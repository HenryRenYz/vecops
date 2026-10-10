# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Same-round MatmulBench/MatmulFusionBench shapes and transform pipelines.

No JIT in timings. Native inner loops, identical metadata and buffers,
randomized alternating order; pin timing only. C++ uses the current public
operator including its automatically selected families, not a slowed baseline.
"""

import argparse
import ast
import copy
import csv
import json
import os
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments, default_cpu, is_arm, packing_geometry
from dsl_timing import isolate_compilation, reserve_cpu
from dsl_transpose import aligned_empty, measure, numpy_dtype

ROOT = Path(__file__).resolve().parents[1]


def catalog(integer=False, dtype=None):
  dtype = dtype or ("int8" if integer else "bfloat16")
  format_, panel, k_step = packing_geometry(dtype)
  tile = panel // 2 if format_ == "sme" else panel
  values = {"tile": tile, "k_step": k_step}

  def evaluate(source):
    tree = ast.parse(source.strip(), mode="eval").body

    def visit(node):
      if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
      if isinstance(node, ast.Name):
        return values[node.id]
      if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        a, b = visit(node.left), visit(node.right)
        return a + b if isinstance(node.op, ast.Add) else a - b if isinstance(node.op, ast.Sub) else a * b
      raise ValueError("nonconstant benchmark extent")

    return visit(tree)

  cases = []
  for filename in ("MatmulBenchCommon.h", "MatmulScenarioBenchCommon.h"):
    text = (ROOT / "benchmarks/ops" / filename).read_text()
    for group, name, m, n, k in re.findall(r'\{"([\w]+)",\s*"([\w]+)",\s*([^,]+),\s*([^,]+),\s*([^,]+),', text):
      try:
        shape = tuple(evaluate(x) for x in (m, n, k))
      except (KeyError, ValueError, SyntaxError):
        continue
      if all(x > 0 for x in shape):
        cases.append((group, name, *shape))
  # Distinct catalog labels can share a shape. Compile/test/timing once per
  # shape while retaining the first exact catalog label in the CSV.
  return tuple({(m, n, k): (group, name, m, n, k) for group, name, m, n, k in reversed(cases)}.values())[::-1]


PIPELINES = {
  "plain": (0, 0, 1),
  "accumulate": (1, 0, 1),
  "bias_relu": (2, 1, 1),
  "bias_clamp": (2, 2, 1),
  "scale": (0, 3, 1),
  "silu": (0, 4, 1),
  "affine_input": (0, 0, 2),
}

KIND_TYPES = {
  "fp32": ("float32", "float32", "float32", "float32", "float32"),
  "fp16": ("float16", "float16", "float16", "float16", "float32"),
  "fp64": ("float64", "float64", "float64", "float64", "float64"),
  "bf16": ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32"),
  "convert": ("float32", "float32", "bfloat16", "bfloat16", "float32"),
  "i8": ("int8", "int8", "int8", "int8", "int32"),
  "mixed": ("uint8", "int8", "uint8", "int8", "int32"),
}


def templates(m, n, k, mode, kind, pipeline, level, b_layout="col"):
  """Declare original metadata/captures without generating an unused kernel."""
  M, N, K = (m, n, k) if mode == "Const" else (T.dynamic(lower=0) for _ in range(3))
  ad, bd, na, nb, acc = KIND_TYPES[kind]
  prologue, epilogue, input_scale = PIPELINES[pipeline]
  out = acc if epilogue == 0 else "float32"
  DM = 1 if prologue == 2 else M
  b_strides = (N, 1) if b_layout == "row" else (1, K)

  @T.template
  def dsl(
    A: T.Tensor((M, K), (K, 1), ad),
    B: T.Tensor((K, N), b_strides, bd),
    C: T.Tensor((M, N), (N, 1), out, access="output"),
    D: T.Tensor((DM, N), (N, 1), acc),
    repeat: int,
  ):
    for _iteration in T.serial(repeat, unroll=1):
      if input_scale == 2:
        a = T.cast(T.load(A, dtype="float32") * 2 + 1, na)
        b = T.cast(T.load(B, dtype="float32") * 2 + 1, nb)
      else:
        a = T.load(A, dtype=na)
        b = T.load(B, dtype=nb)
      prior = T.load(D) if prologue != 0 else None
      if level == "dot":
        value = T.dot((a, b), C, acc=prior)
      else:
        value = T.library.matmul((a, b), C, acc=prior)
      if epilogue != 0:
        value = T.cast(value, "float32")
        if epilogue == 1:
          value = T.max(value, 0)
        elif epilogue == 2:
          value = T.min(T.max(value, -0.25), 0.25)
        elif epilogue == 3:
          value = value * 0.5
        else:
          value = value / (1 + T.exp(-value))
        T.store(C, value)
      T.compiler_fence()

  NativeA, NativeB = T.DType(getattr(vecops, na)), T.DType(getattr(vecops, nb))
  Prologue, Epilogue, InputScale = T.Const(prologue), T.Const(epilogue), T.Const(input_scale)
  RowB = T.Const(b_layout == "row")

  reference = "dsl_matmul_vector_reference.cpp" if kind == "fp32" and not is_arm() else "dsl_matmul_reference.cpp"

  @T.template(source=ROOT / "benchmarks" / reference)
  def cpp(
    A: T.Tensor((M, K), (K, 1), ad),
    B: T.Tensor((K, N), b_strides, bd),
    C: T.Tensor((M, N), (N, 1), out, access="output"),
    D: T.Tensor((DM, N), (N, 1), acc),
    repeat: int,
  ):
    NativeA, NativeB, Prologue, Epilogue, InputScale, RowB  # noqa: B018 -- specialization captures

  return (dsl, cpp), (
    ad,
    bd,
    na,
    nb,
    acc,
    out,
    prologue,
  )


def kernels(compiler, m, n, k, mode, kind, pipeline, level, pass_options=None, b_layout="col"):
  (dsl, cpp), metadata = templates(m, n, k, mode, kind, pipeline, level, b_layout)
  return (T.compile(dsl, compiler=compiler, pass_options=pass_options), T.compile(cpp, compiler=compiler)), metadata


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul")
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--source-manifest", type=Path, help="Save generated source paths for same-round before/after comparisons"
  )
  parser.add_argument(
    "--baseline-manifest", type=Path, help="Time saved baseline sources alongside current DSL and C++"
  )
  parser.add_argument("--shape", nargs="+")
  parser.add_argument("--kind", nargs="+", default=["bf16"], choices=tuple(KIND_TYPES))
  parser.add_argument("--b-layout", choices=["row", "col"], default="col")
  parser.add_argument("--pipeline", nargs="+", default=["plain"], choices=tuple(PIPELINES))
  parser.add_argument("--mode", nargs="+", default=["Const", "Dynamic"], choices=["Const", "Dynamic"])
  parser.add_argument("--level", nargs="+", default=["dot", "matmul"], choices=["dot", "matmul"])
  parser.add_argument("--rounds", type=int, default=9)
  parser.add_argument("--jobs", type=int, default=4)
  add_toolchain_arguments(parser)
  parser.add_argument("--cpu", type=int, default=default_cpu())
  parser.add_argument("--max-flops", type=int, default=300_000_000, help="Skip huge cases; set 0 for the full catalog")
  parser.add_argument("--fusion", choices=["auto", "always", "never"], default="auto")
  parser.add_argument("--schedule-pipeline", choices=["auto", "off", "prefetch", "staged"], default="auto")
  parser.add_argument("--gemv-k-unroll", type=int, default=2, choices=(1, 2, 4))
  args = parser.parse_args()
  catalogs = {kind: catalog(dtype=KIND_TYPES[kind][2]) for kind in args.kind}
  known_shapes = {f"{m}x{n}x{k}" for entries in catalogs.values() for _, _, m, n, k in entries}
  unknown_shapes = set(args.shape or ()) - known_shapes
  if unknown_shapes:
    parser.error(f"shapes absent from the selected dtype catalogs: {', '.join(sorted(unknown_shapes))}")
  catalogs = {
    kind: tuple(
      entry
      for entry in entries
      if (not args.shape or f"{entry[2]}x{entry[3]}x{entry[4]}" in args.shape)
      and (not args.max_flops or 2 * entry[2] * entry[3] * entry[4] <= args.max_flops)
    )
    for kind, entries in catalogs.items()
  }
  if not any(catalogs.values()):
    parser.error("no matching benchmark cases; check filters or use --max-flops 0 for the full catalog")
  isolate_compilation(args.cpu)
  args.build_dir.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  cases = []
  baseline_sources = json.loads(args.baseline_manifest.read_text()) if args.baseline_manifest else {}
  sources = {}
  kernel_cache = {}
  with vecops.collect_compile_requests() as collector:
    for kind in args.kind:
      for group, name, m, n, k in catalogs[kind]:
        for mode in args.mode:
          for pipeline in args.pipeline:
            for level in args.level:
              key = "/".join(map(str, (m, n, k, mode, kind, pipeline, level)))
              if args.b_layout != "col":
                key += "/" + args.b_layout
              # One dynamic schema covers every runtime shape. Construct and
              # prepare it once rather than compiling identical schedules in
              # independent Operator instances for every catalog entry.
              kernel_key = (
                mode,
                kind,
                pipeline,
                level,
                (m, n, k) if mode == "Const" else None,
                baseline_sources.get(key),
              )
              new_kernel = kernel_key not in kernel_cache
              if new_kernel:
                pair, types = kernels(
                  compiler,
                  m,
                  n,
                  k,
                  mode,
                  kind,
                  pipeline,
                  level,
                  {
                    "cpu_matmul_fusion": args.fusion,
                    "cpu_gemv_k_unroll": args.gemv_k_unroll,
                    "cpu_matmul_pipeline": args.schedule_pipeline,
                  },
                  b_layout=args.b_layout,
                )
                if args.baseline_manifest:
                  # Keep the exact old C++ schedule with the same Python schema,
                  # metadata and compiler. No monkey-patching compiler modules.
                  baseline = copy.copy(pair[0].template)
                  baseline.ir = replace(baseline.ir, source=Path(baseline_sources[key]))
                  pair = (*pair, T.compile(baseline, compiler=compiler))
                kernel_cache[kernel_key] = pair, types
              pair, types = kernel_cache[kernel_key]
              sources[key] = str(pair[0].source.resolve())
              ad, bd, na, nb, acc, out, prologue = types
              meta = (
                vecops.TensorMeta((m, k), dtype=getattr(vecops, ad)),
                vecops.TensorMeta(
                  (k, n), strides=(n, 1) if args.b_layout == "row" else (1, k), dtype=getattr(vecops, bd)
                ),
                vecops.TensorMeta((m, n), dtype=getattr(vecops, out)),
                vecops.TensorMeta((1 if prologue == 2 else m, n), dtype=getattr(vecops, acc)),
              )
              if new_kernel:
                for kernel in pair:
                  kernel.operator(*meta, 1, **kernel._core.specialization_values)
              cases.append((group, name, m, n, k, mode, kind, pipeline, level, pair, types))
  if args.source_manifest:
    args.source_manifest.write_text(json.dumps(sources, indent=2) + "\n")
  print(f"{len(cases)} cases, {len(collector)} artifacts", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  timing_lease = reserve_cpu(args.cpu)
  os.sched_setaffinity(0, {args.cpu})
  stream = args.output.open("w") if args.output else __import__("sys").stdout
  writer = csv.writer(stream)
  columns = ("group", "name", "m", "n", "k", "meta", "dtype", "pipeline", "level", "dsl_us", "cpp_us", "dsl_over_cpp")
  if args.baseline_manifest:
    columns += ("baseline_us", "new_over_baseline", "baseline_over_cpp")
  writer.writerow(columns)
  rng = np.random.default_rng(17)
  for group, name, m, n, k, mode, kind, pipeline, level, pair, types in cases:
    ad, bd, na, nb, acc, out, prologue = types
    a = aligned_empty((m, k), ad)
    b = aligned_empty((k, n), bd) if args.b_layout == "row" else aligned_empty((n, k), bd).T
    c = aligned_empty((m, n), out)
    d = aligned_empty((1 if prologue == 2 else m, n), acc)
    a[:] = rng.integers(0 if ad == "uint8" else -3, 4, size=a.shape) / (8 if "float" in ad else 1)
    b[:] = rng.integers(-3, 4, size=b.shape) / (8 if "float" in bd else 1)
    d[:] = rng.integers(-3, 4, size=d.shape)

    def calls(repeat, pair=pair, a=a, b=b, c=c, d=d):
      return tuple(
        lambda kernel=kernel: kernel.operator.native(a, b, c, d, repeat, **kernel._core.specialization_values)
        for kernel in pair
      )

    ds, cp, *prior = calls(1)
    cp()
    reference = c.copy()
    for before in prior:
      before()
      np.testing.assert_allclose(c, reference, rtol=5e-3, atol=3e-4)
    ds()
    np.testing.assert_allclose(c, reference, rtol=5e-3, atol=3e-4)
    x, y = a.astype(numpy_dtype(acc)), b.astype(numpy_dtype(acc))
    if pipeline == "affine_input":
      x, y = x * 2 + 1, y * 2 + 1
    expected = x.astype(numpy_dtype(na)).astype(numpy_dtype(acc)) @ y.astype(numpy_dtype(nb)).astype(numpy_dtype(acc))
    if prologue:
      expected = expected + d
    epilogue = PIPELINES[pipeline][1]
    if epilogue:
      expected = expected.astype(np.float32)
    if epilogue == 1:
      expected = np.maximum(expected, 0)
    if epilogue == 2:
      expected = np.clip(expected, -0.25, 0.25)
    if epilogue == 3:
      expected = expected * 0.5
    if epilogue == 4:
      expected = expected / (1 + np.exp(-expected))
    np.testing.assert_allclose(c, expected.astype(c.dtype), rtol=5e-3, atol=3e-4)
    repeat = max(2, min(4000, 20_000_000 // (m * n * k)))
    td, tc, *old = measure(calls(repeat), repeat, args.rounds)
    row = (group, name, m, n, k, mode, kind, pipeline, level, f"{td:.4f}", f"{tc:.4f}", f"{td / tc:.4f}")
    if old:
      row += (f"{old[0]:.4f}", f"{td / old[0]:.4f}", f"{old[0] / tc:.4f}")
    writer.writerow(row)
    stream.flush()
    if args.output:
      suffix = f", new/baseline {td / old[0]:.3f}x" if old else ""
      print(f"{mode} {m}x{n}x{k} {kind}/{pipeline}/{level}: {td / tc:.3f}x{suffix}", flush=True)
  if args.output:
    stream.close()
  timing_lease.close()


if __name__ == "__main__":
  main()
