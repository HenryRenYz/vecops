# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Frozen transpose/matmul/packing refactor versus its checkpoint and public C++.

Uses the original TransposeBench metadata/input/oracle. Source emission runs
in fresh snapshot processes. All three sources execute in one process with
balanced ordering; every clock reading is preserved, with no filtered trials.
"""

import argparse
import copy
import hashlib
import itertools
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

from dsl_snapshot_entry import select_snapshot


def matmul_workload(compiler, args):
  """Adapt original MatmulBench metadata and its independent NumPy oracle."""
  import numpy as np
  import vecops
  import vecops.language as T
  from dsl_matmul import KIND_TYPES, PIPELINES, catalog, kernels
  from dsl_transpose import aligned_empty, numpy_dtype

  cases = [
    [m, n, k, mode, kind, pipeline, level, args.b_layout]
    for kind in args.kind
    for _, _, m, n, k in catalog(dtype=KIND_TYPES[kind][2])
    if not args.shape or f"{m}x{n}x{k}" in args.shape
    for mode in args.mode
    for pipeline in args.pipeline
    for level in args.level
  ]

  def prototype(case):
    m, n, k, mode, kind, pipeline, level, layout = case
    dsl, cpp = kernels(compiler, m, n, k, mode, kind, pipeline, level, b_layout=layout)[0]
    if args.shared_amx_cache:
      dsl = T.compile(dsl.template, compiler=compiler, pass_options={"cpu_amx_shared_cache_dispatch": True})
    return dsl, cpp

  def metadata(case):
    m, n, k, _, kind, pipeline, _, layout = case
    ad, bd, _, _, acc = KIND_TYPES[kind]
    prologue, epilogue, _ = PIPELINES[pipeline]
    out = "float32" if epilogue else acc
    return (
      vecops.TensorMeta((m, k), dtype=ad),
      vecops.TensorMeta((k, n), strides=(n, 1) if layout == "row" else (1, k), dtype=bd),
      vecops.TensorMeta((m, n), dtype=out),
      vecops.TensorMeta((1 if prologue == 2 else m, n), dtype=acc),
    )

  def inputs(case):
    m, n, k, _, kind, pipeline, _, layout = case
    ad, bd, na, nb, acc = KIND_TYPES[kind]
    prologue, epilogue, _ = PIPELINES[pipeline]
    a = aligned_empty((m, k), ad)
    b = aligned_empty((k, n), bd) if layout == "row" else aligned_empty((n, k), bd).T
    c = aligned_empty((m, n), "float32" if epilogue else acc)
    d = aligned_empty((1 if prologue == 2 else m, n), acc)
    rng = np.random.default_rng(17)
    a[:] = rng.integers(0 if ad == "uint8" else -3, 4, size=a.shape) / (8 if "float" in ad else 1)
    b[:] = rng.integers(-3, 4, size=b.shape) / (8 if "float" in bd else 1)
    d[:] = rng.integers(-3, 4, size=d.shape)
    x, y = a.astype(numpy_dtype(acc)), b.astype(numpy_dtype(acc))
    if pipeline == "affine_input":
      x, y = x * 2 + 1, y * 2 + 1
    expected = x.astype(numpy_dtype(na)).astype(numpy_dtype(acc)) @ y.astype(numpy_dtype(nb)).astype(numpy_dtype(acc))
    if prologue:
      expected = expected + d
    if epilogue:
      expected = expected.astype(np.float32)
    if epilogue == 1:
      expected = np.maximum(expected, 0)
    elif epilogue == 2:
      expected = np.clip(expected, -0.25, 0.25)
    elif epilogue == 3:
      expected = expected * 0.5
    elif epilogue == 4:
      expected = expected / (1 + np.exp(-expected))
    return (a, b, c, d), c, expected.astype(c.dtype), max(2, min(4000, 20_000_000 // (m * n * k)))

  def calls(pair, data, repeat):
    return tuple(
      lambda kernel=kernel: kernel.operator.native(*data, repeat, **kernel._core.specialization_values)
      for kernel in pair
    )

  def check(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=3e-4)

  return cases, prototype, metadata, inputs, calls, check


def packing_workload(compiler, args):
  """Original MatmulPackBench shapes with an independent SME layout oracle."""
  import numpy as np
  import vecops
  import vecops.language as T
  from dsl_benchmark_utils import packed_size, packing_geometry
  from dsl_matmul_pack import catalog, kernels
  from dsl_transpose import aligned_empty, numpy_dtype

  if platform.machine().lower() not in {"aarch64", "arm64"}:
    raise ValueError("SME packing A/B validation requires an actual ARM target")
  cases = [
    [s, k, mode, kind, side, pipeline]
    for s, k in catalog()
    if not args.shape or f"{s}x{k}" in args.shape
    for mode in args.mode
    for kind in args.kind
    for side in args.side
    for pipeline in args.pipeline
  ]

  def prototype(case):
    s, k, mode, kind, side, pipeline = case
    pair, _ = kernels(compiler, s, k, mode, kind, side, pipeline == "affine")
    dsl, cpp = pair
    if args.dsl_pack:
      dsl = T.compile(dsl.template, compiler=compiler, pass_options={"cpu_sme_packing_dsl": True})
    return dsl, cpp

  def types(kind):
    return {"bf16": ("bfloat16", "bfloat16"), "convert": ("float32", "bfloat16"), "i8": ("int8", "int8")}[kind]

  def metadata(case):
    s, k, _, kind, side, _ = case
    memory, native = types(kind)
    return (
      vecops.TensorMeta((s, k) if side == "A" else (k, s), strides=(k, 1) if side == "A" else (1, k), dtype=memory),
      vecops.TensorMeta((packed_size(s, k, native),), dtype=native),
    )

  def inputs(case):
    s, k, _, kind, side, pipeline = case
    memory, native = types(kind)
    data = aligned_empty((s, k), memory)
    rng = np.random.default_rng(23)
    data[:] = rng.integers(-3, 4, size=data.shape) / (8 if "float" in memory else 1)
    output = aligned_empty((packed_size(s, k, native),), native)
    _, panel, group = packing_geometry(native)
    values = data.astype(np.float32) * 2 + 1 if pipeline == "affine" else data
    values = values.astype(numpy_dtype(native))
    padded = np.zeros(((s + panel - 1) // panel * panel, (k + group - 1) // group * group), dtype=values.dtype)
    padded[:s, :k] = values
    # Both operand sides have [spatial-panel, K-group, spatial-lane, K-lane]
    # storage order. Derive it from the format rather than call a packer.
    expected = padded.reshape(-1, panel, padded.shape[1] // group, group).transpose(0, 2, 1, 3).reshape(-1)
    return (data if side == "A" else data.T, output), output, expected, max(2, min(4000, 10_000_000 // (s * k)))

  def calls(pair, data, repeat):
    return tuple(
      lambda kernel=kernel: kernel.operator.native(*data, repeat, **kernel._core.specialization_values)
      for kernel in pair
    )

  return cases, prototype, metadata, inputs, calls, np.testing.assert_array_equal


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--emit", type=Path)
  parser.add_argument("--baseline", type=Path)
  parser.add_argument("--candidate", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--shape", nargs="+")
  parser.add_argument("--dtype", nargs="+")
  parser.add_argument("--mode", nargs="+", default=["Const", "Any", "Aligned"])
  parser.add_argument("--operator", choices=("transpose", "matmul", "packing"), default="transpose")
  parser.add_argument("--kind", nargs="+", default=["bf16"])
  parser.add_argument("--pipeline", nargs="+", default=["plain"])
  parser.add_argument("--level", nargs="+", default=["dot", "matmul"])
  parser.add_argument("--b-layout", choices=("row", "col"), default="col")
  parser.add_argument("--side", nargs="+", default=["A", "B"], choices=("A", "B"))
  parser.add_argument("--implementation", default="auto")
  parser.add_argument(
    "--dsl-sme", action="store_true", help="Stage the experimental general SME transpose library driver"
  )
  parser.add_argument("--dsl-pack", action="store_true", help="Stage the experimental SME packing library driver")
  parser.add_argument(
    "--shared-amx-cache", action="store_true", help="Measure the experimental shared AMX cache driver"
  )
  parser.add_argument("--target", default="Native")
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--rounds", type=int, default=24)
  parser.add_argument("--prepare-only", action="store_true")
  args = parser.parse_args()
  if (
    args.operator == "matmul"
    and not args.emit
    and "fp16" in args.kind
    and args.target.lower() == "native"
    and platform.machine().lower() in {"x86_64", "amd64"}
  ):
    features = Path("/proc/cpuinfo").read_text()
    if "amx_fp16" not in features:
      parser.error(
        "native public C++ FP16 Matmul requires AMX-FP16; this CPU only supports the DSL vector fallback. Select supported paired kinds (e.g. bf16 i8), rather than benchmarking a different dtype as its reference"
      )
  root = select_snapshot(args.root)
  sys.path.insert(0, str(root / "benchmarks"))
  import numpy as np
  import vecops
  import vecops.language as T
  from dsl_timing import reserve_cpu
  from dsl_transpose import MODES, aligned_empty, calls, catalog, kernels, numpy_dtype

  args.output = args.output.resolve()
  args.output.mkdir(parents=True, exist_ok=True)
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cflags=("-ffp-contract=off",),
    cache_dir=args.output / "cache",
    build_dir=args.output / "build",
  )

  if args.operator in {"matmul", "packing"}:
    if args.operator == "packing":
      if any(kind not in {"bf16", "convert", "i8"} for kind in args.kind):
        parser.error("packing requires --kind bf16, convert and/or i8")
      if any(pipeline not in {"plain", "affine"} for pipeline in args.pipeline):
        parser.error("packing requires --pipeline plain and/or affine")
      cases, prototype, metadata, inputs, invoke_all, check = packing_workload(compiler, args)
    else:
      cases, prototype, metadata, inputs, invoke_all, check = matmul_workload(compiler, args)
    if any(mode not in ("Const", "Dynamic") for mode in args.mode):
      parser.error("matmul/packing requires --mode Const and/or Dynamic")
  else:
    cases = [
      (*case, mode)
      for case in catalog()
      for mode in MODES
      if (not args.shape or f"{case[0]}x{case[1]}" in args.shape)
      and (not args.dtype or case[2] in args.dtype)
      and mode in args.mode
    ]

    def prototype(case):
      m, n, ad, cd, compute, mode = case
      dsl, cpp = kernels(compiler, m, n, mode, ad, cd, compute, args.implementation)
      if args.dsl_sme:
        dsl = T.compile(dsl.template, compiler=compiler, pass_options={"cpu_transpose_sme_dsl": True})
      return dsl, cpp

    def metadata(case):
      m, n, ad, cd, _, _ = case
      return vecops.TensorMeta((m, n), dtype=ad), vecops.TensorMeta((n, m), dtype=cd)

    def inputs(case):
      m, n, ad, cd, compute, _ = case
      a, c = aligned_empty((m, n), ad), aligned_empty((n, m), cd)
      a[:] = ((np.arange(m * n, dtype=np.int32) % 251) - 125).reshape(m, n)
      expected = a.astype(numpy_dtype(compute)).astype(c.dtype).T
      return (a, c), c, expected, max(2, min(10000, 4_000_000 // (m * n)))

    def invoke_all(pair, data, repeat):
      return calls(pair, *data, repeat)

    check = np.testing.assert_array_equal
  if not cases:
    parser.error("no original benchmark metadata selected")
  if args.emit:
    records = []
    for case in cases:
      dsl, _ = prototype(case)
      records.append({"case": case, "source": dsl.generated_source})
    args.emit.write_text(json.dumps({"root": str(root), "native": vecops._C.__file__, "records": records}))
    print(f"Emitted {len(records)} original {args.operator} metadata from {root}", flush=True)
    return
  if not args.baseline or not args.candidate or args.rounds <= 0 or args.rounds % 6:
    parser.error("requires two catalogs and a positive multiple of six rounds")
  old, new = (json.loads(path.read_text())["records"] for path in (args.baseline, args.candidate))
  assert [record["case"] for record in old] == [record["case"] for record in new]
  selected = {tuple(case) for case in cases}
  old = [record for record in old if tuple(record["case"]) in selected]
  new = [record for record in new if tuple(record["case"]) in selected]
  if not old:
    parser.error("the paired source catalogs contain no metadata selected by these filters")
  prepared, unique = [], {}
  with vecops.collect_compile_requests() as collector:
    for before, after in zip(old, new):
      key = tuple(before["case"])
      if key not in unique:
        declaration_kernel, cpp = prototype(before["case"])
        variants = []
        for record in (before, after):
          digest = hashlib.sha256(record["source"].encode()).hexdigest()
          source = args.output / (digest + ".cpp")
          source.write_text(record["source"])
          declaration = copy.copy(declaration_kernel.template)
          declaration.ir = replace(declaration.ir, source=source)
          variants.append(T.compile(declaration, compiler=compiler))
        pair = (*variants, cpp)
        meta = metadata(before["case"])
        for kernel in pair:
          kernel.operator(*meta, 1, **kernel._core.specialization_values)
        unique[key] = pair
      prepared.append((before["case"], unique[key]))
  print(f"Preparing {len(collector)} requests for {len(prepared)} original cases", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  if args.prepare_only:
    return
  cpu = min(os.sched_getaffinity(0))
  lease = reserve_cpu(cpu)
  os.sched_setaffinity(0, {cpu})
  rows, orders = [], tuple(itertools.permutations(range(3)))
  try:
    for case, pair in prepared:
      data, c, expected, repeat = inputs(case)
      for invoke in invoke_all(pair, data, 1):
        invoke()
        check(c, expected)
      invocations = invoke_all(pair, data, repeat)
      for invoke in invocations:
        invoke()
      samples, raw = [[], [], []], []
      for round_ in range(args.rounds):
        for position, label in enumerate(orders[round_ % len(orders)]):
          begin = time.perf_counter_ns()
          invocations[label]()
          end = time.perf_counter_ns()
          samples[label].append((end - begin) / repeat)
          raw.append(
            {"round": round_, "position": position, "label": label, "repeat": repeat, "begin_ns": begin, "end_ns": end}
          )
      medians = [statistics.median(values) for values in samples]
      rows.append(
        {
          "case": case,
          "median_ns": medians,
          "new_over_old": medians[1] / medians[0],
          "old_over_cpp": medians[0] / medians[2],
          "new_over_cpp": medians[1] / medians[2],
          "raw": raw,
        }
      )
      for invoke in invocations:
        invoke()
        check(c, expected)
      print(case, rows[-1]["new_over_old"], flush=True)
      (args.output / "results.json").write_text(json.dumps({"cpu": cpu, "root": str(root), "rows": rows}, indent=2))
  finally:
    lease.close()


if __name__ == "__main__":
  main()
