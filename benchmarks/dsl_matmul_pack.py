# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Time standalone packing against public MatmulPackBench shape/dtype metadata."""

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import torch
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments, default_cpu, packed_size, packing_geometry
from dsl_matmul import ROOT
from dsl_timing import isolate_compilation, reserve_cpu
from dsl_transpose import aligned_empty, measure


def catalog():
  text = (ROOT / "benchmarks/ops/MatmulPackBenchCommon.h").read_text()
  return tuple(dict.fromkeys((int(s), int(k)) for s, k in re.findall(r"PackCase<\w+,\s*\w+,\s*(\d+),\s*(\d+)", text)))


def kernels(compiler, s, k, mode, kind, side, transform):
  S, K = (s, k) if mode == "Const" else (T.dynamic(lower=0) for _ in range(2))
  P = T.dynamic(lower=0)
  memory, native = {"bf16": ("bfloat16", "bfloat16"), "convert": ("float32", "bfloat16"), "i8": ("int8", "int8")}[kind]
  format_ = packing_geometry(native)[0]
  shape, strides = ((S, K), (K, 1)) if side == "A" else ((K, S), (1, K))

  @T.template
  def dsl(A: T.Tensor(shape, strides, memory), O: T.Tensor((P,), dtype=native, access="output"), repeat: int):
    for iteration in T.serial(repeat, unroll=1):
      if transform:
        value = T.load(A, dtype="float32") * 2 + 1
      else:
        value = T.load(A)
      T.pack(value, O, side=side, format=format_, dtype=native)
      T.compiler_fence()

  Native, SideB, Transform = T.DType(getattr(vecops, native)), T.Const(side == "B"), T.Const(transform)

  @T.template(source=ROOT / "benchmarks/dsl_matmul_pack_reference.cpp")
  def cpp(A: T.Tensor(shape, strides, memory), O: T.Tensor((P,), dtype=native, access="output"), repeat: int):
    Native, SideB, Transform  # noqa: B018 -- C++ specialization captures

  return (T.compile(dsl, compiler=compiler), T.compile(cpp, compiler=compiler)), (memory, native)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul-pack")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--shape", nargs="+")
  parser.add_argument("--kind", nargs="+", default=["bf16", "convert", "i8"], choices=["bf16", "convert", "i8"])
  parser.add_argument("--mode", nargs="+", default=["Const", "Dynamic"], choices=["Const", "Dynamic"])
  parser.add_argument("--side", nargs="+", default=["A", "B"], choices=["A", "B"])
  parser.add_argument("--pipeline", nargs="+", default=["plain", "affine"], choices=["plain", "affine"])
  parser.add_argument("--max-elements", type=int, default=2_000_000, help="0 includes all model-weight cases")
  parser.add_argument("--jobs", type=int, default=6)
  parser.add_argument("--rounds", type=int, default=9)
  parser.add_argument("--cpu", type=int, default=default_cpu())
  add_toolchain_arguments(parser)
  args = parser.parse_args()
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
  with vecops.collect_compile_requests() as collector:
    for s, k in catalog():
      if args.shape and f"{s}x{k}" not in args.shape or args.max_elements and s * k > args.max_elements:
        continue
      for kind in args.kind:
        for mode in args.mode:
          for side in args.side:
            for pipeline in args.pipeline:
              pair, types = kernels(compiler, s, k, mode, kind, side, pipeline == "affine")
              memory, native = types
              size = packed_size(s, k, native)
              meta = (
                vecops.TensorMeta(
                  (s, k) if side == "A" else (k, s),
                  strides=(k, 1) if side == "A" else (1, k),
                  dtype=getattr(vecops, memory),
                ),
                vecops.TensorMeta((size,), dtype=getattr(vecops, native)),
              )
              for kernel in pair:
                kernel.operator(*meta, 1, **kernel._core.specialization_values)
              cases.append((s, k, kind, mode, side, pipeline, pair, types, size))
  print(f"{len(cases)} cases, {len(collector)} artifacts", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  lease = reserve_cpu(args.cpu)
  os.sched_setaffinity(0, {args.cpu})
  rng = np.random.default_rng(23)
  with args.output.open("w") as stream:
    writer = csv.writer(stream)
    writer.writerow(("spatial", "k", "dtype", "meta", "side", "pipeline", "dsl_us", "cpp_us", "dsl_over_cpp"))
    for s, k, kind, mode, side, pipeline, pair, types, size in cases:
      memory, native = types
      a = aligned_empty((s, k), memory)
      a[:] = rng.integers(-3, 4, size=(s, k)) / (8 if "float" in memory else 1)
      if side == "B":
        a = a.T
      output = aligned_empty((size,), native)
      calls = lambda repeat, a=a, output=output, pair=pair: tuple(
        lambda kernel=kernel: kernel.operator.native(a, output, repeat, **kernel._core.specialization_values)
        for kernel in pair
      )
      ds, cp = calls(1)
      output.fill(0)
      cp()
      expected = output.copy()
      output.fill(0)
      ds()
      np.testing.assert_array_equal(output, expected)
      repeat = max(2, min(4000, 10_000_000 // (s * k)))
      td, tc = measure(calls(repeat), repeat, args.rounds)
      writer.writerow((s, k, kind, mode, side, pipeline, f"{td:.4f}", f"{tc:.4f}", f"{td / tc:.4f}"))
      stream.flush()
      print(f"{mode} {s}x{k} {kind}/{side}/{pipeline}: {td / tc:.3f}x", flush=True)
  lease.close()


if __name__ == "__main__":
  main()
