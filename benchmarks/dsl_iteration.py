# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Native paired timings for SIMD, custom moments and fused prefix consumers.

Compilation is excluded, the timing core and sibling are reserved, and C++
references use named registers / force-inlined helpers rather than arrays or
an artificially scalar reduction. Run at the current governor settings.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T
from dsl_timing import isolate_compilation, reserve_cpu
from dsl_transpose import aligned_empty, measure

ROOT = Path(__file__).resolve().parents[1]


def kernels(compiler, size, kind):
  N, Kind = T.Const(size), T.Const(kind)

  @T.func
  def affine(x):
    return x * 2 + 1

  @T.reducer(identity=(0.0, 0.0), associative=True)
  def moments(a, b):
    return a[0] + b[0], a[1] + b[1]

  @T.template
  def dsl(
    A: T.Tensor((N,), dtype="float32"),
    B: T.Tensor((N,), dtype="float32", access="inout"),
    C: T.Tensor((N,), dtype="float32", access="output"),
    R: T.Tensor((2,), dtype="float32", access="output"),
    repeat: int,
  ):
    for _iteration in T.serial(repeat, unroll=1):
      if kind == 0:
        for i in T.simd(N, unroll=2):
          T.scatter(B, i, affine(T.gather(A, i)), unique=True)
        for j in T.simd(N, unroll=2):
          T.scatter(C, j, T.gather(B, j) * 3 - 1, unique=True)
      elif kind == 1:
        x = T.load(A)
        r = T.reduce((x, x * x), reducer=moments, unroll=2)
        R[0] = r[0]
        R[1] = r[1]
      else:
        T.store(C, T.scan(T.load(A) * 2) * 3 + 1)
      T.compiler_fence()

  @T.template(source=ROOT / "benchmarks/dsl_iteration_reference.cpp")
  def cpp(
    A: T.Tensor((N,), dtype="float32"),
    B: T.Tensor((N,), dtype="float32", access="inout"),
    C: T.Tensor((N,), dtype="float32", access="output"),
    R: T.Tensor((2,), dtype="float32", access="output"),
    repeat: int,
  ):
    N, Kind  # noqa: B018 -- explicit C++ specialization captures

  return (
    T.compile(dsl, compiler=compiler),
    T.compile(dsl, compiler=compiler, pass_options={"loop_fusion": False, "max_fused_reductions": 1}),
    T.compile(cpp, compiler=compiler),
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--size", type=int, nargs="+", default=[257, 4096, 65536])
  parser.add_argument("--kind", type=int, nargs="+", default=[0, 1, 2], choices=[0, 1, 2])
  parser.add_argument("--repeat", type=int, default=2000)
  parser.add_argument("--rounds", type=int, default=9)
  parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
  parser.add_argument("--target", default=os.environ.get("VECOPS_TEST_TARGET", "Native"))
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--output", type=Path)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-iteration")
  args = parser.parse_args()
  if min([*args.size, args.repeat, args.rounds]) <= 0:
    parser.error("sizes/repetition counts must be positive")
  allowed = os.sched_getaffinity(0)
  isolate_compilation(args.cpu)
  build_affinity = os.sched_getaffinity(0)
  compiler = vecops.Compiler(
    target=args.target, cc=args.cc, cxx=args.cxx, jobs=6,
    build_dir=args.build_dir / "build", cache_dir=args.build_dir / "cache"
  )
  cases = []
  with vecops.collect_compile_requests() as collector:
    for kind in args.kind:
      for size in args.size:
        variants = kernels(compiler, size, kind)
        meta = tuple(vecops.TensorMeta((n,), dtype=vecops.float32) for n in (size, size, size, 2))
        for kernel in variants:
          kernel.operator(*meta, args.repeat, **kernel._core.specialization_values)
        cases.append((kind, size, variants))
  vecops.compile_batch(collector.requests, parallelism=6, execution_parallelism=1)
  results = []
  try:
    with reserve_cpu(args.cpu):
      for kind, size, variants in cases:
        os.sched_setaffinity(0, build_affinity)
        a, b, c, r = (aligned_empty((n,), "float32") for n in (size, size, size, 2))
        a[:] = np.linspace(-0.25, 0.25, size)
        for kernel in variants:
          kernel(a, b, c, r, 1)
          if kind == 1:
            np.testing.assert_allclose(r, [a.sum(), np.square(a).sum()], rtol=2e-5, atol=3e-4)
          else:
            expected = (a * 2 + 1) * 3 - 1 if kind == 0 else np.cumsum(a * 2) * 3 + 1
            np.testing.assert_allclose(c, expected, rtol=1e-5, atol=3e-3)
        os.sched_setaffinity(0, {args.cpu})
        calls = tuple(
          lambda k=k, a=a, b=b, c=c, r=r: k.operator.native(a, b, c, r, args.repeat, **k._core.specialization_values)
          for k in variants
        )
        times = measure(calls, args.repeat, args.rounds)
        row = {
          "kind": ("simd", "moments", "scan")[kind],
          "size": size,
          "dsl_us": times[0],
          "off_us": times[1],
          "cpp_us": times[2],
          "ratio_cpp": times[0] / times[2],
          "ratio_off": times[0] / times[1],
          "source": str(variants[0].source.resolve()),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
  finally:
    os.sched_setaffinity(0, allowed)
  if args.output:
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
  main()
