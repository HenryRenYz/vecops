# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Register recurrence vs explicit C++ and deliberate scratch placement.

Native repeat excludes Python dispatch, with shared aligned I/O, alternating
timing order and a CPU affinity applied only after compilation.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T
from dsl_transpose import aligned_empty, measure

ROOT = Path(__file__).resolve().parents[1]


def kernels(compiler, width):
  Width = T.Const(width)

  @T.template
  def dsl(
    A: T.Tensor((Width,), dtype=T.float32),
    C: T.Tensor((Width,), dtype=T.float32, access="output"),
    count: int,
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      x = T.load(A)
      for i in T.serial(count):
        old = x
        x = x + 1
        x = x + old * 0.25
      T.store(C, x)
      T.compiler_fence()

  @T.template(source=ROOT / "benchmarks/dsl_state_reference.cpp")
  def cpp(
    A: T.Tensor((Width,), dtype=T.float32),
    C: T.Tensor((Width,), dtype=T.float32, access="output"),
    count: int,
    repeat: int,
  ):
    Width  # noqa: B018 -- source template capture declaration

  return (
    T.compile(dsl, compiler=compiler),
    T.compile(dsl, compiler=compiler, pass_options={"cpu_state_register_budget": 0}),
    T.compile(cpp, compiler=compiler),
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--width", type=int, nargs="+", default=[7, 17, 37, 64])
  parser.add_argument("--count", type=int, default=32)
  parser.add_argument("--repeat", type=int, default=10000)
  parser.add_argument("--rounds", type=int, default=9)
  parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
  parser.add_argument("--target", default=os.environ.get("VECOPS_TEST_TARGET", "Native"))
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-state")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if min(args.width + [args.count, args.repeat, args.rounds]) <= 0:
    parser.error("shapes and repetition counts must be positive")
  allowed = os.sched_getaffinity(0)
  if args.cpu not in allowed:
    parser.error("timing CPU is outside the caller's allowed affinity")
  args.build_dir.mkdir(parents=True, exist_ok=True)
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=6,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  results = []
  try:
    for width in args.width:
      os.sched_setaffinity(0, allowed - {args.cpu} or allowed)
      variants = kernels(compiler, width)
      a, c = (aligned_empty((width,), "float32") for _ in range(2))
      a[:] = np.linspace(-0.5, 0.5, width)
      expected = a.copy()
      for _ in range(args.count):
        expected = expected + 1 + expected * np.float32(0.25)
      for kernel in variants:
        kernel(a, c, args.count, 1)
        np.testing.assert_allclose(c, expected, rtol=3e-6, atol=3e-6)
      os.sched_setaffinity(0, {args.cpu})
      calls = tuple(
        lambda k=k, a=a, c=c: k.operator.native(a, c, args.count, args.repeat, **k._core.specialization_values)
        for k in variants
      )
      times = measure(calls, args.repeat, args.rounds)
      row = {"width": width, "count": args.count, "register_us": times[0], "scratch_us": times[1], "cpp_us": times[2]}
      results.append(row)
      print(json.dumps(row), flush=True)
  finally:
    os.sched_setaffinity(0, allowed)
  if args.output:
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
  main()
