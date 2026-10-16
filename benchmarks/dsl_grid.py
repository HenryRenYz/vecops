# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Native-repeat CPU grid scheduling benchmark, including dispatch and barriers.

Run in af3. Compilation is excluded and timings alternate variants using shared
aligned arrays. Affinity is restricted only for this process, never frequency.
Use --multiprocess for spawn-based independent per-process CPU partitions.
"""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T
from dsl_transpose import aligned_empty, measure

ROOT = Path(__file__).resolve().parents[1]


def kernel(compiler, width, operation, schedule="static", mode="auto", parallel="auto"):
  M = T.dynamic(lower=0)

  @T.template
  def compute(
    A: T.Tensor((M, width), dtype="float32"),
    B: T.Tensor((M, width), dtype="float32", access="inout"),
    C: T.Tensor((M, width), dtype="float32", access="output"),
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      for row in T.grid(M, schedule=schedule, chunk=2, parallel=parallel):
        x = T.load(A[row, :])
        if operation == "imbalance":
          if row < M // 4:
            T.store(B[row, :], T.softmax(x))
          else:
            T.store(B[row, :], x + 1)
        elif operation == "softmax":
          y = T.softmax(x)
        elif operation == "layernorm":
          y = T.layernorm(x, variance="moments")
        else:
          y = x + 1
        if operation != "imbalance":
          T.store(B[row, :], y)
      if operation == "multigrid":
        scale = 3.0
        for row in T.grid(M, schedule=schedule, chunk=2, parallel=parallel):
          T.store(C[row, :], T.load(B[row, :]) * scale)
        for row in T.grid(M, schedule=schedule, chunk=2, parallel=parallel):
          T.store(B[row, :], T.load(C[row, :]) - 2)
      T.compiler_fence()

  return T.compile(compute, compiler=compiler, pass_options={"cpu_grid_regions": mode})


def reference(a, operation):
  if operation == "layernorm":
    mean = a.mean(-1, keepdims=True)
    return (a - mean) / np.sqrt((a * a).mean(-1, keepdims=True) - mean * mean + 1e-5)
  if operation in {"softmax", "imbalance"}:
    exp = np.exp(a - a.max(-1, keepdims=True))
    out = exp / exp.sum(-1, keepdims=True)
    if operation == "imbalance":
      out[a.shape[0] // 4 :] = a[a.shape[0] // 4 :] + 1
    return out
  return (a + 1) * 3 - 2 if operation == "multigrid" else a + 1


def check_outputs(a, b, c, operation):
  np.testing.assert_allclose(b, reference(a, operation), rtol=4e-5, atol=4e-6)
  if operation == "multigrid":
    np.testing.assert_allclose(c, (a + 1) * 3, rtol=4e-5, atol=4e-6)


def worker(arguments):
  cpus, directory, width, rows, operation, repeat, rounds, target, cc, cxx = arguments
  os.sched_setaffinity(0, cpus)
  compiler = vecops.Compiler(
    target=target, cc=cc, cxx=cxx, jobs=4, cache_dir=directory / "cache", build_dir=directory / "build"
  )
  rng = np.random.default_rng(42)
  a, b, c = (aligned_empty((rows, width), "float32") for _ in range(3))
  a[:] = rng.normal(size=a.shape)
  compiled = kernel(compiler, width, operation)
  with vecops.cpu_thread_limit(len(cpus)):
    compiled(a, b, c, 1)
    check_outputs(a, b, c, operation)
    warmup = {}
    (elapsed,) = measure((lambda: compiled.operator.native(a, b, c, repeat),), repeat, rounds, warmup_record=warmup)
  return {"pid": os.getpid(), "cpus": sorted(cpus), "us": elapsed, "correct": True, "warmup": warmup}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
  parser.add_argument("--rows", type=int, nargs="+", default=[3, 9, 37, 256])
  parser.add_argument("--width", type=int, default=4096)
  parser.add_argument(
    "--operation",
    nargs="+",
    choices=["softmax", "layernorm", "multigrid", "imbalance"],
    default=["softmax", "layernorm", "multigrid", "imbalance"],
  )
  parser.add_argument("--repeat", type=int, default=100)
  parser.add_argument("--rounds", type=int, default=9)
  parser.add_argument("--cpus", type=int, nargs="+")
  parser.add_argument("--multiprocess", type=int, default=0)
  parser.add_argument("--target", default=os.environ.get("VECOPS_TEST_TARGET", "Native"))
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-grid")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--prepare-only", action="store_true", help="Build and validate without timing")
  parser.add_argument("--cache-only", action="store_true", help="Require retained native artifacts")
  args = parser.parse_args()
  if args.multiprocess and (args.prepare_only or args.cache_only):
    parser.error("prepare/cache-only currently require the single-process driver")
  cpus = sorted(args.cpus or os.sched_getaffinity(0))[: max(args.threads) * max(1, args.multiprocess)]
  available = os.sched_getaffinity(0)
  if not set(cpus) <= available:
    parser.error("benchmark CPUs must be inside the caller's allowed CPU set")
  compilation_cpus = available - set(cpus)
  if min(args.threads + args.rows + [args.width, args.repeat, args.rounds]) <= 0:
    parser.error("thread counts, shapes and timing counts must be positive")
  if args.multiprocess < 0:
    parser.error("multiprocess count must be nonnegative")
  args.build_dir.mkdir(parents=True, exist_ok=True)
  if args.multiprocess:
    if len(cpus) < args.multiprocess * max(args.threads):
      parser.error("not enough allowed CPUs for disjoint process partitions")
    partition = max(args.threads)
    tasks = [
      (
        set(cpus[i * partition : (i + 1) * partition]),
        args.build_dir / f"process-{i}",
        args.width,
        max(args.rows),
        args.operation[0],
        args.repeat,
        args.rounds,
        args.target,
        args.cc,
        args.cxx,
      )
      for i in range(args.multiprocess)
    ]
    with multiprocessing.get_context("spawn").Pool(args.multiprocess) as pool:
      results = pool.map(worker, tasks)
  else:
    compiler = vecops.Compiler(
      target=args.target,
      cc=args.cc,
      cxx=args.cxx,
      jobs=6,
      cache_dir=args.build_dir / "cache",
      build_dir=args.build_dir / "build",
      cache_mode="cache-only" if args.cache_only else "read-write",
    )
    results = []
    rng = np.random.default_rng(42)
    prepared = {}
    if compilation_cpus:
      os.sched_setaffinity(0, compilation_cpus)
    with vecops.collect_compile_requests() as collector:
      for operation in args.operation:
        variants = [
          ("serial", kernel(compiler, args.width, operation, parallel="serial")),
          ("static", kernel(compiler, args.width, operation)),
          ("dynamic", kernel(compiler, args.width, operation, schedule="dynamic")),
        ]
        if operation == "multigrid":
          variants += [
            (mode, kernel(compiler, args.width, operation, mode=mode))
            for mode in ("individual", "adjacent", "template")
          ]
        prepared[operation] = variants
        shape = (max(args.rows), args.width)
        meta = (
          vecops.TensorMeta(shape, dtype=vecops.float32),
          vecops.TensorMeta(shape, dtype=vecops.float32, access=vecops.TensorAccess.inout),
          vecops.TensorMeta(shape, dtype=vecops.float32, access=vecops.TensorAccess.output),
        )
        for _, compiled in variants:
          compiled.operator(*meta, 1)
    vecops.compile_batch(collector.requests, parallelism=6, execution_parallelism=1)
    for operation, variants in prepared.items():
      for rows in args.rows:
        a, b, c = (aligned_empty((rows, args.width), "float32") for _ in range(3))
        a[:] = rng.normal(size=a.shape)
        for _, compiled in variants:
          compiled(a, b, c, 1)
          check_outputs(a, b, c, operation)
        os.sched_setaffinity(0, set(cpus))
        for threads in args.threads:
          with vecops.cpu_thread_limit(threads):
            calls = tuple(lambda k=k, a=a, b=b, c=c: k.operator.native(a, b, c, args.repeat) for _, k in variants)
            warmup = {}
            times = None if args.prepare_only else measure(calls, args.repeat, args.rounds, warmup_record=warmup)
            # Check every budget/provider path, outside the timing window.
            for _, compiled in variants:
              compiled.operator.native(a, b, c, 1)
              check_outputs(a, b, c, operation)
            if times is None:
              results.extend(
                {"operation": operation, "rows": rows, "width": args.width, "threads": threads,
                 "schedule": name, "validated": True, "timed": False}
                for name, _ in variants
              )
              continue
            for (name, _), elapsed in zip(variants, times):
              row = {
                "warmup": warmup,
                "operation": operation,
                "rows": rows,
                "width": args.width,
                "threads": threads,
                "schedule": name,
                "us": elapsed,
                "over_serial": elapsed / times[0],
              }
              results.append(row)
              print(json.dumps(row), flush=True)
  if args.output:
    args.output.write_text(json.dumps(results, indent=2) + "\n")
  if args.multiprocess:
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
  main()
