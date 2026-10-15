# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Measure disjoint top-level scratch lifetimes and native ABI call latency.

Repetition lives in the timing harness, outside the DSL kernel. An enclosing
DSL repeat loop would conservatively overlap both scratch intervals and would
not exercise the late storage planner. Peak bytes describe instrumented DSL
scratch lifetimes, not process RSS. Timings include native ABI dispatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T
from dsl_timing import reserve_cpu


def declaration(width):
  N = T.dynamic(lower=1)
  S0, S1 = T.dynamic(), T.dynamic()

  @T.template
  def scratch(
    A: T.Tensor((N, width), (S0, S1), "float32"),
    C: T.Tensor((N, width), dtype="float32", access="output"),
    D: T.Tensor((N, width), dtype="float32", access="output"),
  ):
    for row in T.serial(N, unroll=1):
      first = T.alloc((width,), dtype="float32")
      T.store(first, T.load(A[row, :]) + 1)
      T.store(C[row, :], T.load(first) * 2)
    T.compiler_fence()
    for row in T.serial(N, unroll=1):
      second = T.alloc((width,), dtype="float32")
      T.store(second, T.load(A[row, :]) - 1)
      T.store(D[row, :], T.load(second) * 3)

  return scratch


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--phase", choices=("build", "time"), required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--widths", type=int, nargs="+", default=(17, 257, 4096))
  parser.add_argument("--rows", type=int, nargs="+", default=(1, 32))
  parser.add_argument("--cpu", type=int, required=True)
  parser.add_argument("--rounds", type=int, default=11)
  parser.add_argument("--cc", required=True)
  parser.add_argument("--cxx", required=True)
  args = parser.parse_args()
  if min(*args.rows, *args.widths, args.rounds) <= 0:
    parser.error("rows, widths and rounds must be positive")
  args.work_dir = args.work_dir.resolve()
  args.work_dir.mkdir(parents=True, exist_ok=True)
  manifest = args.work_dir / "manifest.json"
  from vecops._identity import implementation_identity

  identity = {
    "widths": list(args.widths), "rows": list(args.rows), "cpu": args.cpu, "rounds": args.rounds,
    "cc": args.cc, "cxx": args.cxx, "implementation": implementation_identity(),
    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  }
  previous = None
  if args.phase == "time":
    previous = json.loads(manifest.read_text())
    if previous["configuration"] != identity:
      raise ValueError("driver, compiler, package or configuration changed after preparation")
  elif manifest.exists():
    raise ValueError("preserve prepared evidence in a new work directory")
  compiler = vecops.Compiler(
    target="Native", cc=args.cc, cxx=args.cxx, jobs=4,
    cflags=("-ffp-contract=off",),
    cache_dir=args.work_dir / "cache", build_dir=args.work_dir / "build",
    cache_mode="cache-only" if previous else "read-write",
  )
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for width in args.widths:
      template = declaration(width)
      ordinary = T.compile(template, compiler=compiler)
      instrumented = T.compile(template, compiler=compiler, debug=T.DebugOptions(memory=True))
      kernels[width] = ordinary, instrumented
      for rows in args.rows:
        metadata = vecops.TensorMeta((rows, width), dtype="float32")
        for kernel in kernels[width]:
          kernel.operator(metadata, metadata, metadata, **kernel._core.specialization_values)
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  allowed = os.sched_getaffinity(0)
  results = []
  try:
    with reserve_cpu(args.cpu):
      os.sched_setaffinity(0, {args.cpu})
      for width in args.widths:
        ordinary, instrumented = kernels[width]
        digest = hashlib.sha256(ordinary.generated_source.encode()).hexdigest()
        plan = getattr(ordinary.backend_ir, "storage", None)
        slots = len(set(plan.slots.values())) if plan is not None else None
        for rows in args.rows:
          a = (np.arange(rows * width, dtype=np.float32) / np.float32(8)).reshape(rows, width)[:, ::-1]
          c, d = np.empty_like(a), np.empty_like(a)
          for kernel in (ordinary, instrumented):
            kernel.operator.native(a, c, d, **kernel._core.specialization_values)
            np.testing.assert_array_equal(c, (a + 1) * 2)
            np.testing.assert_array_equal(d, (a - 1) * 3)
          profile = instrumented.profile(a, c, d, warmup=0, repeat=1, mode="native", memory=True)
          peak = profile.memory_summary["instrumented_peak_bytes"]
          if slots is not None and (slots != 1 or peak != width * 4):
            raise ValueError("the benchmark did not exercise one physical scratch slot")
          if peak not in (width * 4, width * 8):
            raise ValueError("unexpected scratch ownership or missing instrumentation")
          record = {
            "rows": rows, "width": width, "correct": True, "source_sha256": digest,
            "instrumented_peak_bytes": peak, "physical_slots": slots,
          }
          if previous:
            saved = next(r for r in previous["results"] if (r["rows"], r["width"]) == (rows, width))
            if any(record[k] != saved[k] for k in record):
              raise ValueError("source, numeric validation or storage ownership changed")
            call = vecops._C.KernelCall(
              [vecops._C.TensorView(a), vecops._C.TensorView(c, writable=True), vecops._C.TensorView(d, writable=True)],
              ordinary._core.specialization_values,
            )
            invoke = ordinary.operator.native.invoke
            repeat = 1
            while True:
              begin = time.perf_counter_ns()
              for _ in range(repeat):
                invoke(call)
              elapsed = time.perf_counter_ns() - begin
              if elapsed >= 25_000_000 or repeat >= 2**24:
                break
              repeat *= 2
            samples = []
            for _ in range(args.rounds):
              begin = time.perf_counter_ns()
              for _ in range(repeat):
                invoke(call)
              samples.append((time.perf_counter_ns() - begin) / repeat)
            record.update(median_ns=statistics.median(samples), samples_ns=samples, repeat=repeat)
            np.testing.assert_array_equal(c, (a + 1) * 2)
            np.testing.assert_array_equal(d, (a - 1) * 3)
          results.append(record)
          print(json.dumps(record), flush=True)
  finally:
    os.sched_setaffinity(0, allowed)
  document = {"configuration": identity, "results": results, "timed": bool(previous), "timing_scope": "Python loop and native ABI dispatch included"}
  if not previous:
    manifest.write_text(json.dumps(document, indent=2) + "\n")
  args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
  main()
