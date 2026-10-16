# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Loop-local integer facts: congruence, complete panels and reverse ranges.

Build and timing are separate, with source/implementation identity guards,
fixed first-touch buffer layouts and full output/input/canary checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

# Share the allocation protocol rather than maintaining a second layout.
import dsl_compiler_optimizations as buffer_driver
import numpy as np
import vecops
import vecops.language as T
from dsl_timing import isolate_compilation, reserve_cpu, warmup_native
from vecops._identity import implementation_identity

CASES = ("forward-congruence", "full-span", "reverse-range", "captured-full-span")
BUFFER_PROTOCOL = {**buffer_driver.BUFFER_PROTOCOL, "version": "loop-facts-fixed-buffers-v1"}
fixed_buffers = buffer_driver.fixed_buffers


def declaration(case, width):
  N = T.dynamic(lower=0, upper=1_048_576)
  Width = T.Const(width)
  Kind = case

  @T.template
  def program(
    A: T.Tensor((N, Width), dtype=T.float32),
    C: T.Tensor((N, Width), dtype=T.float32, access="output"),
    D: T.Tensor((N, Width), dtype=T.float32, access="output"),
    mode: int,
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      if Kind == "forward-congruence":
        for begin in T.serial(0, N, 4, unroll=1):
          if begin % 4 == 0:
            T.store(C[begin:T.min(begin + 4, N), :], T.load(A[begin:T.min(begin + 4, N), :]) + mode)
          else:
            T.store(C[begin:T.min(begin + 4, N), :], T.load(A[begin:T.min(begin + 4, N), :]) - 99)
          T.store(D[begin:T.min(begin + 4, N), :], T.load(A[begin:T.min(begin + 4, N), :]) * 2)
      elif Kind == "full-span":
        for row in T.serial(N, unroll=1):
          for column in T.serial(0, Width - Width % 16, 16, unroll=1):
            if T.min(column + 16, Width) - column == 16:
              T.store(C[row, column:column + 16], T.load(A[row, column:column + 16]) + mode)
            else:
              T.store(C[row, column:column + 16], T.load(A[row, column:column + 16]) - 99)
            T.store(D[row, column:column + 16], T.load(A[row, column:column + 16]) * 2)
          if Width % 16 != 0:
            T.store(C[row, Width - Width % 16:], T.load(A[row, Width - Width % 16:]) + mode)
            T.store(D[row, Width - Width % 16:], T.load(A[row, Width - Width % 16:]) * 2)
      elif Kind == "reverse-range":
        for row in T.serial(N - 1, -1, -1, unroll=1):
          if row >= 0:
            T.store(C[row, :], T.load(A[row, :]) + mode)
          else:
            T.store(C[row, :], T.load(A[row, :]) - 99)
          T.store(D[row, :], T.load(A[row, :]) * 3)
      else:
        for row in T.serial(N, unroll=1):
          for column in T.serial(0, Width - Width % 16, 16, unroll=1):
            end = T.min(column + 16, Width)
            span = end - column
            if span == 16:
              T.store(C[row, column:end], T.load(A[row, column:end]) + mode)
            else:
              T.store(C[row, column:end], T.load(A[row, column:end]) - 99)
            T.store(D[row, column:end], T.load(A[row, column:end]) * 2)
          if Width % 16 != 0:
            T.store(C[row, Width - Width % 16:], T.load(A[row, Width - Width % 16:]) + mode)
            T.store(D[row, Width - Width % 16:], T.load(A[row, Width - Width % 16:]) * 2)
      T.compiler_fence()

  return program


def expected(case, a, mode):
  return a + mode, a * (3 if case == "reverse-range" else 2)


def validate(compiled, a, c, d, owner, original, case, mode):
  c.fill(np.nan)
  d.fill(np.nan)
  compiled(a, c, d, mode, 1)
  first, second = expected(case, a, mode)
  np.testing.assert_allclose(c, first, rtol=2e-6, atol=2e-6)
  np.testing.assert_allclose(d, second, rtol=2e-6, atol=2e-6)
  cursor = 0
  for output in sorted((c, d), key=lambda array: array.ctypes.data):
    begin = int(output.ctypes.data) - int(owner.ctypes.data)
    np.testing.assert_array_equal(owner[cursor:begin], original[cursor:begin])
    cursor = begin + output.nbytes
  np.testing.assert_array_equal(owner[cursor:], original[cursor:])


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--phase", choices=("build", "time"), required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
  parser.add_argument("--widths", nargs="+", type=int, default=(17, 257, 4096))
  parser.add_argument("--rows", type=int, default=33)
  parser.add_argument("--validation-rows", nargs="+", type=int, default=(0, 1, 3, 4, 5, 33))
  parser.add_argument("--repeat", type=int, default=512)
  parser.add_argument("--rounds", type=int, default=11)
  parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--target", default="Native")
  args = parser.parse_args(argv)
  if min((*args.widths, args.repeat, args.rounds)) < 1 or args.rows < 0 or min(args.validation_rows) < 0:
    parser.error("widths, repeats and rounds must be positive; rows must be nonnegative")
  allowed = os.sched_getaffinity(0)
  if args.cpu not in allowed:
    parser.error("CPU must belong to the calling process affinity")
  args.work_dir.mkdir(parents=True, exist_ok=True)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  identity = {
    "cases": list(args.cases),
    "widths": list(args.widths),
    "rows": args.rows,
    "validation_rows": list(args.validation_rows),
    "repeat": args.repeat,
    "rounds": args.rounds,
    "cpu": args.cpu,
    "target": args.target,
    "cc": args.cc,
    "cxx": args.cxx,
    "implementation": implementation_identity(),
    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "buffer_helper_sha256": hashlib.sha256(Path(buffer_driver.__file__).read_bytes()).hexdigest(),
    "buffer_protocol": BUFFER_PROTOCOL,
  }
  manifest = args.work_dir / "manifest.json"
  if args.phase == "time" and (not manifest.is_file() or json.loads(manifest.read_text())["configuration"] != identity):
    raise ValueError("timing requires a completed build with exactly matching configuration")
  isolate_compilation(args.cpu)
  prepared, rows = [], []
  try:
    compilation_affinity = os.sched_getaffinity(0)
    lease = reserve_cpu(args.cpu)
    try:
      os.sched_setaffinity(0, {args.cpu})
      # Allocate and first-touch the whole selected suite before any factory
      # or compiler work can change the allocator's history. Retain owners.
      buffer_sets = [fixed_buffers(args.rows, width) for _case in args.cases for width in args.widths]
      original_owners = [buffers[0].copy() for buffers in buffer_sets]
      validation_sets = [
        [(fixed_buffers(count, width), count) for count in args.validation_rows]
        for _case in args.cases for width in args.widths
      ]
      validation_originals = [[buffers[0].copy() for buffers, _count in group] for group in validation_sets]
    finally:
      os.sched_setaffinity(0, compilation_affinity)
      lease.close()
    buffer_inputs = iter(zip(buffer_sets, original_owners, validation_sets, validation_originals))
    compiler = vecops.Compiler(
      target=args.target,
      cc=args.cc,
      cxx=args.cxx,
      jobs=4,
      cache_mode="cache-only" if args.phase == "time" else "read-write",
      cache_dir=args.work_dir / "cache",
      build_dir=args.work_dir / "build",
    )
    for case in args.cases:
      for width in args.widths:
        (_owner, a, c, d, buffer_layout), original, validations, originals = next(buffer_inputs)
        template = declaration(case, width)
        start = time.perf_counter()
        compiled = T.compile(template, compiler=compiler)
        lowering_seconds = time.perf_counter() - start
        start = time.perf_counter()
        compiled.compile_for(a, c, d, 1, args.repeat)
        build_seconds = time.perf_counter() - start
        for mode in (-1, 1):
          validate(compiled, a, c, d, _owner, original, case, mode)
        for (buffers, count), initial in zip(validations, originals):
          owner, aa, cc, dd, _layout = buffers
          for mode in (-1, 1):
            validate(compiled, aa, cc, dd, owner, initial, case, mode)
        source = compiled.generated_source
        row = {
          "case": case,
          "width": width,
          "rows": args.rows,
          "correct": True,
          "validation_rows_correct": list(args.validation_rows),
          "input_and_owner_canaries_unchanged": True,
          "lowering_seconds": lowering_seconds,
          "build_seconds": build_seconds,
          "source_bytes": len(source.encode()),
          "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
          "kernel_pass_count": len(compiled.kernel_ir.pass_trace),
          "buffers": buffer_layout,
        }
        rows.append(row)
        prepared.append((compiled, a, c, d, _owner, original, row))
    if args.phase == "build":
      document = {
        "configuration": identity,
        "results": rows,
        "native_bytes": sum(p.stat().st_size for p in (args.work_dir / "cache").rglob("*.so")),
      }
      manifest.write_text(json.dumps(document, indent=2) + "\n")
    else:
      built = json.loads(manifest.read_text())
      expected_sources = {(row["case"], row["width"]): row["source_sha256"] for row in built["results"]}
      if any(expected_sources.get((row["case"], row["width"])) != row["source_sha256"] for row in rows):
        raise ValueError("generated source changed after the build phase")
      # Repin after compilation/cache preparation. All buffers were already
      # first-touched on this CPU; cache-only forbids background compilation.
      lease = reserve_cpu(args.cpu)
      try:
        os.sched_setaffinity(0, {args.cpu})
        for compiled, a, c, d, owner, original, row in prepared:
          invoke = lambda k=compiled, a=a, c=c, d=d: k.operator.native(
            a, c, d, 1, args.repeat, **k._core.specialization_values
          )
          row["warmup"] = warmup_native(invoke)
          samples = []
          for _ in range(args.rounds):
            start = time.perf_counter_ns()
            invoke()
            samples.append((time.perf_counter_ns() - start) / args.repeat)
          row.update(samples_ns=samples, median_ns=statistics.median(samples))
          for mode in (-1, 1):
            validate(compiled, a, c, d, owner, original, row["case"], mode)
          row["post_timing_correct"] = True
      finally:
        lease.close()
      document = {"configuration": identity, "results": rows, "build": built}
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({
      "phase": args.phase,
      "cases": len(rows),
      "output": str(args.output),
      "buffer_protocol": BUFFER_PROTOCOL,
      "primary_metric": "wall_ns_per_repeat",
      "layout_scope": "fixed 64-byte alignment and 4096-byte offsets; absolute/physical addresses and cache placement can differ",
      "comparison_scope": "requires the same buffer protocol; historical layouts are separate results",
    }), flush=True)
  finally:
    os.sched_setaffinity(0, allowed)


if __name__ == "__main__":
  main()
