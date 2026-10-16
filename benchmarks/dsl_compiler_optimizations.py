# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fixed compiler-regression kernels with separate build and quiet timing phases.

Run this same driver in each frozen baseline/candidate snapshot. Native repeat
loops amortize Python dispatch; compilation is forbidden in the timing phase.
Results retain correctness checks, source/DSO sizes and lowering/build times.
This is a fixed evaluation suite, not a configuration search or autotuner.
The versioned buffer protocol fixes allocation order, first-touch CPU and
4096-byte address-window offsets; results require a matching-protocol baseline.
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
from dsl_timing import isolate_compilation, reserve_cpu, warmup_native
from vecops._identity import implementation_identity

CASES = ("invariant-metadata", "invariant-branch", "scratch-lifetimes", "fusion-pressure", "dead-store")
BUFFER_PROTOCOL = {
  "version": "compileropt-fixed-buffers-v1",
  "address_window_bytes": 4096,
  "tensor_offsets": {"A": 0, "C": 64, "D": 128},
  "padding_windows_per_block": 1,
  "allocation_and_first_touch": "reserved-cpu-before-factories",
  "ownership": "one-byte-allocation-per-case-with-disjoint-float32-views",
}


def fixed_buffers(rows, width):
  """Allocate one fixed layout; callers pin the CPU before the first write.

  The address window is a protocol constant, not an OS page-size assumption.
  Each tensor gets a page-rounded payload plus padding, with a fixed offset
  inside its own block. Keeping the owner alive retains all three views.
  """
  window = BUFFER_PROTOCOL["address_window_bytes"]
  payload_bytes = rows * width * np.dtype(np.float32).itemsize
  block_bytes = ((payload_bytes + window - 1) // window + BUFFER_PROTOCOL["padding_windows_per_block"]) * window
  owner = np.empty(3 * block_bytes + window - 1, dtype=np.uint8)
  base_offset = (-int(owner.ctypes.data)) % window
  arrays = {
    name: np.ndarray((rows, width), dtype=np.float32, buffer=owner,
                     offset=base_offset + index * block_bytes + offset)
    for index, (name, offset) in enumerate(BUFFER_PROTOCOL["tensor_offsets"].items())
  }
  owner.fill(0)
  arrays["A"][:] = np.linspace(-1, 1, rows * width, dtype=np.float32).reshape(rows, width)
  arrays["C"].fill(np.nan)
  arrays["D"].fill(np.nan)
  layout = {
    "owner_address": int(owner.ctypes.data),
    "owner_bytes": owner.nbytes,
    "aligned_base_address": int(owner.ctypes.data) + base_offset,
    "block_bytes": block_bytes,
    "payload_bytes": payload_bytes,
    "tensors": {
      name: {
        "address": int(array.ctypes.data),
        "address_mod_4096": int(array.ctypes.data) % window,
        "address_mod_64": int(array.ctypes.data) % 64,
        "shape": list(array.shape),
        "strides": list(array.strides),
      }
      for name, array in arrays.items()
    },
  }
  return owner, arrays["A"], arrays["C"], arrays["D"], layout


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
      if Kind == "invariant-metadata":
        for row in T.serial(N, unroll=1):
          metadata = N * 3 + Width
          T.store(C[row, :], T.load(A[row, :]) + T.cast(metadata, "float32"))
          T.store(D[row, :], T.load(A[row, :]) * 2)
      elif Kind == "invariant-branch":
        for row in T.serial(N, unroll=1):
          if mode > 0:
            T.store(C[row, :], T.load(A[row, :]) + 1)
            T.store(D[row, :], T.load(A[row, :]) * 2)
          else:
            T.store(C[row, :], T.load(A[row, :]) - 1)
            T.store(D[row, :], T.load(A[row, :]) * 3)
      elif Kind == "scratch-lifetimes":
        for row in T.serial(N, unroll=1):
          first = T.alloc((Width,), dtype="float32")
          T.store(first, T.load(A[row, :]) + 1)
          T.store(C[row, :], T.load(first) * 2)
        for row in T.serial(N, unroll=1):
          second = T.alloc((Width,), dtype="float32")
          T.store(second, T.load(A[row, :]) - 1)
          T.store(D[row, :], T.load(second) * 3)
      elif Kind == "dead-store":
        for row in T.serial(N, unroll=1):
          original = T.load(A[row, :])
          temporary = T.alloc((Width,), dtype="float32")
          T.store(temporary, T.zeros((Width,), dtype="float32"))
          T.store(temporary, original + 1)
          T.store(C[row, :], T.load(temporary) * 2)
          T.store(D[row, :], T.load(A[row, :]) * 3)
      else:
        for row in T.serial(N, unroll=1):
          x = T.load(A[row, :])
          p = x * x + 1
          q = p * x + 2
          T.store(C[row, :], q + p)
          T.store(D[row, :], q - p)
      T.compiler_fence()

  return program


def expected(case, a, mode):
  if case == "invariant-metadata":
    return a + np.float32(a.shape[0] * 3 + a.shape[1]), a * 2
  if case == "invariant-branch":
    return (a + 1, a * 2) if mode > 0 else (a - 1, a * 3)
  if case == "scratch-lifetimes":
    return (a + 1) * 2, (a - 1) * 3
  if case == "dead-store":
    return (a + 1) * 2, a * 3
  p = a * a + 1
  q = p * a + 2
  return q + p, q - p


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--phase", choices=("build", "time"), required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
  parser.add_argument("--widths", nargs="+", type=int, default=(17, 257, 4096))
  parser.add_argument("--rows", type=int, default=32)
  parser.add_argument("--repeat", type=int, default=512)
  parser.add_argument("--rounds", type=int, default=11)
  parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--target", default="Native")
  args = parser.parse_args(argv)
  if min((*args.widths, args.rows, args.repeat, args.rounds)) < 1:
    parser.error("dimensions, repeats and rounds must be positive")
  allowed = os.sched_getaffinity(0)
  if args.cpu not in allowed:
    parser.error("CPU must belong to the calling process affinity")
  args.work_dir.mkdir(parents=True, exist_ok=True)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  identity = {
    "cases": list(args.cases),
    "widths": list(args.widths),
    "rows": args.rows,
    "repeat": args.repeat,
    "rounds": args.rounds,
    "cpu": args.cpu,
    "target": args.target,
    "cc": args.cc,
    "cxx": args.cxx,
    "implementation": implementation_identity(),
    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    finally:
      os.sched_setaffinity(0, compilation_affinity)
      lease.close()
    buffer_inputs = iter(buffer_sets)
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
        _owner, a, c, d, buffer_layout = next(buffer_inputs)
        template = declaration(case, width)
        start = time.perf_counter()
        compiled = T.compile(template, compiler=compiler)
        lowering_seconds = time.perf_counter() - start
        start = time.perf_counter()
        compiled.compile_for(a, c, d, 1, args.repeat)
        build_seconds = time.perf_counter() - start
        for mode in (-1, 1):
          compiled(a, c, d, mode, 1)
          e, f = expected(case, a, mode)
          np.testing.assert_allclose(c, e, rtol=2e-6, atol=2e-6)
          np.testing.assert_allclose(d, f, rtol=2e-6, atol=2e-6)
        source = compiled.generated_source
        row = {
          "case": case,
          "width": width,
          "rows": args.rows,
          "correct": True,
          "lowering_seconds": lowering_seconds,
          "build_seconds": build_seconds,
          "source_bytes": len(source.encode()),
          "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
          "kernel_pass_count": len(compiled.kernel_ir.pass_trace),
          "buffers": buffer_layout,
        }
        rows.append(row)
        prepared.append((compiled, a, c, d, row))
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
        for compiled, a, c, d, row in prepared:
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
            compiled(a, c, d, mode, 1)
            e, f = expected(row["case"], a, mode)
            np.testing.assert_allclose(c, e, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(d, f, rtol=2e-6, atol=2e-6)
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
