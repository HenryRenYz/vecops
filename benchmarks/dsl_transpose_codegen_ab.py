# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Isolate compiler row-unrolling pressure in tiny constant conversions.

This is a generated-code diagnostic, not a new lowering or default policy.
Every candidate is the current DSL source with only LLVM loop hints changed.
The unchanged source, two/four-row hints, a disabled-unroll hint and the
maintained C++ reference are timed together on the same CPU in nine rounds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments
from dsl_transpose import FIELDS, aligned_empty, calls, catalog, kernels, measure, numpy_dtype

ROOT = Path(__file__).resolve().parents[1]
HINTS = ("unchanged", "two", "four", "disable")


def selected_cases():
  """All original-catalog 16x16 conversions, never just favorable pairs."""
  return tuple((*case, "Const") for case in catalog() if case[:2] == (16, 16) and len(set(case[2:])) != 1)


def row_hint_source(source, hint):
  if hint not in HINTS:
    raise ValueError(f"unsupported row-unroll hint {hint!r}")
  if hint == "unchanged":
    return source, 0
  before, separator, body = source.partition("void __kernel__(")
  if not separator:
    raise ValueError("missing generated DSL entry point")
  count = {"two": 2, "four": 4}.get(hint)
  pragma = "unroll(disable)" if hint == "disable" else f"unroll_count({count})"
  # A lowered row loop may already forbid Clang unrolling. Replace that
  # attached hint rather than stacking incompatible LLVM loop directives.
  # Keep the existing GCC branch, if present: this experiment changes only
  # Clang's row scheduling, not the generated loop or another compiler's hint.
  pattern = (
    r"(?m)^([ \t]*)"
    r"(?:#if defined\(__clang__\)\n\1#pragma clang loop unroll(?:\(disable\)|_count\(\d+\))\n"
    r"(?P<gcc>\1#elif defined\(__GNUC__\)\n\1#pragma GCC unroll \d+\n)?\1#endif\n\1)?"
    r"(?P<loop>for \(vecops::nint_t dsl_c_axis_\d+ = 0;)"
  )

  def replace(match):
    indent = match[1]
    gcc = match["gcc"] or ""
    return (
      f"{indent}#if defined(__clang__)\n{indent}#pragma clang loop {pragma}\n"
      f"{gcc}{indent}#endif\n{indent}{match['loop']}"
    )

  body, count = re.subn(pattern, replace, body)
  if not count:
    raise ValueError("the generated entry has no ordinary row traversal")
  return before + separator + body, count


def source_template(path, case):
  m, n, ad, cd, _, _ = case

  @T.template(source=path)
  def candidate(
    A: T.Tensor((m, n), (n, 1), dtype=ad),
    C: T.Tensor((n, m), (m, 1), dtype=cd, access="output"),
    repeat: int,
  ):
    pass

  return candidate


def original_pair(compiler, case):
  """Translate catalog column order to the maintained kernel factory API."""
  m, n, input_dtype, output_dtype, compute_dtype, mode = case
  return kernels(compiler, m, n, mode, input_dtype, output_dtype, compute_dtype)


def validate(case, compiled):
  m, n, ad, cd, compute, _ = case
  a = aligned_empty((m, n), ad)
  a[:] = ((np.arange(m * n, dtype=np.int32) % 251) - 125).reshape(m, n)
  expected = a.astype(numpy_dtype(compute)).astype(numpy_dtype(cd)).T
  for kernel in compiled:
    storage = aligned_empty((m * n + 16,), cd)
    storage[:] = -100
    output = storage[8:-8].reshape(n, m)
    kernel.operator.native(a, output, 1, **kernel._core.specialization_values)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(storage[:8], -100)
    np.testing.assert_array_equal(storage[-8:], -100)
  # One invocation with overlapping, differently typed views. The reference
  # Transpose contract is not used as an oracle for overlapping storage.
  size = max(m * n * numpy_dtype(ad).itemsize, m * n * numpy_dtype(cd).itemsize)
  for kernel in compiled[:-1]:
    storage = np.full(size + 32, 0xA5, dtype=np.uint8)
    a_alias = np.ndarray((m, n), dtype=numpy_dtype(ad), buffer=storage, offset=16)
    c_alias = np.ndarray((n, m), dtype=numpy_dtype(cd), buffer=storage, offset=16)
    a_alias[:] = a
    kernel.operator.native(a_alias, c_alias, 1, **kernel._core.specialization_values)
    np.testing.assert_array_equal(c_alias, expected)
    np.testing.assert_array_equal(storage[:16], 0xA5)
    np.testing.assert_array_equal(storage[-16:], 0xA5)
  return a, aligned_empty((n, m), cd)


def main(arguments=None):
  parser = argparse.ArgumentParser(description=__doc__)
  add_toolchain_arguments(parser)
  parser.add_argument("--source-archive-sha256", required=True)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--rounds", type=int, default=9)
  parser.add_argument("--cpu", type=int)
  parser.add_argument("--build-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args(arguments)
  cases = selected_cases()
  if args.list:
    print(json.dumps({"cases": cases, "hints": HINTS}))
    return
  if not re.fullmatch(r"[0-9a-f]{64}", args.source_archive_sha256) or args.rounds < 1 or args.jobs < 1:
    parser.error("verified source hash and positive rounds/jobs are required")
  manifest_path = args.output.with_suffix(".manifest.json")
  if args.output.exists() or manifest_path.exists():
    parser.error("choose fresh diagnostic outputs")
  allowed = os.sched_getaffinity(0)
  cpu = min(allowed) if args.cpu is None else args.cpu
  if cpu not in allowed:
    parser.error("timing CPU is outside the allocation")
  arguments = {"target": args.target, "cc": args.cc, "cxx": args.cxx, "jobs": args.jobs}
  compiler = vecops.Compiler(
    **arguments, cache_dir=args.build_dir / "cpp-cache", build_dir=args.build_dir / "cpp-build"
  )
  strict_compiler = vecops.Compiler(
    **arguments,
    cflags=("-ffp-contract=off",),
    cache_dir=args.build_dir / "dsl-cache",
    build_dir=args.build_dir / "dsl-build",
  )
  sources_dir = args.build_dir / "derived-sources"
  sources_dir.mkdir(parents=True, exist_ok=False)
  prepared, source_records = [], []
  with vecops.collect_compile_requests() as collector:
    for index, case in enumerate(cases):
      m, n, ad, cd, _, _ = case
      original, cpp = original_pair(compiler, case)
      original_source = original.get_kernel_source()
      candidates = []
      for hint in HINTS:
        source, hint_count = row_hint_source(original_source, hint)
        path = sources_dir / f"case-{index}-{hint}.cpp"
        path.write_text(source)
        candidate = T.compile(source_template(path, case), compiler=strict_compiler)
        candidates.append(candidate)
        source_records.append(
          {"case": case, "hint": hint, "count": hint_count, "sha256": hashlib.sha256(source.encode()).hexdigest()}
        )
      for kernel in (*candidates, cpp):
        kernel.operator(
          vecops.TensorMeta((m, n), dtype=ad),
          vecops.TensorMeta((n, m), dtype=cd),
          1,
          **kernel._core.specialization_values,
        )
      prepared.append((case, tuple(candidates) + (cpp,)))
  manifest = {
    "source_root": str(ROOT),
    "source_archive_sha256": args.source_archive_sha256,
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cases": cases,
    "hints": HINTS,
    "source_variants": source_records,
    "cpp_reference_sha256": hashlib.sha256((ROOT / "benchmarks/dsl_transpose_reference.cpp").read_bytes()).hexdigest(),
    "compiler": arguments,
    "candidate_extra_flags": ["-ffp-contract=off"],
    "cpp_extra_flags": [],
    "cpu": cpu,
    "rounds": args.rounds,
    "logical_requests": len(collector),
    "scope": "All original-catalog tiny conversions; generated-code hint experiment, not a default change",
    "oracle": "Exact input->compute->output transpose and canaries; each DSL-derived source additionally tests overlap",
  }
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  try:
    os.sched_setaffinity(0, {cpu})
    with args.output.open("w", newline="") as stream:
      writer = csv.writer(stream)
      writer.writerow((*FIELDS[:6], "hint", "candidate_us", "cpp_us", "candidate_over_cpp", "candidate_over_unchanged"))
      for case, compiled in prepared:
        a, c = validate(case, compiled)
        repeat = 10000
        timings = measure(calls(compiled, a, c, repeat), repeat, args.rounds)
        for hint, duration in zip(HINTS, timings[:-1]):
          writer.writerow(
            (
              *case,
              hint,
              f"{duration:.6f}",
              f"{timings[-1]:.6f}",
              f"{duration / timings[-1]:.6f}",
              f"{duration / timings[0]:.6f}",
            )
          )
        stream.flush()
  finally:
    os.sched_setaffinity(0, allowed)


if __name__ == "__main__":
  main()
