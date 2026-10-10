# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Balanced cache-K diagnostics on exact, externally packed registrations.

Preserve the catalog's tensors, transforms, independent numerical oracle and
public C++ control. Offline packing is validated before timing. Changing BK
may change accumulator traffic and arithmetic grouping; every profile must
pass the original numerical tolerance. No runtime result is frequency-adjusted
or filtered. This diagnostic is not full-catalog performance acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments, is_arm
from dsl_matmul_source_catalog import (
  ROOT,
  load_cases,
  result_metadata,
  scenarios,
  select_scenarios,
  source_arguments,
  source_metadata,
  source_templates,
  validate_kernel_pair,
)

PROFILES = (
  ("auto", (None, None, None), "matmul"),
  ("128x128-k256", (128, 128, 256), "matmul"),
  ("128x128-k512", (128, 128, 512), "matmul"),
  ("128x128-k4096", (128, 128, 4096), "matmul"),
  ("whole-tile-dot", (None, None, None), "dot"),
)
LABELS = tuple(label for label, _, _ in PROFILES) + ("cpp",)
_BYPASS_LABEL = "whole-problem-library"


def diagnostic_profiles(*, packed_ab_bypass=False):
  """Keep six-label balance; replace one explicit-block tuning probe only."""
  if not packed_ab_bypass:
    return PROFILES
  return (PROFILES[0], (_BYPASS_LABEL, (None, None, None), "matmul"), *PROFILES[2:])


def profile_options(label):
  # Explicit blocks select blocked probes; automatic blocks use the source
  # recipe's residency decision. No retired compiler policy is toggled.
  return {}


def balanced_order(round_):
  """Six-label Williams design balances positions and directed neighbors."""
  return tuple((index + round_) % 6 for index in (0, 1, 5, 2, 4, 3))


def native_calls(compiled, values, repeat):
  return tuple(
    lambda kernel=kernel: kernel.operator.native(*values, repeat, **kernel._core.specialization_values)
    for kernel in compiled
  )


def validate_packers(scenario, packers, values):
  """Both public and DSL packing must produce exactly the same physical ABI."""
  expected = None
  for index, packer in enumerate(packers):
    native_calls((packer,), values, 1)[0]()
    packed = tuple(values[axis].copy() for axis in (11, 12))
    if index == 0:
      expected = packed
    else:
      for actual, reference, active in zip(packed, expected, (scenario.pack_a, scenario.pack_b), strict=True):
        if active:
          np.testing.assert_array_equal(actual, reference)


def validate_profiles(scenario, compiled, values, expected):
  output = values[2]
  tolerance = 2e-2 if output.dtype.itemsize <= 2 else 2e-10 if output.dtype.itemsize == 8 else 8e-4
  calls = native_calls(compiled, values, 1)
  for invoke in calls[:-1]:
    validate_kernel_pair(invoke, calls[-1], output, expected, tolerance=tolerance)


def time_case(scenario, compiled, values, rounds, stream, *, labels=LABELS):
  case = scenario.case
  repeat = max(1, min(10000, 8_000_000 // math.prod((case.m, case.n, case.k))))
  calls = native_calls(compiled, values, repeat)
  if len(calls) != len(labels) or len(labels) != 6:
    raise ValueError("exactly five DSL profiles and one C++ control required")
  for invoke in calls:
    invoke()
  durations = {label: [] for label in labels}
  for round_ in range(rounds):
    for position, index in enumerate(balanced_order(round_)):
      begin = time.perf_counter_ns()
      calls[index]()
      end = time.perf_counter_ns()
      if end <= begin:
        raise RuntimeError("nonpositive raw timing")
      duration = (end - begin) / repeat
      durations[labels[index]].append(duration)
      stream.write(
        json.dumps(
          {
            "execution_id": scenario.execution_id,
            "round": round_,
            "position": position,
            "label": labels[index],
            "repeat": repeat,
            "begin_ns": begin,
            "end_ns": end,
            "ns": duration,
          }
        )
        + "\n"
      )
      stream.flush()
  medians = {label: statistics.median(values) for label, values in durations.items()}
  return {
    **result_metadata(scenario),
    "median_ns": medians,
    "over_cpp": {label: value / medians["cpp"] for label, value in medians.items()},
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  add_toolchain_arguments(parser)
  parser.add_argument("--shape", nargs="+", default=("128x4096x4096",))
  parser.add_argument("--mode", nargs="+", choices=("Const", "Dynamic"), default=("Const", "Dynamic"))
  parser.add_argument("--packing", nargs="+", choices=("packedb", "packedab"), default=("packedab",))
  parser.add_argument("--svl-bytes", type=int, help="listing only: execution must match current process SVL")
  parser.add_argument("--rounds", type=int, default=18)
  parser.add_argument(
    "--sme-packed-ab-bypass",
    action="store_true",
    help="replace the K256 probe with opt-in whole-problem library traversal",
  )
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--cpu", type=int)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul-cache")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--source-archive-sha256", required=True)
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args(argv)
  if len(args.source_archive_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.source_archive_sha256):
    parser.error("a verified source archive SHA256 is required")
  if args.rounds <= 0 or args.rounds % 6 or args.jobs <= 0:
    parser.error("positive jobs and rounds divisible by six are required")
  profiles = diagnostic_profiles(packed_ab_bypass=args.sme_packed_ab_bypass)
  labels = tuple(label for label, _, _ in profiles) + ("cpp",)
  selected = select_scenarios(
    tuple(scenarios(load_cases(svl_bytes=args.svl_bytes), levels=("matmul",))),
    cohorts=("base", "packed"),
    modes=args.mode,
    ranks=(2,),
    native=("bf16",),
    shapes=args.shape,
    packing=args.packing,
  )
  if args.list:
    writer = csv.writer(sys.stdout)
    writer.writerow(("execution_id", "key", "profile", "blocks", "level"))
    writer.writerows(
      (case.execution_id, case.key, label, blocks, level) for case in selected for label, blocks, level in profiles
    )
    writer.writerows((case.execution_id, case.key, "cpp", "unchanged public reference", "matmul") for case in selected)
    return
  if args.output is None:
    parser.error("--output is required to retain unfiltered raw measurements")
  manifest_path, summary_path = args.output.with_suffix(".manifest.json"), args.output.with_suffix(".summary.json")
  if any(path.exists() for path in (args.output, manifest_path, summary_path)):
    parser.error("select fresh paths; previous evidence is never overwritten")
  if not is_arm():
    parser.error("execution requires a scheduled ARM SME target")
  from vecops._arm_vector_state import get_vector_state

  vector_state = get_vector_state()
  if args.svl_bytes is not None and vector_state.sme_bytes != args.svl_bytes:
    parser.error("audit SVL differs from the execution thread's actual SVL")
  allowed = os.sched_getaffinity(0)
  cpu = min(allowed) if args.cpu is None else args.cpu
  if cpu not in allowed:
    parser.error("timing CPU must be in the allocated affinity")
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
  cache, prepared, source_paths = {}, [], {}
  with vecops.collect_compile_requests() as collector:
    for case in selected:
      key = case.compilation_key
      if key not in cache:
        variants = []
        for label, blocks, level in profiles:
          pair, _ = source_templates(replace(case, level=level), blocks=blocks)
          variants.append(T.compile(pair[0], compiler=compiler, pass_options=profile_options(label)))
        pair, packing = source_templates(case)
        if packing is None:
          raise ValueError("diagnostic requires externally prepared packed inputs")
        cpp = T.compile(pair[1], compiler=compiler)
        packers = tuple(T.compile(template, compiler=compiler) for template in packing)
        cache[key] = ((*variants, cpp), packers)
      compiled, packers = cache[key]
      for kernel in (*compiled, *packers):
        kernel.operator(*source_metadata(case), 1, **kernel._core.specialization_values)
      source_paths[case.execution_id] = {
        **{label: str(kernel.source) for label, kernel in zip(labels, compiled, strict=True)},
        "packing": tuple(str(kernel.source) for kernel in packers),
      }
      prepared.append((case, compiled, packers))
  manifest_path.write_text(
    json.dumps(
      {
        "source_archive_sha256": args.source_archive_sha256,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [result_metadata(case) | {"key": case.key} for case in selected],
        "profiles": profiles,
        "profile_pass_options": {label: profile_options(label) for label, _, _ in profiles},
        "labels": labels,
        "sources": source_paths,
        "rounds": args.rounds,
        "cpu": cpu,
        "compiler": {"cc": args.cc, "cxx": args.cxx, "target": args.target},
        "vector_state": {"sve_bytes": vector_state.sve_bytes, "sme_bytes": vector_state.sme_bytes},
        "logical_requests": len(collector),
        "order": "six-label Williams design",
        "offline_packing_inside_timing": False,
        "scope": "exact-source-metadata diagnostic only, not full-catalog performance acceptance",
      },
      indent=2,
    )
    + "\n"
  )
  print(f"[cache diagnostic] compiling {len(collector)} logical requests", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  summaries = []
  try:
    os.sched_setaffinity(0, {cpu})
    with args.output.open("x") as stream:
      for case, compiled, packers in prepared:
        values, expected = source_arguments(case)
        validate_packers(case, packers, values)
        validate_profiles(case, compiled, values, expected)
        record = time_case(case, compiled, values, args.rounds, stream, labels=labels)
        summaries.append(record)
        print(json.dumps(record), flush=True)
  finally:
    os.sched_setaffinity(0, allowed)
  summary_path.write_text(
    json.dumps(
      {
        "cases": summaries,
        "raw_samples": len(selected) * args.rounds * len(labels),
        "raw_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "correctness": "PASS",
        "outliers_removed": False,
      },
      indent=2,
    )
    + "\n"
  )


if __name__ == "__main__":
  main()
