# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Paired singleton-input diagnostics using unchanged original catalog metadata.

Only the opt-in code-generation flag changes. Offline packing, transforms,
independent numerical oracle and public C++ reference remain unchanged. Every
permutation of the three variants is used, balancing position and directed
neighbors. Raw timestamps are retained without filtering or adjustment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments, is_arm
from dsl_matmul_ab_utils import balanced_order as _balanced_order
from dsl_matmul_ab_utils import select_cases
from dsl_matmul_ab_utils import time_case as _time_case
from dsl_matmul_cache_ab import validate_packers, validate_profiles
from dsl_matmul_source_catalog import (
  ROOT,
  result_metadata,
  source_arguments,
  source_metadata,
  source_templates,
)

LABELS = ("unchanged", "singleton-input", "cpp")
DEFAULT_KEYS = ("e6209a02fb9b668c8ff1e15a", "a42a69d0befab988d3973925")
OPTIONS = {"cpu_sme_narrow_singleton_slices": True}


def balanced_order(round_):
  return _balanced_order(round_)


def time_case(scenario, compiled, values, rounds, stream):
  return _time_case(scenario, compiled, values, rounds, stream, labels=LABELS)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  add_toolchain_arguments(parser)
  parser.add_argument("--execution-id", nargs="+", default=DEFAULT_KEYS)
  parser.add_argument("--svl-bytes", type=int, help="listing only; execution must match the current process")
  parser.add_argument("--rounds", type=int, default=18)
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--cpu", type=int)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul-singleton")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--source-archive-sha256", required=True)
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args(argv)
  if len(args.source_archive_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.source_archive_sha256):
    parser.error("a verified source archive SHA256 is required")
  if args.rounds <= 0 or args.rounds % 6 or args.jobs <= 0:
    parser.error("positive jobs and rounds divisible by six are required")
  try:
    selected = select_cases(args.execution_id, svl_bytes=args.svl_bytes)
  except ValueError as error:
    parser.error(str(error))
  if args.list:
    writer = csv.writer(sys.stdout)
    writer.writerow(("execution_id", "key", "profile"))
    writer.writerows((case.execution_id, case.key, label) for case in selected for label in LABELS)
    return
  if args.output is None:
    parser.error("--output is required to retain every raw observation")
  manifest_path = args.output.with_suffix(".manifest.json")
  summary_path = args.output.with_suffix(".summary.json")
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
  prepared, sources = [], {}
  with vecops.collect_compile_requests() as collector:
    for case in selected:
      pair, packing = source_templates(case)
      if packing is None:
        raise ValueError("diagnostic requires the original offline packed input")
      compiled = (
        T.compile(pair[0], compiler=compiler),
        T.compile(pair[0], compiler=compiler, pass_options=OPTIONS),
        T.compile(pair[1], compiler=compiler),
      )
      packers = tuple(T.compile(template, compiler=compiler) for template in packing)
      for kernel in (*compiled, *packers):
        kernel.operator(*source_metadata(case), 1, **kernel._core.specialization_values)
      marker = "proven singleton spatial input: one value per K slot"
      if marker in compiled[0].source.read_text() or marker not in compiled[1].source.read_text():
        raise RuntimeError(f"original key does not prove a changed singleton input route: {case.execution_id}")
      sources[case.execution_id] = {
        **{label: str(kernel.source) for label, kernel in zip(LABELS, compiled, strict=True)},
        "packing": [str(kernel.source) for kernel in packers],
      }
      prepared.append((case, compiled, packers))
  manifest_path.write_text(
    json.dumps(
      {
        "source_archive_sha256": args.source_archive_sha256,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [result_metadata(case) | {"key": case.key} for case in selected],
        "labels": LABELS,
        "profile_pass_options": {"unchanged": {}, "singleton-input": OPTIONS},
        "sources": sources,
        "rounds": args.rounds,
        "cpu": cpu,
        "compiler": {"cc": args.cc, "cxx": args.cxx, "target": args.target},
        "vector_state": {"sve_bytes": vector_state.sve_bytes, "sme_bytes": vector_state.sme_bytes},
        "logical_requests": len(collector),
        "order": "all three-label permutations, repeated without filtering",
        "offline_packing_inside_timing": False,
        "scope": "exact original metadata diagnostic; not full-catalog performance acceptance",
      },
      indent=2,
    )
    + "\n"
  )
  print(f"[singleton diagnostic] compiling {len(collector)} logical requests", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  summaries = []
  try:
    os.sched_setaffinity(0, {cpu})
    with args.output.open("x") as stream:
      for case, compiled, packers in prepared:
        values, expected = source_arguments(case)
        validate_packers(case, packers, values)
        validate_profiles(case, compiled, values, expected)
        record = time_case(case, compiled, values, args.rounds, stream)
        summaries.append(record)
        print(json.dumps(record), flush=True)
  finally:
    os.sched_setaffinity(0, allowed)
  summary_path.write_text(
    json.dumps(
      {
        "cases": summaries,
        "raw_samples": len(selected) * args.rounds * len(LABELS),
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
