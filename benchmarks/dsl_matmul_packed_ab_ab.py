"""Complete geometry cohort for the bounded packed-AB cache bypass.

Keep every original registration meeting the cheap geometry gate, including
GEMV, policy/fusion variants and cache-budget rejects. Do not cherry-pick the
two previously profitable keys or exclude a case based on measured speed.
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
from dsl_matmul_ab_utils import time_case
from dsl_matmul_cache_ab import validate_packers, validate_profiles
from dsl_matmul_source_catalog import (
  ROOT,
  load_cases,
  result_metadata,
  scenarios,
  source_arguments,
  source_metadata,
  source_templates,
)
from vecops.language._compiler.backends.cpu.cost.sme import PackedABTraversalPolicy

LABELS = ("unchanged", "resident-packed-ab", "cpp")
CONTROL_OPTIONS = {"cpu_sme_packed_ab_cache_bypass": False}
OPTIONS = {"cpu_sme_packed_ab_cache_bypass": True}
MARKER = "/* SME packed-AB whole-problem geometry */"


def geometry_cases(*, svl_bytes):
  """Select by actual captured SVL and central policy, not a fixed byte width.

  Cache residency deliberately is NOT a selection criterion: a runtime
  rejection must preserve correctness and incur no significant regression.
  """
  if type(svl_bytes) is not int or not 16 <= svl_bytes <= 256 or svl_bytes & (svl_bytes - 1):
    raise ValueError("SVL must be a power-of-two byte length in [16,256]")
  policy = PackedABTraversalPolicy()
  return tuple(
    case
    for case in scenarios(load_cases(svl_bytes=svl_bytes), levels=("matmul",))
    if case.packing == "packedab"
    and not case.batched
    and case.case.native_a == case.case.native_b == "bf16"
    and case.case.accumulator == "fp32"
    and policy.shape_candidate(case.case.m, case.case.n, case.case.k, lanes=svl_bytes // 4, k_group=2)
  )


def shard_cases(cases, *, index, count):
  """Deterministic disjoint shards preserve the full original cohort order."""
  if type(count) is not int or count <= 0 or type(index) is not int or not 0 <= index < count:
    raise ValueError("shard index must be in [0, positive shard count)")
  selected = tuple(cases[index::count])
  if not selected:
    raise ValueError("requested shard is empty")
  return selected


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  add_toolchain_arguments(parser)
  parser.add_argument("--svl-bytes", type=int, help="listing only; execution verifies the current process")
  parser.add_argument("--rounds", type=int, default=18)
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--cpu", type=int)
  parser.add_argument("--shard-index", type=int, default=0)
  parser.add_argument("--shard-count", type=int, default=1)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul-packed-ab-cohort")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--source-archive-sha256", required=True)
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args(argv)
  if len(args.source_archive_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.source_archive_sha256):
    parser.error("a verified source archive SHA256 is required")
  if args.rounds <= 0 or args.rounds % 6 or args.jobs <= 0:
    parser.error("positive jobs and rounds divisible by six are required")
  vector_state = None
  if args.list:
    if args.svl_bytes is None:
      parser.error("listing requires explicit --svl-bytes; no fixed-width assumption")
    svl_bytes = args.svl_bytes
  else:
    if not is_arm() or not os.environ.get("CCS_JOB_ID"):
      parser.error("execution requires a scheduled ARM SME target")
    from vecops._arm_vector_state import get_vector_state

    vector_state = get_vector_state()
    svl_bytes = vector_state.sme_bytes
    if args.svl_bytes is not None and svl_bytes != args.svl_bytes:
      parser.error("requested SVL differs from the execution thread's actual SVL")
  try:
    cohort = geometry_cases(svl_bytes=svl_bytes)
    selected = shard_cases(cohort, index=args.shard_index, count=args.shard_count)
  except ValueError as error:
    parser.error(str(error))
  if args.list:
    writer = csv.writer(sys.stdout)
    writer.writerow(("execution_id", "key", "profile"))
    writer.writerows((case.execution_id, case.key, label) for case in selected for label in LABELS)
    return
  if args.output is None:
    parser.error("--output is required to retain every raw observation")
  manifest_path, summary_path = args.output.with_suffix(".manifest.json"), args.output.with_suffix(".summary.json")
  if any(path.exists() for path in (args.output, manifest_path, summary_path)) or args.build_dir.exists():
    parser.error("fresh output/build paths required; previous evidence is never overwritten")
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
        raise ValueError("original externally packed registration must provide its packers")
      compiled = (
        T.compile(pair[0], compiler=compiler, pass_options=CONTROL_OPTIONS),
        T.compile(pair[0], compiler=compiler, pass_options=OPTIONS),
        T.compile(pair[1], compiler=compiler),
      )
      packers = tuple(T.compile(template, compiler=compiler) for template in packing)
      for kernel in (*compiled, *packers):
        kernel.operator(*source_metadata(case), 1, **kernel._core.specialization_values)
      before, after = (kernel.source.read_text() for kernel in compiled[:2])
      changed, guard = before != after, MARKER in after
      if MARKER in before or changed != guard:
        raise RuntimeError(f"original key changed outside the candidate runtime policy guard: {case.execution_id}")
      sources[case.execution_id] = {
        **{label: str(kernel.source) for label, kernel in zip(LABELS, compiled, strict=True)},
        "packing": [str(kernel.source) for kernel in packers],
        "candidate_guard_present": guard,
        "candidate_source_changed": changed,
      }
      prepared.append((case, compiled, packers))
  manifest_path.write_text(
    json.dumps(
      {
        "source_archive_sha256": args.source_archive_sha256,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [result_metadata(case) | {"key": case.key} for case in selected],
        "complete_cohort_execution_ids": [case.execution_id for case in cohort],
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "labels": LABELS,
        "profile_pass_options": {"unchanged": CONTROL_OPTIONS, "resident-packed-ab": OPTIONS},
        "sources": sources,
        "rounds": args.rounds,
        "cpu": cpu,
        "compiler": {"cc": args.cc, "cxx": args.cxx, "target": args.target},
        "vector_state": {"sve_bytes": vector_state.sve_bytes, "sme_bytes": vector_state.sme_bytes},
        "logical_requests": len(collector),
        "order": "all three-label permutations, repeated without filtering",
        "offline_packing_inside_timing": False,
        "scope": "complete original geometry cohort, not full-catalog performance acceptance",
      },
      indent=2,
    )
    + "\n"
  )
  print(
    f"[resident packed-AB] compiling {len(collector)} logical requests for {len(selected)}/{len(cohort)} keys",
    flush=True,
  )
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  summaries = []
  try:
    os.sched_setaffinity(0, {cpu})
    with args.output.open("x") as stream:
      for case, compiled, packers in prepared:
        values, expected = source_arguments(case)
        validate_packers(case, packers, values)
        validate_profiles(case, compiled, values, expected)
        record = time_case(case, compiled, values, args.rounds, stream, labels=LABELS)
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
