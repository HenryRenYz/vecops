"""Balanced, raw-timing normalization scheduling diagnostics.

Use the maintained rank-two library/C++ factories, inputs and numerical
oracles. Vary only SIMD unrolling and a proven private register-layout policy;
ordinary FP operations remain uncontracted. This is not full-catalog coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import vecops
from dsl_benchmark_utils import add_toolchain_arguments
from dsl_catalog_utils import validate_catalog_filters
from dsl_norm_catalog import AFFINE, calls, kernels, metadata, validate
from dsl_norm_library import catalog, library_kernel

ROOT = Path(__file__).resolve().parents[1]
PROFILES = (
  ("off-u2", 2, "off"),
  ("auto-u2", 2, "auto"),
  ("force-u2", 2, "force"),
  ("off-u4", 4, "off"),
  ("force-u4", 4, "force"),
)
LABELS = tuple(profile[0] for profile in PROFILES) + ("cpp",)


def pass_options(policy):
  if policy not in {"off", "auto", "force"}:
    raise ValueError("unknown register-layout policy")
  result = {"cpu_math_chunk_pow": 1}
  if policy == "force":
    result["cpu_pointwise_register_layout"] = True
  else:
    result["cpu_pointwise_register_layout_policy"] = policy
  return result


def balanced_order(round_):
  """Six-label Williams design: positions and directed neighbors balance."""
  return tuple((index + round_) % 6 for index in (0, 1, 5, 2, 4, 3))


def time_case(case, compiled, operands, rounds, stream):
  repeat = max(2, min(10000, 4_000_000 // math.prod(case.shape)))
  native_calls = calls(compiled, operands, repeat)
  if len(native_calls) != len(LABELS):
    raise ValueError("exactly five DSL profiles and one C++ control required")
  for invoke in native_calls:
    invoke()
  durations = {label: [] for label in LABELS}
  for round_ in range(rounds):
    for position, index in enumerate(balanced_order(round_)):
      begin = time.perf_counter_ns()
      native_calls[index]()
      end = time.perf_counter_ns()
      if end <= begin:
        raise RuntimeError("nonpositive raw timing")
      duration = (end - begin) / repeat
      durations[LABELS[index]].append(duration)
      stream.write(
        json.dumps(
          {
            "key": case.key,
            "round": round_,
            "position": position,
            "label": LABELS[index],
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
    "key": case.key,
    "median_ns": medians,
    "over_cpp": {label: value / medians["cpp"] for label, value in medians.items()},
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  add_toolchain_arguments(parser)
  parser.add_argument("--shape", nargs="+", default=("1x4096", "64x768", "4096x4096"))
  parser.add_argument("--dtype", nargs="+", choices=("float16", "bfloat16"), default=("float16", "bfloat16"))
  parser.add_argument("--affine", nargs="+", choices=AFFINE, default=AFFINE)
  parser.add_argument("--rounds", type=int, default=18)
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--cpu", type=int)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-norm-schedule")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--source-archive-sha256", required=True)
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args(argv)
  if len(args.source_archive_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.source_archive_sha256):
    parser.error("a verified source archive SHA256 is required")
  if args.rounds <= 0 or args.rounds % 6 or args.jobs <= 0:
    parser.error("positive jobs and rounds divisible by six are required")
  selected = validate_catalog_filters(
    parser,
    tuple(case for case in catalog() if case.operation == "layernorm"),
    (
      ("--shape", args.shape, lambda case: case.key[1]),
      ("--dtype", args.dtype, lambda case: case.dtype),
      ("--affine", args.affine, lambda case: case.affine),
    ),
  )
  if args.list:
    writer = csv.writer(__import__("sys").stdout)
    writer.writerow(("operation", "shape", "dtype", "meta", "affine", "accuracy", "variance", "profile"))
    writer.writerows((*case.key, label) for case in selected for label in LABELS)
    return
  if args.output is None:
    parser.error("--output is required to retain unfiltered raw measurements")
  manifest_path, summary_path = args.output.with_suffix(".manifest.json"), args.output.with_suffix(".summary.json")
  if any(path.exists() for path in (args.output, manifest_path, summary_path)):
    parser.error("select fresh paths; previous evidence is never overwritten")
  allowed = os.sched_getaffinity(0)
  cpu = min(allowed) if args.cpu is None else args.cpu
  if cpu not in allowed:
    parser.error("timing CPU must be in the allocated affinity")
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  by_schema, prepared = {}, []
  with vecops.collect_compile_requests() as collector:
    for case in selected:
      if case.schema_key not in by_schema:
        variants = tuple(
          library_kernel(compiler, case, unroll=unroll, pass_options=pass_options(policy))
          for _, unroll, policy in PROFILES
        )
        _, cpp = kernels(compiler, case, unroll=2)
        by_schema[case.schema_key] = (*variants, cpp)
        for compiled in by_schema[case.schema_key]:
          compiled.operator(*metadata(case), 1, **compiled._core.specialization_values)
      prepared.append((case, by_schema[case.schema_key]))
  manifest = {
    "source_archive_sha256": args.source_archive_sha256,
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cases": [case.key for case in selected],
    "profiles": PROFILES,
    "pass_options": {label: pass_options(policy) for label, _, policy in PROFILES},
    "labels": LABELS,
    "rounds": args.rounds,
    "cpu": cpu,
    "compiler": {"cc": args.cc, "cxx": args.cxx, "target": args.target},
    "logical_requests": len(collector),
    "order": "six-label Williams design",
    "scope": "maintained-metadata diagnostic only, not full-catalog performance acceptance",
    "fp_contract_changed": False,
  }
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  summaries = []
  try:
    os.sched_setaffinity(0, {cpu})
    with args.output.open("x") as stream:
      for case, compiled in prepared:
        # The original public oracle is used for every profile before timing.
        operands = validate(case, (compiled[0], compiled[-1]))
        for variant in compiled[1:-1]:
          validate(case, (variant, compiled[-1]))
        record = time_case(case, compiled, operands, args.rounds, stream)
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
