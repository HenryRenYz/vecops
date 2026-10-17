# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Measure public DSL lowering on fixed catalog groups, without native builds.

Imports, template construction, and compiler policy initialization occur before
each measurement. Both T.compile and lazy generated_source materialization are
measured. Repeats must emit identical source. This is a fixed evaluation, not a
schedule search; compare snapshots under the same plan, CPU, and frequency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_policy():
  """Record policy and sampled frequency for every allowed CPU."""
  result = {}
  for cpu in sorted(os.sched_getaffinity(0)):
    directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    result[str(cpu)] = {
      name: (directory / name).read_text().strip()
      for name in ("scaling_governor", "scaling_min_freq", "scaling_max_freq", "scaling_cur_freq")
      if (directory / name).is_file()
    }
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--snapshot", type=Path, required=True)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--group-id", action="append", required=True)
  parser.add_argument("--role", choices=("dsl", "pack_dsl", "both"), default="dsl")
  parser.add_argument("--repeat", type=int, default=3)
  parser.add_argument("--cpu", type=int, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.repeat < 1 or args.cpu not in os.sched_getaffinity(0):
    parser.error("repeat must be positive and cpu must be allowed")
  if args.output.exists():
    parser.error("output must be a new directory")
  args.snapshot = args.snapshot.resolve()
  plan = json.loads(args.plan.read_text())
  groups = {group["id"]: group for group in plan["groups"]}
  if len(set(args.group_id)) != len(args.group_id) or not set(args.group_id) <= groups.keys():
    parser.error("group IDs must be unique and present in the plan")
  os.sched_setaffinity(0, {args.cpu})

  # Select one snapshot before importing vecops or the catalog.
  import sys

  sys.path.insert(0, str(args.snapshot / "benchmarks"))
  from dsl_snapshot_entry import select_snapshot

  select_snapshot(args.snapshot)
  import dsl_matmul_artifact_ab as artifacts
  import dsl_matmul_source_catalog as catalog
  import vecops
  import vecops.language as T
  from dsl_matmul_full_suite import subset_manifest, verify_plan_sources

  verify_plan_sources(args.snapshot, plan)
  args.output.mkdir(parents=True)
  report = {
    "protocol": "public-compile-and-generated-source-v1",
    "snapshot": str(args.snapshot),
    "plan_sha256": sha(args.plan),
    "host": platform.node(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "cpu": args.cpu,
    "native_build_requested": False,
    "pass_options": plan["pass_options"],
    "compiler_cflags": plan["compiler_cflags"],
    "toolchain": plan["toolchain"],
    "target": plan["target"],
    "compiler_sources": {
      str(path.relative_to(args.snapshot)): sha(path)
      for path in sorted((args.snapshot / "python/vecops/language").rglob("*.py"))
    },
    "rows": [],
  }
  for gid in args.group_id:
    group = groups[gid]
    scenario = artifacts.source_scenarios(subset_manifest(plan, group["execution_ids"]))[0]
    compute, packing = catalog.source_templates(scenario)
    pairs = {"dsl": compute, "pack_dsl": packing}
    roles = [role for role, pair in pairs.items() if pair is not None] if args.role == "both" else [args.role]
    for role in roles:
      if pairs[role] is None:
        raise ValueError(f"group {gid} has no {role} program")
      expected = None
      for repeat in range(args.repeat):
        compiler = vecops.Compiler(
          target=plan["target"],
          cc=plan["toolchain"]["cc"],
          cxx=plan["toolchain"]["cxx"],
          cflags=plan["compiler_cflags"],
          cache_dir=args.output / "policy-cache",
          build_dir=args.output / "unused-native-build",
        )
        before = cpu_policy()
        cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
        kernel = T.compile(pairs[role][0], compiler=compiler, pass_options=plan["pass_options"])
        source = kernel.generated_source.encode()
        wall_ns, cpu_ns = time.perf_counter_ns() - wall_start, time.process_time_ns() - cpu_start
        after = cpu_policy()
        digest = hashlib.sha256(source).hexdigest()
        if expected is not None and digest != expected:
          raise ValueError(f"nondeterministic generated source: {gid}/{role}")
        expected = digest
        row = {
          "group_id": gid,
          "role": role,
          "repeat": repeat,
          "wall_ns": wall_ns,
          "process_cpu_ns": cpu_ns,
          "source_sha256": digest,
          "source_bytes": len(source),
          "cpu_policy_before": before,
          "cpu_policy_after": after,
        }
        report["rows"].append(row)
        # Preserve completed samples if a later group fails.
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
  verify_plan_sources(args.snapshot, plan)


if __name__ == "__main__":
  main()
