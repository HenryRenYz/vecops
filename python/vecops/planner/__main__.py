# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Command-line entry point for offline vecops memory planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vecops.graph import ExecutionGraph

from . import (
  MachineProfile,
  PlannerConfig,
  audit_plan,
  plan_memory,
  plan_summary,
  profile_from_system,
)


_UNITS = {
  "": 1,
  "b": 1,
  "kib": 1 << 10,
  "mib": 1 << 20,
  "gib": 1 << 30,
  "tib": 1 << 40,
  "kb": 1000,
  "mb": 1000**2,
  "gb": 1000**3,
}


def _bytes(value: str) -> int:
  normalized = value.strip().lower().replace("_", "")
  for suffix in sorted(_UNITS, key=len, reverse=True):
    if suffix and normalized.endswith(suffix):
      number = normalized[: -len(suffix)]
      return int(float(number) * _UNITS[suffix])
  return int(normalized)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Plan HBM/DDR placement for a concrete vecops execution graph."
  )
  parser.add_argument("graph", type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--report", type=Path)
  parser.add_argument("--name", default="two-tier-cpu")
  parser.add_argument(
    "--discover-system",
    action="store_true",
    help="Derive bandwidth and target metadata from vecops.memory.System.",
  )
  parser.add_argument("--fast-capacity", required=True, type=_bytes)
  parser.add_argument("--slow-capacity", default="1TiB", type=_bytes)
  parser.add_argument("--fast-bandwidth-mib-s", type=float)
  parser.add_argument("--slow-bandwidth-mib-s", type=float)
  parser.add_argument("--transfer-bandwidth-mib-s", type=float)
  parser.add_argument("--transfer-latency-ns", default=0.0, type=float)
  parser.add_argument("--alignment", default=64, type=int)
  parser.add_argument("--minimum-fast-bytes", default=4096, type=_bytes)
  parser.add_argument("--minimum-weighted-accesses", default=1.0, type=float)
  parser.add_argument(
    "--prefer-all-fast-when-fit",
    action="store_true",
    help="Place eligible values in fast memory even when modeled benefit is non-positive.",
  )
  return parser


def main() -> None:
  args = _parser().parse_args()
  graph = ExecutionGraph.read(args.graph)
  if args.discover_system:
    from vecops import memory

    machine = profile_from_system(
      memory.System(),
      fast_capacity=args.fast_capacity,
      slow_capacity=args.slow_capacity,
      fast_bandwidth_mib_s=args.fast_bandwidth_mib_s,
      slow_bandwidth_mib_s=args.slow_bandwidth_mib_s,
      transfer_bandwidth_mib_s=args.transfer_bandwidth_mib_s,
      alignment=args.alignment,
      name=args.name,
    )
  else:
    machine = MachineProfile(
      fast_capacity=args.fast_capacity,
      slow_capacity=args.slow_capacity,
      fast_bandwidth_mib_s=(
        256000.0
        if args.fast_bandwidth_mib_s is None
        else args.fast_bandwidth_mib_s
      ),
      slow_bandwidth_mib_s=(
        25600.0
        if args.slow_bandwidth_mib_s is None
        else args.slow_bandwidth_mib_s
      ),
      transfer_bandwidth_mib_s=(
        25600.0
        if args.transfer_bandwidth_mib_s is None
        else args.transfer_bandwidth_mib_s
      ),
      transfer_latency_ns=args.transfer_latency_ns,
      alignment=args.alignment,
      name=args.name,
    )
  plan = plan_memory(
    graph,
    machine,
    config=PlannerConfig(
      minimum_fast_bytes=args.minimum_fast_bytes,
      minimum_weighted_accesses=args.minimum_weighted_accesses,
      prefer_all_fast_when_fit=args.prefer_all_fast_when_fit,
    ),
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  plan.write(args.output)
  summary = {**plan_summary(plan), "audit": audit_plan(graph, plan)}
  if args.report is not None:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
