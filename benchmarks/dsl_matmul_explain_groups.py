# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Explain a full-suite plan/ledger without changing or running its artifacts.

Performance statistics are registration-weighted and accept only passed ledger
records whose retained durations reproduce their ratio. Physical-workload
deduplication remains the separate full_summary report's responsibility.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def classification(row, stage="compute"):
  family, policy = row["reference_family"], row["reference_policy"]
  reference = (
    "forced_family:" + family
    if family != "Automatic"
    else "forced_policy:" + policy
    if policy != "Automatic"
    else "automatic"
  )
  if family != "Automatic" and policy != "Automatic":
    reference += "+forced_policy:" + policy
  m, n, k = (int(row[field]) for field in ("m", "n", "k"))
  geometry = (
    "gemv"
    if min(m, n) == 1
    else "tiny_area"
    if m * n <= 64
    else "skinny"
    if min(m, n) < 16
    else "large_work"
    if m * n * k >= 2**26
    else "matrix"
  )
  transforms = any(
    row.get(field, "Convert") not in {"Convert", "Identity"} for field in ("input_pipeline_a", "input_pipeline_b")
  )
  conversion = any(row.get("memory_" + side) != row.get("native_" + side) for side in ("a", "b"))
  epilogue = row.get("output_pipeline", "Convert") not in {"Convert", "Identity"}
  output_cast = row.get("memory_c") != row.get("accumulator")
  pipeline = (
    "input_transform+epilogue"
    if transforms and epilogue
    else "input_transform"
    if transforms
    else "epilogue"
    if epilogue
    else "dtype_conversion"
    if conversion or output_cast
    else "plain_native"
  )
  packing = row["packing"]
  boundary = (
    "standalone_packing"
    if stage == "packing"
    else "explicit_online_pack+compute"
    if packing.startswith("online")
    else "prepacked_compute"
    if packing != "raw"
    else "raw_compute_internal_preparation_allowed"
  )
  return {
    "reference_selector": reference,
    "dsl_selector": "automatic" if stage == "compute" else "standalone_packer",
    "rank": "batched" if row.get("batch") is not None else "rank2",
    "geometry": geometry,
    "k_band": "long_k" if k > 1024 else "medium_k" if k > 64 else "short_k",
    "pipeline": pipeline,
    "input_dtype_conversion": conversion,
    "output_dtype_conversion": output_cast,
    "registration_kind": row["catalog"],
    "timed_boundary": boundary,
    "stage": stage,
  }


def explain(plan, ledger):
  declared = {row["execution_id"]: row for row in plan["executions"]}
  if len(declared) != len(plan["executions"]):
    raise ValueError("duplicate execution identity in plan")
  if any(ledger.get(field) != plan.get(field) for field in ("architecture", "snapshot", "label")):
    raise ValueError("ledger does not belong to this plan")
  records = {row["execution_id"]: row for row in ledger["rows"]}
  if len(records) != len(ledger["rows"]) or set(records) - declared.keys():
    raise ValueError("duplicate or undeclared ledger identities")
  groups = defaultdict(
    lambda: {"status": Counter(), "build_status": Counter(), "ratios": [], "ids": [], "applicable_registration_rows": 0}
  )
  classifications = {}
  for identity, row in declared.items():
    for stage in plan["timing"]["stages"]:
      labels = classification(row, stage)
      key = tuple(
        labels[field] for field in ("stage", "reference_selector", "rank", "geometry", "pipeline", "timed_boundary")
      )
      entry = groups[key]
      entry["labels"] = {
        field: labels[field]
        for field in ("stage", "reference_selector", "rank", "geometry", "pipeline", "timed_boundary")
      }
      record = records.get(identity, {}).get("stages", {}).get(stage, {"status": "pending"})
      entry["status"][record["status"]] += 1
      entry["build_status"][records.get(identity, {}).get("build", "pending")] += 1
      entry["applicable_registration_rows"] += int(stage == "compute" or row["packing"] != "raw")
      entry["ids"].append(identity)
      classifications[identity + "/" + stage] = labels
      if record["status"] == "passed":
        candidate, cpp = (float(record["median_ns"][side]) for side in ("candidate", "cpp"))
        ratio = float(record["candidate_over_cpp"])
        if min(candidate, cpp, ratio) <= 0 or not all(map(math.isfinite, (candidate, cpp, ratio))):
          raise ValueError(f"invalid duration/ratio: {identity}/{stage}")
        if not math.isclose(candidate / cpp, ratio, rel_tol=1e-12):
          raise ValueError(f"duration/ratio mismatch: {identity}/{stage}")
        entry["ratios"].append(ratio)
  output = []
  for key in sorted(groups):
    entry = groups[key]
    ratios = entry.pop("ratios")
    entry["declared_stage_rows_including_not_applicable"] = len(entry["ids"])
    if ratios:
      entry["registration_weighted"] = {
        "passed": len(ratios),
        "median_dsl_over_cpp": statistics.median(ratios),
        "geomean_dsl_over_cpp": math.exp(statistics.fmean(map(math.log, ratios))),
      }
    output.append(entry)
  return {
    "ratio_definition": "DSL median ns / C++ median ns; below one is faster",
    "weighting": "registration-weighted; use full_summary for physical-workload deduplication",
    "rules": {
      "selector": "C++ reference selectors; DSL continues automatic dispatch",
      "rank": "plan batch is non-null, including batch=1",
      "geometry": "priority GEMV=min(M,N)=1; tiny=M*N<=64; skinny=min(M,N)<16; large=M*N*K>=2^26; otherwise matrix",
      "packing": "online explicit pack every native repeat; offline pack outside compute; raw allows each implementation's internal preparation",
      "timing": "steady-state native repeat throughput; workspace/session setup may be amortized, not full Python-call latency or prepared-lifecycle latency",
    },
    "groups": output,
    "classification_by_execution_stage": classifications,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--ledger", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = explain(json.loads(args.plan.read_text()), json.loads(args.ledger.read_text()))
  result.update(plan=str(args.plan.resolve()), ledger=str(args.ledger.resolve()))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(f"Wrote {len(result['groups'])} explanation groups to {args.output}")


if __name__ == "__main__":
  main()
