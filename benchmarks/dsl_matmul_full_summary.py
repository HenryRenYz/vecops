# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Summarize complete catalog timings without hiding unmeasured executions.

Ratios are DSL time / C++ time: below one is faster. Registration-weighted
and workload-deduplicated summaries are both emitted; source aliases must not
silently give a repeated physical workload extra weight. Only validated rows
written by the paired timing runner enter performance statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def statistics_for(rows):
  ratios = sorted(float(row["candidate_over_cpp"]) for row in rows)
  if not ratios:
    return {"count": 0}
  if not all(math.isfinite(value) and value > 0 for value in ratios):
    raise ValueError("nonpositive or nonfinite paired timing")

  def percentile(fraction):
    position = (len(ratios) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ratios[low] + (ratios[high] - ratios[low]) * (position - low)

  return {
    "count": len(ratios),
    "geomean_dsl_over_cpp": math.exp(statistics.fmean(map(math.log, ratios))),
    "median_dsl_over_cpp": statistics.median(ratios),
    "p10": percentile(.1), "p90": percentile(.9),
    "min": ratios[0], "max": ratios[-1],
    "dsl_faster_by_over_10_percent": sum(value < .9 for value in ratios),
    "within_10_percent": sum(.9 <= value <= 1.1 for value in ratios),
    "dsl_slower_by_over_10_percent": sum(value > 1.1 for value in ratios),
    "dsl_over_2x": sum(value > 2 for value in ratios),
    "dsl_over_5x": sum(value > 5 for value in ratios),
  }


# Source spelling, registration line, and human-readable labels are aliases.
# Memory/native dtypes and every observable execution selector are retained.
WORKLOAD_FIELDS = (
  "m", "n", "k", "batch", "broadcast", "meta", "level", "packing",
  "native_a", "native_b", "accumulator", "memory_a", "memory_b",
  "memory_c", "memory_c_input", "input_pipeline_a", "input_pipeline_b",
  "output_pipeline", "reference_family", "reference_policy",
  "reference_bound_invocation", "reference_compensated_pack",
  "batch_strategy", "orientation", "swap", "stage",
)


def summarize(manifest, documents, *, ledger=None):
  declared = {row["execution_id"]: row for row in manifest["executions"]}
  selected = set(manifest.get("selected_execution_ids", declared))
  if not selected <= declared.keys():
    raise ValueError("manifest selects undeclared execution IDs")
  measured, excluded = {}, []
  ledger_rows = {row["execution_id"]: row for row in ledger["rows"]} if ledger is not None else None
  for document in documents:
    for result in document.get("rows", []):
      identity = result["execution_id"]
      if identity not in selected:
        raise ValueError(f"timing outside selected manifest: {identity}")
      key = (identity, result.get("stage", "compute"))
      if ledger_rows is not None:
        record = ledger_rows.get(identity, {}).get("stages", {}).get(key[1], {})
        if record.get("status") != "passed":
          excluded.append({"execution_id": identity, "stage": key[1],
                           "ledger_status": record.get("status", "missing")})
          continue
        if any(record.get(field) != result.get(field) for field in ("median_ns", "artifacts", "candidate_over_cpp")):
          raise ValueError(f"timing result differs from passed ledger: {key}")
      if key in measured:
        raise ValueError(f"duplicate timing (choose one run explicitly): {key}")
      row = {**declared[identity], **result}
      row["swap"] = "EnableSwapAB=false" not in (row.get("configuration") or "")
      row["stage"] = key[1]
      if row["median_ns"]["candidate"] <= 0 or row["median_ns"]["cpp"] <= 0:
        raise ValueError(f"invalid native duration: {key}")
      ratio = row["median_ns"]["candidate"] / row["median_ns"]["cpp"]
      if not math.isclose(ratio, row["candidate_over_cpp"], rel_tol=1e-12):
        raise ValueError(f"ratio does not match retained durations: {key}")
      measured[key] = row
  rows = list(measured.values())
  output = {"ratio_definition": "DSL median ns / C++ median ns; lower is better",
            "selected_executions": len(selected), "excluded_unconfirmed_results": excluded, "stages": {}}
  if ledger_rows is not None:
    output["build_status_by_execution"] = dict(Counter(
      ledger_rows.get(identity, {}).get("build", "missing") for identity in selected))
  requested_stages = set(manifest.get("timing", {}).get("stages", ("compute",)))
  for stage in sorted({row["stage"] for row in rows} | requested_stages):
    stage_rows = [row for row in rows if row["stage"] == stage]
    measured_ids = {row["execution_id"] for row in stage_rows}
    expected = selected if stage == "compute" else {
      identity for identity in selected
      if declared[identity]["packing"] != "raw"
    }
    unique = defaultdict(list)
    for row in stage_rows:
      unique[tuple(json.dumps(row.get(field), sort_keys=True) for field in WORKLOAD_FIELDS)].append(row)
    deduplicated = []
    for aliases in unique.values():
      representative = dict(aliases[0])
      representative["candidate_over_cpp"] = statistics.median(
        row["candidate_over_cpp"] for row in aliases)
      deduplicated.append(representative)
    stage_result = {
      "expected": len(expected), "measured": len(stage_rows),
      "unmeasured_execution_ids": sorted(expected - measured_ids),
      "registration_weighted": statistics_for(stage_rows),
      "unique_workloads": statistics_for(deduplicated),
      "groups": {},
      "slowest": sorted(stage_rows, key=lambda row: row["candidate_over_cpp"], reverse=True)[:25],
      "fastest": sorted(stage_rows, key=lambda row: row["candidate_over_cpp"])[:25],
    }
    if ledger_rows is not None:
      stage_result["validation_status"] = dict(Counter(
        ledger_rows.get(identity, {}).get("stages", {}).get(stage, {}).get("status", "missing")
        for identity in expected))
      stage_result["passed_without_result"] = sorted(
        identity for identity in expected - measured_ids
        if ledger_rows.get(identity, {}).get("stages", {}).get(stage, {}).get("status") == "passed")
    for fields in (("catalog",), ("native_a", "native_b"), ("meta",),
                   ("packing",), ("reference_family",), ("level",),
                   ("catalog", "meta"), ("native_a", "native_b", "packing")):
      groups = defaultdict(list)
      for row in stage_rows:
        groups[" / ".join(str(row.get(field)) for field in fields)].append(row)
      stage_result["groups"][",".join(fields)] = {
        key: statistics_for(values) for key, values in sorted(groups.items())
      }
    output["stages"][stage] = stage_result
  return output, rows


def validated_document(path, manifest):
  """Check the local per-execution manifest, including copied remote evidence."""
  evidence = json.loads(path.with_name("manifest.json").read_text())
  for field in ("source_hashes", "runner_hashes", "pass_options"):
    if evidence.get(field) != manifest.get(field):
      raise ValueError(f"stale or incompatible {field}: {path}")
  document = json.loads(path.read_text())
  expected = set(evidence["selected_execution_ids"])
  if any(row["execution_id"] not in expected for row in document.get("rows", [])):
    raise ValueError(f"result outside its per-execution manifest: {path}")
  return document


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--results", nargs="+", type=Path, required=True,
                      help="explicit results.json files or directories recursively containing them")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--ledger", type=Path, required=True,
                      help="only matching passed validation/timing ledger records enter statistics")
  args = parser.parse_args()
  paths = set()
  for path in args.results:
    paths.update(path.rglob("results.json") if path.is_dir() else (path,))
  manifest = json.loads(args.manifest.read_text())
  summary, rows = summarize(manifest, [validated_document(path, manifest) for path in sorted(paths)],
                            ledger=json.loads(args.ledger.read_text()))
  summary["manifest"] = str(args.manifest.resolve())
  summary["result_files"] = [str(path.resolve()) for path in sorted(paths)]
  args.output_dir.mkdir(parents=True, exist_ok=True)
  (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  fields = ("execution_id", "catalog", "case", *WORKLOAD_FIELDS,
            "dsl_ns", "cpp_ns", "candidate_over_cpp")
  with (args.output_dir / "timings.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow({**row, "dsl_ns": row["median_ns"]["candidate"],
                       "cpp_ns": row["median_ns"]["cpp"]})
  print(json.dumps({stage: {key: value for key, value in result.items()
                            if key in {"expected", "measured", "registration_weighted", "unique_workloads"}}
                    for stage, result in summary["stages"].items()}, indent=2))


if __name__ == "__main__":
  main()
