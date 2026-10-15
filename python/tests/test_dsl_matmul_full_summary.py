# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Coverage and weighting checks for the full paired benchmark report."""

import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "benchmarks/dsl_matmul_full_summary.py"
_spec = importlib.util.spec_from_file_location("matmul_full_summary", _path)
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)


def test_unmeasured_and_registration_aliases_are_explicit():
  manifest = {"executions": [
    {"execution_id": identity, "m": 2, "n": 3, "k": 4,
     "catalog": "base", "packing": "raw", "native_a": "bf16"}
    for identity in ("alias-a", "alias-b", "failed")
  ]}
  documents = [{"rows": [
    {"execution_id": identity, "median_ns": {"candidate": ratio * 100, "cpp": 100},
     "candidate_over_cpp": ratio}
    for identity, ratio in (("alias-a", 2), ("alias-b", 4))
  ]}]
  result, rows = summary.summarize(manifest, documents)
  compute = result["stages"]["compute"]
  assert compute["expected"] == 3
  assert compute["unmeasured_execution_ids"] == ["failed"]
  assert compute["registration_weighted"]["count"] == 2
  assert compute["unique_workloads"]["count"] == 1
  assert compute["unique_workloads"]["median_dsl_over_cpp"] == 3
  assert len(rows) == 2
  with pytest.raises(ValueError, match="duplicate timing"):
    summary.summarize(manifest, documents + documents)


def test_timing_ratio_must_match_durations():
  manifest = {"executions": [{"execution_id": "a", "packing": "raw"}]}
  with pytest.raises(ValueError, match="ratio does not match"):
    summary.summarize(manifest, [{"rows": [{"execution_id": "a",
      "median_ns": {"candidate": 20, "cpp": 10}, "candidate_over_cpp": .5}]}])


def test_statistics_cutoffs_and_invalid_ratios():
  result = summary.statistics_for([
    {"candidate_over_cpp": value} for value in (.5, .9, 1, 1.1, 6)])
  assert result["dsl_faster_by_over_10_percent"] == 1
  assert result["within_10_percent"] == 3
  assert result["dsl_slower_by_over_10_percent"] == 1
  assert result["dsl_over_5x"] == 1
  with pytest.raises(ValueError, match="nonpositive or nonfinite"):
    summary.statistics_for([{"candidate_over_cpp": float("nan")}])


def test_swap_selectors_stay_distinct_and_online_packing_is_expected():
  manifest = {"executions": [
    {"execution_id": identity, "packing": "onlinepackedb", "configuration": selector}
    for identity, selector in (("a", "EnableSwapAB=false"), ("b", "EnableSwapAB=true"))
  ]}
  rows = [{"execution_id": identity, "stage": "packing",
           "median_ns": {"candidate": 20, "cpp": 10}, "candidate_over_cpp": 2}
          for identity in ("a", "b")]
  result, _ = summary.summarize(manifest, [{"rows": rows}])
  assert result["stages"]["packing"]["expected"] == 2
  assert result["stages"]["packing"]["unique_workloads"]["count"] == 2


def test_result_written_before_passed_marker_is_excluded():
  manifest = {"executions": [{"execution_id": "a", "packing": "raw"}]}
  row = {"execution_id": "a", "median_ns": {"candidate": 20, "cpp": 10},
         "candidate_over_cpp": 2}
  ledger = {"rows": [{"execution_id": "a", "stages": {"compute": {"status": "running"}}}]}
  result, rows = summary.summarize(manifest, [{"rows": [row]}], ledger=ledger)
  assert rows == []
  assert result["stages"]["compute"]["unmeasured_execution_ids"] == ["a"]
  ledger["rows"][0]["stages"]["compute"] = {**row, "status": "passed"}
  _, rows = summary.summarize(manifest, [{"rows": [row]}], ledger=ledger)
  assert len(rows) == 1


def test_source_identity_checked_without_remote_absolute_paths(tmp_path):
  identity = {"source_hashes": {"reference.cpp": "current"}, "runner_hashes": {}, "pass_options": {}}
  (tmp_path / "manifest.json").write_text(json.dumps({**identity, "selected_execution_ids": ["a"]}))
  path = tmp_path / "results.json"
  path.write_text(json.dumps({"rows": [{"execution_id": "a"}]}))
  assert summary.validated_document(path, identity)["rows"][0]["execution_id"] == "a"
  with pytest.raises(ValueError, match="stale or incompatible"):
    summary.validated_document(path, {**identity, "source_hashes": {"reference.cpp": "old"}})


def test_entirely_unmeasured_requested_stage_remains_in_report():
  manifest = {"executions": [{"execution_id": "a", "packing": "packedb"}],
              "timing": {"stages": ["compute", "packing"]}}
  ledger = {"rows": [{"execution_id": "a", "build": "compile_failed",
                       "stages": {stage: {"status": "not_run"}
                                  for stage in ("compute", "packing")}}]}
  result, rows = summary.summarize(manifest, [], ledger=ledger)
  assert rows == []
  assert result["build_status_by_execution"] == {"compile_failed": 1}
  assert result["stages"]["packing"]["expected"] == 1
  assert result["stages"]["packing"]["measured"] == 0
  assert result["stages"]["packing"]["validation_status"] == {"not_run": 1}
  assert result["stages"]["packing"]["unmeasured_execution_ids"] == ["a"]
