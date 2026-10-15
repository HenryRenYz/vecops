# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Interpretation boundaries for the full-suite result tables."""

import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "benchmarks/dsl_matmul_explain_groups.py"
_spec = importlib.util.spec_from_file_location("matmul_explain_groups", _path)
explain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(explain)


def row(**changes):
  return {
    "execution_id": "a",
    "m": 32,
    "n": 32,
    "k": 17,
    "batch": None,
    "catalog": "MatmulFusionExpanded",
    "reference_family": "Automatic",
    "reference_policy": "Automatic",
    "memory_a": "bf16",
    "memory_b": "bf16",
    "native_a": "bf16",
    "native_b": "bf16",
    "memory_c": "fp32",
    "accumulator": "fp32",
    "input_pipeline_a": "Convert",
    "input_pipeline_b": "Convert",
    "output_pipeline": "Convert",
    "packing": "raw",
    **changes,
  }


def test_forced_reference_is_not_claimed_as_forced_dsl():
  labels = explain.classification(row(reference_family="ResidualSplit"))
  assert labels["reference_selector"] == "forced_family:ResidualSplit"
  assert labels["dsl_selector"] == "automatic"
  assert (
    explain.classification(row(reference_policy="FourRegions"))["reference_selector"] == "forced_policy:FourRegions"
  )


def test_rank_shape_pipeline_and_packing_are_independent():
  labels = explain.classification(row(batch="1", m=1, n=8, k=4097, packing="onlinepackedb"))
  assert labels["rank"] == "batched" and labels["geometry"] == "gemv" and labels["k_band"] == "long_k"
  assert labels["timed_boundary"] == "explicit_online_pack+compute"
  assert labels["pipeline"] == "plain_native"  # A fusion registration can be plain.
  assert explain.classification(row(memory_a="fp32"))["pipeline"] == "dtype_conversion"
  assert explain.classification(row(output_pipeline="BiasRelu"))["pipeline"] == "epilogue"
  assert explain.classification(row(packing="packedab"))["timed_boundary"] == "prepacked_compute"
  assert explain.classification(row(packing="onlinepackedb"), "packing")["timed_boundary"] == "standalone_packing"


def test_only_consistent_passed_records_enter_performance_statistics():
  plan = {
    "executions": [row(), row(execution_id="b")],
    "architecture": "x86",
    "snapshot": "frozen",
    "label": "default",
    "timing": {"stages": ["compute", "packing"]},
  }
  record = {"status": "passed", "median_ns": {"candidate": 200, "cpp": 100}, "candidate_over_cpp": 2}
  ledger = {
    "architecture": "x86",
    "snapshot": "frozen",
    "label": "default",
    "rows": [
      {"execution_id": "a", "build": "built", "stages": {"compute": record}},
      {"execution_id": "b", "build": "oom", "stages": {"compute": {"status": "failed"}}},
    ],
  }
  result = explain.explain(plan, ledger)
  compute = next(g for g in result["groups"] if g["labels"]["stage"] == "compute")
  assert compute["registration_weighted"]["passed"] == 1
  assert compute["status"] == {"passed": 1, "failed": 1}
  packing = next(g for g in result["groups"] if g["labels"]["stage"] == "packing")
  assert packing["applicable_registration_rows"] == 0
  record["candidate_over_cpp"] = 3
  with pytest.raises(ValueError, match="duration/ratio mismatch"):
    explain.explain(plan, ledger)
