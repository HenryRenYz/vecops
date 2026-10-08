# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""A source registration is executed exactly, never replaced by a shape list."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import dsl_matmul_source_catalog as runner
from vecops.language._compiler.frontend.parser import lower

CASES = runner.load_cases(svl_bytes=64)
EXECUTIONS = tuple(runner.scenarios(CASES))
SCHEMAS = {}
for _scenario in EXECUTIONS:
  if _scenario.mode == "Dynamic" or _scenario.case.catalog == "MatmulOrientation":
    SCHEMAS.setdefault(_scenario.compilation_key, _scenario)


def test_numerical_diagnostic_identifies_reference_failure_before_dsl():
  output = np.zeros(2, dtype=np.float32)

  def invalid_cpp():
    output[:] = (1, 0)

  def uncalled_dsl():
    pytest.fail("a rejected C++ reference must not become a DSL validation result")

  with pytest.raises(AssertionError, match="C\\+\\+ reference vs independent oracle"):
    runner.validate_kernel_pair(uncalled_dsl, invalid_cpp, output, np.ones(2), tolerance=8e-4)


@pytest.mark.parametrize("dtype", (np.int32, np.float32))
def test_numerical_diagnostic_identifies_dsl_failure_without_weakening_comparison(dtype):
  output = np.zeros(2, dtype=dtype)

  def valid_cpp():
    output[:] = 1

  def invalid_dsl():
    output[:] = (1, 2)

  with pytest.raises(AssertionError, match="DSL vs C\\+\\+ reference"):
    runner.validate_kernel_pair(invalid_dsl, valid_cpp, output, np.ones(2), tolerance=8e-4)

  # An integer error must stay an exact failure even with a very loose
  # floating-point oracle tolerance, as in the pre-existing runner.
  if np.issubdtype(dtype, np.integer):
    with pytest.raises(AssertionError, match="DSL vs C\\+\\+ reference"):
      runner.validate_kernel_pair(invalid_dsl, valid_cpp, output, np.ones(2), tolerance=100)


@pytest.mark.parametrize("dtype", (np.int32, np.float32))
def test_numerical_pair_calls_reference_then_dsl_with_unchanged_success_contract(dtype):
  output = np.zeros(2, dtype=dtype)
  calls = []

  def cpp():
    calls.append("cpp")
    output[:] = 3

  def dsl():
    calls.append("dsl")
    output[:] = 3

  runner.validate_kernel_pair(dsl, cpp, output, np.full(2, 3), tolerance=8e-4)
  assert calls == ["cpp", "dsl"]
  np.testing.assert_array_equal(output, (3, 3))


@pytest.fixture(autouse=True)
def arm_packing_geometry(monkeypatch):
  monkeypatch.setattr(runner, "packing_geometry", lambda dtype: ("sme", 16 if dtype == "float64" else 32, 1))


def test_every_registration_metadata_and_packing_axis_is_expanded_exactly():
  expected = sum(
    len(case.metadata) * len(case.input_modes) for case in CASES if case.catalog in runner.COHORTS.values()
  )
  assert len(EXECUTIONS) == expected
  assert len({scenario.execution_id for scenario in EXECUTIONS}) == expected
  assert {s.case.catalog for s in EXECUTIONS} == set(runner.COHORTS.values())
  packed_batches = [s for s in EXECUTIONS if s.case.catalog == "MatmulPackedExpanded" and s.batched]
  assert len(packed_batches) == 81
  assert all(s.shared_b for s in packed_batches)
  assert any(s.online for s in packed_batches)
  assert any(s.pack_a and s.pack_b for s in EXECUTIONS)
  assert any(s.case.native_a == "int8" and s.case.native_b == "uint8" for s in EXECUTIONS)


@pytest.mark.parametrize("scenario", tuple(SCHEMAS.values()), ids=lambda s: s.execution_id)
def test_every_unique_source_schema_and_offline_packer_parses(scenario):
  scenario.validate()
  pair, packing = runner.source_templates(scenario)
  assert lower(pair[0].ir).body.statements
  assert pair[1].ir.source.name == "dsl_matmul_source_reference.cpp"
  assert "SwapAB" in {getattr(capture, "name", None) for capture in pair[1].ir.captures}
  if packing:
    assert lower(packing[0].ir).body.statements
    assert packing[1].ir.source == pair[1].ir.source
  assert len(runner.source_metadata(scenario)) == 13


def test_family_policy_reference_is_explicit_and_unknown_selector_fails():
  assert {s.reference[0] for s in EXECUTIONS if s.case.catalog == "MatmulFamilyExpanded"} == set(runner.FAMILIES)
  assert {s.reference[1] for s in EXECUTIONS if s.case.catalog == "MatmulPolicyExpanded"} == set(runner.POLICIES)
  scenario = next(s for s in EXECUTIONS if s.case.catalog == "MatmulFamilyExpanded")
  with pytest.raises(ValueError, match=r"unresolved C\+\+ family selector"):
    replace(scenario, case=replace(scenario.case, configuration="UnknownFamily")).validate()


def test_reference_keeps_native_rank_three_and_timing_boundaries():
  source = (ROOT / "benchmarks/dsl_matmul_source_reference.cpp").read_text()
  assert "family_selection::Require<kernel_family::SmallVector>" in source
  assert "family_selection::Require<kernel_family::PackedDot>" in source
  assert "family_selection::Require<kernel_family::RuntimeQuantInt8>" in source
  assert "operation(execution, m, n, k, a, b, prior..., out)" in source
  assert "if constexpr (spec::OnlyPack)" in source
  assert "if constexpr (spec::OnlinePack)" in source
  assert "ops::matmul_pack" in source
  # No per-batch loop replaces the optimized public Matmul reference.
  assert "batch++" not in source and "++batch" not in source


def test_geometry_and_user_level_batch_algorithm_have_distinct_execution_ids():
  scenario = next(s for s in EXECUTIONS if s.batched and s.shared_b)
  assert scenario.execution_id != replace(scenario, batch_strategy="serial").execution_id
  assert scenario.execution_id != replace(scenario, case=replace(scenario.case, svl_bytes=32)).execution_id


def test_manifest_verification_rejects_missing_failed_duplicate_and_stale_rows(tmp_path):
  scenarios = EXECUTIONS[:2]
  audit = runner.audit_document(CASES, scenarios)
  ids = [s.execution_id for s in scenarios]
  audit.update(selected_execution_ids=ids, results={ids[0]: {"status": "passed"}})
  path = tmp_path / "results.json"

  def verify():
    path.write_text(json.dumps(audit))
    return runner.verify_manifests([path], scenarios, runner.audit_document(CASES, scenarios))

  result = verify()
  assert result["missing"] == [ids[1]] and not result["complete"]
  audit["results"][ids[1]] = {"status": "compile_failed"}
  assert verify()["failed"] == [ids[1]]
  audit["results"][ids[1]] = {"status": "passed"}
  assert verify()["complete"]
  assert runner.verify_manifests([path, path], scenarios, audit)["duplicates"] == ids
  audit["runner_hashes"] = {}
  assert verify()["stale"] == [str(path)]


def test_selection_rejects_unknown_and_partially_matched_filters():
  scenario = EXECUTIONS[0]
  options = {"cohorts": list(runner.COHORTS), "modes": ["Const", "Dynamic"], "ranks": [2, 3]}
  shape = f"{scenario.case.m}x{scenario.case.n}x{scenario.case.k}"
  assert runner.select_scenarios(EXECUTIONS, **options, shapes=[shape])
  for filters, label in (
    ({"shapes": [shape, "999x997x991"]}, "shape filters"),
    ({"native": [scenario.case.native_a, "unknown"]}, "native dtype filters"),
    ({"packing": ["raw", "unknown"]}, "packing filters"),
    ({"cohorts": ["orientation"], "ranks": [3]}, "zero executions"),
  ):
    with pytest.raises(ValueError, match=label):
      runner.select_scenarios(EXECUTIONS, **(options | filters))


def test_empty_list_and_verification_cannot_report_success():
  with pytest.raises(ValueError, match="shape filters"):
    runner.main(["--list", "--shape", "999x997x991", "--svl-bytes", "64"])
  with pytest.raises(ValueError, match="zero executions"):
    runner.verify_manifests([], [], {})


@pytest.mark.parametrize("flag, expected", [(None, None), ("--runtime-quant", True), ("--no-runtime-quant", False)])
def test_runtime_quant_cli_is_explicit_only_and_retained_in_manifest(monkeypatch, capsys, flag, expected):
  cases = tuple(
    case
    for case in CASES
    if case.catalog == runner.COHORTS["family"] and case.input_pipeline_a == "RuntimePerRowAsymmetricQuantize"
  )
  assert len(cases) == 5
  monkeypatch.setattr(runner, "load_cases", lambda **_: cases)
  monkeypatch.setattr(runner.vecops, "Compiler", lambda **_: pytest.fail("audit must not build"))
  args = ["--list", "--cohort", "family", "--native", "uint8", "--level", "dot", "matmul"]
  if flag:
    args.append(flag)
  assert runner.main(args) == 0
  audit = json.loads(capsys.readouterr().out)
  assert audit["selected"] == 20  # Five source registrations, both metadata modes and both levels.
  assert not audit["executed"] and audit["completed"] == 0
  options = audit["pass_options"]
  if expected is None:
    assert "cpu_sve_runtime_quant" not in options
  else:
    assert options["cpu_sve_runtime_quant"] is expected
  assert options["cpu_matmul_fusion"] == "auto"
  assert options["cpu_matmul_pipeline"] == "auto"
  # The experimental DSL flag must not rewrite the actual C++ family selectors.
  assert {row["reference_family"] for row in audit["executions"]} == {"Automatic", "RuntimeQuantInt8"}


def test_manifest_verification_does_not_mix_runtime_quant_experiments(tmp_path):
  scenarios = EXECUTIONS[:1]
  audit = runner.audit_document(CASES, scenarios)
  key = scenarios[0].execution_id
  audit.update(
    pass_options={"cpu_sve_runtime_quant": True},
    selected_execution_ids=[key],
    results={key: {"status": "passed"}},
  )
  path = tmp_path / "runtime-quant-enabled.json"
  path.write_text(json.dumps(audit))
  assert runner.verify_manifests([path], scenarios, audit)["complete"]
  identity = audit | {"pass_options": {"cpu_sve_runtime_quant": False}}
  result = runner.verify_manifests([path], scenarios, identity)
  assert result["stale"] == [str(path)] and not result["complete"]
