# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source lifecycle selectors and representative paired native numerics."""

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import dsl_matmul_source_catalog as source
from dsl_matmul_artifact_ab import template_contract


def executions():
  return list(source.scenarios(source.load_cases(architecture="x86", complete_registration=True)))


def test_lifecycle_selector_is_separate_from_execution_identity():
  rows = executions()
  for scenario in rows:
    assert scenario.reference_bound_invocation == (
      scenario.case.catalog in {"MatmulScenarioExpanded", "MatmulFusionExpanded", "MatmulPackedExpanded"}
    )
  original = next(s for s in rows if s.case.catalog == "MatmulScenarioExpanded")
  # Only the compiler grouping distinguishes bind-per-call and bind-once.
  alias = replace(original, case=replace(original.case, catalog="MatmulBaseExpanded"))
  assert original.compilation_key != alias.compilation_key
  assert "ReferenceBoundInvocation" not in original.key
  frozen = ROOT / "build-matmul-full-20261014/x86/plan.json"
  if frozen.exists():
    import json

    assert {s.execution_id for s in rows} == {r["execution_id"] for r in json.loads(frozen.read_text())["executions"]}


def test_cpp_prior_is_mutable_without_changing_ordinary_dsl_prior():
  scenario = next(s for s in executions() if s.has_prior and not s.reference_compensated_pack)
  dsl, cpp = source.source_templates(scenario)[0]
  dsl_contract, _ = template_contract(dsl)
  cpp_contract, captures = template_contract(cpp)
  assert dsl_contract[0][3][7] == 1  # Read-only D.
  assert cpp_contract[0][3][7] == 3  # Original mutable benchmark tensor.
  assert captures["ReferenceBoundInvocation"] == int(scenario.reference_bound_invocation)


def test_online_compensation_scope_and_both_template_roles():
  rows = [s for s in executions() if s.reference_compensated_pack]
  assert len(rows) == 40
  assert len({s.compilation_key for s in rows}) == 26
  for pair in source.source_templates(rows[0]):
    for template in pair:
      contract, captures = template_contract(template)
      assert contract[0][3][7] == 3
      if captures:
        assert captures["ReferenceCompensatedPack"] == captures["ReferenceBoundInvocation"] == 1


@pytest.mark.parametrize("kind", ["bound_plain", "shared_a_prior", "online_sidecar"])
def test_native_reference_bound_and_sidecar_oracle(kind, tmp_path):
  import vecops
  import vecops.language as T

  rows = executions()
  if kind == "bound_plain":
    scenario = next(
      s
      for s in rows
      if s.case.catalog == "MatmulScenarioExpanded"
      and not s.batched
      and not s.has_prior
      and s.case.native_a == "bf16"
      and s.case.memory_a == "bf16"
      and s.mode == "Const"
    )
  elif kind == "shared_a_prior":
    scenario = next(
      s
      for s in rows
      if s.case.catalog == "MatmulScenarioExpanded"
      and s.shared_a
      and s.has_prior
      and s.case.output_pipeline == "Accumulate"
      and s.case.n == 16
      and s.mode == "Const"
    )
  else:
    scenario = next(
      s
      for s in rows
      if s.reference_compensated_pack
      and not s.batched
      and s.case.m == 19
      and s.packing == "onlinepackedb"
      and s.mode == "Dynamic"
    )
  compiler = vecops.Compiler(
    target="Native",
    cc=os.environ.get("VECOPS_TEST_CC", "/usr/bin/gcc"),
    cxx=os.environ.get("VECOPS_TEST_CXX", "/usr/bin/g++"),
    jobs=1,
    cache_dir=tmp_path / "cache",
    build_dir=tmp_path / "build",
  )
  kernels = []
  with vecops.collect_compile_requests() as collector:
    for pair in source.source_templates(scenario):
      if pair is None:
        continue
      compiled = tuple(T.compile(template, compiler=compiler) for template in pair)
      for kernel in compiled:
        kernel.operator(*source.source_metadata(scenario), 1, **kernel._core.specialization_values)
      kernels.append(compiled)
  vecops.compile_batch(collector.requests, parallelism=1, execution_parallelism=1)
  values, expected = source.source_arguments(scenario)
  if scenario.reference_compensated_pack:
    correction = values[3].copy()
    for kernel in kernels[1]:
      values[3][:] = np.iinfo(np.int32).min
      values[12].view(np.uint8).fill(255)
      kernel.operator.native(*values, 1, **kernel._core.specialization_values)
      np.testing.assert_array_equal(values[3], correction)
  for kernel in kernels[0]:
    kernel.operator.native(*values, 1, **kernel._core.specialization_values)
    np.testing.assert_allclose(values[2], expected, rtol=8e-4, atol=8e-4)
