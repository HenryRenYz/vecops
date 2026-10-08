# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""The paired ARM benchmark must audit every rank-two fusion/orientation schema."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import dsl_matmul_fusion_catalog as runner
from vecops.language._compiler.frontend.parser import lower


def unique_scenarios(catalog, *, batched=False):
  schemas = {}
  for case in runner.load_cases(ROOT, svl_bytes=64):
    if case.catalog == catalog and (case.batch is not None) == batched:
      scenario = runner.Scenario(case, "Dynamic", "matmul")
      schemas.setdefault(scenario.compilation_key, scenario)
  return tuple(schemas.values())


FUSION = unique_scenarios("MatmulFusionExpanded")
ORIENTATION = unique_scenarios("MatmulOrientation")
BATCHED = unique_scenarios("MatmulFusionExpanded", batched=True)


def test_pipeline_contract_and_complete_rank_two_catalog():
  runner.validate_pipeline_contract()
  assert len(FUSION) == 45
  assert len(ORIENTATION) >= 16
  assert len(BATCHED) == 90
  cases = runner.load_cases(ROOT, svl_bytes=64)
  assert sum(case.catalog == "MatmulFusionExpanded" and case.batch is None for case in cases) == 360
  assert sum(case.catalog == "MatmulFusionExpanded" and case.batch is not None for case in cases) == 180


@pytest.mark.parametrize("scenario", FUSION + ORIENTATION + BATCHED, ids=lambda s: s.key)
def test_all_registered_rank_two_schemas_parse_and_match_metadata(scenario):
  scenario.validate()
  dsl, cpp = runner.templates(scenario)
  module = lower(dsl.ir)
  assert module.body.statements
  assert cpp.ir.source is not None
  values, golden = runner.arguments(scenario)
  metadata = runner.metadata(scenario)
  assert len(values) == len(metadata) == 11
  for value, meta in zip(values, metadata):
    assert value.shape == meta.shape
    assert tuple(stride // value.dtype.itemsize for stride in value.strides) == meta.strides
  assert golden.shape == values[2].shape


@pytest.mark.parametrize(
  "pipeline", ("AsymmetricQuantize4Zp3", "AsymmetricQuantize4Zp3Zp5", "RuntimePerRowAsymmetricQuantize")
)
def test_asymmetric_compensation_is_untimed_and_centered(pipeline):
  scenario = next(s for s in FUSION if s.case.input_pipeline_a == pipeline)
  values, golden = runner.arguments(scenario)
  a, b, _, correction, _, _, _, row_quant, row_dequant, weight, zero = values
  native_a, native_b = scenario.types[4:6]
  aa = runner.transformed_input(a, native_a, pipeline, "A", row_quant, zero).astype(np.int32)
  bb = runner.transformed_input(b, native_b, scenario.case.input_pipeline_b, "B", row_quant, zero).astype(np.int32)
  zp_a = 7 if pipeline == "RuntimePerRowAsymmetricQuantize" else 3
  zp_b = 5 if pipeline == "AsymmetricQuantize4Zp3Zp5" else 0
  centered = (aa - zp_a) @ (bb - zp_b)
  np.testing.assert_array_equal(aa @ bb + correction, centered)
  expected = centered * row_dequant * weight if pipeline == "RuntimePerRowAsymmetricQuantize" else centered * 0.0625
  np.testing.assert_allclose(golden, expected, rtol=1e-6, atol=1e-6)


def test_unknown_pipeline_is_rejected_instead_of_silently_filtered():
  from dataclasses import replace

  scenario = runner.Scenario(replace(FUSION[0].case, output_pipeline="NewUnsupportedTransform"), "Dynamic", "dot")
  with pytest.raises(ValueError, match="unknown output transform"):
    scenario.validate()


def test_batched_native_reference_keeps_public_rank_three_matmul():
  source = (ROOT / "benchmarks/dsl_matmul_fusion_reference.cpp").read_text()
  assert "tensor::transpose<rank - 2, rank - 1>(B)" in source
  assert "row_batch_stride(A)" in source
  assert "operation(execution, m, n, k, a, b, prior, out)" in source
  # Independent/shared-B inputs are passed whole to native Matmul, not a
  # per-batch rank-two baseline that could discard its packing optimizations.
  assert "for (" not in source.split("void __kernel__", 1)[1].split("if constexpr", 1)[0]


def test_shared_b_flatten_is_explicit_dsl_algorithm_and_serial_is_available():
  from dataclasses import replace

  scenario = next(s for s in BATCHED if s.shared_b)
  assert scenario.flatten_batch
  serial = replace(scenario, batch_strategy="serial")
  assert not serial.flatten_batch
  assert scenario.compilation_key != serial.compilation_key
  lower(runner.templates(serial)[0].ir)


def test_shared_a_wrapper_preserves_zero_batch_strides_and_row_parameters():
  from dataclasses import replace

  scenario = next(s for s in BATCHED if s.case.input_pipeline_a == "RuntimePerRowAsymmetricQuantize")
  shared_a = replace(scenario, case=replace(scenario.case, broadcast="A"))
  assert not shared_a.flatten_batch
  lower(runner.templates(shared_a)[0].ir)
  values, _ = runner.arguments(shared_a)
  metas = runner.metadata(shared_a)
  for index in (0, 7, 8):
    assert values[index].strides[0] == metas[index].strides[0] == 0
  assert values[1].strides[0] != 0
