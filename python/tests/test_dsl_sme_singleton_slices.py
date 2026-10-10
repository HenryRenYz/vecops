# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Narrow singleton memory work without narrowing the native ZA instruction."""

import platform

import pytest
from test_dsl_sme_packed_ab_col import _OPTIONS, _run_native_column_cases, _template
from test_dsl_sme_packed_col import _template as _single_packed_template
from test_dsl_sme_packed_row import _run_native_row_cases
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def _lower(template, *, enabled=None, vl=64, svl=64):
  options = {**_OPTIONS, "_vector_state": VectorState(vl, svl)}
  if enabled is not None:
    options["cpu_sme_narrow_singleton_slices"] = enabled
  return CPUBackend().lower(template.ir, (), target="SME", options=options).source


@pytest.mark.parametrize("level", ("dot", "matmul", "blocked"))
@pytest.mark.parametrize("shape", ((64, 1, 257), (1, 64, 257), (13, 1, 5), (64, 2, 257), None))
def test_default_and_false_keep_identical_generated_source(level, shape):
  template = _template(shape=shape, level=level)
  assert _lower(template) == _lower(template, enabled=False)


@pytest.mark.parametrize("shape", ((1, 64, 257), (64, 2, 257), None))
def test_other_horizontal_extents_and_dynamic_one_remain_unchanged(shape):
  template = _template(shape=shape)
  assert _lower(template, enabled=True) == _lower(template)


@pytest.mark.parametrize("dtype,shape", (("float16", (1, 16, 257)), ("bfloat16", (2, 64, 257))))
def test_other_unpacked_operand_shapes_and_dtypes_keep_original_source(dtype, shape):
  template = _single_packed_template(dtype, shape=shape, transformed=True)
  assert _lower(template, enabled=True) == _lower(template)


def test_native_singleton_spatial_input_preserves_row_fusion_alias_and_tails(tmp_path):
  if platform.machine().lower() not in {"aarch64", "arm64"} or get_vector_state().sme_bytes is None:
    pytest.skip("singleton spatial-input acceptance requires Native SME")
  _run_native_row_cases("bfloat16", tmp_path, options={**_OPTIONS, "cpu_sme_narrow_singleton_slices": True})


def test_native_singleton_slices_fusion_alias_split_k_and_dynamic_fallback(tmp_path):
  if platform.machine().lower() not in {"aarch64", "arm64"} or get_vector_state().sme_bytes is None:
    pytest.skip("singleton ZA slice acceptance requires Native SME")
  # Reuse all 144 original exact dyadic/canary calls, including negative
  # strides, alias/broadcast priors, full/tail K, split-K and the Dynamic
  # non-specialized route. No oracle, metadata or tolerance is weakened.
  _run_native_column_cases(tmp_path, options={**_OPTIONS, "cpu_sme_narrow_singleton_slices": True})
