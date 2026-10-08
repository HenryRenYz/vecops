# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Narrow singleton memory work without narrowing the native ZA instruction."""

import platform
import re

import pytest
from dsl_test_utils import assert_no_vec_mask_arrays
from test_dsl_matmul_sve_optimized import _cross
from test_dsl_sme_family_owners import _owners
from test_dsl_sme_packed_ab_col import _OPTIONS, _run_native_column_cases, _template, _wrapper
from test_dsl_sme_packed_col import _template as _single_packed_template
from test_dsl_sme_packed_col import _wrapper as _single_packed_wrapper
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


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64), (16, 256)))
@pytest.mark.parametrize("level", ("dot", "matmul", "blocked"))
def test_constant_column_narrows_memory_but_keeps_native_za_geometry(vl, svl, level):
  template = _template(shape=(4 * (svl // 4), 1, 257), level=level)
  source = _lower(template, enabled=True, vl=vl, svl=svl)
  assert "proven singleton ZA slice" in source
  assert "FixedTag<float, 1>" in source
  assert "ScalableTag<float, 0>" in source
  # The public ZA read argument remains a native word, not FixedTag<T,1>.
  fixed_tags = re.findall(r"using (\w+) = vecops::vec::FixedTag<float, 1>;", source)
  read_tags = re.findall(r"sme::read_(?:hor|ver)<\d+>\((\w+)\{\}", source)
  assert fixed_tags and read_tags and set(fixed_tags).isdisjoint(read_tags)
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("shape", ((1, 64, 257), (64, 2, 257), None))
def test_other_horizontal_extents_and_dynamic_one_remain_unchanged(shape):
  template = _template(shape=shape)
  assert _lower(template, enabled=True) == _lower(template)


@pytest.mark.parametrize("value", (1, 0, "true", (), []))
def test_option_is_strict_bool(value):
  with pytest.raises(ValueError, match="cpu_sme_narrow_singleton_slices must be bool"):
    _lower(_template(shape=(64, 1, 257)), enabled=value)


@pytest.mark.parametrize(
  "shape,level,vl,svl", (((64, 1, 257), "dot", 64, 64), ((13, 1, 5), "dot", 64, 16), ((64, 1, 257), "blocked", 16, 64))
)
def test_real_cross_singleton_column_owners_keep_no_calls_and_no_vector_spills(tmp_path, shape, level, vl, svl):
  source = _lower(_template(shape=shape, level=level), enabled=True, vl=vl, svl=svl)
  assembly = _cross(tmp_path, source, lambda src: _wrapper(src, shape=shape), vl=vl, svl=svl)
  owners = _owners(assembly, vl=vl, svl=svl)
  assert owners and all(re.search(r"^\s*bfmopa\b", owner, re.MULTILINE) for owner in owners)
  for owner in owners:
    assert not re.search(r"\b(?:str|ldr)\s+[zp]\d+,\s*\[sp", owner), "real vector/predicate spill remains"


def test_constant_single_packed_row_narrows_each_unpacked_spatial_k_slot():
  template = _single_packed_template("bfloat16", shape=(1, 64, 257), transformed=True)
  assert _lower(template) == _lower(template, enabled=False)
  source = _lower(template, enabled=True)
  assert "proven singleton spatial input: one value per K slot" in source
  assert "FixedTag<vecops::bfloat16_t, 1>" in source
  assert "vecops::vec::Twice<" in source and "sme::mopa<" in source
  assert_no_vec_mask_arrays(source)


def test_real_cross_single_packed_row_avoids_wide_index_predicate_spills(tmp_path):
  shape = (1, 64, 257)
  source = _lower(_single_packed_template("bfloat16", shape=shape, transformed=True), enabled=True)
  assembly = _cross(
    tmp_path, source, lambda src: _single_packed_wrapper(src, "bfloat16", shape=shape, transformed=True), vl=64, svl=64
  )
  owners = _owners(assembly, vl=64, svl=64)
  assert owners and re.search(r"^\s*bfmopa\b", assembly, re.MULTILINE)
  for owner in owners:
    assert not re.search(r"\b(?:str|ldr)\s+[zp]\d+,\s*\[sp", owner), "real vector/predicate spill remains"


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
