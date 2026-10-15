# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Dot omits optional online packs while instruction layout staging survives."""

import pytest
import vecops.language as T
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu import recipes
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._library.sme_matmul import dot_into as sme_dot_into


def public_dot_template(dtype="float32", *, repeated=False, fused=False):
  @T.template
  def kernel(
    A: T.Tensor((64, 128), dtype=dtype),
    B: T.Tensor((128, 64), (1, 128), dtype),
    C: T.Tensor((64, 64), dtype=dtype, access="output"),
  ):
    if repeated:
      for repeat in T.serial(2):
        T.dot((A, B), C, implementation="sme")
    elif fused:
      value = T.dot((A, B), C, implementation="sme")
      T.store(C, T.max(value * 0.5 + 1, 0))
    else:
      T.dot((A, B), C, implementation="sme")

  return kernel


def leaf_template(dtype, acc_dtype, *, optional):
  @T.template
  def kernel(
    A: T.Tensor((64, 129), dtype=dtype),
    B: T.Tensor((129, 64), (1, 129), dtype),
    C: T.Tensor((64, 64), dtype=acc_dtype, access="output"),
  ):
    sme_dot_into(
      (T.load(A), T.load(B)),
      C,
      acc_dtype=acc_dtype,
      raw_a=A,
      raw_b=B,
      pack_raw_inputs=optional,
    )

  return kernel


def source(kernel):
  return CPUBackend().lower(kernel.ir, (), target="SME", options={"_vector_state": VectorState(64, 64)}).source


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("repeated,fused", ((False, False), (True, False), (False, True)))
def test_public_raw_dot_never_queries_optional_packing_profitability(dtype, repeated, fused):
  generated = source(public_dot_template(dtype, repeated=repeated, fused=fused))
  assert "vecops::platform::cache_info().l2_bytes" not in generated
  assert "vecops::vec::sme::" in generated


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_instruction_leaf_requires_explicit_optional_raw_packing_policy(dtype):
  disabled = source(leaf_template(dtype, dtype, optional=False))
  enabled = source(leaf_template(dtype, dtype, optional=True))
  assert "vecops::platform::cache_info().l2_bytes" not in disabled
  assert "vecops::platform::cache_info().l2_bytes" in enabled
  assert disabled != enabled


@pytest.mark.parametrize(
  "dtype,acc_dtype", (("bfloat16", "float32"), ("float16", "float32"), ("int8", "int32"), ("int16", "int64"))
)
def test_mandatory_grouped_layout_staging_is_independent_of_optional_raw_packing(dtype, acc_dtype):
  disabled = source(leaf_template(dtype, acc_dtype, optional=False))
  enabled = source(leaf_template(dtype, acc_dtype, optional=True))
  assert disabled == enabled
  # Operand-layout staging uses SME transpose/move owners before the
  # product's MOPA owner; changing reuse policy must retain those owners.
  assert "vecops::vec::sme::" in disabled


@pytest.mark.parametrize(
  "packing,optional", (("auto", True), ("none", False), ("A", False), ("B", False), ("AB", False))
)
def test_native_matmul_selects_optional_raw_packing_only_for_auto(monkeypatch, packing, optional):
  policies = []
  original = recipes.expand_helper

  def record(helper, *args, **kwargs):
    if helper.function.__name__ == "blocked_matrix":
      policies.append(kwargs.get("keyword_arguments", {}).get("pack_raw_inputs", False))
    return original(helper, *args, **kwargs)

  monkeypatch.setattr(recipes, "expand_helper", record)

  @T.template
  def kernel(
    A: T.Tensor((64, 128), dtype="float32"),
    B: T.Tensor((128, 64), (1, 128), "float32"),
    C: T.Tensor((64, 64), dtype="float32", access="output"),
  ):
    T.matmul((A, B), C, implementation="sme", packing=packing, swap=False)

  source(kernel)
  assert policies == [optional]
