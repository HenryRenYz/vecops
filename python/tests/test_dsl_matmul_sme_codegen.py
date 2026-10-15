# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source matrix recipes are independent of private native schedulers."""

import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, assert_source_algorithms_lowered
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.environment import freeze


@pytest.mark.parametrize("target,cost_target", (("SME", "AVX512"), ("AVX512", "SME"), ("generic", "SME")))
def test_matmul_codegen_architecture_is_independent_of_cost_overrides(target, cost_target):
  M, N, K = (T.dynamic(lower=1) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    T.dot((A, B), C)

  options = {"cpu_cost_target": cost_target}
  environment = freeze(options, target=target)
  assert environment.codegen_target.architecture == {"SME": "arm", "AVX512": "x86", "generic": "generic"}[target]
  artifact = CPUBackend().lower(kernel.ir, (), target=target, options=options)
  assert_source_algorithms_lowered(artifact.backend_ir)
  assert "dsl_recipe_dot_" in artifact.source
  assert "vecops::ops::Matmul" not in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


def test_frozen_codegen_target_participates_in_environment_identity():
  arm = freeze({"cpu_cost_target": "generic"}, target="SME")
  x86 = freeze({"cpu_cost_target": "generic"}, target="AVX512")
  assert arm.target == x86.target
  assert arm.fingerprint != x86.fingerprint


@pytest.mark.parametrize("dtype,acc_dtype", (("bfloat16", "float32"), ("float16", "float32"), ("int8", "int32")))
@pytest.mark.parametrize("column_major", (False, True))
@pytest.mark.parametrize("step", (-53, 53))
def test_source_dot_seed_and_epilogue_support_signed_output_layout(dtype, acc_dtype, column_major, step):
  stride = (1, step) if column_major else (step, 1)

  @T.template
  def kernel(
    A: T.Tensor((25, 33), dtype=dtype),
    B: T.Tensor((33, 41), (1, 33), dtype),
    D: T.Tensor((1, 41), dtype=acc_dtype),
    C: T.Tensor((25, 41), stride, acc_dtype, access="output"),
  ):
    value = T.dot((A, B), C, acc=T.load(D), implementation="sme")
    T.store(C, T.max(value, 0))

  artifact = CPUBackend().lower(kernel.ir, (), target="SME")
  assert_source_algorithms_lowered(artifact.backend_ir)
  assert "dsl_recipe_dot_" in artifact.source
  assert "vecops::vec::max(" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize(
  "dtype,acc_dtype", (("bfloat16", "float32"), ("float16", "float32"), ("int8", "int32"), ("uint8", "int32"))
)
@pytest.mark.parametrize("k", (17, 19))
def test_sme_packed_tail_is_checked_and_uses_named_registers(dtype, acc_dtype, k):
  @T.template
  def kernel(
    AP: T.Tensor((1024,), dtype=dtype),
    BP: T.Tensor((1024,), dtype=dtype),
    C: T.Tensor((2, 3), dtype=acc_dtype, access="output"),
  ):
    a = T.packed(AP, (2, k), side="A", format="sme")
    b = T.packed(BP, (k, 3), side="B", format="sme")
    T.dot((a, b), C, acc_dtype=acc_dtype, implementation="vector")

  artifact = CPUBackend().lower(kernel.ir, (), target="SME")
  assert "packed input buffer too small" in artifact.source
  assert "pack_offset" in artifact.source
  assert_source_algorithms_lowered(artifact.backend_ir)
  assert_no_vec_mask_arrays(artifact.source)
