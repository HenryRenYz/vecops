# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ordinary Python integer hints must remain compile-time cache tile values."""

import pytest
import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.frontend.parser import lower


@pytest.mark.parametrize("axis", ("BM", "BN", "BK"))
def test_library_matmul_accepts_captured_python_cache_tile(axis):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  blocks = {axis: 32}
  BM, BN, BK = (blocks.get(key) for key in ("BM", "BN", "BK"))

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C, BM=BM, BN=BN, BK=BK, implementation="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert "=vecops_dsl::matmul_blocks" in source
  assert "bypass cache panels" not in source


def test_cache_tiles_from_captured_tuple_and_helper_default_are_constants():
  blocks = (32, 48, 64)

  @T.func
  def multiply(a, b, c, block_m=32):
    return T.library.matmul((a, b), c, BM=block_m, BN=blocks[1], BK=blocks[2])

  @T.template
  def kernel(
    A: T.Tensor((4, 4), dtype="float32"),
    B: T.Tensor((4, 4), dtype="float32"),
    C: T.Tensor((4, 4), dtype="float32", access="output"),
  ):
    multiply(A, B, C)

  matmuls = [
    statement.value
    for statement in lower(kernel.ir).body.statements
    if getattr(statement, "value", None) is not None and getattr(statement.value, "op", None) == "dot"
  ]
  # Before optimization, helper result aliases can reference the same dot
  # expression more than once. All references must retain the constant hints.
  assert matmuls
  assert all((value.data.block_m, value.data.block_n, value.data.block_k) == blocks for value in matmuls)


@pytest.mark.parametrize("block", (True, False, 0, -1, 0.5))
def test_captured_cache_tile_retains_positive_integer_validation(block):
  @T.template
  def kernel(
    A: T.Tensor((4, 4), dtype="float32"),
    B: T.Tensor((4, 4), dtype="float32"),
    C: T.Tensor((4, 4), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C, BM=block)

  with pytest.raises(TypeError, match="BM must be None or a positive constant"):
    lower(kernel.ir)
