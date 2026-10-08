# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source-level contracts for visible normalization, workspace and SDPA."""

import re

import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def test_row_statistics_are_hoisted_out_of_elementwise_consumers():
  @T.template
  def kernel(
    A: T.Tensor((17, 65), dtype="float32"),
    Mask: T.Tensor((17, 65), dtype="int32"),
    C: T.Tensor((17, 65), dtype="float32", access="output"),
  ):
    T.store(C, T.library.masked_softmax(A, Mask) * 2 + 1)

  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  assert "dsl_o_map_1_row" in source
  assert "safe_total" in source
  assert "axis_acc" not in source
  assert "ops::Softmax" not in source


def test_streamed_attention_keeps_named_vectors_and_inline_callbacks():
  @T.template
  def kernel(
    Q: T.Tensor((3, 16), dtype="bfloat16"),
    K: T.Tensor((13, 16), dtype="bfloat16"),
    V: T.Tensor((13, 7), dtype="bfloat16"),
    C: T.Tensor((3, 7), dtype="float32", access="output"),
  ):
    T.library.scaled_dot_product_attention(Q, K, V, C, block_k=4, causal=True)

  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  assert not re.search(r"(?:Vec|Mask)<[^;\n]+>\s+\w+\s*\[", source)
  assert not re.search(r"(?:array|vector)<\s*vecops::vec::(?:Vec|Mask)", source)
  for line in source.splitlines():
    if "[&](" in line:
      assert "VECOPS_INLINE_LAMBDA" in line or "VECOPS_KERNEL_LAMBDA" in line
  assert "ops::Attention" not in source
  assert "vecops::vec::amx::dot<" in source


def test_executor_only_kernels_keep_lightweight_source_contract():
  @T.template
  def kernel(C: T.Tensor((11,), dtype="int64", access="output")):
    for i in T.grid(11):
      C[i] = i

  source = CPUBackend().lower(kernel.ir, ()).source
  assert "#define VECOPS_KERNEL_EXECUTION_CONTEXT 1" in source
  assert "WorkspaceContext& dsl_workspace" not in source


def test_constant_helper_returns_keep_compile_time_identity():
  @T.func
  def constant():
    count = 7
    return count == 7

  @T.template
  def kernel(C: T.Tensor((1,), dtype="int64", access="output"), stop: int):
    T.static_assert(constant())
    value = 0
    for i in T.serial(stop):
      value = value + i
    C[0] = value

  source = CPUBackend().lower(kernel.ir, ()).source
  assert "helper_result" not in source
  assert re.search(r"auto dsl_f_value_\d+ = std::int64_t\{0\}", source)


def test_constant_between_grids_needs_no_publication_barrier():
  @T.template
  def kernel(
    C: T.Tensor((11,), dtype="float32", access="output"),
    D: T.Tensor((11,), dtype="float32", access="output"),
  ):
    for i in T.grid(11):
      C[i] = float(i)
    scale = 3.0
    for i in T.grid(11):
      D[i] = float(i) * scale

  source = CPUBackend().lower(kernel.ir, (), options={"cpu_grid_regions": "template"}).source
  assert "shared grid team" in source
  assert "publish single-worker scalar" not in source
