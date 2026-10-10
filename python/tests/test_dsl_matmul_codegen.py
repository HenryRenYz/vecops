# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Memory layout and dispatch contracts in generated matrix schedules."""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import amx_available_for_test, make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.planning import snapshot_strides


@pytest.mark.parametrize("target", ["AVX512", "SME", "generic"])
def test_explicit_cache_sizes_are_source_loop_steps_not_native_family_parameters(target):
  @T.template
  def kernel(
    A: T.Tensor((73, 79), dtype="bfloat16"),
    B: T.Tensor((79, 83), dtype="bfloat16"),
    C: T.Tensor((73, 83), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C, BM=32, BN=48, BK=64)

  from vecops.language._compiler.ir.traversal import nodes

  artifact = CPUBackend().lower(kernel.ir, (), target=target)
  steps = {I.constant(n.step) for n in nodes(artifact.kernel.body) if isinstance(n, I.For)}
  assert {32, 48, 64} <= steps
  assert not artifact.backend_ir.matmuls
  assert "matmul_blocks(" not in artifact.source
  assert "ops::Matmul" not in artifact.source


def test_constant_vector_row_domain_eliminates_impossible_row_tail_guards():
  @T.template
  def kernel(
    A: T.Tensor((16, 17), dtype="float32"),
    B: T.Tensor((17, 19), dtype="float32"),
    C: T.Tensor((16, 19), dtype="float32", access="output"),
  ):
    T.dot((A, B), C, implementation="vector")

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  assert "dsl_recipe_dot_" in artifact.source
  assert not re.search(r"if \(\(.*row_\d+ \+ std::int64_t\{[123]\}.* < std::int64_t\{16\}", artifact.source)


def test_snapshot_preserves_unit_stride_without_padding_or_overlap():
  shape = tuple(I.literal(v) for v in (13, 17))
  for source, expected in (((1, 19), (1, 13)), ((23, 1), (17, 1)), ((-1, 19), (17, 1)), ((0, 1), (17, 1))):
    view = I.View("A", "float32", shape, tuple(I.literal(v) for v in source), I.literal(0))
    assert tuple(I.constant(v) for v in snapshot_strides(view)) == expected
  shape = tuple(I.literal(v) for v in (2, 3, 4))
  view = I.View("A", "float32", shape, tuple(I.literal(v) for v in (27, 1, 5)), I.literal(0))
  assert tuple(I.constant(v) for v in snapshot_strides(view)) == (12, 1, 3)


@pytest.mark.skipif(not amx_available_for_test(), reason="explicit AMX matmul requires an x86 AMX target")
def test_dense_packed_operands_do_not_assume_zero_padded_k(tmp_path):
  import ml_dtypes

  @T.template
  def kernel(
    A: T.Tensor((4, 33), dtype="bfloat16"),
    B: T.Tensor((33, 5), (1, 33), "bfloat16"),
    C: T.Tensor((4, 5), dtype="float32", access="output"),
  ):
    a = T.pack(A, side="A", format="dense")
    b = T.pack(B, side="B", format="dense")
    T.dot((a, b), C, implementation="amx")

  compiler = make_compiler(
    target="Native",
    jobs=4,
    cache_dir=tmp_path / "cache",
    build_dir=tmp_path / "build",
  )
  compiled = T.compile(kernel, compiler=compiler)
  rng = np.random.default_rng(59)
  a = (rng.integers(-3, 4, (4, 33)) / 8).astype(ml_dtypes.bfloat16)
  b = (rng.integers(-3, 4, (5, 33)) / 8).astype(ml_dtypes.bfloat16).T
  c = np.empty((4, 5), np.float32)
  assert "packed input buffer too small" in compiled.generated_source
  compiled.operator.native(a, b, c)
  np.testing.assert_array_equal(c, a.astype(np.float32) @ b.astype(np.float32))


@pytest.mark.skipif(not amx_available_for_test(), reason="explicit AMX matmul requires an x86 AMX target")
def test_dynamic_k_cursors_guard_zero_length_storage(tmp_path):
  import ml_dtypes

  K = T.dynamic(lower=0)
  AS = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((64, K), (AS, 1), "bfloat16"),
    B: T.Tensor((K, 64), dtype="bfloat16"),
    C: T.Tensor((64, 64), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C, implementation="amx")

  compiler = make_compiler(
    target="Native", cc="gcc", cxx="g++", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"
  )
  compiled = T.compile(kernel, compiler=compiler)
  assert "dsl_recipe_dot_" in compiled.generated_source
  rng = np.random.default_rng(61)
  for k in (0, 33, 64):
    a = (rng.integers(-3, 4, (64, k + 1)) / 8).astype(ml_dtypes.bfloat16)[:, :k]
    b = (rng.integers(-3, 4, (k, 64)) / 8).astype(ml_dtypes.bfloat16)
    c = np.empty((64, 64), np.float32)
    compiled.operator.native(a, b, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, a.astype(np.float32) @ b.astype(np.float32))


def test_matrix_alias_snapshots_preserve_column_major_inputs(tmp_path):
  import ml_dtypes

  dtype = np.dtype(ml_dtypes.bfloat16)
  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), "bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="bfloat16", access="output"),
  ):
    T.library.matmul((A, B), C)

  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  rng = np.random.default_rng(47)
  for m, n, k in ((1, 17, 13), (17, 1, 13), (9, 11, 13), (17, 19, 33)):
    for side in ("A", "B"):
      backing = np.empty(max(m * k, k * n, m * n), dtype)
      a = backing[: m * k].reshape(m, k) if side == "A" else np.empty((m, k), dtype)
      b = (backing[: k * n].reshape(n, k) if side == "B" else np.empty((n, k), dtype)).T
      a[:] = rng.integers(-3, 4, size=a.shape) / 8
      b[:] = rng.integers(-3, 4, size=b.shape) / 8
      expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(dtype)
      c = backing[: m * n].reshape(m, n)
      compiled.operator.native(a, b, c, **compiled._core.specialization_values)
      np.testing.assert_array_equal(c, expected)


def test_partially_static_amx_families_have_no_empty_instruction_tiles(tmp_path):
  import ml_dtypes

  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  requests = []
  with vecops.collect_compile_requests() as collector:
    for static_m in (True, False):
      M = 17 if static_m else T.dynamic(lower=0)
      N = T.dynamic(lower=0) if static_m else 17
      K = T.dynamic(lower=0)

      @T.template
      def kernel(
        A: T.Tensor((M, K), (K, 1), "bfloat16"),
        B: T.Tensor((K, N), (1, K), "bfloat16"),
        C: T.Tensor((M, N), dtype="float32", access="output"),
      ):
        T.library.matmul((A, B), C)

      compiled = T.compile(kernel, compiler=compiler)
      shape = (17, 8, 33) if static_m else (8, 17, 33)
      m, n, k = shape
      compiled.operator(
        vecops.TensorMeta((m, k), dtype=vecops.bfloat16),
        vecops.TensorMeta((k, n), strides=(1, k), dtype=vecops.bfloat16),
        vecops.TensorMeta((m, n), dtype=vecops.float32),
        **compiled._core.specialization_values,
      )
      requests.append((compiled, shape))
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  rng = np.random.default_rng(51)
  for compiled, (m, n, k) in requests:
    a = (rng.integers(-3, 4, (m, k)) / 8).astype(ml_dtypes.bfloat16)
    b = (rng.integers(-3, 4, (n, k)) / 8).astype(ml_dtypes.bfloat16).T
    c = np.empty((m, n), np.float32)
    compiled.operator.native(a, b, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, a.astype(np.float32) @ b.astype(np.float32))
