# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Streaming source-level vector matmul covers dynamic geometry and fusion."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, assert_source_algorithms_lowered, make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._library.matmul import vector_dot_into


@T.func
def finish(value, row, column):
  return T.max(value * 0.5 + 1, 0)


def template(dtype="float32", *, seed=False, affine=False):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype=dtype),
    B: T.Tensor((k, n), dtype=dtype),
    D: T.Tensor((1, n), dtype="float32"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    a, b = T.load(A), T.load(B)
    if affine:
      a, b = a * 2 + 1, b * 2 + 1
    initial = T.load(D) if seed else None
    vector_dot_into((a, b), C, acc=initial, epilogue=finish)

  return kernel


def test_streaming_vector_driver_is_ordinary_ir_not_opaque_dot():
  artifact = CPUBackend().lower(template(seed=True, affine=True).ir, (), target="AVX512")
  assert_source_algorithms_lowered(artifact.backend_ir)
  assert artifact.backend_ir.states and all(plan.pieces for plan in artifact.backend_ir.states.values())
  assert "vecops::vec::fmadd" in artifact.source
  assert "clipped SIMD extent" in artifact.source
  assert "vecops::vec::scatter(" not in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("seed,affine", [(False, False), (True, False), (True, True)])
def test_native_streaming_vector_driver_tails_empty_shapes_and_epilogue(dtype, seed, affine, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(
    template(dtype, seed=seed, affine=affine),
    compiler=compiler,
    pass_options={"cpu_ordered_loop_unroll": True, "cpu_state_vector_reuse": True},
  )
  rng = np.random.default_rng(17)
  for m, n, k in ((0, 17, 3), (7, 0, 3), (7, 37, 0), (1, 1, 17), (7, 37, 17), (8, 32, 31)):
    a = (rng.integers(-3, 4, size=(m, k)) / 8).astype(dtype)
    b = (rng.integers(-3, 4, size=(k, n)) / 8).astype(dtype)
    initial = (rng.integers(-3, 4, size=(1, n)) / 8).astype(np.float32)
    backing = np.full(m * n + 8, -19, dtype=np.float32)
    c = np.ndarray((m, n), dtype=np.float32, buffer=backing, offset=3 * 4, strides=(n * 4, 4))
    x, y = a, b
    if affine:
      x, y = (a * 2 + 1).astype(dtype), (b * 2 + 1).astype(dtype)
    expected = x.astype(np.float32) @ y.astype(np.float32)
    if seed:
      expected += initial
    expected = np.maximum(expected * 0.5 + 1, 0)
    kernel.operator.native(a, b, initial, c, **kernel._core.specialization_values)
    np.testing.assert_array_equal(c, expected)
    np.testing.assert_array_equal(backing[:3], np.full(3, -19, np.float32))
    np.testing.assert_array_equal(backing[3 + m * n :], np.full(5, -19, np.float32))


def test_native_streaming_vector_driver_preserves_overlapping_input(tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(template(seed=True, affine=True), compiler=compiler)
  rng = np.random.default_rng(29)
  a = (rng.integers(-3, 4, size=(7, 37)) / 8).astype(np.float32)
  b = (rng.integers(-3, 4, size=(37, 37)) / 8).astype(np.float32)
  seed = (rng.integers(-3, 4, size=(1, 37)) / 8).astype(np.float32)
  expected = np.maximum(((a.copy() * 2 + 1) @ (b * 2 + 1) + seed) * 0.5 + 1, 0)
  kernel.operator.native(a, b, seed, a, **kernel._core.specialization_values)
  np.testing.assert_array_equal(a, expected)


def test_value_disjointness_expands_all_producer_reads_without_recomputation():
  @T.template
  def kernel(
    A: T.Tensor((4, 17), dtype="float32"),
    B: T.Tensor((4, 17), dtype="float32"),
    C: T.Tensor((4, 17), dtype="float32", access="output"),
  ):
    x = T.load(A) * 2 + T.load(B)
    if T.view_disjoint(x, C):
      T.store(C, x)

  from vecops.language._compiler import ir as I
  from vecops.language._compiler.frontend.parser import lower
  from vecops.language._compiler.ir.traversal import nodes

  ir = lower(kernel.ir)
  guards = [node for node in nodes(ir.body) if isinstance(node, I.Expr) and node.op == "view_disjoint"]
  assert {tuple(view.storage for view in guard.data) for guard in guards} == {("A", "C"), ("B", "C")}
