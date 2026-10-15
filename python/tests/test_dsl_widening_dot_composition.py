# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Grouped-dot geometry composes with broadcasting and checked Debug shapes."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("grouped-dot-composition")
  return make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build")


def broadcast_template(*, singleton, dynamic=False):
  k = T.dynamic(lower=0) if dynamic else (3 if singleton else 16)
  shape = (19,) if singleton else (19, 4)
  strides = (1,) if singleton else (1, 21)

  @T.template
  def kernel(
    A: T.Tensor((k,), dtype="int8"),
    B: T.Tensor((k,), dtype="uint8"),
    C: T.Tensor(shape, strides, "int32", access="output"),
  ):
    grouped = T.widening_dot(T.load(A), T.load(B), dtype="int32")
    T.store(C, T.broadcast_to(grouped, shape))

  return kernel


@pytest.mark.parametrize("singleton,dynamic", ((True, False), (True, True), (False, False)))
def test_grouped_result_broadcasts_singleton_and_absent_vector_axis(compiler, singleton, dynamic):
  count = 3 if singleton else 16
  a = (np.arange(count) * 3 - 17).astype(np.int8)
  b = (np.arange(count) * 7 + 129).astype(np.uint8)
  expected = np.pad(a.astype(np.int64) * b.astype(np.int64), (0, (-count) % 4)).reshape(-1, 4).sum(axis=1)
  shape = (19,) if singleton else (19, 4)
  raw = np.full((21,) if singleton else (4, 21), 0x12345678, dtype=np.int32)
  output = raw[1:-1] if singleton else raw[:, 1:-1].T
  compiled = T.compile(broadcast_template(singleton=singleton, dynamic=dynamic), compiler=compiler)
  compiled.operator.native(a, b, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, np.broadcast_to(expected.astype(np.int32), shape))
  np.testing.assert_array_equal(raw[[0, -1]] if singleton else raw[:, [0, -1]], 0x12345678)


def sliced_template():
  @T.template
  def kernel(
    A: T.Tensor((2, 16), dtype="int8"),
    B: T.Tensor((32,), dtype="uint8"),
    C: T.Tensor((3, 2), dtype="int32", access="output"),
  ):
    a = T.reshape(T.transpose(T.load(A)), (32,))
    grouped = T.widening_dot(a, T.load(B), dtype="int32")
    T.store(C, T.transpose(T.reshape(grouped[1:7], (2, 3))))

  return kernel


def test_grouped_dot_source_transpose_and_consumer_slice_reshape(compiler):
  a = (np.arange(32).reshape(2, 16) * 5 - 83).astype(np.int8)
  b = (np.arange(32) * 11 + 17).astype(np.uint8)
  grouped = (a.T.reshape(-1).astype(np.int64) * b.astype(np.int64)).reshape(-1, 4).sum(axis=1)
  output = np.empty((3, 2), dtype=np.int32)
  compiled = T.compile(sliced_template(), compiler=compiler)
  compiled.operator.native(a, b, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, grouped[1:7].reshape(2, 3).T.astype(np.int32))


def independent_lengths_template(*, reduce=False):
  ka, kb = T.dynamic(lower=0), T.dynamic(lower=0)
  output_shape = (1,) if reduce else ((ka + 3) // 4,)

  @T.template
  def kernel(
    A: T.Tensor((ka,), dtype="uint8"),
    B: T.Tensor((kb,), dtype="uint8"),
    C: T.Tensor(output_shape, dtype="uint32", access="output"),
  ):
    grouped = T.widening_dot(T.load(A), T.load(B), dtype="uint32")
    if reduce:
      T.store(C, T.full((1,), T.reduce_add(grouped), dtype="uint32"))
    else:
      T.store(C, grouped)

  return kernel


@pytest.mark.parametrize("ka,kb", ((3, 1), (0, 1), (1, 0)))
def test_debug_rejects_independent_mismatched_lengths(compiler, ka, kb):
  compiled = T.compile(independent_lengths_template(), compiler=compiler)
  with pytest.raises(AssertionError, match="widening_dot input length mismatch"):
    compiled.interpret(
      np.ones(ka, dtype=np.uint8), np.ones(kb, dtype=np.uint8), np.empty((ka + 3) // 4, dtype=np.uint32)
    )


@pytest.mark.parametrize("count", (0, 1, 5))
def test_debug_grouped_empty_and_partial_groups(compiler, count):
  a = np.arange(count, dtype=np.uint8) + 3
  b = np.arange(count, dtype=np.uint8) + 7
  expected = np.pad(a.astype(np.int64) * b.astype(np.int64), (0, (-count) % 4)).reshape(-1, 4).sum(axis=1)
  output = np.empty((count + 3) // 4, dtype=np.uint32)
  compiled = T.compile(independent_lengths_template(), compiler=compiler)
  compiled.interpret(a, b, output)
  np.testing.assert_array_equal(output, expected.astype(np.uint32))


def test_debug_grouped_integer_reduction_keeps_uint32_modulo_width(compiler):
  count = 66053
  compiled = T.compile(independent_lengths_template(reduce=True), compiler=compiler)
  output = np.empty(1, dtype=np.uint32)
  compiled.interpret(np.full(count, 255, dtype=np.uint8), np.full(count, 255, dtype=np.uint8), output)
  np.testing.assert_array_equal(output, np.asarray([(count * 255 * 255) & 0xFFFFFFFF], dtype=np.uint32))


@pytest.mark.parametrize("count", (1, 2))
def test_bf16_runtime_singleton_group_broadcast(compiler, count):
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  k = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((k,), dtype="bfloat16"),
    B: T.Tensor((k,), dtype="bfloat16"),
    C: T.Tensor((19,), dtype="float32", access="output"),
  ):
    T.store(C, T.broadcast_to(T.widening_dot(T.load(A), T.load(B)), (19,)))

  a = np.asarray([0.5, 0.25][:count], dtype=bf16)
  b = np.asarray([2, 4][:count], dtype=bf16)
  output = np.empty(19, dtype=np.float32)
  compiled = T.compile(kernel, compiler=compiler)
  compiled.operator.native(a, b, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, np.full(19, np.sum(a.astype(np.float32) * b.astype(np.float32))))
