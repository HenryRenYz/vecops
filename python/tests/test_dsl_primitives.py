"""Composable register operations, checked indexed memory and masks."""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-primitives")
  return make_compiler(
    target="Native", jobs=4, cache_dir=root / "cache", build_dir=root / "build"
  )


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_indexed_conversion_mask_alias(compiler, dtype):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(
      A: T.Tensor((19,), dtype=dtype), Idx: T.Tensor((19,), dtype="int64"), C: T.Tensor((19,), dtype="float32")
    ):
      indices = T.load(Idx)
      T.store(C, T.gather(A, indices, dtype="float32", mask=indices >= 0, other=-3) + 1)

    return kernel

  a = np.arange(19, dtype=dtype)
  indices = np.arange(18, -1, -1, dtype=np.int64)
  indices[2] = -100
  expected = np.where(indices >= 0, a[np.maximum(indices, 0)].astype(np.float32), -3) + 1
  k = factory()
  np.testing.assert_array_equal(k(a, indices), expected)
  if dtype == "float32":
    k(a, indices, a)
    np.testing.assert_array_equal(a, expected)
  assert "opt::indexed" in k.generated_source
  assert not re.search(r"(?:Vec|Mask)<[^;\n]+>\s+\w+\s*\[", k.generated_source)


@pytest.mark.parametrize("unique", [False, True])
def test_scatter(compiler, unique):
  @T.jit(compiler=compiler)
  def factory():
    @T.template
    def kernel(
      A: T.Tensor((19,), dtype="float32"),
      Idx: T.Tensor((19,), dtype="int64"),
      C: T.Tensor((19,), dtype="float16", access="inout"),
    ):
      idx = T.load(Idx)
      T.scatter(C, idx, T.load(A) + 2, mask=idx >= 0, unique=unique)

    return kernel

  a = np.arange(19, dtype=np.float32)
  idx = np.arange(18, -1, -1, dtype=np.int64)
  if not unique:
    idx[3:8] = 2
  idx[1] = -9
  c = np.zeros(19, dtype=np.float16)
  expected = c.copy()
  for i, index in enumerate(idx):
    if index >= 0:
      expected[index] = a[i] + 2
  factory()(a, idx, c)
  np.testing.assert_array_equal(c, expected)


def test_mask_queries_and_empty(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory(n):
    @T.template
    def kernel(A: T.Tensor((n,), dtype="float32"), C: T.Tensor((6,), dtype="int64")):
      mask = T.load(A) > 3
      C[0] = int(T.mask_all(mask))
      C[1] = int(T.mask_any(mask))
      C[2] = int(T.mask_none(mask))
      C[3] = T.mask_count(mask)
      C[4] = T.mask_first(mask)
      C[5] = T.mask_last(mask)

    return kernel

  for n in (0, 7, 37):
    a = np.arange(n, dtype=np.float32)
    np.testing.assert_array_equal(
      factory(n)(a), [n == 0, n > 4, n <= 4, max(0, n - 4), 4 if n > 4 else -1, n - 1 if n > 4 else -1]
    )


def test_value_transforms(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((3, 7), dtype="float32"), C: T.Tensor((7, 3), dtype="float32")):
      x = T.load(A) * 2 + 1
      T.store(C, T.transpose(x))

    return kernel

  a = np.arange(21, dtype=np.float32).reshape(3, 7)
  np.testing.assert_array_equal(factory()(a), (a * 2 + 1).T)


def test_local_shuffle(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((32,), dtype="float32"), C: T.Tensor((32,), dtype="float32")):
      T.store(C, T.local_shuf(T.load(A), (3, 2, 1, 0)))

    return kernel

  a = np.arange(32, dtype=np.float32)
  np.testing.assert_array_equal(factory()(a), a.reshape(-1, 4)[:, ::-1].reshape(-1))


def test_shuffle_state_and_scalar_consumer(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((7,), dtype="float32"), C: T.Tensor((7,), dtype="float32")):
      x = T.load(A)
      for i in T.serial(1):
        x = T.local_shuf(x, (3, 2, 1, 0))
      total = T.reduce(x)
      C[0] = total
      C[1] = T.element_at(T.local_shuf(T.load(A), (3, 2, 1, 0)), (5,))
      T.store(C[2:], T.load(A[2:]))

    return kernel

  a = np.arange(7, dtype=np.float32)
  np.testing.assert_array_equal(factory()(a), [17, 6, 2, 3, 4, 5, 6])


@pytest.mark.parametrize("strides", [(7, 1), (1, 3)])
def test_value_reshape(compiler, strides):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((3, 7), strides, dtype="float32"), C: T.Tensor((7, 3), dtype="float32")):
      T.store(C, T.reshape(T.load(A) * 2 + 1, (7, 3)))

    return kernel

  a = np.arange(21, dtype=np.float32).reshape(3, 7)
  if strides == (1, 3):
    a = np.asfortranarray(a)
  compiled = factory()
  np.testing.assert_array_equal(compiled(a), (a * 2 + 1).reshape(7, 3))
  if strides == (7, 1):
    assert "reshape_lane" not in compiled.generated_source


def test_indexed_negative_stride_and_bounds(compiler):
  stride = T.dynamic()

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(
      A: T.Tensor((19,), (stride,), dtype="float32"),
      Idx: T.Tensor((19,), dtype="int32"),
      C: T.Tensor((19,), dtype="float32"),
    ):
      T.store(C, T.gather(A, T.load(Idx)))

    return kernel

  a = np.arange(19, dtype=np.float32)[::-1]
  indices = np.arange(18, -1, -1, dtype=np.int32)
  np.testing.assert_array_equal(factory()(a, indices), a[indices])
  indices[17] = 19
  with pytest.raises(Exception, match="out of bounds"):
    factory()(a, indices)


def test_scalar_tuple_swap(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(C: T.Tensor((1,), dtype="int64")):
      a, b = 1, 2
      for i in T.serial(3):
        a, b = b, a
      C[0] = a * 10 + b

    return kernel

  c = np.empty(1, dtype=np.int64)
  factory()(c)
  np.testing.assert_array_equal(c, [21])


def test_word_shuffle(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((7,), dtype="float32"), C: T.Tensor((7,), dtype="float32")):
      x = T.load(A)
      for i in T.serial(3):
        x = T.shuf(x, 6 - T.arange(7))
      T.store(C, x)

    return kernel

  a = np.arange(7, dtype=np.float32)
  np.testing.assert_array_equal(factory()(a), a[::-1])
