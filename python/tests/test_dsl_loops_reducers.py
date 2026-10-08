# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Local execution layouts and general ordered folds."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-loops-reducers")
  return make_compiler(
    target="Native", jobs=4, cache_dir=root / "cache", build_dir=root / "build"
  )


@pytest.mark.parametrize("fixed", [True, False])
def test_explicit_simd(compiler, fixed):
  descriptor = T.Vector("float32", N=16) if fixed else T.Vector("float32")

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype="float32"), C: T.Tensor((37,), dtype="float32")):
      for index in T.simd(37, vector=descriptor):
        T.scatter(C, index, T.gather(A, index) * 2 + 1, unique=True)

    return kernel

  a = np.arange(37, dtype=np.float32)
  np.testing.assert_array_equal(factory()(a), a * 2 + 1)


def test_local_layout(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((3, 37), dtype="float32"), C: T.Tensor((3, 37), dtype="float32")):
      for i, j in T.parallel(3, 37, layout=T.ExecutionLayout((0, 1), vector_axis=1, vector=T.Vector("float32", N=16))):
        T.scatter(C[i, :], j, T.gather(A[i, :], j) + i, unique=True)

    return kernel

  a = np.arange(111, dtype=np.float32).reshape(3, 37)
  np.testing.assert_array_equal(factory()(a), a + np.arange(3, dtype=np.float32)[:, None])


@pytest.mark.parametrize("axis", [None, 1])
def test_general_tuple_reducer(compiler, axis):
  @T.reducer(identity=(float("-inf"), -1), associative=True)
  def argmax(a, b):
    if b[0] > a[0]:
      return b
    return a

  output_shape = (1,) if axis is None else (3,)

  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((3, 7), dtype="float32"), C: T.Tensor(output_shape, dtype="int64")):
      x = T.load(A)
      idx = T.broadcast_to(T.arange(7), (3, 7))
      pair = T.reduce((x, idx), reducer=argmax, axis=axis)
      T.store(C, pair[1])

    return kernel

  a = np.arange(21, dtype=np.float32).reshape(3, 7)
  a[1, 3] = 100
  expected = [3] if axis is None else np.argmax(a, axis=1)
  np.testing.assert_array_equal(factory()(a), expected)


def test_tuple_scan(compiler):
  @T.reducer(identity=(0.0, -1))
  def sum_last(a, b):
    return a[0] + b[0], T.max(a[1], b[1])

  @T.template
  def kernel(
    A: T.Tensor((3, 7), dtype="float32"),
    C: T.Tensor((3, 7), dtype="float32", access="output"),
    Index: T.Tensor((3, 7), dtype="int64", access="output"),
  ):
    values = T.load(A), T.broadcast_to(T.arange(7), (3, 7))
    result = T.scan(values, reducer=sum_last, axis=1, exclusive=True)
    T.store(C, result[0])
    T.store(Index, result[1])

  a = np.arange(21, dtype=np.float32).reshape(3, 7)
  c, index = np.empty_like(a), np.empty((3, 7), dtype=np.int64)
  T.compile(kernel, compiler=compiler)(a, c, index)
  expected = np.concatenate((np.zeros((3, 1), dtype=np.float32), np.cumsum(a, axis=1)[:, :-1]), axis=1)
  np.testing.assert_array_equal(c, expected)
  np.testing.assert_array_equal(index, np.broadcast_to(np.arange(-1, 6), (3, 7)))


@pytest.mark.parametrize("extent", [0, 7])
def test_general_fold_empty_identity(compiler, extent):
  @T.reducer(identity=0.0)
  def sum_(a, b):
    return a + b

  @T.template
  def kernel(
    A: T.Tensor((extent,), dtype="float32"),
    C: T.Tensor((1,), dtype="float32", access="output"),
    Prefix: T.Tensor((extent,), dtype="float32", access="output"),
  ):
    C[0] = T.reduce(T.load(A), reducer=sum_)
    T.store(Prefix, T.scan(T.load(A), reducer=sum_, reverse=True))

  a = np.arange(extent, dtype=np.float32)
  c, prefix = np.empty(1, dtype=np.float32), np.empty_like(a)
  T.compile(kernel, compiler=compiler)(a, c, prefix)
  np.testing.assert_array_equal(c, [a.sum()])
  np.testing.assert_array_equal(prefix, np.cumsum(a[::-1])[::-1])


def test_invalid_fold_inputs():
  from vecops.language._compiler.frontend.parser import lower

  @T.template
  def empty(A: T.Tensor((7,), dtype="float32")):
    T.reduce(())

  @T.template
  def predicate(A: T.Tensor((7,), dtype="float32")):
    T.reduce(T.load(A) > 0)

  with pytest.raises(TypeError, match="cannot be empty"):
    lower(empty.ir)
  with pytest.raises(TypeError, match="numeric values"):
    lower(predicate.ir)


def test_invalid_vector_axis():
  from vecops.language._compiler.frontend.parser import lower

  @T.template
  def kernel(A: T.Tensor((7,), dtype="float32")):
    for i in T.parallel(7, layout=T.ExecutionLayout(vector_axis=1)):
      pass

  with pytest.raises(TypeError, match="exceeds the iteration rank"):
    lower(kernel.ir)


@pytest.mark.parametrize("exclusive,reverse", [(False, False), (True, False), (False, True), (True, True)])
def test_scan(compiler, exclusive, reverse):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((3, 7), dtype="float32"), C: T.Tensor((3, 7), dtype="float32")):
      T.store(C, T.scan(T.load(A), axis=1, exclusive=exclusive, reverse=reverse))

    return kernel

  a = np.arange(21, dtype=np.float32).reshape(3, 7)
  ordered = a[:, ::-1] if reverse else a
  expected = np.cumsum(ordered, axis=1)
  if exclusive:
    expected = np.concatenate((np.zeros((3, 1), dtype=np.float32), expected[:, :-1]), axis=1)
  if reverse:
    expected = expected[:, ::-1]
  np.testing.assert_array_equal(factory()(a), expected)
