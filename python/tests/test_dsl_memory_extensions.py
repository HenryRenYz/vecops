"""Multidimensional coordinates, singleton transforms and planned workspace."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from vecops.language._compiler.frontend.parser import lower


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-memory-extensions")
  return vecops.Compiler(
    target="Native", cc="gcc", cxx="g++", jobs=4, cache_dir=root / "cache", build_dir=root / "build"
  )


@pytest.mark.parametrize("negative", [False, True])
def test_gather_coordinates_conversion(compiler, negative):
  row_stride = -10 if negative else 10

  @T.template
  def kernel(
    A: T.Tensor((6, 5), (row_stride, 2), "float16"),
    Idx: T.Tensor((7,), dtype="int64"),
    C: T.Tensor((7, 5), dtype="float32", access="output"),
  ):
    rows = T.expand_dims(T.load(Idx), 1)
    columns = T.indices((7, 5), axis=1)
    T.store(C, T.gather(A, (rows, columns), dtype="float32", mask=rows >= 0, other=-7))

  a = np.arange(60, dtype=np.float16).reshape(6, 10)[:, ::2]
  if negative:
    a = a[::-1]
  indices = np.array([3, 1, -1, 4, 1, 0, 5], dtype=np.int64)
  c = np.empty((7, 5), dtype=np.float32)
  T.compile(kernel, compiler=compiler)(a, indices, c)
  expected = a[np.maximum(indices, 0)].astype(np.float32)
  expected[indices < 0] = -7
  np.testing.assert_array_equal(c, expected)


def test_gather_multidimensional_alias(compiler):
  @T.template
  def kernel(A: T.Tensor((6, 5), dtype="float32", access="inout")):
    row = 5 - T.indices((6, 5), axis=0)
    col = T.indices((6, 5), axis=1)
    T.store(A, T.gather(A, (row, col)))

  a = np.arange(30, dtype=np.float32).reshape(6, 5)
  expected = a[::-1].copy()
  T.compile(kernel, compiler=compiler)(a)
  np.testing.assert_array_equal(a, expected)


def test_axis_statistics_and_singletons(compiler):
  @T.template
  def kernel(A: T.Tensor((3, 7), dtype="float32"), C: T.Tensor((3, 7), dtype="float32", access="output")):
    x = T.load(T.squeeze(T.expand_dims(A, 1), 1))
    row = T.expand_dims(T.reduce_add(x, axis=1), 1)
    T.store(C, x + row)

  a = np.arange(21, dtype=np.float32).reshape(3, 7)
  c = np.empty_like(a)
  T.compile(kernel, compiler=compiler)(a, c)
  np.testing.assert_array_equal(c, a + a.sum(axis=1, keepdims=True))


def test_workspace_replay_grid_and_budgets(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  rows = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((rows, 17), dtype="float32"), C: T.Tensor((rows, 17), dtype="float32", access="output")):
    scratch = T.workspace((rows, 17), dtype="float32")
    T.store(scratch, T.load(A) + 2)
    for i in T.grid(rows, parallel="require"):
      T.store(C[i, :], T.load(scratch[i, :]) * 3)

  compiled = T.compile(kernel, compiler=compiler)
  for limit in (4, 2, 4):
    for n in (0, 1, 11, 11):
      a = np.arange(n * 17, dtype=np.float32).reshape(n, 17)
      c = np.empty_like(a)
      with vecops.cpu_thread_limit(limit):
        compiled(a, c)
      np.testing.assert_array_equal(c, (a + 2) * 3)
  assert "WorkspaceContext&" in compiled.generated_source
  assert "dsl_workspace.tensor<" in compiled.generated_source


def test_transform_broadcast_into_higher_rank(compiler):
  @T.template
  def kernel(A: T.Tensor((3,), dtype="float32"), C: T.Tensor((2, 3, 7), dtype="float32", access="output")):
    row = T.expand_dims(T.load(A) + 1, 1)
    T.store(C, row + T.indices((2, 3, 7), axis=2))

  a = np.arange(3, dtype=np.float32)
  c = np.empty((2, 3, 7), dtype=np.float32)
  T.compile(kernel, compiler=compiler)(a, c)
  expected = np.broadcast_to(a[None, :, None] + 1 + np.arange(7), c.shape)
  np.testing.assert_array_equal(c, expected)


def test_worker_workspace_replicas(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")

  @T.template
  def kernel(C: T.Tensor((4,), dtype="int64", access="output")):
    scratch = T.workspace((T.worker_capacity(), 19), dtype="int64")
    with T.team(workers=4) as worker:
      T.store(scratch[worker, :], T.indices((19,)) + worker)
      C[worker] = T.reduce_add(T.load(scratch[worker, :]))

  compiled = T.compile(kernel, compiler=compiler)
  for limit in (4, 2, 4):
    c = np.full(4, -1, dtype=np.int64)
    with vecops.cpu_thread_limit(limit):
      compiled(c)
      compiled(c)
    expected = np.full(4, -1, dtype=np.int64)
    expected[:limit] = 171 + np.arange(limit) * 19
    np.testing.assert_array_equal(c, expected)


def test_workspace_serial_site_rules():
  @T.template
  def inside(A: T.Tensor((7,), dtype="float32")):
    for i in T.serial(3):
      T.workspace((7,))

  @T.template
  def runtime_extent(A: T.Tensor((7,), dtype="float32"), count: int):
    T.workspace((count,))

  with pytest.raises(TypeError, match="must be declared before"):
    lower(inside.ir)
  with pytest.raises(TypeError, match="tensor metadata"):
    lower(runtime_extent.ir)
