"""Resident synchronization, atomics and synchronous software pipelines."""

import numpy as np
import pytest
import vecops
import vecops.language as T


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-collectives-pipeline")
  return vecops.Compiler(
    target="Native", cc="gcc", cxx="g++", jobs=4, cache_dir=root / "cache", build_dir=root / "build"
  )


def test_team_barrier_reduce_and_atomics(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")

  @T.template
  def kernel(C: T.Tensor((4,), dtype="int64", access="output")):
    shared = T.shared((4,), dtype="int64")
    T.store(shared, T.full((4,), 0, dtype="int64"))
    with T.team(workers=4) as worker:
      shared[worker] = worker + 1
      T.barrier()
      total = T.team_reduce(T.load(shared[worker]), op="add")
      C[worker] = total

  compiled = T.compile(kernel, compiler=compiler)
  for _ in range(10):
    c = np.zeros(4, dtype=np.int64)
    compiled(c)
    np.testing.assert_array_equal(c, [10] * 4)
  assert "grid_team" in compiled.generated_source


def test_parallel_atomics(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")

  @T.template
  def kernel(C: T.Tensor((1,), dtype="int64", access="inout")):
    with T.team(workers=4):
      for i in T.serial(31):
        T.atomic_add(C[0], 1)

  compiled = T.compile(kernel, compiler=compiler)
  c = np.zeros(1, dtype=np.int64)
  compiled(c)
  np.testing.assert_array_equal(c, [124])


def test_team_honors_call_budget(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")

  @T.template
  def kernel(C: T.Tensor((4,), dtype="int64", access="output")):
    with T.team(workers=4) as worker:
      C[worker] = T.worker_count()

  compiled = T.compile(kernel, compiler=compiler)
  c = np.zeros(4, dtype=np.int64)
  with vecops.cpu_thread_limit(2):
    compiled(c)
  np.testing.assert_array_equal(c, [2, 2, 0, 0])


@pytest.mark.parametrize("dtype", ["float32", "int64"])
def test_atomic_returns(compiler, dtype):
  @T.template
  def kernel(C: T.Tensor((1,), dtype=dtype, access="inout"), Out: T.Tensor((5,), dtype=dtype, access="output")):
    Out[0] = T.atomic_exchange(C[0], 5, order="seq_cst")
    Out[1] = T.atomic_add(C[0], 2)
    Out[2] = T.atomic_min(C[0], 3, order="acq_rel")
    Out[3] = T.atomic_max(C[0], 9)
    Out[4] = T.atomic_compare_exchange(C[0], 9, 11)

  c, out = np.zeros(1, dtype=dtype), np.empty(5, dtype=dtype)
  T.compile(kernel, compiler=compiler)(c, out)
  np.testing.assert_array_equal(out, [0, 5, 7, 3, 9])
  np.testing.assert_array_equal(c, [11])


def test_divergent_barrier_rejected(compiler):
  @T.template
  def kernel(C: T.Tensor((1,), dtype="int64", access="output")):
    with T.team(workers=4):
      if T.worker_id() == 0:
        T.barrier()

  with pytest.raises(TypeError, match="top level"):
    T.compile(kernel, compiler=compiler)


def test_team_failure_cancels_barrier(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")

  @T.template
  def kernel(C: T.Tensor((4,), dtype="int64", access="output")):
    with T.team(workers=4):
      assert T.worker_id() != 0, "worker failure"
      T.barrier()
      C[T.worker_id()] = 1

  with pytest.raises(Exception, match="worker failure|cancel"):
    T.compile(kernel, compiler=compiler)(np.zeros(4, dtype=np.int64))


def test_raw_amx_tdp(compiler):
  @T.template
  def kernel(
    A: T.Tensor((2, 8), dtype="int8"),
    B: T.Tensor((2, 8), dtype="int8"),
    C: T.Tensor((2, 2), dtype="int32", access="output"),
  ):
    with T.amx.region(((2, 8), (2, 8), (2, 8))):
      acc = T.amx.zero(0, dtype="int32")
      a = T.amx.load(1, A)
      b = T.amx.load(2, B)
      T.amx.tdp(acc, a, b)
      T.amx.store(acc, C)

  a = np.arange(16, dtype=np.int8).reshape(2, 8)
  original = np.arange(16, dtype=np.int8).reshape(8, 2)
  packed = original.reshape(2, 4, 2).transpose(0, 2, 1).reshape(2, 8).copy()
  c = np.empty((2, 2), dtype=np.int32)
  compiled = T.compile(kernel, compiler=compiler)
  compiled(a, packed, c)
  np.testing.assert_array_equal(c, a.astype(np.int32) @ original.astype(np.int32))


def test_lookahead_prefetch_and_double_buffer(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((5, 19), dtype="float32"), C: T.Tensor((5, 19), dtype="float32")):
      banks = T.double_buffer((19,), dtype="float32")
      T.copy(A[0, :], banks[0, :])
      for i in T.pipelined(5, distance=1):
        if i + 1 < 5:
          T.copy(A[i + 1, :], banks[(i + 1) % 2, :])
        T.store(C[i, :], T.load(banks[i % 2, :]) * 2 + 1)

    return kernel

  a = np.arange(95, dtype=np.float32).reshape(5, 19)
  np.testing.assert_array_equal(factory()(a), a * 2 + 1)
  assert "plan-software-pipeline" in factory().kernel_ir.pass_trace


def test_pipeline_prefetch_tail(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((5, 19), dtype="float32"), C: T.Tensor((5, 19), dtype="float32")):
      for i in T.pipelined(5):
        T.store(C[i, :], T.load(A[i, :]) + 1)

    return kernel

  a = np.arange(95, dtype=np.float32).reshape(5, 19)
  compiled = factory()
  np.testing.assert_array_equal(compiled(a), a + 1)
  assert "prefetch_l1" in compiled.generated_source


@pytest.mark.parametrize("rows", [0, 1, 5])
def test_automatic_double_buffer_alias_and_tail(compiler, rows):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((rows, 19), dtype="float32"), C: T.Tensor((rows, 19), dtype="float32")):
      for i in T.pipelined(rows, buffering=True):
        T.store(C[i, :], T.load(A[i, :]) * 2 + 1)

    return kernel

  a = np.arange(rows * 19, dtype=np.float32).reshape(rows, 19)
  compiled = factory()
  np.testing.assert_array_equal(compiled(a), a * 2 + 1)
  assert "two-bank loads guarded" in " ".join(compiled.kernel_ir.diagnostics)
  compiled(a, a)
  np.testing.assert_array_equal(a, np.arange(rows * 19, dtype=np.float32).reshape(rows, 19) * 2 + 1)


def test_automatic_double_buffer_shifted_alias(compiler):
  @T.template
  def kernel(A: T.Tensor((4, 19), dtype="float32"), C: T.Tensor((4, 19), dtype="float32", access="output")):
    for i in T.pipelined(4, buffering=True):
      T.store(C[i, :], T.load(A[i, :]) + 1)

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(95, dtype=np.float32).reshape(5, 19)
  expected = a.copy()
  for i in range(4):
    expected[i + 1] = expected[i] + 1
  compiled(a[:-1], a[1:])
  np.testing.assert_array_equal(a, expected)
