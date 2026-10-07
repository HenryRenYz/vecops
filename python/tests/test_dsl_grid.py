"""Grid ownership, scheduling and native multi-stage execution regressions."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  directory = tmp_path_factory.mktemp("dsl-grid")
  return make_compiler(target="Native", jobs=4, cache_dir=directory / "cache", build_dir=directory / "build")


@pytest.mark.parametrize("schedule", ["static", "dynamic"])
def test_grid_rows_dynamic_sizes_layouts_and_private_scratch(compiler, monkeypatch, schedule):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  M, N, S0, S1 = (T.dynamic(lower=0), T.dynamic(lower=0), T.dynamic(), T.dynamic())

  @T.template
  def kernel(A: T.Tensor((M, N), (S0, S1), "float32"), C: T.Tensor((M, N), dtype="float32", access="output")):
    for row in T.grid(M, schedule=schedule, chunk=3, parallel="require"):
      scratch = T.alloc((N,), dtype="float32")
      T.store(scratch, T.load(A[row, :]) + 1)
      T.store(C[row, :], T.softmax(T.load(scratch)))

  compiled = T.compile(kernel, compiler=compiler)
  assert "grid_lanes" in compiled.generated_source
  rng = np.random.default_rng(1)
  for rows in (0, 1, 3, 4, 5, 9, 41):
    for width in (0, 17, 257):
      x = rng.normal(size=(width, rows)).astype(np.float32).T
      out = np.empty((rows, width), dtype=np.float32)
      compiled(x, out)
      if width and rows:
        exp = np.exp(x - x.max(-1, keepdims=True))
        np.testing.assert_allclose(out, exp / exp.sum(-1, keepdims=True), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("mode", ["individual", "adjacent", "template"])
def test_multigrid_dependencies_and_single_worker_scalar(compiler, monkeypatch, mode):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  M, N = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((M, N), dtype="float32"),
    B: T.Tensor((M, N), dtype="float32", access="inout"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
    *,
    scale_input: float = 3.0,
  ):
    for row in T.grid(M):
      T.store(B[row, :], T.load(A[row, :]) + 1)
    # A compile-time literal needs no worker publication. Keep this test
    # about actual runtime serial content between the parallel phases.
    scale = scale_input
    for row in T.grid(M, schedule="dynamic", chunk=2):
      T.store(C[row, :], T.load(B[row, :]) * scale)
    for row in T.grid(M):
      T.store(B[row, :], T.load(C[row, :]) - 2)

  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_grid_regions": mode})
  if mode == "template":
    assert "publish single-worker scalar" in compiled.generated_source
  rng = np.random.default_rng(4)
  for rows in (0, 1, 5, 13, 37):
    a = rng.normal(size=(rows, 129)).astype(np.float32)
    b, c = np.empty_like(a), np.empty_like(a)
    for _ in range(5):
      compiled(a, b, c)
      np.testing.assert_allclose(c, (a + 1) * 3, atol=1e-6)
      np.testing.assert_allclose(b, (a + 1) * 3 - 2, atol=1e-6)


def test_grid_alias_guard_preserves_order_and_strict_rejects(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  M = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((M,), dtype="float32"), C: T.Tensor((M,), dtype="float32", access="output")):
    for i in T.grid(M):
      C[i] = T.load(A[i]) + 1

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(65, dtype=np.float32) * 2
  compiled(a[:-1], a[1:])
  # A shifted write feeds the next program's read: copying all input first
  # would produce different values and violate the ordered fallback contract.
  np.testing.assert_array_equal(a, np.arange(65, dtype=np.float32))

  @T.template
  def strict(A: T.Tensor((M,), dtype="float32"), C: T.Tensor((M,), dtype="float32", access="output")):
    for i in T.grid(M, parallel="require"):
      C[i] = T.load(A[i]) + 2

  with pytest.raises(Exception, match="alias/layout guard"):
    T.compile(strict, compiler=compiler)(a[:-1], a[1:])


def test_grid_provenance_for_clipped_rectangular_tiles(compiler):
  @T.template
  def kernel(A: T.Tensor((33, 35), dtype="float32"), C: T.Tensor((33, 35), dtype="float32", access="output")):
    for i, j in T.grid(5, 5, parallel="require"):
      T.store(C[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8], T.load(A[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]) + 1)

  artifact = CPUBackend().lower(kernel.ir, ())
  assert "verified grid ownership" in artifact.source
  assert not any("retained serial" in message for message in artifact.kernel.diagnostics)
  a = np.arange(33 * 35, dtype=np.float32).reshape(33, 35)
  c = np.empty_like(a)
  T.compile(kernel, compiler=compiler)(a, c)
  np.testing.assert_array_equal(c, a + 1)


def test_unsafe_grid_does_not_infer_independence():
  @T.template
  def kernel(C: T.Tensor((1,), dtype="float32", access="output")):
    for i in T.grid(12, parallel="require"):
      C[0] = i

  with pytest.raises(TypeError, match="same region"):
    CPUBackend().lower(kernel.ir, ())


def test_grid_region_omits_barrier_only_for_independent_effects():
  @T.template
  def kernel(
    A: T.Tensor((9, 1024), dtype="float32"),
    B: T.Tensor((9, 1024), dtype="float32", access="output"),
    C: T.Tensor((9, 1024), dtype="float32", access="output"),
  ):
    for i in T.grid(9):
      T.store(B[i, :], T.load(A[i, :]) + 1)
    for i in T.grid(9):
      T.store(C[i, :], T.load(A[i, :]) + 2)

  source = CPUBackend().lower(kernel.ir, ()).source
  assert "shared grid team" in source
  assert "RAW/WAR/WAW phase edge" in source
  assert "if (true)" not in source.split("shared grid team", 1)[1].split("logical-task-only executor", 1)[0]


def test_thread_budget_nesting_affinity_and_environment(monkeypatch):
  import os

  monkeypatch.setenv("VECOPS_NUM_THREADS", "3")
  assert 1 <= vecops.cpu_thread_pool_capacity() <= 3
  with vecops.cpu_thread_limit(1):
    assert vecops.cpu_thread_pool_capacity() == 1
    with vecops.cpu_thread_limit(2):
      assert 1 <= vecops.cpu_thread_pool_capacity() <= 2
    assert vecops.cpu_thread_pool_capacity() == 1
  assert 1 <= vecops.cpu_thread_pool_capacity() <= 3
  if hasattr(os, "sched_getaffinity"):
    previous = os.sched_getaffinity(0)
    try:
      os.sched_setaffinity(0, {min(previous)})
      assert vecops.cpu_thread_pool_capacity() == 1
    finally:
      os.sched_setaffinity(0, previous)
  for value in (-1, 2**32):
    with pytest.raises(ValueError):
      with vecops.cpu_thread_limit(value):
        pass
  with pytest.raises(TypeError):
    with vecops.cpu_thread_limit(True):
      pass


def test_grid_war_waw_and_runtime_alias_phase_dependencies(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  M, N = 37, 2048

  @T.template
  def kernel(
    A: T.Tensor((M, N), dtype="float32", access="inout"),
    B: T.Tensor((M, N), dtype="float32", access="inout"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    for i in T.grid(M, schedule="dynamic", chunk=3):
      T.store(C[i, :], T.load(A[i, :]) + 1)
    for i in T.grid(M):
      T.store(B[i, :], 2)
    for i in T.grid(M):
      T.store(C[i, :], T.load(C[i, :]) + 3)

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(M * N, dtype=np.float32).reshape(M, N)
  original = a.copy()
  c = np.empty_like(a)
  for _ in range(8):
    a[:] = original
    compiled(a, a, c)  # A/B alias only across grids: runtime WAR barrier
    np.testing.assert_array_equal(c, original + 4)
    np.testing.assert_array_equal(a, np.full_like(a, 2))


def test_grid_exception_cancels_resident_team_and_recovers(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  M, N = 37, 4096

  @T.template
  def kernel(
    A: T.Tensor((M, N), dtype="float32"),
    B: T.Tensor((M, N), dtype="float32", access="inout"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
    fail: bool,
  ):
    for i in T.grid(M):
      assert not fail or i != 0, "grid failure probe"
      T.store(B[i, :], T.load(A[i, :]) + 1)
    for i in T.grid(M):
      T.store(C[i, :], T.load(B[i, :]) + 1)

  compiled = T.compile(kernel, compiler=compiler)
  a = np.ones((M, N), dtype=np.float32)
  b, c = np.empty_like(a), np.empty_like(a)
  with pytest.raises(Exception, match="grid failure probe|cancelled"):
    compiled(a, b, c, True)
  compiled(a, b, c, False)
  np.testing.assert_array_equal(c, np.full_like(c, 3))


def test_grid_metadata_not_hoisted_across_new_scalar():
  @T.template
  def kernel(
    A: T.Tensor((9, 1024), dtype="float32"),
    C: T.Tensor((9, 1024), dtype="float32", access="output"),
    count_input: int,
  ):
    for i in T.grid(9):
      T.store(C[i, :], T.load(A[i, :]))
    count = count_input
    for i in T.grid(count):
      T.store(C[i, :], T.load(C[i, :]) + 1)

  source = CPUBackend().lower(kernel.ir, ()).source
  assert "shared grid team" not in source


@pytest.mark.parametrize("reduction", [False, True])
def test_grid_guard_cannot_hoist_memory_dependent_coordinates(reduction):
  @T.template
  def kernel(A: T.Tensor((9,), dtype="float32"), C: T.Tensor((9,), dtype="float32", access="output")):
    for i in T.grid(9, parallel="require"):
      shift = T.cast(T.reduce_add(T.load(A)) if reduction else T.load(A[0]), T.int64)
      C[i + shift] = i

  with pytest.raises(TypeError, match="program-local state"):
    CPUBackend().lower(kernel.ir, ())


def test_third_grid_metadata_cannot_hoist_an_earlier_single_worker_scalar(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")

  @T.template
  def kernel(
    A: T.Tensor((9, 2048), dtype="float32"),
    C: T.Tensor((9, 2048), dtype="float32", access="output"),
    *,
    count_input: int = 9,
  ):
    for i in T.grid(9):
      T.store(C[i, :], T.load(A[i, :]) + 1)
    count = count_input
    for i in T.grid(9):
      T.store(C[i, :], T.load(C[i, :]) + 2)
    for i in T.grid(count):
      T.store(C[i, :], T.load(C[i, :]) + 3)

  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_grid_regions": "template"})
  assert compiled.generated_source.count("shared grid team") == 1
  a = np.arange(9 * 2048, dtype=np.float32).reshape(9, 2048)
  out = np.empty_like(a)
  compiled(a, out)
  np.testing.assert_array_equal(out, a + 6)


def test_grid_inner_loop_coordinates_retain_serial_order():
  @T.template
  def kernel(C: T.Tensor((9, 17), dtype="float32", access="output")):
    for i in T.grid(9):
      for j in T.serial(17):
        C[i, j] = i + j

  # The inner serial axis is not a partition axis, so this remains parallel.
  source = CPUBackend().lower(kernel.ir, ()).source
  assert "verified grid ownership" in source


def test_grid_budget_changes_do_not_create_jit_artifacts(compiler):
  @T.template
  def kernel(A: T.Tensor((37, 2048), dtype="float32"), C: T.Tensor((37, 2048), dtype="float32", access="output")):
    for i in T.grid(37):
      T.store(C[i, :], T.softmax(T.load(A[i, :])))

  compiled = T.compile(kernel, compiler=compiler)
  assert compiled.kernel_def.runtime_parallelism
  a = np.ones((37, 2048), dtype=np.float32)
  c = np.empty_like(a)
  # The metadata-only path has no execution context and prepares P=1.
  compiled.operator.compile_for(
    vecops.TensorMeta(a.shape, dtype="float32"),
    vecops.TensorMeta(a.shape, dtype="float32", access=vecops.TensorAccess.output),
  )
  before = set(compiler.cache_dir.rglob("*.so"))
  assert before
  for budget in (1, 2, 4, 8, 1):
    with vecops.cpu_thread_limit(budget):
      compiled(a, c)
      np.testing.assert_allclose(c, np.full_like(c, 1 / 2048), rtol=3e-6)
    assert set(compiler.cache_dir.rglob("*.so")) == before


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_batched_dot_keeps_packing_and_hardware_state_private(compiler, monkeypatch, dtype):
  import ml_dtypes

  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  Batch, M, N, K = 17, 32, 128, 128

  @T.template
  def kernel(
    A: T.Tensor((Batch, M, K), dtype=dtype),
    B: T.Tensor((K, N), (1, K), dtype),
    C: T.Tensor((Batch, M, N), dtype="float32", access="inout"),
  ):
    for i in T.grid(Batch, parallel="require"):
      x = T.dot((T.load(A[i, :, :]), T.load(B)), C[i, :, :], acc=T.load(C[i, :, :]))
      T.store(C[i, :, :], T.max(x, 0))

  storage_dtype = np.float32 if dtype == "float32" else ml_dtypes.bfloat16
  rng = np.random.default_rng(3)
  a = rng.integers(-3, 4, size=(Batch, M, K)).astype(storage_dtype)
  b = rng.integers(-3, 4, size=(N, K)).astype(storage_dtype).T
  c = rng.integers(-3, 4, size=(Batch, M, N)).astype(np.float32)
  expected = np.maximum(a.astype(np.float32) @ b.astype(np.float32) + c, 0)
  compiled = T.compile(kernel, compiler=compiler)
  assert_no_vec_mask_arrays(compiled.generated_source)
  compiled(a, b, c)
  # Grid-local matmul packing must not place target register vectors in arrays.
  # This kernel is executed with multiple workers on both x86 and ARM targets.
  np.testing.assert_allclose(c, expected, rtol=2e-5, atol=2e-5)


def test_grid_transposed_and_negative_stride_outputs(compiler, monkeypatch):
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  M, N = 37, 257
  S0, S1 = T.dynamic(), T.dynamic()

  @T.template
  def kernel(A: T.Tensor((M, N), dtype="float32"), C: T.Tensor((M, N), (S0, S1), "float32", access="output")):
    for i in T.grid(M, parallel="require"):
      T.store(C[i, :], T.load(A[i, :]) + 1)

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(M * N, dtype=np.float32).reshape(M, N)
  for out in (np.empty((N, M), dtype=np.float32).T, np.empty_like(a)[::-1]):
    compiled(a, out)
    np.testing.assert_array_equal(out, a + 1)


@pytest.mark.parametrize("provider", ["torch", "native"])
def test_torch_logical_tasks_and_native_team_fallbacks(compiler, monkeypatch, provider):
  import torch

  monkeypatch.setenv("VECOPS_THREAD_POOL", provider)
  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  monkeypatch.delenv("VECOPS_NATIVE_THREAD_POOL_OPS", raising=False)
  M, N = 37, 2048
  previous = torch.get_num_threads()
  schedule = "dynamic" if provider == "native" else "static"
  torch.set_num_threads(4)
  try:

    @T.template
    def kernel(
      A: T.Tensor((M, N), dtype="float32"),
      B: T.Tensor((M, N), dtype="float32", access="inout"),
      C: T.Tensor((M, N), dtype="float32", access="output"),
    ):
      for i in T.grid(M, schedule=schedule):
        T.store(B[i, :], T.softmax(T.load(A[i, :])))
      for i in T.grid(M, schedule=schedule):
        T.store(C[i, :], T.load(B[i, :]) * 2)

    compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_grid_regions": "template"})
    a = torch.randn(M, N)
    b, c = torch.empty_like(a), torch.empty_like(a)
    compiled(a, b, c)
    torch.testing.assert_close(c, a.softmax(-1) * 2, rtol=4e-5, atol=3e-6)
  finally:
    torch.set_num_threads(previous)


def test_grid_unrolled_inner_coordinate_is_not_a_parallel_partition():
  @T.template
  def kernel(C: T.Tensor((12,), dtype="int64", access="output"), limit: int):
    for i in T.grid(9, parallel="require"):
      for j in T.serial(limit, unroll=2):
        C[i + j] = i

  with pytest.raises(TypeError, match="program-local state"):
    CPUBackend().lower(kernel.ir, ())


def test_grid_extent_read_cannot_move_before_an_earlier_write(compiler):
  @T.template
  def kernel(B: T.Tensor((9,), dtype="int64", access="inout"), C: T.Tensor((5,), dtype="int64", access="output")):
    for i in T.grid(9):
      B[i] = 5
    for i in T.grid(T.load(B[0])):
      C[i] = i + 1

  compiled = T.compile(kernel, compiler=compiler)
  assert "shared grid team" not in compiled.generated_source
  b = np.zeros(9, dtype=np.int64)
  c = np.zeros(5, dtype=np.int64)
  compiled(b, c)
  np.testing.assert_array_equal(b, np.full_like(b, 5))
  np.testing.assert_array_equal(c, np.arange(1, 6, dtype=np.int64))


def test_completed_phase_dependencies_do_not_insert_redundant_barriers():
  @T.template
  def kernel(
    A: T.Tensor((37, 2048), dtype="float32"),
    B: T.Tensor((37, 2048), dtype="float32", access="inout"),
    C: T.Tensor((37, 2048), dtype="float32", access="output"),
    D: T.Tensor((37, 2048), dtype="float32", access="output"),
  ):
    for i in T.grid(37):
      T.store(B[i, :], T.load(A[i, :]) + 1)
    for i in T.grid(37):
      T.store(C[i, :], T.load(B[i, :]) + 2)
    for i in T.grid(37):
      T.store(D[i, :], T.load(B[i, :]) + 3)

  source = CPUBackend().lower(kernel.ir, ()).source
  assert "shared grid team" in source
  assert "if (true" not in source
  edges = [line for line in source.splitlines() if "RAW/WAR/WAW phase edge" in line]
  assert len(edges) == 2  # One definite edge; C/D runtime alias remains guarded.
  assert sum("if (" not in line for line in edges) == 1
