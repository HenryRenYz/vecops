"""Memory layout and dispatch contracts in generated matrix schedules."""

import re

import numpy as np
import vecops
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.planning import prepare, snapshot_strides


def test_snapshot_preserves_unit_stride_without_padding_or_overlap():
  shape = tuple(I.literal(v) for v in (13, 17))
  for source, expected in (((1, 19), (1, 13)), ((23, 1), (17, 1)), ((-1, 19), (17, 1)), ((0, 1), (17, 1))):
    view = I.View("A", "float32", shape, tuple(I.literal(v) for v in source), I.literal(0))
    assert tuple(I.constant(v) for v in snapshot_strides(view)) == expected
  shape = tuple(I.literal(v) for v in (2, 3, 4))
  view = I.View("A", "float32", shape, tuple(I.literal(v) for v in (27, 1, 5)), I.literal(0))
  assert tuple(I.constant(v) for v in snapshot_strides(view)) == (12, 1, 3)


def test_gemv_selects_snapshot_address_not_complete_load_dags():
  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), "bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C)

  source = CPUBackend().lower(kernel.ir, ()).source
  assert "widening vector dot" in source
  assert not re.search(r"alias_\d+\s*\?\s*vecops::vec::load_convert", source)
  assert "single-column output remainder" in source
  # Cache size/rounding work belongs only to the non-small branch.
  assert source.index("/* bypass cache panels") < source.index("=vecops_dsl::matmul_blocks")


def test_gemv_k_banks_follow_target_policy_and_explicit_override():
  @T.template
  def kernel(
    A: T.Tensor((1, 129), dtype="bfloat16"),
    B: T.Tensor((129, 257), (1, 129), "bfloat16"),
    C: T.Tensor((1, 257), dtype="float32", access="output"),
  ):
    T.dot((A, B), C)

  for target, expected in (("AVX512", 1), ("SVE", 2)):
    artifact = CPUBackend().lower(kernel.ir, (), target=target)
    assert next(iter(artifact.lowered.manager.options["_compiler_environment"].costs.profiles)).name == "default"
    assert next(iter(prepare(artifact.lowered).matmuls.values())).k_unroll == expected
  artifact = CPUBackend().lower(kernel.ir, (), options={"cpu_gemv_k_unroll": 4}, target="AVX512")
  assert next(iter(prepare(artifact.lowered).matmuls.values())).k_unroll == 4


def test_explicit_cache_blocks_do_not_take_runtime_bypass():
  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), "bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C, BM=32, BN=48, BK=64)

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  assert "bypass cache panels" not in artifact.source
  # Other ISA/vector alternatives have their own planners; the AMX runtime
  # family-quantum planner must occur only once, outside family dispatch.
  planners = [line for line in artifact.source.splitlines() if "=vecops_dsl::matmul_blocks" in line]
  assert sum("?" in line for line in planners) == 1
  (plan,) = prepare(artifact.lowered).matmuls.values()
  assert plan.unblocked_bytes == 512 * 1024
  assert plan.runtime_unblocked_bytes == 128 * 1024


def test_known_long_gemv_retains_two_banks_unless_overridden():
  @T.template
  def kernel(
    A: T.Tensor((1, 1025), dtype="bfloat16"),
    B: T.Tensor((1025, 128), (1, 1025), "bfloat16"),
    C: T.Tensor((1, 128), dtype="float32", access="output"),
  ):
    T.dot((A, B), C)

  for options, expected in (({}, 2), ({"cpu_gemv_k_unroll": 1}, 1)):
    artifact = CPUBackend().lower(kernel.ir, (), options=options, target="AVX512")
    (plan,) = prepare(artifact.lowered).matmuls.values()
    assert plan.k_unroll == expected


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

  compiler = vecops.Compiler(
    target="Native", cc="gcc", cxx="g++", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"
  )
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

  compiler = vecops.Compiler(
    target="Native", cc="gcc", cxx="g++", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"
  )
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
