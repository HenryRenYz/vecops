"""Memory layout and dispatch contracts in generated matrix schedules."""

import re
from dataclasses import replace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import amx_available_for_test, make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.matmul import MatmulEmitter
from vecops.language._compiler.backends.cpu.planning import prepare, snapshot_strides
from vecops.language._compiler.cost.model import CostProfile


def test_amx_b_panel_budget_is_target_policy_and_keeps_explicit_bn():
  @T.template
  def kernel(
    A: T.Tensor((128, 4096), dtype="bfloat16"),
    B: T.Tensor((4096, 4096), (1, 4096), "bfloat16"),
    C: T.Tensor((128, 4096), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C)

  for target, divisor in (("AVX512", 4), ("SVE", 0)):
    artifact = CPUBackend().lower(kernel.ir, (), target=target)
    (plan,) = prepare(artifact.lowered).matmuls.values()
    assert plan.amx_cache_b_panel_divisor == divisor
    assert plan.amx_cache_b_panel_max_m == (256 if target == "AVX512" else 0)
    if target == "AVX512":
      assert "matmul_blocks(32,32,32,2,2,4,std::int64_t{4096},4)" in artifact.source
  disabled = CostProfile("unit-panel-disabled", policies=(("amx_cache_b_panel_divisor", 0),))
  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512", options={"cost_profiles": (disabled,)})
  assert "matmul_blocks(32,32,32,2,2,4,std::int64_t{4096},0)" in artifact.source

  @T.template
  def explicit_kernel(
    A: T.Tensor((128, 4096), dtype="bfloat16"),
    B: T.Tensor((4096, 4096), (1, 4096), "bfloat16"),
    C: T.Tensor((128, 4096), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C, BN=256)

  source = CPUBackend().lower(explicit_kernel.ir, (), target="AVX512").source
  assert re.search(r"block_n_\d+\+=256\)", source)
  assert "matmul_blocks(32,32,32,2,2,4,std::int64_t{4096},0)" in source

  @T.template
  def large_m_kernel(
    A: T.Tensor((512, 4096), dtype="bfloat16"),
    B: T.Tensor((4096, 512), (1, 4096), "bfloat16"),
    C: T.Tensor((512, 512), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C)

  source = CPUBackend().lower(large_m_kernel.ir, (), target="AVX512").source
  assert "matmul_blocks(32,32,32,2,2,4,std::int64_t{4096},0)" in source


def test_amx_b_panel_budget_rejects_invalid_policy():
  import pytest

  @T.template
  def kernel(
    A: T.Tensor((32, 32), dtype="float32"),
    B: T.Tensor((32, 32), dtype="float32"),
    C: T.Tensor((32, 32), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C)

  for setting in ("amx_cache_b_panel_divisor", "amx_cache_b_panel_max_m"):
    for invalid in (-1, True, 2.5):
      with pytest.raises(ValueError, match="policy settings|amx_cache_b_panel_"):
        policy = CostProfile("unit-panel-invalid", policies=((setting, invalid),))
        CPUBackend().lower(kernel.ir, (), target="AVX512", options={"cost_profiles": (policy,)})


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


def test_aligned_cache_panels_keep_a_branch_free_amx_k_body():
  @T.template
  def kernel(
    A: T.Tensor((128, 768), dtype="bfloat16"),
    B: T.Tensor((768, 1024), (1, 768), "bfloat16"),
    C: T.Tensor((128, 1024), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C)

  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  match = re.search(r"/\* full K: unmasked load/convert \*/\s*for \(;[^\n]+\{(.*?)\n\s*\}", source, re.DOTALL)
  assert match is not None
  body = match[1]
  assert body.count("vecops::vec::amx::dot<") == 4
  assert body.count("vecops::vec::amx::load<") == 4
  assert "amx_k_cursor" in body
  assert "pack_offset" not in body
  assert "if (" not in body


def test_tall_traversal_policy_is_target_specific_without_kernel_cloning():
  @T.template
  def kernel(
    A: T.Tensor((3136, 576), dtype="bfloat16"),
    B: T.Tensor((576, 64), (1, 576), "bfloat16"),
    C: T.Tensor((3136, 64), dtype="float32", access="output"),
  ):
    T.dot((A, B), C)

  for target, expected in (("AVX512", True), ("SVE", False)):
    artifact = CPUBackend().lower(kernel.ir, (), target=target)
    (plan,) = prepare(artifact.lowered).matmuls.values()
    assert I.constant(plan.amx_m_major) is expected

  @T.template
  def cached_kernel(
    A: T.Tensor((3136, 576), dtype="bfloat16"),
    B: T.Tensor((576, 64), (1, 576), "bfloat16"),
    C: T.Tensor((3136, 64), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C)

  artifact = CPUBackend().lower(cached_kernel.ir, (), target="AVX512")
  (plan,) = prepare(artifact.lowered).matmuls.values()
  assert I.constant(plan.amx_m_major) is False

  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def dynamic_kernel(
    A: T.Tensor((M, K), (K, 1), "bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    T.dot((A, B), C)

  artifact = CPUBackend().lower(dynamic_kernel.ir, (), target="AVX512")
  assert "tall strip traversal" in artifact.source
  (plan,) = prepare(artifact.lowered).matmuls.values()
  assert I.constant(plan.amx_m_major) is None


def test_native_b_packing_crossover_preserves_transformed_and_dynamic_inputs():
  @T.template
  def kernel(
    A: T.Tensor((64, 768), dtype="bfloat16"),
    B: T.Tensor((768, 1024), (1, 768), "bfloat16"),
    C: T.Tensor((64, 1024), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C)

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  (plan,) = prepare(artifact.lowered).matmuls.values()
  emitter = MatmulEmitter()
  emitter.active_matmul_plan = plan
  b = plan.operands[1].expression
  assert plan.amx_native_b_pack_min_m == 128
  assert not emitter.amx_automatic_b_pack(b, False)
  assert emitter.amx_automatic_b_pack(I.binary("mul", b, I.literal(2)), False)
  assert emitter.amx_automatic_b_pack(replace(b, data=replace(b.data, dtype="float32")), False)
  assert emitter.amx_automatic_b_pack(
    replace(b, data=replace(b.data, strides=(I.literal(2), b.data.strides[1]))), False
  )
  for m in (
    I.literal(33),
    I.literal(35),
    I.literal(127),
    I.literal(128),
    I.Expr(I.ScalarType("int64"), "raw", data="runtime_m"),
  ):
    product = replace(plan.expression, type=replace(plan.expression.type, shape=(m, plan.expression.type.shape[1])))
    emitter.active_matmul_plan = replace(plan, expression=product)
    assert emitter.amx_automatic_b_pack(b, False)
    assert not emitter.amx_automatic_b_pack(b, True)
  product = replace(plan.expression, type=replace(plan.expression.type, shape=(I.literal(64), I.literal(35))))
  emitter.active_matmul_plan = replace(plan, expression=product)
  assert emitter.amx_automatic_b_pack(b, False)

  @T.template
  def small_kernel(
    A: T.Tensor((64, 256), dtype="bfloat16"),
    B: T.Tensor((256, 256), (1, 256), "bfloat16"),
    C: T.Tensor((64, 256), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C, implementation="amx")

  # Small, resident packed buffers avoid repeating the raw transpose network.
  source = CPUBackend().lower(small_kernel.ir, (), target="AVX512").source
  assert re.search(r"matmul_packed_\d+\.resize\(vecops_dsl::pack_size<vecops::bfloat16_t, 1>", source)


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
  assert "one zero-padded K tail" in compiled.generated_source
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
  from dsl_test_utils import parameter_snapshot_name

  base = re.escape(parameter_snapshot_name(compiled.generated_source, "A.data()"))
  assert re.search(rf"amx_k_cursor_\d+=\(\([^\n]+>0 \?[^\n]+: {base}\)", compiled.generated_source)
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
