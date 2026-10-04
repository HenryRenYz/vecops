"""Execution and codegen tests for source-less vector kernels."""
import numpy as np
import pytest
import vecops
import vecops.language as T


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  path = tmp_path_factory.mktemp("dsl")
  return vecops.Compiler(target="Native", cc="gcc", cxx="g++", jobs=4,
                         cache_dir=path / "cache", build_dir=path / "build")


def normalization(compiler, which, dtype="float32", output_dtype="float32"):
  M = T.dynamic(lower=0)
  N = T.dynamic(lower=0)
  AS0 = T.dynamic()
  AS1 = T.dynamic()
  CS0 = T.dynamic()
  CS1 = T.dynamic()
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((M, N), (AS0, AS1), dtype),
               C: T.Tensor((M, N), (CS0, CS1), output_dtype)):
      for row in T.serial(M):
        x = T.load(A[row, :], dtype=T.float32)
        if which == "layernorm":
          y = T.layernorm(x)
        else:
          y = T.softmax(x)
        T.store(C[row, :], y)
    return kernel
  return factory()


@pytest.mark.parametrize("which", ["layernorm", "softmax"])
def test_normalization_dynamic_layouts(compiler, which):
  kernel = normalization(compiler, which)
  rng = np.random.default_rng(15)
  for width in (0, 1, 7, 17, 65, 257):
    for layout in ("dense", "strided", "transposed", "negative"):
      if layout == "dense":
        x = rng.normal(size=(3, width)).astype(np.float32)
      elif layout == "strided":
        x = rng.normal(size=(3, width * 2)).astype(np.float32)[:, ::2]
      elif layout == "transposed":
        x = rng.normal(size=(width, 3)).astype(np.float32).T
      else:
        x = rng.normal(size=(3, width)).astype(np.float32)[:, ::-1]
      backing = np.full((width * 2, 3), -123, dtype=np.float32)
      y = backing[::2, :].T
      kernel(x, y)
      if width:
        if which == "layernorm":
          mean = x.mean(-1, keepdims=True)
          expected = (x - mean) / np.sqrt(((x - mean) ** 2).mean(-1, keepdims=True) + 1e-5)
        else:
          ex = np.exp(x - x.max(-1, keepdims=True))
          expected = ex / ex.sum(-1, keepdims=True)
        np.testing.assert_allclose(y, expected, atol=3e-6, rtol=3e-5)
      assert np.all(backing[1::2] == -123)
  assert "load_convert" in kernel.generated_source
  assert "store_convert" in kernel.generated_source
  assert "opt::strided" in kernel.generated_source
  assert "vecops::ops::" not in kernel.generated_source
  assert "dsl_f_scratch_" not in kernel.generated_source


def test_control_flow(compiler):
  N = T.dynamic(lower=0)
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((N,), dtype=T.float32),
               C: T.Tensor((N,), dtype=T.float32), *, limit: int = 3):
      factor = 1
      i = 0
      while i < limit:
        i += 1
        if i == 2:
          continue
        factor += i
        if factor > 10:
          break
      for j in T.unroll(2):
        factor += j
      for j in T.serial(N):
        if j % 2 == 0:
          C[j] = T.load(A[j]) * factor
        else:
          C[j] = -T.load(A[j])
    return kernel
  kernel = factory()
  x = np.arange(11, dtype=np.float32)
  y = np.empty_like(x)
  kernel(x, y)
  expected = -x
  expected[::2] = x[::2] * 6
  np.testing.assert_array_equal(y, expected)


def test_transpose_convert_and_scratch_forwarding(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((5, 13), dtype=T.float16),
               C: T.Tensor((13, 5), dtype=T.float32)):
      region = T.transpose(T.tile(A))
      x = T.load(region, dtype=T.float32)
      tmp = T.alloc((13, 5), dtype=T.float32)
      T.store(tmp, x * 2 + 1)
      T.store(C, T.load(tmp))
    return kernel
  kernel = factory()
  x = np.arange(65, dtype=np.float16).reshape(5, 13)
  np.testing.assert_array_equal(kernel(x), x.astype(np.float32).T * 2 + 1)
  assert "dsl_f_scratch_" not in kernel.generated_source
  assert any("transpose" in v for v in kernel.kernel_ir.diagnostics)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("which", ["layernorm", "softmax"])
def test_torch_storage_dtypes(compiler, dtype, which):
  import torch
  kernel = normalization(compiler, which, dtype, dtype)
  x = torch.randn((2, 37), dtype=getattr(torch, dtype))
  y = torch.empty_like(x)
  kernel(x, y)
  expected = torch.softmax(x.float(), -1) if which == "softmax" else torch.nn.functional.layer_norm(x.float(), (37,))
  torch.testing.assert_close(y, expected.to(x.dtype), rtol=3e-3, atol=3e-3)


def test_virtual_vector_fixed_extent(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype=T.float32), C: T.Tensor((37,), dtype=T.float32)):
      x = T.load(A, vector=T.Vector(T.float32, N=37))
      T.store(C, T.where(x > 0, x, -x))
    return kernel
  kernel = factory()
  x = np.linspace(-3, 3, 37, dtype=np.float32)
  np.testing.assert_array_equal(kernel(x), np.abs(x))


def test_located_errors_and_host_execution(compiler):
  with pytest.raises(TypeError, match="only available inside"):
    T.load(None)
  @T.template
  def invalid(A: T.Tensor((3,), dtype=T.float32)):
    try:
      pass
    except Exception:
      pass
  with pytest.raises(TypeError, match=r"test_dsl.py:\d+: unsupported statement Try"):
    T.compile(invalid, compiler=compiler)
  @T.template
  def invalid_grid(A: T.Tensor((3,), dtype=T.float32)):
    for i in T.grid(1.5):
      pass
  with pytest.raises(TypeError, match=r"test_dsl.py:\d+: grid bounds must be scalar integers"):
    T.compile(invalid_grid, compiler=compiler)
  @T.template
  def invalid_assert(A: T.Tensor((3,), dtype=T.float32)):
    assert T.load(A) > 0
  with pytest.raises(TypeError, match=r"test_dsl.py:\d+: assert requires a scalar condition"):
    T.compile(invalid_assert, compiler=compiler)


def test_aliasing_snapshots_and_scalar_capture(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype=T.float32),
               B: T.Tensor((37,), dtype=T.float32, access="inout"),
               C: T.Tensor((37,), dtype=T.float32)):
      factor = 2
      x = T.load(A) * factor
      factor = 9
      T.store(B, x)
      T.store(C, x + 3)
    return kernel
  kernel = factory()
  original = np.arange(37, dtype=np.float32)
  a = original.copy()
  b = np.empty_like(a)
  kernel(a, a, b)
  np.testing.assert_array_equal(a, original * 2)
  np.testing.assert_array_equal(b, original * 2 + 3)


def test_inplace_transpose_and_shifted_copy(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def transpose():
    @T.template
    def kernel(A: T.Tensor((17, 17), dtype=T.float32),
               C: T.Tensor((17, 17), dtype=T.float32)):
      T.copy(T.transpose(A), C)
    return kernel
  kernel = transpose()
  x = np.arange(289, dtype=np.float32).reshape(17, 17)
  expected = x.T.copy()
  kernel(x, x)
  np.testing.assert_array_equal(x, expected)


def test_axis_reductions_grid_and_reshape(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((7, 19), dtype=T.float32),
               C: T.Tensor((7,), dtype=T.float32)):
      x = T.load(A)
      T.store(C, T.reduce_add(x, axis=-1))
    return kernel
  kernel = factory()
  x = np.arange(133, dtype=np.float32).reshape(7, 19)
  np.testing.assert_array_equal(kernel(x), x.sum(-1))


def test_conversion_rounding_is_not_eliminated(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype=T.float32), C: T.Tensor((37,), dtype=T.float32)):
      x = T.load(A)
      rounded = T.cast(T.cast(x, T.float16), T.float32)
      T.store(C, rounded + 0.001)
    return kernel
  kernel = factory()
  x = np.linspace(0.99, 1.01, 37, dtype=np.float32)
  np.testing.assert_allclose(kernel(x), x.astype(np.float16).astype(np.float32) + 0.001, atol=1e-7)


def test_large_scalable_vector_is_stripmined(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((4096,), dtype=T.float32), C: T.Tensor((4096,), dtype=T.float32)):
      descriptor = T.Vector(T.float32, pow=9)
      x = T.load(A, vector=descriptor)
      T.store(C[:T.lanes(descriptor)], x + 1)
    return kernel
  kernel = factory()
  x = np.arange(4096, dtype=np.float32)
  np.testing.assert_array_equal(kernel(x), x + 1)
  # Large logical powers survive in Vector IR, while target guards select
  # legal streaming pieces instead of instantiating an unsupported native Tag.
  assert kernel.vector_ir.plans[next(k for k, r in kernel.vector_ir.plans.items() if r.base_power == 9)].base_power == 9
  assert "#elif VEC_MAX_POW >= 9" in kernel.generated_source
  assert "#else" in kernel.generated_source and "ScalableTag<float, 0>" in kernel.generated_source


def test_grid_reshape_scalar_select_and_division(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((3, 5), dtype=T.float32), C: T.Tensor((15,), dtype=T.float32), *, flag: bool = True):
      view = T.reshape(C, (3, 5))
      for row, col in T.grid(3, 5):
        view[row, col] = T.load(A[row, col]) / 2
      x = T.load(C)
      T.store(C, T.where(flag, x, -x) + 1 / 2)
    return kernel
  kernel = factory()
  a = np.arange(15, dtype=np.float32).reshape(3, 5)
  np.testing.assert_array_equal(kernel(a), a.ravel() / 2 + 0.5)
  np.testing.assert_array_equal(kernel(a, flag=False), -a.ravel() / 2 + 0.5)


def test_promoted_scratch_materializes_on_partial_update(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((17,), dtype=T.float32), C: T.Tensor((17,), dtype=T.float32)):
      scratch = T.alloc((17,), dtype=T.float16)
      T.store(scratch, T.load(A))
      T.store(scratch[0:1], 42)
      T.store(C, T.load(scratch, dtype=T.float32))
    return kernel
  kernel = factory()
  a = np.linspace(0.99, 1.01, 17, dtype=np.float32)
  expected = a.astype(np.float16).astype(np.float32)
  expected[0] = 42
  np.testing.assert_array_equal(kernel(a), expected)


def test_shifted_alias_negative_output_and_inline_reduction(compiler):
  N = T.dynamic(lower=0)
  S = T.dynamic()
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((N,), dtype=T.float32), C: T.Tensor((N,), (S,), T.float32)):
      T.store(C, T.load(A) + T.reduce_add(T.load(A)))
    return kernel
  kernel = factory()
  a = np.arange(66, dtype=np.float32)
  original = a[:65].copy()
  kernel(a[:65], a[1:])
  np.testing.assert_array_equal(a[1:], original + original.sum())
  original = np.arange(65, dtype=np.float32)
  result = np.empty_like(original)
  kernel(original, result[::-1])
  np.testing.assert_array_equal(result[::-1], original + original.sum())
  # The fused wider branch has one horizontal reduction; the fallback branch
  # combines two smaller horizontal reductions rather than duplicating work.
  assert kernel.generated_source.count(" = vecops::vec::reduce_add") == 1
  assert "/* reduction region: no capture boundary */" in kernel.generated_source
  assert "([&]() {" not in kernel.generated_source


@pytest.mark.parametrize("which", ["layernorm", "softmax"])
def test_float64_compute_and_numerical_edges(compiler, which):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((3, 37), dtype=T.float64), C: T.Tensor((3, 37), dtype=T.float64)):
      for row in range(3):
        x = T.load(A[row, :])
        if which == "layernorm":
          T.store(C[row, :], T.layernorm(x))
        else:
          T.store(C[row, :], T.softmax(x))
    return kernel
  kernel = factory()
  a = np.stack([np.ones(37) * 1e9, np.linspace(-700, 700, 37), 1e5 + np.linspace(-0.01, 0.01, 37)])
  if which == "layernorm":
    centered = a - a.mean(-1, keepdims=True)
    expected = centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + 1e-5)
  else:
    exponent = np.exp(a - a.max(-1, keepdims=True))
    expected = exponent / exponent.sum(-1, keepdims=True)
  np.testing.assert_allclose(kernel(a), expected, atol=2e-8, rtol=2e-8)


def test_dsl_batch_capture_isolation(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory(N, factor):
    @T.func
    def transform(x):
      return x * factor
    @T.template
    def kernel(A: T.Tensor((N,), dtype=T.float32), C: T.Tensor((N,), dtype=T.float32)):
      T.store(C, transform(T.load(A)))
    return kernel
  first, second = factory(17, 2), factory(37, 3)
  with vecops.collect_compile_requests() as collection:
    first(np.ones(17, dtype=np.float32))
    second(np.ones(37, dtype=np.float32))
  assert len(collection.requests) == 2
  built = vecops.compile_batch(collection.requests, parallelism=2, execution_parallelism=1)
  assert built.prepared == 2
  np.testing.assert_array_equal(first(np.ones(17, dtype=np.float32)), 2)
  np.testing.assert_array_equal(second(np.ones(37, dtype=np.float32)), 3)
  assert first.template.ir.body.identity != second.template.ir.body.identity


def test_cache_only_dsl_never_creates_directories(tmp_path):
  @T.template
  def kernel(A: T.Tensor((3,), dtype=T.float32), C: T.Tensor((3,), dtype=T.float32, access="output")):
    T.copy(A, C)
  compiler = vecops.Compiler(cache_mode="cache-only", cache_dir=tmp_path / "absent" / "cache", build_dir=tmp_path / "build", cc="missing", cxx="missing")
  with pytest.raises(ImportError, match="cache-only"):
    T.compile(kernel, compiler=compiler)
  assert not (tmp_path / "absent").exists()
  assert not (tmp_path / "build").exists()


def test_overlapping_scatter_output_is_rejected(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((3,), dtype=T.float32), C: T.Tensor((3,), (0,), T.float32)):
      T.copy(A, C)
    return kernel
  kernel = factory()
  a = np.arange(3, dtype=np.float32)
  with pytest.raises((ValueError, RuntimeError), match="overlapping DSL output"):
    kernel(a)


def test_masked_converting_memory_and_integer_predicates(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype=T.float16), C: T.Tensor((37,), dtype=T.float32)):
      indices = T.arange(37)
      mask = (indices & 1) == 0
      x = T.load(A, dtype=T.float32, mask=mask, other=-5)
      T.store(C, x + 1)
      T.store(C, 99, mask=~mask)
    return kernel
  kernel = factory()
  a = np.arange(37, dtype=np.float16)
  expected = a.astype(np.float32) + 1
  expected[1::2] = 99
  np.testing.assert_array_equal(kernel(a), expected)
  assert "opt::masked" in kernel.generated_source
  assert "opt::merge" in kernel.generated_source


def test_load_conversion_rounding_and_promoted_rounding(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype=T.float32), C: T.Tensor((37,), dtype=T.float32)):
      rounded = T.cast(T.load(A, dtype=T.float16), T.float32)
      tmp = T.alloc((37,), dtype=T.float16)
      T.store(tmp, rounded)
      T.store(C, T.load(tmp, dtype=T.float32) + 0.001)
    return kernel
  kernel = factory()
  a = np.linspace(0.99, 1.01, 37, dtype=np.float32)
  np.testing.assert_allclose(kernel(a), a.astype(np.float16).astype(np.float32) + 0.001, atol=1e-7)
  assert "dsl_f_scratch_" not in kernel.generated_source


def test_owned_scratch_inplace_transpose(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((5, 5), dtype=T.float32), C: T.Tensor((5, 5), dtype=T.float32)):
      tmp = T.alloc((5, 5), dtype=T.float32)
      T.store(tmp, T.load(A))
      T.copy(T.transpose(tmp), tmp)
      T.copy(tmp, C)
    return kernel
  kernel = factory()
  a = np.arange(25, dtype=np.float32).reshape(5, 5)
  np.testing.assert_array_equal(kernel(a), a.T)


def test_loop_carried_tile_compiles(compiler):
  @T.template
  def kernel(A: T.Tensor((3,), dtype=T.float32)):
    x = T.load(A)
    for i in range(2):
      x = x + 1
  T.compile(kernel, compiler=compiler)


def test_floor_modulo_and_true_division_vectors(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype=T.int64), C: T.Tensor((37,), dtype=T.float64)):
      x = T.load(A)
      y = x // 3 + x % -3 + x // 4 + x % 4
      T.store(C, y + x / 2)
    return kernel
  kernel = factory()
  a = np.arange(-18, 19, dtype=np.int64)
  np.testing.assert_array_equal(kernel(a), a // 3 + a % -3 + a // 4 + a % 4 + a / 2)


def test_float_floor_modulo_and_negative_scalar_index(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((5,), dtype=T.float64), C: T.Tensor((5,), dtype=T.float64)):
      x = T.load(A)
      T.store(C, x // 0.1 + x % -0.1 + T.load(A[-1]))
    return kernel
  kernel = factory()
  a = np.array([-3.7, -1.0, -0.0, 1.0, 3.7])
  np.testing.assert_allclose(kernel(a), a // 0.1 + a % -0.1 + a[-1], atol=1e-12)


def test_runtime_assertions_and_reshape_are_checked_in_release(compiler):
  S = T.dynamic()
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((15,), (S,), T.float32), C: T.Tensor((3, 5), dtype=T.float32), *, flag: bool = True):
      assert flag, "flag must be true"
      view = T.reshape(A, (3, 5))
      T.copy(view, C)
    return kernel
  kernel = factory()
  with pytest.raises((ValueError, RuntimeError), match="flag must be true"):
    kernel(np.ones(15, dtype=np.float32), flag=False)
  with pytest.raises((ValueError, RuntimeError), match="reshape requires a dense view"):
    kernel(np.ones(30, dtype=np.float32)[::2])


@pytest.mark.parametrize("target", ["Scalar", "AVX2"])
def test_portable_cpu_targets_and_cache_only_hit(tmp_path, target):
  @T.template
  def kernel(A: T.Tensor((37,), dtype=T.float32), C: T.Tensor((37,), dtype=T.float32, access="output")):
    T.store(C, T.softmax(T.load(A)))
  builder = vecops.Compiler(target=target, cc="gcc", cxx="g++", jobs=4,
                           cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  first = T.compile(kernel, out_idx=[-1], compiler=builder)
  a = np.linspace(-3, 3, 37, dtype=np.float32)
  expected = np.exp(a - a.max()) / np.exp(a - a.max()).sum()
  np.testing.assert_allclose(first(a), expected, atol=1e-7)
  reader = vecops.Compiler(target=target, cache_mode="cache-only", cc="missing", cxx="missing",
                          cache_dir=tmp_path / "cache", build_dir=tmp_path / "must-not-create")
  cached = T.compile(kernel, out_idx=[-1], compiler=reader)
  np.testing.assert_allclose(cached(a), expected, atol=1e-7)
  assert cached.source.is_file()
  assert not reader.build_dir.exists()


def test_helper_capture_is_frozen_and_tuple_arguments_are_recursive(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory(config):
    factor = 2
    @T.func
    def transform(x):
      return x * factor + config[0][0]
    @T.template
    def kernel(A: T.Tensor((3,), dtype=T.float32), C: T.Tensor((3,), dtype=T.float32)):
      T.store(C, transform(T.load(A)))
    factor = 9
    return kernel
  kernel = factory(((3,), None))
  np.testing.assert_array_equal(kernel(np.ones(3, dtype=np.float32)), 5)


def test_independent_reductions_fuse_one_memory_pass(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((257,), dtype=T.float32), C: T.Tensor((2,), dtype=T.float32)):
      x = T.load(A)
      total = T.reduce_add(x)
      squared = T.reduce_add(x * x)
      C[0] = total
      C[1] = squared
    return kernel
  kernel = factory()
  a = np.linspace(-3, 3, 257, dtype=np.float32)
  np.testing.assert_allclose(kernel(a), [a.sum(), (a * a).sum()], rtol=1e-6, atol=1e-5)
  assert "std::tuple{" in kernel.generated_source
  assert len([r for r in kernel.vector_ir.regions if len(r.roots) == 2]) == 1


def test_layernorm_moments_is_visible_and_fused(compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((257,), dtype=T.float32), C: T.Tensor((257,), dtype=T.float32)):
      x = T.load(A)
      T.store(C, T.layernorm(x, variance="moments"))
    return kernel
  kernel = factory()
  a = np.linspace(-3, 3, 257, dtype=np.float32)
  np.testing.assert_allclose(kernel(a), (a - a.mean()) / np.sqrt(a.var() + 1e-5), atol=1e-6, rtol=1e-5)
  assert "std::tuple{" in kernel.generated_source
  assert "vecops::ops::" not in kernel.generated_source


def test_blocked_regions_freeze_coordinates_and_validate_dynamic_shapes(compiler):
  N = T.dynamic(lower=0)
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((N,), dtype=T.float32), C: T.Tensor((N,), dtype=T.float32)):
      for block in T.serial(T.ceildiv(N, 32)):
        begin = block * 32
        end = begin + 32
        x = T.load(A[begin:end])
        T.store(C[begin:end], x + 1)
    return kernel
  kernel = factory()
  for n in (0, 1, 37, 257):
    a = np.arange(n, dtype=np.float32)
    np.testing.assert_array_equal(kernel(a), a + 1)


def test_dynamic_broadcast_dimensions_are_checked(compiler):
  M = T.dynamic(lower=0)
  N = T.dynamic(lower=0)
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((M,), dtype=T.float32),
               B: T.Tensor((N,), dtype=T.float32),
               C: T.Tensor((N,), dtype=T.float32)):
      T.store(C, T.load(A) + T.load(B))
    return kernel
  kernel = factory()
  np.testing.assert_array_equal(kernel(np.array([2], dtype=np.float32), np.arange(37, dtype=np.float32)), np.arange(37, dtype=np.float32) + 2)
  a = np.arange(37, dtype=np.float32)
  np.testing.assert_array_equal(kernel(a, a), a * 2)
  with pytest.raises((ValueError, RuntimeError), match="broadcast"):
    kernel(np.ones(3, dtype=np.float32), np.ones(5, dtype=np.float32))
