"""Register layout conversion, fused conversions and hardware ownership."""
import re
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.transpose import TransposePlan


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  path = tmp_path_factory.mktemp("dsl-transpose")
  return vecops.Compiler(target="Native", cc="gcc", cxx="g++", jobs=4,
                         cache_dir=path / "cache", build_dir=path / "build")


def kernel(compiler, ad, cd, compute):
  M, N = T.dynamic(lower=0), T.dynamic(lower=0)
  S0, S1, D0, D1 = (T.dynamic() for _ in range(4))
  @T.template
  def transpose(A: T.Tensor((M, N), (S0, S1), ad),
                C: T.Tensor((N, M), (D0, D1), cd, access="output")):
    T.transpose_copy(A, C, compute_dtype=compute)
  return T.compile(transpose, compiler=compiler)


PAIRS = [(d, d, d) for d in ("int8", "float16", "bfloat16", "float32", "float64")]
PAIRS += [(a, b, "float64" if "float64" in (a, b) else "float32")
          for a in ("int8", "float16", "float32", "float64")
          for b in ("int8", "float16", "float32", "float64") if a != b]


def dtype(name):
  if name == "bfloat16":
    return np.dtype(pytest.importorskip("ml_dtypes").bfloat16)
  return np.dtype(name)


@pytest.mark.parametrize("ad,cd,compute", PAIRS)
def test_dtype_shapes_and_layouts(compiler, ad, cd, compute):
  k = kernel(compiler, ad, cd, compute)
  for m, n in ((0, 17), (17, 0), (1, 32), (32, 1), (3, 5), (16, 16), (17, 19), (63, 129), (128, 128)):
    a = ((np.arange(m*n) % 251) - 125).astype(dtype(ad)).reshape(m, n)
    for layout in ("dense", "column-strided", "negative", "output-negative"):
      x = a
      backing = np.empty((n, m*2), dtype=dtype(cd))
      c = backing[:, ::2] if layout == "column-strided" else backing[:, :m]
      if layout == "negative":
        x = a[:, ::-1]
      if layout == "output-negative":
        c = c[::-1]
      k.operator.native(x, c, **k._core.specialization_values)
      np.testing.assert_array_equal(c, x.astype(dtype(compute)).astype(dtype(cd)).T)
  assert any(isinstance(region, TransposePlan) for region in k.vector_ir.regions)
  assert "local_interleave" in k.generated_source
  assert "vecops::ops::" not in k.generated_source
  assert not re.search(r"std::(?:array|vector)\s*<\s*vecops::vec::(?:Vec|Mask)", k.generated_source)


def test_batch_fusion_and_inplace(compiler):
  @T.template
  def fused(A: T.Tensor((2, 17, 19), dtype="float16"),
            C: T.Tensor((2, 19, 17), dtype="float32", access="output")):
    T.store(C, T.load(T.transpose(A, axes=(0, 2, 1)), dtype="float32") * 2 + 1)
  k = T.compile(fused, compiler=compiler)
  a = np.arange(2*17*19, dtype=np.float16).reshape(2, 17, 19)
  c = np.empty((2, 19, 17), dtype=np.float32)
  k.operator.native(a, c)
  np.testing.assert_array_equal(c, a.transpose(0, 2, 1).astype(np.float32)*2+1)
  inplace = kernel(compiler, "float32", "float32", "float32")
  a = np.arange(32*32, dtype=np.float32).reshape(32, 32)
  expected = a.T.copy()
  inplace.operator.native(a, a, **inplace._core.specialization_values)
  np.testing.assert_array_equal(a, expected)


def test_gather_override_and_hardware_validation(compiler):
  @T.template
  def gathered(A: T.Tensor((17, 19), dtype="float32"),
               C: T.Tensor((19, 17), dtype="float32", access="output")):
    T.transpose_copy(A, C, implementation="gather")
  k = T.compile(gathered, compiler=compiler)
  assert not any(isinstance(region, TransposePlan) for region in k.vector_ir.regions)
  @T.template
  def invalid():
    T.amx.zero(0)
  with pytest.raises(TypeError, match="matching lexical"):
    CPUBackend().lower(invalid.ir, ())
  @T.template
  def invalid_config():
    with T.amx.region(((17, 64),)):
      T.amx.zero(0)
  with pytest.raises(TypeError, match="rows in"):
    CPUBackend().lower(invalid_config.ir, ())
  @T.template
  def escape():
    with T.sme.region():
      x = T.zeros((16,))
    T.reduce_add(x)
  with pytest.raises(TypeError, match="unbound kernel name"):
    CPUBackend().lower(escape.ir, ())


@pytest.mark.skipif("amx_tile" not in Path("/proc/cpuinfo").read_text(), reason="AMX-TILE target required")
def test_amx_tile_memory_and_release(compiler):
  @T.template
  def amx_copy(A: T.Tensor((16, 16), dtype="int32"),
               C: T.Tensor((16, 16), dtype="int32", access="output")):
    with T.amx.region(((16, 64),)):
      T.amx.load(0, A)
      T.amx.store(0, C)
    with T.amx.region(((16, 64),)):
      T.amx.zero(0)
      T.amx.store(0, C)
  k = T.compile(amx_copy, compiler=compiler)
  a = np.arange(256, dtype=np.int32).reshape(16, 16)
  c = np.empty_like(a)
  k.operator.native(a, c)
  np.testing.assert_array_equal(c, np.zeros_like(a))
  assert "with_resources" in k.generated_source
  assert "with_configuration" in k.generated_source
  assert "VECOPS_INLINE_LAMBDA_NOEXCEPT" in k.generated_source


@pytest.mark.skipif("amx_tile" not in Path("/proc/cpuinfo").read_text(), reason="AMX-TILE target required")
def test_amx_materializes_scratch_and_preserves_prior_load(compiler):
  @T.template
  def moves(A: T.Tensor((16, 16), dtype="int32"),
            C: T.Tensor((16, 16), dtype="int32", access="inout"),
            D: T.Tensor((16, 16), dtype="int32", access="output")):
    tmp = T.alloc((16, 16), dtype="int32")
    T.store(tmp, T.load(A) + 1)
    original = T.load(C)
    with T.amx.region(((16, 64),)):
      T.amx.load(0, tmp)
      T.amx.store(0, C)
    T.store(D, original)
  k = T.compile(moves, compiler=compiler)
  a = np.arange(256, dtype=np.int32).reshape(16, 16)
  c, d = np.full_like(a, 42), np.empty_like(a)
  k.operator.native(a, c, d)
  np.testing.assert_array_equal(c, a + 1)
  np.testing.assert_array_equal(d, np.full_like(a, 42))


def test_explicit_sme_source_and_no_vector_arrays():
  @T.template
  def sme(A: T.Tensor((17, 19), dtype="float16"),
          C: T.Tensor((19, 17), dtype="float32", access="output")):
    with T.sme.region():
      T.transpose_copy(A, C, compute_dtype="float32", implementation="sme")
  artifact = CPUBackend().lower(sme.ir, ())
  assert artifact.source.count(".with_resources(") == 1
  assert "vecops::vec::sme::write_hor" in artifact.source
  assert "vecops::vec::sme::read_ver" in artifact.source
  assert "vecops/vec/details/" not in artifact.source
  assert "std::array<vecops::vec::" not in artifact.source


def test_benchmark_catalog_matches_cpp():
  import importlib.util
  path = Path(__file__).parents[2] / "benchmarks/dsl_transpose.py"
  spec = importlib.util.spec_from_file_location("transpose_bench", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  assert len(module.catalog()) == 133
  assert len(module.catalog()) * 3 == 399


def test_memory_pass_registration_and_layout_plans():
  from vecops.language._compiler.pipeline import cpu_memory_pipeline
  assert "cpu-plan-transpose" in [spec.name for spec in cpu_memory_pipeline().specs]
  @T.template
  def two_outputs(A: T.Tensor((16, 16), dtype="float16"),
                  C: T.Tensor((16, 16), dtype="float64", access="output"),
                  D: T.Tensor((16, 16), dtype="float64", access="output")):
    T.transpose_copy(A, C, compute_dtype="float64")
    T.transpose_copy(A, D, compute_dtype="float64")
  artifact = CPUBackend().lower(two_outputs.ir, ())
  plans = [region for region in artifact.lowered.regions if isinstance(region, TransposePlan)]
  assert len(plans) >= 2
  assert all(plan.dense_destination for plan in plans)
  assert any(network.dtype == "float16" for plan in plans for _, network in plan.variants)
  assert artifact.source.count("/* register layout transpose */") >= 2
  with pytest.raises(ValueError, match="multiple of 16"):
    CPUBackend().lower(two_outputs.ir, (), options={"cpu_transpose_cache_block": 7})


def test_numpy_bfloat16_view_and_empty_strides():
  a = np.zeros((2, 3), dtype=dtype("bfloat16"))
  view = vecops._C.TensorView(a, writable=True)
  assert view.dtype == vecops.bfloat16
  assert tuple(view.sizes) == (2, 3)
  assert tuple(view.strides) == (3, 1)
  a.flags.writeable = False
  with pytest.raises(ValueError, match="writable"):
    vecops._C.TensorView(a, writable=True)
  with pytest.raises(TypeError, match="unsupported"):
    vecops._C.TensorView(np.zeros((3,), dtype="V2"), writable=False)
  empty = vecops._C.TensorView(np.empty((0,), dtype=np.float32), writable=False)
  assert tuple(empty.strides) == (1,)


def test_unavailable_or_illegal_explicit_sme_does_not_silently_fallback():
  @T.template
  def invalid(A: T.Tensor((17, 19), dtype="float32"),
              C: T.Tensor((19, 17), dtype="float32", access="output")):
    T.store(C, T.load(T.transpose(A)) * 2, implementation="sme")
  with pytest.raises(TypeError, match="explicit SME transpose"):
    CPUBackend().lower(invalid.ir, ())
  @T.template
  def bad_config():
    with T.amx.region((16, 64)):
      T.amx.zero(0)
  with pytest.raises(TypeError, match="configuration entries"):
    CPUBackend().lower(bad_config.ir, ())


def test_half_widening_and_bitwise_transpose(compiler):
  random = np.random.default_rng(42)
  half = random.integers(0, 65536, size=(16, 16), dtype=np.uint16).view(np.float16)
  same = kernel(compiler, "float16", "float16", "float16")
  c = np.empty_like(half)
  same.operator.native(half, c, **same._core.specialization_values)
  np.testing.assert_array_equal(c.view(np.uint16), half.T.copy().view(np.uint16))
  widened = kernel(compiler, "float16", "float64", "float64")
  d = np.empty(half.shape, dtype=np.float64)
  widened.operator.native(half, d, **widened._core.specialization_values)
  np.testing.assert_array_equal(d, half.astype(np.float64).T)
  finite = np.isfinite(d)
  np.testing.assert_array_equal(np.signbit(d[finite]), np.signbit(half.T[finite]))
