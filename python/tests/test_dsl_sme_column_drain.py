"""Direct ZA column stores need a proven unit stride, never an assumed layout."""

import platform
import re

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_matmul_sve_optimized import _cross
from test_dsl_sme_exact_blocks import _body
from test_dsl_sme_family_owners import _assert_inlined_owners, _owners
from test_dsl_sme_mixed_packed_spatial import _packer
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES


def _template(dtype="bfloat16", *, static_n=None, seed=False, packed=True, transformed=False):
  M, K, PS, CS = (T.dynamic() for _ in range(4))
  N = T.dynamic() if static_n is None else static_n
  acc_dtype = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  c_stride = (CS, 1) if static_n is None else (N, 1)

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), dtype),
    B: T.Tensor((K, N), (1, K), dtype),
    P: T.Tensor((PS,), dtype=dtype),
    D: T.Tensor((M, N), dtype=acc_dtype),
    C: T.Tensor((M, N), c_stride, dtype=acc_dtype, access="output"),
  ):
    b = T.packed(P, (K, N), side="B", format="auto") if packed else T.load(B)
    value = T.dot((T.load(A), b), C, acc=T.load(D) if seed else None, implementation="sme")
    T.store(C, T.max(value * 0.5 + 0.125, 0) if transformed else value)

  return kernel


def _lower(**kwargs):
  return (
    CPUBackend()
    .lower(
      _template(**kwargs).ir,
      (),
      target="SME",
      options={"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always", "_vector_state": VectorState(64, 64)},
    )
    .source
  )


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64", "int8", "uint8"))
def test_static_single_column_uses_vertical_direct_seed_and_store(dtype):
  source = _lower(dtype=dtype, static_n=1, seed=True)
  assert "vecops::vec::sme::load_ver<" in source
  assert "vecops::vec::sme::store_ver<" in source
  assert "runtime contiguous column" not in source
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64", "int8", "uint8"))
def test_dynamic_column_direct_store_has_exact_unit_stride_and_extent_guards(dtype):
  source = _lower(dtype=dtype)
  guards = [line for line in source.splitlines() if "runtime contiguous column" in line]
  assert guards
  for line in guards:
    stride, columns, rows = line.split(" && ")
    assert re.search(r"parameter_C_stride_0_\d+\)==1$", stride)
    assert any(name in columns for name in ("sme_extent_n_value", "sme_matrix_lanes")) and columns.endswith(")==1")
    assert any(name in rows for name in ("sme_extent_m_value", "sme_matrix_lanes"))
    assert rows.endswith(")>1) /* runtime contiguous column: one vertical ZA store */ {")
  assert "vecops::vec::sme::store_ver<" in source
  assert "vecops::vec::sme::store_hor<" in source
  assert "original ZA slice orientation" in source
  assert_no_vec_mask_arrays(source)


def test_transformed_output_does_not_enter_direct_runtime_store():
  source = _lower(transformed=True)
  assert "runtime contiguous column" not in source
  assert "vecops::vec::sme::read_hor<" in source


def test_runtime_single_column_proof_emits_one_constant_index_store_without_loop():
  source = _lower()
  fast = _body(source, "/* runtime contiguous column: one vertical ZA store */")
  assert "/* exactly one active vertical slice */" in fast
  assert re.search(r"constexpr vecops::nint_t \w+=0;", fast)
  assert fast.count("vecops::vec::sme::store_ver<") == 1
  assert not re.search(r"\b(?:for|while)\s*\(", fast)


def _wrapper(source, dtype):
  scalar = CPP_TYPES[dtype]
  return "\n".join(
    (
      '#include "vecops/Kernel.h"',
      "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
      source,
      'extern "C" void column_drain(void* ap,void* bp,void* pp,void* dp,void* cp,vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t count,vecops::nint_t cs) {',
      f"auto A=vecops::tensor::make_tensor(static_cast<{scalar}*>(ap),vecops::tensor::make_shape(vecops::meta::Any{{m}},vecops::meta::Any{{k}}),vecops::tensor::make_strides(vecops::meta::Any{{k}},vecops::meta::cint<1>));",
      f"auto B=vecops::tensor::make_tensor(static_cast<{scalar}*>(bp),vecops::tensor::make_shape(vecops::meta::Any{{k}},vecops::meta::Any{{n}}),vecops::tensor::make_strides(vecops::meta::cint<1>,vecops::meta::Any{{k}}));",
      f"auto P=vecops::tensor::make_tensor(static_cast<{scalar}*>(pp),vecops::tensor::make_shape(vecops::meta::Any{{count}}));",
      "auto shape=vecops::tensor::make_shape(vecops::meta::Any{m},vecops::meta::Any{n});",
      "auto D=vecops::tensor::make_tensor(static_cast<float*>(dp),shape);",
      "auto C=vecops::tensor::make_tensor(static_cast<float*>(cp),shape,vecops::tensor::make_strides(vecops::meta::Any{cs},vecops::meta::cint<1>));",
      "__kernel__(A,B,P,D,C);",
      "}",
    )
  )


@pytest.mark.parametrize("dtype", ("bfloat16", "float32"))
def test_real_cross_direct_dynamic_store_has_vertical_and_horizontal_inline_owners(dtype, tmp_path):
  assembly = _cross(tmp_path, _lower(dtype=dtype, seed=True), lambda source: _wrapper(source, dtype), vl=64, svl=64)
  assert re.search(r"\bst1w\s+\{za\dv\.s", assembly)
  assert re.search(r"\bst1w\s+\{za\dh\.s", assembly)
  owners = _owners(assembly, vl=64, svl=64)
  assert owners
  _assert_inlined_owners(assembly, len(owners), vl=64, svl=64)
  assert not any(re.search(r"^\s*(?:str|ldr|stp|ldp)\s+(?:z\d|p\d)", owner, re.MULTILINE) for owner in owners)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64", "int8", "uint8"))
def test_native_column_drain_prior_fringe_zero_and_signed_output_strides(dtype, tmp_path):
  if platform.machine().lower() not in {"aarch64", "arm64"}:
    pytest.skip("direct ZA memory instructions require the target ARM CPU")
  if dtype == "bfloat16":
    np.dtype(pytest.importorskip("ml_dtypes").bfloat16)
  state = get_vector_state()
  assert state.sme_bytes is not None
  acc_dtype = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  element = np.dtype(acc_dtype).itemsize
  lanes = state.sme_bytes // element
  group = max(1, 4 // np.dtype(dtype).itemsize)
  count = 8 * lanes * ((258 + group - 1) // group) * group
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  options = {"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"}
  pack = T.compile(_packer(dtype, "B"), compiler=compiler, pass_options=options)
  kernels = {
    (static, seed, packed): T.compile(
      _template(dtype, static_n=1 if static else None, seed=seed, packed=packed),
      compiler=compiler,
      pass_options=options,
    )
    for static in (False, True)
    for seed in (False, True)
    for packed in (False, True)
  }
  rng = np.random.default_rng(557)
  lower, scale = (0 if dtype == "uint8" else -2), (1 if dtype in {"int8", "uint8"} else 8)
  shapes = (
    (0, 1, 3),
    (1, 0, 3),
    (4 * lanes, 1, 0),
    (4 * lanes - 1, 1, 5),
    (4 * lanes, 1, 257),
    (4 * lanes + 1, 1, 3),
    (8 * lanes, 1, 7),
    (3 * lanes + 1, 1, 1),
    (2 * lanes, 2, 5),
  )
  calls = 0
  for m, n, k in shapes:
    a = np.ndarray(
      (m, k),
      dtype=dtype,
      buffer=np.empty(max(1, m * k), dtype=dtype),
      strides=(k * np.dtype(dtype).itemsize, np.dtype(dtype).itemsize),
    )
    a[:] = (rng.integers(lower, 3, size=(m, k)) / scale).astype(dtype)
    b = np.ndarray(
      (k, n),
      dtype=dtype,
      buffer=np.empty(max(1, k * n), dtype=dtype),
      strides=(np.dtype(dtype).itemsize, k * np.dtype(dtype).itemsize),
    )
    b[:] = (rng.integers(lower, 3, size=(k, n)) / scale).astype(dtype)
    packed = np.full(count, 7, dtype=dtype)
    pack.operator.native(b, packed, **pack._core.specialization_values)
    prior = np.ndarray(
      (m, n), dtype=acc_dtype, buffer=np.empty(max(1, m * n), dtype=acc_dtype), strides=(n * element, element)
    )
    prior[:] = (rng.integers(-2, 3, size=(m, n)) / (1 if scale == 1 else 16)).astype(acc_dtype)
    product = a.astype(acc_dtype) @ b.astype(acc_dtype)
    for (static, seed, _), kernel in kernels.items():
      if static and n != 1:
        continue
      strides = (n,) if static else (n, 2 * n + 3, -(2 * n + 3))
      for stride in strides:
        lo, hi = min(0, (m - 1) * stride) if m else 0, max(0, (m - 1) * stride) if m else 0
        protected = np.full(max(1, hi - lo + n) + 32, -7, dtype=acc_dtype)
        expected_backing = protected.copy()
        kwargs = {
          "shape": (m, n),
          "dtype": acc_dtype,
          "offset": (16 - lo) * element,
          "strides": (stride * element, element),
        }
        output = np.ndarray(buffer=protected, **kwargs)
        expected_view = np.ndarray(buffer=expected_backing, **kwargs)
        expected_view[:] = product + prior if seed else product
        kernel.operator.native(a, b, packed, prior, output, **kernel._core.specialization_values)
        np.testing.assert_array_equal(protected, expected_backing)
        calls += 1
        if seed and stride == n:
          # A raw prior may alias the destination. Every tile must read its
          # initial accumulator before the direct ZA store changes memory.
          protected.fill(-7)
          expected_backing.fill(-7)
          output[:] = prior
          expected_view[:] = product + prior
          kernel.operator.native(a, b, packed, output, output, **kernel._core.specialization_values)
          np.testing.assert_array_equal(protected, expected_backing)
          calls += 1
  assert calls == 168
