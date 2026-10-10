# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Prove raw/whole-native-packed spatial bounds independently, never K bounds.

The narrow candidate deliberately leaves sliced packed origins on their prior
path. Native numerical tests derive panel capacity from the calling thread,
check physical canaries, and retain odd/zero K, priors and fused epilogues.
"""

import platform

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES


def _template(dtype="bfloat16", *, side="B", level="dot", packed_origin=False, implementation="sme"):
  M, N, K, PS = (T.dynamic(lower=0) for _ in range(4))
  acc_dtype = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), dtype),
    B: T.Tensor((K, N), (1, K), dtype),
    P: T.Tensor((PS,), dtype=dtype),
    D: T.Tensor((M, N), dtype=acc_dtype),
    C: T.Tensor((M, N), dtype=acc_dtype, access="output"),
  ):
    a = T.packed(P, (M, K), side="A", format="auto") if side == "A" else T.load(A)
    b = T.packed(P, (K, N), side="B", format="auto") if side == "B" else T.load(B)
    if packed_origin:
      # Explicit cache panels create packed origins, including the first
      # zero-origin panel; these must not get the new whole-panel proof.
      value = T.library.matmul(
        (a, b), C, acc=T.load(D), BM=64, BN=64, BK=3, packing="none", swap=False, implementation=implementation
      )
    elif level == "matmul":
      value = T.library.matmul((a, b), C, acc=T.load(D), packing="none", swap=False, implementation=implementation)
    else:
      value = T.dot((a, b), C, acc=T.load(D), implementation=implementation)
    T.store(C, T.max(value * 0.5 + 0.125, 0))

  return kernel


def _lower(**kwargs):
  return (
    CPUBackend()
    .lower(
      _template(**kwargs).ir,
      (),
      target="SME",
      options={
        "cpu_matmul_pipeline": "off",
        "cpu_matmul_fusion": "always",
        "_vector_state": VectorState(64, 64),
      },
    )
    .source
  )


def _wrapper(source, dtype):
  element = CPP_TYPES[dtype]
  return "\n".join(
    (
      '#include "vecops/Kernel.h"',
      "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
      source,
      (
        'extern "C" void mixed_spatial(void* ap,void* bp,void* pp,void* dp,void* cp,'
        "vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t count) {"
      ),
      f"auto A=vecops::tensor::make_tensor(static_cast<{element}*>(ap),vecops::tensor::make_shape(vecops::meta::Any{{m}},vecops::meta::Any{{k}}),vecops::tensor::make_strides(vecops::meta::Any{{k}},vecops::meta::cint<1>));",
      f"auto B=vecops::tensor::make_tensor(static_cast<{element}*>(bp),vecops::tensor::make_shape(vecops::meta::Any{{k}},vecops::meta::Any{{n}}),vecops::tensor::make_strides(vecops::meta::cint<1>,vecops::meta::Any{{k}}));",
      f"auto P=vecops::tensor::make_tensor(static_cast<{element}*>(pp),vecops::tensor::make_shape(vecops::meta::Any{{count}}));",
      "auto D=vecops::tensor::make_tensor(static_cast<float*>(dp),vecops::tensor::make_shape(vecops::meta::Any{m},vecops::meta::Any{n}));",
      "auto C=vecops::tensor::make_tensor(static_cast<float*>(cp),vecops::tensor::make_shape(vecops::meta::Any{m},vecops::meta::Any{n}));",
      "__kernel__(A,B,P,D,C);",
      "}",
    )
  )


def _packer(dtype, side):
  R, S, PS = (T.dynamic(lower=0) for _ in range(3))
  stride = (S, 1) if side == "A" else (1, R)

  @T.template
  def kernel(A: T.Tensor((R, S), stride, dtype), P: T.Tensor((PS,), dtype=dtype, access="output")):
    T.pack(A, P, side=side, format="auto")

  return kernel


@pytest.mark.parametrize("side", ("A", "B"))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64", "int8", "uint8"))
def test_native_mixed_spatial_full_fringe_zero_odd_k_fusion_prior_and_canaries(side, dtype, tmp_path):
  if dtype == "bfloat16":
    np.dtype(pytest.importorskip("ml_dtypes").bfloat16)
  state = get_vector_state()
  arm = platform.machine().lower() in {"aarch64", "arm64"}
  integer = dtype in {"int8", "uint8"}
  acc_dtype = "float64" if dtype == "float64" else "int32" if integer else "float32"
  if arm:
    assert state.sme_bytes is not None
    lanes = state.sme_bytes // np.dtype(acc_dtype).itemsize
    panel, group = 2 * lanes, max(1, 4 // np.dtype(dtype).itemsize)
    implementation = "sme"
  else:
    if dtype not in {"float16", "bfloat16"}:
      pytest.skip("additional mixed native SME dtype groups require the ARM target")
    lanes, panel, group, implementation = 16, 16, 32, "auto"
  shape = (4 * lanes, 1, 257) if side == "B" else (1, 4 * lanes, 257)
  max_spatial = 8 * lanes
  count = ((max_spatial + panel - 1) // panel) * panel * ((258 + group - 1) // group) * group
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  options = {"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"}
  pack = T.compile(_packer(dtype, side), compiler=compiler, pass_options=options)
  kernels = {
    (level, window): T.compile(
      _template(dtype, side=side, level=level, packed_origin=window, implementation=implementation),
      compiler=compiler,
      pass_options=options,
    )
    for level, window in (("dot", False), ("matmul", False), ("matmul", True))
  }
  m, n, k = shape
  with vecops.collect_compile_requests() as collector:
    pack.operator(
      vecops.TensorMeta((m, k) if side == "A" else (k, n), strides=(k, 1) if side == "A" else (1, k), dtype=dtype),
      vecops.TensorMeta((count,), dtype=dtype),
      **pack._core.specialization_values,
    )
    for kernel in kernels.values():
      kernel.operator(
        vecops.TensorMeta((m, k), strides=(k, 1), dtype=dtype),
        vecops.TensorMeta((k, n), strides=(1, k), dtype=dtype),
        vecops.TensorMeta((count,), dtype=dtype),
        vecops.TensorMeta((m, n), dtype=acc_dtype),
        vecops.TensorMeta((m, n), dtype=acc_dtype),
        **kernel._core.specialization_values,
      )
  assert len(collector.requests) == 4
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  rng = np.random.default_rng(548)
  shapes = [
    (0, 1, 3),
    (1, 0, 3),
    (4 * lanes, 1, 0),
    (4 * lanes - 1, 1, 5),
    (4 * lanes, 1, 257),
    (4 * lanes + 1, 1, 5),
    (8 * lanes, 1, 7),
    (3 * lanes + 1, 1, 1),
    (3 * lanes, 1, 258),
    (lanes, 1, 2),
    (2 * lanes + 1, 1, 3),
    (4 * lanes, 2 * lanes + 1, 5),
  ]
  if side == "A":
    shapes = [(n, m, k) for m, n, k in shapes]
  calls = 0
  for m, n, k in shapes:
    lower = 0 if dtype == "uint8" else -2
    scale = 1 if integer else 8
    a = (rng.integers(lower, 3, size=(m, k)) / scale).astype(dtype)
    # Empty ndarray strides must retain the explicit column layout.
    b = np.ndarray(
      (k, n),
      dtype=dtype,
      buffer=np.empty(max(1, k * n), dtype=dtype),
      strides=(np.dtype(dtype).itemsize, k * np.dtype(dtype).itemsize),
    )
    b[:] = (rng.integers(lower, 3, size=(k, n)) / scale).astype(dtype)
    protected = np.full(count + 32, 7, dtype=dtype)
    packed = protected[16:-16]
    pack.operator.native(a if side == "A" else b, packed, **pack._core.specialization_values)
    prior = (rng.integers(-2, 3, size=(m, n)) / (1 if integer else 16)).astype(acc_dtype)
    product = a.astype(acc_dtype) @ b.astype(acc_dtype) + prior
    expected = np.maximum(product * 0.5 + 0.125, 0).astype(acc_dtype)
    for kernel in kernels.values():
      protected_c = np.full(m * n + 32, -7, dtype=acc_dtype)
      # Reshaping an empty interior slice can yield a nonzero row stride for
      # N=0. Supply the declared dense ABI explicitly, as for the column B.
      element = np.dtype(acc_dtype).itemsize
      c = np.ndarray((m, n), dtype=acc_dtype, buffer=protected_c, offset=16 * element, strides=(n * element, element))
      kernel.operator.native(a, b, packed, prior, c, **kernel._core.specialization_values)
      np.testing.assert_array_equal(c, expected)
      np.testing.assert_array_equal(protected_c[:16], np.full(16, -7, dtype=acc_dtype))
      np.testing.assert_array_equal(protected_c[-16:], np.full(16, -7, dtype=acc_dtype))
      calls += 1
    np.testing.assert_array_equal(protected[:16], np.full(16, 7, dtype=dtype))
    np.testing.assert_array_equal(protected[-16:], np.full(16, 7, dtype=dtype))
  assert calls == 36
