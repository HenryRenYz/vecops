# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""NC-local native A auto packing; strict semantics and pre-K dispatch."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from test_dsl_matmul_sve_optimized import _strided_output
from test_dsl_sve_packed_mmla import _sme_available
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def _template(*, shape=None, library=True, packing="auto", bm=None, bn=None, bk=None, row_a=False, dtype="bfloat16"):
  M, N, K = tuple(T.Const(v) for v in shape) if shape else tuple(T.Dynamic(name=n, lower=0) for n in ("M", "N", "K"))
  A0, C0, C1, D0, D1 = (T.dynamic() for _ in range(5))
  if row_a:
    A0 = K
  acc_dtype = "float64" if dtype == "float64" else "float32"

  @T.template
  def native_a(
    A: T.Tensor((M, K), (A0, 1), dtype),
    B: T.Tensor((K, N), (1, K), dtype),
    C: T.Tensor((M, N), (C0, C1), acc_dtype, access="output"),
    D: T.Tensor((M, N), (D0, D1), acc_dtype),
    Bias: T.Tensor((1, N), dtype=acc_dtype),
  ):
    product = (
      T.library.matmul((T.load(A), T.load(B)), C, acc=T.load(D), packing=packing, swap=False, BM=bm, BN=bn, BK=bk)
      if library
      else T.dot((T.load(A), T.load(B)), C, acc=T.load(D))
    )
    T.store(C, T.max((product + T.load(Bias)) * 0.5 + 0.125, 0))

  return native_a


def _lower(*, vl=64, svl=64, **kwargs):
  result = CPUBackend().lower(
    _template(**kwargs).ir,
    (),
    target="SME",
    options={"cpu_matmul_pipeline": "off", "_vector_state": VectorState(vl, svl)},
  )
  return result


def _wrapper(source, *, dtype="bfloat16"):
  input_type, acc_type = ("double", "double") if dtype == "float64" else ("vecops::bfloat16_t", "float")
  definitions = (
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    'extern "C" void native_a(void* ap,void* bp,void* cp,void* dp,void* xp,vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t as) {',
    "auto M=vecops::meta::Any{m},N=vecops::meta::Any{n},K=vecops::meta::Any{k};",
    f"auto A=vecops::tensor::make_tensor(static_cast<{input_type}*>(ap),vecops::tensor::make_shape(M,K),vecops::tensor::make_strides(vecops::meta::Any{{as}},vecops::meta::cint<1>));",
    f"auto B=vecops::tensor::make_tensor(static_cast<{input_type}*>(bp),vecops::tensor::make_shape(K,N),vecops::tensor::make_strides(vecops::meta::cint<1>,K));",
    f"auto C=vecops::tensor::make_tensor(static_cast<{acc_type}*>(cp),vecops::tensor::make_shape(M,N),vecops::tensor::make_strides(N,vecops::meta::cint<1>));",
    f"auto D=vecops::tensor::make_tensor(static_cast<{acc_type}*>(dp),vecops::tensor::make_shape(M,N),vecops::tensor::make_strides(N,vecops::meta::cint<1>));",
    f"auto Bias=vecops::tensor::make_tensor(static_cast<{acc_type}*>(xp),vecops::tensor::make_shape(vecops::meta::cint<1>,N),vecops::tensor::make_strides(N,vecops::meta::cint<1>));",
    "__kernel__(A,B,C,D,Bias);}",
  )
  return "\n".join(definitions)


@pytest.mark.skipif(not _sme_available(), reason="actual ARM SME required")
@pytest.mark.parametrize("blocked", (False, True))
def test_native_a_arm_native_profit_and_fallback_alias_strides(blocked, tmp_path):
  pytest.importorskip("ml_dtypes")
  template = _template(bm=64 if blocked else None, bn=256 if blocked else None, bk=65 if blocked else None)
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(template, compiler=compiler)
  rng = np.random.default_rng(22)
  for m, n, k in ((1, 16, 5), (2, 5, 7), (8, 256, 129), (64, 256, 129), (128, 256, 257)):
    a = (rng.integers(-3, 4, size=(m, k)) / 8).astype("bfloat16")
    backing = (rng.integers(-3, 4, size=(n, k)) / 8).astype("bfloat16")
    b = backing.T
    bias = rng.integers(-2, 3, size=(1, n)).astype(np.float32)
    for negative_a in (False, True):
      av = a[::-1] if negative_a else a
      for layout in ("row", "column", "negative"):
        c = _strided_output(m, n, np.float32, layout)
        c[:] = rng.integers(-2, 3, size=(m, n))
        for broadcast in (False, True):
          d = np.broadcast_to(c[:1], (m, n)) if broadcast else c
          expected = np.maximum((av.astype(np.float32) @ b.astype(np.float32) + d.copy() + bias) * 0.5 + 0.125, 0)
          kernel.operator.native(av, b, c, d, bias, **kernel._core.specialization_values)
          np.testing.assert_array_equal(
            c, expected, err_msg=f"{blocked=}, {(m, n, k)}, {negative_a=}, {layout=}, {broadcast=}"
          )
