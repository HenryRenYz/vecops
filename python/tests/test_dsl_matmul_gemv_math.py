# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Completed GEMV dots share a vector epilogue, including wide seed/output."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_source_algorithms_lowered, make_compiler


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-gemv-math")
  return make_compiler(target="Native", jobs=6, cache_dir=root / "cache", build_dir=root / "build")


@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
def test_grouped_gemv_math_and_cache_continuation(compiler, column, wide, blocked):
  m, n, k = (7, 1, 65) if column else (1, 7, 65)
  dtype = "float64" if wide else "float32"

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype="float32"),
    B: T.Tensor((k, n), (1, k), "float32"),
    D: T.Tensor((m, n), dtype=dtype),
    C: T.Tensor((m, n), dtype=dtype, access="output"),
  ):
    if blocked:
      value = T.library.matmul((A, B), C, acc=T.load(D), BM=16, BN=16, BK=32, implementation="vector")
    else:
      value = T.dot((A, B), C, acc=T.load(D), implementation="vector")
    value = T.cast(value, dtype)
    T.store(C, value / (1 + T.exp(-value)))

  compiled = T.compile(kernel, compiler=compiler)
  rng = np.random.default_rng(13)
  a = rng.normal(size=(m, k)).astype(np.float32)
  b = rng.normal(size=(n, k)).astype(np.float32).T
  d = rng.normal(size=(m, n)).astype(dtype)
  c = np.empty((m, n), dtype)
  compiled.operator.native(a, b, d, c)
  value = (a @ b + d.astype(np.float32)).astype(dtype)
  np.testing.assert_allclose(c, value / (1 + np.exp(-value)), rtol=3e-5, atol=3e-5)
  assert_source_algorithms_lowered(compiled.backend_ir)
