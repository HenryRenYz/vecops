# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Native complete-capture output acceptance with the independent NumPy oracle."""

import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from test_dsl_sme_prepared_a_native import _arguments
from vecops._arm_vector_state import get_vector_state
from vecops.language._library.matrix_recipe import _whole_k_matrix
from vecops.language._library.sme_matmul import dot_into, prepare_a, prepare_b

pytestmark = pytest.mark.skipif(
  platform.machine().lower() not in {"aarch64", "arm64"} or get_vector_state().sme_bytes is None,
  reason="native complete-capture output acceptance requires SME",
)


def template(a_dtype, b_dtype, acc_dtype):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), acc_dtype),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), acc_dtype),
    Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), acc_dtype),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), acc_dtype, access="output"),
  ):
    a = T.cast(T.load(A) * 2 + 1, a_dtype)
    b = T.cast(T.load(B) - 1, b_dtype)
    _whole_k_matrix(
      (a, b),
      C,
      dot_into,
      prepare_a,
      prepare_b,
      T.max(1, n),
      acc=T.load(Prior),
      acc_dtype=acc_dtype,
      epilogue=None,
      raw_a=None,
      raw_b=None,
    )

  return kernel


@pytest.fixture(scope="module", params=[("bfloat16", "bfloat16", "float32"), ("uint8", "int8", "int32")])
def captured_kernel(request, tmp_path_factory):
  family = request.param
  root = tmp_path_factory.mktemp("sme-captured-output-" + family[0])
  compiled = T.compile(
    template(*family), compiler=make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build")
  )
  return compiled, family


def test_native_complete_capture_preserves_alias_seed_tail_and_zero_k(captured_kernel):
  compiled, family = captured_kernel
  reduction = 4097 if family[0] == "bfloat16" else 65
  for m, n, k, reverse, alias in (
    (65, 71, reduction, False, None),
    (65, 71, reduction, True, None),
    (65, 71, reduction, True, "A"),
    (65, 71, reduction, True, "B"),
    (17, 19, 0, False, None),
    (0, 19, 7, False, None),
    (17, 0, 7, False, None),
  ):
    a, b, prior, c, transformed_expected, root, before, mask = _arguments(family, m, n, k, reverse, alias)
    # The shared setup uses an exact independent integer oracle followed by
    # 2*x+1. This kernel has no epilogue: recover the exact pre-epilogue result.
    expected = ((transformed_expected.astype(np.int64) - 1) // 2).astype(c.dtype)
    if m == 0:
      prior = np.zeros((1, n), dtype=c.dtype)
    compiled(a, b, prior, c)
    np.testing.assert_array_equal(c, expected)
    np.testing.assert_array_equal(root[mask], before[mask])
