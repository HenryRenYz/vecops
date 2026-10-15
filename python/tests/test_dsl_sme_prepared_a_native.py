# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Native full-K SME reuse against an independent exact integer oracle."""

import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import get_vector_state
from vecops.language._library.matrix_recipe import _whole_k_matrix
from vecops.language._library.sme_matmul import dot_into, prepare_a, prepare_b

pytestmark = pytest.mark.skipif(
  platform.machine().lower() not in {"aarch64", "arm64"} or get_vector_state().sme_bytes is None,
  reason="native prepared-A acceptance requires SME",
)


@T.func
def _epilogue(value, row, column):
  return value * 2 + 1


def _template(a_dtype, b_dtype, acc_dtype):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), acc_dtype),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), acc_dtype),
    Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), acc_dtype),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), acc_dtype, access="output"),
  ):
    # Both conversion DAGs must be packed, never replaced by raw layout reads.
    a = T.cast(T.load(A) * 2 + 1, a_dtype)
    b = T.cast(T.load(B) - 1, b_dtype)
    _whole_k_matrix((a, b), C, dot_into, prepare_a, prepare_b, 33,
                    acc=T.load(Prior), acc_dtype=acc_dtype, epilogue=_epilogue, raw_a=None, raw_b=None)

  return kernel


@pytest.fixture(scope="module", params=[("bfloat16", "bfloat16", "float32"), ("uint8", "int8", "int32")])
def full_k_kernel(request, tmp_path_factory):
  family = request.param
  root = tmp_path_factory.mktemp("sme-prepared-a-" + family[0])
  compiled = T.compile(_template(*family), compiler=make_compiler(
    jobs=2, cache_dir=root / "cache", build_dir=root / "build"))
  return compiled, family


def _arguments(family, m, n, k, reverse, alias):
  memory_dtype = np.dtype(family[2])
  rows = max(m, k) + 4 if alias == "B" else m + 4
  columns = 2 * max(n, k) + 7 if alias == "A" else 2 * n + 7
  # A/C share a root for alias A; B/C share one for alias B. Each source
  # gets fully initialized before the oracle snapshots the overlapped views.
  output_root = np.full((rows, columns), -111, dtype=memory_dtype)
  a_root = output_root if alias == "A" else np.full((m + 4, 2 * k + 7), -111, dtype=memory_dtype)
  b_root = output_root if alias == "B" else np.full((k + 4, 2 * n + 7), -111, dtype=memory_dtype)
  a = a_root[1:m + 1, 2:2 + 2 * k:2]
  b = b_root[1:k + 1, 2:2 + 2 * n:2]
  c = output_root[1:m + 1, 2:2 + 2 * n:2]
  if reverse:
    a, b, c = a[::-1, ::-1], b[::-1, ::-1], c[::-1, ::-1]
  a[:] = (np.arange(m * k).reshape(m, k) % 4) if family[0] == "uint8" else (np.arange(m * k).reshape(m, k) % 5 - 2)
  b[:] = np.arange(k * n).reshape(k, n) % 5 - 2
  prior = c[:1]
  prior[:] = np.arange(n) % 5
  # Small integers remain exactly representable after native narrowing,
  # products and every partial sum. This oracle uses no vecops packing/dot.
  left = a.copy().astype(np.int64) * 2 + 1
  right = b.copy().astype(np.int64) - 1
  expected = ((left @ right) + prior.copy().astype(np.int64)) * 2 + 1
  before = output_root.copy()
  mask = np.ones(output_root.shape, dtype=bool)
  mask[1:m + 1, 2:2 + 2 * n:2] = False
  return a, b, prior, c, expected.astype(memory_dtype), output_root, before, mask


def test_native_full_k_multiple_m_tiles_n_tail_conversion_alias_and_zero_k(full_k_kernel):
  compiled, family = full_k_kernel
  reduction = 4097 if family[0] == "bfloat16" else 65
  for m, n, k, reverse, alias in (
    (65, 71, reduction, False, None),
    (65, 71, reduction, True, None),
    (65, 71, reduction, True, "A"),
    (65, 71, reduction, True, "B"),
    (65, 71, 0, True, "A"),
    (65, 71, 0, True, "B"),
  ):
    a, b, prior, c, expected, root, before, mask = _arguments(family, m, n, k, reverse, alias)
    compiled.operator.native(a, b, prior, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, expected, err_msg=f"{family} K={k} reverse={reverse} alias={alias}")
    np.testing.assert_array_equal(root[mask], before[mask])
