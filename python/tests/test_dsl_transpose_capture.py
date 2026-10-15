# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Transpose preserves captured values across writes and isolated helpers."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler


@T.func
def transpose_value(value, output):
  T.store(output, T.transpose(value))


@pytest.mark.parametrize("helper", [False, True])
def test_native_transpose_retains_original_read_after_write(helper, tmp_path):
  m, n = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((m, n), dtype="float32", access="inout"),
             C: T.Tensor((n, m), dtype="float32", access="output")):
    old = T.load(A)
    T.store(A, T.zeros(A.shape, dtype=A.dtype))
    if helper:
      transpose_value(old, C)
    else:
      T.store(C, T.transpose(old))

  compiled = T.compile(kernel, compiler=make_compiler(
    jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  for shape in [(3, 7), (17, 19)]:
    for alias in [False, True]:
      a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
      expected = a.T.copy()
      c = a.reshape(shape[::-1]) if alias else np.empty(shape[::-1], dtype=np.float32)
      compiled.operator.native(a, c, **compiled._core.specialization_values)
      np.testing.assert_array_equal(c, expected)
      if not alias:
        np.testing.assert_array_equal(a, 0)


def test_native_matmul_shape_rebinding_retains_captured_read(tmp_path):
  import platform

  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  family = "sme" if platform.machine().lower() in {"aarch64", "arm64"} else "amx"
  m, q, n, k = (T.dynamic(lower=0) for _ in range(4))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="bfloat16"),
             D: T.Tensor((q, k), dtype="bfloat16"),
             B: T.Tensor((k, n), dtype="bfloat16", access="inout"),
             C: T.Tensor((m, n), dtype="float32", access="output")):
    a = T.load(A) + T.load(D)
    old_b = T.load(B)
    T.store(B, T.zeros(B.shape, dtype=B.dtype))
    T.dot((a, old_b), C, implementation=family)

  compiled = T.compile(kernel, compiler=make_compiler(
    jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  for rows, columns, reduction, broadcast_rows in [(16, 16, 32, 1), (17, 19, 33, 17)]:
    a = ((np.arange(rows * reduction).reshape(rows, reduction) % 13 - 6) / 8).astype(bf16)
    d = np.full((broadcast_rows, reduction), 0.25, dtype=bf16)
    b = ((np.arange(reduction * columns).reshape(reduction, columns) % 11 - 5) / 8).astype(bf16)
    c = np.empty((rows, columns), dtype=np.float32)
    expected = (a.astype(np.float32) + d.astype(np.float32)).astype(bf16).astype(np.float32) @ b.astype(np.float32)
    compiled.operator.native(a, d, b, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, expected)
    np.testing.assert_array_equal(b.astype(np.float32), 0)
