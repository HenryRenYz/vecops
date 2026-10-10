# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Selected SME owners keep producer addressing in callback-owned SSA.

Source/cross tests do not claim ARM speed. The Native fixture also runs on
x86 to protect the unchanged DAG's strict arithmetic and historical aliases;
the same fixture on ARM exercises the selected streaming owners.
"""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from test_dsl_sme_exact_blocks import _body
from test_dsl_sme_family_owners import _OWNER


def _owners(source):
  return [_body(source, match[0]) for match in re.finditer(_OWNER, source)]


def _k_bodies(owner):
  return [
    _body(owner[match.start() :], match[0])
    for match in re.finditer(r"/\* (?:full K groups: no per-element tail tests|one partial K group) \*/", owner)
  ]


def _native_template(dtype):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  A0, A1, B0, B1, C0, C1 = (T.dynamic() for _ in range(6))
  acc = "float64" if dtype == "float64" else "float32"

  @T.template
  def kernel(
    A: T.Tensor((M, K), (A0, A1), dtype),
    B: T.Tensor((K, N), (B0, B1), dtype),
    C: T.Tensor((M, N), (C0, C1), acc, access="inout"),
  ):
    captured = T.load(A)
    # A=C aliases exercise historical read protection before owner entry;
    # the following matmul must not reload this captured read after the write.
    T.store(C, T.cast(captured, acc))
    a = T.cast(T.cast(captured, acc) * 0.5 + 0.125, dtype)
    value = T.dot((a, T.load(B)), C, acc=T.load(C))
    T.store(C, T.max(value * 0.5 + 0.125, 0))

  return kernel


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_native_owner_loader_metadata_strict_and_historical_alias(dtype, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(
    _native_template(dtype),
    compiler=compiler,
    pass_options={"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"},
  )
  with vecops.collect_compile_requests() as collector:
    kernel.operator(
      vecops.TensorMeta((3, 17), strides=(37, 2), dtype=dtype),
      vecops.TensorMeta((17, 17), strides=(37, 2), dtype=dtype),
      vecops.TensorMeta((3, 17), strides=(37, 2), dtype=dtype),
      **kernel._core.specialization_values,
    )
  assert len(collector.requests) == 1
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  rng = np.random.default_rng(531)
  for m, n in ((3, 17), (64, 1), (1, 64), (19, 23)):
    for negative in (False, True):
      for alias in (False, True):
        a = rng.integers(-4, 5, size=(m, 2 * n + 3)).astype(dtype)[:, : 2 * n : 2]
        b = rng.integers(-4, 5, size=(n, 2 * n + 3)).astype(dtype)[:, : 2 * n : 2]
        a[:] /= np.dtype(dtype).type(16)
        b[:] /= np.dtype(dtype).type(16)
        if negative:
          a, b = a[::-1, ::-1], b[::-1, ::-1]
        c = a if alias else np.empty_like(a)
        original = a.copy()
        transformed = (original * np.dtype(dtype).type(0.5) + np.dtype(dtype).type(0.125)).astype(dtype)
        expected = np.maximum((transformed @ b + original) * np.dtype(dtype).type(0.5) + np.dtype(dtype).type(0.125), 0)
        kernel.operator.native(a, b, c, **kernel._core.specialization_values)
        np.testing.assert_array_equal(c, expected, err_msg=f"{dtype} {(m, n)} negative={negative} alias={alias}")
