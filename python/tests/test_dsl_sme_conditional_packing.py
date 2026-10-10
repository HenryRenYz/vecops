# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Inactive conditional caches may use ZA; active/history paths remain intact.

The Native library oracle checks strict conversion before the dot, an
independent prior snapshot, split-K and aliasing. ARM-only physical pack
checks reuse the independent existing SME pack oracle and runtime geometry.
No native packer or mathematical tolerance is substituted for the fallback.
"""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler


def library_template(dtype="bfloat16", *, captured=False):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  AS, AK, BKs, BS, CS, CN, DS, DN = (T.dynamic() for _ in range(8))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (AS, AK), "float32"),
    B: T.Tensor((K, N), (BKs, BS), "float32"),
    C: T.Tensor((M, N), (CS, CN), "float32", access="output"),
    D: T.Tensor((M, N), (DS, DN), "float32"),
  ):
    a = T.load(A)
    if captured:
      T.store(C, T.broadcast_to(T.expand_dims(T.reduce_add(a, axis=1), 1), (M, N)))
    left = T.cast(a * 0.5 + 0.125, dtype)
    right = T.cast(T.load(B), dtype)
    result = T.library.matmul((left, right), C, acc=T.load(D), BM=16, BN=4, BK=3, packing="AB", swap=False)
    T.store(C, T.max(result * 0.5, 0.0))

  return kernel


@pytest.mark.parametrize("fusion", ("auto", "never"))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_native_conditional_pack_library_preserves_conversion_prior_and_aliases(tmp_path, fusion, dtype):
  ml_dtypes = pytest.importorskip("ml_dtypes")
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(library_template(dtype), compiler=compiler, pass_options={"cpu_matmul_fusion": fusion})
  specs = []
  for m, n, k in ((3, 8, 5), (17, 80, 9), (0, 8, 5), (3, 8, 0)):
    specs.append(
      (
        vecops.TensorMeta((m, k), strides=(max(k, 1), 1), dtype="float32"),
        vecops.TensorMeta((k, n), strides=(1, max(k, 1)), dtype="float32"),
        vecops.TensorMeta((m, n), strides=(max(n, 1), 1), dtype="float32"),
        vecops.TensorMeta((m, n), strides=(max(n, 1), 1), dtype="float32"),
      )
    )
  with vecops.collect_compile_requests() as collector:
    for spec in specs:
      compiled.operator(*spec)
  assert len(collector.requests) == 4
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  native = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.float16
  rng = np.random.default_rng(6207)
  calls = 0
  for m, n, k in ((3, 8, 5), (17, 80, 9), (3, 8, 0)):
    for layout in ("dense", "reverse", "k_stride", "alias_input", "alias_prior"):
      size = max(m * max(k, 1), m * n, 1)
      storage = rng.integers(-8, 9, size * 2).astype(np.float32) / np.float32(8)
      a = np.ndarray((m, k), np.float32, buffer=storage, strides=(max(k, 1) * 4, 4))
      b_backing = rng.integers(-8, 9, max(n * max(k, 1), 1)).astype(np.float32) / np.float32(8)
      b = np.ndarray((k, n), np.float32, buffer=b_backing, strides=(4, max(k, 1) * 4))
      c = np.full((m, n), np.nan, np.float32)
      prior = rng.integers(-4, 5, (m, n)).astype(np.float32) / np.float32(16)
      if layout == "reverse":
        a, b, c, prior = a[::-1], b[:, ::-1], c[:, ::-1], prior[::-1, ::-1]
      elif layout == "k_stride":
        expanded = np.zeros((m, max(k, 1) * 2), np.float32)
        expanded[:, : k * 2 : 2] = a
        a = expanded[:, : k * 2 : 2]
      elif layout == "alias_input":
        c = np.ndarray((m, n), np.float32, buffer=storage, strides=(n * 4, 4))
      elif layout == "alias_prior":
        c[...] = prior
        prior = c
      before_a, before_b, before_prior = a.copy(), b.copy(), prior.copy()
      converted_a = (before_a * np.float32(0.5) + np.float32(0.125)).astype(native).astype(np.float32)
      converted_b = before_b.astype(native).astype(np.float32)
      expected = np.maximum((converted_a @ converted_b + before_prior) * np.float32(0.5), np.float32(0))
      compiled.operator.native(a, b, c, prior, **compiled._core.specialization_values)
      # Dyadic fixture values keep multiply/add and split-K exact; no
      # approximate math or relaxed reduction tolerance can hide an alias bug.
      np.testing.assert_array_equal(c, expected)
      calls += 1
  assert calls == 15


@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_native_history_survives_an_intervening_alias_write_before_packing(tmp_path, dtype):
  ml_dtypes = pytest.importorskip("ml_dtypes")
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(library_template(dtype, captured=True), compiler=compiler)
  with vecops.collect_compile_requests() as collector:
    compiled.operator(
      vecops.TensorMeta((3, 5), dtype="float32"),
      vecops.TensorMeta((5, 8), strides=(1, 5), dtype="float32"),
      vecops.TensorMeta((3, 8), dtype="float32"),
      vecops.TensorMeta((3, 8), dtype="float32"),
    )
  assert len(collector.requests) == 1
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  narrow = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.float16
  for m, n, k in ((3, 8, 5), (17, 80, 9)):
    for aliases in (False, True):
      storage = np.arange(max(m * k, m * n), dtype=np.float32) % 9 / np.float32(8)
      a = np.ndarray((m, k), np.float32, buffer=storage, strides=(k * 4, 4))
      b = (np.arange(n * k, dtype=np.float32).reshape(n, k) % 7 / np.float32(8)).T
      c = (
        np.ndarray((m, n), np.float32, buffer=storage, strides=(n * 4, 4)) if aliases else np.empty((m, n), np.float32)
      )
      prior = np.full((m, n), np.float32(0.25))
      original = a.copy()
      left = (original * np.float32(0.5) + np.float32(0.125)).astype(narrow).astype(np.float32)
      right = b.astype(narrow).astype(np.float32)
      expected = np.maximum((left @ right + prior.copy()) * np.float32(0.5), np.float32(0))
      compiled.operator.native(a, b, c, prior, **compiled._core.specialization_values)
      np.testing.assert_array_equal(c, expected)
