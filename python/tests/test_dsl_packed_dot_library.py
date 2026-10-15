# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Explicit packed MMLA library leaf, independent of automatic recipe policy."""

import os

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._library.packed_dot import dot_into


def template(dtype="bfloat16", m=2, n=8):
  ca, cb = T.dynamic(lower=0), T.dynamic(lower=0)
  accumulator = "float32" if dtype == "bfloat16" else "int32"

  @T.template
  def kernel(AP: T.Tensor((ca,), dtype=dtype), BP: T.Tensor((cb,), dtype=dtype),
             D: T.Tensor((1, n), dtype=accumulator),
             C: T.Tensor((m, n), dtype=accumulator, access="output"), K: T.int64):
    a = T.packed(AP, (m, K), side="A", format="sme")
    b = T.packed(BP, (K, n), side="B", format="sme")
    dot_into((a, b), C, packed_a=AP, packed_b=BP, acc=T.load(D), acc_dtype=accumulator)

  return kernel


@pytest.mark.parametrize("dtype", ["bfloat16", "int8", "uint8"])
@pytest.mark.parametrize("shape", [(2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (8, 2)])
def test_packed_leaf_has_mmla_and_final_group_mask(dtype, shape):
  source = CPUBackend().lower(template(dtype, *shape).ir, (), target="SME",
                              options={"_vector_state": VectorState(64, 64)}).source
  assert "vecops::vec::sve::mmla" in source
  assert "vecops::vec::sve::join_k_groups" in source
  assert "vecops::vec::iota(" in source
  assert "vecops::vec::shr(" in source
  assert "vecops::vec::bit_and(" in source
  assert "vecops::vec::div(" not in source
  assert "vecops::vec::mod(" not in source
  assert "packed MMLA requires matching ordinary VL and streaming SVL" in source
  assert ".with_resources(" not in source.partition("void __kernel__")[2]


@pytest.mark.skipif(os.uname().machine.lower() not in {"aarch64", "arm64"}, reason="requires native SVE MMLA and SME packing metadata")
@pytest.mark.parametrize("dtype", ["bfloat16", "int8", "uint8"])
def test_native_packed_leaf_odd_groups_poison_seed_and_output(dtype, tmp_path):
  pytest.importorskip("ml_dtypes")
  state = get_vector_state()
  if state.sve_bytes != state.sme_bytes or (state.sve_bytes or 0) < 64:
    pytest.skip("test catalog requires matching VL/SVL >=64 bytes")
  dt = np.dtype(dtype)
  span, group = 2 * state.sme_bytes // 4, 4 // dt.itemsize
  compiler = make_compiler(target="Native", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  for m, n in [(2, 2), (2, 8), (4, 4), (8, 2)]:
    compiled = T.compile(template(dtype, m, n), compiler=compiler)
    for k in [0, 1, group + 1, 3 * group, 5 * group - 1]:
      a = (np.arange(m * k).reshape(m, k) % 7 - 3).astype(dt)
      b = (np.arange(k * n).reshape(k, n) % 5 - 2).astype(dt)
      poison = np.nan if dtype == "bfloat16" else 127
      ap = np.full(((k + group - 1) // group, span, group), poison, dtype=dt)
      bp = np.full_like(ap, poison)
      for row in range(m):
        for kk in range(k):
          ap[kk // group, row, kk % group] = a[row, kk]
      for col in range(n):
        for kk in range(k):
          bp[kk // group, col, kk % group] = b[kk, col]
      compute = np.float32 if dtype == "bfloat16" else np.int32
      seed = (np.arange(n, dtype=compute).reshape(1, n) if dtype == "bfloat16" else
              np.full((1, n), 0x7FFFFFFF, dtype=np.int32))
      result = np.empty((m, n), dtype=compute)
      compiled.operator.native(ap.reshape(-1), bp.reshape(-1), seed, result, k,
                               **compiled._core.specialization_values)
      if dtype == "bfloat16":
        np.testing.assert_allclose(result, a.astype(compute) @ b.astype(compute) + seed, rtol=1e-5, atol=1e-5)
      else:
        expected = a.astype(np.int64) @ b.astype(np.int64) + seed.astype(np.int64)
        np.testing.assert_array_equal(result, (expected & 0xFFFFFFFF).astype(np.uint32).view(np.int32))


def test_automatic_recipe_is_guarded_and_keeps_fallback():
  ca, cb = T.dynamic(lower=0), T.dynamic(lower=0)
  m, n = (T.dynamic(lower=0) for _ in range(2))

  @T.template
  def kernel(AP: T.Tensor((ca,), dtype="bfloat16"), BP: T.Tensor((cb,), dtype="bfloat16"),
             C: T.Tensor((m, n), dtype="float32", access="output"), K: T.int64):
    a = T.packed(AP, (m, K), side="A", format="sme")
    b = T.packed(BP, (K, n), side="B", format="sme")
    T.dot((a, b), C)

  options = {"_vector_state": VectorState(64, 64), "cpu_packed_dot": True}
  source = CPUBackend().lower(kernel.ir, (), target="SME", options=options).source
  assert "defined(__ARM_FEATURE_SVE_BF16)" in source
  assert "vecops::vec::sve::mmla" in source
  assert "vecops_dsl::private_call" in source.partition("void __kernel__")[2]
  assert "1025" in source
  options["_vector_state"] = VectorState(32, 64)
  source = CPUBackend().lower(kernel.ir, (), target="SME", options=options).source
  assert "vecops::vec::sve::mmla" not in source


@pytest.mark.skipif(os.uname().machine.lower() not in {"aarch64", "arm64"}, reason="requires native packed-dot dispatch")
def test_native_automatic_packed_recipe_keeps_k_tail_and_fallback(tmp_path):
  pytest.importorskip("ml_dtypes")
  ca, cb = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(AP: T.Tensor((ca,), dtype="bfloat16"), BP: T.Tensor((cb,), dtype="bfloat16"),
             C: T.Tensor((2, 2), dtype="float32", access="output"), K: T.int64):
    a = T.packed(AP, (2, K), side="A", format="sme")
    b = T.packed(BP, (K, 2), side="B", format="sme")
    T.dot((a, b), C)

  span = get_vector_state().sme_bytes // 2
  compiler = make_compiler(target="Native", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_packed_dot": True})
  # K=1026 exercises the retained native fallback above the PackedDot limit.
  for k in (0, 1, 65, 1026):
    a = (np.arange(2 * k).reshape(2, k) % 7 - 3).astype("bfloat16")
    b = (np.arange(k * 2).reshape(k, 2) % 5 - 2).astype("bfloat16")
    ap = np.zeros(((k + 1) // 2, span, 2), dtype="bfloat16")
    bp = np.zeros_like(ap)
    for kk in range(k):
      ap[kk // 2, :2, kk % 2] = a[:, kk]
      bp[kk // 2, :2, kk % 2] = b[kk, :]
    if k % 2:
      ap[-1, :, 1] = np.nan
      bp[-1, :, 1] = np.nan
    result = np.empty((2, 2), dtype=np.float32)
    compiled.operator.native(ap.reshape(-1), bp.reshape(-1), result, k,
                             **compiled._core.specialization_values)
    np.testing.assert_allclose(result, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-5, atol=1e-5)
