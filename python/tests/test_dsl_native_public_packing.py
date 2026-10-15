# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Public native pack ABI: exact padding, payloads, excess capacity and aliases."""

import os

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(dtype, side, format_):
  s, k, capacity = (T.dynamic(lower=0) for _ in range(3))
  ss, ks = T.dynamic(), T.dynamic()
  shape, strides = ((s, k), (ss, ks)) if side == "A" else ((k, s), (ks, ss))

  @T.template
  def kernel(X: T.Tensor(shape, strides, dtype),
             P: T.Tensor((capacity,), dtype=dtype, access="output")):
    T.pack(X, P, side=side, format=format_)

  return kernel


def historical_template(dtype, format_):
  capacity = T.dynamic(lower=0)

  @T.template
  def kernel(X: T.Tensor((19, 17), dtype=dtype, access="inout"),
             P: T.Tensor((capacity,), dtype=dtype, access="output")):
    old = T.load(X)
    T.store(X, T.full(X.shape, 7, dtype=dtype))
    T.pack(old, P, side="A", format=format_)

  return kernel


def test_sme_public_packer_retains_historical_load_before_source_write():
  source = CPUBackend().lower(historical_template("bfloat16", "sme").ir, (), target="SME").source
  assert "packed output buffer too small" in source
  assert "vecops::vec::sme::" in source


@pytest.mark.skipif(os.uname().machine.lower() not in {"aarch64", "arm64"}, reason="requires native SME packing")
def test_native_sme_public_packer_preserves_historical_load(tmp_path):
  pytest.importorskip("ml_dtypes")
  from vecops._arm_vector_state import get_vector_state
  span = get_vector_state().sme_bytes // 2
  dt = np.dtype("bfloat16")
  x = np.arange(19 * 17, dtype=np.float32).reshape(19, 17).astype(dt)
  old = x.copy()
  expected = np.zeros(((19 + span - 1) // span, 9, span, 2), dtype=dt)
  for s in range(19):
    for k in range(17):
      expected[s // span, k // 2, s % span, k % 2] = old[s, k]
  output = np.empty(expected.size, dtype=dt)
  compiler = make_compiler(target="Native", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(historical_template("bfloat16", "sme"), compiler=compiler)
  compiled.operator.native(x, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output.view(np.uint16), expected.reshape(-1).view(np.uint16))
  np.testing.assert_array_equal(x.astype(np.float32), 7)


def packed_dot_template(dtype):
  m, n, ca, cb = (T.dynamic(lower=0) for _ in range(4))
  acc_dtype = "float32" if dtype == "bfloat16" else "int32"

  @T.template
  def kernel(AP: T.Tensor((ca,), dtype=dtype), BP: T.Tensor((cb,), dtype=dtype),
             C: T.Tensor((m, n), dtype=acc_dtype, access="output"), K: T.int64):
    a = T.packed(AP, (m, K), side="A", format="amx")
    b = T.packed(BP, (K, n), side="B", format="amx")
    T.dot((a, b), C, implementation="amx")

  return kernel


@pytest.mark.parametrize("dtype", ["uint8", "bfloat16"])
@pytest.mark.skipif(os.uname().machine.lower() not in {"x86_64", "amd64"}, reason="requires native AMX packing")
def test_native_amx_terminal_scratch_ignores_poison(dtype, tmp_path):
  pytest.importorskip("ml_dtypes")
  dt = np.dtype(dtype)
  kr, group = 64 // dt.itemsize, 4 // dt.itemsize
  compiler = make_compiler(target="Native", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(packed_dot_template(dtype), compiler=compiler)
  for m, n, k in [(32, 32, kr + 1), (17, 19, kr - 1), (16, 16, 1)]:
    a = (np.arange(m * k).reshape(m, k) % 7).astype(dt)
    b = (np.arange(k * n).reshape(k, n) % 5).astype(dt)
    kp = (k + kr - 1) // kr * kr
    poison = np.nan if dtype == "bfloat16" else 255
    ap = np.full(((m + 15) // 16, kp // kr, 16, kr), poison, dtype=dt)
    bp = np.full(((n + 15) // 16, kp // kr, 16, 16 * group), poison, dtype=dt)
    for row in range(m):
      for kk in range(k):
        ap[row // 16, kk // kr, row % 16, kk % kr] = a[row, kk]
    for col in range(n):
      for kk in range(k):
        bp[col // 16, kk // kr, kk % kr // group, col % 16 * group + kk % group] = b[kk, col]
    saved_a, saved_b = ap.view(np.uint8).copy(), bp.view(np.uint8).copy()
    c = np.empty((m, n), dtype=np.float32 if dtype == "bfloat16" else np.int32)
    compiled.operator.native(ap.reshape(-1), bp.reshape(-1), c, k, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, a.astype(c.dtype) @ b.astype(c.dtype))
    np.testing.assert_array_equal(ap.view(np.uint8), saved_a)
    np.testing.assert_array_equal(bp.view(np.uint8), saved_b)


@pytest.mark.parametrize("side", ["A", "B"])
def test_amx_public_packing_uses_native_panel_schedule(side):
  source = CPUBackend().lower(template("uint8", side, "amx").ir, (), target="AVX512").source
  assert "pack_offset" not in source.partition("void __kernel__")[2]
  assert "packed output buffer too small" in source


@pytest.mark.parametrize("dtype", ["uint8", "bfloat16"])
@pytest.mark.parametrize("side", ["A", "B"])
@pytest.mark.skipif(os.uname().machine.lower() not in {"x86_64", "amd64"}, reason="requires native AMX packing")
def test_native_amx_public_packing_bits_and_aliases(dtype, side, tmp_path):
  pytest.importorskip("ml_dtypes")
  memory_dtype = np.dtype(dtype)
  itemsize = memory_dtype.itemsize
  bits = np.dtype(f"uint{itemsize * 8}")
  kr, group = 64 // itemsize, 4 // itemsize
  compiler = make_compiler(target="Native", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(template(dtype, side, "amx"), compiler=compiler)
  rng = np.random.default_rng(718)
  for s, k in [(0, 9), (17, 0), (1, 1), (17, kr + 1), (32, 2 * kr)]:
    count = ((s + 15) // 16) * ((k + kr - 1) // kr) * 16 * kr
    storage = rng.integers(0, 256, size=max(count, s * k) * itemsize + 16 * itemsize,
                           dtype=np.uint8).view(memory_dtype)
    logical = storage[:s * k].reshape(s, k)[::-1]
    expected = np.zeros((count,), dtype=bits)
    for si in range(s):
      for ki in range(k):
        base = ((si // 16) * ((k + kr - 1) // kr) + ki // kr) * 16 * kr
        offset = si % 16 * kr + ki % kr if side == "A" else ki % kr // group * 16 * group + si % 16 * group + ki % group
        expected[base + offset] = logical.view(bits)[si, ki]
    sentinel = storage.view(bits)[count:].copy()
    compiled.operator.native(logical if side == "A" else logical.T, storage,
                             **compiled._core.specialization_values)
    np.testing.assert_array_equal(storage.view(bits)[:count], expected)
    np.testing.assert_array_equal(storage.view(bits)[count:], sentinel)
