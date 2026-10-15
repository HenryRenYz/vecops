# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fused bounded B packing: byte-exact ABI and modulo-32 correction."""

import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(family, memory="float32", native="int8"):
  k, n, capacity = (T.dynamic(lower=0) for _ in range(3))
  ks, ns = T.dynamic(), T.dynamic()

  @T.template
  def kernel(B: T.Tensor((k, n), (ks, ns), dtype=memory),
             P: T.Tensor((capacity,), dtype=native, access="output"),
             D: T.Tensor((n,), dtype="int32", access="output"), Z: T.int32):
    value = T.cast(T.load(B) * 4, native) if memory == "float32" else T.load(B)
    T.library.pack_b_compensated(value, P, D, zero_point=Z, format=family)

  return kernel


@pytest.mark.parametrize("family", ["amx", "sme"])
def test_compensated_packer_is_bounded_source_recipe(family):
  result = CPUBackend().lower(template(family).ir, (), target="AVX512" if family == "amx" else "SME",
                              options={"_vector_state": VectorState(64, 64)})
  assert "compensation sidecar length mismatch" in result.source
  assert "packed output and compensation may not overlap" in result.source
  assert "packed output buffer too small" in result.source


def oracle(value, family, span):
  k, n = value.shape
  quantum = 64 if family == "amx" else 4
  kp = (k + quantum - 1) // quantum * quantum
  out = np.zeros(((n + span - 1) // span) * kp * span, dtype=value.dtype)
  for col in range(n):
    for row in range(k):
      if family == "amx":
        offset = (col // 16 * (kp // 64) + row // 64) * 1024 + row % 64 // 4 * 64 + col % 16 * 4 + row % 4
      else:
        offset = (col // span * (kp // 4) + row // 4) * span * 4 + col % span * 4 + row % 4
      out[offset] = value[row, col]
  return out


@pytest.mark.parametrize("memory,native", [("float32", "int8"), ("int8", "int8"), ("uint8", "uint8")])
def test_native_compensated_pack_bits_sums_tails_and_aliases(memory, native, tmp_path):
  family = "sme" if platform.machine().lower() in {"aarch64", "arm64"} else "amx"
  span = 2 * (get_vector_state().sme_bytes or 64) // 4 if family == "sme" else 16
  compiled = T.compile(template(family, memory, native), compiler=make_compiler(
      jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  for k, n in [(0, 17), (1, 1), (63, 15), (65, 17), (255, 32), (257, 33), (1024, 64)]:
    source = np.empty((n, k), dtype=memory)
    source[...] = (np.arange(n * k).reshape(n, k) % 101 - (0 if native == "uint8" else 50))
    source = source.T[::-1]
    converted = np.clip(np.trunc(source * 4), -128, 127).astype(native) if memory == "float32" else source.copy()
    expected = oracle(converted, family, span)
    for alias in ([False, True] if memory != "float32" and k and n else [False]):
      packed = np.full(expected.size + 16, 91, dtype=native)
      if alias:
        packed[:k * n] = source.reshape(-1)
        actual_source = packed[:k * n].reshape(k, n)
      else:
        actual_source = source
      d = np.empty(n, dtype=np.int32)
      zero_point = -17 if alias or k == 255 else 2147483647
      compiled.operator.native(actual_source, packed, d, zero_point, **compiled._core.specialization_values)
      np.testing.assert_array_equal(packed[:expected.size], expected)
      np.testing.assert_array_equal(packed[expected.size:], 91)
      sums = converted.astype(np.int64).sum(axis=0)
      expected_d = ((-sums * zero_point) & 0xffffffff).astype(np.uint32).view(np.int32)
      np.testing.assert_array_equal(d, expected_d)
