# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fresh online packed captures borrow panels; historical captures stay values."""

import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(family, historical=False, tiled=False, shape=None):
  m, n, k = shape if shape is not None else (T.dynamic(lower=0) for _ in range(3))
  ca, cb = (T.dynamic(lower=0) for _ in range(2))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="bfloat16"),
             B: T.Tensor((k, n), dtype="bfloat16", access="inout"),
             AP: T.Tensor((ca,), dtype="bfloat16", access="inout"),
             BP: T.Tensor((cb,), dtype="bfloat16", access="inout"),
             C: T.Tensor((m, n), dtype="float32", access="output")):
    T.pack(A, AP, side="A", format=family)
    T.pack(B, BP, side="B", format=family)
    a = T.packed(AP, (m, k), side="A", format=family)
    b = T.load(B) if historical else T.packed(BP, (k, n), side="B", format=family)
    if historical:
      T.store(B, T.zeros(B.shape, dtype=B.dtype))
    if tiled:
      T.library.matmul((a, b), C, implementation=family)
    else:
      T.dot((a, b), C, implementation=family)

  return kernel


@pytest.mark.parametrize("family", ["amx", "sme"])
@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
def test_online_physical_binding_obeys_capture_order(family, historical, tiled):
  source = CPUBackend().lower(template(family, historical, tiled).ir, (),
                              target="AVX512" if family == "amx" else "SME",
                              options={"_vector_state": VectorState(64, 64)}).source
  assert ("dsl_recipe_dot_1_packed_a_capacity" in source) is not historical
  assert ("dsl_recipe_dot_1_packed_b_capacity" in source) is not historical


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
def test_native_online_packed_values_survive_later_writes(historical, tiled, tmp_path):
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  family = "sme" if platform.machine().lower() in {"aarch64", "arm64"} else "amx"
  compiled = T.compile(template(family, historical, tiled), compiler=make_compiler(
      jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  for m, n, k in [(16, 16, 32), (17, 19, 33), (1, 64, 65), (64, 16, 0)]:
    a = ((np.arange(m * k).reshape(m, k) % 13 - 6) / 8).astype(bf16)
    b = ((np.arange(k * n).reshape(k, n) % 11 - 5) / 8).astype(bf16)
    # Excess capacity is legal and must not affect physical pitch or results.
    ap, bp = np.empty(m * k + 16384, dtype=bf16), np.empty(n * k + 16384, dtype=bf16)
    c = np.empty((m, n), dtype=np.float32)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    compiled.operator.native(a, b, ap, bp, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, expected)
    if historical:
      np.testing.assert_array_equal(b.astype(np.float32), 0)


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("shape", [(16, 16, 32), (17, 19, 33)])
def test_native_static_historical_input_remains_a_value(tiled, shape, tmp_path):
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  family = "sme" if platform.machine().lower() in {"aarch64", "arm64"} else "amx"
  compiled = T.compile(template(family, historical=True, tiled=tiled, shape=shape), compiler=make_compiler(
      jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  m, n, k = shape
  a = ((np.arange(m * k).reshape(m, k) % 13 - 6) / 8).astype(bf16)
  b = ((np.arange(k * n).reshape(k, n) % 11 - 5) / 8).astype(bf16)
  ap, bp = np.empty(m * k + 16384, dtype=bf16), np.empty(n * k + 16384, dtype=bf16)
  c = np.empty((m, n), dtype=np.float32)
  expected = a.astype(np.float32) @ b.astype(np.float32)
  compiled.operator.native(a, b, ap, bp, c, **compiled._core.specialization_values)
  np.testing.assert_array_equal(c, expected)
  np.testing.assert_array_equal(b.astype(np.float32), 0)
