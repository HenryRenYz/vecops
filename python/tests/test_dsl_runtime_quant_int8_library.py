# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Explicit grouped runtime quantization preserves conversion and modular sums."""

import numpy as np
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._library.runtime_quant_int8 import dot_into, quantize


def template():
  m, n, k, capacity = (T.dynamic(lower=0) for _ in range(4))
  a0, a1 = T.dynamic(), T.dynamic()

  @T.template
  def kernel(A: T.Tensor((m, k), (a0, a1), "float32"), BP: T.Tensor((capacity,), dtype="int8"),
             R: T.Tensor((m, 1), dtype="float32"), Z: T.Tensor((1,), dtype="int32"),
             D: T.Tensor((1, n), dtype="int32"), C: T.Tensor((m, n), dtype="int32", access="output")):
    q = quantize(T.load(A), T.load(R), T.element_at(T.load(Z), (0,)))
    b = T.packed(BP, (k, n), side="B", format="sme")
    dot_into((q, b), C, packed_b=BP, acc=T.load(D), panel=32)

  return kernel


def test_runtime_quant_source_is_grouped_dot_without_quantized_matrix():
  source = CPUBackend().lower(template().ir, (), target="AVX512").source
  assert "vecops::vec::widening_dot" in source
  assert ".with_resources(" not in source
  assert "runtime quant packed B capacity too small" in source


def test_native_runtime_quant_tail_poison_nonfinite_modulo_and_alias(tmp_path):
  compiler = make_compiler(target="Native", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(template(), compiler=compiler)
  for m, n, k in [(2, 35, 7), (3, 5, 0), (1, 1, 1), (1, 17, 9)]:
    a = np.resize(np.asarray([-np.inf, -300.5, -0.5, 0.5, 12.9, 255.9, np.inf, np.nan], np.float32), m * k).reshape(m, k)
    scales = np.ones((m, 1), np.float32)
    zp = np.asarray([3], np.int32)
    weights = ((np.arange(k * n).reshape(k, n) % 11) - 5).astype(np.int8)
    groups, panels = (k + 3) // 4, (n + 31) // 32
    packed = np.full((panels, groups, 32, 4), 127, np.int8)
    for col in range(n):
      for kk in range(k):
        packed[col // 32, kk // 4, col % 32, kk % 4] = weights[kk, col]
    initial = np.full((1, n), 0x7FFFFFFF, np.int32)
    output = np.empty((m, n), np.int32)
    output[0] = initial[0]
    seed_alias = output[:1]
    with np.errstate(invalid="ignore"):
      affine = a * scales + zp.astype(np.float32)
      q = np.clip(np.where(np.isnan(affine), 0, affine), 0, 255).astype(np.uint8)
    expected = q.astype(np.int64) @ weights.astype(np.int64) + initial.astype(np.int64)
    expected = (expected & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    compiled.operator.native(a, packed.reshape(-1), scales, zp, seed_alias, output,
                             **compiled._core.specialization_values)
    np.testing.assert_array_equal(output, expected)
