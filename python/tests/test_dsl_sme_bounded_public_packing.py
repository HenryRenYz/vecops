# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Opt-in cache-block conversion staging preserves the public SME ABI."""

import os

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def converted_template():
  m, k, capacity = (T.dynamic(lower=0) for _ in range(3))
  sx, sy = T.dynamic(), T.dynamic()

  @T.template
  def kernel(X: T.Tensor((m, k), (sx, sy), dtype="float32"),
             P: T.Tensor((capacity,), dtype="bfloat16", access="output")):
    T.pack(T.cast(T.load(X) * 1.25 + 0.5, "bfloat16"), P, side="A", format="sme")

  return kernel


def test_bounded_converted_packing_is_opt_in_with_alias_capture_fallback():
  template = converted_template()
  default = CPUBackend().lower(template.ir, (), target="SME").source
  bounded = CPUBackend().lower(template.ir, (), target="SME", options={"cpu_pack_bounded": True}).source
  assert "65536" not in default
  assert "65536" in bounded
  assert "vecops_dsl::overlap(" in bounded
  assert "vecops::vec::sme::" in bounded
  with pytest.raises(TypeError, match="cpu_pack_bounded"):
    CPUBackend().lower(template.ir, (), target="SME", options={"cpu_pack_bounded": 1})


@pytest.mark.skipif(os.uname().machine.lower() not in {"aarch64", "arm64"}, reason="requires native SME")
def test_native_bounded_conversion_group_padding_and_alias(tmp_path):
  pytest.importorskip("ml_dtypes")
  from vecops._arm_vector_state import get_vector_state
  span = get_vector_state().sme_bytes // 2
  compiled = T.compile(converted_template(), compiler=make_compiler(target="Native", jobs=2,
    cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"), pass_options={"cpu_pack_bounded": True})
  for m, k in ((0, 1), (1, 0), (1, 1), (span + 1, 1031), (2 * span + 1, 65)):
    original = np.arange(m * k, dtype=np.float32).reshape(m, k) / 13
    logical = (original * np.float32(1.25) + np.float32(0.5)).astype("bfloat16")
    expected = np.zeros(((m + span - 1) // span, (k + 1) // 2, span, 2), dtype="bfloat16")
    for row in range(m):
      for col in range(k):
        expected[row // span, col // 2, row % span, col % 2] = logical[row, col]
    for alias in (False, True):
      size = max(original.nbytes, expected.nbytes) + 32
      memory = np.full(size, 19, dtype=np.uint8)
      x = np.ndarray((m, k), dtype=np.float32, buffer=memory)
      x[:] = original
      output_memory = memory if alias else np.full(expected.nbytes + 32, 19, dtype=np.uint8)
      output = np.ndarray((expected.size,), dtype="bfloat16", buffer=output_memory)
      untouched = output_memory[expected.nbytes:].copy()
      compiled.operator.native(x, output, **compiled._core.specialization_values)
      np.testing.assert_array_equal(output.view(np.uint16), expected.reshape(-1).view(np.uint16))
      np.testing.assert_array_equal(output_memory[expected.nbytes:], untouched)
