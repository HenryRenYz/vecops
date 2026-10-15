# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Uniform word bitcasts retain packed-byte order in projected prefixes."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(source_dtype, output_dtype):
  n = T.dynamic(lower=0)

  @T.template
  def kernel(W: T.Tensor((1,), dtype=source_dtype), O: T.Tensor((n,), dtype=output_dtype, access="output"),
             offset: T.int64):
    word = T.element_at(T.load(W), (0,))
    words = T.full(T.Vector(source_dtype, N=32), word)
    values = T.bitcast(words, output_dtype)
    T.store(O, values[offset:offset + n])

  return kernel


@pytest.mark.parametrize("target", ["AVX512", "SME"])
def test_projected_uniform_word_uses_rotated_broadcast(target):
  source = CPUBackend().lower(template("uint32", "uint8").ir, (), target=target).source
  assert "std::rotr" in source
  assert "vecops::vec::bitcast" in source


@pytest.mark.parametrize("source_dtype,output_dtype", [("uint32", "uint8"), ("uint64", "uint16"), ("float32", "uint8")])
def test_native_projected_word_bitcast(source_dtype, output_dtype, tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(template(source_dtype, output_dtype), compiler=compiler)
  width = np.dtype(source_dtype).itemsize
  word = np.asarray([0xA1B2C3D4 if width == 4 else 0xA1B2C3D4E5F60718], dtype=f"uint{width * 8}").view(source_dtype)
  expected = np.tile(word, 32).view(output_dtype)
  for offset in range(width // np.dtype(output_dtype).itemsize):
    for n in [0, 1, 3, 7, 17, 31]:
      storage = np.full(n + 4, 91, dtype=output_dtype)
      output = storage[2:-2]
      compiled.operator.native(word, output, offset, **compiled._core.specialization_values)
      np.testing.assert_array_equal(output, expected[offset:offset + n])
      np.testing.assert_array_equal(storage[[0, 1, -2, -1]], 91)
