# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Integer facts preserve the existing generated C++ scalar promotions."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler


def test_native_unsigned_comparison_retains_cpp_usual_conversions(tmp_path):
  @T.template
  def program(u: T.uint64, C: T.Tensor((2,), dtype="int64", access="output")):
    if u >= -1:
      T.store(C[0], 1)
    else:
      T.store(C[0], 0)
    if u <= -1:
      T.store(C[1], 1)
    else:
      T.store(C[1], 0)

  compiled = T.compile(program, compiler=make_compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  maximum = (1 << 64) - 1
  for value in (0, 1, (1 << 63) - 1, 1 << 63, maximum):
    output = np.full(2, -17, dtype=np.int64)
    compiled(value, output)
    # Both comparisons use uint64 after C++ converts the signed -1.
    np.testing.assert_array_equal(output, [int(value == maximum), 1])


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "uint32"])
def test_native_auto_local_keeps_wider_cpp_addition(tmp_path, dtype):
  scalar = getattr(T, dtype)
  limits = np.iinfo(dtype)
  upper = int(limits.max)

  @T.template
  def program(x: scalar, C: T.Tensor((1,), dtype="int64", access="output")):
    y = x + 1
    if y <= upper:
      T.store(C[0], 1)
    else:
      T.store(C[0], 0)

  compiled = T.compile(program, compiler=make_compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  for value in (int(limits.min), upper - 1, upper):
    output = np.full(1, -17, dtype=np.int64)
    compiled(value, output)
    # The weak integer literal emits int64; auto y retains that wider type.
    np.testing.assert_array_equal(output, [int(value + 1 <= upper)])


def test_native_unsigned_auto_local_keeps_defined_cpp_wraparound(tmp_path):
  @T.template
  def program(x: T.uint64, C: T.Tensor((1,), dtype="int64", access="output")):
    y = x + 1
    if y <= 0:
      T.store(C[0], 1)
    else:
      T.store(C[0], 0)

  compiled = T.compile(program, compiler=make_compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  maximum = (1 << 64) - 1
  for value in (0, 1, maximum - 1, maximum):
    output = np.full(1, -17, dtype=np.int64)
    compiled(value, output)
    np.testing.assert_array_equal(output, [int(value == maximum)])
