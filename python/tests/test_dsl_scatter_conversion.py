# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Executable converting scatter regression coverage, including ordered writes."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("scatter-conversion")
  return make_compiler(target="Native", jobs=4, cache_dir=root / "cache", build_dir=root / "build")


@pytest.mark.parametrize("source", ["float32", "int32"])
@pytest.mark.parametrize("destination", ["int8", "uint8"])
@pytest.mark.parametrize("unique", [False, True])
@pytest.mark.parametrize("stride", [1, -2])
def test_scatter_default_conversion(compiler, source, destination, unique, stride):
  @T.jit(compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype=source), Idx: T.Tensor((37,), dtype="int64"),
               C: T.Tensor((41,), strides=(stride,), dtype=destination, access="inout")):
      indices = T.load(Idx)
      T.scatter(C, indices, T.load(A), mask=indices >= 0, unique=unique)
    return kernel

  values = [-1000, -129, -128, -127, -1, 0, 1, 126, 127, 128, 254, 255, 256, 1000]
  if source == "float32":
    values += [-1.9, 1.9, np.nan, -np.inf, np.inf]
  a = np.resize(np.asarray(values, dtype=source), 37)
  idx = np.arange(37, dtype=np.int64)
  if not unique:
    idx[1:10] = 2
    idx[35:] = 2  # duplicate across vector chunks, including the tail
  idx[10] = -1000  # masked invalid index must never be dereferenced
  bounds = np.iinfo(destination)
  converted = np.asarray([
    0 if np.isnan(v) else bounds.max if v >= bounds.max else bounds.min if v <= bounds.min else int(v)
    for v in a
  ], dtype=destination)
  backing = np.full(82, 17, dtype=destination)
  c = backing[:41] if stride == 1 else backing[::-2]
  expected = c.copy()
  for index, value in zip(idx, converted):
    if index >= 0:
      expected[index] = value
  kernel = factory()
  kernel(a, idx, c)
  np.testing.assert_array_equal(c, expected)
  if stride == -2:
    np.testing.assert_array_equal(backing[::2], 17)


def test_scatter_alias_snapshot(compiler):
  @T.jit(compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((37,), dtype="int16", access="inout"),
               Idx: T.Tensor((37,), dtype="int64")):
      indices = T.load(Idx)
      T.scatter(A, indices, T.load(A), mask=indices >= 0)
    return kernel
  idx = np.arange(36, -1, -1, dtype=np.int64)
  idx[1:8] = 2
  idx[9] = -99
  original = np.arange(37, dtype=np.int16)
  expected = original.copy()
  for index, value in zip(idx, original):
    if index >= 0:
      expected[index] = value
  kernel = factory()
  actual = original.copy()
  kernel(actual, idx)
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fast_math", [False, True])
def test_public_float_integer_conversion_scalar_oracle(tmp_path, fast_math):
  import subprocess
  from pathlib import Path
  from dsl_test_utils import cpp_compile_command

  root = Path(__file__).resolve().parents[2]
  executable = tmp_path / "float-integer-conversion"
  flags = ["-std=c++20", "-O2", "-march=native", "-Wno-ignored-attributes",
           "-I", str(root / "include"), "-I", str(root / "include/vecops")]
  import platform
  if platform.machine().lower() in {"aarch64", "arm64"}:
    from vecops._arm_vector_state import get_vector_state
    # The oracle exercises multiword FixedTags, which require a fixed SVE
    # vector length matching the executing thread.
    flags.append(f"-msve-vector-bits={get_vector_state().sve_bytes * 8}")
  if fast_math:
    flags.append("-ffast-math")
  command = cpp_compile_command(root / "tests/vec/FloatIntegerSaturatingConversionTest.cpp", executable, flags=flags)
  build = subprocess.run(command, capture_output=True, text=True)
  assert build.returncode == 0, build.stderr
  run = subprocess.run([str(executable)], capture_output=True, text=True)
  assert run.returncode == 0, run.stderr
