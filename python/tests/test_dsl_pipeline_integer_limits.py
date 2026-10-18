# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Pipeline-generated arithmetic stays defined for legal extreme-bound loops."""

import os
import subprocess

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler

MIN = -(1 << 63)
MAX = (1 << 63) - 1
CASES = [
  (2, 2, False, MAX - 8, MAX - 1, 3),
  (-2, 2, False, MIN + 8, MIN + 1, 3),
  (1 << 61, 1, True, MIN + 7, MIN + 7 + 6 * (1 << 61), 5),
  (-(1 << 61), 1, True, MAX - 7, MAX - 7 - 6 * (1 << 61), 5),
  (MAX, 4, False, MIN + 7, MIN + 8, 0),
]


@pytest.fixture(scope="module")
def sanitized_compiler(tmp_path_factory):
  # GNU links its UBSan runtime into a shared kernel with ordinary sanitizer
  # link flags. Clang's shared-runtime discovery has a separate deployment
  # contract; run these probes with an explicitly selected GNU toolchain.
  cxx = os.environ.get("CXX", "g++")
  version = subprocess.check_output([cxx, "--version"], text=True)
  if "clang" in version.lower():
    pytest.skip("select GNU CXX to run shared-kernel UBSan probes")
  root = tmp_path_factory.mktemp("pipeline-ubsan")
  return make_compiler(target="Scalar", jobs=1,
      cflags=("-fsanitize=undefined", "-fno-sanitize-recover=undefined"),
      ldflags=("-fsanitize=undefined",), cache_dir=root / "cache", build_dir=root / "build")


@pytest.mark.parametrize("step,distance,buffering,start,stop,last", CASES)
def test_pipeline_generated_arithmetic_under_ubsan(sanitized_compiler, step, distance, buffering, start, stop, last):
  magnitude = abs(step)

  @T.template
  def kernel(A: T.Tensor((1,), dtype="float32"),
             C: T.Tensor((1,), dtype="float32", access="output"), lo: T.int64, hi: T.int64):
    for i in T.pipelined(lo, hi, step, distance=distance, buffering=buffering):
      if step > 0:
        travelled = T.bitcast(i, "uint64") - T.bitcast(lo, "uint64")
      else:
        travelled = T.bitcast(lo, "uint64") - T.bitcast(i, "uint64")
      ordinal = travelled // T.cast(magnitude, "uint64")
      T.store(C[:], T.load(A[:]) + T.cast(ordinal, "float32"))

  compiled = T.compile(kernel, compiler=sanitized_compiler)
  a, c = np.array([11], dtype=np.float32), np.array([-99], dtype=np.float32)
  compiled(a, c, start, stop)
  np.testing.assert_array_equal(c, [11 + last])
  if buffering:
    assert "two-bank loads guarded" in " ".join(compiled.kernel_ir.diagnostics)
  if magnitude * distance > (1 << 64) - 1:
    assert "lookahead stride is not representable" in " ".join(compiled.kernel_ir.diagnostics)
