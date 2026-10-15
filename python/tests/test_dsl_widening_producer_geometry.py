# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Grouped contractions reserve register width for converted producers."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(dtype):
  k = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((k,), dtype=dtype),
    B: T.Tensor((k,), dtype="int8"),
    Scale: T.Tensor((1,), dtype=dtype),
    C: T.Tensor(((k + 3) // 4,), dtype="int32", access="output"),
  ):
    converted = T.cast(T.load(A) * T.load(Scale) + 3, "uint8")
    T.store(C, T.widening_dot(converted, T.load(B), dtype="int32"))

  return kernel


@pytest.mark.parametrize("dtype,headroom", [("float32", 2), ("float64", 3)])
def test_sve_grouped_producer_accounts_for_input_lane_expansion(dtype, headroom):
  artifact = CPUBackend().lower(template(dtype).ir, (), target="SVE")
  regions = artifact.lowered.regions
  relevant = [r for r in regions if any(op.expression.op == "widening_dot" for op in r.operations)]
  assert relevant and all(r.rebind_headroom == headroom for r in relevant)
  if dtype == "float64":
    assert "ScalableTag<std::int32_t, -1>" in artifact.source
    assert "dot_fixed_input" in artifact.source


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_grouped_converted_producer_full_and_partial(dtype, tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(template(dtype), compiler=compiler)
  for k in [0, 1, 3, 4, 17, 65, 1025]:
    a = (np.arange(k) % 173 - 35).astype(dtype) * 0.25
    b = (np.arange(k) % 13 - 6).astype(np.int8)
    scale = np.asarray([1.25], dtype=dtype)
    output = np.empty((k + 3) // 4, np.int32)
    converted = np.clip(a * scale + 3, 0, 255).astype(np.uint8)
    values = converted.astype(np.int64) * b.astype(np.int64)
    expected = np.pad(values, (0, (-k) % 4)).reshape(-1, 4).sum(axis=1).astype(np.int32)
    compiled.operator.native(a, b, scale, output, **compiled._core.specialization_values)
    np.testing.assert_array_equal(output, expected)
