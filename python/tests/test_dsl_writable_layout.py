# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Small-rank layout proofs preserve the generic sorted-axis contract."""

from itertools import product

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.cpp.support import _PREAMBLE


def reference_layout_safe(shape, strides):
  if 0 in shape:
    return True
  covered = 1
  for stride, extent in sorted((abs(s), n) for s, n in zip(strides, shape) if n > 1):
    if stride < covered:
      return False
    covered += (extent - 1) * stride
  return True


def view_with_layout(shape, strides):
  low = sum(min(0, (n - 1) * s) for n, s in zip(shape, strides)) if all(shape) else 0
  high = sum(max(0, (n - 1) * s) for n, s in zip(shape, strides)) if all(shape) else 0
  storage = np.full(high - low + 1, -3, dtype=np.float32)
  return storage, np.ndarray(
    shape, dtype=np.float32, buffer=storage, offset=-low * 4, strides=tuple(s * 4 for s in strides)
  )


def layout_template(rank):
  shape = tuple(T.dynamic(lower=0) for _ in range(rank))
  strides = tuple(T.dynamic() for _ in range(rank))

  @T.template
  def kernel(C: T.Tensor(shape, strides, T.float32, access="output")):
    T.store(C, 7)

  return kernel


def test_two_axis_proof_uses_a_compare_swap_without_a_runtime_sort():
  start = _PREAMBLE.index("if constexpr (Rank == 2)")
  two_axis = _PREAMBLE[start : _PREAMBLE.index("else if constexpr (Rank > 2)", start)]
  assert "count == 2" in two_axis
  assert "axes[1].first < axes[0].first" in two_axis
  assert "std::swap(axes[0], axes[1])" in two_axis
  assert "std::sort(" not in two_axis
  assert "__builtin_mul_overflow(extent - 1, stride" in _PREAMBLE
  assert "__builtin_add_overflow(covered, delta" in _PREAMBLE


@pytest.fixture(scope="module")
def native_layout_kernels(tmp_path_factory):
  root = tmp_path_factory.mktemp("writable-layout")
  compiler = make_compiler(jobs=3, cache_dir=root / "cache", build_dir=root / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for rank in (1, 2, 3):
      compiled = T.compile(layout_template(rank), compiler=compiler)
      compiled.operator(vecops.TensorMeta((2,) * rank, dtype="float32"), **compiled._core.specialization_values)
      kernels[rank] = compiled
  vecops.compile_batch(collector.requests, parallelism=3, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("rank", (1, 2, 3))
def test_native_writable_layout_proof_matches_sorted_axes(native_layout_kernels, rank):
  # Empty/unit axes, both signs, broadcast strides, dense C/F permutations,
  # padded layouts, overlapping layouts and conservative interleaved cases.
  shape_cases = ((0,) + (2,) * (rank - 1), (1,) * rank, (2,) * rank)
  for shape, strides in product(shape_cases, product((-4, -1, 0, 1, 4), repeat=rank)):
    storage, output = view_with_layout(shape, strides)
    label = f"shape={shape}, strides={strides}"
    if reference_layout_safe(shape, strides):
      native_layout_kernels[rank](output)
      np.testing.assert_array_equal(output, 7, err_msg=label)
    else:
      with pytest.raises((RuntimeError, ValueError), match="overlap|writable"):
        native_layout_kernels[rank](output)
      np.testing.assert_array_equal(storage, -3, err_msg=label)
