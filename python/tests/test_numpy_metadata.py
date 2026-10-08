# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""NumPy's empty contiguity flags must not discard explicit ABI layouts."""

import numpy as np
import pytest
import vecops


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64", "int8", "uint8"))
@pytest.mark.parametrize("shape", ((7, 0), (0, 7), (7, 1), (1, 7), (0, 0)))
@pytest.mark.parametrize("order", ("C", "F"))
def test_explicit_empty_and_singleton_numpy_strides_survive_native_binding(dtype, shape, order):
  if dtype == "bfloat16":
    pytest.importorskip("ml_dtypes")
  element = np.dtype(dtype)
  rows, columns = shape
  strides = (columns, 1) if order == "C" else (1, rows)
  array = np.ndarray(
    shape,
    dtype=element,
    buffer=np.empty(max(1, rows * columns), dtype=element),
    strides=tuple(stride * element.itemsize for stride in strides),
  )
  view = vecops._C.TensorView(array, writable=True)
  assert tuple(view.sizes) == shape
  assert tuple(view.strides) == strides
  assert view.array is array
  schema = vecops.KernelDef("numpy::explicit_layout", [vecops.TensorDef("B", shape, strides, dtype=dtype)])
  # Test the real ndarray adapter followed by strict schema validation. A
  # manually supplied TensorMeta would miss the bug in actual invocation.
  vecops._C.bind_kernel_call(schema._native(), vecops._C.KernelCall([view]))


@pytest.mark.parametrize("shape", ((0,), (7, 0), (0, 7), (0, 1, 5)))
def test_default_zero_stride_empty_numpy_arrays_keep_legacy_dense_layout(shape):
  array = np.empty(shape, dtype=np.float32)
  assert not any(array.strides)
  expected, stride = [], 1
  for extent in reversed(shape):
    expected.append(stride)
    stride *= extent
  assert tuple(vecops._C.TensorView(array).strides) == tuple(reversed(expected))


def test_explicit_empty_negative_numpy_layout_is_not_replaced_by_dense():
  array = np.ndarray((0, 7), dtype=np.float32, buffer=np.empty(1, np.float32), strides=(-16, -8))
  assert tuple(vecops._C.TensorView(array).strides) == (-4, -2)
