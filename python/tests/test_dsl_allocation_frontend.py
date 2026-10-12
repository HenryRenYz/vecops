# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Owned storage intrinsics consume parsed shapes exactly once."""

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import lower


@pytest.mark.parametrize("operation", ["alloc", "shared", "double_buffer"])
@pytest.mark.parametrize("kind", ["scalar", "literal", "captured", "helper"])
def test_owned_allocation_shapes_share_one_parsing_path(operation, kind, capsys):
  allocate = getattr(T, operation)
  captured = (19,)

  @T.func
  def helper():
    T.static_print("shape-helper-expanded")
    return (19,)

  @T.template
  def kernel(C: T.Tensor((19,), dtype="float32", access="output")):
    if kind == "scalar":
      banks = allocate(19, dtype="float32")
    elif kind == "literal":
      banks = allocate((19,), dtype="float32")
    elif kind == "captured":
      banks = allocate(captured, dtype="float32")
    else:
      banks = allocate(helper(), dtype="float32")
    T.store(banks, T.full(banks.shape, 1))

  result = lower(kernel.ir)
  allocations = [statement for statement in result.body.statements if isinstance(statement, I.Allocate)]
  assert len(allocations) == 1
  shape = tuple(I.constant(extent) for extent in allocations[0].view.shape)
  assert shape == ((2, 19) if operation == "double_buffer" else (19,))
  output = capsys.readouterr().out
  assert output.count("shape-helper-expanded") == (1 if kind == "helper" else 0)
  assert all(location[1] > 0 and location[2] >= 0 for location in result.origins.values())


@pytest.mark.parametrize("operation", ["alloc", "shared", "double_buffer"])
def test_owned_allocation_duplicate_dtype_is_diagnosed_at_original_call(operation):
  allocate = getattr(T, operation)

  @T.template
  def kernel(C: T.Tensor((19,), dtype="float32", access="output")):
    banks = allocate((19,), "float32", dtype="float32")
    T.store(banks, T.full(banks.shape, 1))

  with pytest.raises(TypeError, match="allocation expects a shape and optional dtype"):
    lower(kernel.ir)


@pytest.mark.parametrize("shape,prefix", [((8,), (3,)), ((4, 8), (2, 3))])
def test_owned_zero_offset_subview_preserves_surrounding_initializer(shape, prefix):
  import numpy as np
  from vecops.language._debug.interpreter import Interpreter

  @T.template
  def kernel(C: T.Tensor(shape, dtype="int32", access="output")):
    storage = T.alloc(shape, dtype="int32")
    T.store(storage, T.full(shape, 7, dtype="int32"))
    if len(shape) == 1:
      T.store(storage[:prefix[0]], T.full(prefix, 11, dtype="int32"))
    else:
      T.store(storage[:prefix[0], :prefix[1]], T.full(prefix, 11, dtype="int32"))
    T.store(C, T.load(storage))

  ir = lower(kernel.ir)
  out = np.empty(shape, np.int32)
  Interpreter(ir, {"C": out}).block(ir.body)
  expected = np.full(shape, 7, np.int32)
  expected[tuple(slice(0, extent) for extent in prefix)] = 11
  np.testing.assert_array_equal(out, expected)
