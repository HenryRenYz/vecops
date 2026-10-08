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
