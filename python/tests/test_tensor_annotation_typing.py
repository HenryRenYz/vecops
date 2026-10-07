"""Tensor annotation hashes are stable while symbolic captures are bound."""

from __future__ import annotations

from typing import Optional, Union

import pytest
import vecops
import vecops.language as T
from vecops.language._compiler.backends.cpu.source import _build_definition
from vecops.language.types import normalize_tensor_annotation


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("strides", (False, True))
def test_optional_tensor_is_hashable_without_hashing_symbolic_dimensions(symbolic, strides):
  size = T.dynamic(lower=0) if symbolic else T.Const(2)
  dtype = T.DType("float32")
  tensor = T.Tensor((size, 4), (4, 1) if strides else None, dtype)
  original_hash = hash(tensor)
  # Exercise the strict set-based deduplication in older Python's typing.
  assert len({tensor, type(None)}) == 2
  optional = Optional[tensor]  # Deliberately exercise the pre-PEP604 typing API.
  assert Optional[tensor] is optional
  size.bind_name("Rows")
  dtype.bind_name("StorageDType")
  assert hash(tensor) == original_hash
  assert Optional[tensor] is optional
  marker = normalize_tensor_annotation(optional)
  assert marker.optional and marker.shape[0] is size and marker.dtype is dtype
  assert marker.strides is tensor.strides
  assert marker.concrete_strides[-1].value == 1


def test_equal_tensor_descriptors_have_equal_stable_hashes():
  a = T.Tensor((2, 4), (4, 1), "float32")
  b = T.Tensor((2, 4), (4, 1), "float32")
  assert a == b and hash(a) == hash(b)
  assert len({a, b}) == 1
  assert normalize_tensor_annotation(Union[a, b, None]).optional
  a.shape[0].bind_name("BoundRows")
  assert a == b and hash(a) == hash(b)


def test_template_optional_typing_preserves_shared_dimensions_and_dtype():
  size, dtype = T.dynamic(lower=0), T.DType("float32")
  tensor = T.Tensor((size,), dtype=dtype)

  @T.template
  def kernel(input: Optional[tensor] = None):  # noqa: UP045
    pass

  annotation = kernel.ir.annotations["input"]
  assert annotation.optional and annotation.shape[0] is size and annotation.dtype is dtype
  definition, _, _ = _build_definition(kernel.ir, ())
  assert definition.parameters[0].optional
  assert definition.parameters[0].access is vecops.TensorAccess.input
