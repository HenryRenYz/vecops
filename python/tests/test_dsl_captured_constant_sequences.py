"""Captured immutable integer tuples share the literal-parameter frontend path."""

from __future__ import annotations

import vecops.language as T
from vecops.language._compiler.frontend.parser import lower


def test_captured_transpose_axes_are_normalized_like_literal_axes():
  axes = (1, 0)

  @T.template
  def kernel(
    A: T.Tensor((2, 3), dtype="float32"),
    C: T.Tensor((3, 2), dtype="float32", access="output"),
  ):
    T.copy(T.transpose(A, axes=axes), C)

  result = lower(kernel.ir)
  assert result.body.statements


def test_captured_reshape_shape_uses_scalar_normalization_for_each_extent():
  shape = (3, 2)

  @T.template
  def kernel(
    A: T.Tensor((2, 3), dtype="float32"),
    C: T.Tensor((3, 2), dtype="float32", access="output"),
  ):
    T.copy(T.reshape(A, shape), C)

  result = lower(kernel.ir)
  assert result.body.statements
