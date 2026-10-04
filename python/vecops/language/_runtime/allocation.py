"""Strided output storage allocation, including overlap and signed spans."""

from __future__ import annotations

from typing import Any

from ..._dtype import framework_dtype
from ..._schema import TensorAccess, TensorMeta
from ..types import TensorType
from .binding import _evaluate, _metadata


def _storage_plan(shape: tuple[int, ...], strides: tuple[int, ...]) -> tuple[int, int]:
  if len(shape) != len(strides):
    raise ValueError("output shape and stride ranks differ")
  if any(size < 0 for size in shape):
    raise ValueError("output shape contains a negative dimension")
  if any(size == 0 for size in shape):
    return 0, 0
  minimum = 0
  maximum = 0
  for size, stride in zip(shape, strides):
    span = (size - 1) * stride
    minimum += min(0, span)
    maximum += max(0, span)
  elements = maximum - minimum + 1
  if elements > 2**63 - 1:
    raise OverflowError("output storage span exceeds int64")
  return elements, -minimum


def _allocate(
  annotation: TensorType,
  environment: dict[str, int],
  reference: Any,
  inferred_dtype: Any,
):
  shape = tuple(_evaluate(item, environment) for item in annotation.shape)
  strides = tuple(_evaluate(item, environment) for item in annotation.concrete_strides)
  metadata = _metadata(reference) if reference is not None else None
  framework = metadata[3] if metadata is not None else "meta"
  dtype = annotation.dtype
  if dtype is None:
    if inferred_dtype is None:
      raise ValueError("cannot infer dtype for an automatically allocated output")
    dtype = inferred_dtype
  elements, offset = _storage_plan(shape, strides)
  if framework == "torch":
    if any(stride < 0 for stride in strides):
      raise ValueError("Torch cannot automatically allocate a tensor with negative strides")
    import torch

    base = torch.empty((elements,), dtype=framework_dtype(dtype, "torch"), device=reference.device)
    return torch.as_strided(base, shape, strides, offset)
  if framework == "numpy":
    import numpy as np

    np_dtype = framework_dtype(dtype, "numpy")
    base = np.empty((elements,), dtype=np_dtype)
    return np.ndarray(
      shape,
      dtype=np_dtype,
      buffer=base,
      offset=offset * np_dtype.itemsize,
      strides=tuple(item * np_dtype.itemsize for item in strides),
    )
  return TensorMeta(shape, dtype=dtype, strides=strides, access=TensorAccess.output)
