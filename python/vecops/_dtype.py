# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Public dtype vocabulary and optional-framework normalization.

NumPy and Torch are deliberately inspected only when they are already loaded;
importing vecops never adds either framework as an import-time dependency.
"""

from __future__ import annotations

import sys
from typing import Any

from . import _C


class DType:
  """Marker for a named dtype specialization value."""


class Const:
  """Marker for an anonymous or named compile-time integer."""


bool_ = _C.DType.bool_
int8 = _C.DType.int8
uint8 = _C.DType.uint8
int16 = _C.DType.int16
uint16 = _C.DType.uint16
int32 = _C.DType.int32
uint32 = _C.DType.uint32
int64 = _C.DType.int64
uint64 = _C.DType.uint64
float16 = _C.DType.float16
bfloat16 = _C.DType.bfloat16
float32 = _C.DType.float32
float64 = _C.DType.float64

_NAMES = {
  "bool": bool_,
  "bool_": bool_,
  "int8": int8,
  "uint8": uint8,
  "int16": int16,
  "uint16": uint16,
  "int32": int32,
  "uint32": uint32,
  "int64": int64,
  "uint64": uint64,
  "float16": float16,
  "bfloat16": bfloat16,
  "float32": float32,
  "float64": float64,
}


def normalize_dtype(value: Any) -> _C.DType:
  """Normalize vecops, NumPy, or Torch dtype values without importing either framework."""
  if isinstance(value, _C.DType):
    return value

  torch = sys.modules.get("torch")
  if torch is not None:
    torch_dtypes = {
      torch.bool: bool_,
      torch.int8: int8,
      torch.uint8: uint8,
      torch.int16: int16,
      torch.int32: int32,
      torch.int64: int64,
      torch.float16: float16,
      torch.bfloat16: bfloat16,
      torch.float32: float32,
      torch.float64: float64,
    }
    try:
      converted = torch_dtypes.get(value)
    except TypeError:
      converted = None
    if converted is not None:
      return converted

  numpy = sys.modules.get("numpy")
  if numpy is not None:
    try:
      dtype = numpy.dtype(value)
    except (TypeError, ValueError):
      pass
    else:
      name = dtype.name
      if name in _NAMES:
        return _NAMES[name]

  raise TypeError(f"unsupported dtype {value!r}; expected a vecops, NumPy, or Torch dtype")


def scalar_dtype(annotation: Any) -> _C.DType:
  """Map Python scalar annotations to ABI types, then normalize concrete dtypes."""
  if annotation is bool:
    return bool_
  if annotation is int:
    return int64
  if annotation is float:
    return float64
  return normalize_dtype(annotation)


__all__ = [
  "Const",
  "DType",
  "bfloat16",
  "bool_",
  "float16",
  "float32",
  "float64",
  "int8",
  "int16",
  "int32",
  "int64",
  "normalize_dtype",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
]
