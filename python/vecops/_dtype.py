"""Public dtype vocabulary and optional-framework normalization.

NumPy and Torch are deliberately inspected only when they are already loaded;
importing vecops never adds either framework as an import-time dependency.
"""

from __future__ import annotations

import sys
from functools import lru_cache
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

_DTYPES = {
  "bool": (bool_, 1),
  "int8": (int8, 1),
  "uint8": (uint8, 1),
  "int16": (int16, 2),
  "uint16": (uint16, 2),
  "int32": (int32, 4),
  "uint32": (uint32, 4),
  "int64": (int64, 8),
  "uint64": (uint64, 8),
  "float16": (float16, 2),
  "bfloat16": (bfloat16, 2),
  "float32": (float32, 4),
  "float64": (float64, 8),
}
_NAMES = {name: dtype for name, (dtype, _) in _DTYPES.items()}
_NAMES["bool_"] = bool_
_BY_VALUE = {dtype: (name, size) for name, (dtype, size) in _DTYPES.items()}
ELEMENT_BYTES = {name: size for name, (_, size) in _DTYPES.items()}


def is_dtype_name(value: Any) -> bool:
  """Recognize canonical names only, independently of framework imports."""
  return isinstance(value, str) and value in _NAMES


@lru_cache(maxsize=1)
def _torch_dtypes(torch):
  # Derive optional Torch types from the same registry. Older Torch versions
  # need not provide every unsigned type; importing vecops never imports Torch.
  return {getattr(torch, name): dtype for name, (dtype, _) in _DTYPES.items() if hasattr(torch, name)}


def normalize_dtype(value: Any) -> _C.DType:
  """Normalize native dtypes, canonical names and already-loaded framework dtypes.

  Canonical names, including ``bfloat16``, never depend on NumPy or Torch
  being imported. Framework-specific aliases remain the framework's concern.
  Dtype wrappers may implement ``__vecops_dtype__`` (as ``T.DType`` does).
  ``None`` is an unspecified schema dtype, not a concrete dtype. Non-native
  NumPy byte orders are rejected, matching the native tensor adapter.
  """
  if isinstance(value, _C.DType):
    return value
  unwrap = getattr(value, "__vecops_dtype__", None)
  if callable(unwrap):
    converted = unwrap()
    if not isinstance(converted, _C.DType):
      raise TypeError("__vecops_dtype__ must return a vecops dtype")
    return converted
  if is_dtype_name(value):
    return _NAMES[value]

  torch = sys.modules.get("torch")
  if torch is not None and isinstance(value, torch.dtype):
    converted = _torch_dtypes(torch).get(value)
    if converted is not None:
      return converted

  numpy = sys.modules.get("numpy")
  if numpy is not None and value is not None:
    try:
      dtype = numpy.dtype(value)
    except (TypeError, ValueError):
      pass
    else:
      if not dtype.isnative:
        raise TypeError("vecops only supports native-endian NumPy dtypes")
      name = dtype.name
      if name in _NAMES:
        return _NAMES[name]

  raise TypeError(f"unsupported dtype {value!r}; expected a canonical name or a vecops, NumPy, or Torch dtype")


def dtype_name(value: Any) -> str:
  """Return the canonical (framework-neutral) name, spelling boolean as bool."""
  return _BY_VALUE[normalize_dtype(value)][0]


def dtype_itemsize(value: Any) -> int:
  """Return the element width in bytes, without importing a framework."""
  return _BY_VALUE[normalize_dtype(value)][1]


def framework_dtype(value: Any, framework: str, *, allow_framework_only: bool = False):
  """Adapt a concrete dtype for allocation in the explicitly requested framework.

  NumPy BF16 requires an already-registered extension dtype (e.g. importing
  ml_dtypes in the application). No optional extension is imported implicitly.
  Memory-only adapters may opt into framework dtypes outside the kernel ABI
  (e.g. complex NumPy arrays). That does not make them supported kernel inputs.
  The Torch adapter also accepts serialized ``torch.<name>`` descriptors.
  """
  if framework == "torch":
    import torch

    if isinstance(value, str):
      candidate = getattr(torch, value.removeprefix("torch."), None)
      if isinstance(candidate, torch.dtype):
        value = candidate
    if allow_framework_only and isinstance(value, torch.dtype):
      return value
    name = dtype_name(value)
    result = getattr(torch, name, None)
    if result is None:
      raise TypeError(f"this Torch version does not support dtype {name}")
    return result
  if framework == "numpy":
    import numpy as np

    try:
      normalized = normalize_dtype(value)
    except TypeError:
      if not allow_framework_only:
        raise
      return np.dtype(value)
    name = dtype_name(normalized)
    try:
      return np.dtype(name)
    except TypeError as error:
      raise TypeError(
        f"NumPy dtype {name} is unavailable; BF16 requires a registered extension such as ml_dtypes"
      ) from error
  return normalize_dtype(value)


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
