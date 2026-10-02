"""Shared dtype normalization; no parser or target-code dependencies."""
from ..._dtype import normalize_dtype, _NAMES
from ..expr import DType

ELEMENT_BYTES = {"bool": 1, "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
                 **{f"{s}int{n}": n // 8 for s in ("", "u") for n in (8, 16, 32, 64)}}

def dtype_name(value):
  if isinstance(value, DType):
    value = value.value
  if value in (float, int, bool):
    return {float: "float64", int: "int64", bool: "bool"}[value]
  if isinstance(value, str) and value in _NAMES:
    value = _NAMES[value]
  return str(normalize_dtype(value)).split(".")[-1].replace("bool_", "bool")
