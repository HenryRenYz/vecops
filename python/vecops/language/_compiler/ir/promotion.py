# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Target-independent floating promotion, including weak Python literals.

Integer/integer widths and signedness deliberately remain the existing IR
policy. Weak literals do not make an existing floating operand wider. A weak
float with integer operands instead selects F32, except for a finite, nonzero
value outside F32's nonzero magnitude range. This is a range rule, not a promise
that an arbitrary Python float is represented exactly. NaN, infinity and signed
zero select F32; explicit typed scalar values participate with their dtype.
"""

from __future__ import annotations

import math
import struct

from .types import ScalarType, TileType

FLOAT_DTYPES = frozenset(("float16", "bfloat16", "float32", "float64"))
F32_MIN_SUBNORMAL = 2.0**-149
F32_MAX = float.fromhex("0x1.fffffep127")


def weak_literal(value):
  return value.op == "literal" and isinstance(value.type, ScalarType) and value.type.weak


def floating_dtype(operands, *, shaped):
  """Select only floating computation types; return None for integer-only IR.

  Mask dtype is its lane basis, not a numeric operand dtype. Never cast a
  predicate to a numeric vector merely because its basis happens to be F32.
  Legacy scalar half arithmetic computes in F32; shaped half values keep their
  declared dtype, with mixed F16/BF16 requiring F32 to retain both ranges.
  """
  if any(getattr(value.type, "mask", False) for value in operands):
    return None
  strong = [value.type.dtype for value in operands if not weak_literal(value)]
  floating = set(strong) & FLOAT_DTYPES
  if floating:
    if "float64" in floating:
      return "float64"
    if "float32" in floating or floating == {"float16", "bfloat16"} or not shaped:
      return "float32"
    return next(iter(floating))
  weak_floats = [value.data for value in operands if weak_literal(value) and value.type.dtype in FLOAT_DTYPES]
  if not weak_floats:
    return None
  if not strong and not shaped:
    # Pure Python scalar expressions keep their existing double semantics.
    # The weak-literal rule applies when a typed value supplies a context.
    return "float64"
  for value in weak_floats:
    magnitude = abs(value)
    if math.isfinite(value) and magnitude != 0 and not F32_MIN_SUBNORMAL <= magnitude <= F32_MAX:
      return "float64"
  return "float32"


def cast_operands(operands, dtype, expr, shaped_type, *, shaped=False):
  """Make the selected computation dtype executable, not just an IR label."""
  result = []
  for value in operands:
    # A shaped operation's scalar fill already converts directly to the
    # selected element type. Keep weak constants unchanged so existing F32
    # kernels retain their DAG, fusion cost and canonical generated body.
    if value.type.dtype == dtype or shaped and weak_literal(value):
      result.append(value)
    else:
      typ = shaped_type(dtype, value.type.shape, value.type) if isinstance(value.type, TileType) else ScalarType(dtype)
      result.append(expr(typ, "cast", (value,)))
  return tuple(result)


def rounded_float(value, dtype):
  """Match scalar constructors and computation boundaries for typed literals.

  The public small-float constructors first convert Python's double to F32,
  then round to F16/BF16. Do not replace that documented double-rounding chain
  with a direct double-to-half conversion.
  """
  if dtype in {"float16", "bfloat16"}:
    value = rounded_float(value, "float32")
    if dtype == "float16":
      try:
        return struct.unpack("=e", struct.pack("=e", value))[0]
      except OverflowError:
        return math.copysign(math.inf, value)
    if math.isnan(value):
      return math.nan
    bits = struct.unpack("=I", struct.pack("=f", value))[0]
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return struct.unpack("=f", struct.pack("=I", bits << 16))[0]
  if dtype == "float32":
    try:
      return struct.unpack("=f", struct.pack("=f", value))[0]
    except OverflowError:
      return math.copysign(math.inf, value)
  return value
