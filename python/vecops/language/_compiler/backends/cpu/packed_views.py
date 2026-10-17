# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Prove aligned logical windows of the public native packing ABI.

Retain the original panel pitch, including its original K padding. Only
complete instruction K groups are borrowed: arbitrary external padding is
never promoted to trusted zero padding. Capacity/density/output nonalias
validation belongs to the caller and must precede these physical views.
"""

from dataclasses import replace

from ... import ir as I
from ...dtypes import ELEMENT_BYTES


def aligned_packed_view(operand, family, acc_dtype, streaming_bytes=None):
  value = operand
  origins = [0, 0]
  sliced = False
  while value.op == "value_transform" and value.data == ("slice", (0, 1)):
    offsets = [I.constant(v) for v in value.args[1:]]
    if len(offsets) != 2 or any(type(v) is not int for v in offsets):
      return None
    origins = [a + b for a, b in zip(origins, offsets)]
    value = value.args[0]
    sliced = True
  if not sliced or value.op != "packed_load" or value.data.format not in {family, "auto"}:
    return None
  packed = value.data
  if packed.origins or acc_dtype == "int64":
    return None
  width = ELEMENT_BYTES[operand.type.dtype]
  if family == "amx":
    span, quantum = 16, 64 // width
  elif family == "sme" and streaming_bytes:
    span, quantum = 2 * streaming_bytes // ELEMENT_BYTES[acc_dtype], ELEMENT_BYTES[acc_dtype] // width
  else:
    return None
  spatial_axis, k_axis = (0, 1) if packed.side == "A" else (1, 0)
  spatial, k = operand.type.shape[spatial_axis], operand.type.shape[k_axis]
  reduction = I.constant(k)
  if type(reduction) is not int or reduction % quantum or origins[spatial_axis] % span or origins[k_axis] % quantum:
    return None
  root_k = packed.shape[k_axis]
  lit = I.literal
  mul = lambda a, b: I.binary("mul", a, b)
  groups = I.binary("floordiv", I.binary("add", root_k, lit(quantum - 1)), lit(quantum))
  panels = I.binary("floordiv", I.binary("add", spatial, lit(span - 1)), lit(span))
  unit = span * quantum
  panel_stride = mul(groups, lit(unit))
  offset = I.binary(
    "add",
    packed.view.offset,
    I.binary("add", mul(lit(origins[spatial_axis] // span), panel_stride), lit(origins[k_axis] // quantum * unit)),
  )
  if family == "amx":
    group = 4 // width
    shape = (panels, lit(reduction // quantum), lit(16), lit(quantum if packed.side == "A" else 16 * group))
    strides = (panel_stride, lit(unit), shape[3], lit(1))
  else:
    shape = (panels, lit(reduction // quantum), lit(unit))
    strides = (panel_stride, lit(unit), lit(1))
  # A checked physical window becomes a new indexing root. Its absolute
  # offset/strides remain those of the caller's allocation; owned stays false.
  region = I.ViewRegion(
    shape,
    strides,
    (lit(0),) * len(shape),
    shape,
    (1,) * len(shape),
    tuple(range(len(shape))),
    (False,) * len(shape),
    root_offset=offset,
  )
  return replace(packed.view, shape=shape, strides=strides, offset=offset, region=region)
