# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Target-independent vector unrolling, coalescing and type legalization."""

import math
from dataclasses import replace

from .. import ir as I
from ..dtypes import ELEMENT_BYTES
from ..ir import vector as V
from .manager import register_pass


@register_pass("unroll-vector-operations", stage="vector")
def unroll_operations(region, context):
  region.copies = tuple(
    V.OperationCopy(
      operation.index, slot, tuple((operand, slot) if operand is not None else None for operand in operation.operands)
    )
    for operation in region.operations
    for slot in range(region.unroll)
  )
  return region


@register_pass("coalesce-vector-groups", stage="vector", after=("unroll-vector-operations",))
def coalesce_groups(region, context):
  if region.iteration_axis is not None:
    region.groupings = {0: V.group_vectors(region.unroll, region.base_power, region.base_power)}
    return region
  region.groupings = {
    power: V.group_vectors(region.unroll, region.base_power, region.base_power + power)
    for power in range(region.unroll.bit_length())
  }
  return region


@register_pass("legalize-vector-types", stage="vector", after=("coalesce-vector-groups",))
def legalize_types(region, context):
  base = ELEMENT_BYTES[region.dtype]

  # Lane-changing primitives evaluate their producers at a different width.
  # A U8 grouped dot has four input lanes per I32 result: an FP32 producer
  # behind that U8 cast needs four times the result register bytes. Merely
  # taking the largest dtype in the DAG misses this intermediate geometry.
  maximum = int(math.log2(base))
  pending = [(root, 0) for root in region.roots]
  seen = set()
  while pending:
    value, lane_shift = pending.pop()
    if (id(value), lane_shift) in seen:
      continue
    seen.add((id(value), lane_shift))
    if isinstance(value.type, I.TileType):
      maximum = max(maximum, int(math.log2(ELEMENT_BYTES[value.type.dtype])) + lane_shift)
      if value.op == "load":
        maximum = max(maximum, int(math.log2(ELEMENT_BYTES[value.data.dtype])) + lane_shift)
    for argument in value.args:
      shift = lane_shift
      if value.op == "widening_dot":
        shift += int(math.log2(value.data))
      elif value.op == "bitreinterpret":
        shift += int(math.log2(ELEMENT_BYTES[value.type.dtype])) - int(math.log2(ELEMENT_BYTES[argument.type.dtype]))
      pending.append((argument, shift))
  region.rebind_headroom = max(0, maximum - int(math.log2(base)))
  region.operations = [
    replace(op, type=replace(op.type, tag=I.VectorTag(pow=region.base_power, basis_dtype=region.dtype)))
    for op in region.operations
  ]
  return region
