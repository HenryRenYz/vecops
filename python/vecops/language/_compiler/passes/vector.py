"""Target-independent vector unrolling, coalescing and type legalization."""

import math
from dataclasses import replace
from .. import ir as I
from ..ir import vector as V
from ..dtypes import ELEMENT_BYTES
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

  def expressions(value):
    yield value
    for argument in value.args:
      yield from expressions(argument)

  sizes = [
    ELEMENT_BYTES[e.type.dtype] for root in region.roots for e in expressions(root) if isinstance(e.type, I.TileType)
  ]
  sizes.extend(ELEMENT_BYTES[load.data.dtype] for root in region.roots for load in I.loads(root))
  region.rebind_headroom = max(0, int(math.log2(max(sizes, default=base))) - int(math.log2(base)))
  region.operations = [
    replace(op, type=replace(op.type, tag=I.VectorTag(pow=region.base_power, basis_dtype=region.dtype)))
    for op in region.operations
  ]
  return region
