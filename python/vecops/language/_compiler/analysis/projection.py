# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact scalar projections through logical layout and conversion boundaries.

Keep the original memory read as an identity-bearing leaf. Consumers resolve
its snapshot/materialization rather than inventing a later read. Nonlocal,
masked and arithmetic values deliberately keep ordinary vector legalization.
"""

from dataclasses import replace

from .. import ir as I


def complete_grouped_prefix(source_extent, group, lane_multiple):
  """Prove every complete output chunk excludes a partial source group.

  Chunk widths may vary, but must be positive multiples of ``lane_multiple``.
  If the last output is partial and cannot end any such chunk, it belongs
  exclusively to the masked remainder. This also holds for scalable widths.
  The caller must establish traversal of this exact grouped-output domain.
  """
  if type(source_extent) is not int or source_extent < 0:
    return False
  if type(group) is not int or group < 1 or type(lane_multiple) is not int or lane_multiple < 1:
    return False
  return source_extent % group == 0 or ((source_extent + group - 1) // group) % lane_multiple != 0


def broadcast_coordinate(extent, coordinate):
  """Project one broadcast dimension, retaining a dynamic singleton guard."""
  known = I.constant(extent)
  if known == 1:
    return I.literal(0)
  if known is not None:
    return coordinate
  return I.Expr(I.ScalarType("int64"), "where", (I.binary("eq", extent, I.literal(1)), I.literal(0), coordinate))


def streaming_selection_supported(value, basis_dtype, *, staged=False):
  """A closed selection/conversion DAG fitting one native compute word.

  Memory reads, state, shape changes and arithmetic remain on the ordinary
  pointwise path. In particular this does not move floating arithmetic or
  exceptional math into a different architectural floating-point mode.
  """
  from ..dtypes import ELEMENT_BYTES
  from ..ir.traversal import walk_unique

  allowed = {
    "vector_ref",
    "literal",
    "raw",
    "symbol",
    "local",
    "frozen",
    "cast",
    "min",
    "max",
    "where",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and",
    "or",
    "xor",
    "not",
    "mask_xor",
    "bit_and",
    "bit_or",
    "bit_xor",
  }
  if staged:
    allowed.add("region_argument")
  return all(
    node.op in allowed
    and (
      not isinstance(node.type, I.TileType)
      or node.type.mask
      or ELEMENT_BYTES[node.type.dtype] <= ELEMENT_BYTES[basis_dtype]
    )
    for node in walk_unique(value)
  )


def pointwise_row_projection(value, boundary, reference, row, column, extent, catalog):
  """Project a pointwise caller DAG onto a logical/register row without new reads.

  Shape-changing/stateful boundaries are deliberately rejected. Loads retain
  their original identity behind ordinary value windows, so the standard alias
  planner still resolves historical captures. A singleton axis broadcasts
  from zero; it must not advance with the destination's induction variables.
  """
  from ..operations import Trait

  memo = {}

  def visit(value):
    if value is boundary:
      return reference
    if isinstance(value.type, I.ScalarType):
      return value
    if id(value) in memo:
      return memo[id(value)]
    shape = value.type.shape
    if not 1 <= len(shape) <= 2:
      raise TypeError("pointwise row projection requires rank-one/two operands")
    width = I.literal(1) if I.constant(shape[-1]) == 1 else extent
    typ = (
      I.VectorType(value.type.dtype, (width,), value.type.mask, reference.type.tag)
      if isinstance(reference.type, I.VectorType)
      else I.TileType(value.type.dtype, (width,), value.type.mask)
    )
    if catalog.has(value, Trait.MEMORY_READ):
      if value.op != "load":
        raise TypeError("opaque memory projection is unsupported")
      starts = ((I.literal(0) if I.constant(shape[0]) == 1 else row),) if len(shape) == 2 else ()
      starts += (I.literal(0) if I.constant(shape[-1]) == 1 else column,)
      result = I.Expr(typ, "value_transform", (value, *starts), ("slice", (len(shape) - 1,)))
    elif value.op == "arange":
      # A matrix epilogue's row projection changes both rank and origin.
      # Index generators need an explicit coordinate map; retagging a rank-2
      # column index as rank-1 leaves an invalid axis, and a row index must
      # become a broadcast scalar rather than an iota over columns.
      axis = value.data if value.data is not None else len(shape) - 1
      coordinate = column if axis == len(shape) - 1 else row
      coordinate = broadcast_coordinate(shape[axis], coordinate)
      start = I.binary("add", value.args[0], coordinate)
      result = I.Expr(typ, "arange", (start,), 0) if axis == len(shape) - 1 else I.Expr(typ, "fill", (start,))
    elif catalog.has(value, Trait.ELEMENTWISE):
      result = replace(value, type=typ, args=tuple(visit(a) for a in value.args))
    else:
      raise TypeError("non-pointwise projection boundary")
    memo[id(value)] = result
    return result

  return visit(value)


def scalar_memory_projection(value, indices, *, blocked=()):
  """Return (original load, coordinates, inner-to-outer casts), or None.

  This is not scalarization of an elementwise algorithm: casts are explicit
  rounding boundaries and all other accepted nodes only change coordinates.
  """
  if id(value) in blocked or value.type.mask:
    return None
  coordinates = tuple(
    I.literal(0) if I.constant(extent) == 1 else index for extent, index in zip(value.type.shape, indices)
  )
  if value.op == "load" and not value.args:
    return value, coordinates, ()
  if value.op == "cast":
    projected = scalar_memory_projection(value.args[0], coordinates, blocked=blocked)
    if projected is None:
      return None
    load, coordinates, casts = projected
    return load, coordinates, (*casts, value.type.dtype)
  if value.op != "value_transform":
    return None
  source = value.args[0]
  kind, axes = value.data
  if kind == "slice":
    mapped = list(value.args[1:])
    for index, axis in zip(coordinates, axes):
      mapped[axis] = I.binary("add", mapped[axis], index)
  elif kind == "expand_dims":
    mapped = coordinates[:axes] + coordinates[axes + 1 :]
  elif kind == "squeeze":
    mapped = coordinates[:axes] + (I.literal(0),) + coordinates[axes:]
  elif kind == "transpose":
    mapped = [None] * len(axes)
    for index, axis in zip(coordinates, axes):
      mapped[axis] = index
  elif kind == "row":
    mapped = (axes, *coordinates)
  elif kind == "broadcast_to":
    mapped = coordinates[len(coordinates) - len(source.type.shape) :]
    mapped = tuple(broadcast_coordinate(extent, index) for extent, index in zip(source.type.shape, mapped))
  else:
    return None
  return scalar_memory_projection(source, tuple(mapped), blocked=blocked)
