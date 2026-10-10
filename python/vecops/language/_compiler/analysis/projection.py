# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact scalar projections through logical layout and conversion boundaries.

Keep the original memory read as an identity-bearing leaf. Consumers resolve
its snapshot/materialization rather than inventing a later read. Nonlocal,
masked and arithmetic values deliberately keep ordinary vector legalization.
"""

from dataclasses import replace

from .. import ir as I


def pointwise_row_projection(value, boundary, reference, row, column, extent, catalog):
  """Project a pointwise caller DAG onto a register row without new reads.

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
    typ = I.VectorType(value.type.dtype, (width,), value.type.mask, reference.type.tag)
    if catalog.has(value, Trait.MEMORY_READ):
      if value.op != "load":
        raise TypeError("opaque memory projection is unsupported")
      starts = ((I.literal(0) if I.constant(shape[0]) == 1 else row),) if len(shape) == 2 else ()
      starts += (I.literal(0) if I.constant(shape[-1]) == 1 else column,)
      result = I.Expr(typ, "value_transform", (value, *starts), ("slice", (len(shape) - 1,)))
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
  else:
    return None
  return scalar_memory_projection(source, tuple(mapped), blocked=blocked)
