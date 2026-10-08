# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact scalar projections through logical layout and conversion boundaries.

Keep the original memory read as an identity-bearing leaf. Consumers resolve
its snapshot/materialization rather than inventing a later read. Nonlocal,
masked and arithmetic values deliberately keep ordinary vector legalization.
"""

from .. import ir as I


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
