# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source-level register transpose traversal, including bounded gather fringes."""

from .. import dsl as T
from .registers import transpose_registers


@T.func
def _transpose_row(load_row, row, column, extent, dtype, columns):
  if row < extent:
    result = load_row(row, column)
  else:
    result = T.zeros(T.Vector(dtype, N=columns))
  return result


@T.func
def _register_tile(load_row, destination, mi, ni, rows, columns, dtype):
  values = ()
  for r in T.unroll(rows):
    values = values + (_transpose_row(load_row, mi + r, ni, destination.shape[1], dtype, columns),)
  transposed = transpose_registers(values)
  for c in T.unroll(columns):
    region = destination[ni + c, mi : mi + rows]
    T.store(region, transposed[c][: region.shape[0]])


@T.func
def register_transpose_into(source, destination, load_row, *, rows, columns, dtype, cache=128):
  """Register butterflies on complete columns; gather only the final fringe.

  The checked adapter supplies a fixed-extent load projection under the full
  column loop. Short spatial rows are zeroed without forming invalid pointers.
  Memory conversions stay in the load/store boundaries of ordinary lowering.
  """
  source = T.as_value(source)
  for mb in T.serial(0, destination.shape[1], cache, unroll=1):
    for nb in T.serial(0, destination.shape[0], cache, unroll=1):
      mend = T.min(mb + cache, destination.shape[1])
      nend = T.min(nb + cache, destination.shape[0])
      full_n = nb + (nend - nb) // columns * columns
      for mi in T.serial(mb, mend, rows, unroll=1):
        for ni in T.serial(nb, full_n, columns, unroll=1):
          _register_tile(load_row, destination, mi, ni, rows, columns, dtype)
      for ni in T.serial(full_n, nend, unroll=1):
        T.store(destination[ni, mb:mend], source[ni, mb:mend])
