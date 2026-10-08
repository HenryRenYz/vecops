# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""SME-specific DSL state and instruction vocabulary.

`with T.sme.region():` is compiled lexical ownership, never a host Python
context manager. load(view) creates one region-local streaming vector;
mopa(tile, A, B) selects a typed outer-product instruction, store_tile(tile, C)
writes the clipped accumulator tile. T.dot owns a separately planned schedule.
Validation/alias handling is outside SM. Vectors cannot escape into Python;
16/8-bit inputs must already use the architectural grouped-K packing order.
Bounded T.serial loops and pure metadata branches keep one region active.
load_hor/ver(tile, index, view) and store_hor/ver(tile, index, view) transfer
one contiguous rank-one storage slice directly between memory and ZA; they
preserve storage bits without numeric conversion. The view and index must fit
one streaming word; checks precede entry, not individual instructions.
An optional word_dtype selects the ZA word width independently of memory
elements, enabling grouped-K bit packing. Only complete words are transferred;
a fractional terminal word is untouched, never rounded up into an overread.
load_bits(view, dtype="uint32") instead performs a storage-element-masked
vector load followed by a bitcast: a fractional word is zero padded. Its
region-local unsigned vector may be written into ZA. store_bits(vector, view)
stores the requested storage prefix without numerical conversion, including
zero inactive lanes from a masked ZA read. Both fit one actual streaming word.
"""

from . import dsl as T
from .dsl import Intrinsic

region = Intrinsic("sme.region")
zero_za = Intrinsic("sme.zero_za")
streaming_lanes = Intrinsic("sme.streaming_lanes")
load = Intrinsic("sme.load")
load_bits = Intrinsic("sme.load_bits")
mopa = Intrinsic("sme.mopa")
store_tile = Intrinsic("sme.store_tile")
load_hor = Intrinsic("sme.load_hor")
load_ver = Intrinsic("sme.load_ver")
store_hor = Intrinsic("sme.store_hor")
store_ver = Intrinsic("sme.store_ver")
read_hor = Intrinsic("sme.read_hor")
read_ver = Intrinsic("sme.read_ver")
write_hor = Intrinsic("sme.write_hor")
write_ver = Intrinsic("sme.write_ver")
store = Intrinsic("sme.store")
store_bits = Intrinsic("sme.store_bits")


@T.func
def _transpose_input_pair(source, row, compute_dtype):
  if source.dtype == compute_dtype:
    load_hor(0, row, source[row, :])
    load_hor(0, row + 1, source[row + 1, :])
  else:
    first = load(source[row, :], dtype=compute_dtype)
    second = load(source[row + 1, :], dtype=compute_dtype)
    write_hor(0, row, first)
    write_hor(0, row + 1, second)


@T.func
def _transpose_output_pair(destination, column, rows, compute_dtype):
  if destination.dtype == compute_dtype:
    store_ver(0, column, destination[column, :])
    store_ver(0, column + 1, destination[column + 1, :])
  else:
    first = read_ver(0, column, rows, dtype=compute_dtype)
    second = read_ver(0, column + 1, rows, dtype=compute_dtype)
    store(first, destination[column, :])
    store(second, destination[column + 1, :])


@T.func
def _transpose_input(source, compute_dtype, pairs, unroll):
  if pairs:
    for row in T.serial(0, source.shape[0] - 1, 2, unroll=unroll):
      _transpose_input_pair(source, row, compute_dtype)
    start = source.shape[0] // 2 * 2
  else:
    start = 0
  for row in T.serial(start, source.shape[0], unroll=unroll):
    if source.dtype == compute_dtype:
      load_hor(0, row, source[row, :])
    else:
      value = load(source[row, :], dtype=compute_dtype)
      write_hor(0, row, value)


@T.func
def _transpose_output(destination, rows, compute_dtype, pairs, unroll):
  if pairs:
    for column in T.serial(0, destination.shape[0] - 1, 2, unroll=unroll):
      _transpose_output_pair(destination, column, rows, compute_dtype)
    start = destination.shape[0] // 2 * 2
  else:
    start = 0
  for column in T.serial(start, destination.shape[0], unroll=unroll):
    if destination.dtype == compute_dtype:
      store_ver(0, column, destination[column, :])
    else:
      value = read_ver(0, column, rows, dtype=compute_dtype)
      store(value, destination[column, :])


@T.func
def _transpose_scheduled_tile(
  source, destination, compute_dtype, pairs, input_unroll, output_unroll, full_input, full_output
):
  """CPU library schedule selected before printing, with typed cold fringes.

  Constant full-tile branches bound code size and address live ranges. The
  automatic layout-conversion caller proves nonaliasing and unit strides;
  source-level guards keep full schedules away from partial tiles.
  """
  width = streaming_lanes(compute_dtype)
  if full_input is not None:
    if source.shape[0] == width and source.shape[1] == width and width == full_input:
      for row in T.unroll(0, full_input, 2):
        # Explicit vector loads preserve the measured load/load/write/write
        # order instead of the same-type raw ZA memory fast path.
        first = load(source[row, :], dtype=compute_dtype)
        second = load(source[row + 1, :], dtype=compute_dtype)
        write_hor(0, row, first)
        write_hor(0, row + 1, second)
    else:
      _transpose_input(source, compute_dtype, pairs, input_unroll)
  else:
    _transpose_input(source, compute_dtype, pairs, input_unroll)
  if full_output is not None:
    if source.shape[0] == width and source.shape[1] == width and width == full_output:
      for column in T.unroll(0, full_output, 2):
        _transpose_output_pair(destination, column, source.shape[0], compute_dtype)
    else:
      _transpose_output(destination, source.shape[0], compute_dtype, pairs, 1)
  else:
    _transpose_output(destination, source.shape[0], compute_dtype, pairs, output_unroll)


@T.func
def _transpose_scheduled_matrix(
  source, destination, compute_dtype, *, pairs=False, input_unroll=2, output_unroll=2, full_input=None, full_output=None
):
  """Executable SME layout-conversion library driver (not a compiler op)."""
  width = streaming_lanes(compute_dtype)
  for mi in T.serial(0, source.shape[0], width, unroll=1):
    for ni in T.serial(0, source.shape[1], width, unroll=1):
      _transpose_scheduled_tile(
        source[mi : mi + width, ni : ni + width],
        destination[ni : ni + width, mi : mi + width],
        compute_dtype,
        pairs,
        input_unroll,
        output_unroll,
        full_input,
        full_output,
      )


@T.func
def transpose_tile(source, destination, *, compute_dtype=None):
  """ZA transpose of a word-sized matrix in an active region.

  This is inspectable DSL, not a compiler operator. Both dimensions must fit
  the actual streaming word; the backend preflights that before ownership.
  Slices may be partial, but each memory row must have unit element stride.
  Operands may alias: all source rows enter ZA before any output is written.
  Same-type paths use raw LD1/ST1 ZA traffic. Explicit compute_dtype converts
  through region-local vectors with load_convert/store_convert at boundaries.
  """
  T.static_assert(len(source.shape) == 2 and len(destination.shape) == 2)
  if compute_dtype is None:
    compute_dtype = source.dtype
  compute_dtype = T.dtype_name(compute_dtype)
  assert source.shape[0] == destination.shape[1], "ZA transpose shape mismatch"
  assert source.shape[1] == destination.shape[0], "ZA transpose shape mismatch"
  for row in T.serial(source.shape[0], unroll=2):
    if source.dtype == compute_dtype:
      load_hor(0, row, source[row, :])
    else:
      value = load(source[row, :], dtype=compute_dtype)
      write_hor(0, row, value)
  for column in T.serial(source.shape[1], unroll=2):
    if destination.dtype == compute_dtype:
      store_ver(0, column, destination[column, :])
    else:
      value = read_ver(0, column, source.shape[0], dtype=compute_dtype)
      store(value, destination[column, :])


@T.func
def transpose_matrix(source, destination, *, compute_dtype=None):
  """Word-tiled transpose in one active region, for disjoint whole matrices.

  Actual streaming length sets the tile width; clipped input/output regions
  preserve fringes. Same-dtype traffic goes straight to/from ZA; conversions
  compose the region-local vector primitives. Use transpose_tile for a
  word-sized in-place transpose, not this independently traversed algorithm.
  """
  T.static_assert(len(source.shape) == 2 and len(destination.shape) == 2)
  assert T.view_disjoint(source, destination), "SME matrix transpose requires disjoint operands"
  assert source.shape[0] == destination.shape[1], "ZA transpose shape mismatch"
  assert source.shape[1] == destination.shape[0], "ZA transpose shape mismatch"
  if compute_dtype is None:
    compute_dtype = source.dtype
  compute_dtype = T.dtype_name(compute_dtype)
  width = streaming_lanes(compute_dtype)
  for mi in T.serial(0, source.shape[0], width, unroll=1):
    for ni in T.serial(0, source.shape[1], width, unroll=1):
      transpose_tile(
        source[mi : mi + width, ni : ni + width],
        destination[ni : ni + width, mi : mi + width],
        compute_dtype=compute_dtype,
      )


__all__ = [
  "load",
  "load_bits",
  "load_hor",
  "load_ver",
  "mopa",
  "read_hor",
  "read_ver",
  "region",
  "store",
  "store_bits",
  "store_hor",
  "store_tile",
  "store_ver",
  "streaming_lanes",
  "transpose_matrix",
  "transpose_tile",
  "write_hor",
  "write_ver",
  "zero_za",
]
