# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source-level ZA packing schedules; primitive lowering owns no algorithm."""

from .. import dsl as T
from .. import sme


@T.func
def _transfer_words(
  source, destination, origin, k_origin, output_origin, width, chunk, load_base, store_base, prefetch, row_unroll
):
  # Select the hint schedule once per chunk, not once per row. Each helper
  # expansion receives a literal hint flag, so its hot loop has no dispatch.
  if prefetch:
    _transfer_rows(
      source, destination, origin, k_origin, output_origin, width, chunk, load_base, store_base, True, row_unroll
    )
  else:
    _transfer_rows(
      source, destination, origin, k_origin, output_origin, width, chunk, load_base, store_base, False, row_unroll
    )


@T.func
def _transfer_rows(
  source, destination, origin, k_origin, output_origin, width, chunk, load_base, store_base, prefetch, row_unroll
):
  ratio = 4 // T.itemsize(source.dtype)
  for row in T.serial(width, unroll=row_unroll):
    for tile in T.unroll(2):
      if prefetch:
        T.prefetch(source[origin + tile * width + row, k_origin + 4 * chunk])
      sme.load_hor(
        load_base + tile, row, source[origin + tile * width + row, k_origin : k_origin + chunk], word_dtype="uint32"
      )
    if store_base is not None:
      for tile in T.unroll(2):
        sme.store_ver(
          store_base + tile,
          row,
          destination[output_origin + row, tile * width * ratio : (tile + 1) * width * ratio],
          word_dtype="uint32",
        )


@T.func
def _drain_words(destination, output_origin, width, ratio, store_base, row_unroll):
  for group in T.serial(width, unroll=row_unroll):
    for tile in T.unroll(2):
      sme.store_ver(
        store_base + tile,
        group,
        destination[output_origin + group, tile * width * ratio : (tile + 1) * width * ratio],
        word_dtype="uint32",
      )


@T.func
def pack_full_words(source, destination, *, origin, width, chunk, chunks, row_unroll=2):
  """Two ZA tile pairs: prime, load/store overlap, then drain.

  The CPU planner uses this only after proving a complete spatial panel and
  at least one complete K chunk. Fractional K and spatial tails are separate
  schedules. Metadata stays SSA; word moves reinterpret storage, not values.
  """
  ratio = 4 // T.itemsize(source.dtype)
  _transfer_words(source, destination, origin, 0, 0, width, chunk, 0, None, chunks > 4, row_unroll)
  for index in T.serial(1, chunks, unroll=1):
    offset = (index - 1) * width
    if index % 2:
      _transfer_words(
        source, destination, origin, index * chunk, offset, width, chunk, 2, 0, chunks - index > 4, row_unroll
      )
    else:
      _transfer_words(
        source, destination, origin, index * chunk, offset, width, chunk, 0, 2, chunks - index > 4, row_unroll
      )
  offset = (chunks - 1) * width
  if chunks % 2:
    _drain_words(destination, offset, width, ratio, 0, row_unroll)
  else:
    _drain_words(destination, offset, width, ratio, 2, row_unroll)


@T.func
def pack_fringe(
  source,
  destination,
  *,
  origin,
  k_origin,
  width,
  active_k,
  spatial,
  output_origin=0,
  group_unroll=2,
  bounded_rows=False,
):
  """Zero-pad fractional K groups and spatial tails through typed ZA words.

  Narrow storage uses its own view of the physical ZA row, for both complete
  and fractional K groups. LD1's inactive-storage zeroing pads the final word
  without a Z-register round trip or a runtime word-alignment dispatch inside
  the row loop. Masked ZA reads zero absent spatial lanes before full stores.
  Neither path evaluates a padded source address or needs a Vec array.
  """
  word_dtype = "uint64" if T.itemsize(source.dtype) == 8 else "uint32"
  ratio = T.itemsize(word_dtype) // T.itemsize(source.dtype)
  # The spatial prefix does not vary across K groups. Make its predicate a
  # visible register value rather than reconstructing it at every ZA read.
  first_mask = sme.mask(T.max(0, T.min(width, spatial - origin)), dtype=word_dtype)
  second_mask = sme.mask(T.max(0, T.min(width, spatial - origin - width)), dtype=word_dtype)
  for tile in T.unroll(2):
    for row in T.serial(T.max(0, T.min(width, spatial - origin - tile * width)), unroll=2):
      if bounded_rows:
        # The row loop still visits only valid rows. This redundant clamp
        # exposes their bound to a preflight that conservatively separates
        # the enclosing panel interval from the clipped row-loop interval.
        region = source[T.min(origin + tile * width + row, spatial - 1), k_origin : k_origin + active_k]
      else:
        region = source[origin + tile * width + row, k_origin : k_origin + active_k]
      if T.itemsize(source.dtype) >= 4:
        sme.load_hor(tile, row, region, word_dtype=word_dtype)
      else:
        # ZA tile views alias the same physical byte rows. A storage-width
        # LD1 zeros inactive storage lanes, including the unused bytes in
        # the final packed word. Translate the wider-word row into that
        # view instead of round-tripping through Z and WRITE_HOR.
        storage_bytes = T.itemsize(source.dtype)
        storage_word = "uint8" if storage_bytes == 1 else "uint16"
        sme.load_hor(tile % storage_bytes, row * ratio + tile // storage_bytes, region, word_dtype=storage_word)
  for group in T.serial((active_k + ratio - 1) // ratio, unroll=group_unroll):
    for tile in T.unroll(2):
      value = sme.read_ver(tile, group, first_mask if tile == 0 else second_mask, dtype=word_dtype)
      sme.store_bits(
        value,
        destination[output_origin + group, tile * width * ratio : (tile + 1) * width * ratio],
      )


@T.func
def _pack_remainder(source, destination, origin, first_chunk, width, chunk, ratio, group_unroll):
  # Complete chunks have constant masks/group counts. A clipped loop over
  # both full and partial chunks hides that fact even from native unrolling.
  for index in T.serial(first_chunk, source.shape[1] // chunk, unroll=1):
    pack_fringe(
      source,
      destination,
      origin=origin,
      k_origin=index * chunk,
      width=width,
      active_k=chunk,
      spatial=source.shape[0],
      output_origin=index * width,
      group_unroll=group_unroll,
    )
  if source.shape[1] % chunk:
    pack_fringe(
      source,
      destination,
      origin=origin,
      k_origin=source.shape[1] - source.shape[1] % chunk,
      width=width,
      active_k=source.shape[1] % chunk,
      spatial=source.shape[0],
      output_origin=(source.shape[1] - source.shape[1] % chunk) // ratio,
    )


@T.func
def pack_raw(source, destination, *, row_unroll=2, group_unroll=2):
  """Whole K-contiguous packed-layout algorithm in one streaming owner.

  The checked caller binds destination's logical [panel, K-group, elements]
  layout after validating physical capacity, format, nonaliasing and unit K
  stride. Each group contains two ZA words of storage elements, including
  padding. Keeping that hierarchy avoids flat-capacity clamps on every store.
  This recipe owns traversal, double buffering and tails, and derives all
  geometry from the current streaming domain.
  """
  word = "uint64" if T.itemsize(source.dtype) == 8 else "uint32"
  width = sme.streaming_lanes(word)
  chunk = sme.streaming_lanes(source.dtype)
  ratio = T.itemsize(word) // T.itemsize(source.dtype)
  # Empty storage has no panels to visit, even if its other logical extent
  # is enormous. In particular, do not let an empty K domain run an M loop
  # whose final padded induction value could overflow.
  if source.shape[0] > 0 and source.shape[1] > 0:
    if source.shape[0] < 2 * width and source.shape[1] <= chunk:
      # One spatial fringe and one K chunk need neither panel bookkeeping
      # nor a K loop. A complete spatial panel keeps the direct-store path.
      pack_fringe(
        source,
        destination[0, :, :],
        origin=0,
        k_origin=0,
        width=width,
        active_k=source.shape[1],
        spatial=source.shape[0],
      )
    else:
      _pack_panels(source, destination, width, chunk, ratio, row_unroll, group_unroll)


@T.func
def _pack_panels(source, destination, width, chunk, ratio, row_unroll, group_unroll):
  if T.itemsize(source.dtype) <= 4:
    # Complete spatial panels and the sole spatial fringe have disjoint
    # lifetimes. Do not keep every pipeline address induction live across
    # a per-panel full/fringe dispatch, especially when there are no full
    # panels at all. Subtraction avoids a round-up overflow.
    full_spatial = source.shape[0] - source.shape[0] % (2 * width)
    for panel_index in T.serial(source.shape[0] // (2 * width), unroll=1):
      panel_output = destination[panel_index, :, :]
      if source.shape[1] >= chunk:
        pack_full_words(
          source,
          panel_output,
          origin=panel_index * (2 * width),
          width=width,
          chunk=chunk,
          chunks=source.shape[1] // chunk,
          row_unroll=row_unroll,
        )
      _pack_remainder(
        source, panel_output, panel_index * (2 * width), source.shape[1] // chunk, width, chunk, ratio, group_unroll
      )
    if full_spatial < source.shape[0]:
      panel_output = destination[full_spatial // (2 * width), :, :]
      _pack_remainder(source, panel_output, full_spatial, 0, width, chunk, ratio, group_unroll)
  else:
    for origin in T.serial(0, source.shape[0], 2 * width, unroll=1):
      panel_output = destination[origin // (2 * width), :, :]
      _pack_remainder(source, panel_output, origin, 0, width, chunk, ratio, group_unroll)


@T.func
def pack_typed_in_region(source, destination, acc_dtype):
  """Pack a native dense View into preallocated private panels in an owner.

  The caller supplies an active SME owner, a K-contiguous [spatial,K]
  source and whole-chunk [panel,K-group,2*width*group] destination. Keep
  the destination's actual panel/group strides; no public ABI flattening,
  allocation, source conversion or ownership transition occurs here.
  """
  T.static_assert(len(source.shape) == 2 and len(destination.shape) == 3)
  T.static_assert(source.dtype == destination.dtype)
  width = sme.streaming_lanes(acc_dtype)
  chunk = sme.streaming_lanes(source.dtype)
  group = T.itemsize(acc_dtype) // T.itemsize(source.dtype)
  for panel in T.serial(destination.shape[0], unroll=1):
    for chunk_index in T.serial((source.shape[1] + chunk - 1) // chunk, unroll=1):
      ki = chunk_index * chunk
      pack_fringe(
        source,
        destination[panel, :, :],
        origin=panel * 2 * width,
        k_origin=ki,
        width=width,
        active_k=T.min(chunk, source.shape[1] - ki),
        spatial=source.shape[0],
        output_origin=ki // group,
        bounded_rows=True,
      )
