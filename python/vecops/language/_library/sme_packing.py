"""Source-level ZA packing schedules; primitive lowering owns no algorithm."""

from .. import dsl as T
from .. import sme


@T.func
def _transfer_words(
  source, destination, origin, k_origin, output_origin, width, chunk, load_base, store_base, prefetch
):
  ratio = 4 // T.itemsize(source.dtype)
  for row in T.serial(width, unroll=2):
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
          destination[
            output_origin + (row * 2 + tile) * width * ratio : output_origin + (row * 2 + tile + 1) * width * ratio
          ],
          word_dtype="uint32",
        )


@T.func
def _drain_words(destination, output_origin, width, ratio, store_base):
  for group in T.serial(width, unroll=2):
    for tile in T.unroll(2):
      sme.store_ver(
        store_base + tile,
        group,
        destination[
          output_origin + (group * 2 + tile) * width * ratio : output_origin + (group * 2 + tile + 1) * width * ratio
        ],
        word_dtype="uint32",
      )


@T.func
def pack_full_words(source, destination, *, origin, width, chunk, chunks):
  """Two ZA tile pairs: prime, load/store overlap, then drain.

  The CPU planner uses this only after proving a complete spatial panel and
  at least one complete K chunk. Fractional K and spatial tails are separate
  schedules. Metadata stays SSA; word moves reinterpret storage, not values.
  """
  ratio = 4 // T.itemsize(source.dtype)
  _transfer_words(source, destination, origin, 0, 0, width, chunk, 0, None, chunks > 4)
  for index in T.serial(1, chunks, unroll=1):
    offset = (index - 1) * 2 * width * width * ratio
    if index % 2:
      _transfer_words(source, destination, origin, index * chunk, offset, width, chunk, 2, 0, chunks - index > 4)
    else:
      _transfer_words(source, destination, origin, index * chunk, offset, width, chunk, 0, 2, chunks - index > 4)
  offset = (chunks - 1) * 2 * width * width * ratio
  if chunks % 2:
    _drain_words(destination, offset, width, ratio, 0)
  else:
    _drain_words(destination, offset, width, ratio, 2)


@T.func
def pack_fringe(source, destination, *, origin, k_origin, width, active_k, spatial):
  """Zero-pad fractional K groups and spatial tails through typed ZA words.

  Storage-masked loads preserve every input bit and zero a fractional final
  word. Masked ZA reads zero absent spatial lanes before full packed stores.
  Neither path evaluates a padded source address or needs a Vec array.
  """
  word_dtype = "uint64" if T.itemsize(source.dtype) == 8 else "uint32"
  ratio = T.itemsize(word_dtype) // T.itemsize(source.dtype)
  for tile in T.unroll(2):
    for row in T.serial(T.max(0, T.min(width, spatial - origin - tile * width)), unroll=2):
      value = sme.load_bits(source[origin + tile * width + row, k_origin : k_origin + active_k], dtype=word_dtype)
      sme.write_hor(tile, row, value)
  for group in T.serial((active_k + ratio - 1) // ratio, unroll=2):
    for tile in T.unroll(2):
      value = sme.read_ver(tile, group, T.max(0, T.min(width, spatial - origin - tile * width)), dtype=word_dtype)
      sme.store_bits(
        value,
        destination[(group * 2 + tile) * width * ratio : (group * 2 + tile + 1) * width * ratio],
      )


@T.func
def _pack_remainder(source, destination, origin, start, width, chunk, ratio):
  for ki in T.serial(start, source.shape[1], chunk, unroll=1):
    pack_fringe(
      source,
      destination[(ki // ratio) * 2 * width * ratio :],
      origin=origin,
      k_origin=ki,
      width=width,
      active_k=T.min(chunk, source.shape[1] - ki),
      spatial=source.shape[0],
    )


@T.func
def pack_raw(source, destination):
  """Whole K-contiguous packed-layout algorithm in one streaming owner.

  The checked caller validates format, capacity, nonaliasing and unit K
  stride. This recipe owns panel traversal, full-word double buffering and
  both tails. It derives all geometry from the current streaming domain.
  """
  word = "uint64" if T.itemsize(source.dtype) == 8 else "uint32"
  width = sme.streaming_lanes(word)
  chunk = sme.streaming_lanes(source.dtype)
  ratio = T.itemsize(word) // T.itemsize(source.dtype)
  groups = (source.shape[1] + ratio - 1) // ratio
  for origin in T.serial(0, source.shape[0], 2 * width, unroll=1):
    panel_output = destination[(origin // (2 * width)) * groups * 2 * width * ratio :]
    if T.itemsize(source.dtype) <= 4:
      if source.shape[0] - origin >= 2 * width:
        if source.shape[1] >= chunk:
          pack_full_words(
            source, panel_output, origin=origin, width=width, chunk=chunk, chunks=source.shape[1] // chunk
          )
        _pack_remainder(source, panel_output, origin, (source.shape[1] // chunk) * chunk, width, chunk, ratio)
      else:
        _pack_remainder(source, panel_output, origin, 0, width, chunk, ratio)
    else:
      _pack_remainder(source, panel_output, origin, 0, width, chunk, ratio)
