# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Portable packed-ABI construction; ISA-specific raw recipes specialize it."""

from .. import dsl as T
from .. import sme
from .amx_matmul import _pack_b_fringe, _pack_b_full, _pack_b_full_k, prepare_b
from .sme_packing import pack_fringe


@T.func
def pack_sme(source, destination, *, side, raw=None, raw_safe=False, bounded=False):
  """Construct the public group-padded ABI, without private chunk padding.

  Materialize only logical producer indices, preserving every conversion
  boundary. ZA transposes storage words into the exact caller-owned prefix.
  ``bounded`` stages disjoint producers in cache blocks; overlapping inputs
  retain complete capture before the first destination write. Raw views bypass
  staging only when their logical reduction dimension has unit stride.
  """
  source = T.as_value(source)
  logical = source if side == "A" else T.transpose(source)
  word = "uint64" if T.itemsize(destination.dtype) == 8 else "uint32"
  width = sme.streaming_lanes(word)
  group = T.itemsize(word) // T.itemsize(destination.dtype)
  panels = T.reshape(destination, ((logical.shape[0] + 2 * width - 1) // (2 * width),
                                   (logical.shape[1] + group - 1) // group, 2 * width * group))
  if raw is not None:
    if raw_safe and (raw.strides[1] == 1 if side == "A" else raw.strides[0] == 1):
      dense = raw if side == "A" else T.transpose(raw)
      _pack_sme_view(dense, panels, word, group)
    else:
      _pack_sme_producer(logical, panels, destination, word, group, bounded)
  else:
    _pack_sme_producer(logical, panels, destination, word, group, bounded)


@T.func
def _pack_sme_producer(logical, panels, destination, word, group, bounded):
  if bounded:
    if T.view_disjoint(logical, destination):
      _pack_sme_bounded(logical, panels, word, group)
    else:
      _pack_sme_staged(logical, panels, word, group)
  else:
    _pack_sme_staged(logical, panels, word, group)


@T.func
def _pack_sme_bounded(logical, panels, word, group):
  width = sme.streaming_lanes(word)
  chunk = sme.streaming_lanes(panels.dtype)
  # Keep a complete instruction chunk and cap ordinary cache staging at
  # 64 KiB (or one chunk on an unusually large streaming vector length).
  block_k = T.max(chunk, T.min(1024, 65536 // (2 * width) // T.itemsize(panels.dtype)) // chunk * chunk)
  dense = T.alloc((2 * width, block_k), dtype=panels.dtype)
  for panel in T.serial(panels.shape[0], unroll=1):
    rows = T.min(2 * width, logical.shape[0] - panel * 2 * width)
    for block in T.serial((logical.shape[1] + block_k - 1) // block_k, unroll=1):
      begin = block * block_k
      count = T.min(block_k, logical.shape[1] - begin)
      T.store(dense[:rows, :count], logical[panel * 2 * width:panel * 2 * width + rows, begin:begin + count])
      with sme.region():
        pack_width = sme.streaming_lanes(word)
        pack_chunk = sme.streaming_lanes(panels.dtype)
        for ki in T.serial(count // pack_chunk, unroll=1):
          pack_fringe(dense, panels[panel, :, :], origin=0,
                      k_origin=ki * pack_chunk, width=pack_width,
                      active_k=pack_chunk, spatial=rows,
                      output_origin=(begin + ki * pack_chunk) // group, bounded_rows=True)
        if count % pack_chunk:
          pack_fringe(dense, panels[panel, :, :], origin=0,
                      k_origin=count - count % pack_chunk, width=pack_width,
                      active_k=count % pack_chunk, spatial=rows,
                      output_origin=(begin + count - count % pack_chunk) // group, bounded_rows=True)


@T.func
def _pack_sme_staged(logical, panels, word, group):
  dense = T.alloc(logical.shape, dtype=panels.dtype)
  T.store(dense, logical)
  _pack_sme_view(dense, panels, word, group)


@T.func
def _pack_sme_view(dense, panels, word, group):
  spatial, reduction = dense.shape
  with sme.region():
    pack_width = sme.streaming_lanes(word)
    chunk = sme.streaming_lanes(panels.dtype)
    for panel in T.serial(panels.shape[0], unroll=1):
      # Full chunks have constant group counts. Keeping the terminal chunk
      # separate also preserves the destination group's exact upper bound
      # in hardware preflight, without combining unrelated loop extrema.
      for ki in T.serial(reduction // chunk, unroll=1):
        pack_fringe(dense, panels[panel, :, :], origin=panel * 2 * pack_width,
                    k_origin=ki * chunk, width=pack_width,
                    active_k=chunk,
                    spatial=spatial, output_origin=ki * chunk // group, bounded_rows=True)
      if reduction % chunk:
        pack_fringe(dense, panels[panel, :, :], origin=panel * 2 * pack_width,
                    k_origin=reduction - reduction % chunk, width=pack_width,
                    active_k=reduction % chunk, spatial=spatial,
                    output_origin=(reduction - reduction % chunk) // group, bounded_rows=True)


@T.func
def pack_amx(source, destination, *, side, raw=None):
  """Construct public 16-row, 64-byte AMX panels with safe logical staging."""
  source = T.as_value(source)
  if side == "B":
    if T.view_disjoint(source, destination):
      _pack_amx_b(source, destination, raw)
    else:
      # An overlapping caller needs complete preparation before its first
      # write. Keep one staged alias path rather than duplicating all native
      # raw-layout alternatives in the same generated function.
      panels = prepare_b(source, "float32")
      T.store(T.reshape(destination, panels.shape), T.load(panels))
  else:
    if T.view_disjoint(source, destination):
      _pack_amx_a(source, destination)
    else:
      dense = T.alloc(source.shape, dtype=source.dtype)
      T.store(dense, source)
      _pack_amx_a(T.load(dense), destination)


@T.func
def _pack_amx_b(source, destination, raw):
  kr = 64 // T.itemsize(destination.dtype)
  group = 4 // T.itemsize(destination.dtype)
  kp = (source.shape[0] + kr - 1) // kr * kr
  panels = T.reshape(destination, ((source.shape[1] + 15) // 16, kp // kr, 16, 16 * group))
  if raw is not None:
    if source.shape[1] % 16 == 0 and source.shape[0] % kr == 0:
      if raw.strides[0] == 1:
        _pack_b_full_k(source, panels, kr)
      elif raw.strides[1] == 1:
        _pack_b_full(source, panels, kp, kr, group)
      else:
        _pack_b_fringe(source, panels, kp, kr, raw)
    else:
      _pack_b_fringe(source, panels, kp, kr, raw)
  else:
    _pack_b_fringe(source, panels, kp, kr, None)


@T.func
def _pack_amx_a(source, destination):
  kr = 64 // T.itemsize(destination.dtype)
  panels = T.reshape(destination, ((source.shape[0] + 15) // 16,
                                   (source.shape[1] + kr - 1) // kr, 16, kr))
  if source.shape[0] % 16 != 0 or source.shape[1] % kr != 0:
    T.store(destination, T.zeros(destination.shape, dtype=destination.dtype))
  for row in T.serial(source.shape[0], unroll=1):
    for ki in T.serial(panels.shape[1], unroll=1):
      cols = T.min(kr, source.shape[1] - ki * kr)
      value = T.as_vector(T.reshape(source[row, ki * kr:ki * kr + cols], (cols,)), T.Vector(source.dtype, N=kr))
      T.store(panels[row // 16, ki, row % 16, :cols], value)


@T.func
def pack_generic(source, destination, *, side, format):
  """Zero padding before writing logical elements, never evaluate padded DAGs.

  Physical ABI offsets are metadata primitives. Traversal and input
  transformations are source DSL. This conservative route deliberately
  favors correctness for unsupported ISA/type/layout combinations; optimized
  register/ZA recipes can replace its schedule without a native loop printer.
  The checked adapter supplies exactly the required physical prefix.
  """
  source = T.as_value(source)
  if format == "dense":
    if side == "A":
      T.store(T.reshape(destination, source.shape), source)
    else:
      T.store(T.reshape(destination, (source.shape[1], source.shape[0])), T.transpose(source))
  else:
    T.store(destination, T.zeros(destination.shape, dtype=destination.dtype))
    spatial = source.shape[0] if side == "A" else source.shape[1]
    reduction = source.shape[1] if side == "A" else source.shape[0]
    for s in T.serial(spatial, unroll=1):
      for k in T.serial(reduction, unroll=2):
        offset = T._packing_offset(s, k, spatial, reduction, destination.dtype, format, side)
        if side == "A":
          value = T.element_at(source, (s, k))
        else:
          value = T.element_at(source, (k, s))
        T.store(destination[offset], value)


@T.func
def pack_b_compensated(source, destination, correction, *, zero_point, format):
  """Pack byte B and write its N-element int32 zero-point correction.

  ``source`` is the already converted logical [K,N] byte value. ``format``
  is explicit ``amx`` or ``sme``. The correction is -zero_point * sum(B),
  computed modulo 2**32 then interpreted as int32. ``zero_point`` is a
  scalar in the int32 range. Destination padding is
  zero, excess capacity is untouched, and the two outputs must be disjoint.
  Disjoint inputs use bounded byte panels; overlapping inputs are captured
  before either output is written. Input conversion is evaluated once.
  """
  source = T.as_value(source)
  T.static_assert(source.dtype == "int8" or source.dtype == "uint8")
  T.static_assert(destination.dtype == source.dtype)
  T.static_assert(correction.dtype == "int32")
  T.static_assert(len(source.shape) == 2 and len(destination.shape) == 1 and len(correction.shape) == 1)
  T.static_assert(format == "amx" or format == "sme")
  assert correction.shape[0] == source.shape[1], "compensation sidecar length mismatch"
  assert destination.strides[0] == 1, "packed output must be dense"
  assert T.view_disjoint(destination, correction), "packed output and compensation may not overlap"
  span = 16 if format == "amx" else 2 * sme.streaming_lanes("int32")
  quantum = 64 if format == "amx" else 4
  panels = (source.shape[1] + span - 1) // span
  groups = (source.shape[0] + quantum - 1) // quantum
  # Divide before multiplying so a caller cannot wrap the capacity check.
  assert groups <= 9223372036854775807 // span // quantum, "packed shape overflow"
  panel_size = groups * span * quantum
  assert panels <= 9223372036854775807 // T.max(1, panel_size), "packed shape overflow"
  assert destination.shape[0] >= panels * panel_size, "packed output buffer too small"
  assert zero_point >= -2147483648 and zero_point <= 2147483647, "zero point exceeds int32 range"
  zp = T.bitcast(T.cast(zero_point, "int32"), "uint32")
  if T.view_disjoint(source, destination) and T.view_disjoint(source, correction):
    _pack_b_compensated_panels(source, destination, correction, zp, format, span, quantum, panel_size)
  else:
    captured = T.alloc(source.shape, dtype=source.dtype)
    T.store(captured, source)
    _pack_b_compensated_panels(T.load(captured), destination, correction, zp, format, span, quantum, panel_size)


@T.func
def _pack_b_compensated_panels(source, destination, correction, zp, format, span, quantum, panel_size):
  # Byte staging is at most 256 x one physical spatial panel. The same
  # converted bytes feed packing and sum; no K*N wide conversion buffer.
  block_k = 256
  dense = T.alloc((span, block_k), dtype=source.dtype)
  sums = T.alloc((span,), dtype="uint32")
  for panel in T.serial((source.shape[1] + span - 1) // span, unroll=1):
    first = panel * span
    columns = T.min(span, source.shape[1] - first)
    T.store(sums, T.zeros((span,), dtype="uint32"))
    for block in T.serial((source.shape[0] + block_k - 1) // block_k, unroll=1):
      begin = block * block_k
      count = T.min(block_k, source.shape[0] - begin)
      # Spatial-major staging shares the packer's contiguous K words with
      # the column reduction. Passing its transposed View retains this layout
      # proof for the AMX register transpose and the SME panel packer.
      T.store(dense[:columns, :count], T.transpose(source[begin:begin + count, first:first + columns]))
      value = T.load(dense[:columns, :count])
      partial = T.reduce_add(T.bitcast(T.cast(value, "int32"), "uint32"), axis=1)
      T.store(sums[:columns], T.load(sums[:columns]) + partial)
      offset = panel * panel_size + begin * span
      size = ((count + quantum - 1) // quantum) * quantum * span
      T.pack(T.transpose(dense[:columns, :count]), destination[offset:offset + size], side="B", format=format)
    result = T.bitcast(T.cast(0, "uint32") - T.load(sums[:columns]) * zp, "int32")
    T.store(correction[first:first + columns], result)
