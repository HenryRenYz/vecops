# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""AMX source recipes: typed TMUL, one tile owner, padded physical panels.

A is row-major [M,K]; B uses architectural dword K groups. The 16x64-byte
tile geometry and signedness selection match vecops' public AMX primitives.
Packing/prologues and output conversion/epilogues remain ordinary DSL.
"""

from .. import amx
from .. import dsl as T
from .matrix_recipe import finish_matrix, seed_accumulator
from .registers import pack_interleave, transpose_dwords_512


@T.func
def _repair_a_tail(source, destination, reduction, kr):
  # Only the final physical K tile is privately repaired; full K tiles keep
  # their checked external backing and are never copied into scratch.
  T.store(destination, T.load(source[:, source.shape[1] - 1 :, :, :]))
  for panel in T.serial(source.shape[0], unroll=1):
    padding = destination[panel, 0, :, reduction % kr :]
    T.store(padding, T.zeros(padding.shape, dtype=source.dtype))


@T.func
def _repair_b_tail(source, destination, reduction, kr, group):
  # The B panel is already VNNI-interleaved. Repair the last K tile in that
  # layout; do not decode it to dense storage and transpose it back again.
  # where selects zero without arithmetic on untrusted NaN/Inf padding.
  T.store(destination, T.load(source[:, source.shape[1] - 1 :, :, :]))
  offsets = T.cast(T.arange(16 * group), "uint8") & (group - 1)
  for panel in T.serial(source.shape[0], unroll=1):
    # Complete valid dwords are already copied. Only one dword row needs a
    # lane selection; later rows are one contiguous zero-fill, not repeated
    # loads/index vectors/selects of every row in the final instruction tile.
    padding = destination[panel, 0, (reduction % kr + group - 1) // group :, :]
    T.store(padding, T.zeros(padding.shape, dtype=source.dtype))
    if reduction % group != 0:
      ki = reduction % kr // group
      tail = T.load(source[panel, source.shape[1] - 1, ki, :])
      T.store(destination[panel, 0, ki, :], T.where(offsets < T.cast(reduction % group, "uint8"), tail, 0))


@T.func
def _load_a(a, tile, mi, ki, kr):
  if len(a.shape) == 4:
    value = amx.load(tile, a[mi // 16, ki, :, :])
  else:
    value = amx.load(tile, a[mi : mi + 16, ki * kr : ki * kr + kr])
  return value


@T.func
def _tiles(a, b, output, *, initial, kr, tail_a=None, tail_b=None, stream_b=False, paired_fringe=False):
  if output.shape[0] >= 32 and output.shape[1] >= 32:
    _tiles_2x2(
      a, b, output, initial=initial, kr=kr, tail_a=tail_a, tail_b=tail_b, stream_b=stream_b, paired_fringe=paired_fringe
    )
  else:
    _tiles_1x1(a, b, output, initial=initial, kr=kr, tail_a=tail_a, tail_b=tail_b, stream_b=stream_b)


@T.func
def _tiles_2x2(a, b, output, *, initial, kr, tail_a=None, tail_b=None, stream_b=False, paired_fringe=False):
  # Four C, two A and two B registers: reuse each loaded operand twice.
  with amx.region(((16, 64), (16, 64), (16, 64), (16, 64), (16, 64), (16, 64), (16, 64), (16, 64))):
    for mi in T.serial(output.shape[0] // 32, unroll=1):
      for ni in T.serial(output.shape[1] // 32, unroll=1):
        if initial:
          c0 = amx.load(0, output[mi * 32 : mi * 32 + 16, ni * 32 : ni * 32 + 16])
          c1 = amx.load(1, output[mi * 32 : mi * 32 + 16, ni * 32 + 16 : ni * 32 + 32])
          c2 = amx.load(2, output[mi * 32 + 16 : mi * 32 + 32, ni * 32 : ni * 32 + 16])
          c3 = amx.load(3, output[mi * 32 + 16 : mi * 32 + 32, ni * 32 + 16 : ni * 32 + 32])
        else:
          c0 = amx.zero(0, dtype=output.dtype)
          c1 = amx.zero(1, dtype=output.dtype)
          c2 = amx.zero(2, dtype=output.dtype)
          c3 = amx.zero(3, dtype=output.dtype)
        for ki in T.serial(b.shape[1] - (1 if tail_a is not None or tail_b is not None else 0), unroll=2):
          a0 = _load_a(a, 4, mi * 32, ki, kr)
          a1 = _load_a(a, 5, mi * 32 + 16, ki, kr)
          if stream_b:
            b0 = amx.stream_load(6, b[ni * 2, ki, :, :])
          else:
            b0 = amx.load(6, b[ni * 2, ki, :, :])
          if stream_b:
            b1 = amx.stream_load(7, b[ni * 2 + 1, ki, :, :])
          else:
            b1 = amx.load(7, b[ni * 2 + 1, ki, :, :])
          amx.tdp(c0, a0, b0)
          amx.tdp(c1, a0, b1)
          amx.tdp(c2, a1, b0)
          amx.tdp(c3, a1, b1)
        if tail_a is not None or tail_b is not None:
          a0 = _load_a(tail_a if tail_a is not None else a, 4, mi * 32, 0 if tail_a is not None else b.shape[1] - 1, kr)
          a1 = _load_a(
            tail_a if tail_a is not None else a, 5, mi * 32 + 16, 0 if tail_a is not None else b.shape[1] - 1, kr
          )
          if stream_b:
            b0 = amx.stream_load(6, tail_b[ni * 2, 0, :, :] if tail_b is not None else b[ni * 2, b.shape[1] - 1, :, :])
          else:
            b0 = amx.load(6, tail_b[ni * 2, 0, :, :] if tail_b is not None else b[ni * 2, b.shape[1] - 1, :, :])
          if stream_b:
            b1 = amx.stream_load(
              7, tail_b[ni * 2 + 1, 0, :, :] if tail_b is not None else b[ni * 2 + 1, b.shape[1] - 1, :, :]
            )
          else:
            b1 = amx.load(7, tail_b[ni * 2 + 1, 0, :, :] if tail_b is not None else b[ni * 2 + 1, b.shape[1] - 1, :, :])
          amx.tdp(c0, a0, b0)
          amx.tdp(c1, a0, b1)
          amx.tdp(c2, a1, b0)
          amx.tdp(c3, a1, b1)
        amx.store(c0, output[mi * 32 : mi * 32 + 16, ni * 32 : ni * 32 + 16])
        amx.store(c1, output[mi * 32 : mi * 32 + 16, ni * 32 + 16 : ni * 32 + 32])
        amx.store(c2, output[mi * 32 + 16 : mi * 32 + 32, ni * 32 : ni * 32 + 16])
        amx.store(c3, output[mi * 32 + 16 : mi * 32 + 32, ni * 32 + 16 : ni * 32 + 32])
    # Pair the fringe's long dimension to share the other operand load.
    if paired_fringe:
      for mi in T.serial(0, output.shape[0] // 32 * 32, 32, unroll=1):
        for ni in T.serial(output.shape[1] // 32 * 32, output.shape[1], 16, unroll=1):
          _tile_pair(
            a, b, output, mi, ni, rows=True, initial=initial, kr=kr, tail_a=tail_a, tail_b=tail_b, stream_b=stream_b
          )
      for mi in T.serial(output.shape[0] // 32 * 32, output.shape[0], 16, unroll=1):
        for ni in T.serial(0, output.shape[1] // 32 * 32, 32, unroll=1):
          _tile_pair(
            a, b, output, mi, ni, rows=False, initial=initial, kr=kr, tail_a=tail_a, tail_b=tail_b, stream_b=stream_b
          )
        for ni in T.serial(output.shape[1] // 32 * 32, output.shape[1], 16, unroll=1):
          _tile_1x1(a, b, output, mi, ni, initial=initial, kr=kr, tail_a=tail_a, tail_b=tail_b, stream_b=stream_b)
    else:
      for mi in T.serial(0, output.shape[0] // 32 * 32, 16, unroll=1):
        for ni in T.serial(output.shape[1] // 32 * 32, output.shape[1], 16, unroll=1):
          _tile_1x1(a, b, output, mi, ni, initial=initial, kr=kr, tail_a=tail_a, tail_b=tail_b, stream_b=stream_b)
      for mi in T.serial(output.shape[0] // 32 * 32, output.shape[0], 16, unroll=1):
        for ni in T.serial(0, output.shape[1], 16, unroll=1):
          _tile_1x1(a, b, output, mi, ni, initial=initial, kr=kr, tail_a=tail_a, tail_b=tail_b, stream_b=stream_b)


@T.func
def _tile_pair(a, b, output, mi, ni, *, rows, initial, kr, tail_a=None, tail_b=None, stream_b=False):
  """Compute a 2x1 or 1x2 fringe inside the existing eight-tile owner."""
  mi1, ni1 = mi + (16 if rows else 0), ni + (0 if rows else 16)
  if initial:
    c0 = amx.load(0, output[mi : mi + 16, ni : ni + 16])
    c1 = amx.load(2 if rows else 1, output[mi1 : mi1 + 16, ni1 : ni1 + 16])
  else:
    c0 = amx.zero(0, dtype=output.dtype)
    c1 = amx.zero(2 if rows else 1, dtype=output.dtype)
  for ki in T.serial(b.shape[1] - (1 if tail_a is not None or tail_b is not None else 0), unroll=2):
    a0 = _load_a(a, 4, mi, ki, kr)
    if rows:
      a1 = _load_a(a, 5, mi1, ki, kr)
    if stream_b:
      b0 = amx.stream_load(6, b[ni // 16, ki, :, :])
    else:
      b0 = amx.load(6, b[ni // 16, ki, :, :])
    if not rows:
      if stream_b:
        b1 = amx.stream_load(7, b[ni1 // 16, ki, :, :])
      else:
        b1 = amx.load(7, b[ni1 // 16, ki, :, :])
    amx.tdp(c0, a0, b0)
    if rows:
      amx.tdp(c1, a1, b0)
    else:
      amx.tdp(c1, a0, b1)
  if tail_a is not None or tail_b is not None:
    a0 = _load_a(tail_a if tail_a is not None else a, 4, mi, 0 if tail_a is not None else b.shape[1] - 1, kr)
    if rows:
      a1 = _load_a(tail_a if tail_a is not None else a, 5, mi1, 0 if tail_a is not None else b.shape[1] - 1, kr)
    if stream_b:
      b0 = amx.stream_load(6, tail_b[ni // 16, 0, :, :] if tail_b is not None else b[ni // 16, b.shape[1] - 1, :, :])
    else:
      b0 = amx.load(6, tail_b[ni // 16, 0, :, :] if tail_b is not None else b[ni // 16, b.shape[1] - 1, :, :])
    if not rows:
      if stream_b:
        b1 = amx.stream_load(
          7, tail_b[ni1 // 16, 0, :, :] if tail_b is not None else b[ni1 // 16, b.shape[1] - 1, :, :]
        )
      else:
        b1 = amx.load(7, tail_b[ni1 // 16, 0, :, :] if tail_b is not None else b[ni1 // 16, b.shape[1] - 1, :, :])
    amx.tdp(c0, a0, b0)
    if rows:
      amx.tdp(c1, a1, b0)
    else:
      amx.tdp(c1, a0, b1)
  amx.store(c0, output[mi : mi + 16, ni : ni + 16])
  amx.store(c1, output[mi1 : mi1 + 16, ni1 : ni1 + 16])


@T.func
def _tile_1x1(a, b, output, mi, ni, *, initial, kr, tail_a=None, tail_b=None, stream_b=False):
  if initial:
    accumulator = amx.load(0, output[mi : mi + 16, ni : ni + 16])
  else:
    accumulator = amx.zero(0, dtype=output.dtype)
  for ki in T.serial(b.shape[1] - (1 if tail_a is not None or tail_b is not None else 0), unroll=2):
    left = _load_a(a, 1, mi, ki, kr)
    if stream_b:
      right = amx.stream_load(2, b[ni // 16, ki, :, :])
    else:
      right = amx.load(2, b[ni // 16, ki, :, :])
    amx.tdp(accumulator, left, right)
  if tail_a is not None or tail_b is not None:
    left = _load_a(tail_a if tail_a is not None else a, 1, mi, 0 if tail_a is not None else b.shape[1] - 1, kr)
    if stream_b:
      right = amx.stream_load(2, tail_b[ni // 16, 0, :, :] if tail_b is not None else b[ni // 16, b.shape[1] - 1, :, :])
    else:
      right = amx.load(2, tail_b[ni // 16, 0, :, :] if tail_b is not None else b[ni // 16, b.shape[1] - 1, :, :])
    amx.tdp(accumulator, left, right)
  amx.store(accumulator, output[mi : mi + 16, ni : ni + 16])


@T.func
def _tiles_1x1(a, b, output, *, initial, kr, tail_a=None, tail_b=None, stream_b=False):
  with amx.region(((16, 64), (16, 64), (16, 64))):
    for mi in T.serial(0, output.shape[0], 16, unroll=1):
      for ni in T.serial(0, output.shape[1], 16, unroll=1):
        _tile_1x1(a, b, output, mi, ni, initial=initial, kr=kr, tail_a=tail_a, tail_b=tail_b, stream_b=stream_b)


@T.func
def prepare_b(value, acc_dtype, raw=None):
  """Prepare private native B panels for one logical [K,N] cache phase.

  Reuse is valid across M blocks only: the consumer must retain this exact
  logical K/N domain, dtype and physical padding. Producers are evaluated
  only inside their logical bounds; absent K/N storage lanes are zero-filled.
  Empty axes, including K==0, produce correspondingly empty physical panels.
  ``raw`` is only a layout proof; conversion/arithmetic still uses ``value``.
  """
  b = T.as_value(value)
  T.static_assert(T.itemsize(acc_dtype) == 4, "AMX accumulators require four-byte elements")
  kr = 64 // T.itemsize(b.dtype)
  group = 4 // T.itemsize(b.dtype)
  assert b.shape[1] <= 9223372036854775792, "AMX spatial extent overflow"
  assert b.shape[0] <= 9223372036854775807 - (kr - 1), "AMX reduction extent overflow"
  np = (b.shape[1] + 15) // 16 * 16
  kp = (b.shape[0] + kr - 1) // kr * kr
  panels = T.alloc((np // 16, kp // kr, 16, 16 * group), dtype=b.dtype)
  if raw is not None:
    if b.shape[1] % 16 == 0 and b.shape[0] % kr == 0:
      if raw.strides[0] == 1:
        _pack_b_full_k(b, panels, kr)
      elif raw.strides[1] == 1:
        _pack_b_full(b, panels, kp, kr, group)
      else:
        _pack_b_fringe(b, panels, kp, kr, raw)
    else:
      _pack_b_fringe(b, panels, kp, kr, raw)
  else:
    # Preserve the existing no-layout-proof packing and its conversion
    # boundaries; rank-three transpose alone can scalarize this permutation.
    dense = T.alloc((np, kp), dtype=b.dtype)
    if b.shape[1] != np or b.shape[0] != kp:
      T.store(dense, T.zeros(dense.shape, dtype=b.dtype))
    T.store(dense[: b.shape[1], : b.shape[0]], T.transpose(b))
    dense_rows = T.reshape(dense, (np, kp // kr, kr))
    for ni in T.serial(np // 16, unroll=1):
      for ki in T.serial(kp // kr, unroll=1):
        rows = ()
        for row in T.unroll(16):
          word = T.as_vector(T.load(dense_rows[ni * 16 + row, ki, :]), T.Vector(b.dtype, N=kr))
          rows = rows + (T.bitcast(word, "uint32"),)
        columns = transpose_dwords_512(rows)
        for column in T.unroll(16):
          T.store(panels[ni, ki, column, :], T.bitcast(columns[column], b.dtype))
  return panels


@T.func
def dot_into(
  inputs,
  output,
  *,
  acc=None,
  acc_dtype="float32",
  epilogue=None,
  raw_a=None,
  raw_b=None,
  packed_a=None,
  packed_b=None,
  prepared_b=None,
  prepared_a=None,
  stream_b=False,
  paired_fringe=False,
):
  """Compute using optional physical panels for the identical logical phase.

  Prepared panels retain actual physical strides. They contain valid zero
  padding, or represent checked external subviews with complete K tiles,
  and bypass external
  packed-buffer repair. External fractional-K declarations retain their
  full tiles in place and privately repair only the terminal instruction
  tile; the caller's padding is never changed or used in arithmetic.
  """
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  m, n, k = a.shape[0], b.shape[1], a.shape[1]
  if m > 0 and n > 0:
    kr = 64 // T.itemsize(a.dtype)
    group = 4 // T.itemsize(b.dtype)
    assert m <= 9223372036854775792 and n <= 9223372036854775792, "AMX spatial extent overflow"
    assert k <= 9223372036854775807 - (kr - 1), "AMX reduction extent overflow"
    mp, np, kp = (m + 15) // 16 * 16, (n + 15) // 16 * 16, (k + kr - 1) // kr * kr
    if prepared_a is not None:
      T.static_assert(
        packed_a is None and len(prepared_a.shape) == 4 and prepared_a.dtype == a.dtype,
        "prepared AMX A panels require rank four and matching dtype",
      )
      assert (
        prepared_a.shape[0] == mp // 16
        and prepared_a.shape[1] == kp // kr
        and prepared_a.shape[2] == 16
        and prepared_a.shape[3] == kr
      ), "prepared AMX A panel shape mismatch"
      _prepared_a_input_b(
        prepared_a,
        b,
        prepared_b,
        packed_b,
        raw_b,
        output,
        acc,
        acc_dtype,
        epilogue,
        mp,
        np,
        kp,
        kr,
        group,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )
    elif prepared_b is not None:
      T.static_assert(
        len(prepared_b.shape) == 4 and prepared_b.dtype == b.dtype,
        "prepared AMX B panels require rank four and matching dtype",
      )
      assert (
        prepared_b.shape[0] == np // 16
        and prepared_b.shape[1] == kp // kr
        and prepared_b.shape[2] == 16
        and prepared_b.shape[3] == 16 * group
      ), "prepared AMX B panel shape mismatch"
      _input_a(
        a,
        raw_a,
        prepared_b,
        output,
        acc,
        acc_dtype,
        epilogue,
        mp,
        np,
        kp,
        kr,
        packed_a,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )
    elif packed_b is not None:
      # A full K domain never consumes padding from an external declaration.
      # Fractional K privately repairs the final tile in its physical layout.
      if k % kr == 0:
        count = np * kp
        panels = T.reshape(packed_b[:count], (np // 16, kp // kr, 16, 16 * group))
        _input_a(
          a,
          raw_a,
          panels,
          output,
          acc,
          acc_dtype,
          epilogue,
          mp,
          np,
          kp,
          kr,
          packed_a,
          stream_b=stream_b,
          paired_fringe=paired_fringe,
        )
      else:
        count = np * kp
        source = T.reshape(packed_b[:count], (np // 16, kp // kr, 16, 16 * group))
        repaired_b = T.alloc((source.shape[0], 1, source.shape[2], source.shape[3]), dtype=b.dtype)
        _repair_b_tail(source, repaired_b, k, kr, group)
        _input_a(
          a,
          raw_a,
          source,
          output,
          acc,
          acc_dtype,
          epilogue,
          mp,
          np,
          kp,
          kr,
          packed_a,
          tail_b=repaired_b,
          stream_b=stream_b,
          paired_fringe=paired_fringe,
        )
    else:
      _input_b(
        a,
        b,
        raw_a,
        output,
        acc,
        acc_dtype,
        epilogue,
        mp,
        np,
        kp,
        kr,
        group,
        packed_a,
        raw_b,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )


@T.func
def _prepared_a_input_b(
  left,
  b,
  prepared_b,
  packed_b,
  raw_b,
  output,
  acc,
  dtype,
  epilogue,
  mp,
  np,
  kp,
  kr,
  group,
  stream_b=False,
  paired_fringe=False,
):
  if prepared_b is not None:
    T.static_assert(packed_b is None and len(prepared_b.shape) == 4 and prepared_b.dtype == b.dtype)
    assert (
      prepared_b.shape[0] == np // 16
      and prepared_b.shape[1] == kp // kr
      and prepared_b.shape[2] == 16
      and prepared_b.shape[3] == 16 * group
    ), "prepared AMX B panel shape mismatch"
    _output(left, prepared_b, output, acc, dtype, epilogue, mp, np, kr, stream_b=stream_b, paired_fringe=paired_fringe)
  elif packed_b is not None:
    source = T.reshape(packed_b[: np * kp], (np // 16, kp // kr, 16, 16 * group))
    if b.shape[0] % kr == 0:
      _output(left, source, output, acc, dtype, epilogue, mp, np, kr, stream_b=stream_b, paired_fringe=paired_fringe)
    else:
      tail = T.alloc((source.shape[0], 1, source.shape[2], source.shape[3]), dtype=b.dtype)
      _repair_b_tail(source, tail, b.shape[0], kr, group)
      _output(
        left,
        source,
        output,
        acc,
        dtype,
        epilogue,
        mp,
        np,
        kr,
        tail_b=tail,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )
  else:
    panels = prepare_b(b, dtype, raw=raw_b)
    _output(left, panels, output, acc, dtype, epilogue, mp, np, kr, stream_b=stream_b, paired_fringe=paired_fringe)


@T.func
def _input_b(
  a, b, raw_a, output, acc, dtype, epilogue, mp, np, kp, kr, group, packed_a, raw_b, stream_b=False, paired_fringe=False
):
  if raw_b is None:
    _input_b_padded(
      a,
      b,
      raw_a,
      output,
      acc,
      dtype,
      epilogue,
      mp,
      np,
      kp,
      kr,
      group,
      packed_a,
      stream_b=stream_b,
      paired_fringe=paired_fringe,
    )
  else:
    _input_b_proven_layout(
      a,
      b,
      raw_a,
      output,
      acc,
      dtype,
      epilogue,
      mp,
      np,
      kp,
      kr,
      group,
      packed_a,
      raw_b,
      stream_b=stream_b,
      paired_fringe=paired_fringe,
    )


@T.func
def _input_b_proven_layout(
  a, b, raw_a, output, acc, dtype, epilogue, mp, np, kp, kr, group, packed_a, raw_b, stream_b=False, paired_fringe=False
):
  # Keep packing alternatives separate from their common consumer. Duplicating
  # _input_a/_output in each layout branch inflated Dynamic kernels' CFG and
  # stack even when only one branch executed. Complete panels read producers
  # directly into one shared permutation DAG, including boundary conversion.
  panels = T.alloc((np // 16, kp // kr, 16, 16 * group), dtype=b.dtype)
  if raw_b is not None:
    if b.shape[1] % 16 == 0 and b.shape[0] % kr == 0:
      if raw_b.strides[0] == 1:
        _pack_b_full_k(b, panels, kr)
      elif raw_b.strides[1] == 1:
        _pack_b_full(b, panels, kp, kr, group)
      else:
        _pack_b_fringe(b, panels, kp, kr, raw_b)
    else:
      _pack_b_fringe(b, panels, kp, kr, raw_b)
  else:
    _pack_b_fringe(b, panels, kp, kr, raw_b)
  _input_a(
    a,
    raw_a,
    panels,
    output,
    acc,
    dtype,
    epilogue,
    mp,
    np,
    kp,
    kr,
    packed_a,
    stream_b=stream_b,
    paired_fringe=paired_fringe,
  )


@T.func
def _pack_b_full(b, panels, kp, kr, group):
  # A complete spatial word from each member of one K dword group.
  for ni in T.serial(panels.shape[0], unroll=1):
    for kg in T.serial(kp // group, unroll=1):
      columns = ()
      for member in T.unroll(group):
        word = T.reshape(b[kg * group + member, ni * 16 : ni * 16 + 16], (16,))
        value = T.as_vector(word, T.Vector(b.dtype, N=16))
        columns = columns + (value,)
      T.store(panels[ni, kg // 16, kg % 16, :], pack_interleave(columns)[0])


@T.func
def _pack_b_full_k(b, panels, kr):
  # Each producer is evaluated once in the combined 16-output register DAG.
  # A fixed one-word destination also bounds converting inputs to two FP32
  # words, avoiding the unused wide aggregates of a generic row tail.
  for ni in T.serial(panels.shape[0], unroll=1):
    for ki in T.serial(panels.shape[1], unroll=1):
      rows = ()
      for row in T.unroll(16):
        word = T.reshape(b[ki * kr : ki * kr + kr, ni * 16 + row], (kr,))
        value = T.as_vector(word, T.Vector(b.dtype, N=kr))
        rows = rows + (T.bitcast(value, "uint32"),)
      transposed = transpose_dwords_512(rows)
      for column in T.unroll(16):
        T.store(panels[ni, ki, column, :], T.bitcast(transposed[column], b.dtype))


@T.func
def _stage_b_k_rows(b, dense, ni, ki, kr, columns, reduction):
  # Explicit one-word vectors avoid constructing a two-word BF16/four-word
  # FP32 tail for a row that is at most one physical AMX K word. All source
  # coordinates remain inside the logical domain; the fringe fill supplies
  # every absent storage element before the bit-level transpose.
  for row in T.serial(columns, unroll=1):
    word = T.reshape(b[ki * kr : ki * kr + kr, ni * 16 + row], (reduction,))
    value = T.as_vector(word, T.Vector(b.dtype, N=kr))
    T.store(dense[row, :reduction], value)


@T.func
def _pack_b_fringe(b, panels, kp, kr, raw_b):
  # A bounded tile cache retains producer rounding and avoids a full NxK
  # dense transpose. Preserve the register transpose for spatial-contiguous
  # inputs instead of turning their rows into K-strided gathers.
  if kp > 0:
    dense = T.alloc((16, kr), dtype=b.dtype)
    for ni in T.serial(panels.shape[0], unroll=1):
      for ki in T.serial(panels.shape[1], unroll=1):
        columns = T.min(16, b.shape[1] - ni * 16)
        reduction = T.min(kr, b.shape[0] - ki * kr)
        if columns != 16 or reduction != kr:
          T.store(dense, T.zeros(dense.shape, dtype=b.dtype))
        if raw_b is None:
          T.store(dense[:columns, :reduction], T.transpose(b[ki * kr : ki * kr + kr, ni * 16 : ni * 16 + 16]))
        elif raw_b.strides[0] == 1:
          _stage_b_k_rows(b, dense, ni, ki, kr, columns, reduction)
        else:
          T.store(dense[:columns, :reduction], T.transpose(b[ki * kr : ki * kr + kr, ni * 16 : ni * 16 + 16]))
        rows = ()
        for row in T.unroll(16):
          value = T.as_vector(T.load(dense[row, :]), T.Vector(b.dtype, N=kr))
          rows = rows + (T.bitcast(value, "uint32"),)
        transposed = transpose_dwords_512(rows)
        for column in T.unroll(16):
          T.store(panels[ni, ki, column, :], T.bitcast(transposed[column], b.dtype))


@T.func
def _input_b_padded(
  a, b, raw_a, output, acc, dtype, epilogue, mp, np, kp, kr, group, packed_a, stream_b=False, paired_fringe=False
):
  # Reinterpret each spatial row's K groups as 16 dwords, then transpose
  # sixteen registers. This is the historical DSL/C++ AMX packing schedule:
  # ordinary rank-three reshape/transpose falls back to costly scalar gathers.
  dense = T.alloc((np, kp), dtype=b.dtype)
  if b.shape[1] != np or b.shape[0] != kp:
    T.store(dense, T.zeros(dense.shape, dtype=b.dtype))
  T.store(dense[: b.shape[1], : b.shape[0]], T.transpose(b))
  dense_rows = T.reshape(dense, (np, kp // kr, kr))
  panels = T.alloc((np // 16, kp // kr, 16, 16 * group), dtype=b.dtype)
  for ni in T.serial(np // 16, unroll=1):
    for ki in T.serial(kp // kr, unroll=1):
      rows = ()
      for row in T.unroll(16):
        value = T.as_vector(T.load(dense_rows[ni * 16 + row, ki, :]), T.Vector(b.dtype, N=kr))
        rows = rows + (T.bitcast(value, "uint32"),)
      columns = transpose_dwords_512(rows)
      for column in T.unroll(16):
        T.store(panels[ni, ki, column, :], T.bitcast(columns[column], b.dtype))
  _input_a(
    a,
    raw_a,
    panels,
    output,
    acc,
    dtype,
    epilogue,
    mp,
    np,
    kp,
    kr,
    packed_a,
    stream_b=stream_b,
    paired_fringe=paired_fringe,
  )


@T.func
def _input_a(
  a,
  raw_a,
  panels,
  output,
  acc,
  dtype,
  epilogue,
  mp,
  np,
  kp,
  kr,
  packed_a,
  tail_b=None,
  stream_b=False,
  paired_fringe=False,
):
  if packed_a is not None:
    source = T.reshape(packed_a[: mp * kp], (mp // 16, kp // kr, 16, kr))
    if a.shape[1] % kr == 0:
      _output(
        source,
        panels,
        output,
        acc,
        dtype,
        epilogue,
        mp,
        np,
        kr,
        tail_b=tail_b,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )
    else:
      left = T.alloc((source.shape[0], 1, source.shape[2], source.shape[3]), dtype=a.dtype)
      _repair_a_tail(source, left, a.shape[1], kr)
      _output(
        source,
        panels,
        output,
        acc,
        dtype,
        epilogue,
        mp,
        np,
        kr,
        tail_a=left,
        tail_b=tail_b,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )
  elif raw_a is not None:
    if a.shape[0] % 16 == 0 and a.shape[1] % kr == 0 and raw_a.strides[1] == 1 and T.view_disjoint(raw_a, output):
      _output(
        raw_a,
        panels,
        output,
        acc,
        dtype,
        epilogue,
        mp,
        np,
        kr,
        tail_b=tail_b,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )
    else:
      _packed_a(
        a,
        panels,
        output,
        acc,
        dtype,
        epilogue,
        mp,
        np,
        kp,
        kr,
        tail_b=tail_b,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )
  else:
    _packed_a(
      a,
      panels,
      output,
      acc,
      dtype,
      epilogue,
      mp,
      np,
      kp,
      kr,
      tail_b=tail_b,
      stream_b=stream_b,
      paired_fringe=paired_fringe,
    )


@T.func
def _packed_a(
  a, panels, output, acc, dtype, epilogue, mp, np, kp, kr, tail_b=None, stream_b=False, paired_fringe=False
):
  left = T.alloc((mp, kp), dtype=a.dtype)
  if a.shape[0] != mp or a.shape[1] != kp:
    T.store(left, T.zeros(left.shape, dtype=left.dtype))
  T.store(left[: a.shape[0], : a.shape[1]], a)
  _output(
    left,
    panels,
    output,
    acc,
    dtype,
    epilogue,
    mp,
    np,
    kr,
    tail_b=tail_b,
    stream_b=stream_b,
    paired_fringe=paired_fringe,
  )


@T.func
def _output(
  left, panels, output, acc, dtype, epilogue, mp, np, kr, tail_a=None, tail_b=None, stream_b=False, paired_fringe=False
):
  if epilogue is None and output.dtype == dtype:
    if output.shape[0] % 16 == 0 and output.shape[1] % 16 == 0 and output.strides[1] == 1:
      if acc is not None:
        T.store(output, T.broadcast_to(T.cast(T.as_value(acc), dtype), output.shape))
      _tiles(
        left,
        panels,
        output,
        initial=acc is not None,
        kr=kr,
        tail_a=tail_a,
        tail_b=tail_b,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )
    else:
      _buffered_output(
        left,
        panels,
        output,
        acc,
        dtype,
        epilogue,
        mp,
        np,
        kr,
        tail_a=tail_a,
        tail_b=tail_b,
        stream_b=stream_b,
        paired_fringe=paired_fringe,
      )
  else:
    _buffered_output(
      left,
      panels,
      output,
      acc,
      dtype,
      epilogue,
      mp,
      np,
      kr,
      tail_a=tail_a,
      tail_b=tail_b,
      stream_b=stream_b,
      paired_fringe=paired_fringe,
    )


@T.func
def _buffered_output(
  left, panels, output, acc, dtype, epilogue, mp, np, kr, tail_a=None, tail_b=None, stream_b=False, paired_fringe=False
):
  result = T.alloc((mp, np), dtype=dtype)
  if acc is not None:
    seed_accumulator(acc, result, output.shape[0], output.shape[1], dtype)
  _tiles(
    left,
    panels,
    result,
    initial=acc is not None,
    kr=kr,
    tail_a=tail_a,
    tail_b=tail_b,
    stream_b=stream_b,
    paired_fringe=paired_fringe,
  )
  finish_matrix(result, output, epilogue=epilogue)


@T.func
def streamed_dot_into(
  inputs,
  output,
  *,
  acc=None,
  acc_dtype="float32",
  epilogue=None,
  raw_a=None,
  raw_b=None,
  packed_a=None,
  packed_b=None,
  prepared_b=None,
  prepared_a=None,
  paired_fringe=False,
):
  """Experimental B-only non-temporal tile loads; ordinary leaf is default.

  The caller selects this policy explicitly. A cache driver immediately
  reusing the same prepared B across M phases should retain normal loads.
  """
  dot_into(
    inputs,
    output,
    acc=acc,
    acc_dtype=acc_dtype,
    epilogue=epilogue,
    raw_a=raw_a,
    raw_b=raw_b,
    packed_a=packed_a,
    packed_b=packed_b,
    prepared_b=prepared_b,
    prepared_a=prepared_a,
    stream_b=True,
    paired_fringe=paired_fringe,
  )


@T.func
def paired_dot_into(
  inputs,
  output,
  *,
  acc=None,
  acc_dtype="float32",
  epilogue=None,
  raw_a=None,
  raw_b=None,
  packed_a=None,
  packed_b=None,
  prepared_b=None,
  prepared_a=None,
  stream_b=False,
  paired_fringe=False,
):
  """Explicit paired fringe experiment; default leaves retain scalar tiles."""
  dot_into(
    inputs,
    output,
    acc=acc,
    acc_dtype=acc_dtype,
    epilogue=epilogue,
    raw_a=raw_a,
    raw_b=raw_b,
    packed_a=packed_a,
    packed_b=packed_b,
    prepared_b=prepared_b,
    prepared_a=prepared_a,
    stream_b=stream_b,
    paired_fringe=True,
  )
