# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Typed SME matrix recipes with grouped K, ZA seeds and fused boundaries.

I8/I32 uses four-byte groups (all sign pairs), I16/I64 four halfwords,
I16/I32 two halfwords, F16/BF16 two halfwords, and F32/F64 one element.
Producer packing, initial accumulators and output epilogues are source DSL;
only typed vector/ZA instructions execute in the no-throw streaming owner.
"""

from .. import dsl as T
from .. import sme
from .matrix_recipe import finish_matrix, grouped_operand, seed_accumulator
from .sme_packing import pack_fringe


@T.func
def _repair_packed_tail(source, destination, reduction, group):
  """Retain physical panels and clear only invalid lanes of the last group.

  External padding has no numerical promise. A physical copy plus one masked
  group avoids decoding all logical elements and running a ZA transpose again.
  Selection performs no arithmetic on discarded NaN/Inf padding.
  """
  T.store(destination, T.load(source))
  # Only the low one/two index bits matter. A modulo-256 byte sequence keeps
  # the same group offsets even for SVL > 128B, and avoids an int64 index
  # vector wider than scalable SVE's four-word limit. Cast the scalar bound
  # too, so numeric promotion does not widen the comparison back to int64.
  valid = ((T.cast(T.arange(source.shape[2]), "uint8") & (group - 1))
           < T.cast(reduction % group, "uint8"))
  for panel in T.serial(source.shape[0], unroll=1):
    tail = T.load(source[panel, source.shape[1] - 1, :])
    T.store(destination[panel, source.shape[1] - 1, :], T.where(valid, tail, 0))


@T.func
def _packed_operand(value, side, dtype, raw):
  # Keep the public SME [panel, K-group, 2*ZA-width*group] layout. A
  # rank-three logical transpose used to scalarize this entire hot path.
  # ZA's row/column moves transpose storage words, preserving all payload
  # bits, and pad only storage; producers never see a padded logical index.
  spatial = value.shape[0] if side == "A" else value.shape[1]
  reduction = value.shape[1] if side == "A" else value.shape[0]
  width = sme.streaming_lanes(dtype)
  group = T.itemsize(dtype) // T.itemsize(value.dtype)
  chunk = sme.streaming_lanes(value.dtype)
  assert reduction <= 9223372036854775807 - (chunk - 1), "SME packing extent overflow"
  panels = (spatial + 2 * width - 1) // (2 * width)
  # Whole physical K chunks keep interval checks independent of the final
  # logical group count. MOPA still consumes only ceil(logical K/group).
  packed = T.alloc((panels, (reduction + chunk - 1) // chunk * (chunk // group), 2 * width * group), dtype=value.dtype)
  if raw is not None:
    source = raw if side == "A" else T.transpose(raw)
    if source.strides[1] == 1 and spatial % (2 * width) == 0:
      _pack_view(source, packed, dtype, group)
    else:
      _pack_dense(value, side, packed, dtype, group)
  else:
    _pack_dense(value, side, packed, dtype, group)
  return packed


@T.func
def _pack_dense(value, side, packed, dtype, group):
  logical = value if side == "A" else T.transpose(value)
  spatial = packed.shape[0] * 2 * sme.streaming_lanes(dtype)
  dense = T.alloc((spatial, logical.shape[1]), dtype=value.dtype)
  if spatial != logical.shape[0]:
    T.store(dense[logical.shape[0]:, :], T.zeros((spatial - logical.shape[0], logical.shape[1]), dtype=value.dtype))
  T.store(dense[:logical.shape[0], :], logical)
  _pack_view(dense, packed, dtype, group)


@T.func
def _pack_view(dense, packed, dtype, group):
  spatial, reduction = dense.shape
  with sme.region():
    pack_width = sme.streaming_lanes(dtype)
    chunk = sme.streaming_lanes(dense.dtype)
    for panel in T.serial(packed.shape[0], unroll=1):
      for chunk_index in T.serial((reduction + chunk - 1) // chunk, unroll=1):
        ki = chunk_index * chunk
        pack_fringe(dense, packed[panel, :, :], origin=panel * 2 * pack_width,
                    k_origin=ki, width=pack_width, active_k=T.min(chunk, reduction - ki),
                    spatial=spatial, output_origin=ki // group)


@T.func
def _grouped_tiles(a, b, output, *, initial, group, groups=None, raw_a=False, raw_b=False, initial_broadcast=None):
  if groups is None:
    groups = a.shape[0]
  if (output.shape[0] % (2 * sme.streaming_lanes(output.dtype)) == 0 and
      output.shape[1] % (2 * sme.streaming_lanes(output.dtype)) == 0):
    _grouped_tiles_2x2(a, b, output, initial=initial, group=group, groups=groups, raw_a=raw_a, raw_b=raw_b, initial_broadcast=initial_broadcast)
  else:
    _grouped_tiles_1x1(a, b, output, initial=initial, group=group, groups=groups, raw_a=raw_a, raw_b=raw_b, initial_broadcast=initial_broadcast)


@T.func
def _load_a(a, ki, mi, width, group, raw):
  if raw:
    result = sme.load(a[mi:mi + width, ki], strided=True)
  elif len(a.shape) == 3:
    start = (mi % (2 * width)) * group
    result = sme.load(a[mi // (2 * width), ki, start:start + width * group])
  else:
    result = sme.load(a[ki, mi * group:(mi + width) * group])
  return result


@T.func
def _load_b(b, ki, ni, width, group, raw):
  if raw:
    result = sme.load(b[ki, ni:ni + width], strided=True)
  elif len(b.shape) == 3:
    start = (ni % (2 * width)) * group
    result = sme.load(b[ni // (2 * width), ki, start:start + width * group])
  else:
    result = sme.load(b[ki, ni * group:(ni + width) * group])
  return result


@T.func
def _grouped_tiles_2x2(a, b, output, *, initial, group, groups, raw_a, raw_b, initial_broadcast=None):
  with sme.region():
    width = sme.streaming_lanes(output.dtype)
    for mblock in T.serial(output.shape[0] // (2 * width), unroll=1):
      for nblock in T.serial(output.shape[1] // (2 * width), unroll=1):
        mi, ni = mblock * 2 * width, nblock * 2 * width
        sme.zero_za()
        if initial_broadcast is not None:
          for c in T.unroll(2):
            prior = sme.load(initial_broadcast[0, ni + c * width:ni + (c + 1) * width])
            for r in T.unroll(2):
              for row in T.serial(width, unroll=2):
                sme.write_hor(r * 2 + c, row, prior)
        elif initial:
          for r in T.unroll(2):
            for c in T.unroll(2):
              for row in T.serial(width, unroll=2):
                sme.load_hor(r * 2 + c, row, output[mi + r * width + row, ni + c * width:ni + (c + 1) * width])
        for ki in T.serial(groups, unroll=2):
          a0 = _load_a(a, ki, mi, width, group, raw_a)
          a1 = _load_a(a, ki, mi + width, width, group, raw_a)
          b0 = _load_b(b, ki, ni, width, group, raw_b)
          b1 = _load_b(b, ki, ni + width, width, group, raw_b)
          sme.mopa(0, a0, b0, acc_dtype=output.dtype)
          sme.mopa(1, a0, b1, acc_dtype=output.dtype)
          sme.mopa(2, a1, b0, acc_dtype=output.dtype)
          sme.mopa(3, a1, b1, acc_dtype=output.dtype)
        for r in T.unroll(2):
          for c in T.unroll(2):
            for row in T.serial(width, unroll=2):
              sme.store_hor(r * 2 + c, row, output[mi + r * width + row, ni + c * width:ni + (c + 1) * width])


@T.func
def _grouped_tiles_1x1(a, b, output, *, initial, group, groups, raw_a, raw_b, initial_broadcast=None):
  with sme.region():
    width = sme.streaming_lanes(output.dtype)
    for mi in T.serial(0, output.shape[0], width, unroll=1):
      for ni in T.serial(0, output.shape[1], width, unroll=1):
        sme.zero_za()
        if initial_broadcast is not None:
          prior = sme.load(initial_broadcast[0, ni:ni + width])
          for row in T.serial(T.min(width, output.shape[0] - mi), unroll=2):
            sme.write_hor(0, row, prior)
        elif initial:
          for row in T.serial(T.min(width, output.shape[0] - mi), unroll=2):
            sme.load_hor(0, row, output[mi + row, ni:ni + width])
        for ki in T.serial(groups, unroll=2):
          left = _load_a(a, ki, mi, width, group, raw_a)
          right = _load_b(b, ki, ni, width, group, raw_b)
          sme.mopa(0, left, right, acc_dtype=output.dtype)
        for row in T.serial(T.min(width, output.shape[0] - mi), unroll=2):
          sme.store_hor(0, row, output[mi + row, ni:ni + width])


@T.func
def dot_into(inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, raw_a=None, raw_b=None,
             packed_a=None, packed_b=None, broadcast_acc=False, pack_raw_inputs=True):
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  m, n, k = a.shape[0], b.shape[1], a.shape[1]
  if m > 0 and n > 0:
    group = T.itemsize(acc_dtype) // T.itemsize(a.dtype)
    assert k <= 9223372036854775807 - (group - 1), "SME reduction extent overflow"
    kp = (k + group - 1) // group * group
    if group == 1 and raw_a is not None and raw_b is not None:
      if pack_raw_inputs and _prefer_raw_packing(m, n, k, acc_dtype):
        left = _packed_operand(a, "A", acc_dtype, raw_a)
        right = _packed_operand(b, "B", acc_dtype, raw_b)
        _output(left, right, output, acc, acc_dtype, epilogue, group, groups=k, broadcast_acc=broadcast_acc)
      else:
        # Private output protects overlapping raw producers on the gather path.
        _buffered_output(raw_a, raw_b, output, acc, acc_dtype, epilogue, group, groups=k,
                         raw_a=True, raw_b=True, broadcast_acc=broadcast_acc)
    elif acc_dtype == "int64":
      # I16/I64 uses four halfwords per 64-bit accumulator lane. The public
      # I16 SME packing ABI instead groups two halfwords into a dword.
      # Retain the dedicated layout until a separate 64-bit ABI is available.
      left = grouped_operand(a, side="A", group=group, spatial=m, reduction=kp)
      right = grouped_operand(b, side="B", group=group, spatial=n, reduction=kp)
      _output(left, right, output, acc, acc_dtype, epilogue, group, broadcast_acc=broadcast_acc)
    else:
      width = sme.streaming_lanes(acc_dtype)
      assert m <= 9223372036854775807 - (2 * width - 1), "SME spatial extent overflow"
      assert n <= 9223372036854775807 - (2 * width - 1), "SME spatial extent overflow"
      if packed_a is not None:
        shape_a = ((m + 2 * width - 1) // (2 * width), kp // group, 2 * width * group)
        source_a = T.reshape(packed_a[:shape_a[0] * kp * 2 * width], shape_a)
        if k % group == 0:
          _input_b(source_a, b, packed_b, raw_b, output, acc, acc_dtype, epilogue, group, kp, width, broadcast_acc)
        else:
          saved_a = T.alloc(shape_a, dtype=a.dtype)
          _repair_packed_tail(source_a, saved_a, k, group)
          _input_b(saved_a, b, packed_b, raw_b, output, acc, acc_dtype, epilogue, group, kp, width, broadcast_acc)
      else:
        left = _packed_operand(a, "A", acc_dtype, raw_a)
        _input_b(left, b, packed_b, raw_b, output, acc, acc_dtype, epilogue, group, kp, width, broadcast_acc)


@T.func
def _input_b(left, b, packed_b, raw_b, output, acc, dtype, epilogue, group, kp, width, broadcast_acc):
  if packed_b is not None:
    shape_b = ((b.shape[1] + 2 * width - 1) // (2 * width), kp // group, 2 * width * group)
    source_b = T.reshape(packed_b[:shape_b[0] * kp * 2 * width], shape_b)
    if b.shape[0] % group == 0:
      _output(left, source_b, output, acc, dtype, epilogue, group, groups=kp // group, broadcast_acc=broadcast_acc)
    else:
      saved_b = T.alloc(shape_b, dtype=b.dtype)
      _repair_packed_tail(source_b, saved_b, b.shape[0], group)
      _output(left, saved_b, output, acc, dtype, epilogue, group, groups=kp // group, broadcast_acc=broadcast_acc)
  else:
    right = _packed_operand(b, "B", dtype, raw_b)
    _output(left, right, output, acc, dtype, epilogue, group, groups=kp // group, broadcast_acc=broadcast_acc)


@T.func
def _output(left, right, output, acc, acc_dtype, epilogue, group, *, groups=None, broadcast_acc=False):
  if epilogue is None and output.dtype == acc_dtype:
    if output.strides[1] == 1:
      if acc is not None and broadcast_acc:
        prior = _broadcast_seed(acc, output.shape[1], acc_dtype)
        _grouped_tiles(left, right, output, initial=False, group=group, groups=groups, initial_broadcast=prior)
      else:
        if acc is not None:
          T.store(output, T.broadcast_to(T.cast(T.as_value(acc), acc_dtype), output.shape))
        _grouped_tiles(left, right, output, initial=acc is not None, group=group, groups=groups)
    else:
      _buffered_output(left, right, output, acc, acc_dtype, epilogue, group, groups=groups, broadcast_acc=broadcast_acc)
  else:
    _buffered_output(left, right, output, acc, acc_dtype, epilogue, group, groups=groups, broadcast_acc=broadcast_acc)

@T.func
def _broadcast_seed(acc, columns, dtype):
  # Snapshot before any output write, including a bias alias of C's first row.
  # Repeated ZA row writes preserve +/-0 and NaN payloads. MOPA with a vector
  # of ones would introduce floating-point arithmetic into initialization.
  prior = T.alloc((1, columns), dtype=dtype)
  T.store(prior, T.broadcast_to(T.cast(T.as_value(acc), dtype), prior.shape))
  return prior


@T.func
def _buffered_output(left, right, output, acc, dtype, epilogue, group, *, groups=None, raw_a=False, raw_b=False, broadcast_acc=False):
  result = T.alloc(output.shape, dtype=dtype)
  if acc is not None and broadcast_acc:
    prior = _broadcast_seed(acc, output.shape[1], dtype)
    _grouped_tiles(left, right, result, initial=False, group=group, groups=groups,
                   raw_a=raw_a, raw_b=raw_b, initial_broadcast=prior)
  else:
    if acc is not None:
      seed_accumulator(acc, result, output.shape[0], output.shape[1], dtype)
    _grouped_tiles(left, right, result, initial=acc is not None, group=group, groups=groups, raw_a=raw_a, raw_b=raw_b)
  finish_matrix(result, output, epilogue=epilogue)


@T.func
def _prefer_raw_packing(m, n, k, dtype):
  # Amortize transposition over several register tiles and enough K work.
  # Count packing in the operation's timing. Keep tiny and nonresident
  # problems on their bounded direct path. Division avoids M+N overflow.
  return (m >= 2 * sme.streaming_lanes(dtype) and n >= 2 * sme.streaming_lanes(dtype)
          and k >= 64 and m <= 2 * T.cache_bytes("L2") // k // T.itemsize(dtype) - n)


@T.func
def _direct_dot(a, b, output):
  """Source-level tile traversal, with one lexical owner."""
  if output.shape[0] > 0 and output.shape[1] > 0:
    if _prefer_raw_packing(a.shape[0], b.shape[1], a.shape[1], a.dtype):
      left = _packed_operand(T.load(a), "A", a.dtype, a)
      right = _packed_operand(T.load(b), "B", b.dtype, b)
      _grouped_tiles(left, right, output, initial=False, group=1, groups=a.shape[1])
    else:
      _grouped_tiles(a, b, output, initial=False, group=1, groups=a.shape[1], raw_a=True, raw_b=True)


@T.func
def direct_dot(a, b, output):
  """Protect overlapping raw producers before entering the one ZA owner.

  Copies are scalar-element storage and only occur on the overlap branch.
  They are alias preservation, not matrix-instruction packing. Neither dot
  family nor cache traversal is chosen in the native instruction printer.
  """
  T.static_assert(a.dtype == b.dtype and b.dtype == output.dtype)
  T.static_assert(a.dtype == "float32" or a.dtype == "float64")
  assert a.shape[1] == b.shape[0], "incompatible SME dot reduction extents"
  assert output.shape[0] == a.shape[0] and output.shape[1] == b.shape[1], "SME dot output shape mismatch"
  if T.view_disjoint(a, output) and T.view_disjoint(b, output):
    _direct_dot(a, b, output)
  else:
    saved_a = T.alloc(a.shape, dtype=a.dtype)
    saved_b = T.alloc(b.shape, dtype=b.dtype)
    T.copy(a, saved_a)
    T.copy(b, saved_b)
    _direct_dot(saved_a, saved_b, output)
