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


@T.func
def _grouped_tiles(a, b, output, *, initial, group, groups=None, raw_a=False, raw_b=False):
  if groups is None:
    groups = a.shape[0]
  if (output.shape[0] % (2 * sme.streaming_lanes(output.dtype)) == 0 and
      output.shape[1] % (2 * sme.streaming_lanes(output.dtype)) == 0):
    _grouped_tiles_2x2(a, b, output, initial=initial, group=group, groups=groups, raw_a=raw_a, raw_b=raw_b)
  else:
    _grouped_tiles_1x1(a, b, output, initial=initial, group=group, groups=groups, raw_a=raw_a, raw_b=raw_b)


@T.func
def _load_a(a, ki, mi, width, group, raw):
  if raw:
    result = sme.load(a[mi:mi + width, ki], strided=True)
  else:
    result = sme.load(a[ki, mi * group:(mi + width) * group])
  return result


@T.func
def _load_b(b, ki, ni, width, group, raw):
  if raw:
    result = sme.load(b[ki, ni:ni + width], strided=True)
  else:
    result = sme.load(b[ki, ni * group:(ni + width) * group])
  return result


@T.func
def _grouped_tiles_2x2(a, b, output, *, initial, group, groups, raw_a, raw_b):
  with sme.region():
    width = sme.streaming_lanes(output.dtype)
    for mblock in T.serial(output.shape[0] // (2 * width), unroll=1):
      for nblock in T.serial(output.shape[1] // (2 * width), unroll=1):
        mi, ni = mblock * 2 * width, nblock * 2 * width
        sme.zero_za()
        if initial:
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
def _grouped_tiles_1x1(a, b, output, *, initial, group, groups, raw_a, raw_b):
  with sme.region():
    width = sme.streaming_lanes(output.dtype)
    for mi in T.serial(0, output.shape[0], width, unroll=1):
      for ni in T.serial(0, output.shape[1], width, unroll=1):
        sme.zero_za()
        if initial:
          for row in T.serial(T.min(width, output.shape[0] - mi), unroll=2):
            sme.load_hor(0, row, output[mi + row, ni:ni + width])
        for ki in T.serial(groups, unroll=2):
          left = _load_a(a, ki, mi, width, group, raw_a)
          right = _load_b(b, ki, ni, width, group, raw_b)
          sme.mopa(0, left, right, acc_dtype=output.dtype)
        for row in T.serial(T.min(width, output.shape[0] - mi), unroll=2):
          sme.store_hor(0, row, output[mi + row, ni:ni + width])


@T.func
def dot_into(inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, raw_a=None, raw_b=None):
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  m, n, k = a.shape[0], b.shape[1], a.shape[1]
  if m > 0 and n > 0:
    group = T.itemsize(acc_dtype) // T.itemsize(a.dtype)
    assert k <= 9223372036854775807 - (group - 1), "SME reduction extent overflow"
    kp = (k + group - 1) // group * group
    if group == 1 and raw_a is not None and raw_b is not None:
      # Direct strided sources remain legal with an accumulator and epilogue.
      # Output stays private, so even overlapping raw input/output is safe.
      _buffered_output(raw_a, raw_b, output, acc, acc_dtype, epilogue, group, groups=k, raw_a=True, raw_b=True)
    else:
      left = grouped_operand(a, side="A", group=group, spatial=m, reduction=kp)
      right = grouped_operand(b, side="B", group=group, spatial=n, reduction=kp)
      _output(left, right, output, acc, acc_dtype, epilogue, group)


@T.func
def _output(left, right, output, acc, acc_dtype, epilogue, group):
  if epilogue is None and output.dtype == acc_dtype:
    if output.strides[1] == 1:
      if acc is not None:
        T.store(output, T.broadcast_to(T.cast(T.as_value(acc), acc_dtype), output.shape))
      _grouped_tiles(left, right, output, initial=acc is not None, group=group)
    else:
      _buffered_output(left, right, output, acc, acc_dtype, epilogue, group)
  else:
    _buffered_output(left, right, output, acc, acc_dtype, epilogue, group)

@T.func
def _buffered_output(left, right, output, acc, dtype, epilogue, group, *, groups=None, raw_a=False, raw_b=False):
  result = T.alloc(output.shape, dtype=dtype)
  if acc is not None:
    seed_accumulator(acc, result, output.shape[0], output.shape[1], dtype)
  _grouped_tiles(left, right, result, initial=acc is not None, group=group, groups=groups, raw_a=raw_a, raw_b=raw_b)
  finish_matrix(result, output, epilogue=epilogue)


@T.func
def _matrix_tiles(a, b, output):
  """A bounded ZA tile; retain tail correlations in the source iteration."""
  width = sme.streaming_lanes(a.dtype)
  for mi in T.serial(0, a.shape[0], width, unroll=1):
    for ni in T.serial(0, b.shape[1], width, unroll=1):
      sme.zero_za()
      for ki in T.serial(a.shape[1], unroll=2):
        left = sme.load(a[mi : mi + width, ki], strided=True)
        right = sme.load(b[ki, ni : ni + width], strided=True)
        sme.mopa(0, left, right)
      for row in T.serial(T.min(width, a.shape[0] - mi), unroll=2):
        sme.store_hor(0, row, output[mi + row, ni : ni + width])


@T.func
def _direct_dot(a, b, output):
  """Source-level tile traversal, with one lexical owner."""
  if output.shape[0] > 0 and output.shape[1] > 0:
    with sme.region():
      _matrix_tiles(a, b, output)


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
