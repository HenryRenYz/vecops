# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ordinary-SVE MMLA leaves over the existing public SME packed ABI."""

from .. import dsl as T
from .. import sme, sve
from .matrix_recipe import finish_matrix


@T.func
def _group(source, index, logical_k, spatial, group, stride):
  lanes = T.lanes(T.Vector(source.dtype))
  coordinate_dtype = "uint16" if T.itemsize(source.dtype) == 2 else "uint8"
  coordinates = sve.iota(coordinate_dtype)
  shift = 1 if T.itemsize(source.dtype) == 2 else 2
  member_mask = T.cast(group - 1, coordinate_dtype)
  valid = ((coordinates >> shift < T.cast(spatial, coordinate_dtype)) &
           (coordinates & member_mask < T.cast(T.min(group, logical_k - index * group), coordinate_dtype)))
  value = T.load(source[index * stride:index * stride + lanes], vector=T.Vector(source.dtype))
  return T.as_vector(T.where(valid, value, 0), T.Vector(source.dtype))


@T.func
def dot_into(inputs, output, *, packed_a, packed_b, acc=None, acc_dtype="float32", epilogue=None):
  """Small public packed AB leaf; no online packing or streaming owner.

  Supports BF16/FP32 and same-sign bytes/I32 at 2x2, 2x4, 2x8, 4x2,
  4x4 and 8x2. The compiler adapter must first validate dense physical
  capacity and output nonaliasing for the public packed declarations.
  The logical K tail is masked before joining, including poisoned external
  padding. An absent paired group is a zero register, never an extra read.
  Ordinary VL and streaming SVL are independently checked; this conservative
  initial route requires equality and sufficient row-pair capacity.
  Accumulator/epilogue operands are consumed before the first output write.
  BF16 follows MMLA accumulation semantics, not bitwise scalar reassociation.
  """
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  T.static_assert((a.dtype == b.dtype == "bfloat16" and acc_dtype == "float32") or
                  (a.dtype == b.dtype and (a.dtype == "int8" or a.dtype == "uint8") and acc_dtype == "int32"))
  m, k, n = a.shape[0], a.shape[1], b.shape[1]
  assert ((m == 2 and (n == 2 or n == 4 or n == 8)) or
          (m == 4 and (n == 2 or n == 4)) or (m == 8 and n == 2)), "unsupported packed MMLA shape"
  assert b.shape[0] == k and output.shape[0] == m and output.shape[1] == n, "packed MMLA shape mismatch"
  span = 2 * sme.streaming_lanes(acc_dtype)
  group = 4 // T.itemsize(a.dtype)
  assert k <= 9223372036854775807 - (group - 1), "packed MMLA reduction extent overflow"
  assert T.lanes(T.Vector(acc_dtype)) == sme.streaming_lanes(acc_dtype), "packed MMLA requires matching ordinary VL and streaming SVL"
  assert span >= T.max(m, n), "packed MMLA spatial panel too small"
  assert T.lanes(T.Vector(acc_dtype)) >= (m * n if m == 8 else 2 * n), "packed MMLA ordinary VL too small"
  groups = (k + group - 1) // group
  stride = span * group
  result = T.alloc((m, n), dtype=acc_dtype)
  acc0 = T.as_vector(T.zeros((T.lanes(T.Vector(acc_dtype)),), dtype=acc_dtype), T.Vector(acc_dtype))
  acc1 = T.as_vector(T.zeros((T.lanes(T.Vector(acc_dtype)),), dtype=acc_dtype), T.Vector(acc_dtype))
  zero_a = T.as_vector(T.zeros((T.lanes(T.Vector(a.dtype)),), dtype=a.dtype), T.Vector(a.dtype))
  zero_b = T.as_vector(T.zeros((T.lanes(T.Vector(b.dtype)),), dtype=b.dtype), T.Vector(b.dtype))
  for ki in T.serial(0, groups, 2, unroll=1):
    a0 = _group(packed_a, ki, k, m, group, stride)
    b0 = _group(packed_b, ki, k, n, group, stride)
    a1, b1 = zero_a, zero_b
    if ki + 1 < groups:
      a1 = _group(packed_a, ki + 1, k, m, group, stride)
      b1 = _group(packed_b, ki + 1, k, n, group, stride)
    left, right = sve.join_k_groups(a0, a1), sve.join_k_groups(b0, b1)
    if m == 8:
      acc0 = sve.mmla(left, sve.broadcast_segment(right, 0), acc0)
    else:
      acc0 = sve.mmla(sve.broadcast_segment(left, 0), right, acc0)
      if m == 4:
        acc1 = sve.mmla(sve.broadcast_segment(left, 1), right, acc1)
  if m == 8:
    T.store(T.reshape(result, (m * n,)), acc0[:m * n])
  else:
    T.store(result[0, :], sve.row_pair(acc0, 0)[:n])
    T.store(result[1, :], sve.row_pair(acc0, 1)[:n])
    if m == 4:
      T.store(result[2, :], sve.row_pair(acc1, 0)[:n])
      T.store(result[3, :], sve.row_pair(acc1, 1)[:n])
  if acc is not None:
    if acc_dtype == "int32":
      seeded = T.bitcast(T.load(result), "uint32") + T.bitcast(T.cast(T.as_value(acc), "int32"), "uint32")
      T.store(result, T.bitcast(seeded, "int32"))
    else:
      T.store(result, T.load(result) + T.cast(T.as_value(acc), acc_dtype))
  finish_matrix(result, output, epilogue=epilogue)
