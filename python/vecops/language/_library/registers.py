# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Inspectable register algorithms shared by transpose and matrix packing.

The arguments are immutable tuples of logical vectors. Every iteration and
branch specializes away; no Python callback or C++ closure runs in a kernel.
The compiler supplies geometry, not the permutation algorithm.
"""

from .. import dsl as T

_UNSIGNED_BITS = ("uint8", "uint16", "uint32", "uint64")


@T.func
def pack_interleave(values):
  """Interleave grouped-K streams, preserving exact storage payload bits.

  Two 16-bit streams or four 8-bit streams form one packed dword per logical
  spatial element. This algorithm accepts fixed and scalable vectors; only
  its geometry primitives are backend instructions.
  """
  T.static_assert(len(values) == 2 or len(values) == 4)
  pair = T.interleave(values[0], values[1])
  if len(values) == 2:
    return (pair,)
  other = T.interleave(values[2], values[3])
  words = T.interleave(T.bitcast(pair, "uint16"), T.bitcast(other, "uint16"))
  return (T.bitcast(words, values[0].dtype),)


@T.func
def _transpose_square(values):
  lanes = values[0].shape[0]
  dtype = values[0].dtype
  element_bytes = T.itemsize(dtype)
  if lanes == 1:
    return values
  if lanes * element_bytes == 16:
    span = 1
    for stage in T.unroll(T.static_log2(lanes)):
      next_values = ()
      stage_dtype = _UNSIGNED_BITS[T.static_log2(element_bytes * span)]
      for start in T.unroll(0, lanes, span * 2):
        for offset in T.unroll(span):
          a = T.bitcast(values[start + offset], stage_dtype)
          b = T.bitcast(values[start + span + offset], stage_dtype)
          lower = T.bitcast(T.local_interleave_lower(a, b), dtype)
          upper = T.bitcast(T.local_interleave_upper(a, b), dtype)
          next_values = next_values + (lower, upper)
      values = next_values
      span = span * 2
    return values
  half = lanes // 2
  low = ()
  for row in T.unroll(lanes):
    part = T.lower_half(values[row])
    low = low + (part,)
  high = ()
  for row in T.unroll(lanes):
    part = T.upper_half(values[row])
    high = high + (part,)
  tl = _transpose_square(low[:half])
  bl = _transpose_square(low[half:])
  tr = _transpose_square(high[:half])
  br = _transpose_square(high[half:])
  outputs = ()
  for column in T.unroll(half):
    joined = T.concat(tl[column], bl[column])
    outputs = outputs + (joined,)
  for column in T.unroll(half):
    joined = T.concat(tr[column], br[column])
    outputs = outputs + (joined,)
  return outputs


@T.func
def transpose_registers(values):
  rows = len(values)
  columns = values[0].shape[0]
  T.static_assert(rows > 0 and (rows & (rows - 1)) == 0)
  T.static_assert(columns == rows or columns == rows * 2)
  if rows == columns:
    return _transpose_square(values)
  low = ()
  for row in T.unroll(rows):
    part = T.lower_half(values[row])
    low = low + (part,)
  lower = _transpose_square(low)
  high = ()
  for row in T.unroll(rows):
    part = T.upper_half(values[row])
    high = high + (part,)
  upper = _transpose_square(high)
  return lower + upper


@T.func
def transpose_dwords_512(values):
  """16x16 dwords: local unpacks, then 128-bit block permutations."""
  first = ()
  for row in T.unroll(0, 16, 2):
    first = first + (
      T.local_interleave_lower(values[row], values[row + 1]),
      T.local_interleave_upper(values[row], values[row + 1]),
    )
  values = ()
  for start in T.unroll(0, 16, 4):
    for offset in T.unroll(2):
      a = T.bitcast(first[start + offset], "uint64")
      b = T.bitcast(first[start + offset + 2], "uint64")
      values = values + (
        T.bitcast(T.local_interleave_lower(a, b), "uint32"),
        T.bitcast(T.local_interleave_upper(a, b), "uint32"),
      )
  span = 4
  for stage in T.unroll(2):
    next_values = ()
    for start in T.unroll(0, 16, span * 2):
      lower = ()
      upper = ()
      for offset in T.unroll(span):
        a, b = values[start + offset], values[start + offset + span]
        lower = lower + (T.block_shuffle(a, b, (0, 2, 4, 6)),)
        upper = upper + (T.block_shuffle(a, b, (1, 3, 5, 7)),)
      next_values = next_values + lower + upper
    values = next_values
    span = span * 2
  return values
