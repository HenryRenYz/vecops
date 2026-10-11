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
from .registers import transpose_dwords_512


@T.func
def _tiles(a, b, output, *, initial, kr):
  if output.shape[0] % 32 == 0 and output.shape[1] % 32 == 0:
    _tiles_2x2(a, b, output, initial=initial, kr=kr)
  else:
    _tiles_1x1(a, b, output, initial=initial, kr=kr)


@T.func
def _tiles_2x2(a, b, output, *, initial, kr):
  # Four C, two A and two B registers: reuse each loaded operand twice.
  with amx.region(((16, 64), (16, 64), (16, 64), (16, 64),
                   (16, 64), (16, 64), (16, 64), (16, 64))):
    for mi in T.serial(output.shape[0] // 32, unroll=1):
      for ni in T.serial(output.shape[1] // 32, unroll=1):
        if initial:
          c0 = amx.load(0, output[mi * 32:mi * 32 + 16, ni * 32:ni * 32 + 16])
          c1 = amx.load(1, output[mi * 32:mi * 32 + 16, ni * 32 + 16:ni * 32 + 32])
          c2 = amx.load(2, output[mi * 32 + 16:mi * 32 + 32, ni * 32:ni * 32 + 16])
          c3 = amx.load(3, output[mi * 32 + 16:mi * 32 + 32, ni * 32 + 16:ni * 32 + 32])
        else:
          c0 = amx.zero(0, dtype=output.dtype)
          c1 = amx.zero(1, dtype=output.dtype)
          c2 = amx.zero(2, dtype=output.dtype)
          c3 = amx.zero(3, dtype=output.dtype)
        for ki in T.serial(a.shape[1] // kr, unroll=2):
          a0 = amx.load(4, a[mi * 32:mi * 32 + 16, ki * kr:ki * kr + kr])
          a1 = amx.load(5, a[mi * 32 + 16:mi * 32 + 32, ki * kr:ki * kr + kr])
          b0 = amx.load(6, b[ni * 2, ki, :, :])
          b1 = amx.load(7, b[ni * 2 + 1, ki, :, :])
          amx.tdp(c0, a0, b0)
          amx.tdp(c1, a0, b1)
          amx.tdp(c2, a1, b0)
          amx.tdp(c3, a1, b1)
        amx.store(c0, output[mi * 32:mi * 32 + 16, ni * 32:ni * 32 + 16])
        amx.store(c1, output[mi * 32:mi * 32 + 16, ni * 32 + 16:ni * 32 + 32])
        amx.store(c2, output[mi * 32 + 16:mi * 32 + 32, ni * 32:ni * 32 + 16])
        amx.store(c3, output[mi * 32 + 16:mi * 32 + 32, ni * 32 + 16:ni * 32 + 32])


@T.func
def _tiles_1x1(a, b, output, *, initial, kr):
  with amx.region(((16, 64), (16, 64), (16, 64))):
    for mi in T.serial(output.shape[0] // 16, unroll=1):
      for ni in T.serial(output.shape[1] // 16, unroll=1):
        if initial:
          accumulator = amx.load(0, output[mi * 16:mi * 16 + 16, ni * 16:ni * 16 + 16])
        else:
          accumulator = amx.zero(0, dtype=output.dtype)
        for ki in T.serial(a.shape[1] // kr, unroll=2):
          left = amx.load(1, a[mi * 16:mi * 16 + 16, ki * kr:ki * kr + kr])
          right = amx.load(2, b[ni, ki, :, :])
          amx.tdp(accumulator, left, right)
        amx.store(accumulator, output[mi * 16:mi * 16 + 16, ni * 16:ni * 16 + 16])


@T.func
def dot_into(inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, raw_a=None, packed_b=None):
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  m, n, k = a.shape[0], b.shape[1], a.shape[1]
  if m > 0 and n > 0:
    kr = 64 // T.itemsize(a.dtype)
    group = 4 // T.itemsize(b.dtype)
    assert m <= 9223372036854775792 and n <= 9223372036854775792, "AMX spatial extent overflow"
    assert k <= 9223372036854775807 - (kr - 1), "AMX reduction extent overflow"
    mp, np, kp = (m + 15) // 16 * 16, (n + 15) // 16 * 16, (k + kr - 1) // kr * kr
    if packed_b is not None:
      # A full K domain never consumes padding from an external declaration.
      # Fractional K uses the logical decoder and our own zero-padded panels.
      if k % kr == 0:
        count = np * kp
        panels = T.reshape(packed_b[:count], (np // 16, kp // kr, 16, 16 * group))
        _input_a(a, raw_a, panels, output, acc, acc_dtype, epilogue, mp, np, kp, kr)
      else:
        _input_b(a, b, raw_a, output, acc, acc_dtype, epilogue, mp, np, kp, kr, group)
    else:
      _input_b(a, b, raw_a, output, acc, acc_dtype, epilogue, mp, np, kp, kr, group)


@T.func
def _input_b(a, b, raw_a, output, acc, dtype, epilogue, mp, np, kp, kr, group):
  # Reinterpret each spatial row's K groups as 16 dwords, then transpose
  # sixteen registers. This is the historical DSL/C++ AMX packing schedule:
  # ordinary rank-three reshape/transpose falls back to costly scalar gathers.
  dense = T.alloc((np, kp), dtype=b.dtype)
  T.store(dense, T.zeros(dense.shape, dtype=b.dtype))
  T.store(dense[:b.shape[1], :b.shape[0]], T.transpose(b))
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
  _input_a(a, raw_a, panels, output, acc, dtype, epilogue, mp, np, kp, kr)


@T.func
def _input_a(a, raw_a, panels, output, acc, dtype, epilogue, mp, np, kp, kr):
  if raw_a is not None:
    if a.shape[0] % 16 == 0 and a.shape[1] % kr == 0 and raw_a.strides[1] == 1 and T.view_disjoint(raw_a, output):
      _output(raw_a, panels, output, acc, dtype, epilogue, mp, np, kr)
    else:
      _packed_a(a, panels, output, acc, dtype, epilogue, mp, np, kp, kr)
  else:
    _packed_a(a, panels, output, acc, dtype, epilogue, mp, np, kp, kr)


@T.func
def _packed_a(a, panels, output, acc, dtype, epilogue, mp, np, kp, kr):
  left = T.alloc((mp, kp), dtype=a.dtype)
  T.store(left, T.zeros(left.shape, dtype=left.dtype))
  T.store(left[:a.shape[0], :a.shape[1]], a)
  _output(left, panels, output, acc, dtype, epilogue, mp, np, kr)


@T.func
def _output(left, panels, output, acc, dtype, epilogue, mp, np, kr):
  if epilogue is None and output.dtype == dtype:
    if output.shape[0] % 16 == 0 and output.shape[1] % 16 == 0 and output.strides[1] == 1:
      if acc is not None:
        T.store(output, T.broadcast_to(T.cast(T.as_value(acc), dtype), output.shape))
      _tiles(left, panels, output, initial=acc is not None, kr=kr)
    else:
      _buffered_output(left, panels, output, acc, dtype, epilogue, mp, np, kr)
  else:
    _buffered_output(left, panels, output, acc, dtype, epilogue, mp, np, kr)


@T.func
def _buffered_output(left, panels, output, acc, dtype, epilogue, mp, np, kr):
  result = T.alloc((mp, np), dtype=dtype)
  if acc is not None:
    seed_accumulator(acc, result, output.shape[0], output.shape[1], dtype)
  _tiles(left, panels, result, initial=acc is not None, kr=kr)
  finish_matrix(result, output, epilogue=epilogue)
