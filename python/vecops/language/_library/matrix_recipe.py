# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared source-level packing, accumulator and epilogue boundaries.

Instruction owners consume private, zero-padded storage. Producer expressions
are evaluated only over their logical domain, before any architectural state
is acquired. This also retains captured-read and input/output alias semantics.
"""

from .. import dsl as T
from .matmul import _vector_row_store


@T.func
def grouped_operand(value, *, side, group, spatial, reduction):
  value = T.as_value(value)
  assert spatial <= 9223372036854775807 // group, "matrix packing extent overflow"
  dense = T.alloc((spatial, reduction), dtype=value.dtype)
  T.store(dense, T.zeros(dense.shape, dtype=dense.dtype))
  if side == "A":
    T.store(dense[:value.shape[0], :value.shape[1]], value)
  else:
    T.store(dense[:value.shape[1], :value.shape[0]], T.transpose(value))
  groups = reduction // group
  packed = T.alloc((groups, spatial * group), dtype=value.dtype)
  T.store(packed, T.reshape(T.transpose(T.reshape(T.load(dense), (spatial, groups, group)), axes=(1, 0, 2)), packed.shape))
  return packed


@T.func
def seed_accumulator(acc, storage, rows, columns, dtype):
  T.store(storage, T.zeros(storage.shape, dtype=dtype))
  if acc is not None:
    initial = T.broadcast_to(T.cast(T.as_value(acc), dtype), (rows, columns))
    T.store(storage[:rows, :columns], initial)


@T.func
def finish_matrix(storage, output, *, epilogue=None):
  width = T.lanes(T.Vector(storage.dtype))
  for row in T.serial(output.shape[0], unroll=1):
    for column in T.serial(0, output.shape[1], width, unroll=1):
      end = T.min(column + width, output.shape[1])
      value = T.as_vector(T.load(storage[row, column:end]), T.Vector(storage.dtype))
      _vector_row_store(value, output, row, column, width, epilogue)


@T.func
def blocked_matrix(inputs, output, leaf, *, acc=None, acc_dtype="float32", epilogue=None,
                   BM=None, BN=None, BK=None, span=32, quantum=32, packing="auto"):
  """Cache phases: initializer once, continuation, epilogue once.

  Logical slicing keeps producer DAGs and already-packed logical operands
  intact. Each hardware leaf handles physical padding independently of an
  explicit user's cache tile sizes, including a nonaligned final K phase.
  """
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  m, n, k = a.shape[0], b.shape[1], a.shape[1]
  if m > 0 and n > 0:
    if (packing == "auto" and BM is None and BN is None and BK is None and
        m <= T.cache_bytes("L2") // 4 // T.itemsize(acc_dtype) // n and
        k <= T.cache_bytes("L2") // 2 // (m * T.itemsize(a.dtype) + n * T.itemsize(b.dtype))):
      leaf((a, b), output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue)
    else:
      bm = span if BM is None else BM
      bn = span if BN is None else BN
      bk = T.max(quantum, T.min(512, T.cache_bytes("L1") // 2 //
                    (bm * T.itemsize(a.dtype) + bn * T.itemsize(b.dtype))) // quantum * quantum) if BK is None else BK
      assert bm > 0 and bn > 0 and bk > 0, "matrix cache blocks must be positive"
      partial = T.alloc((m, n), dtype=acc_dtype)
      seed_accumulator(acc, partial, m, n, acc_dtype)
      for ni in T.serial(0, n, bn, unroll=1):
        for ki in T.serial(0, k, bk, unroll=1):
          for mi in T.serial(0, m, bm, unroll=1):
            end_m, end_n, end_k = T.min(mi + bm, m), T.min(ni + bn, n), T.min(ki + bk, k)
            left, right = a[mi:end_m, ki:end_k], b[ki:end_k, ni:end_n]
            # Explicit optional dense-cache policies are distinct from the
            # physical grouping/padding required by a matrix instruction.
            if packing == "A" or packing == "AB":
              left = T.pack(left, side="A", format="dense")
            if packing == "B" or packing == "AB":
              right = T.pack(right, side="B", format="dense")
            leaf((left, right), partial[mi:end_m, ni:end_n],
                 acc=T.load(partial[mi:end_m, ni:end_n]), acc_dtype=acc_dtype)
      finish_matrix(partial, output, epilogue=epilogue)
