# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared source-level packing, accumulator and epilogue boundaries.

Instruction owners consume bounded raw/packed views or private staging.
Privately constructed padding never evaluates a producer outside its logical
domain. Producer staging precedes ownership and output writes, retaining
captured-read and input/output alias semantics.
"""

from .. import dsl as T


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
  # Complete initializers overwrite every element. Only absent initializers
  # or padded hardware accumulators require a separate zero-fill pass.
  if acc is None or rows != storage.shape[0] or columns != storage.shape[1]:
    T.store(storage, T.zeros(storage.shape, dtype=dtype))
  if acc is not None:
    initial = T.broadcast_to(T.cast(T.as_value(acc), dtype), (rows, columns))
    T.store(storage[:rows, :columns], initial)


@T.func
def finish_matrix(storage, output, *, epilogue=None):
  if epilogue is None:
    T.store(output, T.load(storage[:output.shape[0], :output.shape[1]]))
  else:
    # Project the caller's pointwise DAG over a whole logical row. Ordinary
    # vector legalization can then share checks and use several vector words
    # per iteration; one-word scatter loops repeat that work at every word.
    for row in T.serial(output.shape[0], unroll=1):
      value = T.load(storage[row, :output.shape[1]])
      T.store(output[row, :], epilogue(value, row, 0))


@T.func
def blocked_matrix(inputs, output, leaf, *, acc=None, acc_dtype="float32", epilogue=None,
                   BM=None, BN=None, BK=None, span=32, quantum=32, packing="auto",
                   raw_a=None, raw_b=None, packed_a=None, packed_b=None, broadcast_acc=False):
  """Cache phases: initializer once, continuation, epilogue once.

  Logical slicing keeps producer DAGs and already-packed logical operands
  intact. Each hardware leaf handles physical padding independently of an
  explicit user's cache tile sizes, including a nonaligned final K phase.
  """
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  m, n, k = a.shape[0], b.shape[1], a.shape[1]
  if m > 0 and n > 0:
    if (packing == "auto" and BM is None and BN is None and BK is None and (
        (m <= T.cache_bytes("L2") // 4 // T.itemsize(acc_dtype) // n and
         k <= T.cache_bytes("L2") // 2 // (m * T.itemsize(a.dtype) + n * T.itemsize(b.dtype))) or
        # For moderate raw FP64 working sets, one resident leaf (with an
        # optional private pack) avoids repeating packed cache phases:
        # the measured 64x256x256 Dynamic case was ~2 ms versus ~350 us dot.
        # Keep large problems and explicit cache policies on the tiled path.
        (acc_dtype == "float64" and raw_a is not None and raw_b is not None and
         m <= T.cache_bytes("L2") // 2 // T.itemsize(acc_dtype) // n and
         k <= 2 * T.cache_bytes("L2") // (m * T.itemsize(a.dtype) + n * T.itemsize(b.dtype))))):
      if broadcast_acc:
        leaf((a, b), output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
             raw_a=raw_a, raw_b=raw_b, packed_a=packed_a, packed_b=packed_b, broadcast_acc=True)
      else:
        leaf((a, b), output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
             raw_a=raw_a, raw_b=raw_b, packed_a=packed_a, packed_b=packed_b)
    else:
      bm = span if BM is None else BM
      bn = span if BN is None else BN
      bk = T.max(quantum, T.min(512, T.cache_bytes("L1") // 2 //
                    (bm * T.itemsize(a.dtype) + bn * T.itemsize(b.dtype))) // quantum * quantum) if BK is None else BK
      assert bm > 0 and bn > 0 and bk > 0, "matrix cache blocks must be positive"
      # Cache panels must cover many register tiles. At 64x256x256 FP64,
      # using one 16x16 SME tile as a cache panel repeated packing/ownership
      # 128 times and measured ~53x C++ latency on 920f-4. Keep K's L1
      # budget, but give each operand up to a quarter of L2 for spatial reuse.
      # Explicit blocks retain their exact (possibly nonaligned) semantics.
      if BM is None:
        bm = T.max(span, T.min(m, T.cache_bytes("L2") // 4 // bk // T.itemsize(a.dtype) // span * span))
      if BN is None:
        bn = T.max(span, T.min(n, T.cache_bytes("L2") // 4 // bk // T.itemsize(b.dtype) // span * span))
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
