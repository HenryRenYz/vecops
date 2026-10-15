# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared source-level packing, accumulator and epilogue boundaries.

Instruction owners consume bounded raw/packed views or private staging.
Privately constructed padding never evaluates a producer outside its logical
domain. Producer staging precedes ownership and output writes, retaining
captured-read and input/output alias semantics.
"""

from .. import dsl as T
from .. import sme


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
def _invoke_leaf(inputs, output, leaf, *, acc=None, acc_dtype="float32", epilogue=None,
                 raw_a=None, raw_b=None, packed_a=None, packed_b=None,
                 broadcast_acc=False, pack_raw_inputs=False):
  # This static capability flag is enabled only for an SME leaf. AMX never
  # receives SME-specific keywords; no runtime family selection is involved.
  if pack_raw_inputs:
    leaf(inputs, output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
         raw_a=raw_a, raw_b=raw_b, packed_a=packed_a, packed_b=packed_b,
         broadcast_acc=broadcast_acc, pack_raw_inputs=True)
  elif broadcast_acc:
    leaf(inputs, output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
         raw_a=raw_a, raw_b=raw_b, packed_a=packed_a, packed_b=packed_b, broadcast_acc=True)
  else:
    leaf(inputs, output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
         raw_a=raw_a, raw_b=raw_b, packed_a=packed_a, packed_b=packed_b)


@T.func
def whole_packed_problem(packed_pair, BM, BN, BK, packing):
  """One static whole-problem policy shared by native and vector fallbacks."""
  return packed_pair and packing == "auto" and BM is None and BN is None and BK is None


@T.func
def _prepared_leaf(inputs, output, leaf, prepared_b, *, acc, acc_dtype, raw_a):
  # The panel belongs to this exact logical N/K phase. It retains the
  # producer's real physical strides, including private K-chunk padding.
  # It is not a flattened public packed descriptor or a packed subview.
  leaf(inputs, output, acc=acc, acc_dtype=acc_dtype, raw_a=raw_a, prepared_b=prepared_b)


@T.func
def blocked_matrix(inputs, output, leaf, *, acc=None, acc_dtype="float32", epilogue=None,
                   BM=None, BN=None, BK=None, span=32, quantum=32, packing="auto",
                   raw_a=None, raw_b=None, packed_a=None, packed_b=None, broadcast_acc=False,
                   pack_raw_inputs=False, prepare_b=None, cache_raw_b=None, whole_b_panel=False, prepare_a=None):
  """Cache traversal with whole-problem execution for externally packed pairs.

  An unconfigured automatic caller with both physical packed descriptors
  enters its leaf directly. The choice is static even for dynamic logical
  extents, so cache traversal and its private full accumulator are not
  expanded. Explicit block sizes or packing policies retain exact traversal.
  Physical packed metadata is forwarded only for the original whole problem.
  """
  if whole_packed_problem(packed_a is not None and packed_b is not None, BM, BN, BK, packing):
    _invoke_leaf(inputs, output, leaf, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
                 raw_a=raw_a, raw_b=raw_b, packed_a=packed_a, packed_b=packed_b,
                 broadcast_acc=broadcast_acc, pack_raw_inputs=pack_raw_inputs)
  else:
    _blocked_matrix_core(inputs, output, leaf, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
                         BM=BM, BN=BN, BK=BK, span=span, quantum=quantum, packing=packing,
                         raw_a=raw_a, raw_b=raw_b, packed_a=packed_a, packed_b=packed_b,
                         broadcast_acc=broadcast_acc, pack_raw_inputs=pack_raw_inputs, prepare_b=prepare_b,
                         cache_raw_b=cache_raw_b, whole_b_panel=whole_b_panel, prepare_a=prepare_a)


@T.func
def _blocked_matrix_core(inputs, output, leaf, *, acc=None, acc_dtype="float32", epilogue=None,
                         BM=None, BN=None, BK=None, span=32, quantum=32, packing="auto",
                         raw_a=None, raw_b=None, packed_a=None, packed_b=None, broadcast_acc=False,
                         pack_raw_inputs=False, prepare_b=None, cache_raw_b=None, whole_b_panel=False, prepare_a=None):
  """Cache phases: initializer once, continuation, epilogue once.

  Logical slicing keeps producer DAGs and already-packed logical operands
  intact. Each hardware leaf handles physical padding independently of an
  explicit user's cache tile sizes, including a nonaligned final K phase.
  An explicit dense B cache pack is shared across every M block in one N/K
  phase. It is a logical dense panel, not a sliced native packed-buffer ABI;
  An automatic native caller may provide its own prepare_b helper to build
  physical B panels at the same reuse boundary. Grouped SME callers may also
  supply prepare_a: bounded full-K inputs then reuse A across N panels and
  retain all K accumulation in the leaf. Explicit cache policies keep their
  requested phase schedule. Output is still private
  until every phase completes, preserving captured inputs and aliasing.
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
      _invoke_leaf((a, b), output, leaf, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
                   raw_a=raw_a, raw_b=raw_b, packed_a=packed_a, packed_b=packed_b,
                   broadcast_acc=broadcast_acc, pack_raw_inputs=pack_raw_inputs)
    else:
      # Preserve the established AMX/raw-float expansion in this lexical
      # body. Extra cold helper bindings changed GCC's hot resident kernel
      # and regressed the small Dynamic BF16 case despite identical dispatch.
      # Only the grouped SME capability needs the additional cache selector.
      if prepare_a is not None:
        _cache_matrix((a, b), output, leaf, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
                      BM=BM, BN=BN, BK=BK, span=span, quantum=quantum, packing=packing,
                      raw_a=raw_a, raw_b=raw_b, pack_raw_inputs=pack_raw_inputs,
                      prepare_b=prepare_b, cache_raw_b=cache_raw_b, whole_b_panel=whole_b_panel,
                      prepare_a=prepare_a, packed_pair=packed_a is not None or packed_b is not None)
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
        if (whole_b_panel and prepare_b is not None and cache_raw_b is not None and
            packing == "auto" and BM is None and BN is None and BK is None):
          # AMX BF16 WholeProblem: one bounded full-K B panel, consumed by
          # all M tiles. Mirror the C++ reuse/budget thresholds conservatively
          # for aligned K; explicit cache blocks never change their schedule.
          panel_n = T.min(256, T.max(64, m // 4 // 16 * 16))
          if (cache_raw_b.strides[0] == 1 and m >= 32 and n > panel_n and k > 0 and
              k % 32 == 0 and k <= 524288 // panel_n and
              (n >= 1 + 8388607 // T.max(1, k) or
               (m >= 512 and n >= T.max(512, 1 + 1048575 // T.max(1, k)) and k >= 256))):
            bm, bn, bk = T.max(1, m), panel_n, T.max(1, k)
        partial = T.alloc((m, n), dtype=acc_dtype)
        seed_accumulator(acc, partial, m, n, acc_dtype)
        # Runtime block selection promotes bm/bn/bk to mutable scalars;
        # retain an explicit positive-step proof at the traversal boundary.
        for ni in T.serial(0, n, T.max(1, bn) if prepare_b is not None else bn, unroll=1):
          for ki in T.serial(0, k, T.max(1, bk) if prepare_b is not None else bk, unroll=1):
            end_n, end_k = T.min(ni + bn, n), T.min(ki + bk, k)
            right = b[ki:end_k, ni:end_n]
            # B is identical for every M block. Build the optional dense copy
            # once at its reuse boundary, preserving the original producer DAG
            # and conversion points. Native packed inputs are decoded through
            # their original logical slice; their physical origin/padding must
            # never be reinterpreted as a directly sliceable packed panel.
            if packing == "B" or packing == "AB":
              right = T.pack(right, side="B", format="dense")
            if prepare_b is not None and packing == "auto":
              right_source = raw_b if cache_raw_b is None else cache_raw_b
              right_layout = None if right_source is None else right_source[ki:end_k, ni:end_n]
              prepared = prepare_b(right, acc_dtype, raw=right_layout)
            for mi in T.serial(0, m, T.max(1, bm) if prepare_b is not None else bm, unroll=1):
              end_m = T.min(mi + bm, m)
              left = a[mi:end_m, ki:end_k]
              # Explicit optional dense-cache policies are distinct from the
              # physical grouping/padding required by a matrix instruction.
              if packing == "A" or packing == "AB":
                left = T.pack(left, side="A", format="dense")
              if prepare_b is not None and packing == "auto":
                left_layout = None if raw_a is None else raw_a[mi:end_m, ki:end_k]
                _prepared_leaf((left, right), partial[mi:end_m, ni:end_n], leaf, prepared,
                               acc=T.load(partial[mi:end_m, ni:end_n]), acc_dtype=acc_dtype,
                               raw_a=left_layout)
              else:
                _invoke_leaf((left, right), partial[mi:end_m, ni:end_n], leaf,
                             acc=T.load(partial[mi:end_m, ni:end_n]), acc_dtype=acc_dtype,
                             pack_raw_inputs=pack_raw_inputs)
        finish_matrix(partial, output, epilogue=epilogue)


@T.func
def _cache_matrix(inputs, output, leaf, *, acc, acc_dtype, epilogue,
                  BM, BN, BK, span, quantum, packing, raw_a, raw_b,
                  pack_raw_inputs, prepare_b, cache_raw_b, whole_b_panel, prepare_a, packed_pair):
  a, b = inputs
  m, k = a.shape[0], a.shape[1]
  if (prepare_a is not None and prepare_b is not None and not packed_pair and
      packing == "auto" and BM is None and BN is None and BK is None):
    # Full-K SME traversal retains A across every N panel. Bound physical
    # padding as well as logical payload: A <= 4 MiB, B <= 1 MiB. The
    # original N/K/M cache schedule remains available for oversized K/A.
    chunk = sme.streaming_lanes(a.dtype)
    assert k <= 9223372036854775807 - (chunk - 1), "SME packing extent overflow"
    padded_k = T.ceildiv(k, chunk) * chunk
    if (k > 0 and padded_k <= 1048576 // span // T.itemsize(b.dtype) and
        T.ceildiv(m, span) <= 4194304 // span // T.itemsize(a.dtype) // T.max(1, padded_k)):
      panel_n = T.max(span, T.min(256, 1048576 // T.itemsize(b.dtype) // T.max(1, padded_k)) // span * span)
      # Native BF16 with substantial M reuse can amortize one bounded whole-B
      # preparation. Cost physical spatial/K padding, not logical payload;
      # all other producers retain the 1 MiB panel policy above.
      if (raw_a is not None and raw_b is not None and a.dtype == "bfloat16" and
          b.dtype == "bfloat16" and acc_dtype == "float32"):
        if (m >= 128 and raw_a.strides[1] == 1 and raw_b.strides[0] == 1 and
            T.ceildiv(b.shape[1], span) <= 33554432 // span // T.itemsize(b.dtype) // T.max(1, padded_k)):
          panel_n = T.max(1, b.shape[1])
      _whole_k_matrix(inputs, output, leaf, prepare_a, prepare_b, panel_n,
                      acc=acc, acc_dtype=acc_dtype, epilogue=epilogue, raw_a=raw_a, raw_b=raw_b)
    else:
      _phased_matrix(inputs, output, leaf, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
                     BM=BM, BN=BN, BK=BK, span=span, quantum=quantum, packing=packing,
                     raw_a=raw_a, raw_b=raw_b, pack_raw_inputs=pack_raw_inputs,
                     prepare_b=prepare_b, cache_raw_b=cache_raw_b, whole_b_panel=whole_b_panel)
  else:
    _phased_matrix(inputs, output, leaf, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue,
                   BM=BM, BN=BN, BK=BK, span=span, quantum=quantum, packing=packing,
                   raw_a=raw_a, raw_b=raw_b, pack_raw_inputs=pack_raw_inputs,
                   prepare_b=prepare_b, cache_raw_b=cache_raw_b, whole_b_panel=whole_b_panel)


@T.func
def _whole_k_matrix(inputs, output, leaf, prepare_a, prepare_b, panel_n, *,
                    acc, acc_dtype, epilogue, raw_a, raw_b):
  """Reuse full-K A, with direct output only after complete input capture.

  A single B panel and an absent epilogue permit output writes after both
  producers have been privately prepared. Snapshot any initializer first;
  its broadcast source may overlap C. Multiple B panels and epilogues retain
  the complete partial until all producer reads have finished.
  """
  a, b = inputs
  m, n = a.shape[0], b.shape[1]
  # No intermediate K-phase stores or initial zero-fill: the leaf initializes
  # its accumulator once and consumes all K. Delay all user output writes
  # until every input panel has been consumed, including A/C and B/C aliases.
  if epilogue is None and panel_n >= n:
    direct_a = prepare_a(a, acc_dtype, raw=raw_a)
    direct_b = prepare_b(b, acc_dtype, raw=raw_b)
    if acc is not None:
      direct_initial = T.alloc((m, n), dtype=acc_dtype)
      seed_accumulator(acc, direct_initial, m, n, acc_dtype)
    leaf((a, b), output, acc=None if acc is None else T.load(direct_initial),
         acc_dtype=acc_dtype, prepared_a=direct_a, prepared_b=direct_b)
  else:
    partial = T.alloc((m, n), dtype=acc_dtype)
    prepared_a = prepare_a(a, acc_dtype, raw=raw_a)
    if acc is not None:
      initial = T.broadcast_to(T.cast(T.as_value(acc), acc_dtype), (m, n))
    for ni in T.serial(0, n, T.max(1, panel_n), unroll=1):
      end_n = T.min(ni + panel_n, n)
      right = b[:, ni:end_n]
      right_layout = None if raw_b is None else raw_b[:, ni:end_n]
      prepared_b = prepare_b(right, acc_dtype, raw=right_layout)
      leaf((a, right), partial[:, ni:end_n], acc=None if acc is None else initial[:, ni:end_n],
           acc_dtype=acc_dtype, prepared_a=prepared_a, prepared_b=prepared_b)
    finish_matrix(partial, output, epilogue=epilogue)


@T.func
def _phased_matrix(inputs, output, leaf, *, acc, acc_dtype, epilogue,
                   BM, BN, BK, span, quantum, packing, raw_a, raw_b,
                   pack_raw_inputs, prepare_b, cache_raw_b, whole_b_panel):
  a, b = inputs
  m, n, k = a.shape[0], b.shape[1], a.shape[1]
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
  if (whole_b_panel and prepare_b is not None and cache_raw_b is not None and
      packing == "auto" and BM is None and BN is None and BK is None):
    # AMX BF16 WholeProblem: one bounded full-K B panel, consumed by
    # all M tiles. Mirror the C++ reuse/budget thresholds conservatively
    # for aligned K; explicit cache blocks never change their schedule.
    panel_n = T.min(256, T.max(64, m // 4 // 16 * 16))
    if (cache_raw_b.strides[0] == 1 and m >= 32 and n > panel_n and k > 0 and
        k % 32 == 0 and k <= 524288 // panel_n and
        (n >= 1 + 8388607 // T.max(1, k) or
         (m >= 512 and n >= T.max(512, 1 + 1048575 // T.max(1, k)) and k >= 256))):
      bm, bn, bk = T.max(1, m), panel_n, T.max(1, k)
  partial = T.alloc((m, n), dtype=acc_dtype)
  seed_accumulator(acc, partial, m, n, acc_dtype)
  # Runtime block selection promotes bm/bn/bk to mutable scalars;
  # retain an explicit positive-step proof at the traversal boundary.
  for ni in T.serial(0, n, T.max(1, bn) if prepare_b is not None else bn, unroll=1):
    for ki in T.serial(0, k, T.max(1, bk) if prepare_b is not None else bk, unroll=1):
      end_n, end_k = T.min(ni + bn, n), T.min(ki + bk, k)
      right = b[ki:end_k, ni:end_n]
      # B is identical for every M block. Build the optional dense copy
      # once at its reuse boundary, preserving the original producer DAG
      # and conversion points. Native packed inputs are decoded through
      # their original logical slice; their physical origin/padding must
      # never be reinterpreted as a directly sliceable packed panel.
      if packing == "B" or packing == "AB":
        right = T.pack(right, side="B", format="dense")
      if prepare_b is not None and packing == "auto":
        right_source = raw_b if cache_raw_b is None else cache_raw_b
        right_layout = None if right_source is None else right_source[ki:end_k, ni:end_n]
        prepared = prepare_b(right, acc_dtype, raw=right_layout)
      for mi in T.serial(0, m, T.max(1, bm) if prepare_b is not None else bm, unroll=1):
        end_m = T.min(mi + bm, m)
        left = a[mi:end_m, ki:end_k]
        # Explicit optional dense-cache policies are distinct from the
        # physical grouping/padding required by a matrix instruction.
        if packing == "A" or packing == "AB":
          left = T.pack(left, side="A", format="dense")
        if prepare_b is not None and packing == "auto":
          left_layout = None if raw_a is None else raw_a[mi:end_m, ki:end_k]
          _prepared_leaf((left, right), partial[mi:end_m, ni:end_n], leaf, prepared,
                         acc=T.load(partial[mi:end_m, ni:end_n]), acc_dtype=acc_dtype,
                         raw_a=left_layout)
        else:
          _invoke_leaf((left, right), partial[mi:end_m, ni:end_n], leaf,
                       acc=T.load(partial[mi:end_m, ni:end_n]), acc_dtype=acc_dtype,
                       pack_raw_inputs=pack_raw_inputs)
  finish_matrix(partial, output, epilogue=epilogue)
