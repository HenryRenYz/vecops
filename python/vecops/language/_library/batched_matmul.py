# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Serial batched products and explicit shared-operand flattening in the DSL."""

from .. import dsl as T


@T.func
def _product(inputs, output, acc, acc_dtype, BM, BN, BK, packing, swap, implementation):
  return T._matmul(
    inputs,
    output,
    acc=acc,
    acc_dtype=acc_dtype,
    BM=BM,
    BN=BN,
    BK=BK,
    packing=packing,
    swap=swap,
    implementation=implementation,
  )


@T.func
def batched_matmul(
  inputs,
  output,
  acc=None,
  *,
  acc_dtype=None,
  BM=None,
  BN=None,
  BK=None,
  packing="auto",
  swap=True,
  implementation="auto",
  strategy="auto",
  raw_inputs=None,
  workspace=None,
  parallel="serial",
  shared_packing=False,
):
  """Compute [batch,M,K] @ [batch,K,N], allowing either input to be shared.

  Inputs are logical views or computation values with rank two or three;
  rank-two inputs are shared, and a rank-three batch extent of one broadcasts.
  Output is a writable rank-three view. Initializers broadcast to its shape.
  ``auto`` uses optional raw_inputs layout hints for bounded shared-input
  folds; hints select scheduling but never replace producer reads. Missing
  or unmergeable hints retain serial products. ``serial`` preserves
  independent rank-two products. Explicit ``rows`` folds
  batch*M for shared B; ``columns`` folds batch*N for shared A. These are
  logical transforms: unusual strides may need staging and are
  never asserted to be zero-copy. Block/packing options reach each rank-two
  product unchanged; the folded shape is the scheduling problem in a fold.

  A complete private result delays output writes until every product has
  consumed its inputs and initializer, protecting cross-batch A/C and B/C
  overlap. The returned value can be used by the caller's ordinary pointwise
  epilogue/store, as with library.matmul. No automatic parallelism or persistent
  packed-content cache is implied. Explicit parallel="auto"/"require" uses
  a batch grid instead of automatic flattening; only the existing grid
  ownership planner can enable workers, with instruction scratch private to
  each worker. Explicit shared_packing=True prepares any rank-two shared
  operand once before the grid or serial traversal. An optional rank-three ``workspace``
  reuses caller-declared T.workspace storage of the exact output shape and
  accumulation dtype. Its disjointness from all producer reads, initializer
  and output is checked before use. Contents are overwritten on every call;
  declare the workspace at kernel top level, outside loops and branches.
  """
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  T.static_assert(len(a.shape) == 2 or len(a.shape) == 3)
  T.static_assert(len(b.shape) == 2 or len(b.shape) == 3)
  T.static_assert(len(output.shape) == 3)
  T.static_assert(strategy == "auto" or strategy == "serial" or strategy == "rows" or strategy == "columns")
  T.static_assert(parallel == "serial" or parallel == "auto" or parallel == "require")
  if parallel != "serial":
    T.static_assert(strategy == "auto" or strategy == "serial")
  batches, m, n = output.shape
  k = a.shape[-1]
  assert a.shape[-2] == m and b.shape[-2] == k and b.shape[-1] == n, "batched matmul shape mismatch"
  if len(a.shape) == 3:
    assert a.shape[0] == batches or a.shape[0] == 1, "incompatible A batch extent"
  if len(b.shape) == 3:
    assert b.shape[0] == batches or b.shape[0] == 1, "incompatible B batch extent"
  if acc_dtype is None:
    if (a.dtype == "int8" or a.dtype == "uint8") and (b.dtype == "int8" or b.dtype == "uint8"):
      acc_dtype = "int32"
    elif (a.dtype == "int16" or a.dtype == "uint16") and (b.dtype == "int16" or b.dtype == "uint16"):
      acc_dtype = "int64"
    elif a.dtype == "float64" or b.dtype == "float64":
      acc_dtype = "float64"
    else:
      acc_dtype = "float32"
  acc_dtype = T.cast(0, acc_dtype).dtype
  if workspace is None:
    partial = T.alloc(output.shape, dtype=acc_dtype)
  else:
    T.static_assert(len(workspace.shape) == 3 and workspace.dtype == acc_dtype)
    assert workspace.shape[0] == batches and workspace.shape[1] == m and workspace.shape[2] == n, (
      "batch workspace shape mismatch"
    )
    assert T.view_disjoint(workspace, a) and T.view_disjoint(workspace, b), "batch workspace overlaps input producer"
    assert T.view_disjoint(workspace, output), "batch workspace overlaps output"
    if acc is not None:
      assert T.view_disjoint(workspace, T.as_value(acc)), "batch workspace overlaps initializer"
    partial = workspace
  if acc is not None:
    initial = T.broadcast_to(T.cast(T.as_value(acc), acc_dtype), output.shape)
  if raw_inputs is not None:
    raw_a, raw_b = raw_inputs
    T.static_assert(len(raw_a.shape) == len(a.shape) and len(raw_b.shape) == len(b.shape))
    assert raw_a.shape[-2] == m and raw_a.shape[-1] == k, "raw A hint shape mismatch"
    assert raw_b.shape[-2] == k and raw_b.shape[-1] == n, "raw B hint shape mismatch"
    if len(a.shape) == 3:
      assert raw_a.shape[0] == a.shape[0], "raw A hint batch mismatch"
    if len(b.shape) == 3:
      assert raw_b.shape[0] == b.shape[0], "raw B hint batch mismatch"
  if shared_packing:
    if len(a.shape) == 2:
      a = T.pack(a, side="A", format="auto")
    if len(b.shape) == 2:
      b = T.pack(b, side="B", format="auto")
  if strategy == "auto" and raw_inputs is not None and parallel == "serial":
    if _auto_rows(a, b, raw_a, batches, m, n, k):
      _execute(
        a,
        b,
        partial,
        None if acc is None else initial,
        acc_dtype,
        BM,
        BN,
        BK,
        packing,
        swap,
        implementation,
        "rows",
        parallel,
      )
    elif _auto_columns(a, b, raw_b, batches, m, n, k):
      _execute(
        a,
        b,
        partial,
        None if acc is None else initial,
        acc_dtype,
        BM,
        BN,
        BK,
        packing,
        swap,
        implementation,
        "columns",
        parallel,
      )
    else:
      _execute(
        a,
        b,
        partial,
        None if acc is None else initial,
        acc_dtype,
        BM,
        BN,
        BK,
        packing,
        swap,
        implementation,
        "serial",
        parallel,
      )
  else:
    _execute(
      a,
      b,
      partial,
      None if acc is None else initial,
      acc_dtype,
      BM,
      BN,
      BK,
      packing,
      swap,
      implementation,
      "serial" if strategy == "auto" else strategy,
      parallel,
    )
  result = T.load(partial)
  T.store(output, result)
  return result


@T.func
def _execute(a, b, partial, initial, acc_dtype, BM, BN, BK, packing, swap, implementation, mode, parallel):
  batches, m, n = partial.shape
  k = a.shape[-1]
  if mode == "rows":
    T.static_assert(len(a.shape) == 3)
    if len(b.shape) == 3:
      assert b.shape[0] == 1, "row flatten requires shared B"
      shared_b = b[0, :, :]
    else:
      shared_b = b
    assert a.shape[0] == batches, "row flatten requires an unbroadcast A batch"
    flat_a = T.reshape(a, (batches * m, k))
    flat_output = T.reshape(partial, (batches * m, n))
    _product(
      (flat_a, shared_b),
      flat_output,
      None if initial is None else T.reshape(initial, (batches * m, n)),
      acc_dtype,
      BM,
      BN,
      BK,
      packing,
      swap,
      implementation,
    )
  elif mode == "columns":
    T.static_assert(len(b.shape) == 3)
    if len(a.shape) == 3:
      assert a.shape[0] == 1, "column flatten requires shared A"
      shared_a = a[0, :, :]
    else:
      shared_a = a
    assert b.shape[0] == batches, "column flatten requires an unbroadcast B batch"
    flat_b = T.reshape(T.transpose(b, axes=(1, 0, 2)), (k, batches * n))
    flat_output = T.alloc((m, batches * n), dtype=acc_dtype)
    _product(
      (shared_a, flat_b),
      flat_output,
      None if initial is None else T.reshape(T.transpose(initial, axes=(1, 0, 2)), (m, batches * n)),
      acc_dtype,
      BM,
      BN,
      BK,
      packing,
      swap,
      implementation,
    )
    for column_batch in T.serial(batches, unroll=1):
      T.store(partial[column_batch, :, :], T.load(flat_output[:, column_batch * n : (column_batch + 1) * n]))
  else:
    if parallel == "serial":
      for batch in T.serial(batches, unroll=1):
        _batch_program(a, b, partial, initial, batch, acc_dtype, BM, BN, BK, packing, swap, implementation)
    else:
      for batch in T.grid(batches, parallel=parallel):
        _batch_program(a, b, partial, initial, batch, acc_dtype, BM, BN, BK, packing, swap, implementation)


@T.func
def _batch_program(a, b, partial, initial, batch, acc_dtype, BM, BN, BK, packing, swap, implementation):
  left = a if len(a.shape) == 2 else a[0 if a.shape[0] == 1 else batch, :, :]
  right = b if len(b.shape) == 2 else b[0 if b.shape[0] == 1 else batch, :, :]
  _product(
    (left, right),
    partial[batch, :, :],
    None if initial is None else initial[batch, :, :],
    acc_dtype,
    BM,
    BN,
    BK,
    packing,
    swap,
    implementation,
  )


@T.func
def _auto_rows(a, b, raw_a, batches, m, n, k):
  if len(a.shape) == 3:
    shared_b = True if len(b.shape) == 2 else b.shape[0] == 1
    result = (
      shared_b
      and a.shape[0] == batches
      and batches > 1
      and m > 0
      and m <= 16
      and n > 0
      and k > 0
      and batches <= 64 // T.max(1, m)
      and raw_a.strides[2] == 1
      and raw_a.strides[1] == k
      and raw_a.strides[0] == m * k
    )
  else:
    result = False
  return result


@T.func
def _auto_columns(a, b, raw_b, batches, m, n, k):
  if len(b.shape) == 3:
    shared_a = True if len(a.shape) == 2 else a.shape[0] == 1
    result = (
      shared_a
      and b.shape[0] == batches
      and batches >= 4
      and m == 1
      and n > 0
      and k > 0
      and n <= 64 // T.max(1, batches)
      and raw_b.strides[2] == 1
      and raw_b.strides[0] == n
      and raw_b.strides[1] == batches * n
    )
  else:
    result = False
  return result
