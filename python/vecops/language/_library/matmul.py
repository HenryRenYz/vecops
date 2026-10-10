# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Composable matrix algorithms built from ordinary value/control primitives.

Dot, cache traversal, packing and epilogues use these recipes by default.
Instruction-specific families can specialize these ordinary DSL algorithms;
no C++ spelling or native operator is embedded in the source algorithms.
"""

from .. import dsl as T


@T.func
def vector_dot_into(
  inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, unroll=2, prefer_k=False, pipeline="off", distance=1
):
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  if a.shape[0] > 0 and b.shape[1] > 0:
    _vector_dot_disjoint(
      (a, b),
      output,
      acc=acc,
      acc_dtype=acc_dtype,
      epilogue=epilogue,
      unroll=unroll,
      prefer_k=prefer_k,
      pipeline=pipeline,
      distance=distance,
    )


@T.func
def blocked_dot_into(
  inputs,
  output,
  *,
  acc=None,
  acc_dtype="float32",
  epilogue=None,
  BM=None,
  BN=None,
  BK=None,
  packing="auto",
  unroll=2,
  pipeline="off",
  distance=1,
  prefer_k=False,
):
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  if a.shape[0] > 0 and b.shape[1] > 0:
    if (
      BM is None
      and BN is None
      and BK is None
      and packing == "auto"
      and (
        a.shape[0] <= T.cache_bytes("L2") // 4 // T.itemsize(acc_dtype) // b.shape[1]
        and a.shape[1]
        <= T.cache_bytes("L2") // 2 // (a.shape[0] * T.itemsize(a.dtype) + b.shape[1] * T.itemsize(b.dtype))
      )
    ):
      vector_dot_into(
        (a, b),
        output,
        acc=acc,
        acc_dtype=acc_dtype,
        epilogue=epilogue,
        unroll=unroll,
        prefer_k=prefer_k,
        pipeline=pipeline,
        distance=distance,
      )
    else:
      _blocked_dot_core(
        (a, b),
        output,
        acc=acc,
        acc_dtype=acc_dtype,
        epilogue=epilogue,
        BM=BM,
        BN=BN,
        BK=BK,
        packing=packing,
        unroll=unroll,
        pipeline=pipeline,
        distance=distance,
      )


@T.func
def vector_tile_dot(inputs, *, acc=None, acc_dtype="float32", unroll=2):
  """Outer-product FMA microtile with register-eligible carried state.

  Constant small tiles are placed into independent register pieces by the
  ordinary state planner. Larger/dynamic tiles retain its safe scratch path;
  callers must choose a microtile rather than assume an entire cache tile
  fits registers. Input transforms and helper results remain argument DAGs.
  """
  a, b = inputs
  a = T.as_value(a)
  b = T.as_value(b)
  T.static_assert(len(a.shape) == 2 and len(b.shape) == 2)
  assert a.shape[1] == b.shape[0], "incompatible vector dot reduction extents"
  if acc is None:
    result = T.zeros((a.shape[0], b.shape[1]), dtype=acc_dtype)
  else:
    result = T.broadcast_to(T.cast(T.as_value(acc), acc_dtype), (a.shape[0], b.shape[1]))
  for k in T.serial(a.shape[1], unroll=unroll):
    left = T.expand_dims(T.cast(a[:, k], acc_dtype), 1)
    right = T.expand_dims(T.cast(b[k, :], acc_dtype), 0)
    if acc_dtype == "float32" or acc_dtype == "float64":
      result = T.fma(left, right, result)
    elif acc_dtype == "int32":
      result = T.bitcast(T.bitcast(result, "uint32") + T.bitcast(left, "uint32") * T.bitcast(right, "uint32"), "int32")
    else:
      result = result + left * right
  return result


@T.func
def _vector_row_update(value, columns, a, row, k, dtype):
  scalar = T.cast(T.element_at(a, (row, k)), dtype)
  if dtype == "float32" or dtype == "float64":
    return T.fma(scalar, columns, value)
  return T.bitcast(T.bitcast(value, "uint32") + T.bitcast(scalar, "uint32") * T.bitcast(columns, "uint32"), "int32")


@T.func
def _vector_row_store(value, output, row, column, width, epilogue):
  if epilogue is not None:
    transformed = epilogue(value, row, column)
  else:
    transformed = value
  # SIMD induction carries the bounded register extent through indexed-memory
  # canonicalization. A generic tile store otherwise groups multiple words
  # and reconstructs a clipped one-word state in its wider tail traversal.
  region = output[row, column : column + width]
  for indices in T.simd(region.shape[0], vector=T.Vector(value.dtype), unroll=1):
    T.scatter(region, indices, transformed)


@T.func
def _vector_dot_core(
  inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, unroll=2, pipeline="off", distance=1
):
  """Four-row streaming vector family in ordinary DSL, including both tails.

  This experimental leaf has no cache blocking or packing. The caller picks
  the outer-product family; K-vector GEMV and native matrix instruction
  families remain separate algorithms. ``epilogue`` is an optional T.func
  taking (vector, global_row, global_column), not a host Python callable.
  Register values are independent source variables, never native arrays.
  """
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  T.static_assert(len(a.shape) == 2 and len(b.shape) == 2 and len(output.shape) == 2)
  T.static_assert(acc_dtype == "float32" or acc_dtype == "float64" or acc_dtype == "int32")
  assert a.shape[1] == b.shape[0], "incompatible vector dot reduction extents"
  assert output.shape[0] == a.shape[0] and output.shape[1] == b.shape[1], "vector dot output shape mismatch"
  descriptor = T.Vector(acc_dtype)
  width = T.lanes(descriptor)
  initial = None if acc is None else T.broadcast_to(T.cast(T.as_value(acc), acc_dtype), output.shape)
  for row in T.serial(0, a.shape[0], 4, unroll=1):
    for column in T.serial(0, b.shape[1], width, unroll=1):
      panel = b[:, column : column + width]
      x0 = T.zeros(descriptor, active=panel.shape[1])
      x1 = T.zeros(descriptor, active=panel.shape[1])
      x2 = T.zeros(descriptor, active=panel.shape[1])
      x3 = T.zeros(descriptor, active=panel.shape[1])
      if initial is not None:
        if row < a.shape[0]:
          x0 = T.as_vector(initial[row, column : column + width], descriptor)
        if row + 1 < a.shape[0]:
          x1 = T.as_vector(initial[row + 1, column : column + width], descriptor)
        if row + 2 < a.shape[0]:
          x2 = T.as_vector(initial[row + 2, column : column + width], descriptor)
        if row + 3 < a.shape[0]:
          x3 = T.as_vector(initial[row + 3, column : column + width], descriptor)
      k_loop = T.serial(a.shape[1], unroll=unroll)
      if pipeline != "off":
        k_loop = T.pipelined(
          a.shape[1], unroll=unroll, distance=distance, buffering=pipeline == "staged" and distance == 1
        )
      for k in k_loop:
        columns = T.cast(panel[k, :], acc_dtype)
        if row < a.shape[0]:
          x0 = _vector_row_update(x0, columns, a, row, k, acc_dtype)
        if row + 1 < a.shape[0]:
          x1 = _vector_row_update(x1, columns, a, row + 1, k, acc_dtype)
        if row + 2 < a.shape[0]:
          x2 = _vector_row_update(x2, columns, a, row + 2, k, acc_dtype)
        if row + 3 < a.shape[0]:
          x3 = _vector_row_update(x3, columns, a, row + 3, k, acc_dtype)
      if row < a.shape[0]:
        _vector_row_store(x0, output, row, column, width, epilogue)
      if row + 1 < a.shape[0]:
        _vector_row_store(x1, output, row + 1, column, width, epilogue)
      if row + 2 < a.shape[0]:
        _vector_row_store(x2, output, row + 2, column, width, epilogue)
      if row + 3 < a.shape[0]:
        _vector_row_store(x3, output, row + 3, column, width, epilogue)


@T.func
def _vector_inner_dot_into(inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, unroll=2):
  """K-vectorized inner products for column GEMV/K-contiguous B layouts.

  Reduction, conversion and input transforms use the ordinary reducer
  pipeline. No scheduler-specific vector reduction is generated in C++.
  """
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  initial = None if acc is None else T.broadcast_to(T.cast(T.as_value(acc), acc_dtype), output.shape)
  for row in T.serial(a.shape[0], unroll=1):
    for column in T.serial(b.shape[1], unroll=1):
      if acc_dtype == "int32":
        product = T.bitcast(
          T.reduce_add(
            T.bitcast(T.cast(a[row, :], "int32"), "uint32") * T.bitcast(T.cast(b[:, column], "int32"), "uint32"),
            unroll=unroll,
          ),
          "int32",
        )
      else:
        product = T.reduce_add(T.cast(a[row, :], acc_dtype) * T.cast(b[:, column], acc_dtype), unroll=unroll)
      value = T.as_vector(T.full((1,), product, dtype=acc_dtype), T.Vector(acc_dtype, N=1))
      if initial is not None:
        if acc_dtype == "int32":
          value = T.bitcast(
            T.bitcast(value, "uint32") + T.bitcast(T.element_at(initial, (row, column)), "uint32"), "int32"
          )
        else:
          value = value + T.cast(T.element_at(initial, (row, column)), acc_dtype)
      _vector_row_store(value, output, row, column, 1, epilogue)


@T.func
def _vector_dispatch(
  inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, unroll=2, prefer_k=False, pipeline="off", distance=1
):
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  if prefer_k or b.shape[1] == 1:
    _vector_inner_dot_into((a, b), output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue, unroll=unroll)
  else:
    _vector_dot_core(
      (a, b),
      output,
      acc=acc,
      acc_dtype=acc_dtype,
      epilogue=epilogue,
      unroll=unroll,
      pipeline=pipeline,
      distance=distance,
    )


@T.func
def _vector_dot_disjoint(
  inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, unroll=2, prefer_k=False, pipeline="off", distance=1
):
  """Version the streaming leaf under whole-producer disjointness proofs.

  A proven disjoint branch avoids repeated loop-level alias preservation.
  The general branch retains ordinary historical snapshots; no unchecked
  noalias promise or operator-specific compiler exemption is introduced.
  """
  a, b = inputs
  if acc is None:
    if T.view_disjoint(a, output) and T.view_disjoint(b, output):
      _vector_dispatch(
        (a, b),
        output,
        acc_dtype=acc_dtype,
        epilogue=epilogue,
        unroll=unroll,
        prefer_k=prefer_k,
        pipeline=pipeline,
        distance=distance,
      )
    else:
      _vector_dispatch(
        (a, b),
        output,
        acc_dtype=acc_dtype,
        epilogue=epilogue,
        unroll=unroll,
        prefer_k=prefer_k,
        pipeline=pipeline,
        distance=distance,
      )
  else:
    if T.view_disjoint(a, output) and T.view_disjoint(b, output) and T.view_disjoint(acc, output):
      _vector_dispatch(
        (a, b),
        output,
        acc=acc,
        acc_dtype=acc_dtype,
        epilogue=epilogue,
        unroll=unroll,
        prefer_k=prefer_k,
        pipeline=pipeline,
        distance=distance,
      )
    else:
      _vector_dispatch(
        (a, b),
        output,
        acc=acc,
        acc_dtype=acc_dtype,
        epilogue=epilogue,
        unroll=unroll,
        prefer_k=prefer_k,
        pipeline=pipeline,
        distance=distance,
      )


@T.func
def _panel_store(source, output, origin_row, origin_column, epilogue):
  """Apply the caller DAG once, after the final K phase, in global coordinates."""
  descriptor = T.Vector(source.dtype)
  width = T.lanes(descriptor)
  for row in T.serial(source.shape[0], unroll=1):
    for column in T.serial(0, source.shape[1], width, unroll=1):
      value = T.as_vector(T.load(source[row, column : column + width]), descriptor)
      if epilogue is not None:
        transformed = epilogue(value, origin_row + row, origin_column + column)
      else:
        transformed = value
      T.store(output[row, column : column + width], transformed)


@T.func
def _blocked_dot_core(
  inputs,
  output,
  *,
  acc=None,
  acc_dtype="float32",
  epilogue=None,
  BM=None,
  BN=None,
  BK=None,
  packing="auto",
  unroll=2,
  pipeline="off",
  distance=1,
):
  """Source-level cache traversal and K-phase accumulation.

  Cache capacities are queried on the execution host. Explicit sizes are
  exact iteration steps, not ISA dimensions. The private accumulator avoids
  repeatedly rounding a narrow output and invokes the epilogue only once.
  The ordinary capture/alias planner protects all input and seed reads.
  """
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  initial = None if acc is None else T.broadcast_to(T.cast(T.as_value(acc), acc_dtype), output.shape)
  width = T.lanes(T.Vector(acc_dtype))
  # Keep two operand panels plus the accumulator inside private cache. The
  # formula is intentionally inspectable DSL, not a native scheduler helper.
  k_block = (
    BK
    if BK is not None
    else T.max(1, T.cache_bytes("L1") // (2 * (4 * T.itemsize(a.dtype) + width * T.itemsize(b.dtype))))
  )
  budget = T.max(1, T.cache_bytes("L2") * 3 // 4)
  m_block = BM if BM is not None else T.max(4, T.min(256, budget // (3 * T.max(1, k_block) * T.itemsize(a.dtype))))
  n_block = BN if BN is not None else T.max(width, T.min(512, budget // (3 * T.max(1, k_block) * T.itemsize(b.dtype))))
  for row in T.serial(0, a.shape[0], m_block, unroll=1):
    for column in T.serial(0, b.shape[1], n_block, unroll=1):
      destination = output[row : row + m_block, column : column + n_block]
      work = T.alloc(destination.shape, dtype=acc_dtype)
      if initial is None:
        T.store(work, T.zeros(work.shape, dtype=acc_dtype))
      else:
        T.store(work, initial[row : row + m_block, column : column + n_block])
      for k in T.serial(0, a.shape[1], k_block, unroll=1):
        left = a[row : row + m_block, k : k + k_block]
        right = b[k : k + k_block, column : column + n_block]
        if packing == "auto" or packing == "A" or packing == "AB":
          left = T.pack(left, side="A", format="dense")
        if packing == "auto" or packing == "B" or packing == "AB":
          right = T.pack(right, side="B", format="dense")
        vector_dot_into(
          (left, right),
          work,
          acc=T.load(work),
          acc_dtype=acc_dtype,
          unroll=unroll,
          prefer_k=packing == "auto" or packing == "B" or packing == "AB",
          pipeline=pipeline,
          distance=distance,
        )
      _panel_store(work, destination, row, column, epilogue)
