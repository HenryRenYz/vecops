"""Composable matrix algorithms built from ordinary value/control primitives.

These recipes are not yet the default dot dispatcher. Backend selection,
packing and cache traversal remain separately gated while codegen/performance
equivalence is established. No C++ spelling or native operator is embedded.
"""

from .. import dsl as T


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
    else:
      result = result + left * right
  return result


@T.func
def _vector_row_update(value, columns, a, row, k, dtype):
  scalar = T.cast(T.element_at(a, (row, k)), dtype)
  return T.fma(scalar, columns, value)


@T.func
def _vector_row_store(value, output, row, column, width, epilogue):
  if epilogue is not None:
    value = epilogue(value, row, column)
  T.store(output[row, column : column + width], value)


@T.func
def _vector_dot_core(inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, unroll=2):
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
  T.static_assert(acc_dtype == "float32" or acc_dtype == "float64")
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
      for k in T.serial(a.shape[1], unroll=unroll):
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
def vector_dot_into(inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, unroll=2):
  """Version the streaming leaf under whole-producer disjointness proofs.

  A proven disjoint branch avoids repeated loop-level alias preservation.
  The general branch retains ordinary historical snapshots; no unchecked
  noalias promise or operator-specific compiler exemption is introduced.
  """
  a, b = inputs
  if acc is None:
    if T.view_disjoint(a, output) and T.view_disjoint(b, output):
      _vector_dot_core((a, b), output, acc_dtype=acc_dtype, epilogue=epilogue, unroll=unroll)
    else:
      _vector_dot_core((a, b), output, acc_dtype=acc_dtype, epilogue=epilogue, unroll=unroll)
  else:
    if T.view_disjoint(a, output) and T.view_disjoint(b, output) and T.view_disjoint(acc, output):
      _vector_dot_core((a, b), output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue, unroll=unroll)
    else:
      _vector_dot_core((a, b), output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue, unroll=unroll)
