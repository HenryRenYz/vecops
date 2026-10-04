"""General scalar/tuple folds expand into typed structured control IR.

Native add/min/max reductions retain the existing vector fast paths. Arbitrary
pure combiners have an ordered fallback, including coupled argmax-like state.
"""

from .. import ir as I
from ..ir.iteration import ReductionSpec
from ..ir.traversal import children, walk
from ..operations import Trait
from .capture import CapturedReducer, symbol


class ReducersFrontend:
  def fold(self, operation, args, kwargs, node):
    unroll = self.unroll_count(kwargs.get("unroll", I.literal(2)), node) if operation == "reduce" else 1
    if not args or len(args) > 2:
      self.error(node, "reduce/scan expects a value and a reducer")
    reducer = kwargs.get("reducer", args[1] if len(args) == 2 else "add")
    values = args[0]
    leaves = []

    def gather(value):
      if isinstance(value, tuple):
        return tuple(gather(v) for v in value)
      value = self.value(value, node)
      if not isinstance(value.type, I.TileType):
        self.error(node, "generic folds require tile values")
      value = self.retain_register_transform(value)
      leaves.append(value)
      return value

    values = gather(values)
    if not leaves:
      self.error(node, "fold input tuple cannot be empty")
    shape = leaves[0].type.shape
    for value in leaves[1:]:
      if len(value.type.shape) != len(shape):
        self.error(node, "tuple reduction inputs must have equal ranks")
      for a, b in zip(shape, value.type.shape):
        self.block.statements.append(I.Assert(I.binary("eq", a, b), "tuple reduction shape mismatch"))
    axis = kwargs.get("axis", I.literal(-1) if operation == "scan" else None)
    if axis is not None:
      axis = I.constant(self.value(axis, node))
      if type(axis) is not int or not -len(shape) <= axis < len(shape):
        self.error(node, "fold axis must be a compile-time in-range integer")
      axis %= len(shape)
    if isinstance(reducer, str):
      if reducer not in {"add", "min", "max"} or isinstance(values, tuple):
        self.error(node, "built-in reducer is add/min/max and requires one value")
      dtype = leaves[0].type.dtype
      if leaves[0].type.mask or dtype == "bool":
        self.error(node, "built-in numeric folds require numeric values; use mask queries for predicates")
      identity = 0 if reducer == "add" else float("inf") if reducer == "min" else float("-inf")
      if not dtype.startswith(("float", "bfloat")) and reducer != "add":
        bits = int(dtype.removeprefix("uint").removeprefix("int"))
        identity = (
          (1 << (bits - (not dtype.startswith("uint")))) - 1
          if reducer == "min"
          else 0
          if dtype.startswith("uint")
          else -(1 << (bits - 1))
        )
      if operation == "reduce":
        if axis is not None:
          output_shape = shape[:axis] + shape[axis + 1 :]
          if output_shape:
            return I.Expr(I.TileType(dtype, output_shape), "axis_reduce", (values,), (axis, reducer, unroll))
        return I.Expr(I.ScalarType(dtype), "reduce_" + reducer, (values,), unroll)
    elif isinstance(reducer, CapturedReducer):
      identity = reducer.identity
    else:
      self.error(node, "reducer must be add/min/max or a T.reducer helper")
    exclusive = I.constant(self.value(kwargs.get("exclusive", False), node))
    reverse = I.constant(self.value(kwargs.get("reverse", False), node))
    if type(exclusive) is not bool or type(reverse) is not bool:
      self.error(node, "scan exclusive/reverse must be compile-time booleans")
    output_shape = shape if operation == "scan" else shape[:axis] + shape[axis + 1 :] if axis is not None else ()
    parent = self.block
    self.block = I.Block()
    outer_axes = tuple(i for i in range(len(shape)) if i != axis) if axis is not None else ()
    coords = [None] * len(shape)
    outer_loops = []
    for a in outer_axes:
      name = self.unique("fold_axis")
      coords[a] = I.Expr(I.ScalarType("int64"), "local", data=name)
      outer_loops.append((name, shape[a]))

    def slots(value, initial):
      if isinstance(value, tuple):
        if not isinstance(initial, tuple) or len(initial) != len(value):
          self.error(node, "reducer identity tuple arity mismatch")
        return tuple(slots(v, init) for v, init in zip(value, initial))
      name = self.unique("fold_acc")
      dtype = "bool" if value.type.mask else value.type.dtype
      self.block.statements.append(I.Assign(name, I.Expr(I.ScalarType(dtype), "cast", (symbol(initial),))))
      return I.Expr(I.ScalarType(dtype), "local", data=name)

    accumulators = slots(values, identity)
    name = self.unique("fold_index")
    index = I.Expr(I.ScalarType("int64"), "local", data=name)
    extent = shape[axis] if axis is not None else I.literal(1)
    if axis is None:
      for n in shape:
        extent = I.binary("mul", extent, n)
      remainder = index
      for a in reversed(range(len(shape))):
        coords[a] = I.binary("mod", remainder, shape[a])
        remainder = I.binary("floordiv", remainder, shape[a])
    else:
      coords[axis] = index

    def output(value):
      if isinstance(value, tuple):
        return tuple(output(v) for v in value)
      strides, size = [None] * len(output_shape), I.literal(1)
      for a in reversed(range(len(output_shape))):
        strides[a], size = size, I.binary("mul", size, output_shape[a])
      view = I.View(
        self.unique("fold_output"), value.type.dtype, output_shape, tuple(strides), I.literal(0), owned=True
      )
      parent.statements.append(I.Allocate(view))
      return view

    allocation_start = len(parent.statements)
    outputs = output(accumulators) if output_shape else None
    prefix = self.block
    self.block = I.Block()

    def point(value):
      return (
        tuple(point(v) for v in value)
        if isinstance(value, tuple)
        else I.Expr(I.ScalarType("bool" if value.type.mask else value.type.dtype), "element_at", (value, *coords))
      )

    def store_outputs(view, value):
      if isinstance(view, tuple):
        for v, x in zip(view, value):
          store_outputs(v, x)
      else:
        c = coords if operation == "scan" else [coords[a] for a in outer_axes]
        offset = I.literal(0)
        for index_, stride in zip(c, view.strides):
          offset = I.binary("add", offset, I.binary("mul", index_, stride))
        self.block.statements.append(I.Store(I.View(view.storage, view.dtype, (), (), offset, owned=True), value))

    if operation == "scan" and exclusive:
      store_outputs(outputs, accumulators)
    if isinstance(reducer, str):
      value = I.binary("add" if reducer == "add" else reducer, accumulators, point(values))
    else:
      start = len(self.block.statements)
      value = self.helper(reducer.body, node, arguments=(accumulators, point(values)))

      def pure(block):
        for s in block.statements:
          if not isinstance(s, (I.Assign, I.If, I.Assert)):
            self.error(node, "reducer helpers must be pure scalar combiners")
          for root in (getattr(s, "value", None), getattr(s, "condition", None)):
            if isinstance(root, I.Expr) and any(
              self.operations.get(e).effects not in {"none", "read"} for e in walk(root)
            ):
              self.error(node, "reducer helper has side effects")
          for child in children(s):
            pure(child)

      pure(I.Block(self.block.statements[start:]))

      if operation == "reduce" and reducer.associative:
        native = self.native_combiner(value, accumulators, point(values), values, identity, axis, unroll)
        if native is not None:
          # Drop the ordered fallback's unused output sites. The recognized
          # fold stays logical until common loop fusion and SIMD legalization.
          del parent.statements[allocation_start:]
          self.block = parent
          return native

    def update(target, source):
      if isinstance(target, tuple):
        if not isinstance(source, tuple) or len(target) != len(source):
          self.error(node, "reducer result tuple arity mismatch")

        def freeze(v):
          return tuple(freeze(x) for x in v) if isinstance(v, tuple) else self.freeze_tile(self.value(v, node))

        frozen = tuple(freeze(v) for v in source)
        for a, b in zip(target, frozen):
          update(a, b)
      else:
        source = self.value(source, node)
        if not isinstance(source.type, I.ScalarType):
          self.error(node, "reducer result must be scalar or tuple of scalars")
        self.block.statements.append(I.Assign(target.data, I.Expr(target.type, "cast", (source,)), False))

    update(accumulators, value)
    if operation == "scan" and not exclusive:
      store_outputs(outputs, accumulators)
    body = self.block
    self.block = prefix
    loop_start = I.binary("sub", extent, I.literal(1)) if reverse else I.literal(0)
    loop_stop = I.literal(-1) if reverse else extent
    self.block.statements.append(
      I.For(
        name,
        loop_start,
        loop_stop,
        I.literal(-1 if reverse else 1),
        body,
        unroll,
        operation,
        isinstance(reducer, CapturedReducer) and reducer.associative,
        tuple(
          ReductionSpec(
            reducer if isinstance(reducer, str) else "custom",
            "bool" if leaf.type.mask else leaf.type.dtype,
            ordering="sequential",
            associative=isinstance(reducer, str) or reducer.associative,
          )
          for leaf in leaves
        ),
      )
    )
    if operation == "reduce" and output_shape:
      store_outputs(outputs, accumulators)
    result_block = self.block
    for target, stop in reversed(outer_loops):
      result_block = I.Block([I.For(target, I.literal(0), stop, I.literal(1), result_block, 1)])
    self.block = parent
    parent.statements.extend(result_block.statements)

    def result(v):
      return (
        tuple(result(x) for x in v) if isinstance(v, tuple) else I.Expr(I.TileType(v.dtype, v.shape), "load", data=v)
      )

    return result(outputs) if outputs is not None else accumulators

  def native_combiner(self, result, left, right, inputs, identity, axis, unroll):
    """Recognize pure independent builtin folds without trusting op names alone.

    An associative promise authorizes the established builtin SIMD order.
    Coupled tuple state, non-neutral initial values, assertions and branches
    retain their ordered implementation instead of silently changing meaning.
    """
    statements = self.block.statements
    if any(not isinstance(s, I.Assign) or not s.declare for s in statements):
      return None
    definitions = {s.target: s.value for s in statements}

    def expand(e, seen=frozenset()):
      if e.op in {"local", "frozen"} and e.data in definitions and e.data not in seen:
        return expand(definitions[e.data], seen | {e.data})
      if e.op == "cast" and e.type.dtype == e.args[0].type.dtype:
        return expand(e.args[0], seen)
      from dataclasses import replace

      return replace(e, args=tuple(expand(a, seen) for a in e.args))

    def recognize(value, a, b, source, initial):
      if isinstance(a, tuple):
        if not isinstance(value, tuple) or len(value) != len(a):
          return None
        outputs = tuple(recognize(v, x, y, z, init) for v, x, y, z, init in zip(value, a, b, source, initial))
        return None if any(v is None for v in outputs) else outputs
      value = expand(value)
      native = next(
        (
          name
          for name in self.operations.names(Trait.SCALAR_REDUCTION)
          if self.operations.get(name).reduction == value.op
        ),
        None,
      )
      if source.type.mask or source.type.dtype == "bool":
        return None
      if (
        native is None
        or len(value.args) != 2
        or not (
          I.same_expr(value.args[0], a)
          and I.same_expr(value.args[1], b)
          or I.same_expr(value.args[1], a)
          and I.same_expr(value.args[0], b)
        )
      ):
        return None
      neutral = 0 if value.op == "add" else float("inf") if value.op == "min" else float("-inf")
      if not source.type.dtype.startswith(("float", "bfloat")) and value.op != "add":
        return None
      if I.constant(symbol(initial)) != neutral:
        return None
      shape = source.type.shape
      output = shape[:axis] + shape[axis + 1 :] if axis is not None else ()
      if output:
        return I.Expr(I.TileType(source.type.dtype, output), "axis_reduce", (source,), (axis, value.op, unroll))
      return I.Expr(I.ScalarType(source.type.dtype), native, (source,), unroll)

    return recognize(result, left, right, inputs, identity)
