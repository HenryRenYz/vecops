"""Local iteration is distinct from task scheduling and worker residency."""

import ast

from ...dsl import ExecutionLayout, Vector
from .. import ir as I
from ..dtypes import dtype_name


class LoopsFrontend:
  def local_loop(self, node, descriptor):
    if any(isinstance(n, (ast.Break, ast.Continue)) for statement in node.body for n in ast.walk(statement)):
      self.error(node, "local parallel/SIMD loops do not support break/continue; use T.serial")
    kind, values, options = descriptor
    unroll = self.unroll_count(options.get("unroll", I.literal(2)), node)
    if kind == "simd":
      if not isinstance(node.target, ast.Name) or not 1 <= len(values) <= 2:
        self.error(node, "simd expects one vector target and stop or start, stop")
      start, stop = (I.literal(0), values[0]) if len(values) == 1 else values
      values = (I.binary("sub", stop, start),)
      layout = ExecutionLayout(vector_axis=0, vector=options.get("vector", Vector()))
      targets = (node.target,)
    else:
      start = I.literal(0)
      targets = tuple(node.target.elts) if isinstance(node.target, ast.Tuple) else (node.target,)
      layout = options.get("layout", ExecutionLayout())
      if not isinstance(layout, ExecutionLayout) or len(targets) != len(values) or not values:
        self.error(node, "parallel needs one target per extent and an ExecutionLayout")
    if any(not isinstance(t, ast.Name) for t in targets):
      self.error(node, "local iteration targets must be names")
    values = tuple(self.value(v, node) for v in values)
    if any(not isinstance(v.type, I.ScalarType) or "int" not in v.type.dtype for v in values):
      self.error(node, "local iteration bounds must be scalar integers")
    order = layout.order or tuple(range(len(values)))
    if sorted(order) != list(range(len(values))):
      self.error(node, "execution order must be an axis permutation")
    va = layout.vector_axis
    if va is not None:
      if not -len(values) <= va < len(values):
        self.error(node, "vector axis exceeds the iteration rank")
      va %= len(values)
      if order[-1] != va:
        self.error(node, "vector axis must be innermost in the execution order")
    self.promote_control_state(node.body, node, ignore=tuple(t.id for t in targets))
    before = dict(self.env)
    loops = []
    for axis in order:
      name = self.unique(targets[axis].id)
      index = I.Expr(I.ScalarType("int64"), "local", data=name)
      step = I.literal(1)
      if axis == va:
        vector = layout.vector
        dtype = dtype_name(vector.dtype)
        step = (
          I.literal(vector.N)
          if vector.N is not None
          else I.Expr(I.ScalarType("int64"), "lanes", data=(dtype, vector.pow))
        )
        width = I.binary("min", step, I.binary("sub", values[axis], index))
        offset = I.binary("add", start, index) if kind == "simd" else index
        typ = I.VectorType("int64", (width,), tag=I.VectorTag(vector.N, vector.pow, basis_dtype=dtype))
        self.env[targets[axis].id] = I.Expr(typ, "arange", (offset,))
      else:
        self.env[targets[axis].id] = index
      loops.append((name, values[axis], step))
    self.loop_depth += 1
    body = self.nested(node.body)
    self.loop_depth -= 1
    self.verify_loop_environment(before, node, tuple(t.id for t in targets))
    self.env = before
    # These are ordered local loops, not independent resident task callbacks.
    # Explicit SIMD is represented by the tagged induction value in Vector IR.
    for name, stop, step in reversed(loops):
      body = I.Block(
        [
          I.For(
            name,
            I.literal(0),
            stop,
            step,
            body,
            unroll if I.constant(step) is not None else 1,
            "simd" if I.constant(step) is None else kind,
          )
        ]
      )
    self.block.statements.extend(body.statements)
