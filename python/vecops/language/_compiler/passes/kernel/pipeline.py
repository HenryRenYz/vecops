"""Plan guarded lookahead prefetches before vector lowering.

CPU prefetch is a hint, not an asynchronous load. Opt-in automatic buffering
requires a straight-line eligibility proof and a runtime non-alias guard;
otherwise retain ordered prefetch hints. No unguarded data read is speculated.
"""

from dataclasses import fields, replace

from ... import ir as I
from ...ir.traversal import children, values, walk
from ..manager import register_pass


@register_pass("plan-software-pipeline", after=("refresh-computation-reuse",))
def plan_software_pipeline(ir, context):
  def visit(block):
    for index, statement in enumerate(block.statements):
      if not isinstance(statement, I.PipelineFor):
        for child in children(statement):
          visit(child)
        continue
      loop = statement.loop
      if statement.buffering:
        from .buffering import buffer_loop

        buffered = buffer_loop(ir, loop, context, loop.target)
        if buffered is not None:
          block.statements[index] = buffered
          continue
        ir.diagnostics.append("CPU double buffering retained ordered prefetch-only loop: eligibility proof failed")
      definitions = {s.target: s.value for s in loop.body.statements if isinstance(s, I.Assign) and s.declare}
      memo = {}
      future = I.binary(
        "add",
        I.Expr(I.ScalarType("int64"), "local", data=loop.target),
        I.binary("mul", loop.step, I.literal(statement.distance)),
      )

      def clone(value, memo=memo, loop=loop, future=future, definitions=definitions):
        if isinstance(value, I.Expr):
          if id(value) not in memo:
            if value.op == "local" and value.data == loop.target:
              return future
            if value.op in {"local", "frozen"} and value.data in definitions:
              return clone(definitions[value.data])
            memo[id(value)] = replace(value, args=tuple(clone(a) for a in value.args), data=clone(value.data))
          return memo[id(value)]
        if isinstance(value, tuple):
          return tuple(clone(v) for v in value)
        if isinstance(value, I.Node):
          return replace(value, **{f.name: clone(getattr(value, f.name)) for f in fields(value)})
        return value

      hints, seen = [], set()
      available = {s.target for s in loop.body.statements if isinstance(s, I.Assign) and s.declare}
      straight = I.Block(
        [
          s
          for s in loop.body.statements
          if isinstance(s, (I.Assign, I.BindTile, I.Store, I.ElementwiseGroup, I.ReductionGroup))
        ]
      )
      for root in values(straight):
        for load in I.loads(root):
          if load.op != "load" or id(load) in seen or load.data.owned:
            continue
          seen.add(id(load))
          view = clone(load.data)
          address = (view.offset, *view.shape, *view.strides)
          if any(
            context.environment.operations.get(e).effects != "none"
            or e.op in {"local", "frozen"}
            and e.data in available
            for value in address
            for e in walk(value)
          ):
            ir.diagnostics.append("pipeline prefetch omitted: lookahead address is not a pure available expression")
            continue
          hints.append(I.Prefetch(view))
      condition = I.binary("lt" if I.constant(loop.step) > 0 else "gt", future, loop.stop)
      prefix = [I.If(condition, I.Block(hints), I.Block())] if hints else []
      block.statements[index] = replace(loop, body=I.Block(prefix + loop.body.statements), unroll=1)
      ir.diagnostics.append(
        f"CPU two-stage pipeline: {len(hints)} guarded lookahead prefetch sites; explicit buffer banks"
      )

  visit(ir.body)
  return ir
