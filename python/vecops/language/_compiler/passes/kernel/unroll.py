"""Kernel passes: unroll."""

from dataclasses import fields, is_dataclass, replace

from ... import ir as I
from ...analysis.loops import trivial_true
from ...ir.traversal import children
from ...operations import Trait
from ..manager import register_pass
from .reuse import plan_computation_reuse


@register_pass("unroll-serial", after=("fuse-elementwise",))
def unroll_serial(ir, context):
  def visit(block):
    for index, statement in enumerate(block.statements):
      for child in children(statement):
        visit(child)
      if not isinstance(statement, I.For) or statement.unroll == 1:
        continue
      if I.constant(statement.step) is None and statement.step.op != "lanes":
        continue
      operations = statement.body.statements
      safe = all(
        isinstance(s, (I.Assign, I.BindTile, I.Store, I.IndexedStore, I.Assert, I.ReductionGroup, I.ElementwiseGroup))
        and (not isinstance(s, I.Assign) or s.declare)
        and (not isinstance(s, I.ReductionGroup) or all(a.declare for a in s.assignments))
        and (not isinstance(s, I.IndexedStore) or s.unique)
        and (not isinstance(s, I.Assert) or trivial_true(s.condition))
        for s in operations
      )
      if not safe:
        ir.diagnostics.append("serial unroll retained ordered loop: control or loop-carried/effect dependency")
        continue
      targets = {s.target for s in operations if isinstance(s, I.Assign)}
      targets.update(a.target for s in operations if isinstance(s, I.ReductionGroup) for a in s.assignments)
      copies, memories = [], [{} for _ in range(statement.unroll)]
      full_vectors = statement.kind == "simd" and I.constant(statement.start) == 0

      def clone(value, slot, memories=memories, full_vectors=full_vectors, statement=statement, targets=targets):
        if isinstance(value, I.Expr):
          memo = memories[slot]
          if id(value) not in memo:
            if (
              full_vectors
              and value.op == "min"
              and I.same_expr(value.args[0], statement.step)
              and value.args[1].op == "sub"
              and I.same_expr(value.args[1].args[0], statement.stop)
              and value.args[1].args[1].op == "local"
              and value.args[1].args[1].data == statement.target
            ):
              memo[id(value)] = statement.step
            elif value.op == "local" and value.data == statement.target:
              memo[id(value)] = I.binary("add", value, I.binary("mul", I.literal(slot), statement.step))
            else:
              data = clone(value.data, slot)
              if context.environment.operations.has(value, Trait.REFERENCE) and value.data in targets:
                data = f"{value.data}_u{slot}"
              memo[id(value)] = replace(
                value, args=tuple(clone(a, slot) for a in value.args), data=data, type=clone(value.type, slot)
              )
          return memo[id(value)]
        if isinstance(value, tuple):
          return tuple(clone(v, slot) for v in value)
        if is_dataclass(value) and isinstance(value, I.Node):
          changes = {f.name: clone(getattr(value, f.name), slot) for f in fields(value)}
          if isinstance(value, I.Assign):
            changes["target"] = f"{value.target}_u{slot}"
          return replace(value, **changes)
        return value

      for operation in operations:
        copies.extend(clone(operation, slot) for slot in range(statement.unroll))
      block.statements[index] = I.UnrolledFor(statement, I.Block(copies), full_vectors)

  visit(ir.body)
  return ir


@register_pass("refresh-computation-reuse", after=("unroll-serial",))
def refresh_computation_reuse(ir, context):
  # Operation copies have independent expression identities and lifetimes.
  # Never share a row's cached exponential with another unrolled row.
  ir.materializations.clear()
  return plan_computation_reuse(ir, context)
