"""Expose nested builtin reductions as logical FoldLoops, without scalarization."""

from dataclasses import replace

from ... import ir as I
from ...ir.iteration import AxisFold, AxisResult, FoldLoop
from ...ir.traversal import children, walk
from ...operations import Trait
from ..manager import register_pass


@register_pass("normalize-iteration-regions", after=("expression-cse",), before=("plan-computation-reuse",))
def normalize_iteration_regions(ir, context):
  serial = 0
  catalog = context.environment.operations

  def visit(block, dominating=None):
    nonlocal serial
    available = dict(dominating or {})
    statements = []
    for statement in block.statements:
      if isinstance(statement, I.ReductionGroup):
        statements.append(FoldLoop(statement.assignments, statement.stores))
        continue
      for child in children(statement):
        visit(child, available)
      prefix = []

      def rewrite(value, prefix=prefix):
        nonlocal serial
        # Keep conditionally evaluated reductions at their original point.
        if catalog.has(value, Trait.CONDITIONAL):
          return value
        args = tuple(rewrite(a) for a in value.args)
        changed = replace(value, args=args) if any(a is not b for a, b in zip(args, value.args)) else value
        if catalog.has(changed, Trait.SCALAR_REDUCTION):
          # Mutable references must not escape their original evaluation site.
          key = id(value)
          reusable = not any(
            catalog.has(e, Trait.STATE_REFERENCE | Trait.REFERENCE | Trait.MEMORY_READ) for e in walk(changed)
          )
          if reusable and key in available:
            return available[key]
          serial += 1
          name = f"dsl_o_fold_{serial}"
          prefix.append(FoldLoop((I.Assign(name, changed),)))
          result = I.Expr(changed.type, "local", data=name)
          if reusable:
            available[key] = result
          return result
        return changed

      if isinstance(statement, I.Assign) and catalog.has(statement.value, Trait.SCALAR_REDUCTION):
        statements.append(FoldLoop((statement,)))
      else:
        changes = {}
        for name in ("value", "mask"):
          root = getattr(statement, name, None)
          if isinstance(root, I.Expr):
            changes[name] = rewrite(root)
        statements.extend(prefix)
        statements.append(replace(statement, **changes) if changes else statement)
    block.statements = statements

  visit(ir.body)
  return ir


@register_pass("fuse-axis-folds", after=("fuse-elementwise",), before=("fuse-loop-regions",))
def fuse_axis_folds(ir, context):
  from ...ir.traversal import rewrite_expressions

  replacements = {}
  limit = context.options.get("max_fused_reductions", 2)

  def visit(block):
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      roots = [getattr(statement, "value", None)]
      if isinstance(statement, I.ElementwiseGroup):
        roots = [s.value for s in statement.stores]
      groups = []
      seen = set()
      for root in roots:
        if not isinstance(root, I.Expr):
          continue
        for e in walk(root):
          if e.op != "axis_reduce" or id(e) in seen or isinstance(e.data, AxisResult):
            continue
          seen.add(id(e))
          # Nested dependent reductions remain separate phases.
          if any(v.op == "axis_reduce" for v in walk(e.args[0])):
            continue
          axis, _, unroll = e.data
          found = next(
            (
              g
              for g in groups
              if g[0].data[0] == axis
              and len(g) < limit
              and g[0].data[2] == unroll
              and e.type.dtype == g[0].type.dtype
              and len(e.args[0].type.shape) == len(g[0].args[0].type.shape)
              and all(I.same_expr(a, b) for a, b in zip(e.args[0].type.shape, g[0].args[0].type.shape))
            ),
            None,
          )
          if found is None:
            groups.append([e])
          else:
            found.append(e)
      for group in groups:
        if len(group) < 2:
          continue
        fold = AxisFold(
          tuple(e.args[0] for e in group), group[0].data[0], tuple(e.data[1] for e in group), group[0].data[2]
        )
        for index, e in enumerate(group):
          replacements[id(e)] = replace(e, data=AxisResult(fold, index))
        ir.diagnostics.append(f"axis fold fusion: {len(group)} independent reductions in one iteration domain")

  visit(ir.body)
  return rewrite_expressions(ir, lambda e: replacements.get(id(e), e))
