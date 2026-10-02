"""Kernel passes: fusion."""
from ... import ir as I
from ...ir.traversal import walk, children
from ..manager import register_pass

def matching_shape(a, b):
  return len(a) == len(b) and all(I.same_expr(x, y) for x, y in zip(a, b))


@register_pass("fuse-reductions", after=("plan-computation-reuse",))
def fuse_reductions(ir, context):
  limit = context.options.get("max_fused_reductions", 2)
  if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
    raise ValueError("max_fused_reductions must be a positive integer")
  def fuse(block):
    result, group = [], []
    def flush():
      if group:
        result.append(I.ReductionGroup(tuple(group)) if len(group) > 1 else group[0])
        group.clear()
    for statement in block.statements:
      reduction = statement.value if isinstance(statement, I.Assign) else None
      if reduction is not None and reduction.op.startswith("reduce_"):
        source = reduction.args[0]
        if group:
          previous = group[0].value.args[0]
          compatible = (source.type.dtype == previous.type.dtype and matching_shape(source.type.shape, previous.type.shape)
                        and reduction.data == group[0].value.data)
          dependent = any(e.op in {"local", "frozen"} and e.data in {a.target for a in group} for e in walk(source))
          if not compatible or dependent or len(group) >= limit:
            flush()
        group.append(statement)
      else:
        flush()
        for child in children(statement):
          fuse(child)
        result.append(statement)
    flush()
    block.statements = result
  fuse(ir.body)
  return ir


@register_pass("fuse-reduction-elementwise", after=("fuse-reductions",))
def fuse_reduction_elementwise(ir, context):
  def fuse(block):
    result = []
    for statement in block.statements:
      for child in children(statement):
        fuse(child)
      previous = result[-1] if result else None
      assignments = (previous,) if isinstance(previous, I.Assign) and previous.value.op.startswith("reduce_") else previous.assignments if isinstance(previous, I.ReductionGroup) else ()
      if assignments and isinstance(statement, I.Store):
        targets = {a.target for a in assignments}
        operands = (statement.value,) + ((statement.mask,) if statement.mask is not None else ())
        dependent = any(e.op in {"local", "frozen"} and e.data in targets for v in operands for e in walk(v))
        if not dependent and matching_shape(assignments[0].value.args[0].type.shape, statement.destination.shape):
          stores = previous.stores if isinstance(previous, I.ReductionGroup) else ()
          if not stores:
            result[-1] = I.ReductionGroup(assignments, (statement,))
            continue
      result.append(statement)
    block.statements = result
  fuse(ir.body)
  return ir


@register_pass("fuse-elementwise", after=("fuse-reduction-elementwise",))
def fuse_elementwise(ir, context):
  def fuse(block):
    result = []
    for statement in block.statements:
      for child in children(statement):
        fuse(child)
      previous = result[-1] if result else None
      stores = (previous,) if isinstance(previous, I.Store) else previous.stores if isinstance(previous, I.ElementwiseGroup) else ()
      if isinstance(statement, I.Store) and stores and matching_shape(stores[0].destination.shape, statement.destination.shape):
        result[-1] = I.ElementwiseGroup((*stores, statement))
      else:
        result.append(statement)
    block.statements = result
  fuse(ir.body)
  return ir
