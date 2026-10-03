"""Normalize costly reduction input permutations through the memory planner.

Do not copy arbitrary noncontiguous views: a rank-one slice, broadcast or
masked read stays a gather. A rank-two column-contiguous reduction source
can use the registered transpose plan once, rather than gather every row.
"""

from dataclasses import fields, replace
from ... import ir as I
from ...ir.traversal import walk, children
from ..manager import register_pass
from ...operations import Trait


@register_pass("normalize-reduction-layout", before=("decompose-row-normalization", "expression-cse"))
def normalize_reduction_layout(ir, context):
  serial = 0

  def visit(block):
    nonlocal serial
    result = []
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      replacements, preparation = {}, []
      expressions = [getattr(statement, name, None) for name in ("value", "condition", "mask")]
      for expression in expressions:
        if not isinstance(expression, I.Expr):
          continue
        for reduction in walk(expression):
          if not context.environment.operations.has(reduction, Trait.REDUCTION):
            continue
          for load in I.loads(reduction.args[0]):
            view = load.data
            if (
              id(load) in replacements
              or load.args
              or len(view.shape) != 2
              or I.constant(view.strides[0]) != 1
              or I.constant(view.strides[1]) == 1
            ):
              continue
            serial += 1
            dense = I.View(
              f"dsl_o_layout_{serial}",
              load.type.dtype,
              view.shape,
              (view.shape[1], I.literal(1)),
              I.literal(0),
              owned=True,
            )
            preparation.extend((I.Allocate(dense), I.Store(dense, load)))
            replacements[id(load)] = I.Expr(load.type, "load", data=dense)
      memo = {}

      def rewrite(value):
        if isinstance(value, I.Expr):
          if id(value) in replacements:
            return replacements[id(value)]
          if id(value) not in memo:
            memo[id(value)] = replace(value, args=tuple(rewrite(a) for a in value.args))
          return memo[id(value)]
        if isinstance(value, tuple):
          return tuple(rewrite(v) for v in value)
        if isinstance(value, I.Node) and not isinstance(value, I.Block):
          return replace(value, **{f.name: rewrite(getattr(value, f.name)) for f in fields(value)})
        return value

      result.extend(preparation)
      result.append(rewrite(statement) if replacements else statement)
    block.statements = result

  visit(ir.body)
  return ir
