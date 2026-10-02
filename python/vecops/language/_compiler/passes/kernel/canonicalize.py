"""Kernel passes: canonicalize."""
from ... import ir as I
from ...ir.traversal import rewrite_expressions
from ..manager import register_pass

@register_pass("canonicalize-memory")
def canonicalize_memory(ir, context):
  def canonicalize(expression):
    if expression.op == "cast" and expression.args[0].op == "load":
      source = expression.args[0]
      if source.type.dtype == source.data.dtype:
        return I.Expr(expression.type, "load", source.args, source.data)
    return expression
  return rewrite_expressions(ir, canonicalize)


@register_pass("expression-cse", after=("canonicalize-memory",))
def expression_cse(ir, context):
  table = {}
  def intern(expression):
    if expression.op in {"load", "cast", "local", "frozen", "raw", "symbol", "axis_reduce"} or expression.op.startswith("reduce_"):
      return expression
    key = (repr(expression.type), expression.op, tuple(map(id, expression.args)), repr(expression.data))
    return table.setdefault(key, expression)
  return rewrite_expressions(ir, intern)
