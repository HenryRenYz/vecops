"""Shared expression DAG and structured-control traversal."""
from dataclasses import fields, is_dataclass, replace
from . import kernel as I

def walk(expression):
  yield expression
  for argument in expression.args:
    yield from walk(argument)


def children(statement):
  if isinstance(statement, I.UnrolledFor):
    yield statement.copies
    yield statement.loop.body
  for name in ("body", "otherwise"):
    child = getattr(statement, name, None)
    if isinstance(child, I.Block):
      yield child


def values(block):
  for statement in block.statements:
    if isinstance(statement, I.ReductionGroup):
      yield from values(I.Block([*statement.assignments, *statement.stores]))
    if isinstance(statement, I.ElementwiseGroup):
      yield from values(I.Block(list(statement.stores)))
    for name in ("value", "condition", "mask"):
      value = getattr(statement, name, None)
      if isinstance(value, I.Expr):
        yield value
    for child in children(statement):
      yield from values(child)


def rewrite_expressions(ir, rewrite):
  memo = {}
  def visit(value):
    if isinstance(value, I.Expr):
      if id(value) not in memo:
        args = tuple(visit(a) for a in value.args)
        # CSE substitutes identity even when dataclass values compare equal.
        # Structural equality here would silently discard the shared DAG edge.
        same_arguments = all(a is b for a, b in zip(args, value.args))
        memo[id(value)] = rewrite(value if same_arguments else replace(value, args=args))
      return memo[id(value)]
    if isinstance(value, tuple):
      return tuple(visit(v) for v in value)
    if isinstance(value, I.Block):
      value.statements = [visit(s) for s in value.statements]
      return value
    if is_dataclass(value) and isinstance(value, I.Node):
      return replace(value, **{f.name: visit(getattr(value, f.name)) for f in fields(value)})
    return value
  visit(ir.body)
  return ir
