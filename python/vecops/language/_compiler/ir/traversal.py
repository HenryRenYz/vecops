"""Shared expression DAG and structured-control traversal."""

from dataclasses import fields, is_dataclass, replace

from . import kernel as I


def walk(expression):
  yield expression
  for argument in expression.args:
    yield from walk(argument)


def walk_unique(expression):
  """Preorder traversal of argument edges, visiting each expression once.

  Use for membership/trait checks, not use counts: ``walk`` deliberately
  preserves repeated incoming edges. Shape/data nodes are not argument edges;
  ``nodes`` includes those when an ownership traversal is required.
  """
  pending, seen = [expression], set()
  while pending:
    value = pending.pop()
    if id(value) in seen:
      continue
    seen.add(id(value))
    yield value
    pending.extend(reversed(value.args))


def nodes(value, seen=None):
  """Walk each owned IR node once, including data and shape metadata.

  Expression/shape graphs share nodes heavily; recursive tree expansion would
  make capability detection exponential for nested helper/reduction DAGs.
  """
  if seen is None:
    seen = set()
  if isinstance(value, (tuple, list)):
    for item in value:
      yield from nodes(item, seen)
  elif isinstance(value, I.Node) and id(value) not in seen:
    seen.add(id(value))
    yield value
    for field in fields(value):
      yield from nodes(getattr(value, field.name), seen)


def children(statement):
  from .pipeline import PipelineFor

  if isinstance(statement, PipelineFor):
    yield statement.loop.body
  if isinstance(statement, I.UnrolledFor):
    yield statement.copies
    yield statement.loop.body
  for name in ("body", "otherwise"):
    child = getattr(statement, name, None)
    if isinstance(child, I.Block):
      yield child


def values(block):
  for statement in block.statements:
    from .printing import FormattedText, Print

    if isinstance(statement, Print):

      def arguments(items):
        for item in items:
          if isinstance(item, I.Expr):
            yield item
          elif isinstance(item, FormattedText):
            yield from arguments(item.arguments)
          elif isinstance(item, tuple):
            yield from arguments(item)

      yield from arguments(statement.arguments)
      yield statement.flush
    if isinstance(statement, I.ReductionGroup):
      yield from values(I.Block([*statement.assignments, *statement.stores]))
    if isinstance(statement, I.ElementwiseGroup):
      yield from values(I.Block(list(statement.stores)))
    for name in ("value", "condition", "mask", "indices"):
      value = getattr(statement, name, None)
      if isinstance(value, I.Expr):
        yield value
    for child in children(statement):
      yield from values(child)


def rewrite_expressions(ir, rewrite, *, with_original=False):
  """Rewrite the shared DAG, optionally exposing its pre-rewrite identity.

  ``with_original`` calls ``rewrite(changed, original)``. Identity-sensitive
  passes can then distinguish lexical captures even when rewriting an argument
  has already cloned the current expression. The default callback contract is
  unchanged; all references to one original still share the memoized result.
  """
  memo = {}

  def visit(value):
    if isinstance(value, I.Expr):
      if id(value) not in memo:
        args = tuple(visit(a) for a in value.args)
        # CSE substitutes identity even when dataclass values compare equal.
        # Structural equality here would silently discard the shared DAG edge.
        same_arguments = all(a is b for a, b in zip(args, value.args))
        changed = value if same_arguments else replace(value, args=args)
        memo[id(value)] = rewrite(changed, value) if with_original else rewrite(changed)
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
