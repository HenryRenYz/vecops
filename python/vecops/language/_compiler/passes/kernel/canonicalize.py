# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Kernel passes: canonicalize."""

from ... import ir as I
from ...analysis.keys import ExpressionKeys
from ...ir.traversal import nodes, rewrite_expressions
from ...operations import Trait, current
from ..manager import PassResult, register_pass


@register_pass("canonicalize-memory")
def canonicalize_memory(ir, context):
  changed = False
  # A captured read is a value, not permission to re-read its storage after an
  # intervening write. Keep its argument edge so dependency/alias analysis and
  # historical snapshots continue to identify that exact lexical capture.
  # CPU cast(load) emission still fuses it to load_convert after resolving the
  # original read's snapshot; an unrelated later load retains its own identity.
  pending = [statement.value for statement in nodes(ir.body) if isinstance(statement, I.BindTile)]
  captured, seen = set(), set()
  catalog = context.environment.operations if context is not None else current()
  # Many successive bindings can share the same producer DAG. Visit their
  # union once, rather than expanding that DAG once per capture statement.
  while pending:
    value = pending.pop()
    if id(value) in seen:
      continue
    seen.add(id(value))
    if catalog.has(value, Trait.MEMORY_READ):
      captured.add(id(value))
    pending.extend(value.args)

  def canonicalize(expression, original):
    nonlocal changed
    if expression.op == "cast" and expression.args[0].op == "load":
      source = expression.args[0]
      if id(original.args[0]) not in captured and source.type.dtype == source.data.dtype:
        changed = True
        return I.Expr(expression.type, "load", source.args, source.data)
    return expression

  return PassResult(rewrite_expressions(ir, canonicalize, with_original=True), changed=changed)


@register_pass("expression-cse", after=("canonicalize-memory",))
def expression_cse(ir, context):
  table = {}
  keys = ExpressionKeys()
  stateful = {}
  changed = False
  declarations, mutable = {}, set()
  for node in nodes(ir.body):
    if isinstance(node, I.Assign):
      declarations[node.target] = declarations.get(node.target, 0) + 1
      if not node.declare:
        mutable.add(node.target)
    elif isinstance(node, I.For):
      mutable.add(node.target)
  mutable.update(name for name, count in declarations.items() if count > 1)

  def has_state(expression):
    if expression not in stateful:
      stateful[expression] = (
        context.environment.operations.has(expression, Trait.STATE_REFERENCE)
        or expression.op == "local"
        and expression.data in mutable
        or any(has_state(argument) for argument in expression.args)
      )
    return stateful[expression]

  def intern(expression):
    nonlocal changed
    if not context.environment.operations.has(expression, Trait.CSE):
      return expression
    if has_state(expression):
      return expression
    key = keys.expression(expression)
    result = table.setdefault(key, expression)
    changed |= result is not expression
    return result

  return PassResult(rewrite_expressions(ir, intern), changed=changed)
