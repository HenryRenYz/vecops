# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Kernel passes: canonicalize."""

from ... import ir as I
from ...ir.traversal import nodes, rewrite_expressions, walk_unique
from ...operations import Trait, current
from ..manager import register_pass


@register_pass("canonicalize-memory")
def canonicalize_memory(ir, context):
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
    if expression.op == "cast" and expression.args[0].op == "load":
      source = expression.args[0]
      if id(original.args[0]) not in captured and source.type.dtype == source.data.dtype:
        return I.Expr(expression.type, "load", source.args, source.data)
    return expression

  return rewrite_expressions(ir, canonicalize, with_original=True)


@register_pass("expression-cse", after=("canonicalize-memory",))
def expression_cse(ir, context):
  table = {}

  def intern(expression):
    if not context.environment.operations.has(expression, Trait.CSE):
      return expression
    if any(context.environment.operations.has(e, Trait.STATE_REFERENCE) for e in walk_unique(expression)):
      return expression
    key = (repr(expression.type), expression.op, tuple(map(id, expression.args)), repr(expression.data))
    return table.setdefault(key, expression)

  return rewrite_expressions(ir, intern)
