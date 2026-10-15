# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fold proven total scalar predicates/index arithmetic using shared ABI facts."""

from ... import ir as I
from ...analysis.effects import ValueEffects
from ...analysis.facts import ValueFacts
from ...ir.traversal import rewrite_expressions
from ..manager import PassResult, register_pass


@register_pass("propagate-value-facts", stage="kernel-cleanup", before="simplify-ir")
def propagate_value_facts(ir, context):
  facts = ValueFacts.from_kernel(ir)
  effects = ValueEffects(context.environment.operations)
  changed = False

  def rewrite(value):
    nonlocal changed
    if value.op == "literal" or not effects.discardable(value):
      return value
    fact = facts.infer(value)
    if fact is None or fact.constant is None:
      return value
    changed = True
    data = bool(fact.constant) if value.type.dtype == "bool" else fact.constant
    return I.Expr(value.type, "literal", data=data)

  return PassResult(rewrite_expressions(ir, rewrite), changed=changed)
