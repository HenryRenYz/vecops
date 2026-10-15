# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared conservative value effects and references for kernel transforms.

Pure does not imply nontrapping or speculatable. In particular, historical
loads, shape validation, allocations and hardware state are never moved on the
strength of an effects='none' label alone. Unknown operations are barriers.
"""

import re

from .. import ir as I
from ..ir.traversal import nodes
from ..operations import Trait
from .facts import ValueFacts, type_range


class ValueEffects:
  def __init__(self, catalog, facts=None):
    self.catalog = catalog
    # Rewrites can release a previously analyzed node before this analysis
    # ends. Expr keys use object identity and retain their subject, so a new
    # effectful expression cannot inherit a recycled object's proof.
    self._discard = {}
    self._speculate = {}
    self.facts = facts or ValueFacts()

  def overflow_safe(self, expression):
    if expression.op not in {"add", "sub", "mul", "neg"} or not expression.type.dtype.startswith("int"):
      return True
    if expression.op == "neg":
      value = self.facts.infer(expression.args[0])
      limits = type_range(expression.type.dtype)
      return value is not None and limits.lower <= -value.upper <= -value.lower <= limits.upper
    a, b = (self.facts.infer(value) for value in expression.args)
    if a is None or b is None:
      return False
    if expression.op == "add":
      lower, upper = a.lower + b.lower, a.upper + b.upper
    elif expression.op == "sub":
      lower, upper = a.lower - b.upper, a.upper - b.lower
    else:
      products = (a.lower * b.lower, a.lower * b.upper, a.upper * b.lower, a.upper * b.upper)
      lower, upper = min(products), max(products)
    limits = type_range(expression.type.dtype)
    return limits.lower <= lower <= upper <= limits.upper

  @staticmethod
  def total_division(expression):
    if expression.op not in {"floordiv", "mod"} or len(expression.args) != 2:
      return False
    divisor = I.constant(expression.args[1])
    return (
      isinstance(expression.type, I.ScalarType)
      and "int" in expression.type.dtype
      and type(divisor) is int
      and 0 < divisor <= (1 << 63) - 1
    )

  def discardable(self, expression):
    """Whether dropping evaluation cannot remove an observable effect/trap."""
    if expression not in self._discard:
      definition = self.catalog.get(expression)
      raw = (
        expression.op != "raw"
        or isinstance(expression.data, str)
        and bool(re.fullmatch(r"[A-Za-z_]\w*(?:\.(?:size|stride)\(\d+\))?", expression.data))
      )
      self._discard[expression] = (
        definition.effects == "none"
        and (not definition.may_trap or self.total_division(expression))
        and raw
        and not self.catalog.has(expression, Trait.STATE_REFERENCE)
        and all(self.discardable(argument) for argument in expression.args)
      )
    return self._discard[expression]

  def speculatable(self, expression):
    """Whether a scalar integer/boolean expression can run on an empty path."""
    if expression not in self._speculate:
      self._speculate[expression] = (
        isinstance(expression.type, I.ScalarType)
        and (expression.type.dtype == "bool" or "int" in expression.type.dtype)
        and (self.catalog.get(expression).speculatable or self.total_division(expression))
        and self.discardable(expression)
        and self.overflow_safe(expression)
        and all(self.speculatable(argument) for argument in expression.args)
      )
    return self._speculate[expression]


def scalar_references(value):
  return {
    node.data
    for node in nodes(value)
    if isinstance(node, I.Expr) and node.op in {"local", "frozen", "symbol"} and isinstance(node.data, str)
  }


def assigned_scalars(value):
  return {node.target for node in nodes(value) if isinstance(node, (I.Assign, I.For))}
