# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared conservative value effects and references for kernel transforms.

Pure does not imply nontrapping or speculatable. In particular, historical
loads, shape validation, allocations and hardware state are never moved on the
strength of an effects='none' label alone. Unknown operations are barriers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

from .. import ir as I
from ..ir.iteration import AxisFold, AxisResult, IterationDomain, LoopSchedule, ReductionSpec
from ..ir.traversal import nodes
from ..operations import Trait
from .facts import ValueFacts, type_range

# Value descriptors own expression dependencies without evaluating statements.
# Use exact classes: an extension subclass can add executable ownership. Other
# Nodes (including Block, RowMap, state updates and hardware owners) remain opaque even
# when every contained expression is individually pure.
_VALUE_METADATA = frozenset((
  I.ScalarType, I.TileType, I.VectorType, I.VectorTag, I.View, I.ViewRegion,
  I.PackedView, I.DotOptions, I.TileState, AxisFold, AxisResult, IterationDomain, LoopSchedule, ReductionSpec,
))


class ValueEffects:
  def __init__(self, catalog, facts=None):
    self.catalog = catalog
    # Rewrites can release a previously analyzed node before this analysis
    # ends. Expr keys use object identity and retain their subject, so a new
    # effectful expression cannot inherit a recycled object's proof.
    self._discard = {}
    self._speculate = {}
    self._owned = {}
    self._metadata = {}
    self.facts = facts or ValueFacts()

  def _metadata_edges(self, value):
    """Immediate owned Expr dependencies, stopping at each expression.

    Cache non-expression ownership too: shared View/type/container diamonds
    must not be re-walked for every consumer. Strong subjects prevent id reuse;
    a provisional unknown result makes ownership cycles conservatively unsafe.
    None denotes opaque ownership, not an empty dependency list: flattening a
    statement to its values would erase its write/control effects.
    """
    if isinstance(value, I.Expr):
      return (value,)
    if not isinstance(value, (I.Node, tuple, list, dict)):
      return ()
    identity = id(value)
    if identity in self._metadata:
      return self._metadata[identity][1]
    self._metadata[identity] = value, None
    if isinstance(value, I.Node):
      if type(value) not in _VALUE_METADATA:
        return None
      items = (getattr(value, field.name) for field in fields(value))
    elif isinstance(value, dict):
      items = (item for pair in value.items() for item in pair)
    else:
      items = iter(value)
    owned = {}
    for item in items:
      edges = self._metadata_edges(item)
      if edges is None:
        return None
      owned.update((edge, None) for edge in edges)
    result = tuple(owned)
    self._metadata[identity] = value, result
    return result

  def owned_expressions(self, expression):
    """Direct value dependencies, or None for opaque/cyclic metadata ownership."""
    if expression not in self._owned:
      kind = self._metadata_edges(expression.type)
      data = self._metadata_edges(expression.data)
      self._owned[expression] = None if kind is None or data is None else tuple(dict.fromkeys((*kind, *data)))
    return self._owned[expression]

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
    if expression.op not in {"floordiv", "ceildiv", "mod"} or len(expression.args) != 2:
      return False
    divisor = I.constant(expression.args[1])
    limits = type_range(expression.type.dtype) if isinstance(expression.type, I.ScalarType) else None
    return (
      isinstance(expression.type, I.ScalarType)
      and "int" in expression.type.dtype
      and type(divisor) is int
      and limits is not None
      and 0 < divisor <= min(limits.upper, (1 << 63) - 1)
    )

  def discardable(self, expression):
    """Whether dropping evaluation cannot remove an observable effect/trap."""
    if expression not in self._discard:
      self._discard[expression] = False
      definition = self.catalog.get(expression)
      owned = self.owned_expressions(expression)
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
        and owned is not None
        and all(self.discardable(argument) for argument in expression.args)
        and all(self.discardable(value) for value in owned)
      )
    return self._discard[expression]

  def speculatable(self, expression):
    """Whether a scalar integer/boolean expression can run on an empty path."""
    if expression not in self._speculate:
      self._speculate[expression] = False
      owned = self.owned_expressions(expression)
      self._speculate[expression] = (
        isinstance(expression.type, I.ScalarType)
        and (expression.type.dtype == "bool" or "int" in expression.type.dtype)
        and (self.catalog.get(expression).speculatable or self.total_division(expression))
        and self.discardable(expression)
        and self.overflow_safe(expression)
        and owned is not None
        and all(self.speculatable(argument) for argument in expression.args)
        and all(self.speculatable(value) for value in owned)
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


@dataclass(frozen=True)
class ReadAccess:
  """A real reader identity and its accessed view; no synthetic load identity."""

  expression: I.Expr
  view: I.View
  indices: I.Expr | None = None


def read_accesses(expressions, catalog):
  """Unconditional read views, or None when alias versioning is unsafe.

  Historical ``loads_unique`` remains an argument-edge identity query. This
  query also follows value-owned metadata and conditional cache storage. An
  unknown cache choice is a barrier: even evaluating its inactive view's
  address for an alias predicate could be invalid. Exact boolean choices visit
  only the executable arm. Shared DAGs/metadata are visited once per query.
  """
  effects = ValueEffects(catalog)
  pending = list(expressions) if type(expressions) in (tuple, list) else [expressions]
  seen, result = {}, []
  while pending:
    expression = pending.pop()
    if type(expression) is not I.Expr:
      return None
    if id(expression) in seen:
      continue
    seen[id(expression)] = expression
    if expression.op == "cached_value":
      data = expression.data
      if type(data) is not tuple or len(data) != 2 or type(data[0]) is not I.View or len(expression.args) != 1:
        return None
      view, condition = data
      choice = I.boolean_literal(condition)
      if choice is None:
        return None
      if choice:
        result.append(ReadAccess(expression, view))
      else:
        pending.extend(expression.args)
      owned = effects._metadata_edges((expression.type, condition, view) if choice else (expression.type, condition))
    else:
      pending.extend(expression.args)
      definition = catalog.get(expression)
      if definition.effects not in {"none", "read"}:
        return None
      if (catalog.has(expression, Trait.MEMORY_READ) or definition.effects == "read") and not catalog.has(
        expression, Trait.STATE_REFERENCE
      ):
        data = expression.data
        if expression.op == "gather":
          view = data[0] if type(data) is tuple else data
          indices = expression.args[0] if expression.args else None
        elif expression.op == "packed_load" and type(data) is I.PackedView:
          view, indices = data.view, None
        else:
          view, indices = data, None
        if type(view) is not I.View:
          return None
        result.append(ReadAccess(expression, view, indices))
      owned = effects.owned_expressions(expression)
    if owned is None:
      return None
    pending.extend(owned)
  return tuple(result)
