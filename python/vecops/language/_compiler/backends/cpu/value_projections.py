# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Plan logical reshape projections without replacing historical producers.

Only immutable entry-derived dimensions participate in suffix equivalence.
Captured spelling is normalized for comparison, never substituted into the IR
or evaluated earlier. A certificate permits vector traversal inside that
suffix; crossing a changed prefix remains ordinary scalar legalization.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

from ... import ir as I
from ...ir.traversal import nodes
from ...operations import Trait
from ...passes.manager import register_pass
from .invariants import EntryGeometry


class EntryExtents:
  """Structural equality through single-assignment, entry-derived aliases."""

  def __init__(self, kernel, catalog):
    self.entry = EntryGeometry(kernel, catalog)
    definitions, excluded = {}, set()
    for node in nodes(kernel.body):
      if isinstance(node, (I.Assign, I.For)):
        definitions.setdefault(node.target, []).append(node)
      if isinstance(node, (I.For, I.TeamReduce)):
        excluded.add(node.target)
      elif isinstance(node, I.Grid):
        excluded.update(node.targets)
    self.definitions = {
      name: sites[0].value
      for name, sites in definitions.items()
      if name not in excluded and len(sites) == 1 and type(sites[0]) is I.Assign and sites[0].declare
    }
    self.memo = {}

  def key(self, value, active=None):
    if type(value) is not I.Expr or type(value.type) is not I.ScalarType or value.type.dtype not in {"int64", "bool"}:
      return None
    if value in self.memo:
      return self.memo[value]
    active = set() if active is None else active
    if value in active:
      return None
    active.add(value)
    result = None
    if value.op in {"symbol", "local", "frozen"} and not value.args and type(value.data) is str:
      definitions = self.entry.bindings if value.op == "symbol" else self.definitions
      definition = definitions.get(value.data)
      if type(definition) is I.Expr and definition.type == value.type:
        result = self.key(definition, active)
    elif value.op == "literal":
      if not value.args and (
        (value.type.dtype == "int64" and type(value.data) is int and -(1 << 63) <= value.data < (1 << 63))
        or (value.type.dtype == "bool" and type(value.data) is bool)
      ):
        result = (value.op, value.type.dtype, value.data)
    elif value.op == "raw":
      if not value.args and value.type.dtype == "int64" and type(value.data) is str and value.data in self.entry.raw:
        result = (value.op, value.type.dtype, value.data)
    elif (
      value.args
      and value.data is None
      and self.entry.catalog.get(value).effects == "none"
      and self.entry.catalog.has(value, Trait.ELEMENTWISE)
      and not self.entry.catalog.has(
        value, Trait.REFERENCE | Trait.MEMORY_READ | Trait.STATE_REFERENCE | Trait.CAPTURE_SCALAR
      )
    ):
      arguments = tuple(self.key(argument, active) for argument in value.args)
      if all(argument is not None for argument in arguments):
        result = (value.op, value.type.dtype, arguments)
    active.remove(value)
    self.memo[value] = result
    return result


@dataclass(frozen=True)
class ReshapeSuffix:
  subject: object
  dimensions: int


def collect_reshape_suffixes(kernel, catalog):
  """Certify exact TileType producers; retain their load/capture identities."""
  extents, plans = EntryExtents(kernel, catalog), {}
  for value in nodes(kernel.body):
    if (
      type(value) is not I.Expr
      or value.op != "value_transform"
      or value.data != ("reshape", ())
      or len(value.args) != 1
      or type(value.type) is not I.TileType
      or type(value.args[0].type) is not I.TileType
    ):
      continue
    source = value.args[0]
    before = tuple(extents.key(n) for n in source.type.shape)
    after = tuple(extents.key(n) for n in value.type.shape)
    # Prefix coordinates must also be statement-free entry geometry. This
    # avoids moving a scalar reduction, mutable reference or opaque callback
    # from the old per-lane guard into vector coordinate preparation.
    if any(key is None for key in (*before, *after)):
      continue
    common = 0
    for first, second in zip(reversed(before), reversed(after)):
      if first != second:
        break
      common += 1
    if common:
      plans[id(value)] = ReshapeSuffix(value, common)
  return MappingProxyType(plans)


@register_pass("cpu-plan-value-projections", stage="cpu-memory", after="cpu-plan-entry-geometry")
def plan_value_projections(prepared, context):
  return replace(
    prepared, reshape_suffixes=collect_reshape_suffixes(prepared.vectors.kernel, context.environment.operations)
  )
