# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Late lexical storage placement and nonoverlapping scratch-slot assignment.

Intervals cover complete top-level regions, including read provenance and
metadata uses. This deliberately overestimates nested-loop/branch lifetimes.
Worker allocations, implicit snapshots, workspace authority and state storage
are separate owners. Reuse never changes logical View/storage identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType

from ... import ir as I
from ...analysis.effects import ValueEffects, assigned_scalars, scalar_references
from ...ir.traversal import children, nodes
from ...operations import Trait
from ...passes.manager import register_pass


def branch_buffer_scopes(block):
  """Find common branch scopes without shortening loop capacity reuse."""
  references, allocations, seen = {}, set(), set()

  def visit(value, path=(), looping=False):
    if isinstance(value, dict):
      for key, item in value.items():
        visit(key, path, looping)
        visit(item, path, looping)
      return
    if isinstance(value, (tuple, list, set, frozenset)):
      for item in value:
        visit(item, path, looping)
      return
    if not is_dataclass(value) or isinstance(value, type):
      return
    key = id(value), path, looping
    if key in seen:
      return
    seen.add(key)
    if isinstance(value, I.View):
      references.setdefault(value.storage, set()).add(path)
    if isinstance(value, I.Allocate):
      allocations.add(value.view.storage)
    if isinstance(value, I.If):
      visit(value.condition, path, looping)
      for child in (value.body, value.otherwise):
        visit(child, path if looping else (*path, id(child)), looping)
      return
    if isinstance(value, (I.Grid, I.Team)):
      for field in fields(value):
        item = getattr(value, field.name)
        visit(item, () if field.name == "body" else path, True if field.name == "body" else looping)
      return
    looping = looping or isinstance(value, (I.For, I.While, I.UnrolledFor))
    for field in fields(value):
      visit(getattr(value, field.name), path, looping)

  visit(block)
  result = {}
  for name in allocations:
    paths = iter(references.get(name, ()))
    common = next(paths, ())
    for path in paths:
      common = common[: next((i for i, (a, b) in enumerate(zip(common, path)) if a != b), min(len(common), len(path)))]
    if common:
      result[name] = common[-1]
  return result


@dataclass(frozen=True)
class ScratchInterval:
  storage: str
  dtype: str
  first: int
  last: int
  scope: int | None
  capacity: int | None
  shape: tuple


@dataclass(frozen=True)
class SnapshotScope:
  """One consumer owns every use of these exact historical read objects."""

  subject: object
  reads: tuple[I.Expr, ...]
  occurrences: int


# Exact built-in classes only: an extension subclass can own hidden executable
# metadata or override attribute access. This is an ownership query, not an
# invitation to inspect arbitrary Python containers or extension properties.
_IR_NODES = frozenset(
  value for value in vars(I).values() if isinstance(value, type) and issubclass(value, I.Node) and value is not I.Node
)


class _SnapshotOwners:
  """Direct ownership DAG with bounded first-owner/escaped propagation.

  Each of the exact-consumer and enclosing-group owners changes at most twice:
  unseen to its first consumer, then to escaped. No node stores a transitive
  read set, and the number of independent reads does not limit the proof.
  """

  escaped = object()

  def __init__(self, catalog):
    self.catalog = catalog
    self.effects = ValueEffects(catalog)
    self.graph, self.postorder, self.memory_reads = {}, [], {}
    self.owners, self.argument_owners, self.pointwise = {}, {}, {}

  def inspect(self, value):
    if value is None or type(value) in (bool, int, float, str, bytes):
      return True
    kind = type(value)
    if kind not in _IR_NODES and kind not in (tuple, list, dict):
      return False
    key = id(value)
    if key in self.graph:
      return self.graph[key][1] is not None
    # Strong subjects and provisional opaque edges reject ownership cycles.
    self.graph[key] = value, None
    if kind is I.Expr and self.catalog.get(value).effects not in {"none", "read"}:
      return False
    if kind in _IR_NODES:
      items = tuple(getattr(value, f.name) for f in fields(value))
    elif kind is dict:
      items = tuple(item for pair in value.items() for item in pair)
    else:
      items = tuple(value)
    if not all(self.inspect(item) for item in items):
      return False
    # Screen exact ownership before the shared query: even isinstance() can
    # consult __class__ on an opaque extension object.
    if kind is I.Expr:
      if self.effects.owned_expressions(value) is None:
        return False
      if self.catalog.has(value, Trait.MEMORY_READ):
        self.memory_reads[key] = value
    self.graph[key] = value, tuple(id(item) for item in items if id(item) in self.graph)
    self.postorder.append(key)
    return True

  def merge(self, previous, incoming):
    if previous is None:
      return incoming
    if incoming is None:
      return previous
    return previous if previous is incoming or previous == incoming else self.escaped

  def consume(self, value, owner, group, *, arguments=False):
    mapping = self.argument_owners if arguments else self.owners
    pending = [(id(value), (owner, group))]
    while pending:
      key, incoming = pending.pop()
      if key not in self.graph:
        continue
      previous = mapping.get(key)
      merged = incoming if previous is None else tuple(self.merge(a, b) for a, b in zip(previous, incoming))
      if previous == merged:
        continue
      mapping[key] = merged
      subject, edges = self.graph[key]
      if arguments:
        edges = tuple(id(arg) for arg in subject.args) if type(subject) is I.Expr else ()
      pending.extend((edge, merged) for edge in edges)

  def eligible_value(self, value):
    if value is None:
      return True
    key = id(value)
    if key in self.pointwise:
      return self.pointwise[key]
    self.pointwise[key] = False
    boundaries = Trait.REDUCTION | Trait.REGISTER_TRANSFORM | Trait.VECTOR_BOUNDARY | Trait.STATE_REFERENCE
    if (
      type(value) is not I.Expr
      or self.catalog.get(value).effects not in {"none", "read"}
      or self.catalog.has(value, boundaries)
      or isinstance(value.type, I.TileType)
      and value.op != "load"
      and not self.catalog.has(value, Trait.ELEMENTWISE)
      or self.catalog.has(value, Trait.MEMORY_READ)
      and (value.op != "load" or type(value.data) is not I.View)
    ):
      return False
    answer = all(self.eligible_value(arg) for arg in value.args)
    self.pointwise[key] = answer
    return answer

  def summaries(self):
    # Four scalar owners per node summarize all reachable reads: exact/group
    # consumers, followed by their argument-only counterparts. An unconsumed
    # or metadata-only read is escaped, never silently omitted.
    result = {}
    for key in self.postorder:
      owner = self.owners.get(key, (self.escaped,) * 2)
      argument = self.argument_owners.get(key, (self.escaped,) * 2)
      summary = (*owner, *argument) if key in self.memory_reads else (None,) * 4
      for edge in self.graph[key][1]:
        summary = tuple(self.merge(a, b) for a, b in zip(summary, result[edge]))
      result[key] = summary
    return result


def snapshot_lifetime_scopes(kernel, catalog):
  """Prove lexical consumers with linear direct-edge ownership propagation.

  BindTile is a historical definition, not a memory evaluation. Actual uses
  include owned metadata and loop/branch headers. Each exact consumer and its
  enclosing group has a first-owner/escaped state; shared syntax, opaque
  ownership and specialized execution retain the existing owner lifetime.
  Emission additionally rejects newly protected future reads.
  """
  query = _SnapshotOwners(catalog)
  if not query.inspect(kernel.body):
    return {}
  candidates, occurrences = [], {}
  serial = 0

  def eligible(store):
    return (
      type(store) is I.Store
      and type(store.destination) is I.View
      and query.eligible_value(store.value)
      and query.eligible_value(store.mask)
    )

  def visit(block, blocked=False, group=None):
    nonlocal serial
    for statement in block.statements:
      serial += 1
      token = serial
      occurrences[id(statement)] = occurrences.get(id(statement), 0) + 1
      if type(statement) is I.ElementwiseGroup:
        visit(I.Block(list(statement.stores)), blocked, token)
        if not blocked and all(eligible(store) for store in statement.stores):
          candidates.append((statement, token, 1))
        continue
      grouped = token if group is None else group
      # BindTile only defines history. A later consumer, including one outside
      # this branch/loop/helper, still propagates its own distinct owner.
      if type(statement) is not I.BindTile:
        for descriptor in fields(statement):
          value = getattr(statement, descriptor.name)
          if type(value) is not I.Block:
            query.consume(value, token, grouped)
      if eligible(statement):
        for value in (statement.value, statement.mask):
          query.consume(value, token, grouped, arguments=True)
        if not blocked:
          candidates.append((statement, token, 0))
      nested_blocked = blocked or type(statement) not in (I.For, I.If, I.While)
      for child in children(statement):
        visit(child, nested_blocked)

  visit(kernel.body)
  summaries = query.summaries()
  reads_by_owner = ({}, {})
  for key, read in query.memory_reads.items():
    for dimension, owner in enumerate(query.owners.get(key, (query.escaped,) * 2)):
      reads_by_owner[dimension].setdefault(owner, []).append(read)
  result = {}
  for subject, token, dimension in candidates:
    summary = summaries[id(subject)]
    if occurrences[id(subject)] == 1 and summary[dimension] == token and summary[dimension + 2] == token:
      result[id(subject)] = SnapshotScope(subject, tuple(reads_by_owner[dimension][token]), 1)
  return result


@dataclass(frozen=True)
class StoragePlan:
  scopes: object
  slots: object
  intervals: tuple[ScratchInterval, ...]
  snapshot_scopes: object = field(default_factory=lambda: MappingProxyType({}))


@register_pass("cpu-plan-storage", stage="cpu-memory", after="cpu-plan-hardware")
def plan_storage(prepared, context):
  local = context.options.get("cpu_branch_local_buffers", True)
  reuse = context.options.get("cpu_scratch_reuse", True)
  snapshots = context.options.get("cpu_snapshot_lifetime_scopes", True)
  if type(local) is not bool:
    raise TypeError("cpu_branch_local_buffers must be bool")
  if type(reuse) is not bool:
    raise TypeError("cpu_scratch_reuse must be bool")
  if type(snapshots) is not bool:
    raise TypeError("cpu_snapshot_lifetime_scopes must be bool")
  kernel = prepared.vectors.kernel
  scopes = branch_buffer_scopes(kernel.body) if local else {}
  intervals, allocations, references, forbidden = [], {}, {}, set()
  effects = ValueEffects(context.environment.operations)
  assigned = assigned_scalars(kernel.body)
  for index, statement in enumerate(kernel.body.statements):
    # Worker callbacks have their own declaration/parallel ownership. Do not
    # assign their storage to invocation-level slots even on a serial fallback.
    for region in nodes(statement):
      if isinstance(region, (I.Grid, I.Team, I.HardwareRegion)):
        forbidden.update(node.view.storage for node in nodes(region) if isinstance(node, I.Allocate))
    for node in nodes(statement):
      if isinstance(node, I.Allocate):
        allocations.setdefault(node.view.storage, node.view)
      elif isinstance(node, I.View):
        first, _ = references.get(node.storage, (index, index))
        references[node.storage] = first, index
  for name, view in allocations.items():
    if not view.owned or name in forbidden or name not in references:
      continue
    dimensions = tuple(I.constant(extent) for extent in view.shape)
    capacity = math.prod(dimensions) if all(type(v) is int and v >= 0 for v in dimensions) else None
    if capacity is None and any(
      not effects.speculatable(extent) or scalar_references(extent) & assigned for extent in view.shape
    ):
      continue
    first, last = references[name]
    intervals.append(ScratchInterval(name, view.dtype, first, last, scopes.get(name), capacity, view.shape))
  intervals.sort(key=lambda value: (value.first, value.storage))
  slots, available = {}, []
  for interval in intervals:
    owner = None
    if reuse:
      for index, (previous, slot) in enumerate(available):
        compatible = (
          previous.last < interval.first
          and previous.dtype == interval.dtype
          and previous.scope == interval.scope
          and (
            previous.capacity is not None
            and interval.capacity is not None
            and previous.capacity >= interval.capacity
            or previous.capacity is None
            and interval.capacity is None
            and len(previous.shape) == len(interval.shape)
            and all(I.same_expr(a, b) for a, b in zip(previous.shape, interval.shape))
          )
        )
        if compatible:
          owner = slot
          # Preserve the slot's actual capacity after a smaller reuse; the
          # Buffer implementation retains capacity on shrinking resize().
          previous = replace(interval, capacity=previous.capacity)
          available[index] = previous, slot
          break
    if owner is None:
      owner = interval.storage
      available.append((interval, owner))
    slots[interval.storage] = owner
  if reuse and len(set(slots.values())) < len(slots):
    kernel.diagnostics.append(
      f"scratch lifetime reuse: {len(slots)} logical allocations in {len(set(slots.values()))} slots"
    )
  snapshot_scopes = snapshot_lifetime_scopes(kernel, context.environment.operations) if snapshots else {}
  plan = StoragePlan(
    MappingProxyType(scopes), MappingProxyType(slots), tuple(intervals), MappingProxyType(snapshot_scopes)
  )
  return replace(prepared, storage=plan)
