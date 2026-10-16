# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Prove broadcast compatibility and total scalar equality in lexical domains.

Historical Tile/View descriptors retain their reader identities. Compatibility
can remove a validation; only equality can remove coordinate projection.
"""

from dataclasses import dataclass
from types import MappingProxyType

from ... import ir as I
from .partitions import walk_lexical_scopes


@dataclass(frozen=True)
class BroadcastProof:
  store: I.Store
  source: I.Expr
  destination: I.Expr
  occurrences: int
  equal: bool = False


@dataclass(frozen=True)
class ScalarGeometry:
  """Equal bounded int64 expressions available at one exact store/group.

  Shapes, offsets and strides share this certificate. It never establishes
  pointer disjointness, authorizes an effect, or rebinds historical readers.
  """
  subject: object
  classes: tuple
  representatives: tuple = ()


@dataclass(frozen=True)
class BroadcastAnalysis:
  proofs: object
  geometry: object


def broadcast_key(store, source, destination):
  return id(store), id(source), id(destination)


class _ScalarEqualityCollector:
  """Intersect equality and available representatives over every occurrence."""

  def __init__(self):
    self.edges, self.failed, self.available = {}, set(), {}

  def observe(self, subject, source, destination, scope):
    key = (id(subject), id(source), id(destination))
    a, b = scope.normalize(source), scope.normalize(destination)
    equal = a is not None and b is not None and scope.range(I.binary("sub", a, b)) == (0, 0)
    if not equal:
      self.failed.add(key)
    visible = {
      id(value) for value in (source, destination)
      if value.op in {"local", "frozen"} and not value.args and scope.aliases.lookup(value) is not None
    }
    self.available[key] = self.available.get(key, visible) & visible
    self.edges[key] = (source, destination)
    return equal

  def confirmed(self):
    by_subject = {}
    for key, pair in self.edges.items():
      if key not in self.failed:
        by_subject.setdefault(key[0], []).append((*pair, self.available[key]))
    return by_subject


def make_broadcast_collector(region_shapes=None):
  """Collect compatibility and equality independently at every occurrence."""
  proofs, counts, failed, subjects = {}, {}, set(), {}
  definitions, excluded = {}, set()
  shape_equalities, group_equalities, metadata_equalities = (_ScalarEqualityCollector() for _ in range(3))
  region_shapes = region_shapes or {}

  def observe(statement, scope):
    assignments = statement.assignments if isinstance(statement, I.ReductionGroup) else (statement,)
    for assignment in assignments:
      if isinstance(assignment, (I.Assign, I.For)):
        definitions.setdefault(assignment.target, {})[id(assignment)] = assignment
    if isinstance(statement, (I.For, I.TeamReduce)):
      excluded.add(statement.target)
    elif isinstance(statement, I.Grid):
      excluded.update(statement.targets)
    stores = statement.stores if type(statement) is I.ElementwiseGroup else (statement,)
    if type(statement) in {I.Store, I.ElementwiseGroup}:
      subjects[id(statement)] = (statement, stores)
    for store in stores:
      if type(store) is not I.Store:
        continue
      subjects[id(store)] = (store, (store,))
      # Reuse total scalar metadata already available at this memory operation.
      # Equality does not establish pointer disjointness or move any read.
      if type(store.destination) is I.View:
        pairs = []
        for operand in (store.value, store.mask):
          if operand is None:
            continue
          for load in I.loads_unique(operand):
            if type(load.data) is not I.View:
              continue
            pairs.append((load.data.offset, store.destination.offset))
            pairs.extend(zip(reversed(load.data.strides), reversed(store.destination.strides)))
        for source, destination in pairs:
          if I.same_extent(source, destination):
            continue
          metadata_equalities.observe(store, source, destination, scope)
      occurrence = set()
      for operand in (store.value, store.mask):
        if operand is None or not isinstance(operand.type, I.TileType):
          continue
        if len(operand.type.shape) > len(store.destination.shape):
          continue
        for source, destination in zip(reversed(operand.type.shape), reversed(store.destination.shape)):
          # Match the unconditional emitter cases before bounded analysis.
          if I.constant(source) == 1 or I.same_extent(source, destination):
            continue
          key = broadcast_key(store, source, destination)
          if key in occurrence:
            continue
          occurrence.add(key)
          counts[key] = counts.get(key, 0) + 1
          equal = shape_equalities.observe(store, source, destination, scope)
          a = scope.normalize(source)
          compatible = equal or a is not None and scope.range(a) == (1, 1)
          if not compatible:
            failed.add(key)
          proofs[key] = BroadcastProof(store, source, destination, counts[key])

    if type(statement) is I.ElementwiseGroup and stores:
      # lower_stores uses the first destination as its iteration root. A
      # prepared plan can retain a distinct historical root, so production
      # supplies the actual final vector region shape instead.
      root = region_shapes.get(id(statement), stores[0].destination.shape)
      occurrence = set()
      for store in stores:
        shapes = [store.destination.shape]
        shapes.extend(
          operand.type.shape
          for operand in (store.value, store.mask)
          if operand is not None and isinstance(operand.type, I.TileType)
        )
        for shape in shapes:
          if len(shape) > len(root):
            continue
          for source, destination in zip(reversed(shape), reversed(root)):
            if source is destination:
              continue
            key = (id(statement), id(source), id(destination))
            if key in occurrence:
              continue
            occurrence.add(key)
            group_equalities.observe(statement, source, destination, scope)

  def finish():
    result = {
      key: BroadcastProof(p.store, p.source, p.destination, p.occurrences, key not in shape_equalities.failed)
      for key, p in proofs.items()
      if key not in failed
    }
    immutable = {
      name
      for name, sites in definitions.items()
      if name not in excluded
      and len(sites) == 1
      and type(next(iter(sites.values()))) is I.Assign
      and next(iter(sites.values())).declare
    }
    by_store, by_group, by_metadata = shape_equalities.confirmed(), group_equalities.confirmed(), metadata_equalities.confirmed()
    geometry = {}
    for identity, (subject, stores) in subjects.items():
      components, visible = [], set()
      edges = [edge for store in stores for edge in by_store.get(id(store), ())]
      edges.extend(edge for store in stores for edge in by_metadata.get(id(store), ()))
      if type(subject) is I.ElementwiseGroup:
        edges.extend(by_group.get(identity, ()))
      for a, b, available_here in edges:
        visible.update(available_here)
        joined, remaining = {id(a): a, id(b): b}, []
        for component in components:
          if any(key in component for key in joined):
            joined.update(component)
          else:
            remaining.append(component)
        components = [*remaining, joined]
      if components:
        # Insertion order follows final IR/store/axis order, never object ids.
        classes = tuple(tuple(c.values()) for c in components)
        representatives = _representatives(classes, immutable, visible)
        geometry[identity] = ScalarGeometry(subject, classes, representatives)
    return BroadcastAnalysis(MappingProxyType(result), MappingProxyType(geometry))

  return observe, finish


def _representatives(classes, immutable, visible):
  """Choose original available values, rejecting any alias dependency cycle."""
  from ...ir.traversal import walk_unique

  representatives = []
  for component in classes:
    candidates = []
    for index, expression in enumerate(component):
      binding = expression.op in {"local", "frozen"}
      if binding and (expression.data not in immutable or id(expression) not in visible):
        continue
      # Prefer an existing dominating immutable scalar over spelling out its
      # larger historical clamp again. Other original expressions use DAG cost
      # and final IR encounter order, never ids or generated-name heuristics.
      candidates.append(((0 if binding else 1, len(tuple(walk_unique(expression))), index), expression))
    representatives.append(min(candidates, key=lambda item: item[0])[1] if candidates else None)
  aliases = {
    id(expression): representative
    for component, representative in zip(classes, representatives)
    if representative is not None
    for expression in component
    if expression is not representative
  }
  active, seen = set(), set()

  def acyclic(expression):
    if id(expression) in active:
      return False
    if id(expression) in seen:
      return True
    active.add(id(expression))
    replacement = aliases.get(id(expression))
    children = (replacement,) if replacement is not None else expression.args
    if not all(acyclic(child) for child in children):
      return False
    active.remove(id(expression))
    seen.add(id(expression))
    return True

  # Two equivalence classes may depend on each other's members even when each
  # original DAG is acyclic. Keep the equality query, but emit original scalar
  # expressions if choosing representatives would introduce such a cycle.
  return (
    tuple(representatives)
    if all(acyclic(value) for component in classes for value in component)
    else (None,) * len(classes)
  )


def collect_broadcast_analysis(kernel, *, region_shapes=None):
  observe, finish = make_broadcast_collector(region_shapes)
  walk_lexical_scopes(kernel, observe)
  return finish()


def collect_broadcast_proofs(kernel):
  """Standalone compatibility analysis, retaining all historical subjects."""
  return collect_broadcast_analysis(kernel).proofs
