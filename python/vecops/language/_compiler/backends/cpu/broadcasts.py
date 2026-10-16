# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Prove broadcast compatibility and equality in final lexical domains.

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
class ExtentGeometry:
  subject: object
  classes: tuple
  representatives: tuple = ()


@dataclass(frozen=True)
class BroadcastAnalysis:
  proofs: object
  geometry: object


def broadcast_key(store, source, destination):
  return id(store), id(source), id(destination)


def make_broadcast_collector(region_shapes=None):
  """Collect compatibility and equality independently at every occurrence."""
  proofs, counts, failed, unequal, subjects = {}, {}, set(), set(), {}
  definitions, excluded, available = {}, set(), {}
  group_edges, group_failed, group_available = {}, set(), {}
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
          a, b = scope.normalize(source), scope.normalize(destination)
          equal = a is not None and b is not None and scope.range(I.binary("sub", a, b)) == (0, 0)
          visible = {
            id(value)
            for value in (source, destination)
            if value.op in {"local", "frozen"} and not value.args and scope.aliases.lookup(value) is not None
          }
          available[key] = available.get(key, visible) & visible
          if not equal:
            unequal.add(key)
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
            a, b = scope.normalize(source), scope.normalize(destination)
            if a is None or b is None or scope.range(I.binary("sub", a, b)) != (0, 0):
              group_failed.add(key)
            visible = {
              id(value)
              for value in (source, destination)
              if value.op in {"local", "frozen"} and not value.args and scope.aliases.lookup(value) is not None
            }
            group_available[key] = group_available.get(key, visible) & visible
            group_edges[key] = (source, destination)

  def finish():
    result = {
      key: BroadcastProof(p.store, p.source, p.destination, p.occurrences, key not in unequal)
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
    by_store = {}
    for proof in result.values():
      if proof.equal:
        by_store.setdefault(id(proof.store), []).append(
          (proof.source, proof.destination, available[broadcast_key(proof.store, proof.source, proof.destination)])
        )
    by_group = {}
    for key, pair in group_edges.items():
      if key not in group_failed:
        by_group.setdefault(key[0], []).append((*pair, group_available[key]))
    geometry = {}
    for identity, (subject, stores) in subjects.items():
      components, visible = [], set()
      edges = [edge for store in stores for edge in by_store.get(id(store), ())]
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
        geometry[identity] = ExtentGeometry(subject, classes, representatives)
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
