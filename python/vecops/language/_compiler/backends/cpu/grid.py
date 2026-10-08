# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Conservative grid ownership proofs and target-local scheduling estimates.

No alias promise is inferred from parameter names. Proven logical ownership
still requires runtime non-overlapping layouts and cross-parameter alias guards.
Lost view provenance, loop-carried state and unknown effects retain serial order.
"""

from __future__ import annotations

from dataclasses import dataclass

from ... import ir as I
from ...analysis.affine import affine
from ...cost import CostQuery, estimate
from ...ir.traversal import children, values, walk
from ...operations import Trait


@dataclass(frozen=True)
class GridPlan:
  safe: bool
  guards: tuple
  reads: frozenset
  writes: frozenset
  private: frozenset
  cycles: object
  reason: str = ""


@dataclass(frozen=True)
class GridAxisGuard:
  base: object
  width: object
  factor: int
  programs: object | None = None
  extent: object | None = None
  clipped: bool = False


def plan_grid(grid, environment):
  definitions, reads, stores, private, unsupported, local_names, mutable, pack_roots = (
    {},
    [],
    [],
    set(),
    [],
    set(),
    set(),
    [],
  )

  def assignment(statement):
    if statement.declare:
      local_names.add(statement.target)
      definitions[statement.target] = statement.value
    else:
      if statement.target not in local_names:
        unsupported.append("shared mutable scalar or loop-carried state")
      mutable.add(statement.target)

  def invariant(expression):
    return all(
      isinstance(e.type, I.ScalarType)
      and environment.operations.get(e).effects == "none"
      and not environment.operations.has(e, Trait.REDUCTION)
      and not (e.op in {"local", "frozen"} and e.data in local_names)
      for e in walk(expression)
    )

  def visit(block):
    for statement in block.statements:
      if isinstance(statement, I.Assign):
        assignment(statement)
      if isinstance(statement, I.ReductionGroup):
        for item in statement.assignments:
          assignment(item)
      if isinstance(statement, I.For):
        local_names.add(statement.target)
      if isinstance(statement, I.UnrolledFor):
        local_names.add(statement.loop.target)
      if isinstance(statement, I.Grid):
        local_names.update(statement.targets)
      if isinstance(statement, I.Allocate):
        private.add(statement.view.storage)
      if isinstance(statement, I.DeclareState):
        private.add(statement.state.name)
      if isinstance(statement, I.UpdateState) and statement.state.name not in private:
        unsupported.append("shared mutable tile state")
      if isinstance(statement, I.Pack):
        private.add(statement.destination.view.storage)
        pack_roots.append(statement.source)
      if isinstance(statement, I.Store):
        stores.append(statement.destination)
      if isinstance(statement, I.IndexedStore):
        unsupported.append("indexed writes have no proven program ownership")
      if isinstance(statement, (I.ElementwiseGroup, I.ReductionGroup)):
        stores.extend(s.destination for s in statement.stores)
      if isinstance(statement, I.ArchitectureCall):
        unsupported.append("explicit hardware effects")
      if isinstance(statement, I.Control) and statement.kind in {"break", "continue"}:
        unsupported.append("grid control transfer")
      for child in children(statement):
        visit(child)

  visit(grid.body)
  for name in mutable:
    definitions.pop(name, None)
  roots = tuple(values(grid.body)) + tuple(pack_roots)
  for root in roots:
    for expression in walk(root):
      effect = environment.operations.get(expression).effects
      if effect not in {"none", "read"}:
        unsupported.append("unknown or stateful expression effects")
      if effect == "read":
        if environment.operations.has(expression, Trait.STATE_REFERENCE) and expression.data.name in private:
          continue
        view = expression.data.view if isinstance(expression.data, I.PackedView) else expression.data
        if isinstance(view, I.View):
          reads.append(view)
        else:
          unsupported.append("read effect has no known memory region")
  writes = [view for view in stores if view.storage not in private]
  read_views = [view for view in reads if view.storage not in private]
  roots_read, roots_write = frozenset(v.storage for v in read_views), frozenset(v.storage for v in writes)
  guards, ownership = [], {}
  variables = set(grid.targets)

  def reject(reason):
    return GridPlan(False, (), roots_read, roots_write, frozenset(private), I.literal(1), reason)

  if unsupported:
    return reject(unsupported[0])
  for view in writes:
    region = view.region
    if region is None or not region.affine_partition:
      return reject("write view has no root-coordinate provenance")
    mapped = {}
    for axis, start in enumerate(region.starts):
      result = affine(start, variables, definitions)
      if result is None or len(result[0]) > 1:
        return reject("nonseparable program coordinates")
      if not result[0]:
        continue
      ((name, factor),) = result[0].items()
      width = affine(region.widths[axis], variables, definitions)
      if factor <= 0 or width is None or width[0] or name in mapped:
        return reject("nonmonotonic or overlapping program regions")
      base = result[1]
      extent = grid.extents[grid.targets.index(name)]
      guards.append(GridAxisGuard(base, width[1], factor, extent, region.root_shape[axis], region.clipped[axis]))
      mapped[name] = (axis, factor)
    if any(name not in mapped and I.constant(extent) != 1 for name, extent in zip(grid.targets, grid.extents)):
      return reject("multiple programs can write the same region")
    old = ownership.setdefault(view.storage, mapped)
    if old != mapped:
      return reject("incompatible output ownership")

  # Same-storage reads must stay within the same program-owned partition.
  # Different parameter storages are checked for runtime aliasing by emission.
  for view in read_views:
    if view.storage not in ownership:
      continue
    if view.region is None or not view.region.affine_partition:
      return reject("read/write dependence has unknown coordinates")
    for name, (axis, factor) in ownership[view.storage].items():
      start = affine(view.region.starts[axis], variables, definitions)
      width = affine(view.region.widths[axis], variables, definitions)
      if start is None or start[0] != {name: factor} or width is None or width[0]:
        return reject("cross-program read/write dependence")
      guards.append(GridAxisGuard(start[1], width[1], factor))

  for guard in guards:
    for expression in (guard.base, guard.width):
      if not invariant(expression):
        return reject("ownership guard depends on program-local state")

  # Cost is per program, not per scalar element: use vector-region word costs.
  scalars = {id(e): e for root in roots for e in walk(root) if isinstance(e.type, I.ScalarType)}
  vectors = {id(e): e for root in roots for e in walk(root) if isinstance(e.type, I.TileType)}
  # Reductions are boundaries for fusion estimation, but their producers still
  # execute in this program. Reuse the centralized DAG model and native lanes.
  cost_roots = roots + tuple(
    e.args[0] for root in roots for e in walk(root) if environment.operations.has(e, Trait.REDUCTION)
  )
  cost = estimate(cost_roots, environment.costs, boundaries=scalars)
  scalar_cost = estimate(tuple(scalars.values()), environment.costs, boundaries=vectors, implementation="scalar")
  reductions = {id(e) for root in roots for e in walk(root) if environment.operations.has(e, Trait.REDUCTION)}
  # Scalar epilogues run once per program. Dependency latency is startup cost,
  # not the throughput cost of every vector word in a long streaming loop.
  cycles = I.literal(1.0 + scalar_cost.estimated_cycles + cost.critical_path_cycles)
  word_cycles = max(1, cost.compute_cycles + len(reductions), cost.memory_cycles + len(writes))
  for statement in writes:
    dimensions = statement.shape
    elements = I.literal(1)
    for dimension in dimensions:
      resolved = affine(dimension, variables, definitions)
      if resolved is None or resolved[0] or not invariant(resolved[1]):
        elements = I.literal(1024)  # conservative estimate only, never a proof
        break
      elements = I.binary("mul", elements, resolved[1])
    lanes = I.Expr(I.ScalarType("int64"), "lanes", data=("uint8" if statement.dtype == "bool" else statement.dtype, 0))
    words = I.binary("div", I.binary("mul", elements, I.literal(1.0)), lanes)
    cycles = I.binary("add", cycles, I.binary("mul", words, I.literal(word_cycles)))
  dots = {id(e): e for root in roots for e in walk(root) if e.op == "dot"}
  for dot in dots.values():
    products = I.literal(1.0)
    for dimension in (*dot.type.shape, dot.args[0].type.shape[1]):
      resolved = affine(dimension, variables, definitions)
      is_invariant = resolved is not None and not resolved[0] and invariant(resolved[1])
      products = I.binary("mul", products, resolved[1] if is_invariant else I.literal(32))
    matrix = dot.data.implementation != "vector" and dot.args[0].type.dtype == dot.args[1].type.dtype
    query = CostQuery(
      "dot" if matrix else "fma",
      "matrix" if matrix else "arithmetic",
      dot.type.dtype,
      dot.args[0].type.dtype,
      implementation="grid-word-mac" if matrix else "vector",
    )
    throughput = environment.costs.lookup(query).throughput_cycles
    lanes = I.Expr(I.ScalarType("int64"), "lanes", data=(dot.type.dtype, 0))
    cycles = I.binary("add", cycles, I.binary("mul", I.binary("div", products, lanes), I.literal(throughput)))
  return GridPlan(True, tuple(guards), roots_read, roots_write, frozenset(private), cycles)


def short_scalar_phase(statement, environment):
  if not isinstance(statement, I.Assign) or not statement.declare:
    return False
  if any(
    isinstance(e.type, I.TileType) or environment.operations.get(e).effects != "none" for e in walk(statement.value)
  ):
    return False
  return estimate(
    statement.value, environment.costs, implementation="scalar"
  ).estimated_cycles <= environment.costs.policy("grid_single_cycles", 64)


def region_metadata(grid, plan):
  yield from grid.extents
  yield plan.cycles
  for guard in plan.guards:
    yield from (guard.base, guard.width, guard.programs, guard.extent)


def plan_regions(block, plans, mode, environment):
  """Group legal phases, without moving program-dependent metadata or effects."""
  regions, consumed = {}, set()
  for index, statement in enumerate(block.statements):
    if index in consumed or not isinstance(statement, I.Grid):
      continue
    phases = [statement]
    if mode != "individual" and statement.parallel != "serial" and plans[id(statement)].safe:
      pending = []
      for later_index, later in enumerate(block.statements[index + 1 :], index + 1):
        if isinstance(later, I.Grid) and later.parallel != "serial" and plans[id(later)].safe:
          names = {s.target for _, s in pending} | {s.target for s in phases if isinstance(s, I.Assign)}
          if any(
            environment.operations.get(e).effects != "none"
            or environment.operations.has(e, Trait.REDUCTION)
            or e.op in {"local", "frozen"}
            and e.data in names
            for value in region_metadata(later, plans[id(later)])
            if value is not None
            for e in walk(value)
          ):
            break
          phases.extend(s for _, s in pending)
          consumed.update(i for i, _ in pending)
          pending.clear()
          phases.append(later)
          consumed.add(later_index)
        elif mode in {"auto", "template"} and short_scalar_phase(later, environment):
          pending.append((later_index, later))
        else:
          break
    regions[id(statement)] = tuple(phases)
  return regions
