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
from ...analysis.affine import affine, affine_contains
from ...cost import CostQuery, estimate
from ...dtypes import ELEMENT_BYTES
from ...ir.traversal import children, nodes, values, walk
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

  # These operations have exact memory effects and region-local registers.
  # Owner preflight still validates every hardware access before entry.
  hardware_reads = {"amx.load", "amx.stream_load", "sme.load", "sme.load_bits", "sme.load_hor", "sme.load_ver"}
  hardware_writes = {"amx.store", "sme.store", "sme.store_bits", "sme.store_tile", "sme.store_hor", "sme.store_ver"}
  hardware_registers = {
    "amx.zero",
    "amx.tdp",
    "sme.zero_za",
    "sme.mopa",
    "sme.mask",
    "sme.apply",
    "sme.read_hor",
    "sme.read_ver",
    "sme.write_hor",
    "sme.write_ver",
  }
  vector_creators = {"sme.load", "sme.load_bits", "sme.mask", "sme.apply", "sme.read_hor", "sme.read_ver"}
  native_products = []

  def instruction(statement, owner, loops):
    operation = statement.operation
    if (
      owner is None
      or not operation.startswith(owner["kind"] + ".")
      or operation not in hardware_reads | hardware_writes | hardware_registers
    ):
      unsupported.append("explicit hardware effects have no known lexical owner")
      return
    handles = [v for v in nodes((statement.vector, statement.operands)) if isinstance(v, (I.SMEVector, I.SMEMask))]
    created = statement.vector if operation in vector_creators else None
    for handle in handles:
      if handle is created:
        continue
      if handle.name not in owner["vectors"]:
        unsupported.append("hardware vector escapes its lexical owner")
    # Slice indices and address/shape metadata can capture payload reads too.
    # Generic values() deliberately does not expose ArchitectureCall fields.
    for expression in nodes(statement):
      if isinstance(expression, I.Expr):
        if expression.op == "vector_ref" and expression.data not in owner["vectors"]:
          unsupported.append("hardware expression captures another owner's vector")
        pack_roots.append(expression)
    if created is not None:
      owner["vectors"].add(created.name)
    if operation in {"amx.tdp", "sme.mopa"}:
      valid = len(statement.operands) == 3
      if operation == "amx.tdp":
        valid = valid and all(isinstance(v, I.AMXTile) for v in statement.operands)
        valid = valid and len({v.region for v in statement.operands}) == 1
        valid = valid and all(0 <= v.index < len(owner["configuration"]) for v in statement.operands)
      else:
        valid = (
          valid
          and isinstance(statement.operands[0], str)
          and all(isinstance(v, I.SMEVector) for v in statement.operands[1:])
        )
      if valid:
        native_products.append((statement, owner["configuration"], loops))
      else:
        unsupported.append("hardware matrix operands have no local owner contract")
    if operation in hardware_reads | hardware_writes:
      if statement.view is None:
        unsupported.append("hardware memory effect has no known view")
      elif operation in hardware_reads:
        reads.append(statement.view)
      else:
        stores.append(statement.view)

  def owner_copy(owner):
    return None if owner is None else {**owner, "vectors": set(owner["vectors"])}

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

  def visit(block, owner=None, loops=()):
    for statement in block.statements:
      if isinstance(statement, I.HardwareRegion):
        if owner is not None or statement.kind not in {"amx", "sme"}:
          unsupported.append("unsupported nested hardware owner")
        else:
          pack_roots.extend(value for value in nodes(statement.configuration) if isinstance(value, I.Expr))
          visit(
            statement.body, {"kind": statement.kind, "vectors": set(), "configuration": statement.configuration}, loops
          )
        continue
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
        # Rank-one scatter retains a checked destination view: emission
        # validates every active index against that view's extent before the
        # write. Its complete root-coordinate partition can therefore prove
        # program ownership, regardless of duplicate indices within a program.
        region = statement.destination.region
        if len(statement.destination.shape) == 1 and region is not None and region.affine_partition:
          stores.append(statement.destination)
        else:
          unsupported.append("indexed writes have no proven program ownership")
      if isinstance(statement, (I.ElementwiseGroup, I.ReductionGroup)):
        stores.extend(s.destination for s in statement.stores)
      if isinstance(statement, I.ArchitectureCall):
        instruction(statement, owner, loops)
      if isinstance(statement, I.Control) and statement.kind in {"break", "continue"}:
        unsupported.append("grid control transfer")
      for child in children(statement):
        visit(child, owner_copy(owner), (*loops, statement) if isinstance(statement, I.For) else loops)

  visit(grid.body)
  for name in mutable:
    definitions.pop(name, None)
  roots = tuple(values(grid.body)) + tuple(pack_roots)
  expressions, seen = [], set()
  pending = list(reversed(roots))
  while pending:
    expression = pending.pop()
    if id(expression) not in seen:
      seen.add(id(expression))
      expressions.append(expression)
      pending.extend(reversed(expression.args))
  for expression in expressions:
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
    if region is None or (not region.affine_partition and region.exact_starts is None):
      return reject("write view has no root-coordinate provenance")
    # Nested clipping can invalidate a requested column origin while leaving
    # an already selected batch coordinate exact. Only untouched root axes
    # may prove ownership in that case; never reuse the requested clamped axis.
    starts = region.starts if region.affine_partition else region.exact_starts
    mapped = {}
    for axis, start in enumerate(starts):
      # An inner serial row/column may vary inside this program's partition.
      # Only axes depending on a grid coordinate determine disjoint programs;
      # frontend view bounds and indexed-memory bounds remain checked.
      if not affine_contains(start, variables, definitions):
        continue
      if not region.affine_partition and region.clipped[axis]:
        return reject("clipped program coordinate has no affine ownership")
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
    if view.region is None or (not view.region.affine_partition and view.region.exact_starts is None):
      return reject("read/write dependence has unknown coordinates")
    starts = view.region.starts if view.region.affine_partition else view.region.exact_starts
    for name, (axis, factor) in ownership[view.storage].items():
      if not view.region.affine_partition and view.region.clipped[axis]:
        return reject("clipped read/write coordinate has no affine ownership")
      start = affine(starts[axis], variables, definitions)
      width = affine(view.region.widths[axis], variables, definitions)
      if start is None or start[0] != {name: factor} or width is None or width[0]:
        return reject("cross-program read/write dependence")
      guards.append(GridAxisGuard(start[1], width[1], factor))

  for guard in guards:
    for expression in (guard.base, guard.width):
      if not invariant(expression):
        return reject("ownership guard depends on program-local state")

  # Cost is per program, not per scalar element: use vector-region word costs.
  scalars = {id(e): e for e in expressions if isinstance(e.type, I.ScalarType)}
  vectors = {id(e): e for e in expressions if isinstance(e.type, I.TileType)}
  # Reductions are boundaries for fusion estimation, but their producers still
  # execute in this program. Reuse the centralized DAG model and native lanes.
  cost_roots = roots + tuple(e.args[0] for e in expressions if environment.operations.has(e, Trait.REDUCTION))
  cost = estimate(cost_roots, environment.costs, boundaries=scalars)
  scalar_cost = estimate(tuple(scalars.values()), environment.costs, boundaries=vectors, implementation="scalar")
  reductions = {id(e) for e in expressions if environment.operations.has(e, Trait.REDUCTION)}
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
  dots = {id(e): e for e in expressions if e.op == "dot"}
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

  # Architectural expansion removes logical dot roots. Account for those
  # instructions using the same matrix cost profile, including enclosing
  # serial trips. This is a scheduling estimate, never an ownership proof.
  def program_constant(value, fallback):
    resolved = affine(value, variables, definitions)
    if resolved is None or resolved[0] or not invariant(resolved[1]):
      return fallback
    # Cost queries are emitted outside the grid. View/proof payloads can hide
    # references which are absent from ordinary argument-edge traversal.
    if any(isinstance(e, I.Expr) and e.op in {"local", "frozen"} and e.data in local_names for e in nodes(resolved[1])):
      return fallback
    return resolved[1]

  def floating(value):
    return I.Expr(I.ScalarType("float64"), "cast", (value,))

  for call, configuration, loops in native_products:
    if call.operation == "amx.tdp":
      acc, a, _ = call.operands
      dtype, operand_dtype = acc.dtype, a.dtype
      rows, columns = configuration[acc.index]
      products = I.literal(
        float(rows * columns // ELEMENT_BYTES[dtype] * configuration[a.index][1] // ELEMENT_BYTES[operand_dtype])
      )
    else:
      dtype, a, b = call.operands
      operand_dtype = a.dtype
      group = ELEMENT_BYTES[dtype] // ELEMENT_BYTES[operand_dtype]
      maximum = I.Expr(I.ScalarType("int64"), "streaming_lanes", data=operand_dtype)
      na, nb = (program_constant(v.extent, maximum) for v in (a, b))
      products = I.binary("div", I.binary("mul", floating(na), floating(nb)), I.literal(float(group)))
    for loop in loops:
      step = program_constant(loop.step, I.literal(0))
      step_value = I.constant(step)
      if type(step_value) is int and step_value > 0:
        start, stop = (program_constant(v, None) for v in (loop.start, loop.stop))
        if start is not None and stop is not None:
          # Convert before subtracting so even a very large legal metadata
          # range cannot overflow signed integer arithmetic in grid_metadata.
          distance = I.binary("sub", floating(stop), floating(start))
          trips = I.binary(
            "max",
            I.literal(0.0),
            I.binary("div", I.binary("add", distance, I.literal(float(step_value - 1))), I.literal(float(step_value))),
          )
        else:
          trips = I.literal(16.0)
      else:
        trips = I.literal(16.0)
      products = I.binary("mul", products, trips)
    query = CostQuery("dot", "matrix", dtype, operand_dtype, implementation="grid-word-mac")
    lanes = I.Expr(I.ScalarType("int64"), "lanes", data=(dtype, 0))
    instruction_cycles = I.binary(
      "mul", I.binary("div", products, lanes), I.literal(environment.costs.lookup(query).throughput_cycles)
    )
    cycles = I.binary("add", cycles, instruction_cycles)
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
