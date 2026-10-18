# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Verify prepared CPU storage and capture plans without rewriting their IR."""

from ...ir import kernel as I
from ...ir.verification import require, verify_kernel, verify_vector


def verify_prepared(prepared):
  verify_kernel(prepared.vectors.kernel, terminal=True)
  seen = set()
  for region in (
    *prepared.vectors.regions,
    *prepared.stores.values(),
    *prepared.snapshots.values(),
    *prepared.axes.values(),
    *prepared.reductions.values(),
    *prepared.primitives.values(),
    *prepared.state_regions.values(),
  ):
    if id(region) not in seen:
      verify_vector(region)
      seen.add(id(region))
  storage = prepared.storage
  if storage is not None:
    slots = {}
    for interval in storage.intervals:
      slot = storage.slots[interval.storage]
      for previous in slots.get(slot, ()):
        require(
          previous.dtype == interval.dtype and previous.scope == interval.scope,
          "scratch slot crosses dtype/ownership scopes",
        )
        require(
          previous.last < interval.first or interval.last < previous.first,
          "scratch slot has overlapping live intervals",
        )
      slots.setdefault(slot, []).append(interval)
    verify_snapshot_scopes(prepared)
  verify_private_captures(prepared)
  entry = getattr(prepared, "entry_geometry", None)
  if entry is not None:
    from .invariants import EntryGeometry

    environment = prepared.vectors.manager.options["_compiler_environment"]
    require(type(entry) is EntryGeometry, "unknown entry geometry proof")
    require(entry.matches(prepared.vectors.kernel, environment.operations), "stale entry geometry proof")
  verify_closed_fixed_states(prepared)
  confirmed = _memory_proofs(prepared)
  verify_partition_proofs(prepared, confirmed.partitions)
  verify_broadcast_proofs(prepared, confirmed.broadcasts)
  verify_scalar_geometry(prepared, confirmed.geometry)
  return prepared


def verify_snapshot_scopes(prepared):
  """Recompute nonescaping read uses against final IR and exact subjects."""
  from ...operations import current
  from .storage import SnapshotScope, snapshot_lifetime_scopes

  if not prepared.storage.snapshot_scopes:
    return
  manager = getattr(prepared.vectors, "manager", None)
  environment = (getattr(manager, "options", {}) or {}).get("_compiler_environment")
  catalog = environment.operations if environment is not None else current()
  confirmed = snapshot_lifetime_scopes(prepared.vectors.kernel, catalog)
  for identity, proof in prepared.storage.snapshot_scopes.items():
    require(type(proof) is SnapshotScope and identity == id(proof.subject), "stale snapshot lifetime subject")
    actual = confirmed.get(identity)
    require(actual is not None and actual.subject is proof.subject, "snapshot read escapes its consumer")
    require(
      type(proof.occurrences) is int and proof.occurrences == actual.occurrences,
      "snapshot lifetime missed an occurrence",
    )
    require(
      len(proof.reads) == len(actual.reads) and all(a is b for a, b in zip(proof.reads, actual.reads)),
      "snapshot lifetime has detached historical reads",
    )


def verify_closed_fixed_states(prepared):
  """Recheck fixed-state placement against all original writes and programs."""
  from ... import ir as I
  from ...ir.traversal import nodes, walk_unique
  from .state import StatePiece, closed_fixed_lanes

  marked = []
  for name, plan in prepared.states.items():
    require(type(plan.closed_fixed) is bool, "invalid closed fixed-state marker")
    if plan.closed_fixed:
      require(plan.state.name == name, "detached closed fixed-state subject")
      marked.append(plan)
  if not marked:
    return
  manager = getattr(prepared.vectors, "manager", None)
  options = getattr(manager, "options", {}) or {}
  environment = options.get("_compiler_environment")
  require(environment is not None, "closed fixed-state proof lacks compiler environment")
  budget = options.get("cpu_state_register_budget", 12)
  require(type(budget) is int and 0 < budget <= 64, "closed fixed-state exceeds register budget")
  definitions, excluded = {}, set()
  for statement in nodes(prepared.vectors.kernel.body):
    if isinstance(statement, (I.DeclareState, I.UpdateState)):
      definitions.setdefault(statement.state.name, []).append(statement)
      if any(e.op == "dot" for e in walk_unique(statement.value)):
        excluded.add(statement.state.name)
    if isinstance(statement, I.ExitRegion) and statement.outlined:
      contained = tuple(nodes(statement.body))
      local = {s.state.name for s in contained if isinstance(s, I.DeclareState)}
      excluded.update(s.name for s in contained if isinstance(s, I.TileState) and s.name not in local)
    value = getattr(statement, "value", None)
    if isinstance(value, I.Expr):
      excluded.update(
        ref.data.name
        for expression in walk_unique(value)
        if expression.op == "dot"
        for accumulator in expression.args[2:]
        for ref in walk_unique(accumulator)
        if ref.op == "state_ref"
      )
  for plan in marked:
    require(plan.state.name not in excluded, "closed fixed-state crosses call or matrix storage")
    lanes = closed_fixed_lanes(
      plan.state,
      definitions.get(plan.state.name, ()),
      prepared.register_programs,
      environment,
    )
    require(lanes is not None, "closed fixed-state is not proved by every original write")
    require(plan.pieces == (StatePiece((0,), lanes),), "closed fixed-state has incompatible physical pieces")


def _memory_proofs(prepared):
  from ...operations import current, use_catalog
  from .memory_proofs import collect_memory_proofs

  manager = getattr(prepared.vectors, "manager", None)
  environment = (getattr(manager, "options", {}) or {}).get("_compiler_environment")
  with use_catalog(environment.operations if environment is not None else current()):
    return collect_memory_proofs(
      prepared.vectors.kernel,
      region_shapes={identity: region.shape for identity, region in getattr(prepared.vectors, "plans", {}).items()},
    )


def verify_broadcast_proofs(prepared, confirmed=None):
  """Recompute extent certificates against final IR and its operation snapshot."""
  from .broadcasts import BroadcastProof, broadcast_key

  if confirmed is None:
    confirmed = _memory_proofs(prepared).broadcasts
  for key, proof in prepared.broadcast_proofs.items():
    require(type(proof) is BroadcastProof, "invalid CPU broadcast certificate")
    require(key == broadcast_key(proof.store, proof.source, proof.destination), "stale CPU broadcast identity")
    actual = confirmed.get(key)
    require(actual is not None, "CPU broadcast certificate is not proved in the final lexical domains")
    require(
      actual.store is proof.store and actual.source is proof.source and actual.destination is proof.destination,
      "detached CPU broadcast subject",
    )
    require(proof.occurrences == actual.occurrences, "CPU broadcast proof missed a lexical occurrence")
    require(type(proof.equal) is bool and proof.equal == actual.equal, "invalid CPU extent equality")


def verify_scalar_geometry(prepared, confirmed=None):
  """Geometry may contain equality classes only, for the exact final subject."""
  from .broadcasts import ScalarGeometry

  if confirmed is None:
    confirmed = _memory_proofs(prepared).geometry
  for key, plan in getattr(prepared, "scalar_geometry", {}).items():
    require(type(plan) is ScalarGeometry and key == id(plan.subject), "stale CPU scalar geometry identity")
    actual = confirmed.get(key)
    require(actual is not None and actual.subject is plan.subject, "detached CPU scalar geometry subject")
    require(
      len(plan.classes) == len(actual.classes)
      and all(len(a) == len(b) and all(x is y for x, y in zip(a, b)) for a, b in zip(plan.classes, actual.classes)),
      "CPU scalar geometry is not proved in the final lexical domains",
    )
    require(
      len(plan.representatives) == len(actual.representatives)
      and all(a is b for a, b in zip(plan.representatives, actual.representatives)),
      "CPU scalar representatives are not proved in the final lexical domains",
    )


def verify_partition_proofs(prepared, confirmed=None):
  """Certificates must retain their final group/views and lexical count."""
  from .partitions import PartitionProof, partition_key

  if confirmed is None:
    confirmed = _memory_proofs(prepared).partitions
  for key, proof in prepared.partition_proofs.items():
    require(type(proof) is PartitionProof, "invalid CPU partition certificate")
    require(key == partition_key(proof.group, proof.first, proof.second), "stale CPU partition identity")
    actual = confirmed.get(key)
    require(actual is not None, "CPU partition certificate is not proved in the final lexical domains")
    require(
      actual.group is proof.group and actual.first is proof.first and actual.second is proof.second,
      "detached CPU partition subject",
    )
    require(proof.occurrences == actual.occurrences, "CPU partition proof missed a lexical occurrence")


def verify_private_captures(prepared):
  """Reject stale, mutable or out-of-scope positive value-capture proofs."""
  from ....types import TensorType
  from ...ir.collective import TeamReduce
  from ...ir.state import ExitRegion
  from ...ir.traversal import children, nodes
  from ...operations import Trait, current
  from .capabilities import CPP_TYPES

  kernel = prepared.vectors.kernel
  definitions, excluded = {}, set()
  for node in nodes(kernel.body):
    if isinstance(node, (I.Assign, I.For)):
      definitions.setdefault(node.target, []).append(node)
    if isinstance(node, (I.For, TeamReduce)):
      excluded.add(node.target)
    elif isinstance(node, I.Grid):
      excluded.update(node.targets)
  immutable = {
    name
    for name, sites in definitions.items()
    if len(sites) == 1
    and isinstance(sites[0], I.Assign)
    and sites[0].declare
    and isinstance(sites[0].value.type, I.ScalarType)
    and sites[0].value.type.dtype in CPP_TYPES
    and name not in excluded
  }
  parameters = {
    name
    for name, typ in kernel.parameters.items()
    if not isinstance(typ, TensorType) and name not in definitions and name not in excluded
  }
  tensors = {name: typ for name, typ in kernel.parameters.items() if isinstance(typ, TensorType)}
  metadata = {
    f"{name}.{operation}({axis})"
    for name, typ in tensors.items()
    for operation in ("size", "stride")
    for axis in range(len(typ.shape))
  }
  seen = set()
  manager = getattr(prepared.vectors, "manager", None)
  environment = (getattr(manager, "options", {}) or {}).get("_compiler_environment")
  catalog = environment.operations if environment is not None else current()

  def read_references(region):
    found, symbols = {}, set()

    def visit(value):
      for node in nodes(value):
        if not isinstance(node, I.Expr):
          continue
        if catalog.has(node, Trait.MEMORY_READ):
          found[id(node)] = node
        if node.op == "symbol" and node.data not in symbols:
          symbols.add(node.data)
          if node.data in kernel.bindings:
            visit(kernel.bindings[node.data])

    visit(region.body)
    return found

  def block(body, available, hardware=False):
    available = set(available)
    for statement in body.statements:
      if isinstance(statement, ExitRegion) and statement.outlined:
        plan = prepared.private_captures.get(id(statement))
        require(plan is not None and plan.region is statement, "private capture plan has a missing/stale subject")
        seen.add(id(statement))
        for names in (plan.scalars, plan.metadata, plan.pointers):
          require(len(set(names)) == len(names), "private capture plan contains duplicate bindings")
        require(set(plan.scalars) <= available, "private scalar capture is mutable or lacks lexical dominance")
        require(set(plan.metadata) <= metadata, "private metadata capture is not an ABI snapshot")
        require(set(plan.pointers) <= tensors.keys(), "private pointer capture is not an ABI tensor")
        reads = read_references(statement)
        require(
          len({id(read) for read in plan.snapshot_reads}) == len(plan.snapshot_reads),
          "private snapshot capture contains duplicate reads",
        )
        require(
          all(reads.get(id(read)) is read for read in plan.snapshot_reads),
          "private snapshot capture has a detached/non-read subject",
        )
        require(
          not hardware or not (plan.scalars or plan.metadata or plan.pointers or plan.snapshot_reads),
          "private value capture crosses an active hardware scope",
        )
      for child in children(statement):
        block(child, available, hardware or isinstance(statement, I.HardwareRegion))
      assignments = statement.assignments if isinstance(statement, I.ReductionGroup) else (statement,)
      for assignment in assignments:
        if isinstance(assignment, I.Assign) and assignment.declare and assignment.target in immutable:
          available.add(assignment.target)

  block(kernel.body, parameters)
  require(set(prepared.private_captures) == seen, "private capture plan contains a detached region")
  return prepared
