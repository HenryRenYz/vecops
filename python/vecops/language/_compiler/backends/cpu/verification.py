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
  verify_private_captures(prepared)
  confirmed = _memory_proofs(prepared)
  verify_partition_proofs(prepared, confirmed.partitions)
  verify_broadcast_proofs(prepared, confirmed.broadcasts)
  verify_extent_geometry(prepared, confirmed.geometry)
  return prepared


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


def verify_extent_geometry(prepared, confirmed=None):
  """Geometry may contain equality classes only, for the exact final subject."""
  from .broadcasts import ExtentGeometry

  if confirmed is None:
    confirmed = _memory_proofs(prepared).geometry
  for key, plan in getattr(prepared, "extent_geometry", {}).items():
    require(type(plan) is ExtentGeometry and key == id(plan.subject), "stale CPU extent geometry identity")
    actual = confirmed.get(key)
    require(actual is not None and actual.subject is plan.subject, "detached CPU extent geometry subject")
    require(
      len(plan.classes) == len(actual.classes)
      and all(len(a) == len(b) and all(x is y for x, y in zip(a, b)) for a, b in zip(plan.classes, actual.classes)),
      "CPU extent geometry is not proved in the final lexical domains",
    )
    require(
      len(plan.representatives) == len(actual.representatives)
      and all(a is b for a, b in zip(plan.representatives, actual.representatives)),
      "CPU extent representatives are not proved in the final lexical domains",
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
