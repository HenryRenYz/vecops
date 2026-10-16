# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Plan immutable value captures without changing private-call boundaries.

This is a positive whitelist, not a closed helper ABI. Unknown bindings,
writeback slots, induction variables, state and storage owners retain the
emitter's reference capture. Only existing scalar values and ABI pointer /
metadata snapshots are copied; no expression is evaluated earlier.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

from ....types import TensorType
from ... import ir as I
from ...ir.traversal import children, nodes
from ...operations import Trait
from ...passes.manager import register_pass
from .capabilities import CPP_TYPES


@dataclass(frozen=True)
class PrivateCapturePlan:
  # Retain the subject so an identity-indexed plan cannot outlive its region.
  region: I.ExitRegion
  scalars: tuple[str, ...]
  metadata: tuple[str, ...]
  pointers: tuple[str, ...]
  snapshot_reads: tuple[I.Expr, ...] = ()


@register_pass(
  "cpu-plan-private-captures", stage="cpu-memory",
  after=("cpu-plan-storage", "cpu-plan-grids", "cpu-plan-pointwise-register-layout"),
)
def plan_private_captures(prepared, context):
  kernel = prepared.vectors.kernel
  catalog = context.environment.operations
  definitions, excluded = {}, set()
  for node in nodes(kernel.body):
    if isinstance(node, (I.Assign, I.For)):
      definitions.setdefault(node.target, []).append(node)
    if isinstance(node, (I.For, I.TeamReduce)):
      excluded.add(node.target)
    elif isinstance(node, I.Grid):
      excluded.update(node.targets)
  # Keep precisely the previous architectural-owner reconstruction proof:
  # one declaring Assign and no other Assign/For definition of that name.
  # Capture copyability/dominance is a separate, stricter decision below.
  geometry = {
    name: sites[0].value for name, sites in definitions.items()
    if len(sites) == 1 and isinstance(sites[0], I.Assign) and sites[0].declare
  }
  immutable = {
    name for name, value in geometry.items()
    if name not in excluded and isinstance(value.type, I.ScalarType) and value.type.dtype in CPP_TYPES
  }
  parameters = {name for name, typ in kernel.parameters.items() if not isinstance(typ, TensorType)}
  parameters.difference_update(definitions)
  parameters.difference_update(excluded)
  tensors = {name: typ for name, typ in kernel.parameters.items() if isinstance(typ, TensorType)}
  entry_metadata = {
    f"{name}.{operation}({axis})"
    for name, typ in tensors.items() for operation in ("size", "stride") for axis in range(len(typ.shape))
  }
  plans = {}

  def references(region):
    scalar, metadata, pointers, symbols = set(), set(), set(), set()
    snapshot_reads = {}

    def visit(value):
      for node in nodes(value):
        if isinstance(node, I.View) and node.storage in tensors:
          pointers.add(node.storage)
          # Alias protection compares the complete ABI root, whose physical
          # metadata need not occur in this logical subview's shape/strides.
          metadata.update(
            f"{node.storage}.{operation}({axis})"
            for operation in ("size", "stride") for axis in range(len(tensors[node.storage].shape))
          )
        if not isinstance(node, I.Expr):
          continue
        if catalog.has(node, Trait.MEMORY_READ):
          snapshot_reads.setdefault(id(node), node)
        if node.op == "symbol" and node.data not in symbols:
          symbols.add(node.data)
          if node.data in kernel.bindings:
            visit(kernel.bindings[node.data])
        elif node.op in {"local", "frozen", "raw"} and isinstance(node.data, str):
          if node.data in immutable or node.data in parameters:
            scalar.add(node.data)
          if node.op == "raw" and node.data in entry_metadata:
            metadata.add(node.data)

    visit(region.body)
    return scalar, metadata, pointers, tuple(snapshot_reads.values())

  def block(body, available, hardware=False):
    available = set(available)
    for statement in body.statements:
      if isinstance(statement, I.ExitRegion) and statement.outlined:
        scalar, metadata, pointers, snapshot_reads = references(statement)
        if hardware:
          scalar, metadata, pointers = set(), set(), set()
          snapshot_reads = ()
        plans[id(statement)] = PrivateCapturePlan(
          statement, tuple(sorted(scalar & available)), tuple(sorted(metadata)), tuple(sorted(pointers)),
          snapshot_reads,
        )
      for child in children(statement):
        block(child, available, hardware or isinstance(statement, I.HardwareRegion))
      assignments = statement.assignments if isinstance(statement, I.ReductionGroup) else (statement,)
      for assignment in assignments:
        if isinstance(assignment, I.Assign) and assignment.declare and assignment.target in immutable:
          available.add(assignment.target)

  block(kernel.body, parameters)
  return replace(prepared, private_captures=MappingProxyType(plans),
                 immutable_geometry=MappingProxyType(geometry))
