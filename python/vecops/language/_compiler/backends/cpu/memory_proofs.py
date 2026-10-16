# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Plan related memory proofs in one shared final lexical traversal."""

from dataclasses import dataclass, replace

from ...passes.manager import register_pass
from .broadcasts import make_broadcast_collector
from .partitions import make_partition_collector, walk_lexical_scopes


@dataclass(frozen=True)
class MemoryProofs:
  partitions: object
  broadcasts: object
  geometry: object


def collect_memory_proofs(kernel, *, region_shapes=None):
  partition, finish_partition = make_partition_collector()
  broadcast, finish_broadcast = make_broadcast_collector(region_shapes)

  def observe(statement, scope):
    partition(statement, scope)
    broadcast(statement, scope)

  walk_lexical_scopes(kernel, observe)
  extents = finish_broadcast()
  return MemoryProofs(finish_partition(), extents.proofs, extents.geometry)


@register_pass("cpu-plan-memory-proofs", stage="cpu-memory", after=("cpu-plan-storage", "cpu-plan-register-programs"))
def plan_memory_proofs(prepared, context):
  proofs = collect_memory_proofs(
    prepared.vectors.kernel,
    region_shapes={identity: region.shape for identity, region in prepared.vectors.plans.items()},
  )
  return replace(
    prepared, partition_proofs=proofs.partitions, broadcast_proofs=proofs.broadcasts, scalar_geometry=proofs.geometry
  )
