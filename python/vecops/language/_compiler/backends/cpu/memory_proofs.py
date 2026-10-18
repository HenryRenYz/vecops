# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Plan related memory proofs in one shared final lexical traversal."""

from dataclasses import dataclass, replace

from ...passes.manager import register_pass
from .alignment import make_alignment_collector
from .broadcasts import make_broadcast_collector
from .partitions import make_partition_collector, walk_lexical_scopes


@dataclass(frozen=True)
class MemoryProofs:
  partitions: object
  broadcasts: object
  geometry: object
  alignment: object


def collect_memory_proofs(kernel, *, region_shapes=None, align_stores=False):
  partition, finish_partition = make_partition_collector()
  broadcast, finish_broadcast = make_broadcast_collector(region_shapes)
  alignment, finish_alignment = make_alignment_collector(kernel) if align_stores else (None, dict)

  def observe(statement, scope):
    partition(statement, scope)
    broadcast(statement, scope)
    if alignment is not None:
      alignment(statement, scope)

  walk_lexical_scopes(kernel, observe)
  extents = finish_broadcast()
  return MemoryProofs(finish_partition(), extents.proofs, extents.geometry, finish_alignment())


@register_pass("cpu-plan-memory-proofs", stage="cpu-memory", after=("cpu-plan-storage", "cpu-plan-register-programs"))
def plan_memory_proofs(prepared, context):
  proofs = collect_memory_proofs(
    prepared.vectors.kernel,
    region_shapes={identity: region.shape for identity, region in prepared.vectors.plans.items()},
    align_stores=context.environment.codegen_target.architecture == "x86",
  )
  return replace(
    prepared, partition_proofs=proofs.partitions, broadcast_proofs=proofs.broadcasts, scalar_geometry=proofs.geometry,
    store_alignment_proofs=proofs.alignment
  )
