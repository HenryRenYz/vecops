# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Deterministic simulator and structural validation for memory plans."""

from __future__ import annotations

import bisect
import heapq
from collections import defaultdict

from vecops.graph import ExecutionGraph, TensorKind

from .ir import ActionKind, MemoryPlan, MemoryTier, PLAN_FORMAT_VERSION


def validate_plan(graph: ExecutionGraph, plan: MemoryPlan) -> None:
  """Raise ``ValueError`` if a plan is stale, incomplete, or unsafe."""
  from .core import graph_digest

  graph.validate()
  plan.machine.validate()
  if plan.format_version != PLAN_FORMAT_VERSION:
    raise ValueError(f"unsupported memory plan format {plan.format_version}")
  if plan.graph_digest != graph_digest(graph):
    raise ValueError("memory plan was produced for a different execution graph")
  if set(plan.placements) != set(graph.storages):
    missing = set(graph.storages) - set(plan.placements)
    extra = set(plan.placements) - set(graph.storages)
    raise ValueError(f"plan storage mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
  if plan.fast_arena_bytes > plan.machine.fast_capacity:
    raise ValueError("fast arena exceeds machine capacity")
  if plan.slow_arena_bytes > plan.machine.slow_capacity:
    raise ValueError("slow arena exceeds machine capacity")
  graph_end = graph.events[-1].index if graph.events else 0
  for loop in graph.loops:
    if isinstance(loop.trip_count, int) and loop.trip_count > 1:
      if len(loop.samples) < 2:
        raise ValueError(f"loop {loop.id} has no steady-state sample")
      for sample in loop.samples:
        if len(sample.input_tensors) != len(sample.output_tensors):
          raise ValueError(
            f"loop {loop.id} iteration {sample.iteration} changes carried-state arity"
          )

  by_tier: dict[MemoryTier, list] = defaultdict(list)
  for storage_id, placement in plan.placements.items():
    storage = graph.storages[storage_id]
    if placement.bytes != storage.bytes:
      raise ValueError(f"storage {storage_id} byte size changed in plan")
    if not 0 <= placement.begin_event <= placement.end_event:
      raise ValueError(f"storage {storage_id} has invalid planned lifetime")
    if placement.offset is not None:
      if placement.offset % placement.alignment:
        raise ValueError(f"storage {storage_id} has a misaligned offset")
      by_tier[placement.tier].append(placement)
    if placement.replicated and not placement.external_home:
      raise ValueError(f"storage {storage_id} replica has no slow home")
    if placement.tier == MemoryTier.fast and placement.first_access is not None:
      if not placement.begin_event <= placement.first_access <= placement.end_event:
        raise ValueError(f"storage {storage_id} is absent at its first access")
      if not placement.begin_event <= placement.last_access <= placement.end_event:
        raise ValueError(f"storage {storage_id} is absent at its last access")
    if any(
      graph.tensors[tensor_id].kind == TensorKind.output
      for tensor_id in storage.tensors
    ) and placement.tier != MemoryTier.slow:
      raise ValueError(f"graph output storage {storage_id} must escape through slow memory")

  for tier, placements in by_tier.items():
    active_by_end: list[tuple[int, str, int]] = []
    active_by_offset: list[tuple[int, str, int]] = []
    for placement in sorted(
      placements,
      key=lambda item: (item.begin_event, item.offset, item.storage_id),
    ):
      while active_by_end and active_by_end[0][0] < placement.begin_event:
        _end, storage_id, offset = heapq.heappop(active_by_end)
        position = bisect.bisect_left(active_by_offset, (offset, storage_id, -1))
        if position == len(active_by_offset) or active_by_offset[position][1] != storage_id:
          raise ValueError(f"lost active slot for storage {storage_id}")
        del active_by_offset[position]
      entry = (placement.offset, placement.storage_id, placement.bytes)
      position = bisect.bisect_left(active_by_offset, entry)
      if position:
        left_offset, left_id, left_bytes = active_by_offset[position - 1]
        if left_offset + left_bytes > placement.offset:
          raise ValueError(
            f"overlapping {tier.value} slots for {left_id} and {placement.storage_id}"
          )
      if position < len(active_by_offset):
        right_offset, right_id, _right_bytes = active_by_offset[position]
        if placement.offset + placement.bytes > right_offset:
          raise ValueError(
            f"overlapping {tier.value} slots for {placement.storage_id} and {right_id}"
          )
      active_by_offset.insert(position, entry)
      heapq.heappush(
        active_by_end,
        (placement.end_event, placement.storage_id, placement.offset),
      )

  action_groups: dict[str, list] = defaultdict(list)
  for action in plan.actions:
    if action.storage_id not in plan.placements:
      raise ValueError(f"action {action.id} references unknown storage")
    if action.phase not in {"before", "after"}:
      raise ValueError(f"action {action.id} has invalid phase {action.phase!r}")
    if not 0 <= action.event <= graph_end:
      raise ValueError(f"action {action.id} is outside the graph event range")
    if action.bytes != plan.placements[action.storage_id].bytes:
      raise ValueError(f"action {action.id} byte size disagrees with placement")
    action_groups[action.storage_id].append(action)

  for storage_id, placement in plan.placements.items():
    actions = action_groups.get(storage_id, [])
    managed = placement.tier == MemoryTier.fast or not placement.external_home
    allocations = [item for item in actions if item.kind == ActionKind.allocate]
    frees = [item for item in actions if item.kind == ActionKind.free]
    copies = [item for item in actions if item.kind == ActionKind.copy]
    if managed and (len(allocations) != 1 or len(frees) != 1):
      raise ValueError(f"storage {storage_id} does not have one allocate/free pair")
    if not managed and (allocations or frees):
      raise ValueError(f"external slow storage {storage_id} must not be allocated")
    if placement.replicated:
      if len(copies) != 1:
        raise ValueError(f"replicated storage {storage_id} does not have one copy")
      copy = copies[0]
      if (
        copy.source_tier != MemoryTier.slow
        or copy.target_tier != MemoryTier.fast
        or copy.event > (placement.first_access or placement.begin_event)
      ):
        raise ValueError(f"storage {storage_id} has an invalid prefetch action")
    elif copies:
      raise ValueError(f"non-replicated storage {storage_id} has a copy action")


def plan_summary(plan: MemoryPlan) -> dict[str, int | float | str | bool]:
  """Return stable headline metrics for logs and regression comparisons."""
  return {
    "graph_name": plan.graph_name,
    "machine": plan.machine.name,
    "fast_capacity_bytes": plan.machine.fast_capacity,
    "slow_capacity_bytes": plan.machine.slow_capacity,
    **plan.metrics,
  }


__all__ = ["plan_summary", "validate_plan"]
