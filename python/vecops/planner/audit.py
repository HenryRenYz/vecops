# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Coverage reports for framework state and planned memory movements."""

from __future__ import annotations

from collections import Counter
from typing import Any

from vecops.graph import ExecutionGraph, TensorKind

from .ir import ActionKind, MemoryPlan, MemoryTier
from .validate import validate_plan


def _summary(rows: list[tuple[str, int, Any]], copy_storages: set[str]) -> dict[str, int]:
  accessed = [row for row in rows if row[2].weighted_accesses > 0]
  fast = [row for row in rows if row[2].tier == MemoryTier.fast]
  return {
    "storages": len(rows),
    "bytes": sum(row[1] for row in rows),
    "accessed_storages": len(accessed),
    "accessed_bytes": sum(row[1] for row in accessed),
    "fast_storages": len(fast),
    "fast_bytes": sum(row[1] for row in fast),
    "copy_to_fast_storages": sum(row[0] in copy_storages for row in rows),
    "copy_to_fast_bytes": sum(
      row[1] for row in rows if row[0] in copy_storages
    ),
  }


def audit_plan(graph: ExecutionGraph, plan: MemoryPlan) -> dict[str, Any]:
  """Summarize parameters, buffers, packed weights, and transfer actions."""
  validate_plan(graph, plan)
  copies = [action for action in plan.actions if action.kind == ActionKind.copy]
  copy_counts = Counter(action.storage_id for action in copies)
  copy_storages = {
    action.storage_id
    for action in copies
    if action.source_tier == MemoryTier.slow
    and action.target_tier == MemoryTier.fast
  }
  rows: dict[str, list[tuple[str, int, Any]]] = {
    "parameters": [],
    "buffers": [],
    "packed_buffers": [],
  }
  unnamed_fast_state: list[str] = []
  for storage_id, storage in graph.storages.items():
    tensors = [graph.tensors[tensor_id] for tensor_id in storage.tensors]
    kinds = {tensor.kind for tensor in tensors}
    names = {
      name
      for tensor in tensors
      for name in tensor.annotations.get("names", ())
    }
    placement = plan.placements[storage_id]
    row = (storage_id, storage.bytes, placement)
    if TensorKind.parameter in kinds:
      rows["parameters"].append(row)
    if TensorKind.buffer in kinds:
      rows["buffers"].append(row)
      if any("packed" in name.lower() for name in names):
        rows["packed_buffers"].append(row)
    if (
      kinds & {TensorKind.parameter, TensorKind.buffer}
      and placement.tier == MemoryTier.fast
      and placement.replicated
      and not names
    ):
      unnamed_fast_state.append(storage_id)

  action_counts = Counter(action.kind.value for action in plan.actions)
  low_reuse_copies = [
    action
    for action in copies
    if plan.placements[action.storage_id].weighted_accesses <= 1
  ]
  copy_backs = [
    action
    for action in copies
    if action.source_tier == MemoryTier.fast
    and action.target_tier == MemoryTier.slow
  ]
  return {
    "parameters": _summary(rows["parameters"], copy_storages),
    "buffers": _summary(rows["buffers"], copy_storages),
    "packed_buffers": _summary(rows["packed_buffers"], copy_storages),
    "actions": dict(sorted(action_counts.items())),
    "movement": {
      "copy_actions": len(copies),
      "copy_bytes": sum(action.bytes for action in copies),
      "unique_copy_storages": len(copy_counts),
      "duplicate_copy_actions": sum(
        max(0, count - 1) for count in copy_counts.values()
      ),
      "low_reuse_copy_actions": len(low_reuse_copies),
      "low_reuse_copy_bytes": sum(action.bytes for action in low_reuse_copies),
      "copy_back_actions": len(copy_backs),
      "copy_back_bytes": sum(action.bytes for action in copy_backs),
    },
    "unnamed_fast_state_storages": sorted(unnamed_fast_state),
  }


__all__ = ["audit_plan"]
