# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Late lexical storage placement and nonoverlapping scratch-slot assignment.

Intervals cover complete top-level regions, including read provenance and
metadata uses. This deliberately overestimates nested-loop/branch lifetimes.
Worker allocations, implicit snapshots, workspace authority and state storage
are separate owners. Reuse never changes logical View/storage identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType

from ... import ir as I
from ...analysis.effects import ValueEffects, assigned_scalars, scalar_references
from ...ir.traversal import nodes
from ...passes.manager import register_pass


def branch_buffer_scopes(block):
  """Find common branch scopes without shortening loop capacity reuse."""
  references, allocations, seen = {}, set(), set()

  def visit(value, path=(), looping=False):
    if isinstance(value, dict):
      for key, item in value.items():
        visit(key, path, looping)
        visit(item, path, looping)
      return
    if isinstance(value, (tuple, list, set, frozenset)):
      for item in value:
        visit(item, path, looping)
      return
    if not is_dataclass(value) or isinstance(value, type):
      return
    key = id(value), path, looping
    if key in seen:
      return
    seen.add(key)
    if isinstance(value, I.View):
      references.setdefault(value.storage, set()).add(path)
    if isinstance(value, I.Allocate):
      allocations.add(value.view.storage)
    if isinstance(value, I.If):
      visit(value.condition, path, looping)
      for child in (value.body, value.otherwise):
        visit(child, path if looping else (*path, id(child)), looping)
      return
    if isinstance(value, (I.Grid, I.Team)):
      for field in fields(value):
        item = getattr(value, field.name)
        visit(item, () if field.name == "body" else path, True if field.name == "body" else looping)
      return
    looping = looping or isinstance(value, (I.For, I.While, I.UnrolledFor))
    for field in fields(value):
      visit(getattr(value, field.name), path, looping)

  visit(block)
  result = {}
  for name in allocations:
    paths = iter(references.get(name, ()))
    common = next(paths, ())
    for path in paths:
      common = common[: next((i for i, (a, b) in enumerate(zip(common, path)) if a != b), min(len(common), len(path)))]
    if common:
      result[name] = common[-1]
  return result


@dataclass(frozen=True)
class ScratchInterval:
  storage: str
  dtype: str
  first: int
  last: int
  scope: int | None
  capacity: int | None
  shape: tuple


@dataclass(frozen=True)
class StoragePlan:
  scopes: object
  slots: object
  intervals: tuple[ScratchInterval, ...]


@register_pass("cpu-plan-storage", stage="cpu-memory", after="cpu-plan-hardware")
def plan_storage(prepared, context):
  local = context.options.get("cpu_branch_local_buffers", True)
  reuse = context.options.get("cpu_scratch_reuse", True)
  if type(local) is not bool:
    raise TypeError("cpu_branch_local_buffers must be bool")
  if type(reuse) is not bool:
    raise TypeError("cpu_scratch_reuse must be bool")
  kernel = prepared.vectors.kernel
  scopes = branch_buffer_scopes(kernel.body) if local else {}
  intervals, allocations, references, forbidden = [], {}, {}, set()
  effects = ValueEffects(context.environment.operations)
  assigned = assigned_scalars(kernel.body)
  for index, statement in enumerate(kernel.body.statements):
    # Worker callbacks have their own declaration/parallel ownership. Do not
    # assign their storage to invocation-level slots even on a serial fallback.
    for region in nodes(statement):
      if isinstance(region, (I.Grid, I.Team, I.HardwareRegion)):
        forbidden.update(node.view.storage for node in nodes(region) if isinstance(node, I.Allocate))
    for node in nodes(statement):
      if isinstance(node, I.Allocate):
        allocations.setdefault(node.view.storage, node.view)
      elif isinstance(node, I.View):
        first, _ = references.get(node.storage, (index, index))
        references[node.storage] = first, index
  for name, view in allocations.items():
    if not view.owned or name in forbidden or name not in references:
      continue
    dimensions = tuple(I.constant(extent) for extent in view.shape)
    capacity = math.prod(dimensions) if all(type(v) is int and v >= 0 for v in dimensions) else None
    if capacity is None and any(
      not effects.speculatable(extent) or scalar_references(extent) & assigned for extent in view.shape
    ):
      continue
    first, last = references[name]
    intervals.append(ScratchInterval(name, view.dtype, first, last, scopes.get(name), capacity, view.shape))
  intervals.sort(key=lambda value: (value.first, value.storage))
  slots, available = {}, []
  for interval in intervals:
    owner = None
    if reuse:
      for index, (previous, slot) in enumerate(available):
        compatible = (
          previous.last < interval.first
          and previous.dtype == interval.dtype
          and previous.scope == interval.scope
          and (
            previous.capacity is not None
            and interval.capacity is not None
            and previous.capacity >= interval.capacity
            or previous.capacity is None
            and interval.capacity is None
            and len(previous.shape) == len(interval.shape)
            and all(I.same_expr(a, b) for a, b in zip(previous.shape, interval.shape))
          )
        )
        if compatible:
          owner = slot
          # Preserve the slot's actual capacity after a smaller reuse; the
          # Buffer implementation retains capacity on shrinking resize().
          previous = replace(interval, capacity=previous.capacity)
          available[index] = previous, slot
          break
    if owner is None:
      owner = interval.storage
      available.append((interval, owner))
    slots[interval.storage] = owner
  if reuse and len(set(slots.values())) < len(slots):
    kernel.diagnostics.append(
      f"scratch lifetime reuse: {len(slots)} logical allocations in {len(set(slots.values()))} slots"
    )
  plan = StoragePlan(MappingProxyType(scopes), MappingProxyType(slots), tuple(intervals))
  return replace(prepared, storage=plan)
