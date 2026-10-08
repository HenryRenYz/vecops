# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Capacity-aware offline planner for :mod:`vecops.graph` execution graphs."""

from __future__ import annotations

import hashlib
import heapq
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from vecops.graph import AccessKind, ExecutionGraph, TensorKind

from .ir import (
  ActionKind,
  MachineProfile,
  MemoryPlan,
  MemoryTier,
  PlanAction,
  StorageClass,
  StoragePlacement,
)


@dataclass(frozen=True)
class PlannerConfig:
  """Stable policy switches for one planner invocation."""

  minimum_fast_bytes: int = 4096
  prefer_all_fast_when_fit: bool = False
  force_outputs_slow: bool = True
  allow_mutable_replication: bool = False
  score_by_residency: bool = True
  eager_external_replicas: bool = True
  allow_temporary_fast: bool = True
  allow_input_fast: bool = True
  allow_persistent_fast: bool = True
  require_replayable_temporaries: bool = False
  minimum_weighted_accesses: float = 1.0


@dataclass
class _Profile:
  storage_id: str
  bytes: int
  storage_class: StorageClass
  begin: int
  end: int
  accesses: list[int]
  weighted_accesses: float
  weighted_access_bytes: float
  read_only: bool
  external_home: bool
  graph_output: bool
  loop_replay_safe: bool
  benefit_ns: float = 0.0

  @property
  def first_access(self) -> int | None:
    return min(self.accesses) if self.accesses else None

  @property
  def last_access(self) -> int | None:
    return max(self.accesses) if self.accesses else None

  @property
  def fast_begin(self) -> int:
    return self.first_access if self.external_home and self.first_access is not None else self.begin

  @property
  def fast_end(self) -> int:
    return self.last_access if self.external_home and self.last_access is not None else self.end


class _RangeAddMax:
  def __init__(self, size: int) -> None:
    leaf_count = 1
    while leaf_count < max(size, 1):
      leaf_count *= 2
    self._leaf_count = leaf_count
    self._maximum = [0] * (2 * leaf_count)
    self._lazy = [0] * (2 * leaf_count)

  @property
  def maximum(self) -> int:
    return self._maximum[1]

  def add(self, left: int, right: int, value: int) -> None:
    if right < left:
      return
    self._add(1, 0, self._leaf_count - 1, left, right, value)

  def _add(
    self,
    node: int,
    begin: int,
    end: int,
    left: int,
    right: int,
    value: int,
  ) -> None:
    if left <= begin and end <= right:
      self._maximum[node] += value
      self._lazy[node] += value
      return
    middle = (begin + end) // 2
    if left <= middle:
      self._add(node * 2, begin, middle, left, right, value)
    if middle < right:
      self._add(node * 2 + 1, middle + 1, end, left, right, value)
    self._maximum[node] = self._lazy[node] + max(
      self._maximum[node * 2], self._maximum[node * 2 + 1]
    )


def _align_up(value: int, alignment: int) -> int:
  return (value + alignment - 1) & -alignment


def graph_digest(graph: ExecutionGraph) -> str:
  """Return the exact content digest used to reject stale plans."""
  return hashlib.sha256(graph.dumps(indent=None).encode("utf-8")).hexdigest()


def _event_weights(graph: ExecutionGraph) -> dict[int, float]:
  weights = {event.index: 1.0 for event in graph.events}
  event_index = {event.id: event.index for event in graph.events}
  for loop in graph.loops:
    if not isinstance(loop.trip_count, int) or loop.trip_count <= 0 or not loop.samples:
      continue
    remaining = max(loop.trip_count - 1, 0)
    for sample_index, sample in enumerate(loop.samples):
      begin = event_index[sample.begin_event]
      end = event_index[sample.end_event] if sample.end_event is not None else begin
      if sample.role == "prologue" or sample_index == 0:
        multiplier = 1
      elif sample.role == "epilogue":
        multiplier = 1
      else:
        multiplier = max(remaining, 1)
      for index in range(begin, end + 1):
        weights[index] *= multiplier
  return weights


def _storage_class(graph: ExecutionGraph, storage_id: str) -> StorageClass:
  storage = graph.storages[storage_id]
  tensors = [graph.tensors[item] for item in storage.tensors]
  if storage.persistent:
    return StorageClass.persistent
  if any(tensor.kind == TensorKind.input for tensor in tensors):
    return StorageClass.input
  if any(tensor.kind == TensorKind.output for tensor in tensors):
    return StorageClass.output
  if storage.allocation_event is not None:
    return StorageClass.temporary
  return StorageClass.external


def _profiles(graph: ExecutionGraph, machine: MachineProfile) -> list[_Profile]:
  weights = _event_weights(graph)
  event_indexes = {event.id: event.index for event in graph.events}
  steady_ranges = [
    (event_indexes[sample.begin_event], event_indexes[sample.end_event])
    for loop in graph.loops
    for sample in loop.samples
    if sample.role == "steady_sample" and sample.end_event is not None
  ]
  accesses: dict[str, list[int]] = defaultdict(list)
  access_bytes: dict[str, float] = defaultdict(float)
  writes: set[str] = set()
  for event in graph.events:
    input_storages = {graph.tensors[use.tensor_id].storage_id for use in event.uses}
    output_storages = {
      graph.tensors[tensor_id].storage_id for tensor_id in event.outputs
    }
    metadata_only = (
      not any(use.access.writes for use in event.uses)
      and bool(output_storages)
      and output_storages <= input_storages
      and not any(use.access == AccessKind.allocate for use in event.uses)
    ) or event.name.startswith("prim.") or event.name.startswith(
      ("aten.size", "aten.sym_size", "aten.stride")
    ) or event.name.startswith(
      ("aten.empty.", "aten.empty_like.")
    )
    event_storage_bytes: dict[str, int] = defaultdict(int)
    for use in event.uses:
      storage_id = graph.tensors[use.tensor_id].storage_id
      if not metadata_only and use.access != AccessKind.alias:
        logical_bytes = graph.tensors[use.tensor_id].metadata.logical_bytes
        touched = (
          logical_bytes
          if isinstance(logical_bytes, int)
          else graph.storages[storage_id].bytes
        )
        if isinstance(touched, int):
          event_storage_bytes[storage_id] = max(event_storage_bytes[storage_id], touched)
      if use.access in {AccessKind.write, AccessKind.read_write}:
        writes.add(storage_id)
    for storage_id, touched in event_storage_bytes.items():
      accesses[storage_id].append(event.index)
      access_bytes[storage_id] += touched * weights[event.index]

  output_storages = {
    graph.tensors[tensor_id].storage_id for tensor_id in graph.outputs
  }
  result = []
  for storage_id, storage in graph.storages.items():
    if not isinstance(storage.bytes, int):
      raise ValueError(
        f"storage {storage_id} has symbolic/unknown bytes {storage.bytes!r}; "
        "plan a concrete precompile signature"
      )
    storage_accesses = accesses.get(storage_id, [])
    weighted = sum(weights[index] for index in storage_accesses)
    read_only = storage_id not in writes
    allocation_index = event_indexes.get(storage.allocation_event, 0)
    loop_replay_safe = all(
      storage.lifetime_end <= end
      for begin, end in steady_ranges
      if begin <= allocation_index <= end
    )
    result.append(
      _Profile(
        storage_id=storage_id,
        bytes=storage.bytes,
        storage_class=_storage_class(graph, storage_id),
        begin=storage.lifetime_start,
        end=storage.lifetime_end,
        accesses=storage_accesses,
        weighted_accesses=weighted,
        weighted_access_bytes=access_bytes.get(storage_id, 0.0),
        read_only=read_only,
        external_home=storage.external or storage.persistent,
        graph_output=storage_id in output_storages,
        loop_replay_safe=loop_replay_safe,
      )
    )

  fast_ns_per_byte = 1e9 / (machine.fast_bandwidth_mib_s * (1 << 20))
  slow_ns_per_byte = 1e9 / (machine.slow_bandwidth_mib_s * (1 << 20))
  copy_ns_per_byte = 1e9 / (machine.transfer_bandwidth_mib_s * (1 << 20))
  for profile in result:
    saved = profile.weighted_access_bytes * (
      slow_ns_per_byte - fast_ns_per_byte
    )
    copy_count = 1 if profile.external_home else 0
    if profile.external_home and not profile.read_only:
      copy_count += 1
    profile.benefit_ns = saved - copy_count * (
      machine.transfer_latency_ns + profile.bytes * copy_ns_per_byte
    )
  return result


def _eligible(profile: _Profile, config: PlannerConfig) -> bool:
  if (
    profile.bytes < config.minimum_fast_bytes
    or profile.weighted_accesses < config.minimum_weighted_accesses
    or not profile.accesses
  ):
    return False
  if config.force_outputs_slow and profile.graph_output:
    return False
  if profile.storage_class == StorageClass.external:
    return False
  if profile.storage_class == StorageClass.temporary and not config.allow_temporary_fast:
    return False
  if (
    profile.storage_class == StorageClass.temporary
    and config.require_replayable_temporaries
    and not profile.loop_replay_safe
  ):
    return False
  if profile.storage_class == StorageClass.input and not config.allow_input_fast:
    return False
  if profile.storage_class == StorageClass.persistent and not config.allow_persistent_fast:
    return False
  if profile.external_home and not profile.read_only and not config.allow_mutable_replication:
    return False
  return True


def _select_fast(
  profiles: list[_Profile],
  graph_end: int,
  capacity: int,
  alignment: int,
  config: PlannerConfig,
) -> tuple[set[str], int]:
  def interval(profile: _Profile) -> tuple[int, int]:
    if config.eager_external_replicas and profile.external_home:
      return 0, graph_end
    return profile.fast_begin, profile.fast_end

  eligible = [profile for profile in profiles if _eligible(profile, config)]
  all_live = _RangeAddMax(graph_end + 2)
  for profile in eligible:
    begin, end = interval(profile)
    all_live.add(begin, end, profile.bytes)
  if config.prefer_all_fast_when_fit and all_live.maximum <= capacity:
    return {profile.storage_id for profile in eligible}, all_live.maximum

  def score(profile: _Profile) -> tuple[float, float, str]:
    begin, end = interval(profile)
    duration = end - begin + 1
    denominator = profile.bytes * (duration if config.score_by_residency else 1)
    return (
      profile.benefit_ns / max(denominator, 1),
      profile.benefit_ns,
      profile.storage_id,
    )

  ordered = [
    profile
    for profile in sorted(eligible, key=score, reverse=True)
    if profile.benefit_ns > 0
  ]

  def extend(seed: set[str], limit: int) -> tuple[set[str], int]:
    selected = set(seed)
    usage = _RangeAddMax(graph_end + 2)
    for profile in eligible:
      if profile.storage_id in selected:
        begin, end = interval(profile)
        usage.add(begin, end, profile.bytes)
    for profile in ordered:
      if profile.storage_id in selected:
        continue
      begin, end = interval(profile)
      usage.add(begin, end, profile.bytes)
      if usage.maximum <= limit:
        selected.add(profile.storage_id)
      else:
        usage.add(begin, end, -profile.bytes)
    return selected, usage.maximum

  seeds: list[tuple[set[str], int]] = []
  for numerator, denominator in ((1, 4), (1, 2), (2, 3), (3, 4), (7, 8), (1, 1)):
    seeds.append(extend(set(), capacity * numerator // denominator))

  profile_by_id = {profile.storage_id: profile for profile in profiles}

  def packed_size(selected: set[str]) -> int:
    placements = []
    for storage_id in selected:
      profile = profile_by_id[storage_id]
      begin, end = interval(profile)
      placements.append(
        StoragePlacement(
          storage_id=storage_id,
          storage_class=profile.storage_class,
          bytes=profile.bytes,
          tier=MemoryTier.fast,
          begin_event=begin,
          end_event=end,
          first_access=profile.first_access,
          last_access=profile.last_access,
        )
      )
    return _pack_for_capacity(placements, alignment, capacity)[1]

  def objective(item: tuple[set[str], int]) -> tuple[float, float, int]:
    selected, peak = item
    benefit = sum(
      profile.benefit_ns
      for profile in eligible
      if profile.storage_id in selected
    )
    traffic = sum(
      profile.weighted_access_bytes
      for profile in eligible
      if profile.storage_id in selected
    )
    return benefit, traffic, -peak

  for candidate in sorted(seeds, key=objective, reverse=True):
    if packed_size(candidate[0]) <= capacity:
      return candidate
  return set(), 0


def _selected_peak(
  profiles: Iterable[_Profile],
  selected: set[str],
  graph_end: int,
  config: PlannerConfig,
) -> int:
  usage = _RangeAddMax(graph_end + 2)
  for profile in profiles:
    if profile.storage_id in selected:
      if config.eager_external_replicas and profile.external_home:
        begin, end = 0, graph_end
      else:
        begin, end = profile.fast_begin, profile.fast_end
      usage.add(begin, end, profile.bytes)
  return usage.maximum


def _profile_peak(
  profiles: Iterable[_Profile], graph_end: int, *, managed_only: bool = False
) -> int:
  usage = _RangeAddMax(graph_end + 2)
  for profile in profiles:
    if managed_only and profile.external_home:
      continue
    usage.add(profile.begin, profile.end, profile.bytes)
  return usage.maximum


def _coalesce(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
  result: list[tuple[int, int]] = []
  for offset, size in sorted(blocks):
    if result and result[-1][0] + result[-1][1] == offset:
      result[-1] = (result[-1][0], result[-1][1] + size)
    else:
      result.append((offset, size))
  return result


def _pack(
  placements: Iterable[StoragePlacement], alignment: int
) -> tuple[dict[str, int], int]:
  ordered = sorted(
    placements,
    key=lambda item: (item.begin_event, -item.bytes, item.storage_id),
  )
  active: list[tuple[int, str, int, int]] = []
  free: list[tuple[int, int]] = []
  offsets: dict[str, int] = {}
  high_water = 0
  for placement in ordered:
    while active and active[0][0] < placement.begin_event:
      _end, _storage_id, offset, size = heapq.heappop(active)
      free.append((offset, size))
    free = _coalesce(free)
    required = _align_up(placement.bytes, alignment)
    chosen = None
    for index, (offset, size) in enumerate(free):
      aligned = _align_up(offset, alignment)
      padding = aligned - offset
      if padding + required <= size:
        chosen = (index, offset, size, aligned, padding)
        break
    if chosen is None:
      offset = _align_up(high_water, alignment)
      high_water = offset + required
    else:
      index, block_offset, block_size, offset, padding = chosen
      del free[index]
      if padding:
        free.append((block_offset, padding))
      tail = block_size - padding - required
      if tail:
        free.append((offset + required, tail))
    offsets[placement.storage_id] = offset
    heapq.heappush(
      active,
      (placement.end_event, placement.storage_id, offset, required),
    )
  return offsets, high_water


def _pack_compact(
  placements: Iterable[StoragePlacement], alignment: int
) -> tuple[dict[str, int], int]:
  """Global decreasing-size first fit for the capacity-constrained tier."""
  placed: list[tuple[StoragePlacement, int, int]] = []
  offsets: dict[str, int] = {}
  high_water = 0
  for placement in sorted(
    placements,
    key=lambda item: (-item.bytes, item.begin_event, item.storage_id),
  ):
    required = _align_up(placement.bytes, alignment)
    cursor = 0
    occupied = sorted(
      (offset, size)
      for existing, offset, size in placed
      if placement.begin_event <= existing.end_event
      and existing.begin_event <= placement.end_event
    )
    for offset, size in occupied:
      cursor = _align_up(cursor, alignment)
      if cursor + required <= offset:
        break
      if cursor < offset + size:
        cursor = offset + size
    cursor = _align_up(cursor, alignment)
    offsets[placement.storage_id] = cursor
    placed.append((placement, cursor, required))
    high_water = max(high_water, cursor + required)
  return offsets, high_water


def _pack_for_capacity(
  placements: Iterable[StoragePlacement], alignment: int, capacity: int
) -> tuple[dict[str, int], int]:
  placements = list(placements)
  offsets, arena = _pack(placements, alignment)
  if arena <= capacity:
    return offsets, arena
  return _pack_compact(placements, alignment)


def _actions(placements: dict[str, StoragePlacement]) -> list[PlanAction]:
  result = []
  counter = 0

  def add(kind: ActionKind, event: int, phase: str, storage_id: str, **kwargs) -> None:
    nonlocal counter
    counter += 1
    result.append(
      PlanAction(
        id=f"a{counter}",
        kind=kind,
        event=event,
        phase=phase,
        storage_id=storage_id,
        **kwargs,
      )
    )

  for placement in placements.values():
    if placement.tier == MemoryTier.fast:
      add(
        ActionKind.allocate,
        placement.begin_event,
        "before",
        placement.storage_id,
        tier=MemoryTier.fast,
        bytes=placement.bytes,
        offset=placement.offset,
      )
      if placement.replicated:
        add(
          ActionKind.copy,
          (
            placement.begin_event
            if placement.annotations.get("eager_external_replica")
            else placement.first_access or placement.begin_event
          ),
          "before",
          placement.storage_id,
          source_tier=MemoryTier.slow,
          target_tier=MemoryTier.fast,
          bytes=placement.bytes,
          offset=placement.offset,
        )
      add(
        ActionKind.free,
        placement.end_event,
        "after",
        placement.storage_id,
        tier=MemoryTier.fast,
        bytes=placement.bytes,
        offset=placement.offset,
      )
    elif not placement.external_home:
      add(
        ActionKind.allocate,
        placement.begin_event,
        "before",
        placement.storage_id,
        tier=MemoryTier.slow,
        bytes=placement.bytes,
        offset=placement.offset,
      )
      add(
        ActionKind.free,
        placement.end_event,
        "after",
        placement.storage_id,
        tier=MemoryTier.slow,
        bytes=placement.bytes,
        offset=placement.offset,
      )
  phase_order = {"before": 0, "after": 1}
  kind_order = {ActionKind.allocate: 0, ActionKind.copy: 1, ActionKind.free: 2}
  return sorted(
    result,
    key=lambda item: (
      item.event,
      phase_order[item.phase],
      kind_order[item.kind],
      int(item.id[1:]),
    ),
  )


def _loop_templates(graph: ExecutionGraph) -> list[dict]:
  result = []
  for loop in graph.loops:
    samples = []
    for sample in loop.samples:
      samples.append(
        {
          "iteration": sample.iteration,
          "role": sample.role,
          "begin_event": sample.begin_event,
          "end_event": sample.end_event,
          "input_tensors": list(sample.input_tensors),
          "output_tensors": list(sample.output_tensors),
        }
      )
    recurrence = []
    for previous, current in zip(loop.samples, loop.samples[1:]):
      for slot, (source, target) in enumerate(
        zip(previous.output_tensors, current.input_tensors)
      ):
        recurrence.append(
          {
            "from_iteration": previous.iteration,
            "to_iteration": current.iteration,
            "slot": slot,
            "source_tensor": source,
            "source_storage": graph.tensors[source].storage_id,
            "target_tensor": target,
            "target_storage": graph.tensors[target].storage_id,
          }
        )
    result.append(
      {
        "id": loop.id,
        "name": loop.name,
        "trip_count": loop.trip_count,
        "samples": samples,
        "recurrence": recurrence,
      }
    )
  return result


def plan_memory(
  graph: ExecutionGraph,
  machine: MachineProfile,
  *,
  config: PlannerConfig | None = None,
) -> MemoryPlan:
  """Plan one concrete graph for bounded fast and slow CPU memory."""
  graph.validate()
  machine.validate()
  config = config or PlannerConfig()
  profiles = _profiles(graph, machine)
  graph_end = graph.events[-1].index if graph.events else 0
  selected, _logical_fast_peak = _select_fast(
    profiles,
    graph_end,
    machine.fast_capacity,
    machine.alignment,
    config,
  )

  placements = {}
  for profile in profiles:
    fast = profile.storage_id in selected
    if fast and config.eager_external_replicas and profile.external_home:
      begin, end = 0, graph_end
    else:
      begin = profile.fast_begin if fast else profile.begin
      end = profile.fast_end if fast else profile.end
    placements[profile.storage_id] = StoragePlacement(
      storage_id=profile.storage_id,
      storage_class=profile.storage_class,
      bytes=profile.bytes,
      tier=MemoryTier.fast if fast else MemoryTier.slow,
      begin_event=begin,
      end_event=end,
      first_access=profile.first_access,
      last_access=profile.last_access,
      alignment=machine.alignment,
      read_only=profile.read_only,
      external_home=profile.external_home,
      replicated=fast and profile.external_home,
      estimated_benefit_ns=profile.benefit_ns if fast else 0.0,
      weighted_accesses=profile.weighted_accesses,
      weighted_access_bytes=profile.weighted_access_bytes,
      annotations={
        "eager_external_replica": bool(
          fast and profile.external_home and config.eager_external_replicas
        ),
        "loop_replay_safe": profile.loop_replay_safe,
      },
    )

  fast_items = [item for item in placements.values() if item.tier == MemoryTier.fast]
  slow_items = [
    item
    for item in placements.values()
    if item.tier == MemoryTier.slow and not item.external_home
  ]
  fast_offsets, fast_arena = _pack_for_capacity(
    fast_items, machine.alignment, machine.fast_capacity
  )
  slow_offsets, slow_arena = _pack(slow_items, machine.alignment)
  profile_by_id = {profile.storage_id: profile for profile in profiles}
  while fast_arena > machine.fast_capacity and selected:
    victim = min(
      selected,
      key=lambda storage_id: (
        placements[storage_id].estimated_benefit_ns
        / max(
          placements[storage_id].bytes
          * (
            placements[storage_id].end_event
            - placements[storage_id].begin_event
            + 1
          ),
          1,
        ),
        storage_id,
      ),
    )
    selected.remove(victim)
    profile = profile_by_id[victim]
    placement = placements[victim]
    placement.tier = MemoryTier.slow
    placement.begin_event = profile.begin
    placement.end_event = profile.end
    placement.replicated = False
    placement.estimated_benefit_ns = 0.0
    placement.offset = None
    fast_items = [item for item in placements.values() if item.tier == MemoryTier.fast]
    slow_items = [
      item
      for item in placements.values()
      if item.tier == MemoryTier.slow and not item.external_home
    ]
    fast_offsets, fast_arena = _pack_for_capacity(
      fast_items, machine.alignment, machine.fast_capacity
    )
    slow_offsets, slow_arena = _pack(slow_items, machine.alignment)
  for storage_id, offset in fast_offsets.items():
    placements[storage_id].offset = offset
  for storage_id, offset in slow_offsets.items():
    placements[storage_id].offset = offset
  if fast_arena > machine.fast_capacity:
    raise ValueError(
      f"packed fast arena {fast_arena} exceeds capacity {machine.fast_capacity}"
    )
  if slow_arena > machine.slow_capacity:
    raise ValueError(
      f"packed slow arena {slow_arena} exceeds capacity {machine.slow_capacity}"
    )

  copy_bytes = sum(item.bytes for item in fast_items if item.replicated)
  fast_weighted_bytes = sum(item.weighted_access_bytes for item in fast_items)
  slow_weighted_bytes = sum(
    item.weighted_access_bytes
    for item in placements.values()
    if item.tier == MemoryTier.slow
  )
  plan = MemoryPlan(
    graph_name=graph.name,
    graph_digest=graph_digest(graph),
    machine=machine,
    placements=placements,
    actions=_actions(placements),
    fast_arena_bytes=fast_arena,
    slow_arena_bytes=slow_arena,
    metrics={
      "graph_events": len(graph.events),
      "graph_storages": len(graph.storages),
      "graph_peak_bytes": _profile_peak(profiles, graph_end),
      "managed_graph_peak_bytes": _profile_peak(
        profiles, graph_end, managed_only=True
      ),
      "fast_storages": len(fast_items),
      "slow_storages": len(placements) - len(fast_items),
      "replicated_storages": sum(item.replicated for item in fast_items),
      "logical_fast_peak_bytes": _selected_peak(
        profiles, selected, graph_end, config
      ),
      "fast_arena_bytes": fast_arena,
      "slow_arena_bytes": slow_arena,
      "copy_to_fast_bytes": copy_bytes,
      "weighted_fast_access_bytes": fast_weighted_bytes,
      "weighted_slow_access_bytes": slow_weighted_bytes,
      "weighted_fast_access_fraction": fast_weighted_bytes
      / max(fast_weighted_bytes + slow_weighted_bytes, 1),
      "estimated_benefit_ns": sum(
        max(item.estimated_benefit_ns, 0.0) for item in fast_items
      ),
    },
    attributes={
      "planner": "weighted-interval-greedy-v1",
      "config": asdict(config),
      "loop_templates": _loop_templates(graph),
    },
  )
  from .validate import validate_plan

  validate_plan(graph, plan)
  return plan


def load_or_plan(
  graph: ExecutionGraph,
  machine: MachineProfile,
  path: str | Path,
  *,
  config: PlannerConfig | None = None,
) -> tuple[MemoryPlan, bool]:
  """Load one compatible cached plan or atomically replace it.

  Returns ``(plan, cache_hit)``. Corrupt, stale, or machine-incompatible cache
  entries are regenerated instead of being replayed.
  """
  from .validate import validate_plan

  destination = Path(path)
  config = config or PlannerConfig()
  if destination.exists():
    try:
      cached = MemoryPlan.read(destination)
      if cached.machine != machine:
        raise ValueError("cached plan uses another machine profile")
      if cached.attributes.get("config") != asdict(config):
        raise ValueError("cached plan uses another planner configuration")
      validate_plan(graph, cached)
      return cached, True
    except (KeyError, TypeError, ValueError, OSError):
      pass
  result = plan_memory(graph, machine, config=config)
  result.write(destination)
  return result, False


__all__ = ["PlannerConfig", "graph_digest", "load_or_plan", "plan_memory"]
