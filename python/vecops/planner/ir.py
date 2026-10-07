# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Framework-neutral memory-plan IR and machine profile."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


PLAN_FORMAT_VERSION = 1


class MemoryTier(str, Enum):
  fast = "fast"
  slow = "slow"


class StorageClass(str, Enum):
  persistent = "persistent"
  input = "input"
  output = "output"
  temporary = "temporary"
  external = "external"


class ActionKind(str, Enum):
  allocate = "allocate"
  copy = "copy"
  free = "free"


@dataclass(frozen=True)
class MachineProfile:
  """Two-tier CPU memory properties used by the offline planner."""

  fast_capacity: int
  slow_capacity: int
  fast_bandwidth_mib_s: float
  slow_bandwidth_mib_s: float
  transfer_bandwidth_mib_s: float
  transfer_latency_ns: float = 0.0
  alignment: int = 64
  name: str = "two-tier-cpu"
  attributes: dict[str, Any] = field(default_factory=dict)

  def validate(self) -> None:
    if self.fast_capacity < 0 or self.slow_capacity < 0:
      raise ValueError("memory capacities must be non-negative")
    if min(
      self.fast_bandwidth_mib_s,
      self.slow_bandwidth_mib_s,
      self.transfer_bandwidth_mib_s,
    ) <= 0:
      raise ValueError("memory bandwidths must be positive")
    if self.transfer_latency_ns < 0:
      raise ValueError("transfer latency must be non-negative")
    if self.alignment <= 0 or self.alignment & (self.alignment - 1):
      raise ValueError("alignment must be a positive power of two")


@dataclass
class StoragePlacement:
  """Placement of one alias group for its captured semantic lifetime."""

  storage_id: str
  storage_class: StorageClass
  bytes: int
  tier: MemoryTier
  begin_event: int
  end_event: int
  first_access: int | None
  last_access: int | None
  offset: int | None = None
  alignment: int = 64
  read_only: bool = False
  external_home: bool = False
  replicated: bool = False
  estimated_benefit_ns: float = 0.0
  weighted_accesses: float = 0.0
  weighted_access_bytes: float = 0.0
  annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanAction:
  """One allocation, transfer, or free anchored to a graph event."""

  id: str
  kind: ActionKind
  event: int
  phase: str
  storage_id: str
  tier: MemoryTier | None = None
  source_tier: MemoryTier | None = None
  target_tier: MemoryTier | None = None
  bytes: int = 0
  offset: int | None = None
  asynchronous: bool = False
  annotations: dict[str, Any] = field(default_factory=dict)


def _plain(value: Any) -> Any:
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, tuple):
    return [_plain(item) for item in value]
  if isinstance(value, list):
    return [_plain(item) for item in value]
  if isinstance(value, dict):
    return {str(key): _plain(item) for key, item in value.items()}
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  return str(value)


@dataclass
class MemoryPlan:
  """Serializable output of model-level memory planning."""

  graph_name: str
  graph_digest: str
  machine: MachineProfile
  placements: dict[str, StoragePlacement]
  actions: list[PlanAction]
  fast_arena_bytes: int
  slow_arena_bytes: int
  metrics: dict[str, int | float | str | bool]
  attributes: dict[str, Any] = field(default_factory=dict)
  format_version: int = PLAN_FORMAT_VERSION

  def to_dict(self) -> dict[str, Any]:
    return _plain(asdict(self))

  @classmethod
  def from_dict(cls, values: dict[str, Any]) -> "MemoryPlan":
    machine = MachineProfile(**dict(values["machine"]))
    placements = {}
    for key, raw in values.get("placements", {}).items():
      item = dict(raw)
      item["storage_class"] = StorageClass(item["storage_class"])
      item["tier"] = MemoryTier(item["tier"])
      placements[key] = StoragePlacement(**item)
    actions = []
    for raw in values.get("actions", []):
      item = dict(raw)
      item["kind"] = ActionKind(item["kind"])
      for name in ("tier", "source_tier", "target_tier"):
        if item.get(name) is not None:
          item[name] = MemoryTier(item[name])
      actions.append(PlanAction(**item))
    return cls(
      graph_name=values["graph_name"],
      graph_digest=values["graph_digest"],
      machine=machine,
      placements=placements,
      actions=actions,
      fast_arena_bytes=int(values["fast_arena_bytes"]),
      slow_arena_bytes=int(values["slow_arena_bytes"]),
      metrics=dict(values.get("metrics", {})),
      attributes=dict(values.get("attributes", {})),
      format_version=int(values.get("format_version", PLAN_FORMAT_VERSION)),
    )

  def dumps(self, *, indent: int | None = 2) -> str:
    return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

  @classmethod
  def loads(cls, document: str) -> "MemoryPlan":
    return cls.from_dict(json.loads(document))

  def write(self, path: str | Path, *, indent: int | None = 2) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
      prefix=f".{destination.name}.",
      suffix=".tmp",
      dir=destination.parent,
      text=True,
    )
    try:
      with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(self.dumps(indent=indent) + "\n")
        output.flush()
        os.fsync(output.fileno())
      os.replace(temporary, destination)
    except BaseException:
      try:
        os.unlink(temporary)
      except FileNotFoundError:
        pass
      raise

  @classmethod
  def read(cls, path: str | Path) -> "MemoryPlan":
    return cls.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
  "ActionKind",
  "MachineProfile",
  "MemoryPlan",
  "MemoryTier",
  "PLAN_FORMAT_VERSION",
  "PlanAction",
  "StorageClass",
  "StoragePlacement",
]
