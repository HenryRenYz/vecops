# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Framework-neutral execution and tensor-storage graph IR.

The IR deliberately contains no Torch objects.  Framework adapters translate
their tensors, operators, alias contracts, and loop structure into the plain
dataclasses below; memory planners can therefore consume a serialized graph
without importing the framework that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


GRAPH_FORMAT_VERSION = 1
Dimension = int | str


class AccessKind(str, Enum):
  """How one event accesses an existing logical tensor value."""

  read = "read"
  write = "write"
  read_write = "read_write"
  alias = "alias"
  allocate = "allocate"

  @property
  def writes(self) -> bool:
    return self in {AccessKind.write, AccessKind.read_write}


class TensorKind(str, Enum):
  """Origin/ownership class of a logical tensor."""

  input = "input"
  output = "output"
  parameter = "parameter"
  buffer = "buffer"
  temporary = "temporary"
  external = "external"
  unknown = "unknown"


@dataclass
class TensorMetadata:
  """Storage-free tensor properties needed by liveness and allocation."""

  shape: tuple[Dimension, ...]
  strides: tuple[Dimension, ...]
  dtype: str
  device: str
  layout: str = "strided"
  element_size: int | None = None
  storage_offset: Dimension = 0
  logical_bytes: int | str | None = None
  requires_grad: bool = False


@dataclass
class TensorUse:
  """One ordered access to a tensor, including the storage version change."""

  tensor_id: str
  access: AccessKind
  argument: str | None = None
  version_before: int = 0
  version_after: int = 0


@dataclass
class Event:
  """One ordered program event at a stable logical scope."""

  id: str
  index: int
  kind: str
  name: str
  scope: tuple[str, ...] = ()
  uses: list[TensorUse] = field(default_factory=list)
  outputs: list[str] = field(default_factory=list)
  attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorValue:
  """One logical tensor view/value; aliases share ``storage_id``."""

  id: str
  metadata: TensorMetadata
  storage_id: str
  name: str | None = None
  kind: TensorKind = TensorKind.unknown
  alias_of: str | None = None
  alias_kind: str | None = None
  producer_event: str | None = None
  lifetime_start: int = 0
  lifetime_end: int = 0
  persistent: bool = False
  read_only: bool = False
  escapes: bool = False
  annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageValue:
  """One physical allocation identity shared by one or more tensor views."""

  id: str
  bytes: int | str | None
  tensors: list[str] = field(default_factory=list)
  allocation_event: str | None = None
  lifetime_start: int = 0
  lifetime_end: int = 0
  external: bool = False
  persistent: bool = False
  annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopSample:
  """One observed prologue/steady/epilogue iteration and its event range."""

  iteration: int | str
  role: str | None
  begin_event: str
  end_event: str | None = None
  input_tensors: list[str] = field(default_factory=list)
  output_tensors: list[str] = field(default_factory=list)


@dataclass
class Loop:
  """A possibly sampled repeated structure in the captured execution."""

  id: str
  name: str
  scope: tuple[str, ...]
  trip_count: int | str
  observed_iterations: list[int | str] = field(default_factory=list)
  samples: list[LoopSample] = field(default_factory=list)
  begin_event: str | None = None
  end_event: str | None = None
  attributes: dict[str, Any] = field(default_factory=dict)


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
class ExecutionGraph:
  """Complete framework-neutral trace produced by one model execution."""

  name: str
  events: list[Event]
  tensors: dict[str, TensorValue]
  storages: dict[str, StorageValue]
  loops: list[Loop] = field(default_factory=list)
  inputs: list[str] = field(default_factory=list)
  outputs: list[str] = field(default_factory=list)
  attributes: dict[str, Any] = field(default_factory=dict)
  format_version: int = GRAPH_FORMAT_VERSION

  def validate(self) -> None:
    """Raise ``ValueError`` when IDs, ordering, aliasing, or lifetimes disagree."""
    if self.format_version != GRAPH_FORMAT_VERSION:
      raise ValueError(f"unsupported execution graph format {self.format_version}")
    event_ids: set[str] = set()
    previous = -1
    for event in self.events:
      if event.id in event_ids:
        raise ValueError(f"duplicate event ID {event.id}")
      if event.index <= previous:
        raise ValueError("event indexes must be strictly increasing")
      previous = event.index
      event_ids.add(event.id)
      for use in event.uses:
        if use.tensor_id not in self.tensors:
          raise ValueError(f"event {event.id} uses unknown tensor {use.tensor_id}")
        if use.version_after < use.version_before:
          raise ValueError(f"event {event.id} moves a storage version backwards")
      for output in event.outputs:
        if output not in self.tensors:
          raise ValueError(f"event {event.id} produces unknown tensor {output}")

    graph_end = self.events[-1].index if self.events else 0
    for tensor_id, tensor in self.tensors.items():
      storage = self.storages.get(tensor.storage_id)
      if storage is None:
        raise ValueError(f"tensor {tensor_id} references unknown storage {tensor.storage_id}")
      if tensor_id not in storage.tensors:
        raise ValueError(f"storage {storage.id} does not list tensor {tensor_id}")
      if tensor.alias_of is not None:
        base = self.tensors.get(tensor.alias_of)
        if base is None:
          raise ValueError(f"tensor {tensor_id} aliases unknown tensor {tensor.alias_of}")
        if base.storage_id != tensor.storage_id:
          raise ValueError(f"tensor {tensor_id} and alias base use different storages")
      if not 0 <= tensor.lifetime_start <= tensor.lifetime_end <= graph_end:
        raise ValueError(f"tensor {tensor_id} has an invalid lifetime")
      metadata = tensor.metadata
      if (
        isinstance(storage.bytes, int)
        and isinstance(metadata.element_size, int)
        and isinstance(metadata.storage_offset, int)
        and all(isinstance(item, int) for item in metadata.shape)
        and all(isinstance(item, int) for item in metadata.strides)
        and len(metadata.shape) == len(metadata.strides)
        and all(item >= 0 for item in metadata.shape)
      ):
        if any(item == 0 for item in metadata.shape):
          required_bytes = 0
        else:
          low = metadata.storage_offset
          high = metadata.storage_offset
          for extent, stride in zip(metadata.shape, metadata.strides):
            distance = (extent - 1) * stride
            low += min(distance, 0)
            high += max(distance, 0)
          if low < 0:
            raise ValueError(f"tensor {tensor_id} addresses before its storage")
          required_bytes = (high + 1) * metadata.element_size
        if required_bytes > storage.bytes:
          raise ValueError(
            f"tensor {tensor_id} requires {required_bytes} bytes from storage {storage.id}, "
            f"which has only {storage.bytes}; shape={metadata.shape}, "
            f"strides={metadata.strides}, storage_offset={metadata.storage_offset}, "
            f"element_size={metadata.element_size}, dtype={metadata.dtype}, "
            f"producer_event={tensor.producer_event}, alias_of={tensor.alias_of}"
          )

    for storage_id, storage in self.storages.items():
      if not storage.tensors:
        raise ValueError(f"storage {storage_id} has no tensor members")
      starts = [self.tensors[item].lifetime_start for item in storage.tensors]
      ends = [self.tensors[item].lifetime_end for item in storage.tensors]
      if storage.lifetime_start != min(starts) or storage.lifetime_end != max(ends):
        raise ValueError(f"storage {storage_id} lifetime is not the union of its aliases")

    for tensor_id in [*self.inputs, *self.outputs]:
      if tensor_id not in self.tensors:
        raise ValueError(f"graph boundary references unknown tensor {tensor_id}")
    for loop in self.loops:
      if loop.begin_event is not None and loop.begin_event not in event_ids:
        raise ValueError(f"loop {loop.id} has an unknown begin event")
      if loop.end_event is not None and loop.end_event not in event_ids:
        raise ValueError(f"loop {loop.id} has an unknown end event")
      if [sample.iteration for sample in loop.samples] != loop.observed_iterations:
        raise ValueError(f"loop {loop.id} samples disagree with observed iterations")
      for sample in loop.samples:
        if sample.begin_event not in event_ids or sample.end_event not in event_ids:
          raise ValueError(f"loop {loop.id} sample has an unknown event range")
        for tensor_id in [*sample.input_tensors, *sample.output_tensors]:
          if tensor_id not in self.tensors:
            raise ValueError(f"loop {loop.id} sample references unknown tensor {tensor_id}")

  def to_dict(self) -> dict[str, Any]:
    return _plain(asdict(self))

  @classmethod
  def from_dict(cls, values: dict[str, Any]) -> "ExecutionGraph":
    def metadata(item):
      item = dict(item)
      item["shape"] = tuple(item.get("shape", ()))
      item["strides"] = tuple(item.get("strides", ()))
      return TensorMetadata(**item)

    tensors = {}
    for key, item in values.get("tensors", {}).items():
      item = dict(item)
      item["metadata"] = metadata(item["metadata"])
      item["kind"] = TensorKind(item.get("kind", TensorKind.unknown.value))
      tensors[key] = TensorValue(**item)
    storages = {
      key: StorageValue(**dict(item))
      for key, item in values.get("storages", {}).items()
    }
    events = []
    for raw in values.get("events", []):
      item = dict(raw)
      item["scope"] = tuple(item.get("scope", ()))
      item["uses"] = [
        TensorUse(
          **{
            **dict(use),
            "access": AccessKind(use["access"]),
          }
        )
        for use in item.get("uses", [])
      ]
      events.append(Event(**item))
    loops = []
    for raw in values.get("loops", []):
      item = dict(raw)
      item["scope"] = tuple(item.get("scope", ()))
      item["samples"] = [LoopSample(**dict(sample)) for sample in item.get("samples", [])]
      loops.append(Loop(**item))
    graph = cls(
      name=values["name"],
      events=events,
      tensors=tensors,
      storages=storages,
      loops=loops,
      inputs=list(values.get("inputs", [])),
      outputs=list(values.get("outputs", [])),
      attributes=dict(values.get("attributes", {})),
      format_version=int(values.get("format_version", GRAPH_FORMAT_VERSION)),
    )
    graph.validate()
    return graph

  def dumps(self, *, indent: int | None = 2) -> str:
    return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

  @classmethod
  def loads(cls, document: str) -> "ExecutionGraph":
    return cls.from_dict(json.loads(document))

  def write(self, path: str | Path, *, indent: int | None = 2) -> None:
    Path(path).write_text(self.dumps(indent=indent) + "\n", encoding="utf-8")

  @classmethod
  def read(cls, path: str | Path) -> "ExecutionGraph":
    return cls.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
  "AccessKind",
  "Dimension",
  "Event",
  "ExecutionGraph",
  "GRAPH_FORMAT_VERSION",
  "Loop",
  "LoopSample",
  "StorageValue",
  "TensorKind",
  "TensorMetadata",
  "TensorUse",
  "TensorValue",
]
