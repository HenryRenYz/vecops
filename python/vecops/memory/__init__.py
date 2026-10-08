# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ergonomic heterogeneous CPU-memory allocation.

The default API uses placement intent names instead of exposing hwloc object
indexes. ``os_numa_id=`` remains available for diagnostics and expert control.
Allocations own their storage and implement the Python buffer protocol.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

from .. import _C

Buffer = _C.MemoryAllocation

_BACKENDS = {
  "auto": _C.MemoryBackend.auto,
  "system": _C.MemoryBackend.system,
  "hwloc": _C.MemoryBackend.hwloc,
}
_PLACEMENTS = {
  "default": _C.MemoryPlacement.default,
  "high_bandwidth": _C.MemoryPlacement.high_bandwidth,
  "low_latency": _C.MemoryPlacement.low_latency,
}
_FALLBACKS = {
  None: _C.MemoryFallback.none,
  "none": _C.MemoryFallback.none,
  "default": _C.MemoryFallback.to_default,
}
_RANKINGS = {
  "bandwidth": _C.MemoryRanking.bandwidth,
  "latency": _C.MemoryRanking.latency,
}
_KINDS = {
  "unknown": _C.MemoryKind.unknown,
  "dram": _C.MemoryKind.dram,
  "hbm": _C.MemoryKind.hbm,
  "cxl": _C.MemoryKind.cxl,
  "pmem": _C.MemoryKind.pmem,
}


def _choice(values: Mapping[Any, Any], value: Any, name: str):
  try:
    return values[value]
  except KeyError as error:
    choices = ", ".join(repr(item) for item in values)
    raise ValueError(f"{name} must be one of {choices}; got {value!r}") from error


class System:
  """One immutable topology snapshot plus process-local budgets and counters."""

  def __init__(
    self,
    *,
    backend: str = "auto",
    target_overrides: Sequence[Mapping[str, Any]] = (),
    path_overrides: Sequence[Mapping[str, Any]] = (),
  ) -> None:
    config = _C.MemoryConfig()
    config.backend = _choice(_BACKENDS, backend, "backend")
    native_targets = []
    for values in target_overrides:
      override = _C.MemoryTargetOverride()
      override.os_numa_id = values.get("os_numa_id")
      kind = values.get("kind")
      if kind is not None:
        override.kind = _choice(_KINDS, str(kind).lower(), "kind")
      override.max_managed_bytes = values.get("max_managed_bytes")
      override.min_free_bytes = values.get("min_free_bytes")
      native_targets.append(override)
    config.target_overrides = native_targets

    native_paths = []
    for values in path_overrides:
      override = _C.MemoryPathOverride()
      override.initiator_os_numa_id = values.get("initiator_os_numa_id")
      override.target_os_numa_id = int(values["target_os_numa_id"])
      override.bandwidth_mib_s = values.get("bandwidth_mib_s")
      override.latency_ns = values.get("latency_ns")
      native_paths.append(override)
    config.path_overrides = native_paths
    self._native = _C.MemorySystem.discover(config)

  def allocate(
    self,
    size: int,
    *,
    placement: str = "default",
    domain: int | None = None,
    fallback: str | None = "default",
    alignment: int = 64,
    tier: int | None = None,
    os_numa_id: int | None = None,
    large_pages: bool = True,
  ) -> Buffer:
    """Allocate uninitialized bytes and return an owning buffer."""
    request = _C.MemoryAllocationRequest()
    request.bytes = int(size)
    request.domain = domain
    request.alignment = int(alignment)
    request.use_large_pages = bool(large_pages)
    request.fallback = _choice(_FALLBACKS, fallback, "fallback")
    request.objective_rank = tier
    if os_numa_id is None:
      if tier is not None and placement not in {"high_bandwidth", "low_latency"}:
        raise ValueError("tier requires high_bandwidth or low_latency placement")
      request.intent = _choice(_PLACEMENTS, placement, "placement")
    else:
      if placement != "default" or tier is not None:
        raise ValueError("os_numa_id cannot be combined with placement or tier")
      request.intent = _C.MemoryPlacement.exact_target
      request.exact_os_numa_id = int(os_numa_id)
    return self._native.allocate(request)

  def topology(self) -> dict[str, Any]:
    return self._native.topology()

  def tiers(self, domain: int | None = None, *, objective: str = "bandwidth") -> list[dict[str, Any]]:
    if domain is None:
      domain = self.current_cpu_domain()
    return self._native.tiers(int(domain), _choice(_RANKINGS, objective, "objective"))

  def current_cpu_domain(self) -> int:
    return self._native.current_cpu_domain()

  def stats(self) -> list[dict[str, Any]]:
    return self._native.stats()

  def describe(self) -> str:
    return self._native.describe()

  def workspace_session(
    self,
    *,
    fast_capacity: int,
    slow_capacity: int | None = None,
    domain: int | None = None,
    allow_fast_fallback: bool = True,
    large_pages: bool = True,
  ):
    """Create one single-use, bounded vecops workspace session."""
    return self._native.workspace_session(
      int(fast_capacity),
      None if slow_capacity is None else int(slow_capacity),
      domain,
      bool(allow_fast_fallback),
      bool(large_pages),
    )


_default: System | None = None
_default_lock = threading.Lock()


def default_system() -> System:
  """Return the lazily discovered process-default memory system."""
  global _default
  if _default is None:
    with _default_lock:
      if _default is None:
        _default = System()
  return _default


def allocate(size: int, **options) -> Buffer:
  """Allocate bytes through :func:`default_system`."""
  return default_system().allocate(size, **options)


def topology() -> dict[str, Any]:
  return default_system().topology()


def tiers(domain: int | None = None, *, objective: str = "bandwidth") -> list[dict[str, Any]]:
  return default_system().tiers(domain, objective=objective)


def stats() -> list[dict[str, Any]]:
  return default_system().stats()


def describe() -> str:
  return default_system().describe()


from . import numpy

__all__ = [
  "Buffer",
  "System",
  "allocate",
  "default_system",
  "describe",
  "numpy",
  "stats",
  "tiers",
  "topology",
]
