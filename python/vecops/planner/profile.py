"""Translate a live ``vecops.memory.System`` into planner cost inputs."""

from __future__ import annotations

from typing import Any

from .ir import MachineProfile


def profile_from_system(
  system: Any,
  *,
  fast_capacity: int,
  slow_capacity: int,
  domain: int | None = None,
  fast_bandwidth_mib_s: float | None = None,
  slow_bandwidth_mib_s: float | None = None,
  transfer_bandwidth_mib_s: float | None = None,
  alignment: int = 64,
  name: str | None = None,
) -> MachineProfile:
  """Build a two-tier profile from one immutable memory topology snapshot."""
  bandwidth_overridden = any(
    value is not None
    for value in (
      fast_bandwidth_mib_s,
      slow_bandwidth_mib_s,
      transfer_bandwidth_mib_s,
    )
  )
  if domain is None:
    domain = system.current_cpu_domain()
  bandwidth_tiers = system.tiers(domain, objective="bandwidth")
  latency_tiers = system.tiers(domain, objective="latency")
  if not bandwidth_tiers or bandwidth_tiers[0].get("value") is None:
    raise ValueError("memory topology has no bandwidth measurement for fast tier")
  discovered_fast_bandwidth = float(bandwidth_tiers[0]["value"])
  discovered_slow_bandwidth = discovered_fast_bandwidth
  for tier in bandwidth_tiers[1:]:
    value = tier.get("value")
    if value is not None and float(value) < discovered_fast_bandwidth:
      discovered_slow_bandwidth = float(value)
      break
  fast_bandwidth = (
    discovered_fast_bandwidth
    if fast_bandwidth_mib_s is None
    else float(fast_bandwidth_mib_s)
  )
  slow_bandwidth = (
    discovered_slow_bandwidth
    if slow_bandwidth_mib_s is None
    else float(slow_bandwidth_mib_s)
  )
  if transfer_bandwidth_mib_s is None:
    transfer_bandwidth_mib_s = slow_bandwidth
  transfer_latency = 0.0
  if latency_tiers and latency_tiers[0].get("value") is not None:
    transfer_latency = float(latency_tiers[0]["value"])
  topology = system.topology()
  return MachineProfile(
    fast_capacity=int(fast_capacity),
    slow_capacity=int(slow_capacity),
    fast_bandwidth_mib_s=fast_bandwidth,
    slow_bandwidth_mib_s=slow_bandwidth,
    transfer_bandwidth_mib_s=float(transfer_bandwidth_mib_s),
    transfer_latency_ns=transfer_latency,
    alignment=alignment,
    name=name or f"{topology.get('backend', 'memory')}:domain-{domain}",
    attributes={
      "domain": domain,
      "topology_backend": topology.get("backend"),
      "discovered_fast_bandwidth_mib_s": discovered_fast_bandwidth,
      "discovered_slow_bandwidth_mib_s": discovered_slow_bandwidth,
      "bandwidth_overridden": bandwidth_overridden,
      "fast_targets": bandwidth_tiers[0].get("targets", []),
      "slow_targets": (
        bandwidth_tiers[1].get("targets", [])
        if len(bandwidth_tiers) > 1
        else bandwidth_tiers[0].get("targets", [])
      ),
    },
  )


__all__ = ["profile_from_system"]
