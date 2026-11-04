# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Framework-neutral logical buffers; available without the physical memory component."""
from __future__ import annotations
import contextvars
import math
from dataclasses import asdict, dataclass
from .. import _C

_current = contextvars.ContextVar('vecops_buffer_session', default=None)


@dataclass(frozen=True)
class BufferRequirement:
  """One storage alias group, including a kernel-internal temporary.

  Lifetimes are half-open execution intervals. Parallel work must be represented
  conservatively unless a completion dependency proves non-overlap. Traffic is
  weighted over expected replays; replicas include their aligned physical span.
  """
  name: str
  bytes: int
  alignment: int = 64
  replicas: int = 1
  placement: str = 'preferred'
  traffic_bytes: float = 0
  lifetimes: tuple[tuple[int, int], ...] = ((0, 1),)


def _requests(requirements):
  values = [asdict(x) if isinstance(x, BufferRequirement) else dict(x) for x in requirements]
  names = [x['name'] for x in values]
  if len(set(names)) != len(names):
    raise ValueError('buffer names must be unique within a plan')
  for x in values:
    if not math.isfinite(x.get('traffic_bytes', 0)) or x.get('traffic_bytes', 0) < 0:
      raise ValueError('buffer traffic must be finite and nonnegative')
  return values


def plan_buffers(requirements, *, fast_capacity, slow_capacity):
  """Use the same native lifetime/coloring planner as C++ kernel workspace."""
  return _C.plan_buffers(_requests(requirements), int(fast_capacity), int(slow_capacity))


def current_session():
  return _current.get()
