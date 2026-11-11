# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Framework-neutral model execution graph capture.

Use :class:`GraphCapture` directly from framework adapters, or pass
``capture_graph=True`` to :func:`vecops.precompile` for the Torch adapter.
"""

from ._discovery import discover_pointwise_requests
from ._execution_weights import event_execution_weights
from .capture import (
  GraphCapture,
  LoopHandle,
  current_capture,
  declare_loop,
  enter_scope,
  leave_scope,
  mark_alias,
  mark_outputs,
  mark_retained,
  mark_tensor,
  wrap_region,
)
from .ir import (
  AccessKind,
  Event,
  ExecutionGraph,
  Loop,
  LoopSample,
  StorageValue,
  TensorKind,
  TensorMetadata,
  TensorUse,
  TensorValue,
)

def provision_communication_capacity(plans,*,providers=(),allow_growth=True):
  """Probe or provision stopped graph transport groups before graph admission."""
  from ._replay import provision_communication_capacity as provision
  return provision(plans,providers=providers,allow_growth=allow_growth)


__all__ = [
  "AccessKind",
  "Event",
  "ExecutionGraph",
  "GraphCapture",
  "discover_pointwise_requests",
  "event_execution_weights",
  "provision_communication_capacity",
  "Loop",
  "LoopSample",
  "LoopHandle",
  "StorageValue",
  "TensorKind",
  "TensorMetadata",
  "TensorUse",
  "TensorValue",
  "current_capture",
  "declare_loop",
  "enter_scope",
  "leave_scope",
  "mark_alias",
  "mark_outputs",
  "mark_retained",
  "mark_tensor",
  "wrap_region",
]
