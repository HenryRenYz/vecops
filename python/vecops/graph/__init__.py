"""Framework-neutral model execution graph capture.

Use :class:`GraphCapture` directly from framework adapters, or pass
``capture_graph=True`` to :func:`vecops.precompile` for the Torch adapter.
"""

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

__all__ = [
  "AccessKind",
  "Event",
  "ExecutionGraph",
  "GraphCapture",
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
