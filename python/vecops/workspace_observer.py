# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Python-side control of the native workspace allocation observer.

The native collector is installed into every loaded Torch bridge DSO
through their exported ``vecops_torch_bridge_set_allocation_observer_v1``
setter.  The C callback only pushes records into a mutex-held native
buffer, so it never touches the GIL and is safe under kernel worker
threads.
"""

from __future__ import annotations

import ctypes
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from . import _C
from . import _schema_bridge


class _ObserverV1(ctypes.Structure):
  """Mirror of ``VecopsAllocationObserverV1`` (native alignment)."""

  _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("user_context", ctypes.c_void_p),
      ("on_workspace_allocation", ctypes.c_void_p),
  ]


_state = threading.Lock()
_observer: _ObserverV1 | None = None
_active = False


def _pointer() -> ctypes.c_void_p:
  """Address of the active observer struct, or NULL when disabled."""
  with _state:
    if not _active or _observer is None:
      return ctypes.c_void_p(None)
    return ctypes.c_void_p(ctypes.addressof(_observer))


_schema_bridge._workspace_observer_factory = _pointer


def _install_into_loaded_bridges() -> int:
  """Push the current observer into every already-loaded bridge DSO.

  ``_loaded_bridge_modules`` is a flat list (bridge DSOs interleaved with
  the native runtime DSO); only bridge DSOs expose the observer setter.
  """
  installed = 0
  pointer = _pointer()
  for module in list(_schema_bridge._loaded_bridge_modules):
    setter = getattr(module, "vecops_torch_bridge_set_allocation_observer_v1", None)
    if setter is None:
      continue
    setter.argtypes = [ctypes.c_void_p]
    setter.restype = None
    setter(pointer)
    installed += 1
  return installed


def drain() -> list[dict[str, Any]]:
  """Return and clear the collected workspace allocation records."""
  return list(_C.drain_workspace_allocations())


@contextmanager
def tracking() -> Iterator[None]:
  """Collect native workspace allocations on this process.

  While active, every kernel-internal workspace allocation (fast/slow
  arena or heap spill) is recorded with its recipe, site, size and tier.
  Records are retrieved with :func:`drain`.
  """
  global _observer, _active
  with _state:
    if _active:
      raise RuntimeError("workspace allocation tracking is already active")
    callback, user_context, struct_size = _C.set_workspace_allocation_tracking(True)
    observer = _ObserverV1()
    if ctypes.sizeof(_ObserverV1) != struct_size:
      raise RuntimeError(
          "workspace observer ABI mismatch: "
          f"python {ctypes.sizeof(_ObserverV1)} != native {struct_size}"
      )
    observer.struct_size = struct_size
    observer.user_context = ctypes.c_void_p(user_context)
    observer.on_workspace_allocation = ctypes.c_void_p(callback)
    _observer = observer
    _active = True
  _install_into_loaded_bridges()
  try:
    yield
  finally:
    with _state:
      _active = False
    _install_into_loaded_bridges()
    _C.set_workspace_allocation_tracking(False)


__all__ = ["tracking", "drain"]
