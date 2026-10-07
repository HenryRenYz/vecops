"""Explicit builtin loading and pipeline assembly; no compilation-global IR."""

import importlib
import threading

from .passes.manager import PassManager

_builtin_lock = threading.RLock()
_builtins_loaded = False
_BUILTIN_MODULES = (
  "passes.kernel.canonicalize",
  "passes.kernel.regions",
  "passes.kernel.transforms",
  "passes.kernel.layout",
  "passes.kernel.matmul",
  "passes.kernel.iteration",
  "passes.kernel.indexed",
  "passes.kernel.consumers",
  "passes.kernel.reuse",
  "passes.kernel.fusion",
  "passes.kernel.unroll",
  "passes.kernel.pipeline",
  "backends.cpu.recipes",
  "passes.vector",
  "backends.cpu.passes",
  "backends.cpu.transpose",
  "backends.cpu.matmul",
  "backends.cpu.schedule",
  "backends.cpu.grid_pass",
  "backends.cpu.state",
  "backends.cpu.register_layout",
  "backends.cpu.registers",
  "backends.cpu.hardware",
)


def load_builtin_passes():
  """Load the builtins once, before taking registry snapshots.

  All builtin registration imports are owned here. Removing a registered pass
  deliberately does not cause a later compile to silently register it again.
  """
  global _builtins_loaded
  with _builtin_lock:
    if not _builtins_loaded:
      for module in _BUILTIN_MODULES:
        importlib.import_module(f"{__package__}.{module}")
      _builtins_loaded = True


def kernel_pipeline(pipeline=None, *, options=None):
  load_builtin_passes()
  manager = pipeline if isinstance(pipeline, PassManager) else PassManager(pipeline, stage="kernel", options=options)
  if manager.stage != "kernel":
    raise ValueError("kernel optimization requires a kernel-stage PassManager")
  return manager


def optimize(ir, *, pipeline=None, options=None):
  return kernel_pipeline(pipeline, options=options).run(ir)


def cpu_vector_pipeline(*, options=None):
  """CPU policy selection belongs to this backend, not to the frontend/IR."""
  load_builtin_passes()
  return PassManager(stage="vector", options={"backend": "cpu", **dict(options or {})})


def cpu_memory_pipeline(*, options=None):
  """Freeze layout/access planning independently of source emission."""
  load_builtin_passes()
  return PassManager(stage="cpu-memory", options={"backend": "cpu", **dict(options or {})})
