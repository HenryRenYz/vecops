# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Explicit builtin loading and pipeline assembly; no compilation-global IR."""

import importlib
import threading

from .passes.manager import PassGroup, PassInvocation, PassManager, registered_passes

_builtin_lock = threading.RLock()
_builtins_loaded = False
_BUILTIN_MODULES = (
  "passes.kernel.canonicalize",
  "passes.kernel.cleanup",
  "passes.kernel.forwarding",
  "passes.kernel.facts",
  "passes.kernel.motion",
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
  "backends.cpu.matrix_values",
  "backends.cpu.packing",
  "backends.cpu.layout_recipes",
  "passes.vector",
  "backends.cpu.passes",
  "backends.cpu.transpose",
  "backends.cpu.grid_pass",
  "backends.cpu.workspace",
  "backends.cpu.state",
  "backends.cpu.register_layout",
  "backends.cpu.registers",
  "backends.cpu.hardware",
  "backends.cpu.storage",
  "backends.cpu.captures",
  "backends.cpu.invariants",
  "backends.cpu.memory_proofs",
  "passes.verification",
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


def kernel_pipeline(pipeline=None, *, options=None, _registry_snapshot=None):
  load_builtin_passes()
  registry = registered_passes() if _registry_snapshot is None else _registry_snapshot

  def manager(sequence=None, *, stage="kernel"):
    return PassManager(sequence, stage=stage, options=options, _registry_snapshot=registry)

  if pipeline is None:
    options = dict(options or {})
    rounds = options.get("cleanup_iterations", 4)
    if type(rounds) is not int or not 1 <= rounds <= 16:
      raise ValueError("cleanup_iterations must be in [1,16]")

    cleanup_passes = []
    for spec in manager(stage="kernel-cleanup").specs:
      if spec.name == "verify-kernel":
        continue
      if spec.name == "dead-code-elimination":
        cleanup_passes.extend(("fold-value-transforms", "canonicalize-memory", "expression-cse"))
      cleanup_passes.append(PassInvocation(spec.name, stage="kernel-cleanup"))

    def cleanup(name):
      return PassGroup(
        name,
        tuple(cleanup_passes),
        max_iterations=rounds,
      )

    sequence = [PassInvocation("verify-kernel", stage="kernel-cleanup")]
    for spec in manager().specs:
      sequence.append(spec.name)
      if spec.name == "expression-cse":
        sequence.append(cleanup("normalize-cleanup"))
      elif spec.name == "cpu-expand-layout-recipes":
        sequence.extend(
          (
            cleanup("recipe-cleanup"),
            PassInvocation("verify-kernel", stage="kernel-cleanup"),
            PassGroup(
              "loop-optimization", tuple(spec.name for spec in manager(stage="kernel-loop").specs), stage="kernel-loop"
            ),
            cleanup("loop-cleanup"),
          )
        )
      elif spec.name == "unroll-serial":
        sequence.append(cleanup("unroll-cleanup"))
      elif spec.name == "plan-software-pipeline":
        sequence.extend(
          (
            cleanup("pipeline-cleanup"),
            "plan-computation-reuse",
            PassInvocation("verify-kernel", stage="kernel-cleanup"),
          )
        )
    pipeline = sequence
  selected = pipeline if isinstance(pipeline, PassManager) else manager(pipeline)
  if selected.stage != "kernel":
    raise ValueError("kernel optimization requires a kernel-stage PassManager")
  return selected


def optimize(ir, *, pipeline=None, options=None):
  return kernel_pipeline(pipeline, options=options).run(ir)


def cpu_vector_pipeline(*, options=None, _registry_snapshot=None):
  """CPU policy selection belongs to this backend, not to the frontend/IR."""
  load_builtin_passes()
  return PassManager(
    stage="vector", options={"backend": "cpu", **dict(options or {})}, _registry_snapshot=_registry_snapshot
  )


def cpu_memory_pipeline(*, options=None, _registry_snapshot=None):
  """Freeze layout/access planning independently of source emission."""
  load_builtin_passes()
  return PassManager(
    stage="cpu-memory", options={"backend": "cpu", **dict(options or {})}, _registry_snapshot=_registry_snapshot
  )
