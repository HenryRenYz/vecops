# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Registered, deterministic compiler pipelines with per-run analysis state.

Pass callbacks receive (ir, context). Return the IR, None for an in-place
transformation, or PassResult to explicitly preserve cached analyses. Registry
snapshots isolate concurrent/batch compilations from later registrations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PassResult:
  ir: object
  changed: bool = True
  preserved: tuple[str, ...] = ()


@dataclass
class PassContext:
  stage: str
  options: dict = field(default_factory=dict)
  analyses: dict = field(default_factory=dict)
  trace: list[str] = field(default_factory=list)

  @property
  def environment(self):
    from ..environment import freeze

    return freeze(self.options)

  def analysis(self, name, compute, ir):
    if name not in self.analyses:
      self.analyses[name] = compute(ir)
    return self.analyses[name]


@dataclass(frozen=True)
class PassSpec:
  name: str
  callback: object
  stage: str
  before: tuple[str, ...]
  after: tuple[str, ...]


_registry = {}
_lock = threading.RLock()


def register_pass(name, *, stage="kernel", before=(), after=()):
  """Register a named pass; ordering constraints are checked by PassManager.

  The built-in stages are kernel, vector and cpu-memory. A custom pass may use another
  stage and an explicitly selected manager; duplicate names are errors.
  """
  if not isinstance(name, str) or not name:
    raise ValueError("compiler pass name must be a nonempty string")
  before = (before,) if isinstance(before, str) else tuple(before)
  after = (after,) if isinstance(after, str) else tuple(after)

  def decorate(callback):
    with _lock:
      if name in _registry:
        raise ValueError(f"compiler pass {name!r} is already registered")
      _registry[name] = PassSpec(name, callback, stage, before, after)
    return callback

  return decorate


def unregister_pass(name):
  """Remove a plugin pass; existing pipeline snapshots remain valid."""
  with _lock:
    del _registry[name]


def registered_passes(*, stage=None):
  with _lock:
    return tuple(spec for spec in _registry.values() if stage is None or spec.stage == stage)


class PassManager:
  """Stable topological order, optional explicit pipeline, no current-kernel global."""

  def __init__(self, pipeline=None, *, stage="kernel", options=None):
    specs = {spec.name: spec for spec in registered_passes(stage=stage)}
    names = list(specs) if pipeline is None else list(pipeline)
    if len(set(names)) != len(names):
      raise ValueError("compiler pipeline contains duplicate passes")
    missing = set(names) - set(specs)
    if missing:
      raise ValueError(f"unknown {stage} passes: {sorted(missing)}")
    edges = {name: set() for name in names}
    for name in names:
      spec = specs[name]
      for predecessor in spec.after:
        if predecessor in edges:
          edges[name].add(predecessor)
        elif pipeline is None:
          raise ValueError(f"pass {name!r} requires unknown predecessor {predecessor!r}")
      for successor in spec.before:
        if successor in edges:
          edges[successor].add(name)
        elif pipeline is None:
          raise ValueError(f"pass {name!r} requires unknown successor {successor!r}")
    ordered = []
    while edges:
      ready = next((name for name, dependencies in edges.items() if not dependencies), None)
      if ready is None:
        raise ValueError(f"compiler pass ordering cycle: {sorted(edges)}")
      ordered.append(specs[ready])
      del edges[ready]
      for dependencies in edges.values():
        dependencies.discard(ready)
    self.specs = tuple(ordered)
    self.stage = stage
    self.options = dict(options or {})
    from ..environment import freeze

    self.options["_compiler_environment"] = freeze(self.options)

  def with_options(self, options):
    """Reuse registered-pass snapshots with a shared compilation environment."""
    result = object.__new__(type(self))
    result.specs, result.stage = self.specs, self.stage
    result.options = {**self.options, **options}
    return result

  def run(self, ir):
    from ..operations import use_catalog

    with use_catalog(self.options["_compiler_environment"].operations):
      return self._run(ir)

  def _run(self, ir):
    context = PassContext(self.stage, dict(self.options))
    session = self.options.get("_debug_session")
    for spec in self.specs:
      begin = session.begin_pass(spec.name, ir) if session is not None else None
      try:
        result = spec.callback(ir, context)
      except BaseException as error:
        if session is not None:
          session.end_pass(self.stage, spec.name, ir, begin, error)
          error.debug_info = session
        raise
      if isinstance(result, PassResult):
        ir = result.ir
        if result.changed:
          context.analyses = {k: v for k, v in context.analyses.items() if k in result.preserved}
      else:
        if result is not None:
          ir = result
        context.analyses.clear()
      context.trace.append(spec.name)
      if session is not None:
        session.end_pass(self.stage, spec.name, ir, begin)
    if hasattr(ir, "pass_trace"):
      ir.pass_trace.extend(context.trace)
    return ir


__all__ = ["PassContext", "PassManager", "PassResult", "register_pass", "registered_passes", "unregister_pass"]
