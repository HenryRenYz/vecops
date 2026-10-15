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
from types import MappingProxyType


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


@dataclass(frozen=True)
class PassInvocation:
  """One invocation of a registered implementation in an explicit sequence.

  Registration names identify implementations, not execution slots. The same
  implementation can occur repeatedly with independent immutable options.
  ``label`` disambiguates those slots in diagnostic reports.
  """

  name: str
  options: object = field(default_factory=dict)
  label: str | None = None
  stage: str | None = None

  def __post_init__(self):
    object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True)
class PassGroup:
  """A named nested sequence, optionally repeated to a bounded fixed point.

  ``stage`` selects the callback contract; None inherits the parent stage.
  A round converges only when every callback reports ``changed=False`` through
  PassResult. Legacy callbacks conservatively count as changed. This avoids
  using object identity or an expensive/possibly lossy textual IR comparison.
  """

  name: str
  pipeline: tuple
  stage: str | None = None
  max_iterations: int = 1

  def __post_init__(self):
    if not self.name or type(self.max_iterations) is not int or self.max_iterations < 1:
      raise ValueError("pass groups require a name and positive max_iterations")
    object.__setattr__(self, "pipeline", tuple(self.pipeline))


@dataclass(frozen=True)
class _Invocation:
  spec: PassSpec
  options: object
  label: str


@dataclass(frozen=True)
class _Group:
  name: str
  stage: str
  steps: tuple
  max_iterations: int


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
  """Frozen pass program with nested scopes and repeatable invocations.

  The default registry pipeline uses stable before/after topological ordering.
  An explicitly supplied pipeline is an executable sequence: its order is
  authoritative, and omitted default passes are not silently inserted.
  """

  def __init__(self, pipeline=None, *, stage="kernel", options=None, _registry_snapshot=None):
    registry = {spec.name: spec for spec in (registered_passes() if _registry_snapshot is None else _registry_snapshot)}
    specs = {name: spec for name, spec in registry.items() if spec.stage == stage}
    if pipeline is None:
      pipeline = self._default_order(specs)

    def resolve(sequence, current_stage):
      result = []
      for item in sequence:
        if isinstance(item, PassGroup):
          child_stage = item.stage or current_stage
          result.append(_Group(item.name, child_stage, resolve(item.pipeline, child_stage), item.max_iterations))
          continue
        invocation = PassInvocation(item) if isinstance(item, str) else item
        if not isinstance(invocation, PassInvocation):
          raise TypeError("pipeline entries must be pass names, PassInvocation or PassGroup")
        spec = registry.get(invocation.name)
        selected_stage = invocation.stage or current_stage
        if spec is None or spec.stage != selected_stage:
          raise ValueError(f"unknown {selected_stage} pass: {invocation.name!r}")
        result.append(_Invocation(spec, invocation.options, invocation.label or invocation.name))
      return tuple(result)

    self.steps = resolve(pipeline, stage)

    def flatten(steps):
      for step in steps:
        if isinstance(step, _Group):
          yield from flatten(step.steps)
        else:
          yield step.spec

    self.specs = tuple(flatten(self.steps))
    self.stage = stage
    self.options = dict(options or {})
    from ..environment import freeze

    self.options["_compiler_environment"] = freeze(self.options)

  @staticmethod
  def _default_order(specs):
    names = list(specs)
    edges = {name: set() for name in names}
    for name in names:
      spec = specs[name]
      for predecessor in spec.after:
        if predecessor in edges:
          edges[name].add(predecessor)
        else:
          raise ValueError(f"pass {name!r} requires unknown predecessor {predecessor!r}")
      for successor in spec.before:
        if successor in edges:
          edges[successor].add(name)
        else:
          raise ValueError(f"pass {name!r} requires unknown successor {successor!r}")
    ordered = []
    while edges:
      ready = next((name for name, dependencies in edges.items() if not dependencies), None)
      if ready is None:
        raise ValueError(f"compiler pass ordering cycle: {sorted(edges)}")
      ordered.append(ready)
      del edges[ready]
      for dependencies in edges.values():
        dependencies.discard(ready)
    return ordered

  def with_options(self, options):
    """Reuse registered-pass snapshots with a shared compilation environment."""
    result = object.__new__(type(self))
    result.specs, result.steps, result.stage = self.specs, self.steps, self.stage
    result.options = {**self.options, **options}
    return result

  def run(self, ir):
    from ..operations import use_catalog

    with use_catalog(self.options["_compiler_environment"].operations):
      return self._run(ir)

  def _run(self, ir):
    context = PassContext(self.stage, dict(self.options))
    session = self.options.get("_debug_session")

    def run_steps(steps, value, stage, path=()):
      changed = False
      for step in steps:
        if isinstance(step, _Group):
          for iteration in range(step.max_iterations):
            value, round_changed = run_steps(step.steps, value, step.stage, (*path, (step.name, iteration)))
            changed |= round_changed
            if not round_changed:
              break
          continue
        spec = step.spec
        previous_options, previous_stage = context.options, context.stage
        if step.options or previous_stage != spec.stage:
          context.analyses.clear()
        context.options = {**previous_options, **step.options}
        # A local option override must not replace the compilation's frozen
        # semantics/target snapshot or debug session.
        context.options["_compiler_environment"] = self.options["_compiler_environment"]
        context.stage = spec.stage
        begin = session.begin_pass(spec.name, value) if session is not None else None
        try:
          result = spec.callback(value, context)
        except BaseException as error:
          if session is not None:
            session.end_pass(spec.stage, spec.name, value, begin, error)
            error.debug_info = session
          raise
        finally:
          context.options, context.stage = previous_options, previous_stage
        if isinstance(result, PassResult):
          value = result.ir
          pass_changed = result.changed
          if pass_changed:
            context.analyses = {k: v for k, v in context.analyses.items() if k in result.preserved}
        else:
          if result is not None:
            value = result
          pass_changed = True
          context.analyses.clear()
        if step.options or previous_stage != spec.stage:
          context.analyses.clear()
        changed |= pass_changed
        context.trace.append(spec.name)
        if session is not None:
          session.end_pass(spec.stage, spec.name, value, begin)
          session.passes[-1].update(invocation=step.label, groups=path, reported_changed=pass_changed)
      return value, changed

    ir, _ = run_steps(self.steps, ir, self.stage)
    if hasattr(ir, "pass_trace"):
      ir.pass_trace.extend(context.trace)
    return ir


__all__ = [
  "PassContext",
  "PassGroup",
  "PassInvocation",
  "PassManager",
  "PassResult",
  "PassSpec",
  "register_pass",
  "registered_passes",
  "unregister_pass",
]
