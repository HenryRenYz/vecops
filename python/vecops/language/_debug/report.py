# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Read-only user-facing inspection and safe reproducibility bundles."""

import json
from dataclasses import dataclass
from pathlib import Path

from .options import DebugOptions
from .session import DebugSession, current_session


@dataclass(frozen=True)
class Unavailable:
  reason: str

  def __str__(self):
    return f"unavailable: {self.reason}"


class DebugInfo:
  format_version = 1

  def __init__(self, session=None, *, name="", cache=None, compiler=None):
    self._session = session or DebugSession()
    self.name, self.compiler = name, compiler
    self._cache = dict(cache or {})
    self.diagnostics = ()
    self.owner = None
    self.children = ()
    self._validate_cache = lambda: True

  def get_source(self, kind="kernel"):
    if kind not in self._session.sources and self.children:
      return {child.name: child.get_source(kind) for child in self.children}
    if kind not in self._session.sources and kind in ("adapter", "spec"):
      artifacts = self.artifacts()
      sources = []
      for artifact in artifacts:
        path = Path(artifact["command"]["file"])
        path = path if kind == "adapter" else path.parent / "vecops_spec.h"
        if path.is_file():
          sources.append(path.read_text())
      if sources:
        return "\n".join(dict.fromkeys(sources))
    return self._session.sources.get(kind, Unavailable(f"{kind} source has not been generated"))

  def get_ir(self, stage="kernel"):
    if stage not in self._session.stages and self.children:
      return {child.name: child.get_ir(stage) for child in self.children}
    return self._session.stages.get(stage, Unavailable(f"{stage} IR was not produced for this kernel"))

  def get_pass_report(self):
    import copy

    return (
      *tuple(copy.deepcopy(p) for p in self._session.passes),
      *(dict(p, kernel=child.name) for child in self.children for p in child.get_pass_report()),
    )

  def explain(self, topic=None):
    messages = (*self._session.messages, *self.diagnostics)
    import copy

    return (
      *tuple(copy.deepcopy(m) for m in messages if topic is None or topic.lower() in str(m).lower()),
      *(m for child in self.children for m in child.explain(topic)),
    )

  def get_cache_info(self):
    return dict(self._cache)

  def get_build_report(self):
    return (
      *(tuple(dict(e) for e in self.compiler.build_events) if self.compiler is not None else ()),
      *(e for child in self.children for e in child.get_build_report()),
    )

  def artifacts(self):
    if self.compiler is None:
      return ()
    root = self.compiler.build_dir
    if not root.is_dir():
      return ()
    # Build metadata identifies the actual module; never silently pick a
    # different kernel's assembly merely because it was compiled most recently.
    identity = self._cache.get("source_path")
    result = []
    for command_file in root.glob("*/build/compile_commands.json"):
      try:
        commands = json.loads(command_file.read_text())
      except (OSError, ValueError):
        continue
      for command in commands:
        source = Path(command["file"])
        if not identity or not source.is_file():
          continue
        text = source.read_text(errors="replace")
        # Different schemas may share a source file. Require the descriptor's
        # kernel name as well as its source path, not merely a matching include.
        if identity in text and json.dumps(self.name) in text:
          result.append({"command": command, "build_dir": str(command_file.parent)})
    return tuple(result)

  def get_assembly(self, specialization=None):
    from .machine import assembly

    return assembly(self, specialization)

  def export(self, directory, *, include_commands=False):
    import re

    for key in (*self._session.sources, *self._session.stages):
      if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError("debug source/stage names must be safe filename components")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
      raise FileExistsError("debug export requires an empty destination")
    for kind, source in self._session.sources.items():
      (destination / f"{kind}.{'py' if kind == 'python' else 'cpp'}").write_text(source)
    for stage, text in self._session.stages.items():
      (destination / f"{stage}.ir.txt").write_text(text)
    artifacts = self.artifacts()
    if not include_commands:
      artifacts = tuple({"source": a["command"]["file"], "build_dir": a["build_dir"]} for a in artifacts)
    for kind in ("adapter", "spec"):
      text = self.get_source(kind)
      if isinstance(text, str):
        (destination / f"{kind}.{'h' if kind == 'spec' else 'cpp'}").write_text(text)
    for index, child in enumerate(self.children):
      child.export(destination / f"child{index}", include_commands=include_commands)
    payload = {
      "format_version": self.format_version,
      "name": self.name,
      "cache": self._cache,
      "passes": self._session.passes,
      "diagnostics": self.explain(),
      "origins": self._session.origins,
      "artifacts": artifacts,
      "children": tuple({"name": child.name, "cache": child.get_cache_info()} for child in self.children),
    }
    (destination / "report.json").write_text(json.dumps(payload, indent=2, default=str))
    return destination


class InspectionMixin:
  def get_kernel_source(self):
    return self.debug.get_source("kernel")

  def get_host_source(self):
    return self.debug.get_source("adapter")

  def get_ir(self, stage="kernel"):
    return self.debug.get_ir(stage)

  def get_pass_report(self):
    return self.debug.get_pass_report()

  def get_cache_info(self):
    return self.debug.get_cache_info()

  def get_build_report(self):
    return self.debug.get_build_report()

  def explain(self, topic=None):
    return self.debug.explain(topic)

  def get_assembly(self, specialization=None):
    return self.debug.get_assembly(specialization)

  def analyze_assembly(self, specialization=None):
    from .machine import machine_report

    return machine_report(self.debug, specialization)

  def export_debug(self, directory, **kwargs):
    return self.debug.export(directory, **kwargs)

  def get_layout(self, *args, **kwargs):
    from .resources import layouts

    return layouts(self, args, kwargs)

  def interpret(self, *args, **kwargs):
    from .interpreter import interpret

    return interpret(self, args, kwargs)

  def analyze(self, *args, **kwargs):
    from .resources import analyze

    return analyze(self, args, kwargs)

  def profile(self, *args, **kwargs):
    from .profile import profile

    return profile(self, args, **kwargs)


def attach_composite(function):
  """Keep Python wrapper call semantics; expose frozen child-kernel reports."""
  import inspect

  children, seen = [], set()

  def visit(value):
    if id(value) in seen:
      return
    seen.add(id(value))
    if isinstance(value, InspectionMixin):
      children.append(value.debug)
    elif isinstance(value, (tuple, list)):
      for item in value:
        visit(item)
    elif isinstance(value, dict):
      for item in value.values():
        visit(item)
    elif inspect.isfunction(value):
      variables = inspect.getclosurevars(value)
      for item in (*variables.nonlocals.values(), *variables.globals.values()):
        if not inspect.ismodule(item):
          visit(item)

  visit(function)
  info = DebugInfo(name=getattr(function, "__qualname__", "wrapper"))
  info.children = tuple(children)
  info._validate_cache = lambda: all(child._validate_cache() for child in children)
  try:
    function.debug = info
    for name in (
      "get_kernel_source",
      "get_host_source",
      "get_ir",
      "get_pass_report",
      "get_cache_info",
      "get_assembly",
      "get_build_report",
      "explain",
      "export_debug",
    ):
      setattr(function, name, getattr(InspectionMixin, name).__get__(function))
  except (AttributeError, TypeError):
    # Callable objects with slots are not mutated; their decorator still
    # caches the original object, never a call-semantics-changing proxy.
    pass
  return info


class LoweredKernel(InspectionMixin):
  def __init__(self, template, artifact, session, compiler=None):
    self.template, self.artifact = template, artifact
    self.kernel_ir = artifact.kernel if artifact is not None else None
    self.vector_ir = artifact.lowered if artifact is not None else None
    self.backend_ir = artifact.backend_ir if artifact is not None else None
    self.debug = DebugInfo(session, name=template.ir.name, compiler=compiler)
    self.debug.owner = self
    if self.kernel_ir:
      self.debug.diagnostics = tuple(self.kernel_ir.diagnostics)


def lower(template, *, target="generic", out_idx=None, pipeline=None, pass_options=None, debug=None):
  from .._compiler.backends.cpu.backend import CPUBackend
  from .._runtime.binding import _normalize_out_idx
  from ..template import TemplateFunction

  if not isinstance(template, TemplateFunction):
    raise TypeError("T.lower expects a TemplateFunction")
  session = DebugSession(debug or DebugOptions())
  session.sources.update(template.debug._session.sources)
  token = current_session.set(session)
  try:
    if template.ir.source is None:
      options = {
        **dict(pass_options or {}),
        "_debug_session": session,
        "debug_prints": session.options.prints,
        "max_print_elements": session.options.max_print_elements,
        "debug_memory": session.options.memory or session.options.timeline,
        "debug_timeline": session.options.timeline,
      }
      artifact = CPUBackend().lower(
        template.ir,
        _normalize_out_idx(out_idx, tuple(template.ir.signature.parameters.values())),
        pipeline=pipeline,
        options=options,
        target=target,
      )
      session.sources["kernel"] = artifact.source
    else:
      artifact = None
      session.sources["kernel"] = template.ir.source.read_text()
    result = LoweredKernel(template, artifact, session)
    template.debug = result.debug
    return result
  except BaseException as error:
    error.debug_info = DebugInfo(session, name=template.ir.name)
    raise
  finally:
    current_session.reset(token)
