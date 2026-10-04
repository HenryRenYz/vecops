"""Compilation-local trace collection; no global hooks or mutable IR snapshots."""

import difflib
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

from .options import DebugOptions
from .printer import format_ir

current_session = ContextVar("vecops_debug_session", default=None)


@dataclass
class DebugSession:
  options: DebugOptions = field(default_factory=DebugOptions)
  stages: dict = field(default_factory=dict)
  passes: list = field(default_factory=list)
  messages: list = field(default_factory=list)
  sources: dict = field(default_factory=dict)
  origins: dict = field(default_factory=dict)

  def stage(self, name, ir):
    self.stages[name] = format_ir(ir)

  def begin_pass(self, name, ir):
    enabled = self.options.trace != "off" and (not self.options.passes or name in self.options.passes)
    before = format_ir(ir) if enabled else None
    return time.perf_counter(), before

  def end_pass(self, stage, name, ir, begin, error=None):
    start, before = begin
    elapsed = time.perf_counter() - start
    after = format_ir(ir) if before is not None else None
    entry = {"stage": stage, "name": name, "seconds": elapsed, "status": "failed" if error else "ok"}
    if before is not None:
      entry["changed"] = before != after
      if self.options.trace == "all" or before != after or error:
        entry.update(
          before=before,
          after=after,
          diff="".join(
            difflib.unified_diff(
              before.splitlines(True), after.splitlines(True), fromfile=f"{name}:before", tofile=f"{name}:after"
            )
          ),
        )
    if error:
      entry["error"] = f"{type(error).__name__}: {error}"
    self.passes.append(entry)
