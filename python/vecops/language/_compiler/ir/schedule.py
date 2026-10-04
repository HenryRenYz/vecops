"""Backend schedules expressed before source emission.

Parameters are bound by the backend printer (VL, native tile width and runtime
cache budgets), not C++ snippets embedded in optimization passes. A schedule
retains loop domains, recurrence and pipeline decisions for inspection and
target-specific transformation. ISA actions remain backend expansion points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .iteration import IterationDomain, LoopSchedule, ReductionSpec
from .kernel import Expr, literal
from .types import Node, ScalarType


def parameter(name):
  return Expr(ScalarType("int64"), "schedule_parameter", data=name)


@dataclass(frozen=True)
class ScheduleAction(Node):
  name: str
  effects: str


@dataclass(frozen=True)
class ScheduleLoop(Node):
  role: str
  domain: IterationDomain
  body: tuple
  schedule: LoopSchedule = field(default_factory=LoopSchedule)
  recurrence: ReductionSpec | None = None
  # full-tiles tests remaining extent >= step; nonempty retains K=0 seed and
  # epilogue behavior of the cache driver. Neither changes the reduction order.
  bound: str = "extent"
  prologue: tuple = ()
  finalization: tuple = ()


@dataclass(frozen=True)
class MatmulSchedule(Node):
  regions: tuple
  pipeline: str
  prefetch_lines: int

  def loop(self, role):
    def find(items):
      for item in items:
        if isinstance(item, ScheduleLoop):
          if item.role == role:
            return item
          result = find(item.body)
          if result is not None:
            return result
      return None

    result = find(self.regions)
    if result is None:
      raise KeyError(f"unplanned matrix loop: {role}")
    return result


def matrix_loop(role, axis, step, body, *, start=None, extent=None, schedule=None, recurrence=None, bound="extent"):
  return ScheduleLoop(
    role,
    IterationDomain(
      (parameter(extent or axis),),
      (literal(0) if start is None else parameter(start),),
      (parameter(step),),
    ),
    tuple(body),
    schedule or LoopSchedule(unroll=1),
    recurrence,
    bound,
  )
