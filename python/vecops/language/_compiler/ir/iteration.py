# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Logical loop contracts; execution mapping and ISA representation are separate.

FoldLoop retains reduction identity and ordering until vector legalization.
Loop consumers use the same domain/state interface for explicit For loops;
neither a scan recurrence nor a resident-team barrier is an independent axis.
"""

from dataclasses import dataclass

from . import kernel as I
from .state import UpdateState
from .types import Node


@dataclass(frozen=True)
class IterationDomain(Node):
  extents: tuple
  starts: tuple = ()
  steps: tuple = ()
  axes: tuple[int, ...] = ()

  def matches(self, other):
    return (
      self.axes == other.axes
      and len(self.extents) == len(other.extents)
      and all(I.same_expr(a, b) for a, b in zip(self.extents, other.extents))
      and len(self.starts) == len(other.starts)
      and all(I.same_expr(a, b) for a, b in zip(self.starts, other.starts))
      and len(self.steps) == len(other.steps)
      and all(I.same_expr(a, b) for a, b in zip(self.steps, other.steps))
    )


@dataclass(frozen=True)
class ReductionSpec(Node):
  operation: str
  dtype: str
  # Builtins retain their established SIMD grouping, not ordered left-fold
  # semantics. Custom combiners opt in explicitly before being recognized.
  ordering: str = "builtin"
  associative: bool = True


@dataclass(frozen=True)
class LoopSchedule(Node):
  unroll: int = 2
  pipeline: str = "off"
  distance: int = 1


@dataclass(frozen=True)
class IterationContract:
  domain: IterationDomain
  schedule: LoopSchedule
  kind: str
  carried: frozenset
  reducers: tuple = ()
  ordered: bool = True
  task_parallel: bool = False


@dataclass(frozen=True)
class FoldLoop(I.ReductionGroup):
  """A logical reduction loop with explicit prologue/body/finalizer contracts.

  Native lowering consumes descriptors rather than scalarizing this region.
  The inherited bindings also keep compatibility with alias/snapshot planning.
  """

  @property
  def domain(self):
    source = self.assignments[0].value.args[0]
    return IterationDomain(source.type.shape, axes=tuple(range(len(source.type.shape))))

  @property
  def reducers(self):
    from ..operations import current

    return tuple(ReductionSpec(current().get(a.value).reduction, a.value.type.dtype) for a in self.assignments)

  @property
  def schedule(self):
    return LoopSchedule(self.assignments[0].value.data)


@dataclass(frozen=True, eq=False)
class AxisFold(Node):
  sources: tuple
  axis: int
  operations: tuple[str, ...]
  unroll: int

  @property
  def domain(self):
    return IterationDomain(self.sources[0].type.shape, axes=(self.axis,))


@dataclass(frozen=True)
class AxisResult(Node):
  fold: AxisFold
  index: int

  def __iter__(self):
    # Existing memory/layout planners can still consume the single-result
    # descriptor; vector emission consumes the shared fold as one loop.
    yield self.fold.axis
    yield self.fold.operations[self.index]
    yield self.fold.unroll


def domain(loop):
  if isinstance(loop, FoldLoop):
    return loop.domain
  if isinstance(loop, I.For):
    return IterationDomain((loop.stop,), (loop.start,), (loop.step,))
  if isinstance(loop, I.Grid):
    return IterationDomain(
      loop.extents,
      tuple(I.literal(0) for _ in loop.extents),
      tuple(I.literal(1) for _ in loop.extents),
      tuple(range(len(loop.extents))),
    )
  if isinstance(loop, I.UnrolledFor):
    return domain(loop.loop)
  from .pipeline import PipelineFor

  if isinstance(loop, PipelineFor):
    return domain(loop.loop)
  if isinstance(loop, AxisFold):
    return loop.domain
  from .schedule import ScheduleLoop

  if isinstance(loop, ScheduleLoop):
    return loop.domain
  return None


def carried_names(loop):
  """Explicit scalar/tile recurrence names, also for ordered custom folds."""
  from .traversal import nodes

  if not isinstance(loop, I.For):
    return frozenset()
  return frozenset(
    s.target if isinstance(s, I.Assign) else s.state.name
    for s in nodes(loop.body)
    if isinstance(s, I.Assign) and not s.declare or isinstance(s, UpdateState)
  )


def contract(loop):
  """Common legality contract, not permission to reassociate every loop.

  Prefix loops are ordered; worker/team collectives are synchronization
  boundaries, not iterations in this interface. ScheduleLoop is the backend
  specialization of the same domain/recurrence contract.
  """
  logical = domain(loop)
  if logical is None:
    return None
  if isinstance(loop, FoldLoop):
    return IterationContract(
      logical, loop.schedule, "fold", frozenset(a.target for a in loop.assignments), loop.reducers, False
    )
  if isinstance(loop, I.Grid):
    return IterationContract(logical, LoopSchedule(1), "grid", frozenset(), ordered=False, task_parallel=True)
  from .schedule import ScheduleLoop

  if isinstance(loop, ScheduleLoop):
    return IterationContract(
      logical,
      loop.schedule,
      "scheduled",
      frozenset(),
      (loop.recurrence,) if loop.recurrence else (),
      bool(loop.recurrence),
    )
  from .pipeline import PipelineFor

  if isinstance(loop, (PipelineFor, I.UnrolledFor)):
    base = contract(loop.loop)
    return IterationContract(
      logical,
      LoopSchedule(
        loop.loop.unroll,
        "staged"
        if isinstance(loop, PipelineFor) and loop.buffering
        else "prefetch"
        if isinstance(loop, PipelineFor)
        else "off",
      ),
      base.kind,
      base.carried,
      base.reducers,
      base.ordered,
    )
  if isinstance(loop, AxisFold):
    return IterationContract(logical, LoopSchedule(loop.unroll), "fold", frozenset(), ordered=False)
  return IterationContract(
    logical,
    LoopSchedule(loop.unroll),
    loop.kind,
    carried_names(loop),
    loop.reducers,
    ordered=loop.kind == "scan" or any(r.ordering == "sequential" for r in loop.reducers) or not loop.associative,
  )
