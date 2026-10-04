"""Emit planned grids, private buffers and dependency-aware resident teams."""

from dataclasses import replace
from .... import ir as I
from ....ir.traversal import values, walk
from ....operations import Trait


class GridEmitter:
  def grid_guard(self, statement):
    plan = self.prepared.grids[id(statement)]
    if not plan.safe or statement.parallel == "serial":
      return "false"
    guards = []
    for guard in plan.guards:
      arguments = [self.scalar_inline(guard.base), self.scalar_inline(guard.width), str(guard.factor)]
      if guard.programs is not None:
        arguments += [self.scalar_inline(guard.programs), self.scalar_inline(guard.extent), str(guard.clipped).lower()]
      guards.append(f"vecops_dsl::grid_axis_safe({','.join(arguments)})")
    for storage in sorted(plan.writes):
      root = self.storage_views[storage]
      shape = ", ".join(self.scalar(v) for v in root.shape)
      strides = ", ".join(self.scalar(v) for v in root.strides)
      guards.append(f"vecops_dsl::grid_layout_safe<{len(root.shape)}>({{{shape}}}, {{{strides}}})")
      for other in sorted((plan.reads | plan.writes) - {storage}):
        guards.append(f"!({self.overlap(root, self.storage_views[other])})")
    return " && ".join(dict.fromkeys(guards)) or "true"

  def grid_metadata(self, statement):
    shape = ", ".join(self.scalar(v) for v in statement.extents)
    count, workers = self.unique("programs"), self.unique("workers")
    self.line(f"const auto {count}=vecops_dsl::grid_count({{{shape}}});")
    plan = self.prepared.grids[id(statement)]
    overhead = self.environment.costs.policy("grid_worker_cycles", 2000)
    self.line(
      f"const auto {workers}=vecops::execution::grid_workers_available(dsl_grid_capacity,dsl_grid_requested,{count},"
      f"static_cast<double>({self.scalar_inline(plan.cycles)}),{overhead});"
    )
    return count, workers

  def ordered_grid(self, statement):
    for name, extent in zip(statement.targets, statement.extents):
      self.opening(f"for (vecops::nint_t {name}=0; {name}<({self.scalar_inline(extent)}); ++{name})")
    self.parallel_depth += 1
    self.block(statement.body)
    self.parallel_depth -= 1
    for _ in statement.targets:
      self.close()

  def grid_program(self, statement, program):
    remainder = program
    for axis in reversed(range(len(statement.targets))):
      extent = self.scalar_inline(statement.extents[axis])
      name = statement.targets[axis]
      coordinate = remainder if axis == 0 else f"({remainder}) % ({extent})"
      self.line(f"const vecops::nint_t {name}=({coordinate});")
      remainder = f"(({remainder}) / ({extent}))"
    self.parallel_depth += 1
    previous_proof = self.verified_grid
    self.verified_grid = self.prepared.grids[id(statement)]
    self.block(statement.body)
    self.verified_grid = previous_proof
    self.parallel_depth -= 1

  def worker_buffers(self, insertion):
    # Hoisted allocations belong to this worker callback, never its parent
    # kernel. A worker can safely reuse them across its sequential programs.
    declarations = list(self.buffer_declarations.values())
    self.lines[insertion:insertion] = ["  " * self.indent + line.strip() for line in declarations]

  def private_materializations(self, bodies):
    produced = {
      id(e.args[0])
      for body in bodies
      for root in values(body)
      for e in walk(root)
      if self.environment.operations.has(e, Trait.SCALAR_REDUCTION)
    }
    from ..capabilities import CPP_TYPES

    for identity, view in self.ir.materializations.items():
      if identity in produced and identity not in self.materialized:
        self.line(f"vecops_dsl::Buffer<{CPP_TYPES[view.dtype]}> {view.storage}; /* worker-private reuse */")

  def emit_grid(self, statement):
    self.protect_region(statement.body, {})
    if self.parallel_depth or statement.parallel == "serial" or not self.prepared.grids[id(statement)].safe:
      self.ordered_grid(statement)
      return
    count, workers = self.grid_metadata(statement)
    guard = self.grid_guard(statement)
    self.opening(f"if ({count}>0 && ({guard})) /* verified grid ownership */")
    program, lane, cursor = self.unique("program"), self.unique("lane"), self.unique("cursor")
    self.line(f"std::atomic<vecops::nint_t> {cursor}{{0}};")
    self.opening(
      f"vecops::execution::grid_lanes(dsl_grid_pool,{workers},"
      f"[&](vecops::execution::ParallelContext {lane}) VECOPS_INLINE_LAMBDA"
    )
    old = self.buffer_declarations
    self.buffer_declarations = {}
    insertion = len(self.lines)
    self.private_materializations((statement.body,))
    self.opening(
      f"vecops::execution::grid_programs({count},{lane}.thread_id,{lane}.num_threads,"
      f"{'true' if statement.schedule == 'dynamic' else 'false'},{statement.chunk},{cursor},"
      f"[&](vecops::nint_t {program}) VECOPS_INLINE_LAMBDA"
    )
    self.grid_program(statement, program)
    self.worker_buffers(insertion)
    self.indent -= 1
    self.line("});")
    self.buffer_declarations = old
    self.indent -= 1
    self.line("});")
    self.close()
    self.opening("else /* alias/layout guard retains ordered execution */")
    if statement.parallel == "require":
      self.line(f'if ({count}>0) throw std::invalid_argument("required parallel grid failed its alias/layout guard");')
    self.ordered_grid(statement)
    self.close()

  def grid_region(self, phases):
    grids = [p for p in phases if isinstance(p, I.Grid)]
    if len(grids) < 2 or self.parallel_depth:
      for phase in phases:
        self.emit_grid(phase)
      return
    names = {id(grid): self.grid_metadata(grid) for grid in grids}
    guards = " && ".join(f"({self.grid_guard(g)})" for g in grids)
    maximum = "std::max<vecops::nint_t>({" + ",".join(names[id(g)][1] for g in grids) + "})"
    team = self.unique("team")
    for phase in phases:
      if isinstance(phase, I.Assign) and phase.declare:
        from ..capabilities import CPP_TYPES

        self.line(f"{CPP_TYPES[phase.value.type.dtype]} {phase.target}{{}};")
    self.opening(f"if (({guards}) && vecops::execution::supports_grid_team(dsl_grid_pool)) /* shared grid team */")
    cursors = {}
    for grid in grids:
      cursor = self.unique("cursor")
      self.line(f"std::atomic<vecops::nint_t> {cursor}{{0}};")
      cursors[id(grid)] = cursor
    self.opening(
      f"vecops::execution::grid_team(dsl_grid_pool,{maximum},[&](vecops::execution::GridTeam {team}) VECOPS_INLINE_LAMBDA"
    )
    old = self.buffer_declarations
    self.buffer_declarations = {}
    insertion = len(self.lines)
    self.private_materializations(tuple(g.body for g in grids))
    pending_read, pending_write = set(), set()
    for phase in phases:
      if isinstance(phase, I.Assign):
        self.line(f"{team}.barrier();")
        self.opening(f"if ({team}.worker_id()==0)")
        self.block(I.Block([replace(phase, declare=False)]))
        self.close()
        self.line(f"{team}.barrier(); /* publish single-worker scalar */")
        pending_read.clear()
        pending_write.clear()
        continue
      plan = self.prepared.grids[id(phase)]
      hazards = []
      for writer, others in ((pending_write, plan.reads | plan.writes), (plan.writes, pending_read)):
        for a in sorted(writer):
          for b in sorted(others):
            hazards.append("true" if a == b else self.overlap(self.storage_views[a], self.storage_views[b]))
      if "true" in hazards:
        self.line(f"{team}.barrier(); /* RAW/WAR/WAW phase edge */")
        # A definite barrier completes every earlier phase. Keeping their
        # effects pending would add redundant edges for later read-only reuse.
        pending_read.clear()
        pending_write.clear()
      elif hazards:
        self.line(f"if ({' || '.join(dict.fromkeys(hazards))}) {team}.barrier(); /* RAW/WAR/WAW phase edge */")
      count = names[id(phase)][0]
      program = self.unique("program")
      self.opening(
        f"vecops::execution::grid_programs({count},{team}.worker_id(),{team}.worker_count(),"
        f"{'true' if phase.schedule == 'dynamic' else 'false'},{phase.chunk},{cursors[id(phase)]},"
        f"[&](vecops::nint_t {program}) VECOPS_INLINE_LAMBDA"
      )
      self.grid_program(phase, program)
      self.indent -= 1
      self.line("});")
      pending_read.update(plan.reads)
      pending_write.update(plan.writes)
    self.worker_buffers(insertion)
    self.buffer_declarations = old
    self.indent -= 1
    self.line("});")
    self.close()
    self.opening("else /* logical-task-only executor or unsafe region */")
    for phase in phases:
      if isinstance(phase, I.Grid):
        self.emit_grid(phase)
      else:
        self.block(I.Block([replace(phase, declare=False)]))
    self.close()
