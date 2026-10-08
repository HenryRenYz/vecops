# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU resident teams, scalar shared slots and portable atomic operations."""

from .... import ir as I
from ....ir.traversal import children
from ..capabilities import CPP_TYPES


def team_statements(block):
  for statement in block.statements:
    yield statement
    for child in children(statement):
      yield from team_statements(child)


class CollectiveEmitter:
  def emit_team(self, statement):
    requested = self.unique("team_requested")
    workers = statement.name + "_workers"
    budget = self.unique("team_budget")
    self.line(f"const vecops::nint_t {requested} = {self.scalar(statement.workers)};")
    self.line(f'if ({requested}<0) throw std::invalid_argument("negative team worker count");')
    self.line(
      f"const auto {budget} = std::max<vecops::nint_t>(1, std::min(dsl_grid_capacity, dsl_grid_requested>0 ? dsl_grid_requested : dsl_grid_capacity));"
    )
    self.line(f"const auto {workers} = std::min({budget}, {requested}>0 ? {requested} : {budget});")
    for collective in team_statements(statement.body):
      if isinstance(collective, I.TeamReduce):
        self.line(f"vecops_dsl::Buffer<{CPP_TYPES[collective.value.type.dtype]}> {collective.target}_slots;")
        self.line(f"{collective.target}_slots.resize({workers});")
    self.opening(
      f"vecops::execution::grid_team(dsl_grid_pool, {workers}, [&](vecops::execution::GridTeam {statement.name}) VECOPS_INLINE_LAMBDA"
    )
    old_buffers = self.buffer_declarations
    old_active, old_snapshots = self.active_loads, self.snapshots
    old_materialized = self.materialized
    old_reductions, old_chunks = self.reduction_materialized, self.chunk_cache
    self.buffer_declarations, self.active_loads, self.snapshots, self.materialized = {}, {}, {}, {}
    self.reduction_materialized, self.chunk_cache = {}, {}
    insertion = len(self.lines)
    self.parallel_depth += 1
    self.block(statement.body)
    self.parallel_depth -= 1
    self.worker_buffers(insertion)
    self.buffer_declarations = old_buffers
    self.active_loads, self.snapshots, self.materialized = old_active, old_snapshots, old_materialized
    self.reduction_materialized, self.chunk_cache = old_reductions, old_chunks
    self.indent -= 1
    self.line("});")

  def team_reduce(self, statement):
    name = statement.team
    target = statement.target
    self.line(f"{target}_slots.data()[{name}.worker_id()] = {self.scalar(statement.value)};")
    self.line(f"{name}.barrier();")
    self.line(f"auto {target} = {target}_slots.data()[0];")
    worker = self.unique("collective_worker")
    self.opening(f"for (vecops::nint_t {worker}=1; {worker}<{name}.worker_count(); ++{worker})")
    operand = f"{target}_slots.data()[{worker}]"
    expression = (
      f"{target}+{operand}" if statement.operation == "add" else f"std::{statement.operation}({target}, {operand})"
    )
    self.line(f"{target} = {expression};")
    self.close()
    self.line(f"{name}.barrier(); /* no slot reuse before every reader finishes */")

  def atomic_scalar(self, value):
    view, order = value.data
    args = ", ".join(f"static_cast<{CPP_TYPES[view.dtype]}>({self.scalar(v)})" for v in value.args)
    return f"vecops::execution::{value.op}({self.pointer(view)}, {args}, std::memory_order_{order})"
