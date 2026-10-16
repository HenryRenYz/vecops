# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""C++ emission: control."""

from __future__ import annotations

import json

from .... import ir as I
from ....analysis.effects import read_accesses
from .memory import scoped_extent_geometry


class ControlEmitter:
  @scoped_extent_geometry
  def elementwise_group(self, group, future):
    body = I.Block(list(group.stores))
    self.protect_region(body, future)
    if self.register_group_store(group, future):
      return
    from ....ir.traversal import walk

    if any(
      id(store) in self.prepared.transposes
      or id(store) in self.prepared.register_programs
      or any(e.op == "dot" for e in walk(store.value))
      for store in group.stores
    ):
      # A generic streaming fusion would bypass the prepared permutation DAG.
      # Keep each register transpose intact until multi-output transpose
      # fusion has its own layout-aware plan.
      self.block(body)
      return
    if not group.stores[0].destination.shape:
      self.block(body)
      return
    guards = []
    accesses = [
      read_accesses((store.value, store.mask) if store.mask is not None else store.value, self.environment.operations)
      for store in group.stores
    ]
    if any(reads is None for reads in accesses):
      # Conditional/opaque read addresses must not be evaluated speculatively
      # to decide whether the stores can share one vector traversal.
      self.block(body)
      return
    for index, store in enumerate(group.stores):
      own_loads = accesses[index]
      for later_index, later in enumerate(group.stores[index + 1 :], index + 1):
        guards.append(f"!({self.overlap(store.destination, later.destination)})")
        guards.extend(f"!({self.overlap(store.destination, load.view)})" for load in accesses[later_index])
        guards.extend(f"!({self.overlap(later.destination, load.view)})" for load in own_loads)
    self.opening("if (" + (" && ".join(guards) or "true") + ") /* independent elementwise fusion */")
    old_snapshots = dict(self.snapshots)
    for store in group.stores:
      self.protect_store(store, future)
      self.check_store(store)
    plan = self.vector_ir.plans[id(group)]
    self.traverse_region(
      plan, lambda tags, coords, active, roots, state: self.write_effects(plan, tags, coords, active, roots)
    )
    self.snapshots = old_snapshots
    self.close()
    self.opening("else /* preserve sequential effects for overlapping operands */")
    self.block(body)
    self.close()

  def unrolled_for(self, statement, future):
    loop = statement.loop
    self.protect_region(loop.body, future)
    dependencies = self.expression_loads(loop.body)
    guards = []
    old_proofs = getattr(self, "disjoint_roots", {})
    proofs = dict(old_proofs)

    def root(view):
      if view.region is None:
        return view
      return I.View(view.storage, view.dtype, view.region.root_shape, view.region.root_strides, I.literal(0))

    for store in self.stores(loop.body):
      output = self.storage_views.get(store.destination.storage)
      if output is None:
        guards.append("false")
        continue
      for load in dependencies.values():
        source = self.storage_views.get(load.data.storage)
        guards.append(f"!({self.overlap(source, output)})" if source is not None else "false")
        if source is not None:
          proofs[tuple(sorted((load.data.storage, store.destination.storage)))] = (
            root(load.data),
            root(store.destination),
          )
    self.opening("if (" + (" && ".join(dict.fromkeys(guards)) or "true") + ") /* operation-major serial unroll */")
    self.disjoint_roots = proofs
    self.line(f"vecops::nint_t {loop.target} = {self.scalar(loop.start)};")
    step = I.constant(loop.step)
    rendered_step = str(step) if step is not None else self.scalar(loop.step)
    increment = str(loop.unroll * step) if step is not None else f"{loop.unroll}*({rendered_step})"
    condition = (
      f"{loop.target}<({self.scalar_inline(loop.stop)}) && (({self.scalar_inline(loop.stop)})-{loop.target})/({rendered_step})>={loop.unroll}"
      if statement.full_vectors
      else f"vecops_dsl::full_group({loop.target}, {self.scalar_inline(loop.stop)}, {rendered_step}, {loop.unroll})"
    )
    self.opening(f"for (; {condition}; {loop.target} += {increment})")
    self.block(statement.copies)
    self.close()
    comparison = "<" if step is None or step > 0 else ">"
    self.opening(
      f"for (; {loop.target} {comparison} ({self.scalar_inline(loop.stop)}); {loop.target} += {rendered_step})"
    )
    self.block(loop.body)
    self.close()
    self.close()
    self.disjoint_roots = old_proofs
    self.opening("else /* alias-dependent iterations retain their original order */")
    self.block(I.Block([loop]))
    self.close()

  def block(self, block):
    before_active, before_snapshots = dict(self.active_loads), dict(self.snapshots)
    before_materialized = dict(self.materialized)
    consumed = set()
    for index, statement in enumerate(block.statements):
      if index in consumed:
        continue
      if isinstance(statement, I.Team):
        self.emit_team(statement)
      elif isinstance(statement, I.Print):
        self.emit_print(statement)
      elif isinstance(statement, I.Barrier):
        self.line(f"{statement.team}.barrier();")
      elif isinstance(statement, I.TeamReduce):
        self.team_reduce(statement)
      elif isinstance(statement, I.Prefetch):
        self.prefetch(statement)
      elif isinstance(statement, I.Grid):
        phases = self.prepared.grid_regions[id(statement)] if not self.parallel_depth else (statement,)
        consumed.update(range(index + 1, index + len(phases)))
        self.protect_region(I.Block(phases), self.expression_loads(I.Block(block.statements[index + len(phases) :])))
        self.grid_region(phases)
      elif isinstance(statement, I.HardwareRegion):
        future = self.expression_loads(I.Block(block.statements[index + 1 :]))
        self.protect_region(statement.body, future)
        self.hardware_region(statement, future)
      elif isinstance(statement, (I.DeclareState, I.UpdateState)):
        self.emit_state(statement)
      elif isinstance(statement, I.ExitRegion):
        self.protect_region(statement.body, self.expression_loads(I.Block(block.statements[index + 1 :])))
        if statement.outlined:
          plan = self.prepared.private_captures.get(id(statement))
          captures = []
          if plan is not None and self.hardware_scope is None and not self.hardware_scalar_bindings:
            captures.extend(plan.scalars)
            captures.extend(self.parameter_metadata[name] for name in plan.metadata if name in self.parameter_metadata)
            captures.extend(self.parameter_pointers[name] for name in plan.pointers if name in self.parameter_pointers)
            # protect_region has already established these const bool
            # conditions. Resolve planned read identities, never recompute a
            # guard or copy its resizable snapshot Buffer owner.
            captures.extend(self.snapshots[id(read)][1] for read in plan.snapshot_reads if id(read) in self.snapshots)
          # The plan names existing immutable values only. Owners, mutable
          # results and any unplanned emission binding stay by reference.
          capture = ", ".join(("&", *dict.fromkeys(captures)))
          self.opening(f"vecops_dsl::private_call([{capture}]()")
        self.opening("/* inlined helper exit region */")
        self.block(statement.body)
        self.close()
        self.line(f"{statement.name}: ;")
        if statement.outlined:
          self.close()
          self.lines[-1] += ");"
      elif isinstance(statement, I.Exit):
        self.line(f"goto {statement.region};")
      elif isinstance(statement, I.Assign):
        value = self.scalar(statement.value)
        self.line(f"{'auto ' if statement.declare else ''}{statement.target} = {value};")
        self.materialized.update(self.reduction_materialized)
        self.reduction_materialized.clear()
      elif isinstance(statement, I.ReductionGroup):
        future = self.expression_loads(I.Block(block.statements[index + 1 :]))
        for store in statement.stores:
          # Protect every reduction input, not just the side output's operand.
          combined = I.Expr(
            store.value.type,
            "fusion_dependencies",
            tuple(a.value.args[0] for a in statement.assignments) + (store.value,),
          )
          self.protect_store(I.Store(store.destination, combined, store.mask), future)
        name = self.unique("reductions")
        reduced = self.reductions(
          tuple(a.value for a in statement.assignments),
          plan=self.vector_ir.plans.get(id(statement)),
          stores=statement.stores,
        )
        # A single reduction with an independent side output still returns a
        # scalar; multi-reductions use a tuple.
        self.line(f"auto {name} = {reduced};")
        for axis, assignment in enumerate(statement.assignments):
          value = f"std::get<{axis}>({name})" if len(statement.assignments) > 1 else name
          self.line(f"{'auto ' if assignment.declare else ''}{assignment.target} = {value};")
        self.materialized.update(self.reduction_materialized)
        self.reduction_materialized.clear()
      elif isinstance(statement, I.ElementwiseGroup):
        future = self.expression_loads(I.Block(block.statements[index + 1 :]))
        self.elementwise_group(statement, future)
      elif isinstance(statement, I.UnrolledFor):
        self.unrolled_for(statement, self.expression_loads(I.Block(block.statements[index + 1 :])))
      elif isinstance(statement, I.For):
        self.protect_region(statement.body, self.expression_loads(I.Block(block.statements[index + 1 :])))
        ordered_unroll = self.vector_ir.manager.options.get("cpu_ordered_loop_unroll", False)
        if type(ordered_unroll) is not bool:
          raise TypeError("cpu_ordered_loop_unroll must be bool")
        if ordered_unroll and statement.unroll > 1:
          # The operation-major pass kept this recurrence ordered. Native
          # unrolling may still duplicate complete iterations without moving
          # one iteration's state/effects ahead of another iteration's input.
          self.line("#if defined(__clang__)")
          self.line(f"#pragma clang loop unroll_count({statement.unroll})")
          self.line("#elif defined(__GNUC__)")
          self.line(f"#pragma GCC unroll {statement.unroll}")
          self.line("#endif")
        comparison = "<" if I.constant(statement.step) is None or I.constant(statement.step) > 0 else ">"
        self.opening(
          f"for (vecops::nint_t {statement.target} = {self.scalar(statement.start)}; {statement.target} {comparison} ({self.scalar_inline(statement.stop)}); {statement.target} += {self.scalar(statement.step)})"
        )
        self.block(statement.body)
        self.close()
      elif isinstance(statement, I.If):
        self.protect_region(I.Block([statement]), self.expression_loads(I.Block(block.statements[index + 1 :])))
        feature = statement.condition.data if statement.condition.op == "compile_feature" else None
        if feature:
          self.line(f"#if {feature}")
          self.opening("/* target instruction recipe */")
        else:
          self.opening(f"if ({self.scalar(statement.condition)})")
        old_proofs = getattr(self, "disjoint_roots", {})
        proofs = dict(old_proofs)

        def collect(condition, proofs=proofs):
          if condition.op == "and":
            for operand in condition.args:
              collect(operand)
          elif condition.op == "view_disjoint":
            a, b = condition.data
            proofs[tuple(sorted((a.storage, b.storage)))] = (a, b)

        collect(statement.condition)
        self.disjoint_roots = proofs
        self.branch_block(statement.body)
        self.disjoint_roots = old_proofs
        self.close()
        if statement.otherwise.statements:
          if feature:
            self.line("#else")
            self.opening("/* target instruction fallback */")
          else:
            self.opening("else")
          self.branch_block(statement.otherwise)
          self.close()
        if feature:
          self.line("#endif")
      elif isinstance(statement, I.While):
        self.protect_region(statement.body, self.expression_loads(I.Block(block.statements[index + 1 :])))
        if self.scalar_has_statements(statement.condition):
          self.opening("while (true)")
          condition = self.scalar(statement.condition)
          self.line(f"if (!({condition})) break;")
        else:
          self.opening(f"while ({self.scalar(statement.condition)})")
        self.block(statement.body)
        self.close()
      elif isinstance(statement, I.IndexedStore):
        self.indexed_store(statement, self.expression_loads(I.Block(block.statements[index + 1 :])))
      elif isinstance(statement, I.Store):
        future = self.expression_loads(I.Block(block.statements[index + 1 :]))
        self.protect_store(statement, future)
        nonempty = " && ".join(f"({self.scalar(n)}) > 0" for n in statement.destination.shape)
        if nonempty:
          self.opening(f"if ({nonempty})")
        self.store(statement)
        if nonempty:
          self.close()
      elif isinstance(statement, I.BindTile):
        self.active_loads.update({id(load): load for load in I.loads(statement.value)})
      elif isinstance(statement, I.WorkspaceAllocate):
        self.storage_views[statement.view.storage] = statement.view
        shape = ", ".join(self.scalar(v) for v in statement.view.shape)
        self.line(f"(void){self.checked_shape(shape)};")
        from ..capabilities import CPP_TYPES

        self.line(
          f'auto {statement.view.storage} = dsl_workspace.tensor<{CPP_TYPES[statement.view.dtype]}>("{statement.view.storage}", vecops::tensor::make_shape({shape}));'
        )
      elif isinstance(statement, I.Allocate):
        self.storage_views[statement.view.storage] = statement.view
        # Promoted scratch sites have no Store/Load using their storage.
        if statement.view.storage in self.used_storage(self.ir.body):
          count = ", ".join(self.scalar(v) for v in statement.view.shape)
          self.allocate_buffer(statement.view.dtype, statement.view.storage, self.checked_shape(count))
      elif isinstance(statement, I.Pack):
        raise TypeError("packing requires the cpu-expand-packing-recipes pass")
      elif isinstance(statement, I.Assert):
        self.line(
          f"if (!({self.scalar(statement.condition)})) throw std::invalid_argument({json.dumps(statement.message)});"
        )
      elif isinstance(statement, I.Control):
        self.line(statement.kind + ";")
      else:
        raise TypeError(f"unhandled IR statement {type(statement).__name__}")
    self.active_loads, self.snapshots = before_active, before_snapshots
    self.materialized = before_materialized
