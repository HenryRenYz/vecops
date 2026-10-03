"""C++ emission: control."""

from __future__ import annotations
import json
from .... import ir as I
from ..capabilities import CPP_TYPES


class ControlEmitter:
  def elementwise_group(self, group, future):
    body = I.Block(list(group.stores))
    self.protect_region(body, future)
    from ....ir.traversal import walk

    if any(
      id(store) in self.prepared.transposes or any(e.op == "dot" for e in walk(store.value)) for store in group.stores
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
    for index, store in enumerate(group.stores):
      own_loads = list(I.loads(store.value)) + (list(I.loads(store.mask)) if store.mask is not None else [])
      for later in group.stores[index + 1 :]:
        guards.append(f"!({self.overlap(store.destination, later.destination)})")
        later_loads = list(I.loads(later.value)) + (list(I.loads(later.mask)) if later.mask is not None else [])
        guards.extend(f"!({self.overlap(store.destination, load.data)})" for load in later_loads)
        guards.extend(f"!({self.overlap(later.destination, load.data)})" for load in own_loads)
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
    for store in self.stores(loop.body):
      output = self.storage_views.get(store.destination.storage)
      if output is None:
        guards.append("false")
        continue
      for load in dependencies.values():
        source = self.storage_views.get(load.data.storage)
        guards.append(f"!({self.overlap(source, output)})" if source is not None else "false")
    self.opening("if (" + (" && ".join(dict.fromkeys(guards)) or "true") + ") /* operation-major serial unroll */")
    self.line(f"vecops::nint_t {loop.target} = {self.scalar(loop.start)};")
    step = I.constant(loop.step)
    self.opening(
      f"for (; vecops_dsl::full_group({loop.target}, {self.scalar_inline(loop.stop)}, {step}, {loop.unroll}); {loop.target} += {loop.unroll * step})"
    )
    self.block(statement.copies)
    self.close()
    comparison = "<" if step > 0 else ">"
    self.opening(f"for (; {loop.target} {comparison} ({self.scalar_inline(loop.stop)}); {loop.target} += {step})")
    self.block(loop.body)
    self.close()
    self.close()
    self.opening("else /* alias-dependent iterations retain their original order */")
    self.block(I.Block([loop]))
    self.close()

  def block(self, block):
    before_active, before_snapshots = dict(self.active_loads), dict(self.snapshots)
    before_materialized = dict(self.materialized)
    for index, statement in enumerate(block.statements):
      if isinstance(statement, I.HardwareRegion):
        future = self.expression_loads(I.Block(block.statements[index + 1 :]))
        self.protect_region(statement.body, future)
        self.hardware_region(statement, future)
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
        comparison = "<" if I.constant(statement.step) > 0 else ">"
        self.opening(
          f"for (vecops::nint_t {statement.target} = {self.scalar(statement.start)}; {statement.target} {comparison} ({self.scalar_inline(statement.stop)}); {statement.target} += {self.scalar(statement.step)})"
        )
        self.block(statement.body)
        self.close()
      elif isinstance(statement, I.If):
        self.protect_region(I.Block([statement]), self.expression_loads(I.Block(block.statements[index + 1 :])))
        self.opening(f"if ({self.scalar(statement.condition)})")
        self.block(statement.body)
        self.close()
        if statement.otherwise.statements:
          self.opening("else")
          self.block(statement.otherwise)
          self.close()
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
      elif isinstance(statement, I.Allocate):
        self.storage_views[statement.view.storage] = statement.view
        # Promoted scratch sites have no Store/Load using their storage.
        if statement.view.storage in self.used_storage(self.ir.body):
          count = ", ".join(self.scalar(v) for v in statement.view.shape)
          self.allocate_buffer(statement.view.dtype, statement.view.storage, f"vecops_dsl::checked_shape({{{count}}})")
      elif isinstance(statement, I.Pack):
        self.pack_matrix(statement)
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
