# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Kernel passes: reuse."""

from ... import ir as I
from ...cost import estimate
from ...ir.traversal import children, values, walk_unique
from ...operations import Trait
from ..manager import register_pass


def _unconditionally_evaluated_reductions(expression, catalog):
  """Walk reductions without crossing a scalar short-circuit boundary.

  Vector selects evaluate their operands in the same lane traversal and are
  safe *inside* a reduction input. Scalar selects/logical operations can
  leave their operand reductions unexecuted, so do not plan caches for those
  reductions. Track guard state separately for DAG nodes shared by both
  unconditional and conditional paths.
  """
  seen = set()

  def visit(value, guarded=False):
    key = (id(value), guarded)
    if key in seen:
      return
    seen.add(key)
    if not guarded and catalog.has(value, Trait.SCALAR_REDUCTION):
      yield value
    conditional = isinstance(value.type, I.ScalarType) and catalog.has(value, Trait.CONDITIONAL)
    for index, argument in enumerate(value.args):
      yield from visit(argument, guarded or conditional and index > 0)

  yield from visit(expression)


@register_pass("plan-computation-reuse", after=("expression-cse",))
def plan_computation_reuse(ir, context):
  # Every invocation owns a fresh plan. Rewrites/unrolling can replace value
  # identities; a pipeline can repeat this planner without a separate refresh
  # shim or retaining stale materialization boundaries.
  ir.materializations.clear()
  environment = context.environment
  catalog, costs = environment.operations, environment.costs

  def visit(block):
    # Reverse ownership traversal computes the last lexical use of the shared
    # expression DAG once. Re-scanning the whole suffix at every binding was
    # quadratic for long generated recipe blocks.
    last_use = {}
    for index in reversed(range(len(block.statements))):
      pending = list(values(I.Block([block.statements[index]])))
      while pending:
        value = pending.pop()
        if id(value) in last_use:
          continue
        last_use[id(value)] = index
        pending.extend(value.args)
    for index, statement in enumerate(block.statements):
      bindings = (
        statement.assignments
        if isinstance(statement, I.ReductionGroup)
        else (statement,)
        if isinstance(statement, I.Assign)
        else ()
      )
      for binding in bindings:
        for reduction in _unconditionally_evaluated_reductions(binding.value, catalog):
          source = reduction.args[0]
          if (
            last_use.get(id(source), -1) <= index
            or any(catalog.has(e, Trait.STATE_REFERENCE) for e in walk_unique(source))
            or source.type.dtype not in {"float32", "float64"}
            or estimate(source, costs, boundaries=ir.materializations).compute_cycles
            < costs.policy("reuse_compute_budget", 6)
            or any(catalog.has(e, Trait.UNSTABLE_EXTENT) for extent in source.type.shape for e in walk_unique(extent))
          ):
            continue
          if id(source) not in ir.materializations:
            strides = [I.literal(1)] * len(source.type.shape)
            for axis in reversed(range(len(strides) - 1)):
              strides[axis] = I.binary("mul", strides[axis + 1], source.type.shape[axis + 1])
            ir.materializations[id(source)] = I.View(
              f"dsl_o_reuse_{len(ir.materializations)}",
              source.type.dtype,
              source.type.shape,
              tuple(strides),
              I.literal(0),
              owned=True,
            )
      for child in children(statement):
        visit(child)

  visit(ir.body)
  return ir
