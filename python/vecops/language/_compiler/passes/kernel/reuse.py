"""Kernel passes: reuse."""

from ... import ir as I
from ...cost import estimate
from ...ir.traversal import children, values, walk
from ...operations import Trait
from ..manager import register_pass


@register_pass("plan-computation-reuse", after=("expression-cse",))
def plan_computation_reuse(ir, context):
  environment = context.environment
  catalog, costs = environment.operations, environment.costs

  def visit(block):
    for index, statement in enumerate(block.statements):
      bindings = (
        statement.assignments
        if isinstance(statement, I.ReductionGroup)
        else (statement,)
        if isinstance(statement, I.Assign)
        else ()
      )
      for binding in bindings:
        if any(catalog.has(e, Trait.CONDITIONAL) for e in walk(binding.value)):
          continue
        future = {id(e) for value in values(I.Block(block.statements[index + 1 :])) for e in walk(value)}
        for reduction in walk(binding.value):
          if not catalog.has(reduction, Trait.SCALAR_REDUCTION):
            continue
          source = reduction.args[0]
          if (
            id(source) not in future
            or any(catalog.has(e, Trait.STATE_REFERENCE) for e in walk(source))
            or source.type.dtype not in {"float32", "float64"}
            or estimate(source, costs, boundaries=ir.materializations).compute_cycles
            < costs.policy("reuse_compute_budget", 6)
            or any(catalog.has(e, Trait.UNSTABLE_EXTENT) for extent in source.type.shape for e in walk(extent))
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
