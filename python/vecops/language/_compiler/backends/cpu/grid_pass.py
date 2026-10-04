"""Registered CPU grid planning, independent of C++ printing."""

from dataclasses import replace
from types import MappingProxyType
from ... import ir as I
from ...ir.traversal import children
from ...passes.manager import register_pass
from .grid import plan_grid, plan_regions


@register_pass("cpu-plan-grids", stage="cpu-memory", after=("cpu-plan-matmul",))
def plan_grids(prepared, context):
  plans, blocks = {}, []
  mode = context.options.get("cpu_grid_regions", "auto")
  if mode not in {"auto", "individual", "adjacent", "template"}:
    raise ValueError("cpu_grid_regions must be auto, individual, adjacent or template")

  def visit(block):
    blocks.append(block)
    for statement in block.statements:
      if isinstance(statement, I.Grid):
        plan = plan_grid(statement, context.environment)
        if not plan.safe:
          prepared.vectors.kernel.diagnostics.append("grid retained serial order: " + plan.reason)
          if statement.parallel == "require":
            raise TypeError("required parallel grid is unsafe: " + plan.reason)
        plans[id(statement)] = plan
      for child in children(statement):
        visit(child)

  visit(prepared.vectors.kernel.body)
  regions = {
    key: phases for block in blocks for key, phases in plan_regions(block, plans, mode, context.environment).items()
  }
  return replace(prepared, grids=MappingProxyType(plans), grid_regions=MappingProxyType(regions))
