"""Kernel passes: reuse."""
from ... import ir as I
from ...ir.traversal import walk, children, values
from ..manager import register_pass

@register_pass("plan-computation-reuse", after=("expression-cse",))
def plan_computation_reuse(ir, context):
  expensive = {"exp", "exp_neg", "exp2", "exp2_neg", "log", "log2"}
  def visit(block):
    for index, statement in enumerate(block.statements):
      if isinstance(statement, I.Assign) and not any(e.op in {"where", "and", "or"} for e in walk(statement.value)):
        future = {id(e) for value in values(I.Block(block.statements[index + 1:])) for e in walk(value)}
        for reduction in walk(statement.value):
          if not reduction.op.startswith("reduce_"):
            continue
          source = reduction.args[0]
          if (id(source) not in future or source.type.dtype not in {"float32", "float64"}
              or not any(e.op in expensive for e in walk(source))
              or any(e.op in {"local", "frozen", "load"} or e.op.startswith("reduce_")
                     for extent in source.type.shape for e in walk(extent))):
            continue
          if id(source) not in ir.materializations:
            strides = [I.literal(1)] * len(source.type.shape)
            for axis in reversed(range(len(strides) - 1)):
              strides[axis] = I.binary("mul", strides[axis + 1], source.type.shape[axis + 1])
            ir.materializations[id(source)] = I.View(
                f"dsl_o_reuse_{len(ir.materializations)}", source.type.dtype,
                source.type.shape, tuple(strides), I.literal(0), owned=True)
      for child in children(statement):
        visit(child)
  visit(ir.body)
  return ir
