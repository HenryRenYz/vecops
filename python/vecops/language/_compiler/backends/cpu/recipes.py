# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Specialize source-parsed library leaves before ordinary kernel/vector passes.

Adapters select a measured legal region, not its arithmetic or instruction
schedule. The recipe's control, state and producer/consumer DAGs then use the
same optimization/legalization pipeline as user-written DSL code.
"""

from dataclasses import replace

from ... import ir as I
from ...analysis.projection import pointwise_row_projection
from ...cost import estimate
from ...dtypes import ELEMENT_BYTES
from ...frontend.helpers import ValueFunction, expand_helper
from ...ir.traversal import children, nodes, walk_unique
from ...operations import Trait
from ...passes.manager import PassResult, register_pass
from .policies import recompute_limit


@register_pass("cpu-expand-dot-recipes", after="orient-matmul", before="fuse-elementwise")
def expand_dot_recipes(ir, context):
  from ...._library.matmul import vector_dot_into, vector_tile_dot

  environment = context.environment
  target = environment.codegen_target
  vector_state = environment.vector_state
  native_bytes = vector_state.sve_bytes or 16 if target.isa in {"sve", "sme"} else (target.vector_bits or 128) // 8
  budget = context.options.get("cpu_state_register_budget", 12)
  serial, rewritten = 0, set()
  fusion = context.options.get("cpu_matmul_fusion", "auto")
  if fusion not in {"auto", "always", "never"}:
    raise ValueError("cpu_matmul_fusion must be auto, always or never")

  def input_policy(product, ordinal, prefix):
    expression = product.args[ordinal]
    cost = estimate(expression, environment.costs, boundaries=ir.materializations).compute_cycles
    if not cost or fusion == "always":
      return expression, []
    condition = (
      I.literal(True)
      if fusion == "never"
      else I.binary("gt", product.type.shape[1 - ordinal], I.literal(int(recompute_limit(cost, environment.costs))))
    )
    if I.constant(condition) is False:
      return expression, []
    shape = expression.type.shape
    view = I.View(
      prefix + "_input_cache", expression.type.dtype, shape, (shape[1], I.literal(1)), I.literal(0), owned=True
    )
    body = I.Block([I.Allocate(view), I.Store(view, expression)])
    if I.constant(condition) is True:
      return I.Expr(expression.type, "load", data=view), body.statements
    flag = prefix + "_cache_enabled"
    enabled = I.Expr(I.ScalarType("bool"), "local", data=flag)
    return I.Expr(expression.type, "cached_value", (expression,), (view, enabled)), [
      I.Assign(flag, condition),
      I.If(enabled, body, I.Block([])),
    ]

  def eligible(product):
    vector_family = product.data.implementation == "vector" or (
      product.data.implementation == "auto"
      and target.architecture == "x86"
      and all(arg.type.dtype == product.type.dtype for arg in product.args[:2])
    )
    if product.data.tiled or not vector_family or product.type.dtype not in {"float32", "float64"}:
      return False
    if any(value.op in {"dot", "packed_load", "state_ref"} for arg in product.args for value in walk_unique(arg)):
      return False
    m, n = map(I.constant, product.type.shape)
    widest = max(
      ELEMENT_BYTES[value.type.dtype] for value in walk_unique(product) if isinstance(value.type, I.TileType)
    )
    lanes = max(1, native_bytes // widest)
    return (
      "tile"
      if type(m) is int and 0 < m <= 4 and type(n) is int and 0 < n <= lanes and m * n.bit_count() <= budget
      else "streaming"
    )

  def replace_product(value, product, result, memo):
    if value is product:
      return result
    if id(value) not in memo:
      args = tuple(replace_product(arg, product, result, memo) for arg in value.args)
      memo[id(value)] = replace(value, args=args) if any(a is not b for a, b in zip(args, value.args)) else value
    return memo[id(value)]

  # A register result is scoped to its sink. Do not replace a product consumed
  # by another sink or a nonlocal/state/conditional consumer with that local.
  consumers = {}
  for statement in nodes(ir.body):
    if isinstance(statement, I.BindTile):
      continue
    for name in ("value", "source", "condition", "mask"):
      value = getattr(statement, name, None)
      if isinstance(value, I.Expr):
        for product in walk_unique(value):
          if product.op == "dot":
            consumers.setdefault(id(product), set()).add(id(statement))

  def visit(block):
    nonlocal serial
    output = []
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      if isinstance(statement, I.Store) and statement.mask is None:
        products = {id(value): value for value in walk_unique(statement.value) if value.op == "dot"}
        if len(products) == 1:
          product = next(iter(products.values()))
          operands = {id(value) for arg in product.args for value in walk_unique(arg)}
          epilogue = all(
            value is product
            or environment.operations.has(value, Trait.ELEMENTWISE | Trait.MEMORY_READ)
            or isinstance(value.type, I.ScalarType)
            for value in walk_unique(statement.value)
            if id(value) not in operands
          )
          family = eligible(product)
          if family and epilogue and consumers[id(product)] == {id(statement)}:
            serial += 1
            kwargs = {"acc": product.args[2] if len(product.args) > 2 else None, "acc_dtype": product.type.dtype}
            arguments = (product.args[:2],)
            if family == "streaming":
              inputs = []
              for ordinal in range(2):
                operand, preamble = input_policy(product, ordinal, f"dsl_recipe_dot_{serial}_{ordinal}")
                output.extend(preamble)
                inputs.append(operand)
              arguments = (tuple(inputs),)
              primary = next(
                (
                  load
                  for load in I.loads_unique(product.args[1])
                  if len(load.type.shape) == 2
                  and all(I.same_expr(a, b) for a, b in zip(load.type.shape, product.args[1].type.shape))
                ),
                None,
              )
              kwargs["prefer_k"] = (
                I.literal(False) if primary is None else I.binary("eq", primary.data.strides[0], I.literal(1))
              )
              extent = I.Expr(I.ScalarType("int64"), "region_argument", data="prefix")
              reference = I.Expr(
                I.VectorType(product.type.dtype, (extent,), tag=I.VectorTag(basis_dtype=product.type.dtype)),
                "region_argument",
                data="result",
              )
              row, column = (I.Expr(I.ScalarType("int64"), "region_argument", data=name) for name in ("row", "column"))
              kwargs["epilogue"] = ValueFunction(
                (reference, row, column),
                pointwise_row_projection(
                  statement.value, product, reference, row, column, extent, environment.operations
                ),
              )
              arguments += (statement.destination,)
            result, body = expand_helper(
              vector_tile_dot if family == "tile" else vector_dot_into,
              arguments,
              keyword_arguments=kwargs,
              catalog=environment.operations,
              prefix=f"dsl_recipe_dot_{serial}",
            )
            output.extend(body)
            rewritten.add(id(product))
            if family == "streaming":
              continue
            statement = replace(statement, value=replace_product(statement.value, product, result, {}))
      output.append(statement)
    block.statements = output

  visit(ir.body)

  # Keep original lazy-load provenance, but discard unused opaque products.
  def provenance(block):
    output = []
    for statement in block.statements:
      for child in children(statement):
        provenance(child)
      if isinstance(statement, I.BindTile) and any(id(value) in rewritten for value in walk_unique(statement.value)):
        output.extend(I.BindTile(load) for load in I.loads_unique(statement.value))
      else:
        output.append(statement)
    block.statements = output

  provenance(ir.body)
  if rewritten:
    # This adapter follows epilogue discovery, after the first layout fold.
    # Normalize newly staged windows with that same generic pass, rather than
    # teach the terminal emitter a recipe-specific projection algorithm.
    from ...passes.kernel.transforms import fold_value_transforms

    ir = fold_value_transforms(ir, context)
    ir.diagnostics.append(f"staged {len(rewritten)} vector dot leaf/leaves from DSL library")
  return PassResult(ir, changed=bool(rewritten))
