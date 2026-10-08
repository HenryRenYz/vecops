# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Specialize source-parsed library leaves before ordinary kernel/vector passes.

Adapters select a measured legal region, not its arithmetic or instruction
schedule. The recipe's control, state and producer/consumer DAGs then use the
same optimization/legalization pipeline as user-written DSL code.
"""

from dataclasses import replace

from ... import ir as I
from ...dtypes import ELEMENT_BYTES
from ...frontend.helpers import expand_helper
from ...ir.traversal import children, nodes, walk_unique
from ...operations import Trait
from ...passes.manager import PassResult, register_pass


@register_pass("cpu-expand-dot-recipes", after="orient-matmul", before="fuse-elementwise")
def expand_dot_recipes(ir, context):
  enabled = context.options.get("cpu_vector_dot_dsl", False)
  if type(enabled) is not bool:
    raise TypeError("cpu_vector_dot_dsl must be bool")
  if not enabled:
    return PassResult(ir, changed=False)
  from ...._library.matmul import vector_tile_dot

  environment = context.environment
  target = environment.codegen_target
  vector_state = environment.vector_state
  native_bytes = vector_state.sve_bytes or 16 if target.isa in {"sve", "sme"} else (target.vector_bits or 128) // 8
  budget = context.options.get("cpu_state_register_budget", 12)
  serial, rewritten = 0, set()

  def eligible(product):
    if (
      product.data.tiled or product.data.implementation != "vector" or product.type.dtype not in {"float32", "float64"}
    ):
      return False
    if any(value.op in {"dot", "packed_load", "state_ref"} for arg in product.args for value in walk_unique(arg)):
      return False
    m, n = map(I.constant, product.type.shape)
    widest = max(
      ELEMENT_BYTES[value.type.dtype] for value in walk_unique(product) if isinstance(value.type, I.TileType)
    )
    lanes = max(1, native_bytes // widest)
    return type(m) is int and 0 < m <= 4 and type(n) is int and 0 < n <= lanes and m * n.bit_count() <= budget

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
          if eligible(product) and epilogue and consumers[id(product)] == {id(statement)}:
            serial += 1
            result, body = expand_helper(
              vector_tile_dot,
              (product.args[:2],),
              keyword_arguments={
                "acc": product.args[2] if len(product.args) > 2 else None,
                "acc_dtype": product.type.dtype,
              },
              catalog=environment.operations,
              prefix=f"dsl_recipe_dot_{serial}",
            )
            output.extend(body)
            statement = replace(statement, value=replace_product(statement.value, product, result, {}))
            rewritten.add(id(product))
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
