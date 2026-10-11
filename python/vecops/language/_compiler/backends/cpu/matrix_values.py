# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Stage non-pointwise/shared matrix results before source algorithm expansion.

This is a value-lifetime transformation, not an operator implementation.
Branch-local results never escape; original read identities survive staging.
"""

from dataclasses import replace

from ... import ir as I
from ...ir.traversal import children, nodes, walk_unique
from ...operations import Trait
from ...passes.manager import PassResult, register_pass


@register_pass("cpu-stage-matrix-values", after="orient-matmul", before="cpu-expand-dot-recipes")
def stage_matrix_values(ir, context):
  consumers, serial = {}, 0
  catalog = context.environment.operations
  for statement in nodes(ir.body):
    if isinstance(statement, I.BindTile):
      continue
    for name in ("value", "source", "mask", "condition"):
      value = getattr(statement, name, None)
      if isinstance(value, I.Expr):
        for product in walk_unique(value):
          if product.op == "dot":
            consumers.setdefault(id(product), set()).add(id(statement))

  def visit(block):
    nonlocal serial
    output, saved = [], {}
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      if isinstance(statement, I.BindTile):
        output.append(statement)
        continue
      updates, memo, retained = {}, {}, set()

      def rewrite(value, memo=memo, retained=retained):
        nonlocal serial
        if id(value) in saved:
          return saved[id(value)]
        if id(value) in memo:
          return memo[id(value)]
        arguments = tuple(rewrite(argument) for argument in value.args)
        result = replace(value, args=arguments) if any(a is not b for a, b in zip(arguments, value.args)) else value
        if value.op == "dot" and id(value) not in retained:
          serial += 1
          view = I.View(
            f"dsl_matrix_value_{serial}",
            result.type.dtype,
            result.type.shape,
            (result.type.shape[1], I.literal(1)),
            I.literal(0),
            owned=True,
            region=I.ViewRegion(result.type.shape, (result.type.shape[1], I.literal(1)),
                                (I.literal(0), I.literal(0)), result.type.shape,
                                (1, 1), (0, 1), (False, False)),
          )
          output.extend((I.Allocate(view), I.Store(view, result)))
          saved[id(value)] = result = I.Expr(result.type, "load", data=view)
        memo[id(value)] = result
        return result

      if isinstance(statement, I.Store) and statement.mask is None:
        products = {id(e): e for e in walk_unique(statement.value) if e.op == "dot"}
        if len(products) == 1:
          product = next(iter(products.values()))
          operands = {id(e) for a in product.args for e in walk_unique(a)}
          if consumers[id(product)] == {id(statement)} and all(
            e is product or isinstance(e.type, I.ScalarType) or catalog.has(e, Trait.ELEMENTWISE | Trait.MEMORY_READ)
            for e in walk_unique(statement.value)
            if id(e) not in operands
          ):
            retained.add(id(product))
      for name in ("value", "source", "mask", "condition"):
        value = getattr(statement, name, None)
        if isinstance(value, I.Expr):
          updates[name] = rewrite(value)
      output.append(replace(statement, **updates) if updates else statement)
    block.statements = output

  visit(ir.body)
  if serial:
    ir.diagnostics.append(f"staged {serial} shared/non-pointwise matrix value(s)")
  return PassResult(ir, changed=bool(serial))
