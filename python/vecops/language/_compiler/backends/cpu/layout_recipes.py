# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bind exact register-load geometry to ordinary source-level layout control."""

from dataclasses import replace

from ... import ir as I
from ...dtypes import ELEMENT_BYTES
from ...frontend.helpers import ValueFunction, expand_helper
from ...ir.traversal import children
from ...passes.manager import PassResult, register_pass
from .transpose import plan_transpose, small_vla_conversion


@register_pass("cpu-expand-layout-recipes", after="cpu-expand-packing-recipes", before="fuse-elementwise")
def expand_layout_recipes(ir, context):
  from ...._library.layout import register_transpose_into

  environment, serial = context.environment, 0
  target = environment.codegen_target
  native_bytes = (target.vector_bits or 128) // 8 if target.architecture == "x86" else 16

  def visit(block):
    nonlocal serial
    output = []
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      plan = plan_transpose(statement, context.options) if isinstance(statement, I.Store) else None
      if (
        plan is None
        or len(statement.destination.shape) != 2
        or target.isa == "scalar"
        or statement.implementation == "sme"
        or small_vla_conversion(plan, context.options)
        or (
          plan.sme
          and target.isa == "sme"
          and (statement.implementation == "sme" or statement.implementation == "auto" and plan.sme_preferred)
        )
      ):
        output.append(statement)
        continue
      serial += 1
      source = statement.value
      if source.op == "cast" and source.type.dtype == statement.destination.dtype:
        source = source.args[0]
      dtype = source.type.dtype
      columns = min(16, native_bytes // plan.lane_bytes)
      rows = min(columns, 8 if ELEMENT_BYTES[dtype] >= 4 else 16)
      row, column = (I.Expr(I.ScalarType("int64"), "region_argument", data=name) for name in ("row", "column"))
      typ = I.VectorType(dtype, (I.literal(columns),), tag=I.VectorTag(N=columns, basis_dtype=dtype))
      load_row = ValueFunction((row, column), I.Expr(typ, "value_transform", (source, column, row), ("slice", (0,))))
      _, body = expand_helper(
        register_transpose_into,
        (source, statement.destination, load_row),
        catalog=environment.operations,
        keyword_arguments={"rows": rows, "columns": columns, "dtype": dtype, "cache": plan.cache_block},
        prefix=f"dsl_layout_{serial}",
      )
      guard = I.binary("eq", statement.destination.strides[-1], I.literal(1))
      for load in I.loads_unique(source):
        guard = I.binary("and", guard, I.binary("eq", load.data.strides[plan.source_axis], I.literal(1)))
        guard = I.binary(
          "and", guard, I.Expr(I.ScalarType("bool"), "view_disjoint", data=(load.data, statement.destination))
        )
      # Alias or unsupported strides use the ordinary historical-read path,
      # not another transpose algorithm hidden in a native printer.
      fallback = replace(statement, implementation="gather")
      output.append(I.If(guard, I.Block(body), I.Block([fallback])))
    block.statements = output

  visit(ir.body)
  if serial:
    from ...passes.kernel.transforms import fold_value_transforms

    ir = fold_value_transforms(ir, context)
    ir.diagnostics.append(f"staged {serial} register transpose traversal(s) from DSL library")
  return PassResult(ir, changed=bool(serial))
