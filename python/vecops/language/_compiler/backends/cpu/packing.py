# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Checked ABI binding for source-level packing algorithms, without loop printing."""

from dataclasses import replace

from ... import ir as I
from ...frontend.helpers import expand_helper
from ...ir.traversal import children
from ...passes.manager import PassResult, register_pass


@register_pass("cpu-expand-packing-recipes", after="cpu-expand-dot-recipes", before="fuse-elementwise")
def expand_packing_recipes(ir, context):
  from ...._library.packing import pack_generic

  environment, serial = context.environment, 0

  def visit(block):
    nonlocal serial
    output = []
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      if not isinstance(statement, I.Pack):
        output.append(statement)
        continue
      serial += 1
      source, packed = statement.source, statement.destination
      view = packed.view
      spatial, k = packed.shape if packed.side == "A" else packed.shape[::-1]
      count = I.Expr(I.ScalarType("int64"), "packing_size", (spatial, k), (view.dtype, packed.format))
      capacity = I.Expr(I.ScalarType("int64"), "checked_shape", view.shape)
      output.append(I.Assert(I.binary("ge", capacity, count), "packed output buffer too small"))
      expected = I.literal(1)
      for extent, stride in zip(reversed(view.shape), reversed(view.strides)):
        output.append(I.Assert(I.binary("eq", stride, expected), "packed output must be dense"))
        expected = I.binary("mul", expected, extent)
      # Bind only the required prefix. Excess caller capacity is untouched.
      flat = replace(view, shape=(count,), strides=(I.literal(1),), region=None)
      _, generic = expand_helper(
        pack_generic,
        (source, flat),
        keyword_arguments={"side": packed.side, "format": packed.format},
        catalog=environment.operations,
        prefix=f"dsl_pack_{serial}",
      )
      output.extend(generic)
    block.statements = output

  visit(ir.body)
  if serial:
    ir.diagnostics.append(f"staged {serial} packed-layout recipe(s) from DSL library")
  return PassResult(ir, changed=bool(serial))
