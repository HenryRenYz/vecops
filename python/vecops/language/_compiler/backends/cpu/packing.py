# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Checked ABI binding for source-level packing algorithms, without loop printing."""

from dataclasses import replace

from ... import ir as I
from ...frontend.helpers import expand_helper
from ...ir.traversal import children, nodes
from ...passes.manager import PassResult, register_pass


@register_pass("cpu-expand-packing-recipes", after="cpu-expand-dot-recipes", before="fuse-elementwise")
def expand_packing_recipes(ir, context):
  from ...._library.packing import pack_amx, pack_generic, pack_sme

  environment, serial = context.environment, 0
  bounded = context.options.get("cpu_pack_bounded", False)
  if type(bounded) is not bool:
    raise TypeError("cpu_pack_bounded must be bool")
  statements = tuple(nodes(ir.body))
  write_views = tuple(
    statement.destination.view if isinstance(statement, I.Pack) else statement.destination
    for statement in statements
    if isinstance(statement, (I.Pack, I.Store, I.IndexedStore))
  )
  # Root regions contain invocation metadata, unlike future slices whose
  # offsets can refer to loop/branch locals unavailable at this pack site.
  writes = tuple(
    replace(view, shape=view.region.root_shape, strides=view.region.root_strides,
            offset=I.literal(0), region=None)
    for view in write_views if not view.owned and view.region is not None
  )
  opaque_hardware = any(isinstance(statement, I.ArchitectureCall) for statement in statements)
  unknown_writes = any(not view.owned and view.region is None for view in write_views)

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
      target = environment.codegen_target
      format_ = packed.format
      if format_ == "auto":
        if target.isa == "sme":
          format_ = "sme"
        elif target.architecture == "x86" and target.isa != "scalar" and view.dtype in {"int8", "uint8", "float16", "bfloat16", "int16", "uint16"}:
          format_ = "amx"
        else:
          format_ = "dense"
      native = format_ == "sme" and target.isa == "sme" or format_ == "amx" and target.architecture == "x86" and target.isa != "scalar" and view.dtype in {"int8", "uint8", "bfloat16", "float16", "int16", "uint16"}
      helper = pack_sme if native and format_ == "sme" else pack_amx if native else pack_generic
      keywords = {"side": packed.side} if native else {"side": packed.side, "format": packed.format}
      if native:
        keywords["raw"] = (
          source.data
          if source.op == "load" and not source.args and source.type.dtype == source.data.dtype
          else None
        )
        if format_ == "sme":
          keywords["bounded"] = bounded
          # A raw View loses a captured load's historical identity. Bypass
          # its DAG only when no kernel write can have changed that backing.
          raw = keywords["raw"]
          safe = I.literal(not opaque_hardware and not unknown_writes)
          if raw is not None:
            if raw.owned and any(view.storage == raw.storage for view in write_views):
              safe = I.literal(False)
            for destination in writes:
              safe = I.binary("and", safe, I.Expr(I.ScalarType("bool"), "view_disjoint", data=(raw, destination)))
          keywords["raw_safe"] = safe
      _, generic = expand_helper(
        helper,
        (source, flat),
        keyword_arguments=keywords,
        catalog=environment.operations,
        prefix=f"dsl_pack_{serial}",
      )
      output.extend(generic)
    block.statements = output

  visit(ir.body)
  if serial:
    ir.diagnostics.append(f"staged {serial} packed-layout recipe(s) from DSL library")
  return PassResult(ir, changed=bool(serial))
