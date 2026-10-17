# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Push logical value transforms to memory when the proof is exact."""

import re
from dataclasses import replace

from ... import ir as I
from ...analysis.effects import ValueEffects, assigned_scalars
from ...ir.traversal import nodes, rewrite_expressions
from ...operations import Trait
from ..manager import PassResult, register_pass


@register_pass("fold-value-transforms", after=("decompose-row-maps",), before=("canonicalize-memory",))
def fold_value_transforms(ir, context):
  changed = False
  effects = ValueEffects(context.environment.operations)
  assigned = assigned_scalars(ir.body)
  stable = {}

  def stable_extent(value):
    if value not in stable:
      stable[value] = False
      if effects.discardable(value):
        if value.op == "literal":
          stable[value] = True
        elif value.op == "symbol" and value.data in ir.bindings:
          stable[value] = stable_extent(ir.bindings[value.data])
        elif value.op == "raw" and type(value.data) is str:
          match = re.fullmatch(r"([A-Za-z_]\w*)(?:\.(?:size|stride)\(\d+\))?", value.data)
          stable[value] = match is not None and match[1] in ir.parameters and match[1] not in assigned
    return stable[value]

  # A later window of a previously bound value must still select that value's
  # historical snapshot. Its view may depend on indices defined only later,
  # so moving the projected load into the old binding is not generally legal.
  captured = {
    id(load)
    for statement in nodes(ir.body)
    if isinstance(statement, I.BindTile)
    for load in I.loads_unique(statement.value)
  }

  def transform(value, kind, axes, shape, starts=()):
    if isinstance(value.type, I.ScalarType):
      return value
    typ = I.TileType(value.type.dtype, shape, value.type.mask)
    if value.op == "load" and not value.args and id(value) not in captured:
      view = value.data
      if kind == "transpose":
        return I.Expr(
          typ, "load", data=replace(view, shape=shape, strides=tuple(view.strides[a] for a in axes), region=None)
        )
      if kind == "slice":
        offset = view.offset
        for start, stride in zip(starts, view.strides):
          offset = I.binary("add", offset, I.binary("mul", start, stride))
        return I.Expr(
          typ,
          "load",
          data=replace(view, shape=shape, strides=tuple(view.strides[a] for a in axes), offset=offset, region=None),
        )
      if kind == "expand_dims":
        strides = view.strides[:axes] + (I.literal(0),) + view.strides[axes:]
        return I.Expr(typ, "load", data=replace(view, shape=shape, strides=strides, region=None))
      if kind == "squeeze":
        strides = view.strides[:axes] + view.strides[axes + 1 :]
        return I.Expr(typ, "load", data=replace(view, shape=shape, strides=strides, region=None))
      if kind != "reshape":
        return I.Expr(typ, "value_transform", (value, *starts), (kind, axes))
      expected = I.literal(1)
      for extent, stride in zip(reversed(view.shape), reversed(view.strides)):
        if not I.same_expr(stride, expected):
          break
        expected = I.binary("mul", expected, extent)
      else:
        strides, expected = [None] * len(shape), I.literal(1)
        for a in reversed(range(len(shape))):
          strides[a], expected = expected, I.binary("mul", expected, shape[a])
        return I.Expr(typ, "load", data=replace(view, shape=shape, strides=tuple(strides), region=None))
    if value.op == "fill":
      return replace(value, type=typ)
    # Index generators depend on the coordinate system, not just their scalar
    # operands. Retain the transform so its coordinate map updates the index
    # axis/start; simply retagging arange corrupts slices/transposes/reshapes.
    if (
      value.op != "arange"
      and context.environment.operations.has(value, Trait.ELEMENTWISE)
      and all(
        isinstance(a.type, I.ScalarType)
        or len(a.type.shape) == len(value.type.shape)
        and all(I.same_expr(x, y) for x, y in zip(a.type.shape, value.type.shape))
        for a in value.args
      )
    ):
      return I.Expr(typ, value.op, tuple(transform(a, kind, axes, shape, starts) for a in value.args), value.data)
    return I.Expr(typ, "value_transform", (value, *starts), (kind, axes))

  def rewrite(value):
    nonlocal changed
    if value.op == "value_transform" and value.data[0] in {"reshape", "transpose", "slice", "expand_dims", "squeeze"}:
      source = value.args[0]
      if (
        value.data[0] == "reshape"
        and type(source.type) is I.TileType
        and type(value.type) is I.TileType
        and source.type.dtype == value.type.dtype
        and source.type.mask == value.type.mask
      ):
        # Reshaping only unit axes is an exact coordinate projection even for
        # an already captured read. Keep that read object rather than folding
        # it into a new memory access (which would lose its historical value).
        def unit(extent):
          return type(I.constant(extent)) is int and I.constant(extent) == 1

        old = tuple(n for n in source.type.shape if not unit(n))
        new = tuple(n for n in value.type.shape if not unit(n))
        if len(old) == len(new) and all(
          a is b or I.same_extent(a, b) and stable_extent(a) and stable_extent(b) for a, b in zip(old, new)
        ):
          result = source
          for axis in reversed(range(len(source.type.shape))):
            if unit(source.type.shape[axis]):
              shape = result.type.shape[:axis] + result.type.shape[axis + 1 :]
              result = I.Expr(
                I.TileType(value.type.dtype, shape, value.type.mask), "value_transform", (result,), ("squeeze", axis)
              )
          for axis, extent in enumerate(value.type.shape):
            if unit(extent):
              shape = result.type.shape[:axis] + (extent,) + result.type.shape[axis:]
              result = I.Expr(
                I.TileType(value.type.dtype, shape, value.type.mask),
                "value_transform",
                (result,),
                ("expand_dims", axis),
              )
          changed = True
          return result
      result = transform(value.args[0], *value.data, value.type.shape, value.args[1:])
      if (
        result.op == value.op
        and result.data == value.data
        and result.type == value.type
        and len(result.args) == len(value.args)
        and all(a is b for a, b in zip(result.args, value.args))
      ):
        return value
      changed = True
      return result
    return value

  return PassResult(rewrite_expressions(ir, rewrite), changed=changed)
