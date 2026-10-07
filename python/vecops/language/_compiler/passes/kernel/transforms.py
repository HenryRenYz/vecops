"""Push logical value transforms to memory when the proof is exact."""

from dataclasses import replace

from ... import ir as I
from ...ir.traversal import nodes, rewrite_expressions
from ...operations import Trait
from ..manager import register_pass


@register_pass("fold-value-transforms", after=("decompose-row-maps",), before=("canonicalize-memory",))
def fold_value_transforms(ir, context):
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
    if context.environment.operations.has(value, Trait.ELEMENTWISE) and all(
      isinstance(a.type, I.ScalarType)
      or len(a.type.shape) == len(value.type.shape)
      and all(I.same_expr(x, y) for x, y in zip(a.type.shape, value.type.shape))
      for a in value.args
    ):
      return I.Expr(typ, value.op, tuple(transform(a, kind, axes, shape, starts) for a in value.args), value.data)
    return I.Expr(typ, "value_transform", (value, *starts), (kind, axes))

  def rewrite(value):
    if value.op == "value_transform" and value.data[0] in {"reshape", "transpose", "slice", "expand_dims", "squeeze"}:
      return transform(value.args[0], *value.data, value.type.shape, value.args[1:])
    return value

  return rewrite_expressions(ir, rewrite)
