"""Push logical value transforms to memory when the proof is exact."""

from dataclasses import replace

from ... import ir as I
from ...ir.traversal import rewrite_expressions
from ...operations import Trait
from ..manager import register_pass


@register_pass("fold-value-transforms", after=("decompose-row-normalization",), before=("canonicalize-memory",))
def fold_value_transforms(ir, context):
  def transform(value, kind, axes, shape):
    if isinstance(value.type, I.ScalarType):
      return value
    typ = I.TileType(value.type.dtype, shape, value.type.mask)
    if value.op == "load" and not value.args:
      view = value.data
      if kind == "transpose":
        return I.Expr(
          typ, "load", data=replace(view, shape=shape, strides=tuple(view.strides[a] for a in axes), region=None)
        )
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
    if context.environment.operations.has(value, Trait.ELEMENTWISE) and all(
      isinstance(a.type, I.ScalarType)
      or len(a.type.shape) == len(value.type.shape)
      and all(I.same_expr(x, y) for x, y in zip(a.type.shape, value.type.shape))
      for a in value.args
    ):
      return I.Expr(typ, value.op, tuple(transform(a, kind, axes, shape) for a in value.args), value.data)
    return I.Expr(typ, "value_transform", (value,), (kind, axes))

  def rewrite(value):
    if value.op == "value_transform" and value.data[0] in {"reshape", "transpose"}:
      return transform(value.args[0], *value.data, value.type.shape)
    return value

  return rewrite_expressions(ir, rewrite)
