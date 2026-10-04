"""Shape-only transforms and axis coordinates, independent of execution layout."""

from dataclasses import replace

from .. import ir as I


class ShapesFrontend:
  def shape_operation(self, operation, args, kwargs, node):
    if operation == "indices":
      if len(args) != 1:
        self.error(node, "indices expects a shape and keyword axis")
      shape = self.shape(args[0], node)
      axis = I.constant(self.value(kwargs.get("axis", -1), node))
      if not shape or type(axis) is not int or not -len(shape) <= axis < len(shape):
        self.error(node, "indices axis must be a compile-time in-range integer")
      for extent in shape:
        if not isinstance(extent.type, I.ScalarType) or "int" not in extent.type.dtype:
          self.error(node, "indices extents must be scalar integers")
        self.block.statements.append(I.Assert(I.binary("ge", extent, I.literal(0)), "negative index extent"))
      return I.Expr(I.TileType("int64", shape), "arange", (I.literal(0),), axis % len(shape))
    if operation not in {"expand_dims", "squeeze"}:
      return NotImplemented
    if not 1 <= len(args) <= 2 or len(args) == 2 and "axis" in kwargs:
      self.error(node, "shape transform expects a value and axis")
    value = args[0]
    if not isinstance(value, I.View) and not (isinstance(value, I.Expr) and isinstance(value.type, I.TileType)):
      self.error(node, "shape transform expects a view or tile value")
    shape = value.shape if isinstance(value, I.View) else value.type.shape
    bound = len(shape) + (operation == "expand_dims")
    axis = I.constant(self.value(kwargs.get("axis", args[1] if len(args) == 2 else -1), node))
    if type(axis) is not int or not -bound <= axis < bound:
      self.error(node, "shape transform axis must be a compile-time in-range integer")
    axis %= bound
    if operation == "expand_dims":
      output = shape[:axis] + (I.literal(1),) + shape[axis:]
    else:
      self.block.statements.append(I.Assert(I.binary("eq", shape[axis], I.literal(1)), "squeeze axis is not one"))
      output = shape[:axis] + shape[axis + 1 :]
    if isinstance(value, I.View):
      strides = (
        value.strides[:axis] + (I.literal(0),) + value.strides[axis:]
        if operation == "expand_dims"
        else value.strides[:axis] + value.strides[axis + 1 :]
      )
      return replace(value, shape=output, strides=strides, region=None)
    value = self.retain_register_transform(value)
    if not output:
      return I.Expr(I.ScalarType("bool" if value.type.mask else value.type.dtype), "element_at", (value, I.literal(0)))
    return I.Expr(I.TileType(value.type.dtype, output, value.type.mask), "value_transform", (value,), (operation, axis))
