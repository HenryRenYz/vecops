"""Shape-only transforms and axis coordinates, independent of execution layout."""

from dataclasses import replace

from ...dsl import Vector
from .. import ir as I
from ..dtypes import dtype_name


class ShapesFrontend:
  def fill_value(self, operation, args, kwargs, node):
    """Fill a logical tile or explicit register geometry without a dummy load.

    An optional active prefix changes logical shape, never physical tag width.
    This lets a carried vector retain its geometry at a dynamic memory tail.
    """
    limits = (1, 2) if operation == "zeros" else (2, 3)
    if not limits[0] <= len(args) <= limits[1]:
      self.error(node, "zeros/full expects a shape or Vector, scalar fill and optional dtype")
    explicit_dtype = kwargs.get("dtype", args[-1] if len(args) == limits[1] else None)
    value = self.value(args[1] if operation == "full" else I.literal(0), node)
    if not isinstance(value.type, I.ScalarType):
      self.error(node, "full requires a scalar fill value; use broadcast_to for tile values")
    descriptor = args[0]
    if not isinstance(descriptor, Vector):
      if "active" in kwargs:
        self.error(node, "active prefix requires an explicit Vector descriptor")
      shape = self.shape(descriptor, node)
      dtype = dtype_name("float32" if explicit_dtype is None else explicit_dtype)
      return I.Expr(I.TileType(dtype, shape), "fill", (value,))
    dtype = dtype_name(descriptor.dtype)
    if dtype == "bool":
      self.error(node, "vector fills require a numeric Vector descriptor")
    if explicit_dtype is not None and dtype_name(explicit_dtype) != dtype:
      self.error(node, "fill dtype conflicts with its Vector descriptor")
    extent = (
      I.literal(descriptor.N)
      if descriptor.N is not None
      else I.Expr(I.ScalarType("int64"), "lanes", data=(dtype, descriptor.pow))
    )
    active = self.value(kwargs.get("active", extent), node)
    if not isinstance(active.type, I.ScalarType) or "int" not in active.type.dtype:
      self.error(node, "vector fill active prefix must be a scalar integer")
    valid = I.binary("and", I.binary("ge", active, I.literal(0)), I.binary("le", active, extent))
    if I.constant(valid) is False:
      self.error(node, "vector fill active prefix exceeds its geometry")
    self.block.statements.append(I.Assert(valid, "vector fill active prefix exceeds its geometry"))
    typ = I.VectorType(dtype, (active,), tag=I.VectorTag(descriptor.N, descriptor.pow, basis_dtype=dtype))
    return I.Expr(typ, "fill", (value,))

  def shape_operation(self, operation, args, kwargs, node):
    if operation == "as_vector":
      if len(args) != 2 or not isinstance(args[1], Vector):
        self.error(node, "as_vector expects a rank-one value and Vector descriptor")
      value, descriptor = args
      value = self.matrix_operand(value, node, allow_scalar=True)
      if (
        not isinstance(value.type, I.TileType)
        or len(value.type.shape) != 1
        or value.type.mask
        or value.type.dtype == "bool"
      ):
        self.error(node, "as_vector expects a numeric rank-one value")
      dtype = dtype_name(descriptor.dtype)
      if dtype == "bool":
        self.error(node, "as_vector requires a numeric Vector descriptor")
      extent = (
        I.literal(descriptor.N)
        if descriptor.N is not None
        else I.Expr(I.ScalarType("int64"), "lanes", data=(dtype, descriptor.pow))
      )
      shape = (I.binary("min", value.type.shape[0], extent),)
      if not I.same_expr(shape[0], value.type.shape[0]):
        value = I.Expr(I.TileType(value.type.dtype, shape), "value_transform", (value, I.literal(0)), ("slice", (0,)))
      if value.type.dtype != dtype:
        value = I.Expr(I.TileType(dtype, shape), "cast", (value,))
      typ = I.VectorType(dtype, shape, tag=I.VectorTag(descriptor.N, descriptor.pow, basis_dtype=dtype))
      return I.Expr(typ, "identity", (value,))
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
