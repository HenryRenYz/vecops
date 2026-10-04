"""Value transforms and explicit indexed memory, independent of CPU tags."""

from dataclasses import replace

from .. import ir as I
from ..dtypes import dtype_name
from ..ir.traversal import walk
from ..operations import Trait


class PrimitivesFrontend:
  def retain_register_transform(self, value):
    if any(self.operations.has(e, Trait.REGISTER_TRANSFORM) for e in walk(value)):
      # Word-local semantics must not change when a later consumer extracts
      # one lane or traverses a different axis. Evaluate once in its own layout.
      return self.declare_state(value)
    return value

  def primitive(self, operation, args, kwargs, node):
    family = self.operations.get(operation).family
    if family == "mask_query":
      if len(args) != 1:
        self.error(node, "mask queries expect one predicate")
      value = self.value(args[0], node)
      if not isinstance(value.type, I.TileType) or not value.type.mask:
        self.error(node, "mask queries require a tile/vector predicate")
      dtype = "bool" if operation in {"mask_all", "mask_any", "mask_none"} else "int64"
      return I.Expr(
        I.ScalarType(dtype), operation, (value,), self.unroll_count(kwargs.get("unroll", I.literal(2)), node)
      )
    if operation == "element_at":
      if len(args) != 2:
        self.error(node, "element_at expects a value and logical coordinates")
      value = self.value(args[0], node)
      coords = self.shape(args[1], node)
      if not isinstance(value.type, I.TileType) or len(coords) != len(value.type.shape):
        self.error(node, "element_at requires one scalar index per tile axis")
      if any(not isinstance(c.type, I.ScalarType) or "int" not in c.type.dtype for c in coords):
        self.error(node, "element_at indices must be scalar integers")
      value = self.retain_register_transform(value)
      return I.Expr(I.ScalarType("bool" if value.type.mask else value.type.dtype), "element_at", (value, *coords))
    if operation in {"shuf", "local_shuf"}:
      if len(args) != 2:
        self.error(node, "register shuffles expect a value and indices/order")
      value = self.value(args[0], node)
      if not isinstance(value.type, I.TileType) or value.type.mask:
        self.error(node, "register shuffles require a numeric tile/vector")
      if operation == "shuf":
        index = self.value(args[1], node)
        if (
          not isinstance(index.type, I.TileType)
          or "int" not in index.type.dtype
          or len(index.type.shape) != len(value.type.shape)
        ):
          self.error(node, "shuf indices must be an integer tile of the same shape")
        for a, b in zip(index.type.shape, value.type.shape):
          self.block.statements.append(I.Assert(I.binary("eq", a, b), "shuf shape mismatch"))
        return I.Expr(value.type, operation, (value, index))
      order = tuple(I.constant(v) for v in self.shape(args[1], node))
      from ..dtypes import ELEMENT_BYTES

      lanes = 16 // ELEMENT_BYTES[value.type.dtype]
      if len(order) != lanes or any(type(v) is not int or not 0 <= v < lanes for v in order):
        self.error(node, "local_shuf order must permute a complete 16-byte block")
      return I.Expr(value.type, operation, (value,), order)
    if operation in {"gather", "scatter"}:
      expected = 2 if operation == "gather" else 3
      if len(args) != expected or not isinstance(args[0], I.View) or not args[0].shape:
        self.error(node, "indexed memory requires a memory view and logical indices")
      view = args[0]
      if operation == "scatter" and (len(view.shape) != 1 or isinstance(args[1], tuple)):
        self.error(node, "scatter currently requires a rank-one memory view and index tile")
      coordinates = args[1] if isinstance(args[1], tuple) else (args[1],)
      if len(coordinates) != len(view.shape):
        self.error(node, "gather needs one logical index per memory axis")
      coordinates = tuple(self.value(v, node) for v in coordinates)
      if any(v.type.dtype not in {"int32", "int64"} for v in coordinates):
        self.error(node, "indexed memory indices must be int32/int64 values")
      tiles = [v for v in coordinates if isinstance(v.type, I.TileType)]
      if not tiles:
        self.error(node, "indexed memory needs an index tile; use a scalar view for scalar loads")
      indices = tiles[0]
      shape = indices.type.shape
      for tile in tiles[1:]:
        shape = I.broadcast_shape(shape, tile.type.shape)
      mask = self.value(kwargs["mask"], node) if kwargs.get("mask") is not None else None
      other = self.value(kwargs.get("other", I.literal(0)), node) if operation == "gather" else None
      if mask is not None and not (mask.type.dtype == "bool" or isinstance(mask.type, I.TileType) and mask.type.mask):
        self.error(node, "indexed memory mask must be a predicate")
      # Shape checks precede writes, including the ordered scatter path.
      for operand in (mask, self.value(args[2], node) if operation == "scatter" else other):
        if operand is not None and isinstance(operand.type, I.TileType):
          if len(operand.type.shape) > len(shape):
            self.error(node, "indexed operands cannot exceed the index rank")
          for a, b in zip(reversed(operand.type.shape), reversed(shape)):
            self.block.statements.append(
              I.Assert(
                I.binary("or", I.binary("eq", a, I.literal(1)), I.binary("eq", a, b)), "indexed operand shape mismatch"
              )
            )
      self.materialize(view.storage)
      if operation == "gather":
        dtype = dtype_name(kwargs.get("dtype") or view.dtype)
        return I.Expr(
          I.shaped_type(dtype, shape, indices.type),
          "gather",
          (*coordinates, mask or I.literal(True), other),
          view,
        )
      if not view.writable:
        self.error(node, "scatter destination is input-only")
      unique = I.constant(self.value(kwargs.get("unique", I.literal(False)), node))
      if type(unique) is not bool:
        self.error(node, "scatter unique must be a compile-time bool")
      self.block.statements.append(
        I.IndexedStore(
          view,
          self.freeze_tile(indices),
          self.freeze_tile(self.value(args[2], node)),
          self.freeze_tile(mask) if mask is not None else None,
          unique,
        )
      )
      return None
    if operation in {"transpose", "reshape", "broadcast_to"}:
      if operation == "transpose":
        if not 1 <= len(args) <= 2 or len(args) == 2 and "axes" in kwargs:
          self.error(node, "transpose expects a value and optional axes")
      elif len(args) != 2:
        self.error(node, "reshape/broadcast_to expects a value and shape")
      value = args[0]
      if not isinstance(value, I.Expr) or not isinstance(value.type, I.TileType):
        if operation == "broadcast_to":
          self.error(node, "broadcast_to expects a tile value")
        return NotImplemented
      value = self.retain_register_transform(value)
      shape = value.type.shape
      if operation == "transpose":
        requested = kwargs.get("axes", args[1] if len(args) > 1 else None)
        axes = (
          tuple(I.constant(self.value(v, node)) for v in requested)
          if requested is not None
          else tuple(reversed(range(len(shape))))
        )
        if sorted(axes) != list(range(len(shape))):
          self.error(node, "transpose axes must be a permutation")
        shape = tuple(shape[a] for a in axes)
        data = (operation, axes)
      else:
        shape = self.shape(args[1], node)
        if operation == "reshape":

          def size(extents):
            result = I.literal(1)
            for extent in extents:
              result = I.binary("mul", result, extent)
            return result

          self.block.statements.append(
            I.Assert(I.binary("eq", size(shape), size(value.type.shape)), "reshape changes element count")
          )
        else:
          if len(shape) < len(value.type.shape):
            self.error(node, "broadcast_to cannot reduce rank")
          for a, b in zip(reversed(value.type.shape), reversed(shape)):
            self.block.statements.append(
              I.Assert(
                I.binary("or", I.binary("eq", a, I.literal(1)), I.binary("eq", a, b)), "broadcast_to shape mismatch"
              )
            )
        for extent in shape:
          self.block.statements.append(I.Assert(I.binary("ge", extent, I.literal(0)), "negative value extent"))
        data = (operation, ())
      # Transposed reads keep the existing DataAccess transpose fast path.
      if operation == "transpose" and value.op == "load" and not value.args:
        view = value.data
        return I.Expr(
          I.TileType(value.type.dtype, shape, value.type.mask),
          "load",
          data=replace(view, shape=shape, strides=tuple(view.strides[a] for a in axes), region=None),
        )
      return I.Expr(I.TileType(value.type.dtype, shape, value.type.mask), "value_transform", (value,), data)
    return NotImplemented
