# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Value transforms and explicit indexed memory, independent of CPU tags."""

from dataclasses import replace

from .. import ir as I
from ..dtypes import ELEMENT_BYTES, dtype_name
from ..ir.traversal import walk, walk_unique
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
    if operation == "_packing_offset":
      if len(args) != 7 or kwargs:
        self.error(node, "_packing_offset expects s, k, spatial, reduction, dtype, format, side")
      s, k, spatial, reduction = (self.value(v, node) for v in args[:4])
      dtype, format_, side = dtype_name(args[4]), args[5], args[6]
      if format_ not in {"auto", "dense", "amx", "sme"} or side not in {"A", "B"}:
        self.error(node, "invalid packing ABI")
      return I.Expr(I.ScalarType("int64"), "packing_offset", (s, k, spatial, reduction), (dtype, format_, side))
    if operation == "cache_bytes":
      if len(args) != 1 or kwargs or args[0] not in {"L1", "L2", "L3", "LLC"}:
        self.error(node, "cache_bytes expects one cache level: L1, L2, L3 or LLC")
      return I.Expr(I.ScalarType("int64"), "cache_bytes", data=args[0])
    if operation == "view_disjoint":
      if len(args) != 2 or kwargs:
        self.error(node, "view_disjoint expects two views or captured values")

      def memory_views(value):
        if isinstance(value, I.View):
          return (value,)
        if not isinstance(value, I.Expr):
          self.error(node, "view_disjoint expects memory views or captured values")
        views = {}
        for expression in walk_unique(value):
          definition = self.operations.get(expression)
          if expression.op == "packed_load":
            views[id(expression.data.view)] = expression.data.view
          elif self.operations.has(expression, Trait.MEMORY_READ):
            if not isinstance(expression.data, I.View):
              self.error(node, "view_disjoint cannot prove an opaque memory access")
            views[id(expression.data)] = expression.data
          elif definition.effects != "none" and not self.operations.has(expression, Trait.STATE_REFERENCE):
            self.error(node, "view_disjoint cannot prove an opaque/effectful producer")
        return tuple(views.values())

      result = I.literal(True)
      for first in memory_views(args[0]):
        for second in memory_views(args[1]):
          result = I.binary("and", result, I.Expr(I.ScalarType("bool"), operation, data=(first, second)))
      return result
    if family == "register_geometry":
      return self.register_geometry(operation, args, node)
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
      if operation == "broadcast_to" and isinstance(value, I.Expr) and isinstance(value.type, I.ScalarType):
        shape = self.shape(args[1], node)
        for extent in shape:
          self.block.statements.append(I.Assert(I.binary("ge", extent, I.literal(0)), "negative value extent"))
        return I.Expr(I.TileType(value.type.dtype, shape), "fill", (value,))
      if not isinstance(value, I.Expr) or not isinstance(value.type, I.TileType):
        if operation == "broadcast_to":
          self.error(node, "broadcast_to expects a tile value")
        return NotImplemented
      value = self.retain_register_transform(value)
      shape = value.type.shape
      if operation == "transpose":
        requested = kwargs.get("axes", args[1] if len(args) > 1 else None)
        axes = (
          tuple(I.constant(v) for v in self.shape(requested, node))
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

  def register_geometry(self, operation, args, node):
    """Fixed logical register geometry, independent of physical Vec limits.

    These are pure bit/lane transformations, not numeric conversions. A
    vector can contain multiple native words; CPU legalization must retain
    this logical geometry when splitting it into individually named values.
    """
    operation = {"lower_half": "lower", "upper_half": "upper"}.get(operation, operation)
    arity = 1 if operation in {"lower", "upper"} else 3 if operation == "block_shuffle" else 2
    if len(args) != arity:
      self.error(node, f"{operation} expects {arity} arguments")
    value = self.value(args[0], node)
    if operation == "bitcast":
      dtype = dtype_name(args[1])
      if value.type.dtype != "bool" and dtype != "bool" and ELEMENT_BYTES[value.type.dtype] == ELEMENT_BYTES[dtype]:
        typ = (
          I.shaped_type(dtype, value.type.shape, value.type, keep_tag=True)
          if isinstance(value.type, I.TileType)
          else I.ScalarType(dtype)
        )
        return I.Expr(typ, "bitreinterpret", (value,))
    if not isinstance(value.type, I.TileType) or value.type.mask or len(value.type.shape) != 1:
      self.error(node, "register geometry requires a numeric rank-one value")
    extent = value.type.shape[0]
    lanes = I.constant(extent)

    def scalable_extent(tag, dtype):
      count = I.Expr(I.ScalarType("int64"), "lanes", data=(tag.basis_dtype, tag.pow))
      if dtype != tag.basis_dtype:
        count = I.binary(
          "floordiv", I.binary("mul", count, I.literal(ELEMENT_BYTES[tag.basis_dtype])), I.literal(ELEMENT_BYTES[dtype])
        )
      return count

    scalable = (
      lanes is None
      and isinstance(value.type, I.VectorType)
      and value.type.tag.N is None
      and I.same_expr(extent, scalable_extent(value.type.tag, value.type.dtype))
    )
    if not scalable and (type(lanes) is not int or lanes <= 0):
      self.error(node, "register geometry currently requires a complete fixed extent")
    tag = value.type.tag if scalable else I.VectorTag(N=lanes, basis_dtype=value.type.dtype)
    minimum_bytes = (16 << tag.pow) if tag.pow >= 0 else (16 >> -tag.pow)
    minimum_lanes = minimum_bytes // ELEMENT_BYTES[value.type.dtype] if scalable else lanes
    if minimum_lanes < 1:
      self.error(node, "register geometry requires at least one guaranteed lane")
    dtype, operands, data = value.type.dtype, (value,), None
    if operation == "bitcast":
      dtype = dtype_name(args[1])
      size = minimum_lanes * ELEMENT_BYTES[value.type.dtype]
      if dtype == "bool" or size % ELEMENT_BYTES[dtype]:
        self.error(node, "bitcast requires equal logical bytes and a numeric dtype")
      extent = I.binary(
        "floordiv", I.binary("mul", extent, I.literal(ELEMENT_BYTES[value.type.dtype])), I.literal(ELEMENT_BYTES[dtype])
      )
    elif operation in {"lower", "upper"}:
      if minimum_lanes < 2 or minimum_lanes % 2:
        self.error(node, "lower/upper require an even logical lane count")
      extent = I.binary("floordiv", extent, I.literal(2))
      tag = replace(tag, pow=tag.pow - 1)
    else:
      right = self.value(args[1], node)
      if not isinstance(right.type, I.TileType) or right.type.mask or right.type.dtype != dtype:
        self.error(node, "register operands must have the same numeric dtype")
      if len(right.type.shape) != 1 or not I.same_expr(right.type.shape[0], extent):
        self.error(node, "register operands must have the same complete extent")
      operands = (value, right)
      if operation in {"concat", "interleave"}:
        extent = I.binary("mul", extent, I.literal(2))
        tag = replace(tag, pow=tag.pow + 1)
      elif operation == "block_shuffle":
        if scalable:
          self.error(node, "block_shuffle currently requires a complete fixed extent")
        # One block is 128 bits, matching vec's local-interleave boundary.
        blocks = lanes * ELEMENT_BYTES[dtype] // 16
        if lanes * ELEMENT_BYTES[dtype] < 16 or lanes * ELEMENT_BYTES[dtype] % 16:
          self.error(node, "block_shuffle requires complete 16-byte blocks")
        order = tuple(I.constant(v) for v in self.shape(args[2], node))
        if len(order) != blocks or any(type(i) is not int or not 0 <= i < 2 * blocks for i in order):
          self.error(node, "block_shuffle order must select one source block per output block")
        data = order
      elif minimum_lanes * ELEMENT_BYTES[dtype] % 16:
        self.error(node, "local interleave requires complete 16-byte blocks")
    if not scalable:
      tag = I.VectorTag(N=I.constant(extent), basis_dtype=dtype)
    else:
      extent = scalable_extent(tag, dtype)
    typ = I.VectorType(dtype, (extent,), tag=tag)
    return I.Expr(typ, operation, operands, data)
