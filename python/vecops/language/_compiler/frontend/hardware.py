# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Architecture operand checks and region-scoped streaming vector handles."""

from .. import ir as I
from ..dtypes import ELEMENT_BYTES, dtype_name
from ..ir.traversal import walk_unique
from ..operations import Trait


class HardwareFrontend:
  def validate_hardware_body(self, body, kind, node):
    """Admit only no-throw structured instruction programs.

    A bounded serial loop keeps ZA/AMX state in the same lexical owner.
    Generic stores remain the existing separately checked transpose route;
    mixing their vector preparation with runtime hardware control is not yet
    legal. All metadata checks are emitted before entering the owner.
    """

    def scalar(value, guard_only=False):
      for expression in walk_unique(value):
        divide = expression.op in {"floordiv", "mod"} and self.positive_extent(expression.args[1])
        if (
          not isinstance(expression.type, I.ScalarType)
          or expression.type.dtype not in {"bool", "int64", "int32"}
          or not (
            self.operations.has(expression, Trait.HARDWARE_SCALAR)
            or divide
            or guard_only
            and expression.op in {"view_disjoint", "view_nonoverlap"}
          )
        ):
          self.error(node, "hardware control/address expressions require pure integer/boolean metadata")

    def visit(block, controlled=False):
      for statement in block.statements:
        if isinstance(statement, I.For):
          if statement.kind != "serial":
            self.error(node, "hardware regions require serial instruction loops")
          for value in (statement.start, statement.stop, statement.step):
            scalar(value)
          visit(statement.body, True)
        elif isinstance(statement, I.If):
          scalar(statement.condition)
          visit(statement.body, True)
          visit(statement.otherwise, True)
        elif isinstance(statement, I.Assert):
          scalar(statement.condition, guard_only=True)
        elif isinstance(statement, I.Assign) and statement.declare:
          scalar(statement.value)
        elif isinstance(statement, I.ArchitectureCall):
          if statement.view is not None:
            for value in (statement.view.offset, *statement.view.shape, *statement.view.strides):
              scalar(value)
          for value in statement.operands:
            if isinstance(value, I.Expr):
              if statement.operation != "sme.apply":
                scalar(value)
          if statement.slice_index is not None:
            scalar(statement.slice_index)
          if statement.vector is not None:
            scalar(statement.vector.extent)
        elif isinstance(statement, I.Prefetch):
          for value in (statement.view.offset, *statement.view.shape, *statement.view.strides):
            scalar(value)
        elif kind == "sme" and not controlled and isinstance(statement, (I.BindTile, I.Store)):
          if isinstance(statement, I.Store) and statement.mask is not None:
            self.error(node, "SME region does not admit masked stores")
        else:
          self.error(
            node,
            "hardware region supports only whole-view transpose/architecture instructions and bounded serial control; no allocation, mutable state or ordinary vector computation",
          )

    visit(body)

  def sme_operation(self, operation, args, kwargs, node):
    region = id(self.hardware_regions[-1])
    if operation == "sme.apply":
      from .capture import Body
      from .helpers import ValueFunction

      if len(args) < 2 or not isinstance(args[0], (Body, ValueFunction)):
        self.error(node, "sme.apply expects a pure source helper and a streaming vector")
      arguments, handles = [], []
      for value in args[1:]:
        if isinstance(value, I.SMEVector) and value.region == region:
          handles.append(value)
          arguments.append(I.Expr(I.TileType(value.dtype, (value.extent,)), "vector_ref", data=value.name))
        elif isinstance(args[0], Body) and isinstance(value, str):
          arguments.append(value)
        else:
          argument = self.value(value, node)
          if not isinstance(argument.type, I.ScalarType):
            self.error(node, "sme.apply operands must be region-local vectors or scalars")
          arguments.append(argument)
      if not handles:
        self.error(node, "sme.apply requires a streaming vector")
      if isinstance(args[0], ValueFunction):
        result = args[0].bind(tuple(arguments))
      else:
        parent, self.block = self.block, I.Block()
        try:
          result = self.helper(args[0], node, arguments=arguments)
          if any(not isinstance(s, I.BindTile) for s in self.block.statements):
            self.error(node, "sme.apply helper must be a pure pointwise value expression")
        finally:
          self.block = parent
      if (not isinstance(result, I.Expr) or not isinstance(result.type, I.TileType)
          or len(result.type.shape) != 1 or result.type.mask or result.type.dtype == "bool"):
        self.error(node, "sme.apply helper must return one numeric vector")
      if any(not I.same_expr(v.extent, handles[0].extent) for v in handles[1:]):
        self.error(node, "sme.apply vectors must have identical prefixes")
      from ..analysis.projection import streaming_selection_supported

      if not streaming_selection_supported(result, handles[0].dtype):
        self.error(node, "sme.apply supports only memory-free conversion, selection and comparison expressions")
      if not I.same_expr(result.type.shape[0], handles[0].extent):
        self.error(node, "sme.apply cannot change the vector prefix")
      handle = I.SMEVector(self.unique("streaming_result"), result.type.dtype, handles[0].extent, region)
      self.block.statements.append(I.ArchitectureCall(operation, vector=handle, operands=(result, handles[0])))
      return handle
    if operation == "sme.mask":
      if len(args) != 1:
        self.error(node, "sme.mask expects one scalar active count")
      extent = self.value(args[0], node)
      dtype = dtype_name(kwargs.get("dtype", "uint32"))
      if dtype == "bool" or not isinstance(extent.type, I.ScalarType) or not extent.type.dtype.startswith("int"):
        self.error(node, "SME mask requires a numeric lane dtype and a scalar integer count")
      mask = I.SMEMask(self.unique("streaming_mask"), dtype, extent, region)
      self.block.statements.append(I.ArchitectureCall(operation, vector=mask))
      return mask
    if operation in {"sme.load_hor", "sme.load_ver", "sme.store_hor", "sme.store_ver"}:
      if len(args) != 3 or not isinstance(args[2], I.View) or len(args[2].shape) != 1:
        self.error(node, "SME slice moves expect (constant tile, scalar slice index, rank-one view)")
      tile, index, view = I.constant(self.value(args[0], node)), self.value(args[1], node), args[2]
      word_dtype = dtype_name(kwargs.get("word_dtype", view.dtype))
      if (
        type(tile) is not int
        or view.dtype == "bool"
        or word_dtype == "bool"
        or not 0 <= tile < ELEMENT_BYTES[word_dtype]
      ):
        self.error(node, "SME raw ZA tile index is out of range for the storage element width")
      if not isinstance(index.type, I.ScalarType) or not index.type.dtype.startswith("int"):
        self.error(node, "SME ZA slice index must be a scalar integer")
      if operation.startswith("sme.store") and not view.writable:
        self.error(node, "SME slice store destination is input-only")
      self.block.statements.append(I.ArchitectureCall(operation, tile, view, operands=(word_dtype,), slice_index=index))
      return None
    if operation in {"sme.read_hor", "sme.read_ver", "sme.write_hor", "sme.write_ver"}:
      if len(args) != 3:
        self.error(node, "SME vector slice moves expect (constant tile, scalar slice index, vector/active count)")
      tile, index = I.constant(self.value(args[0], node)), self.value(args[1], node)
      mask = None
      if operation.startswith("sme.read"):
        mask = args[2] if isinstance(args[2], I.SMEMask) else None
        extent = mask.extent if mask is not None else self.value(args[2], node)
        dtype = dtype_name(kwargs.get("dtype", mask.dtype if mask is not None else "float32"))
        if mask is not None and (mask.region != region or ELEMENT_BYTES[mask.dtype] != ELEMENT_BYTES[dtype]):
          self.error(node, "SME read mask must have the same lane width and belong to this region")
        if not isinstance(extent.type, I.ScalarType) or not extent.type.dtype.startswith("int"):
          self.error(node, "SME vector active count must be a scalar integer")
        vector = I.SMEVector(self.unique("za_vector"), dtype, extent, region)
      else:
        vector = args[2]
        if not isinstance(vector, I.SMEVector) or vector.region != region:
          self.error(node, "SME write requires a streaming vector from this region")
        dtype = vector.dtype
      if dtype == "bool" or type(tile) is not int or not 0 <= tile < ELEMENT_BYTES[dtype]:
        self.error(node, "SME ZA vector slice type/tile is invalid")
      if not isinstance(index.type, I.ScalarType) or not index.type.dtype.startswith("int"):
        self.error(node, "SME ZA slice index must be a scalar integer")
      self.block.statements.append(
        I.ArchitectureCall(operation, tile, slice_index=index, vector=vector, operands=() if mask is None else (mask,))
      )
      return vector if operation.startswith("sme.read") else None
    if operation in {"sme.store", "sme.store_bits"}:
      if (
        len(args) != 2
        or not isinstance(args[0], I.SMEVector)
        or args[0].region != region
        or not isinstance(args[1], I.View)
        or len(args[1].shape) != 1
        or not args[1].writable
        or args[1].dtype == "bool"
      ):
        self.error(node, f"{operation} expects (region-local streaming vector, writable rank-one view)")
      if operation == "sme.store_bits" and args[0].dtype not in {"uint8", "uint16", "uint32", "uint64"}:
        self.error(node, "sme.store_bits requires an unsigned streaming bit vector")
      self.block.statements.append(I.ArchitectureCall(operation, view=args[1], vector=args[0]))
      return None
    if operation in {"sme.load", "sme.load_bits"}:
      if len(args) != 1 or not isinstance(args[0], I.View) or len(args[0].shape) != 1:
        self.error(node, f"{operation} expects a rank-one view")
      if args[0].dtype not in ELEMENT_BYTES or args[0].dtype == "bool":
        self.error(node, "unsupported SME streaming operand dtype")
      dtype = dtype_name(kwargs.get("dtype", "uint32" if operation == "sme.load_bits" else args[0].dtype))
      if dtype not in ELEMENT_BYTES or dtype == "bool":
        self.error(node, "unsupported SME streaming vector dtype")
      extent = args[0].shape[0]
      if operation == "sme.load_bits":
        if dtype not in {"uint8", "uint16", "uint32", "uint64"}:
          self.error(node, "sme.load_bits requires an unsigned word dtype")
        size = ELEMENT_BYTES[dtype]
        extent = I.binary(
          "floordiv",
          I.binary("add", I.binary("mul", extent, I.literal(ELEMENT_BYTES[args[0].dtype])), I.literal(size - 1)),
          I.literal(size),
        )
      handle = I.SMEVector(self.unique("streaming_vector"), dtype, extent, region)
      strided = (
        I.constant(self.value(kwargs.get("strided", I.literal(False)), node)) if operation == "sme.load" else False
      )
      if type(strided) is not bool:
        self.error(node, "SME strided load option must be a compile-time bool")
      self.block.statements.append(
        I.ArchitectureCall(operation, view=args[0], vector=handle, operands=(True,) if strided else ())
      )
      return handle
    if operation == "sme.mopa":
      if len(args) != 3 or any(not isinstance(v, I.SMEVector) or v.region != region for v in args[1:]):
        self.error(node, "sme.mopa expects a ZA tile index and two sme.load handles in this region")
      tile = I.constant(self.value(args[0], node))
      a, b = args[1:]
      da, db = a.dtype, b.dtype
      integer = (da in {"int8", "uint8"} and db in {"int8", "uint8"}) or (
        da in {"int16", "uint16"} and db in {"int16", "uint16"}
      )
      floating = da == db and da in {"float16", "bfloat16", "float32", "float64"}
      if not (integer or floating):
        self.error(node, "unsupported SME mopa operand dtype combination")
      dtype = dtype_name(kwargs.get("acc_dtype", "int64" if da in {"int16", "uint16"} else
                                    "int32" if integer else "float64" if da == "float64" else "float32"))
      legal = (dtype in {"int32", "int64"} if da in {"int16", "uint16"} else
               dtype == "int32" if integer else dtype == ("float64" if da == "float64" else "float32"))
      if not legal:
        self.error(node, "unsupported SME mopa accumulator dtype")
      if dtype == "int32" and da in {"int16", "uint16"} and da != db:
        self.error(node, "SME2 I16I32 requires matching operand signedness")
      limit = ELEMENT_BYTES[dtype]
      if type(tile) is not int or not 0 <= tile < limit:
        self.error(node, "SME accumulator tile index out of range")
      self.block.statements.append(I.ArchitectureCall(operation, tile, operands=(dtype, a, b)))
      return None
    if operation == "sme.store_tile":
      if len(args) != 2 or not isinstance(args[1], I.View) or len(args[1].shape) != 2 or not args[1].writable:
        self.error(node, "sme.store_tile expects a ZA index and writable rank-two view")
      view = args[1]
      tile = I.constant(self.value(args[0], node))
      if (
        view.dtype not in {"float32", "int32", "float64", "int64"}
        or type(tile) is not int
        or not 0 <= tile < ELEMENT_BYTES[view.dtype]
      ):
        self.error(node, "invalid SME store accumulator type/index")
      self.block.statements.append(I.ArchitectureCall(operation, tile, view))
      return None
    return NotImplemented
