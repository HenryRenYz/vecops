# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Python AST partial evaluation into structured kernel IR."""

from __future__ import annotations

import ast
from dataclasses import replace
from types import ModuleType

from ...dsl import DSLFunction, ExecutionLayout, Intrinsic, Vector
from ...expr import Dynamic
from ...types import TensorType
from .. import ir as I
from ..dtypes import dtype_name
from ..ir.promotion import cast_operands, floating_dtype
from ..operations import Trait, use_catalog
from ..operations import current as operation_snapshot
from .capture import DSL_MODULES, Body, SelfCall, capture, symbol
from .collectives import CollectiveFrontend
from .diagnostics import DSLError
from .hardware import HardwareFrontend
from .helpers import HelperFrontend
from .indexing import IndexingFrontend
from .loops import LoopsFrontend
from .pipeline import PipelineFrontend
from .primitives import PrimitivesFrontend
from .printing import PrintFrontend
from .reducers import ReducersFrontend
from .shapes import ShapesFrontend
from .state import StateFrontend
from .workspace import WorkspaceFrontend


class Frontend(
  HelperFrontend,
  IndexingFrontend,
  StateFrontend,
  PrimitivesFrontend,
  LoopsFrontend,
  ReducersFrontend,
  CollectiveFrontend,
  PipelineFrontend,
  HardwareFrontend,
  ShapesFrontend,
  WorkspaceFrontend,
  PrintFrontend,
):
  def __init__(self, ir, *, catalog=None):
    self.operations = catalog or operation_snapshot()
    self.body = ir.body
    self.name = ir.name
    self.parameters = {}
    self.bindings = {}
    self.env = dict(self.body.environment)
    self.block = I.Block()
    self.counter = 0
    self.loop_depth = 0
    self.helper_depth = 0
    self.helper_scopes = []
    self.control_depth = 0
    self.mutable_names = set()
    self.promoted = {}
    self.diagnostics = []
    self.hardware_regions = []
    self.positive_scalars = set()
    self.helper_exits = []
    self.teams = []
    self.origins = {}
    for name, parameter in ir.signature.parameters.items():
      annotation = ir.annotations[name]
      self.parameters[name] = annotation
      if isinstance(annotation, TensorType):
        if annotation.dtype is None:
          raise DSLError("DSL tensor annotations require an explicit dtype")
        if annotation.optional or parameter.default is None:
          raise DSLError("optional DSL tensors are not implemented; select a template in the outer factory")
        shape = tuple(symbol(v) for v in annotation.shape)
        strides = tuple(symbol(v) for v in annotation.concrete_strides)
        self.env[name] = I.View(
          name,
          dtype_name(annotation.dtype),
          shape,
          strides,
          I.literal(0),
          annotation.access != "input",
          region=I.ViewRegion(
            shape,
            strides,
            tuple(I.literal(0) for _ in shape),
            shape,
            (1,) * len(shape),
            tuple(range(len(shape))),
            (False,) * len(shape),
            root_offset=I.literal(0),
          ),
        )
        for axis, value in enumerate(annotation.shape):
          if isinstance(value, Dynamic):
            self.bindings.setdefault(value.name, I.Expr(I.ScalarType("int64"), "raw", data=f"{name}.size({axis})"))
        for axis, value in enumerate(annotation.concrete_strides):
          if isinstance(value, Dynamic):
            self.bindings.setdefault(value.name, I.Expr(I.ScalarType("int64"), "raw", data=f"{name}.stride({axis})"))
      else:
        self.env[name] = I.Expr(I.ScalarType(dtype_name(annotation)), "raw", data=name)

  def unique(self, name="v"):
    self.counter += 1
    return f"dsl_f_{name}_{self.counter}"

  def error(self, node, message):
    raise DSLError(f"{self.body.filename}:{self.body.first_line + getattr(node, 'lineno', 1) - 1}: {message}")

  def compile(self):
    with use_catalog(self.operations):
      self.statements(self.body.tree.body)
    from ..._debug.session import current_session

    session = current_session.get()
    if session is not None:
      session.origins.update({f"node{index}": location for index, location in enumerate(self.origins.values())})
    return I.KernelIR(
      self.block, self.parameters, self.bindings, self.diagnostics, origins=self.origins, name=self.name
    )

  def value(self, obj, node):
    obj = symbol(obj)
    if not isinstance(obj, I.Expr):
      self.error(node, "expected a scalar or tile computation value")
    return obj

  def shape(self, obj, node):
    return tuple(self.value(v, node) for v in (obj if isinstance(obj, tuple) else (obj,)))

  def expression(self, node):
    # Every scalar expression has the same IR representation, whether it came
    # from an AST literal, a captured Python value, tuple indexing, or a helper
    # default. Keep immutable host captures intact for their capture identity;
    # normalize them at the single expression-consumption boundary instead.
    value = symbol(self._expression(node))
    if isinstance(value, I.Node):
      self.origins.setdefault(id(value), (self.body.filename, self.body.first_line + node.lineno - 1, node.col_offset))
    return value

  def _expression(self, node):
    if isinstance(node, ast.JoinedStr):
      return self.joined_string(node)
    if isinstance(node, ast.Constant):
      return symbol(node.value)
    if isinstance(node, ast.Name):
      if node.id in self.env:
        return self.env[node.id]
      if node.id in {"float", "int", "bool"}:
        return {"float": float, "int": int, "bool": bool}[node.id]
      if node.id == "print":
        return Intrinsic("print")
      self.error(node, f"unbound kernel name {node.id!r}")
    if isinstance(node, (ast.Tuple, ast.List)):
      if isinstance(node, ast.List):
        self.error(node, "use immutable tuples in kernel code")
      values = []
      for item in node.elts:
        if isinstance(item, ast.Starred):
          expanded = self.expression(item.value)
          if not isinstance(expanded, tuple):
            self.error(item, "tuple expansion requires an immutable tuple")
          values.extend(expanded)
        else:
          values.append(self.expression(item))
      return tuple(values)
    if isinstance(node, ast.Attribute):
      path = ast.unparse(node)
      if path in self.env:
        return self.env[path]
      obj = self.expression(node.value)
      from .helpers import ValueFunction

      if isinstance(obj, (ValueFunction, Body)) and node.attr == "streaming_selection":
        return obj.streaming_selection if isinstance(obj, ValueFunction) else False
      if node.attr == "T" and isinstance(obj, I.Expr) and isinstance(obj.type, I.TileType):
        return self.primitive("transpose", [obj], {}, node)
      if isinstance(obj, ModuleType):
        value = getattr(obj, node.attr, None)
        if isinstance(value, ModuleType) and value.__name__ in DSL_MODULES:
          return value
        if isinstance(value, DSLFunction):
          return capture(value.function, inline=value.inline)
        from ...._dtype import is_native_dtype
        from ..._debug.options import PrintTarget

        if isinstance(value, (Intrinsic, PrintTarget, type)) or is_native_dtype(value):
          return value
        self.error(node, f"unsupported DSL attribute {node.attr!r}")
      if node.attr in {"shape", "strides"} and isinstance(obj, I.View):
        return getattr(obj, node.attr)
      if node.attr == "shape" and isinstance(obj, I.Expr) and isinstance(obj.type, I.TileType):
        return obj.type.shape
      if node.attr == "dtype" and isinstance(obj, (I.View, I.Expr, I.SMEVector, I.SMEMask)):
        return obj.type.dtype if isinstance(obj, I.Expr) else obj.dtype
      if node.attr == "T" and isinstance(obj, I.View):
        region = replace(obj.region, axes=obj.region.axes[::-1]) if obj.region is not None else None
        return replace(obj, shape=obj.shape[::-1], strides=obj.strides[::-1], region=region)
      self.error(node, f"unsupported attribute {node.attr!r}")
    if isinstance(node, ast.BinOp):
      operators = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mul",
        ast.Div: "div",
        ast.FloorDiv: "floordiv",
        ast.Mod: "mod",
        ast.BitAnd: "and",
        ast.BitOr: "or",
        ast.BitXor: "xor",
        ast.LShift: "shl",
        ast.RShift: "shr",
      }
      if isinstance(node.op, ast.Pow):
        exponent = self.expression(node.right)
        if isinstance(exponent, I.Expr) and I.constant(exponent) == 2:
          value = self.value(self.expression(node.left), node)
          return I.binary("mul", value, value)
        self.error(node, "only squared expressions are supported by **; use explicit DSL math")
      if type(node.op) not in operators:
        self.error(node, "unsupported binary operation")
      raw_left, raw_right = self.expression(node.left), self.expression(node.right)
      if isinstance(node.op, ast.Mult):
        if isinstance(raw_left, (str, tuple)) and isinstance(I.constant(raw_right), int):
          return raw_left * I.constant(raw_right)
        if isinstance(raw_right, (str, tuple)) and isinstance(I.constant(raw_left), int):
          return raw_right * I.constant(raw_left)
      if isinstance(node.op, ast.Add) and isinstance(raw_left, (str, tuple)) and type(raw_right) is type(raw_left):
        return raw_left + raw_right
      left, right = self.value(raw_left, node), self.value(raw_right, node)
      op = operators[type(node.op)]
      if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        predicate = lambda v: v.type.dtype == "bool" or isinstance(v.type, I.TileType) and v.type.mask
        if predicate(left) and predicate(right):
          op = {ast.BitAnd: "and", ast.BitOr: "or", ast.BitXor: "mask_xor"}[type(node.op)]
        else:
          op = {ast.BitAnd: "bit_and", ast.BitOr: "bit_or", ast.BitXor: "bit_xor"}[type(node.op)]
      return I.binary(op, left, right)
    if isinstance(node, ast.UnaryOp):
      value = self.value(self.expression(node.operand), node)
      op = {ast.USub: "neg", ast.UAdd: "identity", ast.Not: "not", ast.Invert: "not"}.get(type(node.op))
      if op is None:
        self.error(node, "unsupported unary operation")
      if value.op == "literal":
        result = (
          ~value.data
          if isinstance(node.op, ast.Invert) and value.type.dtype != "bool"
          else -value.data
          if op == "neg"
          else not value.data
          if op == "not"
          else value.data
        )
        dtype = (
          None
          if value.type.weak
          else "bool"
          if op == "not" and not isinstance(node.op, ast.Invert)
          else value.type.dtype
        )
        return I.literal(result, dtype=dtype)
      predicate = value.type.dtype == "bool" or isinstance(value.type, I.TileType) and value.type.mask
      if isinstance(node.op, ast.Invert) and not predicate:
        op = "bit_not"
      if isinstance(node.op, ast.Not) and isinstance(value.type, I.TileType) and not predicate:
        self.error(node, "not requires a scalar or tile predicate; compare tile values explicitly")
      typ = I.ScalarType("bool") if op == "not" and isinstance(value.type, I.ScalarType) else value.type
      return I.Expr(typ, op, (value,))
    if isinstance(node, ast.Compare):
      left = self.expression(node.left)
      result = I.literal(True)
      for op, rhs in zip(node.ops, node.comparators):
        right = self.expression(rhs)
        if isinstance(op, (ast.Is, ast.IsNot)):
          if right is not None and left is not None:
            self.error(node, "identity comparisons are only supported for None")
          comparison = I.literal((left is right) if isinstance(op, ast.Is) else (left is not right))
          result = I.binary("and", result, comparison)
          left = right
          continue
        if isinstance(left, str) and isinstance(right, str):
          if not isinstance(op, (ast.Eq, ast.NotEq)):
            self.error(node, "only equality is supported for compile-time strings")
          result = I.binary("and", result, I.literal((left == right) if isinstance(op, ast.Eq) else (left != right)))
          left = right
          continue
        left, right = self.value(left, node), self.value(right, node)
        name = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}.get(type(op))
        if name is None:
          self.error(node, "unsupported comparison")
        result = I.binary("and", result, I.binary(name, left, right))
        left = right
      return result
    if isinstance(node, ast.BoolOp):
      values = [self.value(self.expression(v), node) for v in node.values]
      result = values[0]
      for value in values[1:]:
        result = I.binary("and" if isinstance(node.op, ast.And) else "or", result, value)
      return result
    if isinstance(node, ast.IfExp):
      condition = self.value(self.expression(node.test), node)
      if condition.op == "literal":
        return self.expression(node.body if condition.data else node.orelse)
      a, b = self.value(self.expression(node.body), node), self.value(self.expression(node.orelse), node)
      return self.select(condition, a, b)
    if isinstance(node, ast.Subscript):
      return self.index(node)
    if isinstance(node, ast.Call):
      return self.call(node)
    self.error(node, f"unsupported expression {type(node).__name__}")

  def call(self, node):
    if isinstance(node.func, ast.Name) and node.func.id in {
      "range",
      "float",
      "int",
      "bool",
      "len",
      "min",
      "max",
      "abs",
    }:
      operation = node.func.id
    else:
      function = self.expression(node.func)
      if isinstance(function, SelfCall):
        scope = next(
          (
            (body, depth)
            for body, depth in reversed(self.helper_scopes)
            if any(v is function for v in body.environment.values())
          ),
          None,
        )
        if scope is None or self.control_depth > scope[1]:
          self.error(node, "recursive T.func calls require compile-time control and specialization")
        return self.helper(scope[0], node)
      if isinstance(function, Body):
        return self.helper(function, node)
      from .helpers import ValueFunction

      if isinstance(function, ValueFunction):
        if node.keywords:
          self.error(node, "staged value helpers require positional arguments")
        return function.bind(tuple(self.value(self.expression(a), node) for a in node.args))
      if function in (Vector, ExecutionLayout):

        def descriptor_value(value):
          if isinstance(value, I.Expr) and value.op == "literal":
            return value.data
          if isinstance(value, tuple):
            return tuple(descriptor_value(v) for v in value)
          return value

        args = [self.expression(a) for a in node.args]
        kwargs = {k.arg: self.expression(k.value) for k in node.keywords}
        args = [descriptor_value(v) for v in args]
        kwargs = {k: descriptor_value(v) for k, v in kwargs.items()}
        return function(*args, **kwargs)
      if not isinstance(function, Intrinsic):
        self.error(node, "arbitrary Python calls are not allowed; use T.func or a DSL intrinsic")
      operation = function.name
    args = [self.expression(a) for a in node.args]
    kwargs = {k.arg: self.expression(k.value) for k in node.keywords}
    if any(k.arg is None for k in node.keywords):
      self.error(node, "runtime keyword unpacking is not supported")
    if operation in {"print", "static_print"}:
      return self.print_call(operation, args, kwargs, node)
    if operation == "map_rows":
      return self.map_rows(args, kwargs, node)
    if operation in {"dtype_name", "itemsize", "static_log2"}:
      if len(args) != 1 or kwargs:
        self.error(node, f"{operation} expects one compile-time argument")
      if operation == "dtype_name":
        return dtype_name(args[0])
      if operation == "itemsize":
        from ..dtypes import ELEMENT_BYTES

        return I.literal(ELEMENT_BYTES[dtype_name(args[0])])
      value = I.constant(self.value(args[0], node))
      if type(value) is not int or value <= 0 or value & (value - 1):
        self.error(node, "static_log2 requires a positive compile-time power of two")
      return I.literal(value.bit_length() - 1)
    allowed = {
      "load": {"dtype", "vector", "mask", "other"},
      "store": {"mask", "implementation"},
      "copy": {"mask", "implementation"},
      "alloc": {"dtype"},
      "zeros": {"dtype", "active"},
      "full": {"dtype", "active"},
      "as_vector": set(),
      "cast": {"dtype"},
      "transpose": {"axes"},
      "serial": {"unroll"},
      "dot": {"acc", "acc_dtype", "implementation"},
      "_matmul": {"acc", "acc_dtype", "implementation", "packing", "BM", "BN", "BK", "swap"},
      "pack": {"side", "format", "dtype"},
      "packed": {"side", "format"},
    }.get(operation, set(self.operations.get(operation).keywords))
    if set(kwargs) - allowed:
      self.error(node, f"unsupported {operation} keyword(s): {', '.join(sorted(set(kwargs) - allowed))}")
    if operation in {"sme.region", "amx.region"}:
      configuration = ()
      if operation == "amx.region":
        if len(args) != 1 or not isinstance(args[0], tuple) or not 1 <= len(args[0]) <= 8:
          self.error(node, "amx.region expects 1..8 (rows, column_bytes) descriptors")
        if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in args[0]):
          self.error(node, "AMX configuration entries must be (rows, column_bytes) tuples")
        configuration = tuple(tuple(I.constant(self.value(v, node)) for v in pair) for pair in args[0])
        if any(
          len(pair) != 2
          or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)
          or not 1 <= pair[0] <= 16
          or not 1 <= pair[1] <= 64
          for pair in configuration
        ):
          self.error(node, "AMX descriptors require rows in 1..16 and column_bytes in 1..64")
      elif args:
        self.error(node, "sme.region takes no arguments")
      return ("hardware-region", operation[:3], configuration)
    if operation == "sme.streaming_lanes":
      if len(args) != 1:
        self.error(node, "sme.streaming_lanes expects one dtype")
      return I.Expr(I.ScalarType("int64"), "streaming_lanes", data=dtype_name(args[0]))
    if operation.startswith(("sme.", "amx.")):
      kind = operation[:3]
      if not self.hardware_regions or self.hardware_regions[-1][0] != kind:
        self.error(node, f"{operation} requires a matching lexical hardware region")
      if kind == "sme":
        result = self.sme_operation(operation, args, kwargs, node)
        if result is not NotImplemented:
          return result
      if operation == "amx.tdp":
        if len(args) != 3 or any(
          not isinstance(v, I.AMXTile) or v.region != id(self.hardware_regions[-1]) for v in args
        ):
          self.error(node, "amx.tdp expects accumulator, A, B tile handles from this region")
        acc, a, b = args
        integer = a.dtype in {"int8", "uint8"} and b.dtype in {"int8", "uint8"}
        floating = a.dtype == b.dtype and a.dtype in {"float16", "bfloat16"}
        if not (integer and acc.dtype == "int32" or floating and acc.dtype == "float32"):
          self.error(node, "unsupported AMX tdp operand/accumulator dtype combination")
        if len({v.index for v in args}) != 3:
          self.error(node, "AMX tdp requires three distinct tile registers")
        cfg = self.hardware_regions[-1][1]
        m, nbytes = cfg[acc.index]
        am, kbytes = cfg[a.index]
        bk, bnbytes = cfg[b.index]
        if am != m or bnbytes != nbytes or kbytes != bk * 4 or nbytes % 4:
          self.error(node, "AMX tdp configuration dimensions are incompatible")
        self.block.statements.append(I.ArchitectureCall(operation, operands=args))
        return acc
      if operation == "sme.zero_za":
        if args:
          self.error(node, "zero_za takes no arguments")
        self.block.statements.append(I.ArchitectureCall(operation))
      else:
        if operation not in {"amx.zero", "amx.load", "amx.stream_load", "amx.store"}:
          self.error(node, "unsupported architecture instruction")
        count = 1 if operation == "amx.zero" else 2
        if len(args) != count:
          self.error(node, f"{operation} expects {count} arguments")
        tile = args[0].index if isinstance(args[0], I.AMXTile) else I.constant(self.value(args[0], node))
        if isinstance(tile, bool) or not isinstance(tile, int) or not 0 <= tile < len(self.hardware_regions[-1][1]):
          self.error(node, "AMX tile index must be a configured compile-time constant")
        view = args[1] if count == 2 else None
        if view is not None and (not isinstance(view, I.View) or len(view.shape) != 2):
          self.error(node, "AMX memory instructions require a rank-two view")
        if operation == "amx.store" and not view.writable:
          self.error(node, "AMX store destination is input-only")
        self.block.statements.append(I.ArchitectureCall(operation, tile, view))
        if operation in {"amx.zero", "amx.load", "amx.stream_load"}:
          dtype = dtype_name(kwargs.get("dtype", "float32")) if view is None else view.dtype
          return I.AMXTile(tile, dtype, id(self.hardware_regions[-1]))
      return None
    workspace = self.workspace_operation(operation, args, kwargs, node)
    if workspace is not NotImplemented:
      return workspace
    shaped = self.shape_operation(operation, args, kwargs, node)
    if shaped is not NotImplemented:
      return shaped
    primitive = self.primitive(operation, args, kwargs, node)
    if primitive is not NotImplemented:
      return primitive
    collective = self.collective(operation, args, kwargs, node)
    if collective is not NotImplemented:
      return collective
    pipeline = self.pipeline_operation(operation, args, kwargs, node)
    if pipeline is not NotImplemented:
      return pipeline
    if operation in {"reduce", "scan"}:
      return self.fold(operation, args, kwargs, node)
    if operation in {"parallel", "simd"}:
      return (operation, args, kwargs)
    if operation in {"range", "serial", "grid", "unroll"}:
      if operation == "grid":
        if set(kwargs) - {"schedule", "chunk", "parallel"}:
          self.error(node, "unknown grid option")
        schedule, parallel = kwargs.get("schedule", "auto"), kwargs.get("parallel", "auto")
        chunk = I.constant(self.value(kwargs.get("chunk", I.literal(1)), node))
        if schedule not in {"auto", "static", "dynamic"} or parallel not in {"auto", "require", "serial"}:
          self.error(node, "invalid grid schedule or parallel policy")
        if type(chunk) is not int or not 1 <= chunk <= 1048576:
          self.error(node, "grid chunk must be a positive compile-time integer <= 1048576")
        return (operation, args, (schedule, chunk, parallel))
      if operation == "serial":
        return (operation, args, self.unroll_count(kwargs.get("unroll", I.literal(2)), node))
      return (operation, args)
    if operation == "len":
      return args[0].shape[0] if isinstance(args[0], I.View) else I.literal(len(args[0]))
    if operation == "as_value":
      if len(args) != 1:
        self.error(node, "as_value expects one operand")
      return self.matrix_operand(args[0], node, allow_scalar=True)
    if operation in {"dot", "_matmul"}:
      pure = len(args) in (2, 3) and not isinstance(args[0], tuple)
      if len(args) not in (2, 3) or not pure and len(args[0]) != 2:
        self.error(node, "dot/matmul expects ((A, B), output, acc=None)")
      a, b = (self.matrix_operand(v, node) for v in (args[:2] if pure else args[0]))
      output = (
        args[1]
        if not pure
        else I.View(
          self.unique("dot_result"),
          "float32",
          (a.type.shape[0], b.type.shape[1]),
          (b.type.shape[1], I.literal(1)),
          I.literal(0),
          owned=True,
        )
      )
      if not isinstance(output, I.View) or len(output.shape) != 2 or not output.writable:
        self.error(node, "dot output must be a writable rank-two tile")
      if a.op == "packed_load" and a.data.side != "A" or b.op == "packed_load" and b.data.side != "B":
        self.error(node, "packed operand side does not match its dot slot")
      acc = kwargs.get("acc", args[2] if len(args) == 3 else None)
      dtype = dtype_name(
        kwargs.get(
          "acc_dtype",
          "int32"
          if a.type.dtype in {"int8", "uint8"} and b.type.dtype in {"int8", "uint8"}
          else "int64"
          if a.type.dtype in {"int16", "uint16"} and b.type.dtype in {"int16", "uint16"}
          else "float64"
          if "float64" in (a.type.dtype, b.type.dtype)
          else "float32",
        )
      )
      if dtype not in {"float32", "float64", "int32", "int64"}:
        self.error(node, "dot accumulator must be float32, float64, int32 or int64")
      for x, y in (
        (a.type.shape[1], b.type.shape[0]),
        (a.type.shape[0], output.shape[0]),
        (b.type.shape[1], output.shape[1]),
      ):
        self.block.statements.append(I.Assert(I.binary("eq", x, y), "incompatible dot matrix extents"))
      options = {}
      for key in ("BM", "BN", "BK"):
        value = kwargs.get(key)
        count = I.constant(value) if value is not None else None
        if value is not None and (isinstance(count, bool) or not isinstance(count, int) or count <= 0):
          self.error(node, f"{key} must be None or a positive constant")
        options[{"BM": "block_m", "BN": "block_n", "BK": "block_k"}[key]] = count
      implementation = kwargs.get("implementation", "auto")
      packing = kwargs.get("packing", "auto")
      if implementation not in {"auto", "vector", "amx", "sme"} or packing not in {"auto", "none", "A", "B", "AB"}:
        self.error(node, "invalid matmul implementation or packing policy")
      swap = kwargs.get("swap", I.literal(True))
      swap = I.constant(swap)
      if not isinstance(swap, bool):
        self.error(node, "swap must be a compile-time bool")
      shape = output.shape
      operands = (a, b) + ((self.matrix_operand(acc, node, allow_scalar=True),) if acc is not None else ())
      value = self.freeze_tile(
        I.Expr(
          I.TileType(dtype, shape),
          "dot",
          operands,
          I.DotOptions(
            tiled=operation == "_matmul", packing=packing, implementation=implementation, swap=swap, **options
          ),
        )
      )
      if not pure:
        self.write_tile(output, value)
      return value
    if operation in {"pack", "packed"}:
      side, format_ = kwargs.get("side", "A"), kwargs.get("format", "auto")
      if side not in {"A", "B"} or format_ not in {"auto", "amx", "sme", "dense"}:
        self.error(node, "packing requires side A/B and format auto/amx/sme/dense")
      if operation == "packed":
        if len(args) != 2 or not isinstance(args[0], I.View):
          self.error(node, "packed expects (physical_buffer, logical_shape)")
        shape = self.shape(args[1], node)
        if len(shape) != 2:
          self.error(node, "packed logical shape must have rank two")
        view = args[0]
      else:
        if len(args) not in (1, 2):
          self.error(node, "pack expects source and an optional physical output buffer")
        source = self.matrix_operand(args[0], node)
        if source.op == "packed_load":
          self.error(node, "pack source must be an unpacked logical matrix")
        dtype = dtype_name(kwargs.get("dtype", source.type.dtype))
        if dtype != source.type.dtype:
          source = I.Expr(I.TileType(dtype, source.type.shape), "cast", (source,))
        shape = source.type.shape
        spatial, k = shape if side == "A" else shape[::-1]
        count = I.Expr(I.ScalarType("int64"), "packing_size", (spatial, k), (dtype, format_))
        if len(args) == 2:
          view = args[1]
          if not isinstance(view, I.View) or not view.writable or view.dtype != dtype:
            self.error(node, "pack output requires a writable buffer of the packed dtype")
        else:
          view = I.View(self.unique("packed"), dtype, (count,), (I.literal(1),), I.literal(0), owned=True)
          self.block.statements.append(I.Allocate(view))
        self.block.statements.append(I.Pack(self.freeze_tile(source), I.PackedView(view, shape, side, format_)))
      if format_ == "amx" and view.dtype not in {"bfloat16", "float16", "int8", "uint8"}:
        self.error(node, "AMX packing supports bf16/fp16/int8/uint8 only")
      return I.Expr(I.TileType(view.dtype, shape), "packed_load", data=I.PackedView(view, shape, side, format_))
    if operation in {"float", "int", "bool", "cast"}:
      dtype = {"float": "float64", "int": "int64", "bool": "bool"}.get(operation)
      if dtype is None:
        dtype = dtype_name(args[1] if len(args) > 1 else kwargs["dtype"])
      value = self.value(args[0], node)
      typ = (
        I.shaped_type(dtype, value.type.shape, value.type)
        if isinstance(value.type, I.TileType)
        else I.ScalarType(dtype)
      )
      if value.op == "arange" and value.type.dtype == "int64" and dtype.startswith("uint"):
        # Unsigned conversion commutes with the sequence's unit increment:
        # generate modulo-width indices directly instead of building wide
        # int64 vectors and narrowing them. Do not apply this to float casts
        # (rounding) or signed sequences (signed overflow semantics).
        return I.Expr(typ, "arange", value.args, value.data)
      return I.Expr(typ, "cast", (value,))
    if operation == "tile":
      view = args[0]
      if not isinstance(view, I.View):
        self.error(node, "T.tile expects a tensor region")
      if len(args) != 1 or kwargs:
        self.error(node, "use tensor slices to specify T.tile's region")
      return view
    if operation == "transpose":
      view = args[0]
      if not isinstance(view, I.View):
        self.error(node, "T.transpose expects a view")
      axes = (
        args[1] if len(args) > 1 else kwargs.get("axes", tuple(I.literal(i) for i in reversed(range(len(view.shape)))))
      )
      axes = tuple(I.constant(v) for v in self.shape(axes, node))
      if sorted(axes) != list(range(len(view.shape))):
        self.error(node, "transpose axes must be a permutation")
      self.diagnostics.append(
        "transpose view: CPU lowering selects register/ZA tiles or gather/scatter by access legality"
      )
      return I.View(
        view.storage,
        view.dtype,
        tuple(view.shape[i] for i in axes),
        tuple(view.strides[i] for i in axes),
        view.offset,
        view.writable,
        view.owned,
        replace(view.region, axes=tuple(view.region.axes[i] for i in axes)) if view.region is not None else None,
      )
    if operation == "reshape":
      view = args[0]
      if not isinstance(view, I.View):
        self.error(node, "reshape expects a memory view")
      shape = self.shape(args[1], node)
      previous_size, new_size, stride = I.literal(1), I.literal(1), I.literal(1)
      for extent, actual in reversed(list(zip(view.shape, view.strides))):
        self.block.statements.append(I.Assert(I.binary("eq", actual, stride), "reshape requires a dense view"))
        stride = I.binary("mul", stride, extent)
        previous_size = I.binary("mul", previous_size, extent)
      strides = [I.literal(1)] * len(shape)
      for axis in reversed(range(len(shape))):
        self.block.statements.append(I.Assert(I.binary("ge", shape[axis], I.literal(0)), "negative reshape extent"))
        new_size = I.binary("mul", new_size, shape[axis])
        if axis + 1 < len(shape):
          strides[axis] = I.binary("mul", strides[axis + 1], shape[axis + 1])
      self.block.statements.append(I.Assert(I.binary("eq", previous_size, new_size), "reshape changes element count"))
      # A checked dense reshape creates a bounded logical coordinate system
      # for memory preflight. It does not establish an affine grid partition
      # relative to other views of the original storage.
      region = I.ViewRegion(
        shape,
        tuple(strides),
        (I.literal(0),) * len(shape),
        shape,
        (1,) * len(shape),
        tuple(range(len(shape))),
        (False,) * len(shape),
        affine_partition=False,
        root_offset=view.offset,
        span_source=(
          view.region.span_source
          if view.region is not None and view.region.span_source is not None
          else replace(view, region=None)
        ),
      )
      return I.View(view.storage, view.dtype, shape, tuple(strides), view.offset, view.writable, view.owned, region)
    if operation == "load":
      view = args[0]
      if not isinstance(view, I.View):
        self.error(node, "T.load expects a tensor/tile region")
      dtype = dtype_name(kwargs.get("dtype", args[1] if len(args) > 1 else view.dtype))
      if view.storage in self.promoted:
        original, value = self.promoted[view.storage]
        if (
          "mask" not in kwargs
          and I.same_expr(view.offset, original.offset)
          and len(view.shape) == len(original.shape)
          and all(I.same_expr(a, b) for a, b in zip(view.shape, original.shape))
          and all(I.same_expr(a, b) for a, b in zip(view.strides, original.strides))
        ):
          if dtype != value.type.dtype:
            return I.Expr(I.shaped_type(dtype, value.type.shape, value.type), "cast", (value,))
          return value
        self.materialize(view.storage)
      shape = view.shape
      vector = kwargs.get("vector")
      if vector is not None:
        if not isinstance(vector, Vector) or len(shape) != 1:
          self.error(node, "vector loads require a Vector descriptor and a one-dimensional region")
        dtype = dtype_name(vector.dtype)
        requested = (
          I.literal(vector.N)
          if vector.N is not None
          else I.Expr(I.ScalarType("int64"), "lanes", data=(dtype, vector.pow))
        )
        shape = (I.binary("min", shape[0], requested),)
        view = I.View(view.storage, view.dtype, shape, view.strides, view.offset, view.writable, view.owned)
      typ = (
        I.VectorType(dtype, shape, tag=I.VectorTag(vector.N, vector.pow, basis_dtype=dtype))
        if vector is not None
        else I.TileType(dtype, shape)
        if shape
        else I.ScalarType(dtype)
      )
      memory_args = ()
      if "mask" in kwargs:
        mask = self.value(kwargs["mask"], node)
        if not (mask.type.dtype == "bool" or isinstance(mask.type, I.TileType) and mask.type.mask):
          self.error(node, "memory mask must be a predicate")
        if isinstance(mask.type, I.TileType):
          if len(mask.type.shape) > len(shape):
            self.error(node, "load mask cannot have a higher rank than the loaded region")
          for a, b in zip(reversed(mask.type.shape), reversed(shape)):
            if a.op == b.op == "literal" and a.data not in (1, b.data):
              self.error(node, "load mask does not broadcast to the loaded region")
        other = self.value(kwargs.get("other", I.literal(0)), node)
        memory_args = (
          mask,
          I.Expr(I.shaped_type(dtype, other.type.shape, other.type), "cast", (other,))
          if isinstance(other.type, I.TileType)
          else I.Expr(I.ScalarType(dtype), "cast", (other,)),
        )
      elif "other" in kwargs:
        self.error(node, "load other requires a mask")
      return I.Expr(typ, "load", memory_args, view)
    if operation in {"alloc", "zeros", "full"}:
      if operation == "alloc":
        return self.call_allocation(args, kwargs, node)
      return self.fill_value(operation, args, kwargs, node)
    if operation in {"store", "copy"}:
      destination = args[0] if operation == "store" else args[1]
      if operation == "copy":
        source = args[0]
        if not isinstance(source, I.View):
          self.error(node, "copy source must be a memory view")
        value = None
        if source.storage in self.promoted:
          original, promoted = self.promoted[source.storage]
          if (
            I.same_expr(source.offset, original.offset)
            and len(source.shape) == len(original.shape)
            and all(I.same_expr(a, b) for a, b in zip(source.shape, original.shape))
            and all(I.same_expr(a, b) for a, b in zip(source.strides, original.strides))
          ):
            value = promoted
          else:
            self.materialize(source.storage)
        if value is None:
          typ = I.TileType(source.dtype, source.shape) if source.shape else I.ScalarType(source.dtype)
          value = I.Expr(typ, "load", data=source)
      else:
        value = args[1]
      if not isinstance(destination, I.View):
        self.error(node, "store/copy destination must be a view")
      if not destination.writable:
        self.error(node, f"tensor {destination.storage!r} is input-only; annotate access='output' or 'inout'")
      value = self.freeze_tile(self.value(value, node))
      mask = self.freeze_tile(self.value(kwargs["mask"], node)) if "mask" in kwargs else None
      if mask is not None and not (mask.type.dtype == "bool" or isinstance(mask.type, I.TileType) and mask.type.mask):
        self.error(node, "memory mask must be a predicate")
      previous = self.promoted.get(destination.storage)
      same_forwarded_view = previous is None or (
        I.same_expr(destination.offset, previous[0].offset)
        and len(destination.shape) == len(previous[0].shape)
        and all(I.same_expr(a, b) for a, b in zip(destination.shape, previous[0].shape))
        and all(I.same_expr(a, b) for a, b in zip(destination.strides, previous[0].strides))
      )
      if (
        destination.owned
        # Replacing a forwarded whole buffer with a partial view would drop
        # its still-live padding/other elements. Materialize that old view
        # before the partial store, even when both views start at offset zero.
        and same_forwarded_view
        and mask is None
        and not self.control_depth
        and destination.offset.op == "literal"
        and destination.offset.data == 0
        and isinstance(value.type, I.TileType)
        and len(value.type.shape) == len(destination.shape)
        and all(I.same_expr(a, b) for a, b in zip(destination.shape, value.type.shape))
      ):
        # Keep storage-rounding even when forwarding an owned temporary.
        rounded = I.Expr(I.TileType(destination.dtype, destination.shape), "cast", (value,))
        self.promoted[destination.storage] = (destination, rounded)
        self.block.statements.append(I.BindTile(rounded))
        return None
      self.materialize(destination.storage)
      implementation = kwargs.get("implementation", "auto")
      if implementation not in {"auto", "vector", "sme", "gather"}:
        self.error(node, "transpose implementation must be auto, vector, sme or gather")
      self.block.statements.append(I.Store(destination, value, mask, implementation))
      return None
    if self.operations.has(operation, Trait.SCALAR_REDUCTION):
      if len(args) != 1:
        self.error(node, f"T.{operation} expects one computation value")
      value = self.value(args[0], node)
      unroll = self.unroll_count(kwargs.get("unroll", I.literal(2)), node)
      if not isinstance(value.type, I.TileType):
        return value
      axis = kwargs.get("axis")
      if axis is not None:
        axis = I.constant(self.value(axis, node))
        if not isinstance(axis, int) or not -len(value.type.shape) <= axis < len(value.type.shape):
          self.error(node, "reduction axis must be an in-range compile-time integer")
        axis %= len(value.type.shape)
        shape = value.type.shape[:axis] + value.type.shape[axis + 1 :]
        if shape:
          return I.Expr(
            I.TileType(value.type.dtype, shape),
            "axis_reduce",
            (value,),
            (axis, self.operations.get(operation).reduction, unroll),
          )
      return I.Expr(I.ScalarType(value.type.dtype), operation, (value,), unroll)
    if operation in {"min", "max", "ceildiv"}:
      if len(args) != 2:
        self.error(node, f"T.{operation} expects two elementwise operands; use T.reduce_{operation} for a reduction")
      a, b = self.value(args[0], node), self.value(args[1], node)
      if operation == "ceildiv":
        return I.Expr(a.type, "ceildiv", (a, b))
      return I.binary(operation, a, b)
    if self.operations.has(operation, Trait.UNARY_MATH):
      if len(args) != self.operations.get(operation).arity:
        self.error(node, f"T.{operation} expects one operand")
      value = self.value(args[0], node)
      accuracy = kwargs.get("accuracy", "strict")
      if accuracy not in {"strict", "fast", "estimate"}:
        self.error(node, "accuracy must be strict, fast or estimate")
      return I.Expr(value.type, operation, (value,), accuracy)
    if operation == "widening_dot":
      if len(args) != 2:
        self.error(node, "T.widening_dot expects two rank-one computation values")
      a, b = (self.value(v, node) for v in args)
      if any(not isinstance(v.type, I.TileType) or len(v.type.shape) != 1 for v in (a, b)):
        self.error(node, "T.widening_dot expects two rank-one computation values")
      extent = a.type.shape[0]
      from ..analysis.affine import canonical_integer

      if not I.same_expr(canonical_integer(extent), canonical_integer(b.type.shape[0])):
        if I.constant(extent) is not None and I.constant(b.type.shape[0]) is not None:
          self.error(node, "T.widening_dot inputs must have equal logical lengths")
        # Validation must execute even when the result is empty. A check
        # embedded only in its extent can disappear with the consumer loop.
        self.block.statements.append(
          I.Assert(I.binary("eq", extent, b.type.shape[0]), "widening_dot input length mismatch")
        )
      dtype = dtype_name(kwargs.get("dtype", "float32"))
      if a.type.dtype == b.type.dtype == "bfloat16" and dtype == "float32":
        group = 2
      elif a.type.dtype in {"int8", "uint8"} and b.type.dtype in {"int8", "uint8"} and dtype in {"int32", "uint32"}:
        group = 4
      else:
        self.error(node, "T.widening_dot supports BF16 x BF16 -> FP32 and signed/unsigned byte -> I32/U32")
      extent = I.binary("ceildiv", extent, I.literal(group))
      return I.Expr(I.TileType(dtype, (extent,)), "widening_dot", (a, b), group)
    if operation == "fma":
      a, b, c = (self.value(v, node) for v in args)
      shaped = any(isinstance(value.type, I.TileType) for value in (a, b, c))
      dtype = floating_dtype((a, b, c), shaped=shaped)
      if dtype is not None:
        a, b, c = cast_operands((a, b, c), dtype, I.Expr, I.shaped_type, shaped=shaped)
      typ = I.binary("add", I.binary("mul", a, b), c).type
      return I.Expr(typ, "fma", (a, b, c))
    if operation == "where":
      condition, a, b = (self.value(v, node) for v in args)
      return self.select(condition, a, b)
    if operation == "lanes":
      vector = args[0] if args else Vector()
      if not isinstance(vector, Vector):
        self.error(node, "T.lanes expects a Vector descriptor")
      return (
        I.literal(vector.N)
        if vector.N is not None
        else I.Expr(I.ScalarType("int64"), "lanes", data=(dtype_name(vector.dtype), vector.pow))
      )
    if operation == "arange":
      start, stop = (I.literal(0), args[0]) if len(args) == 1 else args[:2]
      return I.Expr(I.TileType("int64", (I.binary("sub", stop, start),)), "arange", (start,))
    if operation == "static_assert":
      value = self.value(args[0], node)
      if value.op != "literal" or not value.data:
        self.error(node, "static_assert condition is not proven true")
      return None
    if operation == "compiler_fence":
      if args or kwargs:
        self.error(node, "compiler_fence takes no arguments")
      self.block.statements.append(I.Control('asm volatile("" ::: "memory")'))
      return None
    self.error(node, f"unsupported DSL operation {operation!r}")

  def matrix_operand(self, value, node, *, allow_scalar=False):
    if isinstance(value, I.View):
      if value.storage in self.promoted:
        original, promoted = self.promoted[value.storage]
        if (
          I.same_expr(value.offset, original.offset)
          and len(value.shape) == len(original.shape)
          and all(I.same_expr(a, b) for a, b in zip(value.shape, original.shape))
        ):
          value = promoted
        else:
          self.materialize(value.storage)
      if isinstance(value, I.View):
        value = I.Expr(
          I.TileType(value.dtype, value.shape) if value.shape else I.ScalarType(value.dtype), "load", data=value
        )
    value = self.value(value, node)
    if not allow_scalar and (not isinstance(value.type, I.TileType) or len(value.type.shape) != 2):
      self.error(node, "dot/pack inputs must be rank-two views or tile expressions")
    return value

  def write_tile(self, output, value):
    if output.owned and not self.control_depth and I.constant(output.offset) == 0:
      rounded = I.Expr(I.TileType(output.dtype, output.shape), "cast", (value,))
      self.promoted[output.storage] = (output, rounded)
      self.block.statements.append(I.BindTile(rounded))
    else:
      self.materialize(output.storage)
      self.block.statements.append(I.Store(output, value))

  def select(self, condition, a, b):
    if isinstance(condition.type, I.TileType) and not condition.type.mask:
      condition = I.binary("ne", condition, I.literal(0))
    elif isinstance(condition.type, I.ScalarType) and condition.type.dtype != "bool":
      condition = I.Expr(I.ScalarType("bool"), "cast", (condition,))
    shaped = any(isinstance(value.type, I.TileType) for value in (condition, a, b))
    dtype = floating_dtype((a, b), shaped=shaped)
    if dtype is not None:
      a, b = cast_operands((a, b), dtype, I.Expr, I.shaped_type, shaped=shaped)
    typ = I.binary("add", a, b).type
    if dtype is not None and dtype != typ.dtype:
      typ = I.shaped_type(dtype, typ.shape, typ) if isinstance(typ, I.TileType) else I.ScalarType(dtype)
    if isinstance(condition.type, I.TileType):
      shape = I.broadcast_shape(typ.shape if isinstance(typ, I.TileType) else (), condition.type.shape)
      typ = I.shaped_type(typ.dtype, shape, typ if isinstance(typ, I.TileType) else condition.type)
    return I.Expr(typ, "where", (condition, a, b))

  def unroll_count(self, value, node):
    count = I.constant(self.value(value, node))
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
      self.error(node, "unroll must be a positive compile-time integer")
    if count > 1024:
      self.error(node, "unroll exceeds the 1024-copy code-size budget")
    return count

  def positive_extent(self, value):
    from ..analysis.affine import positive_extent

    return positive_extent(value, self.operations, self.positive_scalars)

  def assign(self, target, value, node):
    if isinstance(target, ast.Tuple):
      if not isinstance(value, tuple) or len(value) != len(target.elts):
        self.error(node, "tuple assignment arity mismatch")
      value = self.capture_assignment(value)
      for item, val in zip(target.elts, value):
        self.assign(item, val, node)
      return
    if isinstance(target, ast.Subscript):
      destination = self.expression(target)
      if not isinstance(destination, I.View) or not destination.writable:
        self.error(node, "assignment requires a writable view")
      self.materialize(destination.storage)
      self.block.statements.append(I.Store(destination, self.freeze_tile(self.value(value, node))))
      return
    if not isinstance(target, ast.Name):
      self.error(node, "only names, tuples and writable views can be assigned")
    existing = self.env.get(target.id)
    from .state import CarriedTuple

    if isinstance(existing, CarriedTuple):
      self.update_tuple(existing, self.capture_assignment(value), node)
      return
    if isinstance(existing, I.Expr) and existing.op == "state_ref" and target.id in self.mutable_names:
      self.check_state_type(existing.type, value, node)
      self.block.statements.append(I.UpdateState(existing.data, self.freeze_tile(value)))
      return
    if isinstance(value, I.Expr) and isinstance(value.type, I.ScalarType):
      existing = self.env.get(target.id)
      if value is existing:
        return  # A helper's as_value(parameter) is not a scalar mutation.
      if value.op == "literal" and not (isinstance(existing, I.Expr) and existing.op == "local"):
        # Preserve compile-time helper results/locals. Structured control
        # promotes an assigned constant to a runtime slot before entering it.
        self.env[target.id] = value
        return
      # Scalars assigned in control flow are loop-carried C++ locals.
      if isinstance(existing, I.Expr) and existing.op == "local":
        if existing.type.dtype != value.type.dtype:
          self.error(node, "loop-carried scalar dtype changes; use an explicit cast")
        name, declare = existing.data, False
      else:
        name, declare = self.unique(target.id), True
      self.block.statements.append(I.Assign(name, value, declare))
      if self.positive_extent(value):
        self.positive_scalars.add(name)
      else:
        self.positive_scalars.discard(name)
      self.env[target.id] = I.Expr(value.type, "local", data=name)
    else:
      if isinstance(value, I.Expr) and isinstance(value.type, I.TileType):
        value = self.freeze_tile(value)
        value = self.snapshot_state(value)
        self.block.statements.append(I.BindTile(value))
      self.env[target.id] = value

  def materialize(self, storage):
    if storage in self.promoted:
      view, value = self.promoted.pop(storage)
      self.block.statements.append(I.Store(view, value))

  def freeze_tile(self, value):
    """Capture mutable coordinates once per assignment, preserving DAG sharing.

    Memoization is local to this capture boundary: sharing a captured value
    between two assignments would incorrectly freeze a later scalar update.
    The expression graph itself is immutable during this traversal.
    """
    memo, capture_required = {}, {}

    def needs_freeze(expr):
      identity = id(expr)
      if identity not in capture_required:
        capture_required[identity] = self.operations.has(expr, Trait.CAPTURE_SCALAR) or any(
          needs_freeze(argument) for argument in expr.args
        )
      return capture_required[identity]

    def freeze(expr):
      identity = id(expr)
      if identity in memo:
        return memo[identity]
      if isinstance(expr.type, I.ScalarType) and expr.op != "frozen" and needs_freeze(expr):
        name = self.unique("captured")
        self.block.statements.append(I.Assign(name, expr))
        memo[identity] = I.Expr(expr.type, "frozen", data=name)
        return memo[identity]
      arguments = tuple(freeze(a) for a in expr.args)
      data = expr.data
      if self.operations.has(expr, Trait.MEMORY_READ):
        view = expr.data
        shape, strides, offset = (
          tuple(freeze(v) for v in view.shape),
          tuple(freeze(v) for v in view.strides),
          freeze(view.offset),
        )
        if (
          any(a is not b for a, b in zip(shape, view.shape))
          or any(a is not b for a, b in zip(strides, view.strides))
          or offset is not view.offset
        ):
          data = replace(view, shape=shape, strides=strides, offset=offset)
      typ = expr.type
      if isinstance(typ, I.TileType):
        shape = tuple(freeze(v) for v in typ.shape)
        if any(a is not b for a, b in zip(shape, typ.shape)):
          typ = I.shaped_type(typ.dtype, shape, typ, typ.mask, keep_tag=True)
      memo[identity] = (
        expr
        if all(a is b for a, b in zip(arguments, expr.args)) and data is expr.data and typ is expr.type
        else I.Expr(typ, expr.op, arguments, data)
      )
      return memo[identity]

    return freeze(value)

  def statements(self, nodes):
    for node in nodes:
      self.statement(node)
      if isinstance(node, ast.Return):
        break

  def nested(self, nodes):
    previous = self.block
    self.block = I.Block()
    self.control_depth += 1
    try:
      self.statements(nodes)
      return self.block
    finally:
      self.control_depth -= 1
      self.block = previous

  def verify_loop_environment(self, before, node, ignore=()):
    for name, value in before.items():
      if (
        name not in ignore
        and self.env.get(name) is not value
        and not (isinstance(value, I.Expr) and value.op in {"local", "state_ref"})
      ):
        self.error(node, "loop-carried view rebinding requires explicit storage; tile/vector values may carry state")

  def statement(self, node):
    block = self.block
    count = len(block.statements)
    result = self._statement(node)
    for statement in block.statements[count:]:
      self.origins.setdefault(
        id(statement), (self.body.filename, self.body.first_line + node.lineno - 1, node.col_offset)
      )
    return result

  def _statement(self, node):
    if isinstance(node, ast.With):
      if len(node.items) != 1:
        self.error(node, "hardware regions cannot bind host context-manager values")
      region = self.expression(node.items[0].context_expr)
      if isinstance(region, tuple) and region[0] == "team-region":
        self.team_region(node, region)
        return
      if node.items[0].optional_vars is not None:
        self.error(node, "hardware regions cannot bind host context-manager values")
      if not isinstance(region, tuple) or len(region) != 3 or region[0] != "hardware-region":
        self.error(node, "with requires T.sme.region or T.amx.region")
      _, kind, configuration = region
      if self.hardware_regions:
        self.error(node, "explicit nested hardware regions are not supported; use one encompassing region")
      # Raw tile memory instructions are an opaque memory-effect boundary.
      # Flush forwarded owned values before entry, never allocate/materialize
      # them inside a manual SME or AMX interval.
      for storage in tuple(self.promoted):
        self.materialize(storage)
      before = dict(self.env)
      self.hardware_regions.append((kind, configuration))
      try:
        body = self.nested(node.body)
      finally:
        self.hardware_regions.pop()
      self.validate_hardware_body(body, kind, node)
      self.env = before
      self.block.statements.append(I.HardwareRegion(kind, body, configuration))
    elif isinstance(node, ast.Assign):
      value = self.expression(node.value)
      for target in node.targets:
        self.assign(target, value, node)
    elif isinstance(node, ast.AnnAssign):
      if node.value:
        self.assign(node.target, self.expression(node.value), node)
    elif isinstance(node, ast.AugAssign):
      operator = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}.get(type(node.op))
      if operator is None:
        self.error(node, "unsupported augmented assignment")
      self.assign(
        node.target,
        I.binary(
          operator, self.value(self.expression(node.target), node), self.value(self.expression(node.value), node)
        ),
        node,
      )
    elif isinstance(node, ast.Expr):
      self.expression(node.value)
    elif isinstance(node, ast.For):
      for storage in tuple(self.promoted):
        self.materialize(storage)
      if node.orelse:
        self.error(node, "for ... else is not supported")
      loop = self.expression(node.iter)
      if isinstance(loop, tuple) and loop[0] in {"parallel", "simd"}:
        self.local_loop(node, loop)
        return
      if (
        not isinstance(loop, tuple)
        or len(loop) not in (2, 3)
        or loop[0] not in {"range", "serial", "unroll", "grid", "pipelined"}
      ):
        self.error(node, "kernel loops require range/T.serial/T.grid/T.unroll")
      kind, values = loop[:2]
      unroll = loop[2] if len(loop) == 3 else 2
      if kind == "grid":
        targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
        if len(targets) != len(values):
          self.error(node, "grid dimension/target mismatch")
        for target in targets:
          if not isinstance(target, ast.Name):
            self.error(node, "grid targets must be names")
        self.promote_control_state(node.body, node, ignore=tuple(t.id for t in targets))
        before = dict(self.env)

        names, extents = [], []
        for target, value in zip(targets, values):
          stop = self.value(value, node)
          if not isinstance(stop.type, I.ScalarType) or "int" not in stop.type.dtype:
            self.error(node, "grid bounds must be scalar integers")
          name = self.unique(target.id)
          self.env[target.id] = I.Expr(I.ScalarType("int64"), "local", data=name)
          names.append(name)
          extents.append(stop)

        self.loop_depth += 1
        body = self.nested(node.body)
        self.block.statements.append(I.Grid(tuple(names), tuple(extents), body, *unroll))
        self.loop_depth -= 1
        self.verify_loop_environment(before, node, tuple(t.id for t in targets))
        self.env = before
        return
      if not isinstance(node.target, ast.Name):
        self.error(node, "loop target must be a scalar name")
      values = [self.value(v, node) for v in values]
      if any(not isinstance(v.type, I.ScalarType) or "int" not in v.type.dtype for v in values):
        self.error(node, "loop bounds must be scalar integers")
      if not 1 <= len(values) <= 3:
        self.error(node, "range expects one to three arguments")
      start, stop, step = (
        (I.literal(0), values[0], I.literal(1))
        if len(values) == 1
        else (values[0], values[1], values[2] if len(values) == 3 else I.literal(1))
      )
      if step.op != "literal" and not self.positive_extent(step) or step.op == "literal" and step.data == 0:
        self.error(node, "loop step must be a nonzero constant or proven positive native extent")
      if kind == "unroll":
        if any(isinstance(n, (ast.Break, ast.Continue)) for s in node.body for n in ast.walk(s)):
          self.error(node, "break/continue in an unrolled region is not supported; use T.serial")
        if any(v.op != "literal" for v in (start, stop, step)):
          self.error(node, "unroll bounds must be compile-time constants")
        if len(range(start.data, stop.data, step.data)) > 1024:
          self.error(node, "unroll exceeds the 1024-iteration code-size budget")
        self.loop_depth += 1
        for value in range(start.data, stop.data, step.data):
          self.env[node.target.id] = I.literal(value)
          self.statements(node.body)
        self.loop_depth -= 1
      else:
        self.promote_control_state(node.body, node, ignore=(node.target.id,))
        name = self.unique(node.target.id)
        before = dict(self.env)
        self.env[node.target.id] = I.Expr(I.ScalarType("int64"), "local", data=name)
        self.loop_depth += 1
        body = self.nested(node.body)
        self.loop_depth -= 1
        self.verify_loop_environment(before, node, (node.target.id,))
        self.env = before
        if kind == "pipelined":
          self.block.statements.append(
            I.PipelineFor(I.For(name, start, stop, step, body, unroll[2]), *unroll[:2], unroll[3])
          )
        else:
          self.block.statements.append(I.For(name, start, stop, step, body, unroll))
    elif isinstance(node, ast.If):
      condition = self.value(self.expression(node.test), node)
      if not isinstance(condition.type, I.ScalarType):
        self.error(node, "tile predicates require T.where, not Python if")
      if condition.op == "literal":
        self.statements(node.body if condition.data else node.orelse)
      else:
        for storage in tuple(self.promoted):
          self.materialize(storage)
        self.promote_control_state(node.body + node.orelse, node)
        before = dict(self.env)
        body = self.nested(node.body)
        body_env = dict(self.env)
        self.env = dict(before)
        otherwise = self.nested(node.orelse)
        otherwise_env = dict(self.env)
        merged = {}
        for name in sorted(set(body_env) | set(otherwise_env)):
          if name not in before:
            a, b = body_env.get(name), otherwise_env.get(name)
            if name not in body_env or name not in otherwise_env:
              # A name defined on only one path is not exported from the
              # branch. It remains usable locally, but a later read diagnoses
              # an unbound name rather than emitting an uninitialized local.
              continue
            merged[name] = self.join_branch_values(a, b, body, otherwise, before, node)
          elif body_env.get(name) is not before.get(name) or otherwise_env.get(name) is not before.get(name):
            value = before.get(name)
            if not isinstance(value, I.Expr) or value.op not in {"local", "state_ref"}:
              self.error(node, "dynamic branches cannot rebind tile/view values; use T.where or explicit storage")
        self.env = before
        self.env.update(merged)
        self.block.statements.append(I.If(condition, body, otherwise))
    elif isinstance(node, ast.While):
      for storage in tuple(self.promoted):
        self.materialize(storage)
      if node.orelse:
        self.error(node, "while ... else is not supported")
      self.promote_control_state(node.body, node)
      condition = self.value(self.expression(node.test), node)
      if not isinstance(condition.type, I.ScalarType):
        self.error(node, "while condition must be scalar")
      self.loop_depth += 1
      before = dict(self.env)
      body = self.nested(node.body)
      self.loop_depth -= 1
      self.verify_loop_environment(before, node)
      self.env = before
      self.block.statements.append(I.While(condition, body))
    elif isinstance(node, ast.Assert):
      if node.msg is not None and (not isinstance(node.msg, ast.Constant) or not isinstance(node.msg.value, str)):
        self.error(node, "DSL assertion messages must be constant strings")
      condition = self.value(self.expression(node.test), node)
      if not isinstance(condition.type, I.ScalarType):
        self.error(node, "assert requires a scalar condition; reduce tile predicates explicitly")
      self.block.statements.append(
        I.Assert(condition, node.msg.value if isinstance(node.msg, ast.Constant) else "DSL assertion failed")
      )
    elif isinstance(node, (ast.Break, ast.Continue)):
      if not self.loop_depth:
        self.error(node, "break/continue outside loop")
      self.block.statements.append(I.Control("break" if isinstance(node, ast.Break) else "continue"))
    elif isinstance(node, ast.Return):
      if self.helper_exits:
        self.helper_return(self.expression(node.value) if node.value is not None else None, node)
        return
      if node.value is not None:
        self.error(node, "kernels return through output tensors; only T.func helpers return values")
      self.block.statements.append(I.Control("return"))
    elif not isinstance(node, ast.Pass):
      self.error(node, f"unsupported statement {type(node).__name__}")


def lower(ir, *, catalog=None):
  return Frontend(ir, catalog=catalog).compile()
