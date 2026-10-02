"""Python AST partial evaluation into structured kernel IR."""
from __future__ import annotations
import ast
from types import ModuleType
from ...dsl import DSLFunction, Intrinsic, Vector
from ...expr import Dynamic
from ...types import TensorType
from .. import ir as I
from ..dtypes import dtype_name
from .capture import Body, symbol
from .diagnostics import DSLError

class Frontend:
  def __init__(self, ir):
    self.body = ir.body
    self.parameters = {}
    self.bindings = {}
    self.env = dict(self.body.environment)
    self.block = I.Block()
    self.counter = 0
    self.loop_depth = 0
    self.helper_depth = 0
    self.control_depth = 0
    self.promoted = {}
    self.diagnostics = []
    for name, parameter in ir.signature.parameters.items():
      annotation = ir.annotations[name]
      self.parameters[name] = annotation
      if isinstance(annotation, TensorType):
        if annotation.optional or parameter.default is None:
          raise DSLError("optional DSL tensors are not implemented; select a template in the outer factory")
        shape = tuple(symbol(v) for v in annotation.shape)
        strides = tuple(symbol(v) for v in annotation.concrete_strides)
        self.env[name] = I.View(name, dtype_name(annotation.dtype), shape, strides, I.literal(0),
                                annotation.access != "input")
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
    self.statements(self.body.tree.body)
    return I.KernelIR(self.block, self.parameters, self.bindings, self.diagnostics)

  def value(self, obj, node):
    obj = symbol(obj)
    if not isinstance(obj, I.Expr):
      self.error(node, "expected a scalar or tile computation value")
    return obj

  def shape(self, obj, node):
    return tuple(self.value(v, node) for v in (obj if isinstance(obj, tuple) else (obj,)))

  def expression(self, node):
    if isinstance(node, ast.Constant):
      return symbol(node.value)
    if isinstance(node, ast.Name):
      if node.id in self.env:
        return self.env[node.id]
      if node.id in {"float", "int", "bool"}:
        return {"float": float, "int": int, "bool": bool}[node.id]
      self.error(node, f"unbound kernel name {node.id!r}")
    if isinstance(node, (ast.Tuple, ast.List)):
      if isinstance(node, ast.List):
        self.error(node, "use immutable tuples in kernel code")
      return tuple(self.expression(v) for v in node.elts)
    if isinstance(node, ast.Attribute):
      if isinstance(node.value, ast.Name) and f"{node.value.id}.{node.attr}" in self.env:
        return self.env[f"{node.value.id}.{node.attr}"]
      obj = self.expression(node.value)
      if isinstance(obj, ModuleType):
        value = getattr(obj, node.attr, None)
        if isinstance(value, DSLFunction):
          return capture(value.function)
        if isinstance(value, (Intrinsic, type)) or node.attr in {"float16", "bfloat16", "float32", "float64", "int32", "int64", "bool_"}:
          return value
        self.error(node, f"unsupported DSL attribute {node.attr!r}")
      if node.attr in {"shape", "strides"} and isinstance(obj, I.View):
        return getattr(obj, node.attr)
      if node.attr == "shape" and isinstance(obj, I.Expr) and isinstance(obj.type, I.TileType):
        return obj.type.shape
      if node.attr == "dtype" and isinstance(obj, (I.View, I.Expr)):
        return obj.dtype if isinstance(obj, I.View) else obj.type.dtype
      if node.attr == "T" and isinstance(obj, I.View):
        return I.View(obj.storage, obj.dtype, obj.shape[::-1], obj.strides[::-1], obj.offset, obj.writable, obj.owned)
      self.error(node, f"unsupported attribute {node.attr!r}")
    if isinstance(node, ast.BinOp):
      operators = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.FloorDiv: "floordiv", ast.Mod: "mod",
                   ast.BitAnd: "and", ast.BitOr: "or", ast.BitXor: "xor", ast.LShift: "shl", ast.RShift: "shr"}
      if isinstance(node.op, ast.Pow):
        exponent = self.expression(node.right)
        if isinstance(exponent, I.Expr) and I.constant(exponent) == 2:
          value = self.value(self.expression(node.left), node)
          return I.binary("mul", value, value)
        self.error(node, "only squared expressions are supported by **; use explicit DSL math")
      if type(node.op) not in operators:
        self.error(node, "unsupported binary operation")
      left, right = self.value(self.expression(node.left), node), self.value(self.expression(node.right), node)
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
        return I.literal(~value.data if isinstance(node.op, ast.Invert) and value.type.dtype != "bool" else -value.data if op == "neg" else not value.data if op == "not" else value.data)
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
      obj = self.expression(node.value)
      if isinstance(obj, tuple):
        index = I.constant(self.value(self.expression(node.slice), node))
        if not isinstance(index, int):
          self.error(node, "tuple indices must be compile-time integers")
        return obj[index]
      if not isinstance(obj, I.View):
        self.error(node, "only tensor/tile views and tuples can be indexed")
      items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
      shape, strides, offset = [], [], obj.offset
      axis = 0
      for item in items:
        if isinstance(item, ast.Constant) and item.value is Ellipsis:
          remaining = len(obj.shape) - (len(items) - 1)
          shape.extend(obj.shape[axis:axis + remaining]); strides.extend(obj.strides[axis:axis + remaining]); axis += remaining
          continue
        if axis >= len(obj.shape):
          self.error(node, "too many tensor indices")
        extent, stride = obj.shape[axis], obj.strides[axis]
        if isinstance(item, ast.Slice):
          if item.lower is None and item.upper is None and item.step is None:
            shape.append(extent); strides.append(stride); axis += 1
            continue
          step = self.value(self.expression(item.step), node) if item.step else I.literal(1)
          if step.op != "literal" or not isinstance(step.data, int) or isinstance(step.data, bool) or step.data <= 0:
            self.error(node, "slice steps must be positive constants; negative tensor strides are supported")
          start = self.value(self.expression(item.lower), node) if item.lower else I.literal(0)
          stop = self.value(self.expression(item.upper), node) if item.upper else extent
          if any(not isinstance(v.type, I.ScalarType) or "int" not in v.type.dtype for v in (start, stop)):
            self.error(node, "slice bounds must be scalar integers")
          # Region slices clip to bounds. Negative Python indices are normalized.
          if I.constant(start) is not None and I.constant(start) < 0:
            start = I.binary("add", extent, start)
          if I.constant(stop) is not None and I.constant(stop) < 0:
            stop = I.binary("add", extent, stop)
          start = I.binary("max", I.literal(0), I.binary("min", start, extent))
          clipped = I.binary("min", stop, extent)
          # Native lane extents are positive; preserve a canonical min for
          # descriptor loads and matching region stores.
          if I.constant(start) == 0 and stop.op == "lanes":
            stop = clipped
          else:
            stop = I.binary("max", start, clipped)
          length = I.binary("floordiv", I.binary("add", I.binary("sub", stop, start), I.binary("sub", step, I.literal(1))), step)
          offset = I.binary("add", offset, I.binary("mul", start, stride))
          shape.append(length); strides.append(I.binary("mul", stride, step))
        else:
          index = self.value(self.expression(item), node)
          if not isinstance(index.type, I.ScalarType) or "int" not in index.type.dtype:
            self.error(node, "view indices must be scalar integers; use strided regions for vector access")
          if I.constant(index) is not None and I.constant(index) < 0:
            index = I.binary("add", extent, index)
          offset = I.binary("add", offset, I.binary("mul", index, stride))
        axis += 1
      shape.extend(obj.shape[axis:]); strides.extend(obj.strides[axis:])
      return I.View(obj.storage, obj.dtype, tuple(shape), tuple(strides), offset, obj.writable, obj.owned)
    if isinstance(node, ast.Call):
      return self.call(node)
    self.error(node, f"unsupported expression {type(node).__name__}")

  def call(self, node):
    if isinstance(node.func, ast.Name) and node.func.id in {"range", "float", "int", "bool", "len", "min", "max", "abs"}:
      operation = node.func.id
    else:
      function = self.expression(node.func)
      if isinstance(function, Body):
        return self.helper(function, node)
      if function is Vector:
        args = [self.expression(a) for a in node.args]
        kwargs = {k.arg: self.expression(k.value) for k in node.keywords}
        args = [v.data if isinstance(v, I.Expr) and v.op == "literal" else v for v in args]
        kwargs = {k: v.data if isinstance(v, I.Expr) and v.op == "literal" else v for k, v in kwargs.items()}
        return Vector(*args, **kwargs)
      if not isinstance(function, Intrinsic):
        self.error(node, "arbitrary Python calls are not allowed; use T.func or a DSL intrinsic")
      operation = function.name
    args = [self.expression(a) for a in node.args]
    kwargs = {k.arg: self.expression(k.value) for k in node.keywords}
    if any(k.arg is None for k in node.keywords):
      self.error(node, "runtime keyword unpacking is not supported")
    allowed = {
      "load": {"dtype", "vector", "mask", "other"}, "store": {"mask"}, "copy": {"mask"},
      "alloc": {"dtype"}, "zeros": {"dtype"}, "full": {"dtype"}, "cast": {"dtype"},
      "transpose": {"axes"}, "serial": {"unroll"},
      **{name: {"axis", "unroll"} for name in ("reduce_add", "reduce_min", "reduce_max")},
      **{name: {"accuracy"} for name in ("exp", "exp2", "exp_neg", "exp2_neg", "log", "log2", "rsqrt", "rcp")},
    }.get(operation, set())
    if set(kwargs) - allowed:
      self.error(node, f"unsupported {operation} keyword(s): {', '.join(sorted(set(kwargs) - allowed))}")
    if operation in {"range", "serial", "grid", "unroll"}:
      if operation == "serial":
        return (operation, args, self.unroll_count(kwargs.get("unroll", I.literal(2)), node))
      return (operation, args)
    if operation == "len":
      return args[0].shape[0] if isinstance(args[0], I.View) else I.literal(len(args[0]))
    if operation in {"float", "int", "bool", "cast"}:
      dtype = {"float": "float64", "int": "int64", "bool": "bool"}.get(operation)
      if dtype is None:
        dtype = dtype_name(args[1] if len(args) > 1 else kwargs["dtype"])
      value = self.value(args[0], node)
      typ = I.shaped_type(dtype, value.type.shape, value.type) if isinstance(value.type, I.TileType) else I.ScalarType(dtype)
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
      axes = args[1] if len(args) > 1 else kwargs.get("axes", tuple(I.literal(i) for i in reversed(range(len(view.shape)))))
      axes = tuple(I.constant(v) for v in axes)
      if sorted(axes) != list(range(len(view.shape))):
        self.error(node, "transpose axes must be a permutation")
      self.diagnostics.append("TODO: specialized transpose kernel; using gather/scatter")
      return I.View(view.storage, view.dtype, tuple(view.shape[i] for i in axes), tuple(view.strides[i] for i in axes), view.offset, view.writable, view.owned)
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
      return I.View(view.storage, view.dtype, shape, tuple(strides), view.offset, view.writable, view.owned)
    if operation == "load":
      view = args[0]
      if not isinstance(view, I.View):
        self.error(node, "T.load expects a tensor/tile region")
      dtype = dtype_name(kwargs.get("dtype", args[1] if len(args) > 1 else view.dtype))
      if view.storage in self.promoted:
        original, value = self.promoted[view.storage]
        if ("mask" not in kwargs and I.same_expr(view.offset, original.offset) and len(view.shape) == len(original.shape)
            and all(I.same_expr(a, b) for a, b in zip(view.shape, original.shape))
            and all(I.same_expr(a, b) for a, b in zip(view.strides, original.strides))):
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
        requested = I.literal(vector.N) if vector.N is not None else I.Expr(I.ScalarType("int64"), "lanes", data=(dtype, vector.pow))
        shape = (I.binary("min", shape[0], requested),)
        view = I.View(view.storage, view.dtype, shape, view.strides, view.offset, view.writable, view.owned)
      typ = I.VectorType(dtype, shape, tag=I.VectorTag(vector.N, vector.pow, basis_dtype=dtype)) if vector is not None else I.TileType(dtype, shape) if shape else I.ScalarType(dtype)
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
        memory_args = (mask, I.Expr(I.shaped_type(dtype, other.type.shape, other.type), "cast", (other,)) if isinstance(other.type, I.TileType) else I.Expr(I.ScalarType(dtype), "cast", (other,)))
      elif "other" in kwargs:
        self.error(node, "load other requires a mask")
      return I.Expr(typ, "load", memory_args, view)
    if operation in {"alloc", "zeros", "full"}:
      shape = self.shape(args[0], node)
      dtype = dtype_name(kwargs.get("dtype", args[1] if len(args) > 1 and operation != "full" else "float32"))
      if operation != "alloc":
        value = self.value(args[1] if operation == "full" else I.literal(0), node)
        if operation == "full":
          dtype = dtype_name(kwargs.get("dtype", args[2] if len(args) > 2 else "float32"))
        return I.Expr(I.TileType(dtype, shape), "fill", (value,))
      strides = [I.literal(1)] * len(shape)
      for i in reversed(range(len(shape) - 1)):
        strides[i] = I.binary("mul", strides[i + 1], shape[i + 1])
      view = I.View(self.unique("scratch"), dtype, shape, tuple(strides), I.literal(0), owned=True)
      self.block.statements.append(I.Allocate(view))
      return view
    if operation in {"store", "copy"}:
      destination = args[0] if operation == "store" else args[1]
      if operation == "copy":
        source = args[0]
        if not isinstance(source, I.View):
          self.error(node, "copy source must be a memory view")
        value = None
        if source.storage in self.promoted:
          original, promoted = self.promoted[source.storage]
          if (I.same_expr(source.offset, original.offset) and len(source.shape) == len(original.shape)
              and all(I.same_expr(a, b) for a, b in zip(source.shape, original.shape))
              and all(I.same_expr(a, b) for a, b in zip(source.strides, original.strides))):
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
      if destination.owned and mask is None and not self.control_depth and destination.offset.op == "literal" and destination.offset.data == 0 and isinstance(value.type, I.TileType) and len(value.type.shape) == len(destination.shape) and all(I.same_expr(a, b) for a, b in zip(destination.shape, value.type.shape)):
        # Keep storage-rounding even when forwarding an owned temporary.
        rounded = I.Expr(I.TileType(destination.dtype, destination.shape), "cast", (value,))
        self.promoted[destination.storage] = (destination, rounded)
        self.block.statements.append(I.BindTile(rounded))
        return None
      self.materialize(destination.storage)
      self.block.statements.append(I.Store(destination, value, mask))
      return None
    if operation in {"reduce_add", "reduce_max", "reduce_min"}:
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
        shape = value.type.shape[:axis] + value.type.shape[axis + 1:]
        if shape:
          return I.Expr(I.TileType(value.type.dtype, shape), "axis_reduce", (value,), (axis, operation[7:], unroll))
      return I.Expr(I.ScalarType(value.type.dtype), operation, (value,), unroll)
    if operation in {"min", "max", "ceildiv"}:
      if len(args) != 2:
        self.error(node, f"T.{operation} expects two elementwise operands; use T.reduce_{operation} for a reduction")
      a, b = self.value(args[0], node), self.value(args[1], node)
      if operation == "ceildiv":
        return I.Expr(a.type, "ceildiv", (a, b))
      return I.binary(operation, a, b)
    if operation in {"exp", "exp2", "exp_neg", "exp2_neg", "log", "log2", "sqrt", "rsqrt", "rcp", "abs"}:
      value = self.value(args[0], node)
      accuracy = kwargs.get("accuracy", "strict")
      if accuracy not in {"strict", "fast", "estimate"}:
        self.error(node, "accuracy must be strict, fast or estimate")
      return I.Expr(value.type, operation, (value,), accuracy)
    if operation == "fma":
      a, b, c = (self.value(v, node) for v in args)
      typ = I.binary("add", I.binary("mul", a, b), c).type
      return I.Expr(typ, "fma", (a, b, c))
    if operation == "where":
      condition, a, b = (self.value(v, node) for v in args)
      return self.select(condition, a, b)
    if operation == "lanes":
      vector = args[0] if args else Vector()
      if not isinstance(vector, Vector):
        self.error(node, "T.lanes expects a Vector descriptor")
      return I.literal(vector.N) if vector.N is not None else I.Expr(I.ScalarType("int64"), "lanes", data=(dtype_name(vector.dtype), vector.pow))
    if operation == "arange":
      start, stop = (I.literal(0), args[0]) if len(args) == 1 else args[:2]
      return I.Expr(I.TileType("int64", (I.binary("sub", stop, start),)), "arange", (start,))
    if operation == "static_assert":
      value = self.value(args[0], node)
      if value.op != "literal" or not value.data:
        self.error(node, "static_assert condition is not proven true")
      return None
    self.error(node, f"unsupported DSL operation {operation!r}")

  def select(self, condition, a, b):
    if isinstance(condition.type, I.TileType) and not condition.type.mask:
      condition = I.binary("ne", condition, I.literal(0))
    elif isinstance(condition.type, I.ScalarType) and condition.type.dtype != "bool":
      condition = I.Expr(I.ScalarType("bool"), "cast", (condition,))
    typ = I.binary("add", a, b).type
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

  def helper(self, body, node):
    if body.tree.args.vararg or body.tree.args.kwarg or body.tree.args.posonlyargs:
      self.error(node, "DSL helpers currently require named positional/keyword-only parameters")
    if self.helper_depth >= 32:
      self.error(node, "DSL helper expansion depth exceeds 32")
    positional = [a.arg for a in body.tree.args.args]
    parameters = positional + [a.arg for a in body.tree.args.kwonlyargs]
    args = [self.expression(a) for a in node.args]
    kwargs = {k.arg: self.expression(k.value) for k in node.keywords}
    if len(args) > len(positional):
      self.error(node, "too many helper arguments")
    bound = dict(zip(parameters, args))
    for name, value in kwargs.items():
      if name not in parameters or name in bound:
        self.error(node, "invalid or duplicate helper argument")
      bound[name] = value
    defaults = dict(zip(positional[len(positional) - len(body.tree.args.defaults):], body.tree.args.defaults))
    defaults.update({a.arg: v for a, v in zip(body.tree.args.kwonlyargs, body.tree.args.kw_defaults) if v is not None})
    previous_env, previous_body = self.env, self.body
    self.env = dict(body.environment)
    self.env.update(bound)
    self.body = body
    self.helper_depth += 1
    try:
      for name in parameters:
        if name not in self.env:
          if name not in defaults:
            self.error(node, f"missing helper argument {name!r}")
          self.env[name] = self.expression(defaults[name])
      result = None
      for statement in body.tree.body:
        if isinstance(statement, ast.Return):
          result = self.expression(statement.value) if statement.value else None
          break
        self.statement(statement)
      return result
    finally:
      self.helper_depth -= 1
      self.env, self.body = previous_env, previous_body

  def assign(self, target, value, node):
    if isinstance(target, ast.Tuple):
      if not isinstance(value, tuple) or len(value) != len(target.elts):
        self.error(node, "tuple assignment arity mismatch")
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
    if isinstance(value, I.Expr) and isinstance(value.type, I.ScalarType):
      existing = self.env.get(target.id)
      # Scalars assigned in control flow are loop-carried C++ locals.
      if isinstance(existing, I.Expr) and existing.op == "local":
        if existing.type.dtype != value.type.dtype:
          self.error(node, "loop-carried scalar dtype changes; use an explicit cast")
        name, declare = existing.data, False
      else:
        name, declare = self.unique(target.id), True
      self.block.statements.append(I.Assign(name, value, declare))
      self.env[target.id] = I.Expr(value.type, "local", data=name)
    else:
      if isinstance(value, I.Expr) and isinstance(value.type, I.TileType):
        value = self.freeze_tile(value)
        self.block.statements.append(I.BindTile(value))
      self.env[target.id] = value

  def materialize(self, storage):
    if storage in self.promoted:
      view, value = self.promoted.pop(storage)
      self.block.statements.append(I.Store(view, value))

  def freeze_tile(self, value):
    """A delayed computation captures scalar/coordinate values, not mutable locals."""
    copies = {}
    def needs_freeze(expr):
      return (expr.op in {"local", "load"} or expr.op.startswith("reduce_")
              or any(needs_freeze(a) for a in expr.args))
    def freeze(expr):
      if isinstance(expr.type, I.ScalarType) and expr.op != "frozen" and needs_freeze(expr):
        if id(expr) not in copies:
          name = self.unique("captured")
          self.block.statements.append(I.Assign(name, expr))
          copies[id(expr)] = I.Expr(expr.type, "frozen", data=name)
        return copies[id(expr)]
      arguments = tuple(freeze(a) for a in expr.args)
      data = expr.data
      if expr.op == "load":
        view = expr.data
        shape, strides, offset = tuple(freeze(v) for v in view.shape), tuple(freeze(v) for v in view.strides), freeze(view.offset)
        if shape != view.shape or strides != view.strides or offset is not view.offset:
          data = I.View(view.storage, view.dtype, shape, strides, offset, view.writable, view.owned)
      if arguments == expr.args and data is expr.data:
        return expr
      typ = expr.type
      if isinstance(typ, I.TileType):
        typ = I.shaped_type(typ.dtype, tuple(freeze(v) for v in typ.shape), typ, typ.mask, keep_tag=True)
      return I.Expr(typ, expr.op, arguments, data)
    return freeze(value)

  def statements(self, nodes):
    for node in nodes:
      self.statement(node)

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
      if name not in ignore and self.env.get(name) is not value and not (isinstance(value, I.Expr) and value.op == "local"):
        self.error(node, "loop-carried tile/view rebinding requires explicit T.alloc storage")

  def statement(self, node):
    if isinstance(node, ast.Assign):
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
      self.assign(node.target, I.binary(operator, self.value(self.expression(node.target), node), self.value(self.expression(node.value), node)), node)
    elif isinstance(node, ast.Expr):
      self.expression(node.value)
    elif isinstance(node, ast.For):
      for storage in tuple(self.promoted):
        self.materialize(storage)
      if node.orelse:
        self.error(node, "for ... else is not supported")
      loop = self.expression(node.iter)
      if not isinstance(loop, tuple) or len(loop) not in (2, 3) or loop[0] not in {"range", "serial", "unroll", "grid"}:
        self.error(node, "kernel loops require range/T.serial/T.grid/T.unroll")
      kind, values = loop[:2]
      unroll = loop[2] if len(loop) == 3 else 2
      if kind == "grid":
        before = dict(self.env)
        targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
        if len(targets) != len(values):
          self.error(node, "grid dimension/target mismatch")
        for target in targets:
          if not isinstance(target, ast.Name):
            self.error(node, "grid targets must be names")
        def build(axis):
          if axis == len(targets):
            return self.nested(node.body)
          target, stop = targets[axis], self.value(values[axis], node)
          if not isinstance(stop.type, I.ScalarType) or "int" not in stop.type.dtype:
            self.error(node, "grid bounds must be scalar integers")
          name = self.unique(target.id)
          old = self.env.get(target.id)
          self.env[target.id] = I.Expr(I.ScalarType("int64"), "local", data=name)
          body = I.Block([I.For(name, I.literal(0), stop, I.literal(1), build(axis + 1))])
          if old is None: self.env.pop(target.id, None)
          else: self.env[target.id] = old
          return body
        self.loop_depth += 1
        self.block.statements.extend(build(0).statements)
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
      start, stop, step = (I.literal(0), values[0], I.literal(1)) if len(values) == 1 else (values[0], values[1], values[2] if len(values) == 3 else I.literal(1))
      if step.op != "literal" or step.data == 0:
        self.error(node, "loop step must be a nonzero constant")
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
        name = self.unique(node.target.id)
        before = dict(self.env)
        self.env[node.target.id] = I.Expr(I.ScalarType("int64"), "local", data=name)
        self.loop_depth += 1
        body = self.nested(node.body)
        self.loop_depth -= 1
        self.verify_loop_environment(before, node, (node.target.id,))
        self.env = before
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
        before = dict(self.env)
        body = self.nested(node.body)
        body_env = dict(self.env)
        self.env = dict(before)
        otherwise = self.nested(node.orelse)
        for name in set(body_env) | set(self.env):
          if name not in before:
            self.error(node, "initialize branch-carried variables before a dynamic if")
          if body_env.get(name) is not before.get(name) or self.env.get(name) is not before.get(name):
            value = before.get(name)
            if not isinstance(value, I.Expr) or value.op != "local":
              self.error(node, "dynamic branches cannot rebind tile/view values; use T.where or explicit storage")
        self.env = before
        self.block.statements.append(I.If(condition, body, otherwise))
    elif isinstance(node, ast.While):
      for storage in tuple(self.promoted):
        self.materialize(storage)
      if node.orelse:
        self.error(node, "while ... else is not supported")
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
      self.block.statements.append(I.Assert(condition, node.msg.value if isinstance(node.msg, ast.Constant) else "DSL assertion failed"))
    elif isinstance(node, (ast.Break, ast.Continue)):
      if not self.loop_depth:
        self.error(node, "break/continue outside loop")
      self.block.statements.append(I.Control("break" if isinstance(node, ast.Break) else "continue"))
    elif isinstance(node, ast.Return):
      if node.value is not None:
        self.error(node, "kernels return through output tensors; only T.func helpers return values")
      self.block.statements.append(I.Control("return"))
    elif not isinstance(node, ast.Pass):
      self.error(node, f"unsupported statement {type(node).__name__}")


def lower(ir):
  return Frontend(ir).compile()
