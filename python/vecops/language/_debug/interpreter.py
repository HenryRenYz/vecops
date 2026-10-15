# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Small canonical-IR reference interpreter with checked scalar-memory access.

This is a debugging reference, not an ISA emulator. Unsupported intrinsics,
hardware state, atomics and collectives fail explicitly instead of pretending
to reproduce native ordering/approximation behavior.
"""

import ctypes
import math

from ..._dtype import bfloat16, framework_dtype, normalize_dtype
from .._compiler import ir as I
from .._compiler.operations import Trait, current


class UnsupportedInterpretation(NotImplementedError):
  pass


class _HelperExit(Exception):
  def __init__(self, region):
    self.region = region


class Interpreter:
  def __init__(self, kernel, arguments, *, vector_lanes=16, max_steps=1000000):
    global np
    import numpy as np

    self.kernel, self.arguments = kernel, arguments
    self.environment = dict(arguments)
    self.states = {}
    self.vector_lanes, self.max_steps, self.steps = vector_lanes, max_steps, 0
    for name, expression in kernel.bindings.items():
      self.environment[name] = self.value(expression)

  def step(self):
    self.steps += 1
    if self.steps > self.max_steps:
      raise RuntimeError("interpreter step budget exceeded (possible nonterminating loop)")

  def view(self, view):
    root = self.arguments.get(view.storage, self.states.get(view.storage))
    if root is None:
      raise UnsupportedInterpretation(f"unbound memory storage: {view.storage}")
    root = np.asarray(root)
    shape = tuple(int(self.value(n)) for n in view.shape)
    strides = tuple(int(self.value(n)) * root.itemsize for n in view.strides)
    offset = int(self.value(view.offset)) * root.itemsize
    if any(n < 0 for n in shape):
      raise ValueError("negative interpreted view shape")
    if not math.prod(shape):
      return np.empty(shape, dtype=root.dtype)
    origin = root.__array_interface__["data"][0]
    low = sum(min(0, (n - 1) * s) for n, s in zip(root.shape, root.strides))
    high = sum(max(0, (n - 1) * s) for n, s in zip(root.shape, root.strides)) + root.itemsize
    view_low = offset + sum(min(0, (n - 1) * s) for n, s in zip(shape, strides))
    view_high = offset + sum(max(0, (n - 1) * s) for n, s in zip(shape, strides)) + root.itemsize
    if not low <= view_low <= view_high <= high:
      raise IndexError(f"out-of-bounds interpreted access to {view.storage}")
    storage = (ctypes.c_ubyte * (high - low)).from_address(origin + low)
    return np.ndarray(shape, dtype=root.dtype, buffer=storage, offset=offset - low, strides=strides)

  def dtype(self, name):
    if name == "bfloat16":
      raise UnsupportedInterpretation("NumPy reference interpreter has no bfloat16 dtype; use FP32 reference inputs")
    return framework_dtype(name, "numpy")

  def value(self, expression):
    if not isinstance(expression, I.Expr):
      return expression
    op = expression.op
    if op == "view_disjoint":

      def span(view):
        array = self.view(view)
        if not array.size:
          return None
        pointer = array.__array_interface__["data"][0]
        low = sum(min(0, (n - 1) * s) for n, s in zip(array.shape, array.strides))
        high = sum(max(0, (n - 1) * s) for n, s in zip(array.shape, array.strides)) + array.itemsize
        return pointer + low, pointer + high

      a, b = map(span, expression.data)
      return a is None or b is None or a[1] <= b[0] or b[1] <= a[0]
    if op == "load":
      return self.view(expression.data).copy()
    if op in ("symbol", "local", "frozen", "raw"):
      if expression.data in self.environment:
        return self.environment[expression.data]
      import re

      match = re.fullmatch(r"(\w+)\.(size|stride)\((\d+)\)", expression.data)
      if match:
        root = self.arguments[match[1]]
        axis = int(match[3])
        return root.shape[axis] if match[2] == "size" else root.strides[axis] // root.itemsize
      raise UnsupportedInterpretation(f"raw expression: {expression.data}")
    if op == "literal":
      return expression.data
    if op == "register_input":
      return self.environment["__register_inputs"][expression.data]
    if op == "state_ref":
      return self.states[expression.data.name].copy()
    if op == "native_lanes":
      return self.vector_lanes
    args = tuple(self.value(a) for a in expression.args)
    if op == "ceildiv":
      return -(-args[0] // args[1])
    contract = current().get(expression)
    if contract.traits & Trait.SCALAR_REDUCTION:
      operation = contract.reduction
      function = {"add": np.sum, "min": np.min, "max": np.max}.get(operation)
      if function is None:
        raise UnsupportedInterpretation(op)
      if operation == "add":
        return function(args[0], dtype=self.dtype(expression.type.dtype))
      return function(args[0])
    if op == "axis_reduce":
      axis, operation, _ = expression.data
      return {"add": np.sum, "min": np.min, "max": np.max}[operation](args[0], axis=axis)
    if op in ("cast", "convert"):
      return np.asarray(args[0], dtype=self.dtype(expression.type.dtype))
    if op == "widening_dot":
      # Wide integer temporaries preserve each input's signedness. A final
      # unsigned cast specifies the destination-width modulo arithmetic.
      group = expression.data
      integer = expression.type.dtype in {"int32", "uint32"}
      dtype = np.int64 if integer else np.float32
      a, b = (np.asarray(v, dtype=dtype) for v in args)
      if a.shape != b.shape:
        raise ValueError("widening_dot input length mismatch")
      padding = (-a.size) % group
      product = np.pad(a * b, (0, padding)).reshape(-1, group).sum(axis=1, dtype=dtype)
      if integer:
        product = (product & 0xFFFFFFFF).astype(np.uint32)
        return product.view(np.int32) if expression.type.dtype == "int32" else product
      return product
    if op.startswith("sve."):
      operation = op.removeprefix("sve.")
      dtype = self.dtype(expression.type.dtype)
      if operation == "iota":
        return np.arange(self.vector_lanes * 4 // np.dtype(dtype).itemsize, dtype=dtype)
      if operation == "join_k_groups":
        a, b = (np.ascontiguousarray(v).view(np.uint32) for v in args)
        if a.shape != b.shape or a.size % 4:
          raise ValueError("SVE inputs must be matching complete 128-bit segments")
        joined = np.empty_like(a)
        joined[0::2], joined[1::2] = a[:a.size // 2], b[:b.size // 2]
        return joined.view(dtype)
      if operation == "broadcast_segment":
        source = np.ascontiguousarray(args[0]).view(np.uint8)
        start = expression.data * 16
        if source.size % 16 or start + 16 > source.size:
          raise ValueError("SVE segment index exceeds ordinary VL")
        return np.tile(source[start:start + 16], source.size // 16).view(dtype)
      if operation == "row_pair":
        source = np.ascontiguousarray(args[0]).view(np.uint64)
        selected = source[expression.data::2]
        return np.concatenate((selected, selected)).view(dtype)
      a, b, acc = args
      group = 4 if expression.args[0].type.dtype == "bfloat16" else 8
      if a.size != b.size or a.size != acc.size * group // 2 or acc.size % 4:
        raise ValueError("SVE MMLA operands must occupy matching complete registers")
      integer = expression.type.dtype == "int32"
      wide = np.int64 if integer else np.float32
      a = np.asarray(a, dtype=wide).reshape(-1, 2, group)
      b = np.asarray(b, dtype=wide).reshape(-1, 2, group)
      products = np.matmul(a, np.transpose(b, (0, 2, 1)))
      result = products.reshape(-1) + np.asarray(acc, dtype=wide)
      if integer:
        return (result.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
      return result.astype(dtype)
    if op == "bitreinterpret":
      source = expression.args[0]
      return np.asarray(self.value(source), dtype=self.dtype(source.type.dtype)).view(self.dtype(expression.type.dtype))
    if op == "bitcast":
      return np.asarray(args[0]).view(self.dtype(expression.type.dtype))
    if op in ("lower", "upper"):
      half = len(args[0]) // 2
      return args[0][:half] if op == "lower" else args[0][half:]
    if op == "concat":
      return np.concatenate(args)
    if op == "interleave":
      return np.stack(args, axis=-1).reshape(-1)
    if op in ("local_interleave_lower", "local_interleave_upper"):
      width = 16 // np.asarray(args[0]).dtype.itemsize
      half = slice(0, width // 2) if op.endswith("lower") else slice(width // 2, width)
      a, b = (np.asarray(v).reshape(-1, width)[:, half] for v in args)
      return np.stack((a, b), axis=-1).reshape(-1)
    if op == "block_shuffle":
      width = 16 // np.asarray(args[0]).dtype.itemsize
      blocks = np.concatenate([np.asarray(v).reshape(-1, width) for v in args])
      return blocks[list(expression.data)].reshape(-1)
    if op in ("where", "select"):
      return np.where(*args)
    if op == "element_at":
      return args[0][tuple(int(a) for a in args[1:])]
    if op == "value_transform" and expression.data[0] == "slice":
      kept = expression.data[1]
      shape = tuple(int(self.value(n)) for n in expression.type.shape)
      indices = tuple(
        slice(int(start), int(start) + shape[kept.index(axis)]) if axis in kept else int(start)
        for axis, start in enumerate(args[1:])
      )
      return args[0][indices]
    if op == "reshape":
      return np.reshape(args[0], tuple(int(self.value(n)) for n in expression.type.shape))
    if op == "broadcast" or op == "value_transform" and expression.data[0] == "broadcast_to":
      return np.broadcast_to(args[0], tuple(int(self.value(n)) for n in expression.type.shape))
    if op == "dot":
      result = args[0].astype(self.dtype(expression.type.dtype)) @ args[1].astype(self.dtype(expression.type.dtype))
      return result + args[2] if len(args) > 2 else result
    functions = {
      "exp": np.exp,
      "exp2": np.exp2,
      "exp_neg": np.exp,
      "exp2_neg": np.exp2,
      "log": np.log,
      "log2": np.log2,
      "abs": np.abs,
      "sqrt": np.sqrt,
      "rsqrt": lambda x: 1 / np.sqrt(x),
      "rcp": lambda x: 1 / x,
      "neg": np.negative,
      "add": np.add,
      "sub": np.subtract,
      "mul": np.multiply,
      "div": np.divide,
      "floordiv": np.floor_divide,
      "mod": np.mod,
      "min": np.minimum,
      "max": np.maximum,
      "lt": np.less,
      "le": np.less_equal,
      "gt": np.greater,
      "ge": np.greater_equal,
      "eq": np.equal,
      "ne": np.not_equal,
      "and": np.logical_and,
      "or": np.logical_or,
      "not": np.logical_not,
      "bit_and": np.bitwise_and,
      "bit_or": np.bitwise_or,
      "bit_xor": np.bitwise_xor,
      "shl": np.left_shift,
      "shr": np.right_shift,
    }
    if op in functions:
      return functions[op](*args)
    if contract.fold is not None:
      return contract.fold(*args)
    if op == "fill":
      return np.full(
        tuple(int(self.value(n)) for n in expression.type.shape), args[0], dtype=self.dtype(expression.type.dtype)
      )
    raise UnsupportedInterpretation(f"unsupported interpreter operation: {op}")

  def block(self, block):
    from .._compiler.ir.pipeline import PipelineFor

    for statement in block.statements:
      self.step()
      if isinstance(statement, I.Assign):
        self.environment[statement.target] = self.value(statement.value)
      elif isinstance(statement, (I.BindTile, I.Print)):
        # Printing diagnostics are not part of numerical reference results.
        continue
      elif isinstance(statement, I.Store):
        destination = self.view(statement.destination)
        result = self.value(statement.value)
        if statement.mask is None:
          destination[...] = result
        else:
          np.copyto(destination, result, where=self.value(statement.mask))
      elif isinstance(statement, I.DeclareState):
        self.states[statement.state.name] = np.asarray(self.value(statement.value)).copy()
      elif isinstance(statement, I.UpdateState):
        self.states[statement.state.name][...] = self.value(statement.value)
      elif isinstance(statement, I.ExitRegion):
        try:
          self.block(statement.body)
        except _HelperExit as result:
          if result.region != statement.name:
            raise
      elif isinstance(statement, I.Exit):
        raise _HelperExit(statement.region)
      elif isinstance(statement, I.Allocate):
        self.states[statement.view.storage] = np.empty(
          tuple(int(self.value(n)) for n in statement.view.shape), dtype=self.dtype(statement.view.dtype)
        )
      elif isinstance(statement, I.If):
        self.block(statement.body if self.value(statement.condition) else statement.otherwise)
      elif isinstance(statement, I.Assert):
        if not self.value(statement.condition):
          raise AssertionError(statement.message)
      elif isinstance(statement, I.While):
        while self.value(statement.condition):
          self.step()
          self.block(statement.body)
      elif isinstance(statement, (I.For, I.UnrolledFor, PipelineFor)):
        loop = statement if isinstance(statement, I.For) else statement.loop
        for index in range(int(self.value(loop.start)), int(self.value(loop.stop)), int(self.value(loop.step))):
          self.environment[loop.target] = index
          self.block(loop.body)
      elif isinstance(statement, I.Grid):
        import itertools

        for coordinate in itertools.product(*(range(int(self.value(n))) for n in statement.extents)):
          self.environment.update(zip(statement.targets, coordinate))
          self.block(statement.body)
      elif isinstance(statement, (I.ReductionGroup, I.ElementwiseGroup)):
        self.block(I.Block([*getattr(statement, "assignments", ()), *statement.stores]))
      else:
        raise UnsupportedInterpretation(f"unsupported interpreter statement: {type(statement).__name__}")


def interpret(owner, args=(), kwargs=None):
  from .._compiler.frontend.parser import lower
  from .._runtime.kernel import CompiledKernel

  if not isinstance(owner, CompiledKernel):
    raise TypeError("interpret requires a compiled kernel for output binding; compilation itself remains lazy")
  if owner.template.ir.source is not None:
    raise UnsupportedInterpretation("source-backed C++ kernels cannot be interpreted as DSL")
  ordered = owner._materialize_outputs(owner._arguments(args, kwargs or {}))
  values = {}
  for name, value in zip(owner.template.ir.signature.parameters, ordered):
    if hasattr(value, "detach"):
      if normalize_dtype(value.dtype) == bfloat16:
        raise UnsupportedInterpretation("bfloat16 is unavailable in the NumPy reference interpreter")
      value = value.detach().numpy()
    values[name] = value
  kernel = lower(owner.template.ir)
  Interpreter(kernel, values).block(kernel.body)
  outputs = tuple(ordered[i] for i in owner.out_idx)
  return outputs[0] if len(outputs) == 1 else outputs or None
