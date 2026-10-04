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
    if op == "state_ref":
      return self.states[expression.data.name].copy()
    if op == "native_lanes":
      return self.vector_lanes
    args = tuple(self.value(a) for a in expression.args)
    contract = current().get(expression)
    if contract.traits & Trait.SCALAR_REDUCTION:
      operation = contract.reduction
      function = {"add": np.sum, "min": np.min, "max": np.max}.get(operation)
      if function is None:
        raise UnsupportedInterpretation(op)
      return function(args[0])
    if op == "axis_reduce":
      axis, operation, _ = expression.data
      return {"add": np.sum, "min": np.min, "max": np.max}[operation](args[0], axis=axis)
    if op in ("cast", "convert"):
      return np.asarray(args[0], dtype=self.dtype(expression.type.dtype))
    if op in ("where", "select"):
      return np.where(*args)
    if op == "element_at":
      return args[0][tuple(int(a) for a in args[1:])]
    if op == "reshape":
      return np.reshape(args[0], tuple(int(self.value(n)) for n in expression.type.shape))
    if op == "broadcast":
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
      elif isinstance(statement, I.Allocate):
        self.states[statement.view.storage] = np.empty(
          tuple(int(self.value(n)) for n in statement.view.shape), dtype=self.dtype(statement.view.dtype)
        )
      elif isinstance(statement, I.If):
        self.block(statement.body if self.value(statement.condition) else statement.otherwise)
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
