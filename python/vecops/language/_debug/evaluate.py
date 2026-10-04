"""Metadata evaluation shared by resource reports and the reference interpreter."""

import math
import re

from ..._dtype import dtype_itemsize as itemsize
from ..._dtype import dtype_name
from .._compiler import ir as I


def scalar(expression, environment):
  if not isinstance(expression, I.Expr):
    return expression
  if expression.op == "literal":
    return expression.data
  if expression.op == "lanes":
    dtype, power = expression.data
    target = environment.get("__debug_cpu_target")
    if target is None or target.vector_bits is None:
      raise ValueError("scalable vector length is not known for this target")
    native = 1 if target.isa == "scalar" else target.vector_bits // (itemsize(dtype) * 8)
    if power < 0:
      divisor = 1 << -power
      if native < divisor or native % divisor:
        raise ValueError("nonintegral virtual subword extent")
      return native // divisor
    return native * (1 << power)
  if expression.op in ("symbol", "local", "frozen", "raw", "schedule_parameter"):
    name = expression.data
    if name in environment:
      return environment[name]
    match = re.fullmatch(r"(\w+)\.(size|stride)\((\d+)\)", name)
    if match:
      descriptor = environment[match[1]]
      return descriptor["shape" if match[2] == "size" else "strides"][int(match[3])]
    raise ValueError(f"runtime-dependent expression: {name}")
  args = tuple(scalar(a, environment) for a in expression.args)
  from .._compiler.operations import current

  fold = current().get(expression).fold
  if fold is not None:
    return fold(*args)
  if expression.op == "ceildiv":
    return -(-args[0] // args[1])
  if expression.op == "select":
    return args[1] if args[0] else args[2]
  if expression.op == "broadcast_dim":
    return args[1] if args[0] == 1 else args[0]
  if expression.op == "cast":
    return float(args[0]) if "float" in expression.type.dtype else int(args[0])
  if expression.op == "numel":
    return math.prod(args)
  raise ValueError(f"cannot evaluate metadata operation {expression.op}")


def bind(owner, args, kwargs):
  """Bind without allocating omitted output tensors or reading tensor contents."""
  from .._runtime.binding import _AUTO, _bind_runtime_symbol, _evaluate, _metadata
  from ..types import TensorType

  template = getattr(owner, "template", owner)
  parameters = tuple(template.ir.signature.parameters.values())
  if hasattr(owner, "_arguments"):
    ordered = owner._arguments(args, kwargs)
  else:
    bound = template.ir.signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    ordered = [bound.arguments.get(p.name, _AUTO) for p in parameters]
  environment, tensors = {}, {}
  for parameter, value in zip(parameters, ordered):
    annotation = template.ir.annotations[parameter.name]
    if not isinstance(annotation, TensorType):
      if value is not _AUTO:
        environment[parameter.name] = value
      continue
    if value is _AUTO or value is None:
      continue
    metadata = _metadata(value)
    if metadata is None and hasattr(value, "shape") and hasattr(value, "strides"):
      metadata = (value.shape, value.strides, value.dtype, "cpu")
    if metadata is None:
      raise TypeError(f"cannot obtain tensor metadata from {parameter.name}")
    shape, strides, dtype, _ = metadata
    if len(shape) != len(annotation.shape):
      raise ValueError(f"rank mismatch: {parameter.name}")
    for declared, actual in zip((*annotation.shape, *annotation.concrete_strides), (*shape, *strides)):
      _bind_runtime_symbol(declared, actual, environment)
    descriptor = {
      "shape": tuple(shape),
      "strides": tuple(strides),
      "dtype": dtype_name(dtype),
      "value": value,
      "annotation": annotation,
    }
    environment[parameter.name] = tensors[parameter.name] = descriptor
  for parameter, value in zip(parameters, ordered):
    annotation = template.ir.annotations[parameter.name]
    if isinstance(annotation, TensorType) and parameter.name not in tensors:
      shape = tuple(_evaluate(n, environment) for n in annotation.shape)
      strides = tuple(_evaluate(n, environment) for n in annotation.concrete_strides)
      dtype = annotation.dtype
      if dtype is None:
        if not tensors:
          raise ValueError("cannot infer dtype for an automatically allocated output")
        dtype = next(iter(tensors.values()))["dtype"]
      dtype = dtype_name(dtype)
      descriptor = {"shape": shape, "strides": strides, "dtype": dtype, "value": None, "annotation": annotation}
      environment[parameter.name] = tensors[parameter.name] = descriptor
  kernel = getattr(owner, "kernel_ir", None)
  if kernel:
    for name, expression in kernel.bindings.items():
      try:
        environment[name] = scalar(expression, environment)
      except (ValueError, KeyError):
        pass
  return environment, tensors
