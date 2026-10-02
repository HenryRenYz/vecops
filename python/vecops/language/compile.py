"""Lower and cache captured templates, then adapt their runtime calls."""

from __future__ import annotations

import hashlib
import inspect
import threading
import weakref
from dataclasses import dataclass
from typing import Any

from .._compiler import Compiler
from .._dtype import Const as LegacyConst
from .._dtype import DType as LegacyDType
from .._dtype import normalize_dtype
from .._operator import Operator
from .._schema import KernelDef, TensorAccess, TensorDef, TensorMeta, ValueDef
from .expr import Const, DType, Dynamic, IntExpr, SymbolicInt
from .template import TemplateFunction, TemplateIR
from .types import TensorType

_AUTO = object()
_compile_cache: weakref.WeakValueDictionary[tuple[Any, ...], _CompiledCore] = weakref.WeakValueDictionary()
_compile_lock = threading.RLock()


def _normalize_out_idx(
  out_idx: int | list[int] | tuple[int, ...] | None,
  parameters: tuple[inspect.Parameter, ...],
) -> tuple[int, ...]:
  if out_idx is None:
    return ()
  positional = tuple(
    index
    for index, parameter in enumerate(parameters)
    if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
  )
  values = (out_idx,) if isinstance(out_idx, int) and not isinstance(out_idx, bool) else tuple(out_idx)
  result = []
  for value in values:
    if isinstance(value, bool) or not isinstance(value, int):
      raise TypeError("out_idx entries must be integers")
    normalized = value + len(positional) if value < 0 else value
    if normalized < 0 or normalized >= len(positional):
      raise IndexError(f"out_idx {value} is outside a {len(positional)}-positional-parameter template")
    parameter_index = positional[normalized]
    if parameter_index in result:
      raise ValueError(f"out_idx contains duplicate parameter {value}")
    result.append(parameter_index)
  return tuple(result)


def _source_identity(ir: TemplateIR) -> str:
  digest = hashlib.sha256()
  digest.update(b"vecops-language-cpp-v1\0")
  if ir.source is not None:
    digest.update(ir.source.read_bytes())
  for dependency in ir.dependencies:
    digest.update(str(dependency).encode())
    digest.update(b"\0")
    digest.update(dependency.read_bytes())
  return digest.hexdigest()


def _compiler_key(compiler: Compiler):
  return (
    compiler.target,
    str(compiler.cache_dir),
    str(compiler.build_dir),
    compiler.cache_mode,
    compiler.cache_namespace,
    tuple(map(str, compiler.include_dirs)),
    compiler.cflags,
    tuple(map(str, compiler.library_dirs)),
    compiler.libraries,
    compiler.ldflags,
  )


def _specialization_environment(ir: TemplateIR):
  values: dict[str, Any] = {}
  kinds: dict[str, type] = {}
  order: list[str] = []
  for capture in ir.captures:
    name = getattr(capture, "name", None)
    if not name:
      continue
    if isinstance(capture, Const):
      kind = LegacyConst
      value = int(capture.value)
    elif isinstance(capture, DType):
      kind = LegacyDType
      value = capture.value
    else:
      continue
    if name in kinds and (kinds[name] is not kind or values[name] != value):
      raise ValueError(f"captured specialization name {name!r} has conflicting values")
    kinds[name] = kind
    values[name] = value
    if name not in order:
      order.append(name)
  return kinds, values, tuple(order)


def _lower_tensor_dtype(dtype: Any):
  if isinstance(dtype, DType):
    if not dtype.name:
      raise ValueError("a dtype captured by a template annotation must have a name")
    return dtype.name
  return dtype


def _build_definition(ir: TemplateIR, out_idx: tuple[int, ...]):
  parameters = []
  signature_parameters = tuple(ir.signature.parameters.values())
  for index, parameter in enumerate(signature_parameters):
    if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
      raise TypeError("T.template functions do not support *args or **kwargs")
    annotation = ir.annotations.get(parameter.name, parameter.annotation)
    if annotation is inspect.Parameter.empty:
      raise TypeError(f"template parameter {parameter.name!r} requires an annotation")
    if isinstance(annotation, TensorType):
      access_name = annotation.access
      if index in out_idx and access_name == "input":
        access_name = "output"
      access = {
        "input": TensorAccess.input,
        "output": TensorAccess.output,
        "inout": TensorAccess.inout,
      }[access_name]
      optional = annotation.optional or parameter.default is None
      if optional and access != TensorAccess.input:
        raise TypeError(f"output tensor {parameter.name!r} cannot be optional")
      parameters.append(
        TensorDef(
          parameter.name,
          annotation.shape,
          annotation.concrete_strides,
          dtype=_lower_tensor_dtype(annotation.dtype),
          optional=optional,
          access=access,
        )
      )
    else:
      dtype = annotation.value if isinstance(annotation, DType) else annotation
      if parameter.default is inspect.Parameter.empty:
        parameters.append(ValueDef(parameter.name, dtype))
      else:
        parameters.append(ValueDef(parameter.name, dtype, parameter.default))

  for index in out_idx:
    if not isinstance(parameters[index], TensorDef):
      raise TypeError(f"out_idx parameter {signature_parameters[index].name!r} is not a tensor")
    if parameters[index].optional:
      raise TypeError(f"out_idx parameter {signature_parameters[index].name!r} cannot be optional")

  kinds, values, order = _specialization_environment(ir)
  definition = KernelDef(ir.name, parameters, values=kinds)
  return definition, values, order


@dataclass
class _CompiledCore:
  operator: Operator
  definition: KernelDef
  out_idx: tuple[int, ...]
  specialization_values: dict[str, Any]
  specialization_order: tuple[str, ...]
  cache_digest: str
  torch_wrapper: Any = None

  def torch(self):
    if self.torch_wrapper is None:
      from .._schema_bridge import register_torch_operator

      self.torch_wrapper = register_torch_operator(
        "vecops_language",
        f"k_{self.cache_digest[:24]}",
        self.operator.native,
        self.definition._native(),
        compiler=self.operator.compiler,
        spec_order=self.specialization_order,
      )
    return self.torch_wrapper


def _metadata(value: Any):
  if isinstance(value, TensorMeta):
    return value.shape, value.strides or (), normalize_dtype(value.dtype), "meta"
  torch = __import__("sys").modules.get("torch")
  if torch is not None and isinstance(value, torch.Tensor):
    return tuple(map(int, value.shape)), tuple(map(int, value.stride())), normalize_dtype(value.dtype), "torch"
  numpy = __import__("sys").modules.get("numpy")
  if numpy is not None and isinstance(value, numpy.ndarray):
    itemsize = int(value.dtype.itemsize)
    return (
      tuple(map(int, value.shape)),
      tuple(int(item) // itemsize for item in value.strides),
      normalize_dtype(value.dtype),
      "numpy",
    )
  return None


def _bind_runtime_symbol(value: SymbolicInt, observed: int, environment: dict[str, int]) -> None:
  if isinstance(value, Dynamic) and value.name:
    previous = environment.setdefault(value.name, observed)
    if previous != observed:
      raise ValueError(f"runtime dimension {value.name!r} is {previous}, observed {observed}")


def _evaluate(value: SymbolicInt, environment: dict[str, int]) -> int:
  if isinstance(value, Const):
    return int(value.value)
  if isinstance(value, Dynamic):
    if value.name and value.name in environment:
      return environment[value.name]
    if value.observed is not None:
      return value.observed
    raise ValueError(f"cannot resolve dynamic output dimension {value.name or '<anonymous>'!r}")
  if isinstance(value, IntExpr):
    left = _evaluate(value.left, environment)
    right = _evaluate(value.right, environment)
    return {
      "add": lambda: left + right,
      "sub": lambda: left - right,
      "mul": lambda: left * right,
      "floordiv": lambda: left // right,
      "mod": lambda: left % right,
      "min": lambda: min(left, right),
      "max": lambda: max(left, right),
    }[value.op]()
  raise TypeError(f"unsupported output dimension {value!r}")


def _dtype_for_framework(dtype: Any, framework: str):
  normalized = normalize_dtype(dtype)
  name = str(normalized).rsplit(".", 1)[-1]
  if framework == "torch":
    import torch

    return getattr(torch, "bool" if name == "bool_" else name)
  if framework == "numpy":
    import numpy as np

    if name == "bfloat16":
      raise TypeError("NumPy output allocation does not support bfloat16")
    return np.dtype("bool" if name == "bool_" else name)
  return normalized


def _storage_plan(shape: tuple[int, ...], strides: tuple[int, ...]) -> tuple[int, int]:
  if len(shape) != len(strides):
    raise ValueError("output shape and stride ranks differ")
  if any(size < 0 for size in shape):
    raise ValueError("output shape contains a negative dimension")
  if any(size == 0 for size in shape):
    return 0, 0
  minimum = 0
  maximum = 0
  for size, stride in zip(shape, strides):
    span = (size - 1) * stride
    minimum += min(0, span)
    maximum += max(0, span)
  elements = maximum - minimum + 1
  if elements > 2**63 - 1:
    raise OverflowError("output storage span exceeds int64")
  return elements, -minimum


def _allocate(
  annotation: TensorType,
  environment: dict[str, int],
  reference: Any,
  inferred_dtype: Any,
):
  shape = tuple(_evaluate(item, environment) for item in annotation.shape)
  strides = tuple(_evaluate(item, environment) for item in annotation.concrete_strides)
  metadata = _metadata(reference) if reference is not None else None
  framework = metadata[3] if metadata is not None else "meta"
  dtype = annotation.dtype.value if isinstance(annotation.dtype, DType) else annotation.dtype
  if dtype is None:
    if inferred_dtype is None:
      raise ValueError("cannot infer dtype for an automatically allocated output")
    dtype = inferred_dtype
  elements, offset = _storage_plan(shape, strides)
  if framework == "torch":
    if any(stride < 0 for stride in strides):
      raise ValueError("Torch cannot automatically allocate a tensor with negative strides")
    import torch

    base = torch.empty((elements,), dtype=_dtype_for_framework(dtype, "torch"), device=reference.device)
    return torch.as_strided(base, shape, strides, offset)
  if framework == "numpy":
    import numpy as np

    np_dtype = _dtype_for_framework(dtype, "numpy")
    base = np.empty((elements,), dtype=np_dtype)
    return np.ndarray(
      shape,
      dtype=np_dtype,
      buffer=base,
      offset=offset * np_dtype.itemsize,
      strides=tuple(item * np_dtype.itemsize for item in strides),
    )
  return TensorMeta(shape, dtype=dtype, strides=strides, access=TensorAccess.output)


class CompiledKernel:
  """A bound template instance backed by a cached source-kernel operator."""

  def __init__(self, core: _CompiledCore, template: TemplateFunction) -> None:
    self._core = core
    self.template = template
    self.kernel_def = core.definition
    self.source = template.ir.source
    self.out_idx = core.out_idx
    self.__signature__ = template.ir.signature
    self._explicit_torch_function = None

  @property
  def operator(self) -> Operator:
    return self._core.operator

  def explicit_torch_function(self):
    """Return a graph-capture-friendly callable for explicit output calls.

    The returned function accepts every template parameter positionally,
    including output tensors.  It bypasses Python argument binding and output
    allocation, which makes the call suitable for hot paths and TorchDynamo
    graphs after template elaboration has selected this specialization.
    """
    if self._explicit_torch_function is None:
      target = self._core.torch()
      specializations = tuple(
        _dtype_for_framework(self._core.specialization_values[name], "torch")
        if self._core.definition.values[name] is LegacyDType
        else self._core.specialization_values[name]
        for name in self._core.specialization_order
      )

      def invoke(*args):
        return target(*args, *specializations)

      self._explicit_torch_function = invoke
    return self._explicit_torch_function

  def explicit_torch_binding(self):
    """Return the raw Torch op and positional specs after bridge preparation."""
    wrapper = self._core.torch()
    if not wrapper._vecops_ready[0]:
      raise RuntimeError(
        "the Torch bridge must be compiled before graph capture; invoke or "
        "precompile this kernel once first"
      )
    specializations = tuple(
      _dtype_for_framework(self._core.specialization_values[name], "torch")
      if self._core.definition.values[name] is LegacyDType
      else self._core.specialization_values[name]
      for name in self._core.specialization_order
    )
    return wrapper._vecops_torch_op_holder[0], specializations

  def _arguments(self, args, kwargs):
    parameters = tuple(self.template.ir.signature.parameters.values())

    def bind(compact_outputs: bool):
      remaining = dict(kwargs)
      ordered = [_AUTO] * len(parameters)
      position = 0
      for index, parameter in enumerate(parameters):
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
          continue
        if compact_outputs and index in self.out_idx:
          continue
        if position < len(args):
          if parameter.name in remaining:
            raise TypeError(f"multiple values for kernel argument {parameter.name!r}")
          ordered[index] = args[position]
          position += 1
      if position != len(args):
        raise TypeError(f"kernel expected at most {position} positional arguments")
      for index, parameter in enumerate(parameters):
        if parameter.name in remaining:
          if ordered[index] is not _AUTO:
            raise TypeError(f"multiple values for kernel argument {parameter.name!r}")
          ordered[index] = remaining.pop(parameter.name)
        elif ordered[index] is _AUTO and index not in self.out_idx:
          if parameter.default is not inspect.Parameter.empty:
            ordered[index] = parameter.default
          else:
            raise TypeError(f"missing required kernel argument {parameter.name!r}")
      if remaining:
        raise TypeError(f"unexpected kernel argument {next(iter(remaining))!r}")
      return ordered

    try:
      return bind(False)
    except TypeError as explicit_error:
      if not self.out_idx:
        raise
      try:
        return bind(True)
      except TypeError:
        raise explicit_error

  def _materialize_outputs(self, ordered):
    annotations = self.template.ir.annotations
    parameters = tuple(self.template.ir.signature.parameters.values())
    environment: dict[str, int] = {}
    reference = None
    inferred_dtype = None
    for parameter, value in zip(parameters, ordered):
      if value is _AUTO or value is None:
        continue
      annotation = annotations.get(parameter.name, parameter.annotation)
      if not isinstance(annotation, TensorType):
        continue
      metadata = _metadata(value)
      if metadata is None:
        continue
      shape, strides, dtype, _ = metadata
      if reference is None:
        reference = value
        inferred_dtype = dtype
      if len(shape) != len(annotation.shape) or len(strides) != len(annotation.concrete_strides):
        raise ValueError(f"tensor {parameter.name!r} has the wrong rank")
      for declared, observed in zip(annotation.shape, shape):
        _bind_runtime_symbol(declared, observed, environment)
      for declared, observed in zip(annotation.concrete_strides, strides):
        _bind_runtime_symbol(declared, observed, environment)
    for index in self.out_idx:
      if ordered[index] is _AUTO or ordered[index] is None:
        parameter = parameters[index]
        annotation = annotations[parameter.name]
        ordered[index] = _allocate(annotation, environment, reference, inferred_dtype)
    return ordered

  def __call__(self, *args, **kwargs):
    ordered = self._materialize_outputs(self._arguments(args, kwargs))
    torch = __import__("sys").modules.get("torch")
    use_torch = torch is not None and any(isinstance(value, torch.Tensor) for value in ordered)
    target = self._core.torch() if use_torch else self._core.operator
    specializations = self._core.specialization_values
    if use_torch:
      specializations = {
        name: _dtype_for_framework(value, "torch") if self._core.definition.values[name] is LegacyDType else value
        for name, value in specializations.items()
      }
    target(*ordered, **specializations)
    outputs = tuple(ordered[index] for index in self.out_idx)
    if not outputs:
      return None
    return outputs[0] if len(outputs) == 1 else outputs

  def compile_for(self, *args, **kwargs) -> None:
    ordered = self._materialize_outputs(self._arguments(args, kwargs))
    self._core.operator.compile_for(*ordered, **self._core.specialization_values)


def compile(
  template_function: TemplateFunction,
  *,
  out_idx: int | list[int] | tuple[int, ...] | None = None,
  compiler: Compiler | None = None,
) -> CompiledKernel:
  """Lower a TemplateFunction and reuse its immutable compiled-kernel core."""
  if not isinstance(template_function, TemplateFunction):
    raise TypeError("T.compile expects a T.template-decorated function")
  ir = template_function.ir
  if ir.source is None:
    raise NotImplementedError("Python DSL templates are not implemented; use T.template(source=...)")
  selected = compiler or template_function.compiler or Compiler()
  indices = _normalize_out_idx(out_idx, tuple(ir.signature.parameters.values()))
  source_identity = _source_identity(ir)
  key = (ir.canonical(), indices, source_identity, _compiler_key(selected))
  digest = hashlib.sha256(repr(key).encode()).hexdigest()
  with _compile_lock:
    core = _compile_cache.get(key)
    if core is None:
      definition, values, order = _build_definition(ir, indices)
      recipe = ir.recipe_id or f"{ir.name}:{source_identity[:16]}"
      recipe = f"language-v1:{recipe}:{source_identity[:16]}"
      operation = Operator(ir.source, definition, selected, recipe_id=recipe)
      core = _CompiledCore(operation, definition, indices, values, order, digest)
      _compile_cache[key] = core
  return CompiledKernel(core, template_function)


__all__ = ["CompiledKernel", "compile"]
