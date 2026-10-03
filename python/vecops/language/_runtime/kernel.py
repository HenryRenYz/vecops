"""Compiled kernel calls and lazy Torch bridge adaptation."""

from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Any
from ..._dtype import DType as SchemaDType
from ..._operator import Operator
from ..._schema import KernelDef
from ..template import TemplateFunction
from ..types import TensorType
from .binding import _AUTO, _metadata, _bind_runtime_symbol, _dtype_for_framework
from .allocation import _allocate


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
      from ..._schema_bridge import register_torch_operator

      self.torch_wrapper = register_torch_operator(
        "vecops_language",
        f"k_{self.cache_digest[:24]}",
        self.operator.native,
        self.definition._native(),
        compiler=self.operator.compiler,
        spec_order=self.specialization_order,
      )
    return self.torch_wrapper


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
        if self._core.definition.values[name] is SchemaDType
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
        "the Torch bridge must be compiled before graph capture; invoke or precompile this kernel once first"
      )
    specializations = tuple(
      _dtype_for_framework(self._core.specialization_values[name], "torch")
      if self._core.definition.values[name] is SchemaDType
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
        name: _dtype_for_framework(value, "torch") if self._core.definition.values[name] is SchemaDType else value
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
