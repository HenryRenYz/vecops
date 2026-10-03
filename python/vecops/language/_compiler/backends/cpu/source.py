"""CPU source-kernel schema and constexpr specialization adaptation."""

from __future__ import annotations
import inspect
from typing import Any
from ....._dtype import Const as SchemaConst, DType as SchemaDType, normalize_dtype
from ....._schema import KernelDef, TensorAccess, TensorDef, ValueDef
from ....expr import Const, DType
from ....template import TemplateIR
from ....types import TensorType


def _specialization_environment(ir: TemplateIR):
  values: dict[str, Any] = {}
  kinds: dict[str, type] = {}
  order: list[str] = []
  for capture in ir.captures:
    name = getattr(capture, "name", None)
    if not name:
      continue
    if isinstance(capture, Const):
      kind = SchemaConst
      value = int(capture.value)
    elif isinstance(capture, DType):
      kind = SchemaDType
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
  if isinstance(dtype, str):
    from ....._dtype import _NAMES

    if dtype in _NAMES:
      return _NAMES[dtype]
    try:
      return normalize_dtype(dtype)
    except (TypeError, ValueError):
      pass  # Non-dtype strings are explicit low-level schema symbols.
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
