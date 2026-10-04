"""Runtime tensor metadata and symbolic argument/output binding."""

from __future__ import annotations

import inspect
from typing import Any

from ..._dtype import normalize_dtype
from ..._schema import TensorMeta
from ..expr import Const, Dynamic, IntExpr, SymbolicInt

_AUTO = object()


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
