# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Runtime annotations used by :func:`vecops.jit` declarations.

Tensor annotations use ``vt.In(dtype)[shape, strides]`` syntax. Each string
is whitespace-separated and accepts fixed integers, identifier symbols,
``Const``, ``Dynamic``/``Any``, or ``Dynamic<alignment,lower,upper>``. The
three Dynamic expressions are fixed integers or Const symbols; bounds are
inclusive. Supplying only ``[shape]`` creates anonymous Const strides.

Runtime-only ``vt.Tensor`` annotations use the same base grammar with two
additional relations: a leading ``*B`` binds B to the product of arbitrary
leading axes, and ``2*M`` checks an integer multiple. ``@vt.check_tensors``
binds shared symbols across all annotated arguments for Torch, NumPy, and
storage-free TensorMeta values.
"""

from __future__ import annotations

import functools
import inspect
import math
import re
import sys
from dataclasses import dataclass
from typing import Any

from .. import _C
from .._dtype import Const, DType, normalize_dtype
from .._schema import Dynamic, TensorMeta

_INTEGER = re.compile(r"^[+-]?\d+$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PRODUCT = re.compile(
  r"^(?:(?P<left>[1-9][0-9]*)\*(?P<symbol1>[A-Za-z_][A-Za-z0-9_]*)|"
  r"(?P<symbol2>[A-Za-z_][A-Za-z0-9_]*)\*(?P<right>[1-9][0-9]*))$"
)
_VARIADIC = re.compile(r"^\*(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)$")


@dataclass(frozen=True)
class ProductDimension:
  """Runtime-check-only dimension equal to an integer multiple of a symbol."""

  factor: int
  symbol: str


@dataclass(frozen=True)
class VariadicDimension:
  """Runtime-check-only leading axis group bound to its extent product."""

  symbol: str


def _split_dimensions(text: str) -> list[str]:
  tokens: list[str] = []
  begin = 0
  depth = 0
  for index, character in enumerate(text):
    if character == "<":
      depth += 1
    elif character == ">":
      depth -= 1
      if depth < 0:
        raise ValueError(f"unmatched '>' in dimension string {text!r}")
    elif character.isspace() and depth == 0:
      if begin < index:
        tokens.append(text[begin:index])
      begin = index + 1
  if depth:
    raise ValueError(f"unclosed '<' in dimension string {text!r}")
  if begin < len(text):
    tokens.append(text[begin:])
  return tokens


def _expression(text: str) -> int | str:
  if text == "-inf":
    return -(2**63)
  if text == "+inf":
    return 2**63 - 1
  if _INTEGER.fullmatch(text):
    return int(text)
  if _IDENTIFIER.fullmatch(text):
    return text
  raise ValueError(f"invalid Const expression {text!r}")


def _dimension(token: str, *, runtime: bool = False):
  if token == "Const":
    return Const
  if token in {"Dynamic", "Any"}:
    return Dynamic()
  if token.startswith("Dynamic<") and token.endswith(">"):
    arguments = token[len("Dynamic<") : -1].split(",")
    if not 1 <= len(arguments) <= 3 or any(not argument for argument in arguments):
      raise ValueError(f"invalid Dynamic declaration {token!r}")
    values = [_expression(argument) for argument in arguments]
    return Dynamic(*values)
  if _INTEGER.fullmatch(token):
    return int(token)
  if runtime:
    if match := _PRODUCT.fullmatch(token):
      factor = int(match.group("left") or match.group("right"))
      symbol = match.group("symbol1") or match.group("symbol2")
      return ProductDimension(factor, symbol)
    if match := _VARIADIC.fullmatch(token):
      return VariadicDimension(match.group("symbol"))
  if _IDENTIFIER.fullmatch(token):
    return token
  raise ValueError(f"invalid dimension token {token!r}")


def parse_dimensions(text: str, *, runtime: bool = False) -> tuple[Any, ...]:
  """Parse one dimension-string DSL value into KernelDef-compatible objects."""
  if not isinstance(text, str):
    raise TypeError("tensor shape and strides must be strings")
  return tuple(
    _dimension(token, runtime=runtime)
    for token in _split_dimensions(text.strip())
  )


@dataclass(frozen=True)
class TensorAnnotation:
  """Frozen intermediate annotation consumed only by :func:`vecops.jit`."""

  dtype: Any
  access: _C.TensorAccess
  shape: tuple[Any, ...]
  strides: tuple[Any, ...]
  optional: bool = False


class _TensorFactory:
  """Subscription helper returned by :func:`In`, :func:`Out`, and :func:`InOut`."""

  def __init__(self, access: _C.TensorAccess, dtype: Any, optional: bool) -> None:
    self.access = access
    self.dtype = dtype
    self.optional = optional

  def __getitem__(self, dimensions: str | tuple[str, str]) -> TensorAnnotation:
    """Create an annotation from ``"shape"`` or ``("shape", "strides")`` strings."""
    if isinstance(dimensions, str):
      shape = parse_dimensions(dimensions)
      strides = tuple(Const for _ in shape)
    elif isinstance(dimensions, tuple) and len(dimensions) == 2:
      shape = parse_dimensions(dimensions[0])
      strides = parse_dimensions(dimensions[1])
    else:
      raise TypeError("tensor annotation expects ['shape'] or ['shape', 'strides']")
    if len(shape) != len(strides):
      raise ValueError("tensor annotation shape and stride ranks differ")
    return TensorAnnotation(self.dtype, self.access, shape, strides, self.optional)


def In(dtype: Any = None, *, optional: bool = False) -> _TensorFactory:
  """Declare a read-only tensor parameter; only inputs may be optional."""
  return _TensorFactory(_C.TensorAccess.input, dtype, optional)


def Out(dtype: Any = None) -> _TensorFactory:
  """Declare a caller-allocated write-only tensor parameter."""
  return _TensorFactory(_C.TensorAccess.output, dtype, False)


def InOut(dtype: Any = None) -> _TensorFactory:
  """Declare a caller-allocated readable and writable tensor parameter."""
  return _TensorFactory(_C.TensorAccess.inout, dtype, False)


@dataclass(frozen=True)
class _RuntimeTensorMetadata:
  shape: tuple[int, ...]
  strides: tuple[int, ...]
  dtype: Any


def _runtime_tensor_metadata(value: Any) -> _RuntimeTensorMetadata:
  """Adapt Torch, NumPy, or vecops metadata without importing a framework."""
  if isinstance(value, TensorMeta):
    return _RuntimeTensorMetadata(value.shape, value.strides or (), value.dtype)

  torch = sys.modules.get("torch")
  if torch is not None and isinstance(value, torch.Tensor):
    return _RuntimeTensorMetadata(
      tuple(int(axis) for axis in value.shape),
      tuple(int(stride) for stride in value.stride()),
      value.dtype,
    )

  numpy = sys.modules.get("numpy")
  if numpy is not None and isinstance(value, numpy.ndarray):
    item_size = int(value.dtype.itemsize)
    if item_size <= 0 or any(stride % item_size for stride in value.strides):
      raise TypeError("NumPy tensor has strides that are not whole elements")
    return _RuntimeTensorMetadata(
      tuple(int(axis) for axis in value.shape),
      tuple(int(stride // item_size) for stride in value.strides),
      value.dtype,
    )

  if isinstance(value, _C.TensorMeta):
    return _RuntimeTensorMetadata(
      tuple(value.sizes), tuple(value.strides), value.dtype
    )
  raise TypeError(
    "expected a torch.Tensor, numpy.ndarray, vecops.TensorMeta, or vecops._C.TensorMeta"
  )


@dataclass(frozen=True)
class TensorPattern:
  """Framework-neutral runtime tensor metadata contract."""

  dtype: Any
  shape: tuple[Any, ...]
  strides: tuple[Any, ...] | None = None

  def check(
    self,
    value: Any,
    *,
    symbols: dict[str, Dynamic] | None = None,
    name: str = "tensor",
  ) -> Any:
    """Validate one tensor and return it unchanged."""
    _check_tensor_records([(name, self, value)], symbols or {})
    return value


class _RuntimeTensorFactory:
  def __init__(self, dtype: Any = None) -> None:
    self.dtype = dtype

  def __getitem__(self, dimensions: str | tuple[str, str]) -> TensorPattern:
    if isinstance(dimensions, str):
      shape = parse_dimensions(dimensions, runtime=True)
      strides = None
    elif isinstance(dimensions, tuple) and len(dimensions) == 2:
      shape = parse_dimensions(dimensions[0], runtime=True)
      strides = parse_dimensions(dimensions[1], runtime=True)
    else:
      raise TypeError("runtime tensor annotation expects ['shape'] or ['shape', 'strides']")
    variadic = [index for index, item in enumerate(shape) if isinstance(item, VariadicDimension)]
    if variadic and variadic != [0]:
      raise ValueError("a variadic runtime dimension must be the first and only variadic shape token")
    if strides is not None and variadic:
      raise ValueError("variadic runtime shapes cannot declare positional strides")
    if strides is not None and len(shape) != len(strides):
      raise ValueError("runtime tensor annotation shape and stride ranks differ")
    return TensorPattern(self.dtype, shape, strides)


class Tensor:
  """Factory for runtime tensor contracts.

  ``Tensor[shape]`` leaves dtype unconstrained, while
  ``Tensor(dtype)[shape]`` checks a concrete or named dtype.
  """

  def __new__(cls, dtype: Any = None) -> _RuntimeTensorFactory:
    return _RuntimeTensorFactory(dtype)

  def __class_getitem__(cls, dimensions: str | tuple[str, str]) -> TensorPattern:
    return _RuntimeTensorFactory()[dimensions]


def _bind_integer(values: dict[str, int], symbol: str, observed: int, location: str) -> None:
  previous = values.setdefault(symbol, observed)
  if previous != observed:
    raise ValueError(
      f"dimension {symbol!r} is {previous}, but {location} observed {observed}"
    )


def _resolve_runtime_expr(value: int | str, values: dict[str, int], location: str) -> int:
  if isinstance(value, int):
    return value
  try:
    return values[value]
  except KeyError as error:
    raise ValueError(f"{location} requires unresolved dimension {value!r}") from error


def _check_dynamic(
  declaration: Dynamic,
  observed: int,
  values: dict[str, int],
  location: str,
) -> None:
  alignment = _resolve_runtime_expr(declaration.alignment, values, f"{location} alignment")
  lower = _resolve_runtime_expr(declaration.lower, values, f"{location} lower bound")
  upper = _resolve_runtime_expr(declaration.upper, values, f"{location} upper bound")
  if alignment <= 0 or alignment & (alignment - 1):
    raise ValueError(f"{location} alignment must be a positive power of two")
  if lower > upper:
    raise ValueError(f"{location} lower bound exceeds its upper bound")
  if observed < lower or observed > upper or observed % alignment:
    raise ValueError(
      f"{location}={observed} violates Dynamic<{alignment},{lower},{upper}>"
    )


def _observed_dimensions(
  pattern: tuple[Any, ...], observed: tuple[int, ...], location: str
) -> list[tuple[Any, int, str]]:
  if pattern and isinstance(pattern[0], VariadicDimension):
    tail_rank = len(pattern) - 1
    if len(observed) < tail_rank:
      raise ValueError(
        f"{location} has rank {len(observed)}, expected at least {tail_rank}"
      )
    prefix_rank = len(observed) - tail_rank
    prefix = observed[:prefix_rank]
    result = [(pattern[0], math.prod(prefix), f"{location} leading dimensions")]
    result.extend(
      (item, value, f"{location}[{prefix_rank + index}]")
      for index, (item, value) in enumerate(zip(pattern[1:], observed[prefix_rank:]))
    )
    return result
  if len(pattern) != len(observed):
    raise ValueError(
      f"{location} has rank {len(observed)}, expected {len(pattern)}"
    )
  return [
    (item, value, f"{location}[{index}]")
    for index, (item, value) in enumerate(zip(pattern, observed))
  ]


def _check_tensor_records(
  records: list[tuple[str, TensorPattern, Any]],
  declarations: dict[str, Dynamic],
) -> None:
  for symbol, declaration in declarations.items():
    if not isinstance(symbol, str) or not symbol:
      raise TypeError("runtime dimension symbol names must be non-empty strings")
    if not isinstance(declaration, Dynamic):
      raise TypeError(f"runtime dimension symbol {symbol!r} must use vt.Dynamic")

  metadata = [
    (name, pattern, _runtime_tensor_metadata(value))
    for name, pattern, value in records
  ]
  integer_values: dict[str, int] = {}
  dtype_values: dict[str, Any] = {}
  dimensions: list[tuple[Any, int, str]] = []

  for name, pattern, tensor in metadata:
    if any(axis < 0 for axis in tensor.shape):
      raise ValueError(f"{name} has a negative shape dimension")
    if isinstance(pattern.dtype, str):
      observed_dtype = normalize_dtype(tensor.dtype)
      previous = dtype_values.setdefault(pattern.dtype, observed_dtype)
      if previous != observed_dtype:
        raise TypeError(
          f"dtype {pattern.dtype!r} is {previous}, but {name} has {observed_dtype}"
        )
    elif pattern.dtype is not None:
      expected_dtype = normalize_dtype(pattern.dtype)
      observed_dtype = normalize_dtype(tensor.dtype)
      if observed_dtype != expected_dtype:
        raise TypeError(
          f"{name} has dtype {observed_dtype}, expected {expected_dtype}"
        )
    dimensions.extend(_observed_dimensions(pattern.shape, tensor.shape, f"{name}.shape"))
    if pattern.strides is not None:
      dimensions.extend(
        _observed_dimensions(pattern.strides, tensor.strides, f"{name}.strides")
      )

  # First infer every direct symbol so Dynamic bounds and products are order independent.
  for declaration, observed, location in dimensions:
    if isinstance(declaration, str):
      _bind_integer(integer_values, declaration, observed, location)
    elif isinstance(declaration, VariadicDimension):
      _bind_integer(integer_values, declaration.symbol, observed, location)

  unused = declarations.keys() - integer_values.keys()
  if unused:
    raise ValueError(f"unused runtime dimension symbol {next(iter(sorted(unused)))!r}")

  for declaration, observed, location in dimensions:
    if isinstance(declaration, int):
      if observed != declaration:
        raise ValueError(f"{location} is {observed}, expected {declaration}")
    elif declaration is Const:
      continue
    elif isinstance(declaration, Dynamic):
      _check_dynamic(declaration, observed, integer_values, location)
    elif isinstance(declaration, ProductDimension):
      expected = declaration.factor * _resolve_runtime_expr(
        declaration.symbol, integer_values, location
      )
      if observed != expected:
        raise ValueError(f"{location} is {observed}, expected {expected}")
    elif isinstance(declaration, str) and declaration in declarations:
      _check_dynamic(declarations[declaration], observed, integer_values, location)
    elif isinstance(declaration, VariadicDimension) and declaration.symbol in declarations:
      _check_dynamic(
        declarations[declaration.symbol], observed, integer_values, location
      )


def _torch_is_compiling() -> bool:
  torch = sys.modules.get("torch")
  return bool(
    torch is not None
    and hasattr(torch, "compiler")
    and torch.compiler.is_compiling()
  )


def check_tensors(*, symbols: dict[str, Dynamic] | None = None):
  """Validate all :class:`TensorPattern` annotations before an eager call.

  Validation is skipped while Torch Dynamo is tracing; the registered native
  operator still validates its canonical KernelDef metadata at execution.
  """

  declarations = dict(symbols or {})

  def decorate(function):
    signature = inspect.signature(function)
    annotations = inspect.get_annotations(function, eval_str=True)
    patterns = {
      name: annotation
      for name, annotation in annotations.items()
      if name in signature.parameters and isinstance(annotation, TensorPattern)
    }
    if not patterns:
      raise TypeError("@vt.check_tensors requires at least one vt.Tensor annotation")

    @functools.wraps(function)
    def checked(*args, **kwargs):
      if not _torch_is_compiling():
        bound = signature.bind(*args, **kwargs)
        _check_tensor_records(
          [
            (name, pattern, bound.arguments[name])
            for name, pattern in patterns.items()
            if name in bound.arguments
          ],
          declarations,
        )
      return function(*args, **kwargs)

    return checked

  return decorate


@dataclass(frozen=True)
class ScalarAnnotation:
  """Frozen scalar annotation used when a concrete dtype rather than Python type is needed."""

  dtype: Any


class Scalar:
  """Subscription marker spelling an explicit scalar ABI dtype, e.g. ``Scalar[vecops.float32]``."""

  def __class_getitem__(cls, dtype: Any) -> ScalarAnnotation:
    """Return a scalar annotation with the requested concrete dtype."""
    return ScalarAnnotation(dtype)


__all__ = [
  "Const",
  "DType",
  "Dynamic",
  "In",
  "InOut",
  "Out",
  "Tensor",
  "TensorPattern",
  "Scalar",
  "ScalarAnnotation",
  "TensorAnnotation",
  "check_tensors",
]
