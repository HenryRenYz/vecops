# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Pythonic declarative kernel schemas lowered to :mod:`vecops._C`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from . import _C
from ._dtype import Const, DType, normalize_dtype, scalar_dtype

_MISSING = object()
TensorAccess = _C.TensorAccess


@dataclass(frozen=True)
class Dynamic:
  """A runtime dimension constrained by fixed or named Const bounds.

  ``alignment``, ``lower``, and ``upper`` accept an integer or a named
  specialization string. String expressions must resolve to ``vecops.Const``
  values; bounds are inclusive.
  """

  alignment: int | str = 1
  lower: int | str = -(2**63)
  upper: int | str = 2**63 - 1


Dimension = int | str | type[Const] | Dynamic


def _dimension(value: Dimension, symbols: Mapping[str, Dynamic] | None = None) -> _C.DimensionDef:
  if value is Const:
    return _C.DimensionDef.constant()
  if isinstance(value, Dynamic):
    return _C.DimensionDef.dynamic(value.alignment, value.lower, value.upper)
  if isinstance(value, bool) or not isinstance(value, (int, str)):
    raise TypeError(f"invalid dimension declaration {value!r}")
  if isinstance(value, str) and symbols is not None and value in symbols:
    declaration = symbols[value]
    return _C.DimensionDef.named_dynamic(
      value, declaration.alignment, declaration.lower, declaration.upper
    )
  return _C.DimensionDef(value)


def _collect_dimension_symbols(value: Dimension, symbols: dict[str, type]) -> None:
  if isinstance(value, str):
    symbols.setdefault(value, Const)
  elif isinstance(value, Dynamic):
    for expression in (value.alignment, value.lower, value.upper):
      if isinstance(expression, str):
        existing = symbols.setdefault(expression, Const)
        if existing is not Const:
          raise TypeError(f"specialization symbol {expression!r} has conflicting kinds")


@dataclass(frozen=True)
class TensorDef:
  """Declarative tensor parameter for a :class:`KernelDef`.

  Shape and strides accept fixed integers, named Const symbols, ``Const`` for
  an anonymous compile-time value, or :class:`Dynamic`.  ``dtype=None`` is an
  anonymous dtype specialization; a string names a ``DType`` specialization.
  Outputs and in-outs are caller-owned mutable tensors and cannot be optional.
  """

  name: str
  shape: tuple[Dimension, ...]
  strides: tuple[Dimension, ...] | None = None
  dtype: Any = None
  optional: bool = False
  access: _C.TensorAccess = _C.TensorAccess.input
  device_type: int = 1
  device_index: int = -1

  def __init__(
    self,
    name: str,
    shape: Iterable[Dimension],
    strides: Iterable[Dimension] | None = None,
    *,
    dtype: Any = None,
    optional: bool = False,
    access: _C.TensorAccess = _C.TensorAccess.input,
    output: bool = False,
    device_type: int = 1,
    device_index: int = -1,
  ) -> None:
    if output:
      if access != _C.TensorAccess.input:
        raise ValueError("cannot specify output=True together with a non-input access")
      access = _C.TensorAccess.output
    shape_tuple = tuple(shape)
    stride_tuple = tuple(strides) if strides is not None else tuple(Const for _ in shape_tuple)
    if len(shape_tuple) != len(stride_tuple):
      raise ValueError(f"tensor {name!r} shape and stride ranks differ")
    if optional and access != _C.TensorAccess.input:
      raise ValueError("writable tensors cannot be optional")
    object.__setattr__(self, "name", name)
    object.__setattr__(self, "shape", shape_tuple)
    object.__setattr__(self, "strides", stride_tuple)
    object.__setattr__(self, "dtype", dtype)
    object.__setattr__(self, "optional", optional)
    object.__setattr__(self, "access", access)
    object.__setattr__(self, "device_type", device_type)
    object.__setattr__(self, "device_index", device_index)

  @property
  def output(self) -> bool:
    """Compatibility spelling for whether this parameter is writable."""
    return self.access != _C.TensorAccess.input

  def _native(self, symbols: Mapping[str, Dynamic] | None = None) -> _C.TensorDef:
    dtype = self.dtype if isinstance(self.dtype, str) or self.dtype is None else normalize_dtype(self.dtype)
    return _C.TensorDef(
      self.name,
      [_dimension(item, symbols) for item in self.shape],
      [_dimension(item, symbols) for item in self.strides or ()],
      dtype=dtype,
      optional=self.optional,
      access=self.access,
      device_type=self.device_type,
      device_index=self.device_index,
    )

  def _collect_symbols(self, symbols: dict[str, type]) -> None:
    for item in (*self.shape, *(self.strides or ())):
      _collect_dimension_symbols(item, symbols)
    if isinstance(self.dtype, str):
      existing = symbols.setdefault(self.dtype, DType)
      if existing is not DType:
        raise TypeError(f"specialization symbol {self.dtype!r} has conflicting kinds")


@dataclass(frozen=True)
class ValueDef:
  """Runtime scalar parameter with an optional Python default value."""

  name: str
  dtype: Any
  default: Any = _MISSING

  @property
  def has_default(self) -> bool:
    """Whether binding may synthesize this trailing argument from ``default``."""
    return self.default is not _MISSING

  def _native(self) -> _C.ValueDef:
    dtype = scalar_dtype(self.dtype)
    if not self.has_default:
      return _C.ValueDef(self.name, dtype)
    return _C.ValueDef(self.name, dtype, self.default)


ParameterDef = TensorDef | ValueDef


@dataclass(frozen=True)
class TensorMeta:
  """Storage-free tensor metadata used to bind or compile a specialization.

  When strides are omitted, this creates dense row-major element strides. It
  is valid for :meth:`vecops.JitKernel.compile_for`, but cannot execute because
  it intentionally contains no data pointer.
  """

  shape: tuple[int, ...]
  dtype: Any
  strides: tuple[int, ...] | None = None
  access: _C.TensorAccess = _C.TensorAccess.input
  device_type: int = 1
  device_index: int = 0

  def __init__(
    self,
    shape: Iterable[int],
    *,
    dtype: Any,
    strides: Iterable[int] | None = None,
    access: _C.TensorAccess = _C.TensorAccess.input,
    device_type: int = 1,
    device_index: int = 0,
  ) -> None:
    shape = tuple(shape)
    if strides is None:
      computed = [1] * len(shape)
      for axis in range(len(shape) - 2, -1, -1):
        computed[axis] = computed[axis + 1] * shape[axis + 1]
      strides = tuple(computed)
    else:
      strides = tuple(strides)
    if len(shape) != len(strides):
      raise ValueError("TensorMeta shape and strides must have equal lengths")
    object.__setattr__(self, "shape", shape)
    object.__setattr__(self, "dtype", dtype)
    object.__setattr__(self, "strides", strides)
    object.__setattr__(self, "access", access)
    object.__setattr__(self, "device_type", device_type)
    object.__setattr__(self, "device_index", device_index)

  def _native(self) -> _C.TensorMeta:
    flags = 1 if self.access == _C.TensorAccess.input else 2 if self.access == _C.TensorAccess.output else 3
    return _C.TensorMeta(
      normalize_dtype(self.dtype),
      self.shape,
      self.strides,
      flags,
      self.device_type,
      self.device_index,
    )


def _specialization_kind(value: Any) -> Any:
  if value is Const:
    return _C.ConstInt
  if value is DType:
    return _C.DTypeValue
  if isinstance(value, _C.SpecializationType):
    return value
  raise TypeError(f"invalid specialization kind {value!r}; expected vecops.Const or vecops.DType")


class KernelDef:
  """Immutable public source-kernel schema with automatic symbol discovery.

  Dimension and dtype names are collected into specialization ``values`` by
  default. Explicit ``symbols`` map selected dimension names to named Dynamic
  declarations instead; their runtime values enforce equality but do not
  specialize artifacts. Explicit ``values`` may add names used directly by
  kernel C++ as ``vecops::spec`` references. Construction validates the
  lowered native schema without compiling source or allocating tensors.
  """

  def __init__(
    self,
    name: str,
    parameters: Iterable[ParameterDef],
    values: Mapping[str, Any] | None = None,
    *,
    symbols: Mapping[str, Dynamic] | None = None,
  ) -> None:
    self.name = name
    self.parameters = tuple(parameters)
    declared_symbols = dict(symbols or {})
    for symbol, declaration in declared_symbols.items():
      if not isinstance(symbol, str) or not symbol:
        raise TypeError("runtime dimension symbol names must be non-empty strings")
      if not isinstance(declaration, Dynamic):
        raise TypeError(
          f"runtime dimension symbol {symbol!r} must be declared with vecops.Dynamic"
        )
    self.symbols = MappingProxyType(declared_symbols)
    inferred: dict[str, type] = {}
    for symbol, declaration in declared_symbols.items():
      for expression in (
        declaration.alignment,
        declaration.lower,
        declaration.upper,
      ):
        if isinstance(expression, str):
          if expression in declared_symbols:
            raise TypeError(
              f"runtime dimension symbol {symbol!r} constraint cannot reference "
              f"runtime dimension symbol {expression!r}"
            )
          inferred.setdefault(expression, Const)
    used_runtime_symbols: set[str] = set()
    for parameter in self.parameters:
      if isinstance(parameter, TensorDef):
        for item in (*parameter.shape, *(parameter.strides or ())):
          if isinstance(item, str) and item in declared_symbols:
            used_runtime_symbols.add(item)
          else:
            _collect_dimension_symbols(item, inferred)
        if isinstance(parameter.dtype, str):
          if parameter.dtype in declared_symbols:
            raise TypeError(
              f"runtime dimension symbol {parameter.dtype!r} cannot be used as a dtype"
            )
          existing = inferred.setdefault(parameter.dtype, DType)
          if existing is not DType:
            raise TypeError(
              f"specialization symbol {parameter.dtype!r} has conflicting kinds"
            )
      elif not isinstance(parameter, ValueDef):
        raise TypeError(f"unsupported kernel parameter {parameter!r}")
    unused = declared_symbols.keys() - used_runtime_symbols
    if unused:
      raise ValueError(f"unused runtime dimension symbol {next(iter(sorted(unused)))!r}")
    for symbol, kind in (values or {}).items():
      if symbol in declared_symbols:
        raise TypeError(
          f"runtime dimension symbol {symbol!r} conflicts with a specialization value"
        )
      normalized = Const if kind is Const else DType if kind is DType else kind
      if symbol in inferred and inferred[symbol] is not normalized:
        raise TypeError(f"specialization symbol {symbol!r} has conflicting kinds")
      inferred[symbol] = normalized
    self.values = MappingProxyType(inferred)
    self._native().validate()

  @property
  def inputs(self) -> tuple[ParameterDef, ...]:
    """Return ordered parameters; ``inputs`` is retained for C++ API parity."""
    return self.parameters

  def _native(self, *, name: str | None = None) -> _C.KernelDef:
    parameters = [
      parameter._native(self.symbols) if isinstance(parameter, TensorDef) else parameter._native()
      for parameter in self.parameters
    ]
    values = {symbol: _specialization_kind(kind) for symbol, kind in self.values.items()}
    return _C.KernelDef(name or self.name, parameters, values)


__all__ = ["Dynamic", "KernelDef", "TensorAccess", "TensorDef", "TensorMeta", "ValueDef"]
