"""Symbolic compile-time and runtime values for :mod:`vecops.language`."""

from __future__ import annotations

import enum
import operator
from typing import Any as TypingAny

from .. import _C
from .._dtype import is_dtype_name, normalize_dtype

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class Predicate:
  """A comparison whose truth value means "proven true at compile time"."""

  __slots__ = ("_known", "_value")

  def __init__(self, known: bool, value: bool = False) -> None:
    self._known = bool(known)
    self._value = bool(value)

  @property
  def known(self) -> bool:
    return self._known

  @property
  def value(self) -> bool:
    if not self._known:
      raise ValueError("symbolic predicate is not known at compile time")
    return self._value

  def __bool__(self) -> bool:
    return self._known and self._value

  def __repr__(self) -> str:
    return f"Predicate({self._value})" if self._known else "Predicate(unknown)"


class SymbolicInt:
  """Base class for integer values and expression nodes."""

  __slots__ = ()

  def _binary(self, op: str, other: TypingAny):
    return IntExpr(op, self, _as_symbolic_int(other))

  def __add__(self, other):
    return self._binary("add", other)

  def __radd__(self, other):
    return _as_symbolic_int(other)._binary("add", self)

  def __sub__(self, other):
    return self._binary("sub", other)

  def __rsub__(self, other):
    return _as_symbolic_int(other)._binary("sub", self)

  def __mul__(self, other):
    return self._binary("mul", other)

  def __rmul__(self, other):
    return _as_symbolic_int(other)._binary("mul", self)

  def __floordiv__(self, other):
    return self._binary("floordiv", other)

  def __rfloordiv__(self, other):
    return _as_symbolic_int(other)._binary("floordiv", self)

  def __mod__(self, other):
    return self._binary("mod", other)

  def __rmod__(self, other):
    return _as_symbolic_int(other)._binary("mod", self)

  def _compare(self, op, other) -> Predicate:
    right = _as_symbolic_int(other)
    left_value = self.compile_value
    right_value = right.compile_value
    if left_value is not None and right_value is not None:
      return Predicate(True, op(left_value, right_value))
    return Predicate(False)

  def __lt__(self, other):
    return self._compare(operator.lt, other)

  def __le__(self, other):
    return self._compare(operator.le, other)

  def __eq__(self, other):
    try:
      return self._compare(operator.eq, other)
    except TypeError:
      return Predicate(True, False)

  def __ne__(self, other):
    return self._compare(operator.ne, other)

  def __gt__(self, other):
    return self._compare(operator.gt, other)

  def __ge__(self, other):
    return self._compare(operator.ge, other)

  @property
  def compile_value(self) -> int | None:
    raise NotImplementedError

  @property
  def observed(self) -> int | None:
    raise NotImplementedError

  def bind_name(self, name: str) -> None:
    """Assign a stable source-level name when this is an unnamed leaf."""

  def _native_dimension(self):
    raise NotImplementedError

  def _collect_symbols(self, symbols: dict[str, type]) -> None:
    raise NotImplementedError

  def _runtime_symbols(self) -> set[str]:
    raise NotImplementedError

  def canonical(self):
    raise NotImplementedError


class Const(SymbolicInt):
  """An exact compile-time integer or boolean."""

  __slots__ = ("name", "value")

  def __init__(self, value: int | bool, name: str | None = None) -> None:
    if not isinstance(value, (int, bool)):
      raise TypeError("T.Const requires an int or bool")
    integer = int(value)
    if integer < _INT64_MIN or integer > _INT64_MAX:
      raise OverflowError("T.Const value does not fit int64")
    self.value = value
    self.name = name

  @property
  def compile_value(self) -> int:
    return int(self.value)

  @property
  def observed(self) -> int:
    return int(self.value)

  def bind_name(self, name: str) -> None:
    if self.name is None:
      self.name = name

  def _native_dimension(self):
    if isinstance(self.value, bool):
      raise TypeError("boolean T.Const cannot be used as a tensor dimension")
    return _C.DimensionDef(self.name) if self.name else _C.DimensionDef(int(self.value))

  def _collect_symbols(self, symbols: dict[str, type]) -> None:
    if self.name:
      from .._dtype import Const as SchemaConst

      symbols.setdefault(self.name, SchemaConst)

  def _runtime_symbols(self) -> set[str]:
    return set()

  def canonical(self):
    return ("const", self.name, type(self.value).__name__, int(self.value))

  def __int__(self) -> int:
    return int(self.value)

  def __index__(self) -> int:
    return int(self.value)

  def __bool__(self) -> bool:
    return bool(self.value)

  def __repr__(self) -> str:
    return f"Const({self.value!r}, name={self.name!r})"


def _constraint_value(value: int | Const, what: str) -> int | str:
  if isinstance(value, bool):
    raise TypeError(f"dynamic {what} cannot be bool")
  if isinstance(value, int):
    return value
  if isinstance(value, Const):
    if value.name:
      return value.name
    return int(value.value)
  raise TypeError(f"dynamic {what} must be an int or T.Const")


class Dynamic(SymbolicInt):
  """A non-specializing runtime integer with optional observed value."""

  __slots__ = ("_observed", "alignment", "lower", "name", "upper")

  def __init__(
    self,
    alignment: int | Const = 1,
    lower: int | Const = _INT64_MIN,
    upper: int | Const = _INT64_MAX,
    *,
    observed: int | None = None,
    name: str | None = None,
  ) -> None:
    self.alignment = alignment
    self.lower = lower
    self.upper = upper
    self.name = name
    self._observed = int(observed) if observed is not None else None

  @property
  def compile_value(self) -> None:
    return None

  @property
  def observed(self) -> int | None:
    return self._observed

  def bind_name(self, name: str) -> None:
    if self.name is None:
      self.name = name

  def _native_dimension(self):
    arguments = (
      _constraint_value(self.alignment, "alignment"),
      _constraint_value(self.lower, "lower bound"),
      _constraint_value(self.upper, "upper bound"),
    )
    if self.name:
      return _C.DimensionDef.named_dynamic(self.name, *arguments)
    return _C.DimensionDef.dynamic(*arguments)

  def _collect_symbols(self, symbols: dict[str, type]) -> None:
    from .._dtype import Const as SchemaConst

    for value in (self.alignment, self.lower, self.upper):
      if isinstance(value, Const) and value.name:
        symbols.setdefault(value.name, SchemaConst)

  def _runtime_symbols(self) -> set[str]:
    return {self.name} if self.name else set()

  def canonical(self):
    def item(value):
      return value.canonical() if isinstance(value, Const) else ("literal", value)

    return ("dynamic", self.name, item(self.alignment), item(self.lower), item(self.upper))

  def __repr__(self) -> str:
    return f"Dynamic(name={self.name!r}, observed={self._observed!r})"


class IntExpr(SymbolicInt):
  """An immutable binary integer expression."""

  __slots__ = ("left", "op", "right")

  def __init__(self, op: str, left: SymbolicInt, right: SymbolicInt) -> None:
    if op in {"floordiv", "mod"} and right.compile_value == 0:
      raise ZeroDivisionError("symbolic dimension expression divides by zero")
    self.op = op
    self.left = left
    self.right = right

  @property
  def compile_value(self) -> int | None:
    left = self.left.compile_value
    right = self.right.compile_value
    return None if left is None or right is None else _evaluate(self.op, left, right)

  @property
  def observed(self) -> int | None:
    left = self.left.observed
    right = self.right.observed
    return None if left is None or right is None else _evaluate(self.op, left, right)

  def bind_name(self, name: str) -> None:
    del name

  def _native_expr(self):
    left = _native_expr(self.left)
    right = _native_expr(self.right)
    method = {
      "add": "add",
      "sub": "sub",
      "mul": "mul",
      "floordiv": "floordiv",
      "mod": "mod",
      "min": "minimum",
      "max": "maximum",
    }[self.op]
    return getattr(_C.DimensionExpr, method)(left, right)

  def _native_dimension(self):
    return _C.DimensionDef.expression(self._native_expr())

  def _collect_symbols(self, symbols: dict[str, type]) -> None:
    self.left._collect_symbols(symbols)
    self.right._collect_symbols(symbols)

  def _runtime_symbols(self) -> set[str]:
    return self.left._runtime_symbols() | self.right._runtime_symbols()

  def canonical(self):
    return (self.op, self.left.canonical(), self.right.canonical())

  def __repr__(self) -> str:
    return f"IntExpr({self.op}, {self.left!r}, {self.right!r})"


def _native_expr(value: SymbolicInt):
  if isinstance(value, IntExpr):
    return value._native_expr()
  if isinstance(value, Const):
    if isinstance(value.value, bool):
      raise TypeError("boolean T.Const cannot be used in a dimension expression")
    return _C.DimensionExpr.const_symbol(value.name) if value.name else _C.DimensionExpr.literal(int(value.value))
  if isinstance(value, Dynamic):
    if not value.name:
      raise ValueError("an anonymous dynamic value cannot be used in a derived dimension expression")
    return _C.DimensionExpr.runtime_symbol(value.name)
  raise TypeError(f"unsupported dimension expression leaf {value!r}")


def _evaluate(op: str, left: int, right: int) -> int:
  result = {
    "add": lambda: left + right,
    "sub": lambda: left - right,
    "mul": lambda: left * right,
    "floordiv": lambda: left // right,
    "mod": lambda: left % right,
    "min": lambda: min(left, right),
    "max": lambda: max(left, right),
  }[op]()
  if result < _INT64_MIN or result > _INT64_MAX:
    raise OverflowError("symbolic dimension expression overflows int64")
  return result


def _as_symbolic_int(value: TypingAny) -> SymbolicInt:
  if isinstance(value, SymbolicInt):
    return value
  if isinstance(value, (int, bool)):
    return Const(value)
  raise TypeError(f"integer expression expected, got {type(value).__name__}")


class DType:
  """A normalized compile-time dtype value."""

  __slots__ = ("name", "value")

  def __init__(self, value: TypingAny, name: str | None = None) -> None:
    self.value = normalize_dtype(value)
    self.name = name

  def bind_name(self, name: str) -> None:
    if self.name is None:
      self.name = name

  def __vecops_dtype__(self) -> _C.DType:
    """Expose the concrete dtype without coupling normalizers to the DSL."""
    return self.value

  def canonical(self):
    return ("dtype", self.name, str(self.value))

  def __repr__(self) -> str:
    return f"DType({self.value!r}, name={self.name!r})"


def dynamic(
  alignment: int | Const = 1,
  lower: int | Const = _INT64_MIN,
  upper: int | Const = _INT64_MAX,
) -> Dynamic:
  return Dynamic(alignment, lower, upper)


Any = Dynamic
any = Dynamic()


def is_value(value: TypingAny) -> bool:
  return isinstance(value, Const) or not isinstance(value, SymbolicInt)


def minimum(left, right) -> SymbolicInt:
  return IntExpr("min", _as_symbolic_int(left), _as_symbolic_int(right))


def maximum(left, right) -> SymbolicInt:
  return IntExpr("max", _as_symbolic_int(left), _as_symbolic_int(right))


def normalize_value(value: TypingAny, *, name: str | None = None):
  """Normalize one immutable outer-JIT value; mutable containers are rejected."""
  # A normalized value may be forwarded into another T.jit function. Give the
  # callee a fresh leaf named for its own parameter instead of mutating or
  # retaining the caller's specialization name.
  if isinstance(value, Const):
    return Const(value.value, name or value.name)
  if isinstance(value, Dynamic):
    return Dynamic(
      value.alignment,
      value.lower,
      value.upper,
      observed=value.observed,
      name=name or value.name,
    )
  if isinstance(value, DType):
    return DType(value.value, name or value.name)
  if isinstance(value, IntExpr):
    return value
  if isinstance(value, bool):
    return Const(value, name)
  if isinstance(value, int):
    return Const(value, name)
  if value is None:
    return None
  if isinstance(value, tuple):
    return tuple(normalize_value(item, name=f"{name}_{index}" if name else None) for index, item in enumerate(value))
  if isinstance(value, (list, dict, set, bytearray)):
    raise TypeError(f"T.jit does not support mutable argument type {type(value).__name__}")
  if isinstance(value, _C.DType):
    return DType(value, name)
  if is_dtype_name(value):
    return DType(value, name)
  module = getattr(value, "__module__", type(value).__module__).split(".", 1)[0]
  if module in {"torch", "numpy", "ml_dtypes"}:
    try:
      return DType(value, name)
    except TypeError:
      pass
  if isinstance(value, (str, bytes, float, enum.Enum)):
    return value
  try:
    hash(value)
  except TypeError as error:
    raise TypeError(f"T.jit argument {type(value).__name__} is not immutable/hashable") from error
  raise TypeError(f"T.jit does not know how to normalize {type(value).__name__}")


def cache_key(value: TypingAny):
  if isinstance(value, (Const, Dynamic, DType, IntExpr)):
    return value.canonical()
  if isinstance(value, tuple):
    return ("tuple", tuple(cache_key(item) for item in value))
  if value is None:
    return ("none",)
  if isinstance(value, enum.Enum):
    return ("enum", type(value).__module__, type(value).__qualname__, value.name)
  return (type(value).__module__, type(value).__qualname__, value)


__all__ = [
  "Any",
  "Const",
  "DType",
  "Dynamic",
  "IntExpr",
  "Predicate",
  "SymbolicInt",
  "any",
  "cache_key",
  "dynamic",
  "is_value",
  "maximum",
  "minimum",
  "normalize_value",
]
