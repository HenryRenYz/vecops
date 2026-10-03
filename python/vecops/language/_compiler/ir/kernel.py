"""Structured kernel IR and target-independent expression construction."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .types import Node, ScalarType, TileType, VectorTag, VectorType
from ..operations import Trait, current as operation_snapshot


def shaped_type(dtype, shape, prototype=None, mask=False, keep_tag=False):
  if (
    isinstance(prototype, VectorType)
    and len(shape) == 1
    and (keep_tag or all(same_expr(a, b) for a, b in zip(shape, prototype.shape)))
  ):
    return VectorType(dtype, shape, mask, prototype.tag)
  return TileType(dtype, shape, mask)


@dataclass(frozen=True, eq=False)
class Expr(Node):
  type: ScalarType | TileType
  op: str
  args: tuple[Expr, ...] = ()
  data: Any = None


@dataclass(frozen=True)
class View(Node):
  storage: str
  dtype: str
  shape: tuple[Expr, ...]
  strides: tuple[Expr, ...]
  offset: Expr
  writable: bool = True
  owned: bool = False


@dataclass
class Block(Node):
  statements: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Assign(Node):
  target: str
  value: Expr
  declare: bool = True


@dataclass(frozen=True)
class BindTile(Node):
  value: Expr


@dataclass(frozen=True)
class ReductionGroup(Node):
  assignments: tuple[Assign, ...]
  stores: tuple[Store, ...] = ()


@dataclass(frozen=True)
class ElementwiseGroup(Node):
  stores: tuple[Store, ...]


@dataclass(frozen=True)
class For(Node):
  target: str
  start: Expr
  stop: Expr
  step: Expr
  body: Block
  unroll: int = 2


@dataclass(frozen=True)
class UnrolledFor(Node):
  loop: For
  # Operation-major copies; cross-iteration dependencies are checked by a pass.
  copies: Block


@dataclass(frozen=True)
class If(Node):
  condition: Expr
  body: Block
  otherwise: Block


@dataclass(frozen=True)
class While(Node):
  condition: Expr
  body: Block


@dataclass(frozen=True)
class Store(Node):
  destination: View
  value: Expr
  mask: Expr | None = None
  implementation: str = "auto"


@dataclass(frozen=True)
class Allocate(Node):
  view: View


@dataclass(frozen=True)
class Assert(Node):
  condition: Expr
  message: str


@dataclass(frozen=True)
class Control(Node):
  kind: str


@dataclass(frozen=True)
class HardwareRegion(Node):
  kind: str
  body: Block
  configuration: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ArchitectureCall(Node):
  operation: str
  tile: int = 0
  view: View | None = None


@dataclass
class KernelIR(Node):
  body: Block
  parameters: dict[str, Any]
  bindings: dict[str, Expr]
  diagnostics: list[str] = field(default_factory=list)
  materializations: dict[int, View] = field(default_factory=dict)
  pass_trace: list[str] = field(default_factory=list)
  vector_regions: list[Any] = field(default_factory=list)


def literal(value, dtype=None):
  dtype = dtype or ("bool" if isinstance(value, bool) else "int64" if isinstance(value, int) else "float64")
  return Expr(ScalarType(dtype), "literal", data=value)


def constant(expr):
  return expr.data if isinstance(expr, Expr) and expr.op == "literal" else None


def binary(op, left, right):
  divisor = constant(right)
  if (
    op in {"mod", "floordiv"}
    and isinstance(left.type, TileType)
    and "int" in left.type.dtype
    and isinstance(divisor, int)
    and divisor > 0
    and divisor & (divisor - 1) == 0
  ):
    return binary(
      "bit_and" if op == "mod" else "shr", left, literal(divisor - 1 if op == "mod" else divisor.bit_length() - 1)
    )
  if op == "and" and left.op == "literal" and left.data is True:
    return right
  if op == "and" and left.op == "literal" and left.data is False and isinstance(right.type, ScalarType):
    return left
  if op == "or" and left.op == "literal" and left.data is False:
    return right
  if op == "or" and left.op == "literal" and left.data is True and isinstance(right.type, ScalarType):
    return left
  if op in {"min", "max"} and same_expr(left, right):
    return left
  if (
    isinstance(left.type, ScalarType)
    and isinstance(right.type, ScalarType)
    and left.type.dtype.startswith("int")
    and right.type.dtype.startswith("int")
  ):
    if op in {"add", "sub"} and constant(right) == 0:
      return left
    if op == "add" and constant(left) == 0:
      return right
    if op in {"mul", "floordiv"} and constant(right) == 1:
      return left
    if op == "mul" and constant(left) == 1:
      return right
  if left.op == right.op == "literal":
    fold = operation_snapshot().get(op).fold
    if fold is not None:
      return literal(fold(left.data, right.data))
  tiles = [v for v in (left, right) if isinstance(v.type, TileType)]
  if tiles:
    typ = tiles[0].type
    shape = typ.shape
    for tile in tiles[1:]:
      shape = broadcast_shape(shape, tile.type.shape)
    comparison = operation_snapshot().has(op, Trait.COMPARISON)
    float_dtypes = [v.type.dtype for v in tiles if v.type.dtype.startswith("float")]
    dtype = max(float_dtypes, key=lambda d: int(d[5:])) if float_dtypes else "float64" if op == "div" else typ.dtype
    typ = shaped_type(dtype, shape, typ, comparison)
  else:
    floating = [v.type.dtype for v in (left, right) if v.type.dtype.startswith("float")]
    dtype = (
      "float64" if "float64" in floating or op == "div" and not floating else "float32" if floating else left.type.dtype
    )
    typ = ScalarType("bool" if operation_snapshot().has(op, Trait.COMPARISON) else dtype)
  return Expr(typ, op, (left, right))


def same_expr(a, b):
  if (
    a.type.dtype != b.type.dtype
    or isinstance(a.type, TileType) != isinstance(b.type, TileType)
    or getattr(a.type, "mask", False) != getattr(b.type, "mask", False)
  ):
    return False
  if (
    isinstance(a.type, ScalarType)
    and "int" in a.type.dtype
    and a.op == b.op
    and a.op in {"min", "max"}
    and len(a.args) == len(b.args) == 2
  ):
    if same_expr(a.args[0], b.args[1]) and same_expr(a.args[1], b.args[0]):
      return True
  return a is b or (
    a.op == b.op
    and a.data == b.data
    and len(a.args) == len(b.args)
    and all(same_expr(x, y) for x, y in zip(a.args, b.args))
  )


def broadcast_shape(a, b):
  result = []
  for i in range(1, max(len(a), len(b)) + 1):
    x = a[-i] if i <= len(a) else literal(1)
    y = b[-i] if i <= len(b) else literal(1)
    if constant(x) == 1:
      result.append(y)
    elif constant(y) == 1 or same_expr(x, y):
      result.append(x)
    else:
      if x.op == y.op == "literal":
        raise TypeError("tile shapes are not broadcast-compatible")
      result.append(Expr(ScalarType("int64"), "broadcast_dim", (x, y)))
  return tuple(reversed(result))


def loads(expr):
  if expr.op == "load":
    yield expr
  for argument in expr.args:
    yield from loads(argument)
