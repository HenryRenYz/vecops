"""Structured kernel IR and target-independent expression construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..operations import Trait
from ..operations import current as operation_snapshot
from .promotion import FLOAT_DTYPES, cast_operands, floating_dtype, rounded_float, weak_literal
from .types import Node, ScalarType, TileType, VectorTag, VectorType


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
class ViewRegion(Node):
  """Logical root coordinates retained for grid independence proofs only.

  Address generation still uses offset/strides. Losing this provenance is
  legal, but requires conservative scheduling rather than assumed no-alias.
  """

  root_shape: tuple[Expr, ...]
  root_strides: tuple[Expr, ...]
  starts: tuple[Expr, ...]
  widths: tuple[Expr, ...]
  steps: tuple[int, ...]
  axes: tuple[int, ...]
  clipped: tuple[bool, ...]


@dataclass(frozen=True)
class View(Node):
  storage: str
  dtype: str
  shape: tuple[Expr, ...]
  strides: tuple[Expr, ...]
  offset: Expr
  writable: bool = True
  owned: bool = False
  region: ViewRegion | None = None


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
  kind: str = "serial"
  associative: bool = False
  # Reduction/scan legality metadata; the combiner remains executable body IR.
  reducers: tuple = ()


@dataclass(frozen=True)
class Grid(Node):
  targets: tuple[str, ...]
  extents: tuple[Expr, ...]
  body: Block
  schedule: str = "auto"
  chunk: int = 1
  parallel: str = "auto"


@dataclass(frozen=True)
class UnrolledFor(Node):
  loop: For
  # Operation-major copies; cross-iteration dependencies are checked by a pass.
  copies: Block
  full_vectors: bool = False


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
  # Proven clipped local-SIMD extent; not a promise about memory layout.
  vector_extent: VectorTag | None = None


@dataclass(frozen=True)
class IndexedStore(Node):
  destination: View
  indices: Expr
  value: Expr
  mask: Expr | None = None
  unique: bool = False


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
  operands: tuple = ()


@dataclass(frozen=True)
class AMXTile(Node):
  index: int
  dtype: str
  region: int


@dataclass(frozen=True)
class SMEVector(Node):
  name: str
  view: View
  region: int


@dataclass
class KernelIR(Node):
  body: Block
  parameters: dict[str, Any]
  bindings: dict[str, Expr]
  diagnostics: list[str] = field(default_factory=list)
  materializations: dict[int, View] = field(default_factory=dict)
  pass_trace: list[str] = field(default_factory=list)
  vector_regions: list[Any] = field(default_factory=list)
  origins: dict[int, tuple] = field(default_factory=dict)
  name: str = ""


def literal(value, dtype=None):
  weak = dtype is None
  dtype = dtype or ("bool" if isinstance(value, bool) else "int64" if isinstance(value, int) else "float64")
  if not weak and dtype in FLOAT_DTYPES:
    value = rounded_float(value, dtype)
  return Expr(ScalarType(dtype, weak=weak), "literal", data=value)


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
      if weak_literal(left) and weak_literal(right):
        return literal(fold(left.data, right.data))
      dtype = floating_dtype((left, right), shaped=False)
      if dtype is not None:
        result = fold(rounded_float(left.data, dtype), rounded_float(right.data, dtype))
        comparison = operation_snapshot().has(op, Trait.COMPARISON)
        return literal(result if comparison else rounded_float(result, dtype), dtype="bool" if comparison else dtype)
      # Integer-only folding retains its pre-existing Python-result width.
      # Preserve strong provenance without adding signed/width promotion here.
      result = literal(fold(left.data, right.data))
      return Expr(ScalarType(result.type.dtype), "literal", data=result.data)
  tiles = [v for v in (left, right) if isinstance(v.type, TileType)]
  promoted = floating_dtype((left, right), shaped=bool(tiles))
  if promoted is not None and operation_snapshot().get(op).family != "logical":
    left, right = cast_operands((left, right), promoted, Expr, shaped_type, shaped=bool(tiles))
  else:
    promoted = None
  if tiles:
    typ = tiles[0].type
    shape = typ.shape
    for tile in tiles[1:]:
      shape = broadcast_shape(shape, tile.type.shape)
    comparison = operation_snapshot().has(op, Trait.COMPARISON)
    dtype = promoted or ("float64" if op == "div" else typ.dtype)
    typ = shaped_type(dtype, shape, typ, comparison)
  else:
    dtype = promoted or ("float64" if op == "div" else left.type.dtype)
    typ = ScalarType("bool" if operation_snapshot().has(op, Trait.COMPARISON) else dtype)
  return Expr(typ, op, (left, right))


def same_expr(a, b):
  if (
    a.type.dtype != b.type.dtype
    or getattr(a.type, "weak", False) != getattr(b.type, "weak", False)
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
    and same_expr(a.args[0], b.args[1])
    and same_expr(a.args[1], b.args[0])
  ):
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
  """Memory-read occurrences, preserving repeated incoming argument edges."""
  if operation_snapshot().has(expr, Trait.MEMORY_READ):
    yield expr
  for argument in expr.args:
    yield from loads(argument)


def loads_unique(expr):
  """Memory-read identities for dependency/alias membership, not use counts."""
  from .traversal import walk_unique

  catalog = operation_snapshot()
  for value in walk_unique(expr):
    if catalog.has(value, Trait.MEMORY_READ):
      yield value
