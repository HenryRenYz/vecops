"""CPU-independent vector operations between tile IR and native legalization.

Expressions describe logical math; VectorOp/OperationCopy make the operations
and iteration identities explicit. A grouping is not a C++ type: target limits,
dtype lane rebinding and memory effects still have to be legalized afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import kernel as I


@dataclass(frozen=True)
class VectorOp:
  index: int
  expression: I.Expr
  operands: tuple[int | None, ...]
  type: I.VectorType
  opaque: bool = False
  native_math_words: bool = False


@dataclass(frozen=True)
class OperationCopy:
  operation: int
  iteration: int
  operands: tuple[tuple[int, int] | None, ...]


@dataclass(frozen=True)
class VectorGroup:
  first: int
  count: int
  power: int
  # Oversized input vectors can stream a repeated legal piece, without
  # allocating an exponentially large Python list or generating huge code.
  repeat: int = 1


@dataclass(frozen=True)
class VectorTailFragment:
  """A native-width masked piece of an existing full-loop vector group.

  ``first`` is measured in tail-vector extents, not logical unroll copies.
  ``path`` selects lower/upper halves from the original accumulator group;
  restoring the same tree preserves its lane and horizontal-reduction order.
  """

  parent: int
  first: int
  power: int
  path: tuple[bool, ...]


@dataclass(frozen=True)
class VectorEffect:
  """Explicit loop prologue, accumulation, converting store, or epilogue."""

  kind: str
  operation: str = ""
  index: int = 0
  value_root: int = -1
  mask_root: int = -1
  dtype: str = ""
  store: I.Store | None = None


def group_vectors(unroll, base_power=0, max_power=5):
  if unroll < 1:
    raise ValueError("vector unroll must be positive")
  if base_power > max_power:
    return tuple(VectorGroup(i, 1, max_power, 1 << (base_power - max_power)) for i in range(unroll))
  result, first = [], 0
  while first < unroll:
    count = 1 << min((unroll - first).bit_length() - 1, max_power - base_power)
    result.append(VectorGroup(first, count, base_power + count.bit_length() - 1))
    first += count
  return tuple(result)


@dataclass
class VectorRegion:
  shape: tuple[I.Expr, ...]
  dtype: str
  unroll: int
  roots: tuple[I.Expr, ...]
  operations: list[VectorOp]
  root_indices: tuple[int | None, ...]
  base_power: int = 0
  initializations: tuple[VectorEffect, ...] = ()
  effects: tuple[VectorEffect, ...] = ()
  finalizations: tuple[VectorEffect, ...] = ()
  iteration_axis: int | None = None
  copies: tuple[OperationCopy, ...] = ()
  groupings: dict[int, tuple[VectorGroup, ...]] = field(default_factory=dict)
  rebind_headroom: int = 0
  full_chunks: bool = False
  pass_trace: list[str] = field(default_factory=list)
  cost: object | None = None
  single_chunk: bool = False
  native_tails: dict[int | None, tuple[VectorTailFragment, ...]] = field(default_factory=dict)
  # Scheduling hint for the independent outer axes, not the vector operation
  # unroll factor above. None leaves native-compiler heuristics unchanged;
  # one keeps each row rolled, without changing the mathematical iteration.
  outer_unroll: int | None = None


@dataclass
class VectorKernel:
  kernel: I.KernelIR
  manager: object = None
  plans: dict[int, VectorRegion] = field(default_factory=dict)
  regions: list[VectorRegion] = field(default_factory=list)
