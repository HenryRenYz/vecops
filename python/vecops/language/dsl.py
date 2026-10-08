# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Kernel-language vocabulary; these operations are interpreted by the AST frontend.

They deliberately cannot execute tensor computation in the host interpreter.
Source-backed templates remain declarations and never execute this vocabulary.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from ._compiler.operations import Trait
from ._compiler.operations import snapshot as operation_snapshot


@dataclass(frozen=True)
class Vector:
  """Virtual vector extent, legalized independently of C++ Vec representation.

  N is a positive fixed logical lane count; otherwise pow specifies a
  target-relative scalable extent. Large extents are strip-mined, not spilled
  merely because the C++ tag cannot represent them as one value.
  """

  dtype: object = "float32"
  N: int | None = None
  pow: int = 0

  def __post_init__(self):
    from .expr import Const

    if isinstance(self.N, Const):
      object.__setattr__(self, "N", self.N.value)
    if isinstance(self.pow, Const):
      object.__setattr__(self, "pow", self.pow.value)
    if self.N is not None and (not isinstance(self.N, int) or isinstance(self.N, bool) or self.N <= 0):
      raise ValueError("Vector N must be a positive integer")
    if not isinstance(self.pow, int) or isinstance(self.pow, bool):
      raise TypeError("Vector pow must be an integer")
    if self.N is not None and self.pow != 0:
      raise ValueError("Vector N and nonzero pow are alternative extents")


@dataclass(frozen=True)
class Intrinsic:
  name: str

  def __call__(self, *args, **kwargs):
    raise TypeError(f"T.{self.name} is only available inside a source-less T.template or T.func")


@dataclass(frozen=True)
class ExecutionLayout:
  """Logical iteration order and optional vector axis, not a memory layout.

  order lists outer-to-inner axes. A vector axis must be the innermost axis;
  its induction value is a vector of logical indices, including clipped tails.
  Backend workers/warps are deliberately not encoded in this descriptor.
  """

  order: tuple[int, ...] = ()
  vector_axis: int | None = None
  vector: Vector = Vector()

  def __post_init__(self):
    if not isinstance(self.order, tuple) or any(type(i) is not int for i in self.order):
      raise TypeError("execution order must be an immutable tuple of axes")
    if self.vector_axis is not None and type(self.vector_axis) is not int:
      raise TypeError("vector_axis must be an integer or None")
    if not isinstance(self.vector, Vector):
      raise TypeError("execution vector must be a Vector descriptor")


class DSLFunction:
  def __init__(self, function):
    functools.update_wrapper(self, function)
    self.function = function

  def __call__(self, *args, **kwargs):
    raise TypeError("T.func helpers are called by the DSL compiler, not the Python interpreter")


def func(function):
  """Declare a source-parsed, specialized and inlined kernel helper."""
  return DSLFunction(function)


@dataclass(frozen=True)
class Reducer:
  """A source-parsed binary combiner and immutable scalar/tuple identity.

  associative is an explicit reassociation promise, not inferred for floats.
  Generic combiners preserve left-fold order on the current CPU backend.
  """

  function: object
  identity: object
  associative: bool = False


def reducer(*, identity, associative=False):
  """Declare a pure binary scalar/tuple reduction helper."""

  def immutable(value):
    return isinstance(value, (bool, int, float)) or isinstance(value, tuple) and all(immutable(v) for v in value)

  if not immutable(identity) or type(associative) is not bool:
    raise TypeError("reducer requires an immutable numeric identity and boolean associative flag")
  return lambda function: Reducer(function, identity, associative)


_OPERATIONS = (
  "print",
  "static_print",
  "serial",
  "grid",
  "unroll",
  "tile",
  "transpose",
  "reshape",
  "load",
  "store",
  "copy",
  "dot",
  "pack",
  "packed",
  "_matmul",
  "alloc",
  "zeros",
  "full",
  "cast",
  "bitcast",
  "lower_half",
  "upper_half",
  "concat",
  "interleave",
  "view_disjoint",
  "dtype_name",
  "local_interleave_lower",
  "local_interleave_upper",
  "block_shuffle",
  "itemsize",
  "static_log2",
  "where",
  "as_value",
  "as_vector",
  "map_rows",
  "max",
  "min",
  "fma",
  "ceildiv",
  "arange",
  "lanes",
  "static_assert",
  "compiler_fence",
  "broadcast_to",
  "gather",
  "scatter",
  "shuf",
  "local_shuf",
  "element_at",
  "expand_dims",
  "squeeze",
  "indices",
  "workspace",
  "worker_capacity",
  "parallel",
  "simd",
  "reduce",
  "scan",
  "team",
  "worker_id",
  "worker_count",
  "barrier",
  "team_reduce",
  "shared",
  "atomic_add",
  "atomic_min",
  "atomic_max",
  "atomic_exchange",
  "atomic_compare_exchange",
  "prefetch",
  "pipelined",
  "double_buffer",
)
_OPERATIONS += tuple(
  sorted(operation_snapshot().names(Trait.UNARY_MATH | Trait.SCALAR_REDUCTION | Trait.MASK_REDUCTION))
)
globals().update({name: Intrinsic(name) for name in _OPERATIONS})
__all__ = ["Vector", "ExecutionLayout", "Reducer", "reducer", "func", *_OPERATIONS]
