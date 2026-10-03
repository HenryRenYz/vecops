"""Kernel-language vocabulary; these operations are interpreted by the AST frontend.

They deliberately cannot execute tensor computation in the host interpreter.
Source-backed templates remain declarations and never execute this vocabulary.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from ._compiler.operations import Trait, snapshot as operation_snapshot


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


class DSLFunction:
  def __init__(self, function):
    functools.update_wrapper(self, function)
    self.function = function

  def __call__(self, *args, **kwargs):
    raise TypeError("T.func helpers are called by the DSL compiler, not the Python interpreter")


def func(function):
  """Declare a source-parsed, specialized and inlined kernel helper."""
  return DSLFunction(function)


_OPERATIONS = (
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
  "where",
  "as_value",
  "_normalize",
  "max",
  "min",
  "fma",
  "ceildiv",
  "arange",
  "lanes",
  "static_assert",
  "compiler_fence",
)
_OPERATIONS += tuple(sorted(operation_snapshot().names(Trait.UNARY_MATH | Trait.SCALAR_REDUCTION)))
globals().update({name: Intrinsic(name) for name in _OPERATIONS})
__all__ = ["Vector", "func", *_OPERATIONS]
