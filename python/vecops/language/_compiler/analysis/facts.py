# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared integer ranges/congruences from validated ABI metadata and value DAGs.

Arithmetic facts are retained only when the complete interval fits the result
dtype. No wraparound or signed-overflow assumption is used to justify a rewrite.
Facts describe values, not permission to drop their evaluation; callers must
independently check effects/traps. Floating values are deliberately unknown.
"""

import math
import re
from dataclasses import dataclass

from ...expr import Const, Dynamic, IntExpr
from .. import ir as I


@dataclass(frozen=True)
class IntegerFacts:
  lower: int
  upper: int
  divisor: int = 1
  residue: int = 0

  @property
  def constant(self):
    return self.lower if self.lower == self.upper else None

  def multiple_of(self, divisor):
    return divisor > 0 and (
      self.constant is not None
      and self.constant % divisor == 0
      or self.divisor % divisor == 0
      and self.residue % divisor == 0
    )


def type_range(dtype):
  if dtype == "bool":
    return IntegerFacts(0, 1)
  match = re.fullmatch(r"(u?)int(8|16|32|64)", dtype)
  if match is None:
    return None
  bits = int(match[2])
  return IntegerFacts(0, (1 << bits) - 1) if match[1] else IntegerFacts(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def _integer(value):
  value = value.value if isinstance(value, Const) else value
  return value if type(value) is int else None


class ValueFacts:
  """Per-query immutable ABI facts, with DAG memoization and no global state."""

  def __init__(self, bindings=None):
    self.bindings = dict(bindings or {})
    self.memo = {}
    self.owned = {}

  @classmethod
  def from_kernel(cls, kernel):
    bindings = {}

    def collect(value, *, extent=False):
      if isinstance(value, IntExpr):
        collect(value.left)
        collect(value.right)
      elif isinstance(value, Dynamic) and value.name:
        limits = type_range("int64")
        lower, upper = _integer(value.lower), _integer(value.upper)
        alignment = _integer(value.alignment) or 1
        lower = max(limits.lower, lower if lower is not None else limits.lower, 0 if extent else limits.lower)
        upper = min(limits.upper, upper if upper is not None else limits.upper)
        if lower <= upper:
          fact = IntegerFacts(lower, upper, alignment)
          previous = bindings.get(("symbol", value.name))
          if previous is not None:
            fact = IntegerFacts(
              max(previous.lower, fact.lower), min(previous.upper, fact.upper), math.lcm(previous.divisor, fact.divisor)
            )
          bindings[("symbol", value.name)] = fact

    for parameter in kernel.parameters.values():
      for dimension in getattr(parameter, "shape", ()):
        collect(dimension, extent=True)
      for stride in getattr(parameter, "concrete_strides", ()):
        collect(stride)
    for name, value in kernel.bindings.items():
      if value.op == "raw" and ("symbol", name) in bindings:
        bindings[("raw", value.data)] = bindings[("symbol", name)]
    return cls(bindings)

  def infer(self, value):
    if id(value) in self.memo:
      return self.memo[id(value)]
    self.owned[id(value)] = value
    limits = type_range(value.type.dtype) if isinstance(value.type, I.ScalarType) else None
    result = limits
    if limits is not None:
      if value.op == "literal" and type(value.data) in (int, bool) and limits.lower <= value.data <= limits.upper:
        result = IntegerFacts(int(value.data), int(value.data), 0, int(value.data))
      elif value.op in {"symbol", "raw", "local", "frozen"}:
        result = self.bindings.get((value.op, value.data), limits)
      elif len(value.args) == 2:
        a, b = (self.infer(argument) for argument in value.args)
        if a is not None and b is not None:
          result = self._binary(value.op, a, b, limits)
    self.memo[id(value)] = result
    return result

  @staticmethod
  def _binary(op, a, b, limits):
    lo, hi, divisor, residue = limits.lower, limits.upper, 1, 0
    if op in {"add", "sub"}:
      lo = a.lower + b.lower if op == "add" else a.lower - b.upper
      hi = a.upper + b.upper if op == "add" else a.upper - b.lower
      divisor = math.gcd(a.divisor, b.divisor)
      residue = a.residue + b.residue if op == "add" else a.residue - b.residue
    elif op == "mul":
      products = (a.lower * b.lower, a.lower * b.upper, a.upper * b.lower, a.upper * b.upper)
      lo, hi = min(products), max(products)
      if b.constant is not None:
        divisor, residue = a.divisor * abs(b.constant), a.residue * b.constant
      elif a.constant is not None:
        divisor, residue = b.divisor * abs(a.constant), b.residue * a.constant
    elif op in {"min", "max"}:
      operation = min if op == "min" else max
      lo, hi = operation(a.lower, b.lower), operation(a.upper, b.upper)
      divisor = math.gcd(math.gcd(a.divisor, b.divisor), a.residue - b.residue)
      residue = a.residue
    elif op in {"floordiv", "mod"} and b.constant is not None and b.constant > 0:
      d = b.constant
      if op == "floordiv":
        lo, hi = a.lower // d, a.upper // d
        if a.multiple_of(d):
          divisor, residue = a.divisor // d, a.residue // d
      elif a.multiple_of(d):
        lo = hi = 0
      else:
        lo, hi = 0, d - 1
    elif op == "bit_and" and b.constant is not None and b.constant >= 0:
      lo, hi = 0, b.constant
    elif op in {"lt", "le", "gt", "ge", "eq", "ne"}:
      outcome = None
      if op in {"lt", "le", "gt", "ge"}:
        x, y = (b, a) if op in {"gt", "ge"} else (a, b)
        inclusive = op in {"le", "ge"}
        if x.upper <= y.lower if inclusive else x.upper < y.lower:
          outcome = True
        elif x.lower > y.upper if inclusive else x.lower >= y.upper:
          outcome = False
      elif a.upper < b.lower or b.upper < a.lower:
        outcome = op == "ne"
      elif a.constant is not None and b.constant is not None:
        outcome = (a.constant == b.constant) == (op == "eq")
      if outcome is not None:
        lo = hi = int(outcome)
    if not limits.lower <= lo <= hi <= limits.upper:
      return limits
    return IntegerFacts(lo, hi, divisor, residue % divisor if divisor else residue)
