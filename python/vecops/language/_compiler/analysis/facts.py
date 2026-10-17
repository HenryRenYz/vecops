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
from ...types import TensorType
from .. import ir as I
from ..ir.traversal import walk_unique


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


def abi_metadata_constants(kernel):
  """Exact immutable int64 metadata guaranteed by validated tensor arguments.

  Observed dynamic dimensions, bounds and arbitrary raw expressions do not
  establish these equalities. Optional descriptors may be absent, so querying
  their metadata still requires the original presence check.
  """
  constants = {}
  for name, parameter in getattr(kernel, "parameters", {}).items():
    if type(parameter) is not TensorType or parameter.optional:
      continue
    for operation, dimensions in (("size", parameter.shape), ("stride", parameter.concrete_strides)):
      for axis, dimension in enumerate(dimensions):
        if type(dimension) is Const and type(dimension.value) is int and -(1 << 63) <= dimension.value < 1 << 63:
          constants[f"{name}.{operation}({axis})"] = dimension.value
  return constants


class ValueFacts:
  """Program-point facts with DAG memoization and queried binding keys.

  ``queried`` includes absent bindings: adding a fact can invalidate a proof
  that previously lacked a local range or used a full ABI interval. A scope may preserve its
  caches only when none of the bindings queried since its last reset change.
  """

  def __init__(self, bindings=None):
    self.bindings = dict(bindings or {})
    self.memo = {}
    self.owned = {}
    self.queried = set()

  @classmethod
  def from_kernel(cls, kernel):
    bindings = {("raw", name): IntegerFacts(value, value, 0, value)
                for name, value in abi_metadata_constants(kernel).items()}

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
    result = None
    if limits is not None:
      if value.op == "literal" and type(value.data) in (int, bool) and limits.lower <= value.data <= limits.upper:
        result = IntegerFacts(int(value.data), int(value.data), 0, int(value.data))
      elif value.op in {"symbol", "raw", "local", "frozen"}:
        key = value.op, value.data
        self.queried.add(key)
        # ABI leaves have their declared runtime types. An auto local need
        # not: int8 + a Python integer executes as int64 in emitted C++.
        # Only an established definition may give a local a narrow range.
        result = self.bindings.get(key, limits if value.op in {"symbol", "raw"} else None)
        if result is not None and not limits.lower <= result.lower <= result.upper <= limits.upper:
          result = None
      elif len(value.args) == 2 and self.integer_semantics_modeled(value):
        a, b = (self.infer(argument) for argument in value.args)
        if a is not None and b is not None:
          result = self._binary(value.op, a, b, limits)
    self.memo[id(value)] = result
    return result

  def integer_semantics_modeled(self, value):
    """Whether integer facts can preserve this operation's C++ conversions.

    Scalar integer promotion is not normalized in IR. In particular, usual
    signed/unsigned conversions cannot be replaced by mathematical integer
    comparisons, and typed min/division helpers convert their operands first.
    Check this before rewriting operands into literals, whose C++ type may
    differ from that of the original binding.
    """
    # Unsigned IR literals and auto locals need not execute in their nominal
    # unsigned type. Until that promotion is explicit, keep the whole scalar
    # expression rather than partially replacing its operands with literals.
    if isinstance(value.type, I.ScalarType) and value.type.dtype.startswith("uint"):
      return False
    operations = {"add", "sub", "mul", "min", "max", "floordiv", "mod", "bit_and",
                  "lt", "le", "gt", "ge", "eq", "ne"}
    if value.op not in operations or len(value.args) != 2:
      return True
    if any(not isinstance(a.type, I.ScalarType) or type_range(a.type.dtype) is None for a in value.args):
      return False
    if any(a.type.dtype.startswith("uint") for a in value.args):
      return False
    if value.op in {"min", "max", "floordiv", "mod"}:
      limits = type_range(value.type.dtype)
      arguments = tuple(self.infer(a) for a in value.args)
      return limits is not None and all(
        a is not None and limits.lower <= a.lower <= a.upper <= limits.upper for a in arguments
      )
    return True

  def arithmetic_fits(self, value):
    """Prove one integer operation fits, without signed or unsigned wrapping.

    Callers check children separately. An unknown interval is not a proof of
    safe arithmetic, even when a surrounding predicate has a constant range.
    """
    limits = type_range(value.type.dtype) if isinstance(value.type, I.ScalarType) else None
    if limits is None or value.op not in {"add", "sub", "mul", "neg"}:
      return True
    arguments = tuple(self.infer(argument) for argument in value.args)
    if not arguments or any(argument is None for argument in arguments):
      return False
    a = arguments[0]
    if value.op == "neg":
      lower, upper = -a.upper, -a.lower
    else:
      b = arguments[1]
      if value.op == "add":
        lower, upper = a.lower + b.lower, a.upper + b.upper
      elif value.op == "sub":
        lower, upper = a.lower - b.upper, a.upper - b.lower
      else:
        products = (a.lower * b.lower, a.lower * b.upper, a.upper * b.lower, a.upper * b.upper)
        lower, upper = min(products), max(products)
    return limits.lower <= lower <= upper <= limits.upper

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
    else:
      return None
    if not limits.lower <= lo <= hi <= limits.upper:
      # The executable may have been promoted wider than its nominal IR
      # type. Its mathematical result is not bounded by that nominal type.
      return None
    return IntegerFacts(lo, hi, divisor, residue % divisor if divisor else residue)


def induction_facts(loop, facts):
  """Bound a canonical signed-native For inside its body, never after it.

  The caller proves stable/total bounds and no writes to the induction slot.
  The final increment must fit too: no overflow-based loop reasoning, even
  on the increment which would exit the loop.
  """
  step = I.constant(loop.step)
  if type(step) is not int or step == 0:
    return None
  limits = type_range("int64")
  # uint64 comparisons against a negative native counter have different C++
  # promotions. Leave that mixed-sign domain unoptimized.
  if any(value.type.dtype == "uint64" for value in (loop.start, loop.stop, loop.step)):
    return None
  start, stop = facts.infer(loop.start), facts.infer(loop.stop)
  if start is None or stop is None or not limits.lower <= step <= limits.upper:
    return None
  if not all(limits.lower <= value.lower <= value.upper <= limits.upper for value in (start, stop)):
    return None
  lower, upper = (start.lower, stop.upper - 1) if step > 0 else (stop.lower + 1, start.upper)
  divisor = math.gcd(start.divisor, abs(step))
  residue = start.residue % divisor
  lower += (residue - lower) % divisor
  upper -= (upper - residue) % divisor
  if lower > upper or not limits.lower <= lower + step <= upper + step <= limits.upper:
    return None
  return IntegerFacts(lower, upper, divisor, residue)


@dataclass(frozen=True)
class ScalarAlias:
  value: I.Expr
  dependencies: frozenset


class ScalarAliases:
  """Lexical scalar equalities, separately versioned from interval facts.

  Callers prove exact integer types, effects and total arithmetic at definition
  and use. A dependency write invalidates recomputation, not the stored
  snapshot's independent interval. Missing alias lookups are memo dependencies
  too. Definitions retain shared Expr DAGs and never expand recursively at use.
  """

  max_nodes = 64
  max_dependencies = 64

  def __init__(self, records=None, versions=None):
    self.records = dict(records or {})
    self.versions = dict(versions or {})
    self.queried = set()
    self.dependents = {}
    for key, record in self.records.items():
      for dependency, _version in record.dependencies:
        self.dependents.setdefault(dependency, set()).add(key)

  def copy(self):
    return ScalarAliases(self.records, self.versions)

  @classmethod
  def bounded(cls, value):
    return all(index < cls.max_nodes for index, _node in enumerate(walk_unique(value)))

  def lookup(self, value):
    key = value.op, value.data
    self.queried.add(key)
    record = self.records.get(key)
    if record is not None and all(self.versions.get(name, 0) == version for name, version in record.dependencies):
      return record.value
    return None

  def dependencies(self, value):
    result = set()
    for node in walk_unique(value):
      if node.op in {"local", "frozen"}:
        key = node.op, node.data
        result.add((key, self.versions.get(key, 0)))
        record = self.records.get(key)
        if record is not None:
          result.update(record.dependencies)
        if len(result) > self.max_dependencies:
          # Refuse the equality instead of truncating a dependency. A long
          # copy chain can share one Expr yet accumulate unbounded versions.
          return None
    return frozenset(result)

  def write(self, names):
    keys = {(operation, name) for name in names for operation in ("local", "frozen")}
    invalid = keys | {owner for key in keys for owner in self.dependents.get(key, ())}
    changed = set()
    for key in invalid:
      record = self.records.pop(key, None)
      if record is not None:
        changed.add(key)
        for dependency, _version in record.dependencies:
          self.dependents[dependency].discard(key)
          if not self.dependents[dependency]:
            del self.dependents[dependency]
    for key in keys:
      self.versions[key] = self.versions.get(key, 0) + 1
    return changed

  def define(self, name, value, dependencies):
    assert len(dependencies) <= self.max_dependencies
    record = ScalarAlias(value, dependencies)
    keys = {("local", name), ("frozen", name)}
    assert not any(key in self.records for key in keys), "write must precede the new definition"
    for key in keys:
      self.records[key] = record
      for dependency, _version in dependencies:
        self.dependents.setdefault(dependency, set()).add(key)
    return keys
