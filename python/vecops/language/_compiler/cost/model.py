# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Backend-neutral implementation cost rules, in estimated target cycles.

Throughput is reciprocal throughput, not dependency latency. Defaults are
heuristics, not measured timings. All provenance survives profile overlays.
"""

import hashlib
import math
from dataclasses import dataclass, field
from functools import lru_cache

from ..operations import Trait


@dataclass(frozen=True)
class OperationCost:
  throughput_cycles: float
  latency_cycles: float
  temporary_registers: int = 0
  resource: str = "compute"
  confidence: float = 0.25
  provenance: str = "heuristic default"

  def __post_init__(self):
    if any(not math.isfinite(v) or v < 0 for v in (self.throughput_cycles, self.latency_cycles)):
      raise ValueError("cost cycles must be finite and nonnegative")
    if not 0 <= self.confidence <= 1 or self.temporary_registers < 0:
      raise ValueError("invalid cost confidence/register count")


@dataclass(frozen=True)
class CostQuery:
  operation: str
  family: str
  dtype: str
  input_dtype: str = ""
  accuracy: str = "strict"
  implementation: str = "vector"
  words: int = 1
  masked: bool = False
  access: str = "none"


@dataclass(frozen=True)
class CostRule:
  cost: OperationCost
  match: tuple[tuple[str, object], ...] = ()
  priority: int = 0

  def matches(self, query):
    return all(getattr(query, key) == value for key, value in self.match)


@dataclass(frozen=True)
class CostProfile:
  name: str
  rules: tuple[CostRule, ...] = ()
  version: str = "1"
  policies: tuple[tuple[str, float], ...] = ()

  def __post_init__(self):
    fields = CostQuery.__dataclass_fields__
    for rule in self.rules:
      if len(dict(rule.match)) != len(rule.match) or any(key not in fields for key, _ in rule.match):
        raise ValueError("invalid cost rule selector")
    if len(dict(self.policies)) != len(self.policies):
      raise ValueError("duplicate policy setting")
    if any(not math.isfinite(value) or value < 0 for _, value in self.policies):
      raise ValueError("policy settings must be finite and nonnegative")


@dataclass(frozen=True)
class CostModel:
  profiles: tuple[CostProfile, ...]
  catalog: object = field(compare=False, repr=False)

  @property
  def fingerprint(self):
    return hashlib.sha256(repr(self.profiles).encode()).hexdigest()

  def policy(self, name, default):
    for profile in reversed(self.profiles):
      if name in dict(profile.policies):
        return dict(profile.policies)[name]
    return default

  def lookup(self, query):
    return _lookup(self.profiles, query)

  def query(self, expr, *, implementation="vector", words=1):
    definition = self.catalog.get(expr)
    dtype = expr.type.dtype
    input_dtype = expr.args[0].type.dtype if expr.args else dtype
    access = "none"
    if definition.effects == "read" and expr.op == "load":
      input_dtype = expr.data.dtype
      access = "contiguous" if expr.data.strides and getattr(expr.data.strides[-1], "data", None) == 1 else "strided"
    accuracy = expr.data if definition.traits & Trait.ACCURACY else "strict"
    return CostQuery(
      expr.op,
      definition.family,
      dtype,
      input_dtype,
      accuracy or "strict",
      implementation,
      words,
      bool(getattr(expr.type, "mask", False) or expr.op == "load" and expr.args),
      access,
    )


@lru_cache(maxsize=8192)
def _lookup(profiles, query):
  # Bound the cache and retain only immutable profile/query values, not a
  # compilation's semantic catalog, IR or pass context.
  for profile in reversed(profiles):
    matches = [rule for rule in profile.rules if rule.matches(query)]
    if not matches:
      continue
    rank = max((rule.priority, len(rule.match)) for rule in matches)
    selected = [rule for rule in matches if (rule.priority, len(rule.match)) == rank]
    if len(selected) != 1:
      raise ValueError(f"ambiguous cost rules in {profile.name!r} for {query}")
    return selected[0].cost
  # Unknown operations must never become free merely by omission.
  return OperationCost(16, 16, confidence=0, provenance="unknown operation fallback")
