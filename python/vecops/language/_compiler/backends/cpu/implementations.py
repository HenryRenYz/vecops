# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU lowering contracts, distinct from target-independent semantics/costs."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from types import MappingProxyType

from ...operations import Trait
from ...operations import snapshot as operation_snapshot


@dataclass(frozen=True)
class CPUImplementation:
  operation: str
  vector_name: str | None = None
  scalar_kind: str = "special"
  scalar_spelling: str | None = None
  wordwise_math: bool = False
  lane_fallback: bool = False
  # C++ entry points may instantiate Half<Tag> implicitly. -1 refers to
  # the result domain; nonnegative entries refer to operand domains.
  half_domains: tuple[int, ...] = ()
  version: str = "1"


@dataclass(frozen=True)
class Implementations:
  definitions: object
  fingerprint: str

  def get(self, operation):
    return self.definitions.get(operation, CPUImplementation(operation))


_registry = {}
_lock = threading.RLock()
_snapshot = None


def register_implementation(implementation):
  global _snapshot
  with _lock:
    if implementation.operation in _registry:
      raise ValueError(f"CPU implementation {implementation.operation!r} already registered")
    _registry[implementation.operation] = implementation
    _snapshot = None


def unregister_implementation(name):
  global _snapshot
  with _lock:
    del _registry[name]
    _snapshot = None


def snapshot():
  global _snapshot
  with _lock:
    if _snapshot is None:
      definitions = dict(_registry)
      _snapshot = Implementations(
        MappingProxyType(definitions), hashlib.sha256(repr(sorted(definitions.items())).encode()).hexdigest()
      )
    return _snapshot


_INFIX = {
  "add": "+",
  "sub": "-",
  "mul": "*",
  "div": "/",
  "lt": "<",
  "le": "<=",
  "gt": ">",
  "ge": ">=",
  "eq": "==",
  "ne": "!=",
  "and": "&&",
  "or": "||",
  "mask_xor": "!=",
  "bit_and": "&",
  "bit_or": "|",
  "bit_xor": "^",
  "shl": "<<",
  "shr": ">>",
}
_PREFIX = {"neg": "-", "not": "!", "identity": "+", "bit_not": "~"}
_VECTOR = {
  "lt": "cmplt",
  "le": "cmple",
  "gt": "cmpgt",
  "ge": "cmpge",
  "eq": "cmpeq",
  "ne": "cmpne",
  "fma": "fmadd",
  "and": "mask_and",
  "or": "mask_or",
  "not": "mask_not",
}
_HALF_DOMAINS = {"lower": (0,), "upper": (0,), "concat": (-1,), "interleave": (-1,)}
for name, definition in operation_snapshot().definitions.items():
  kind, spelling = "special", None
  if name in _INFIX:
    kind, spelling = "infix", _INFIX[name]
  elif name in _PREFIX:
    kind, spelling = "prefix", _PREFIX[name]
  elif definition.traits & Trait.UNARY_MATH:
    kind, spelling = ("reciprocal", name) if name in {"rcp", "rsqrt"} else ("std_math", name.removesuffix("_neg"))
  elif name in {"min", "max"}:
    kind, spelling = "std_binary", name
  lane_fallback = name in {"floordiv", "mod", "ceildiv"}
  register_implementation(
    CPUImplementation(
      name,
      _VECTOR.get(name, name) if definition.traits & Trait.ELEMENTWISE else None,
      kind,
      spelling,
      bool(definition.traits & Trait.ACCURACY),
      lane_fallback,
      _HALF_DOMAINS.get(name, ()),
    )
  )
