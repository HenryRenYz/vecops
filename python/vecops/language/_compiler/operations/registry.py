"""Target-independent operation contracts, frozen for each compilation.

Traits are semantic promises, never target costs or ISA capabilities. Unknown
operations are conservatively opaque; a backend must explicitly lower them.
"""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import IntFlag, auto
from types import MappingProxyType


class Trait(IntFlag):
  NONE = 0
  ELEMENTWISE = auto()
  UNARY_MATH = auto()
  ACCURACY = auto()
  COMPARISON = auto()
  SCALAR_REDUCTION = auto()
  REDUCTION = auto()
  VECTOR_BOUNDARY = auto()
  REFERENCE = auto()
  RUNTIME_SHAPE = auto()
  CAPTURE_SCALAR = auto()
  UNSTABLE_EXTENT = auto()
  CSE = auto()
  CONDITIONAL = auto()
  TRANSPOSE_STREAMABLE = auto()


@dataclass(frozen=True)
class OpDef:
  name: str
  family: str
  traits: Trait = Trait.NONE
  effects: str = "none"
  arity: int | None = None
  keywords: tuple[str, ...] = ()
  reduction: str | None = None
  fold: object = None
  version: str = "1"


@dataclass(frozen=True)
class OperationCatalog:
  definitions: object
  fingerprint: str

  def get(self, operation):
    name = operation if isinstance(operation, str) else operation.op
    return self.definitions.get(name, OpDef(name, "unknown", effects="unknown"))

  def has(self, operation, trait):
    return bool(self.get(operation).traits & trait)

  def names(self, trait):
    return frozenset(name for name, definition in self.definitions.items() if definition.traits & trait)


_definitions = {}
_lock = threading.RLock()
_snapshot = None
_active_catalog = ContextVar("vecops_operation_catalog", default=None)


def current():
  return _active_catalog.get() or snapshot()


@contextmanager
def use_catalog(catalog):
  token = _active_catalog.set(catalog)
  try:
    yield
  finally:
    _active_catalog.reset(token)


def register_operation(definition):
  """Register a contract before compilation; duplicates are errors.

  Callback semantic changes require a version bump. Existing snapshots remain
  valid when plugins subsequently register or unregister operations.
  """
  if not isinstance(definition, OpDef) or not definition.name or not definition.family:
    raise ValueError("operation registration requires a named OpDef and family")
  if definition.effects not in {"none", "read", "write", "state", "unknown"}:
    raise ValueError("invalid operation effect")
  if definition.traits & Trait.CSE and definition.effects != "none":
    raise ValueError("effectful operations cannot promise unconditional CSE")
  global _snapshot
  with _lock:
    if definition.name in _definitions:
      raise ValueError(f"operation {definition.name!r} is already registered")
    _definitions[definition.name] = definition
    _snapshot = None
  return definition


def unregister_operation(name):
  global _snapshot
  with _lock:
    del _definitions[name]
    _snapshot = None


def snapshot():
  global _snapshot
  with _lock:
    if _snapshot is not None:
      return _snapshot
    definitions = dict(_definitions)
    identity = tuple(
      (
        name,
        definition.family,
        int(definition.traits),
        definition.effects,
        definition.arity,
        definition.keywords,
        definition.reduction,
        definition.version,
        getattr(definition.fold, "__qualname__", None),
      )
      for name, definition in sorted(definitions.items())
    )
    _snapshot = OperationCatalog(MappingProxyType(definitions), hashlib.sha256(repr(identity).encode()).hexdigest())
    return _snapshot
