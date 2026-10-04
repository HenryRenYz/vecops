"""One immutable semantic/capability/cost snapshot for a compilation."""

import hashlib
import threading
from dataclasses import dataclass
from functools import cached_property
from weakref import WeakValueDictionary

from .backends.cpu.cost import CPUTarget, build_model, resolve_target
from .backends.cpu.implementations import snapshot as implementation_snapshot
from .cost.profiles import fingerprint as profile_fingerprint
from .operations import snapshot


@dataclass(frozen=True)
class CompilerEnvironment:
  operations: object
  implementations: object
  costs: object
  target: object

  @cached_property
  def fingerprint(self):
    identity = (self.operations.fingerprint, self.implementations.fingerprint, self.costs.fingerprint, self.target)
    return hashlib.sha256(repr(identity).encode()).hexdigest()


_cache = WeakValueDictionary()
_lock = threading.RLock()


def freeze(options=None, *, target="generic"):
  options = options or {}
  previous = options.get("_compiler_environment")
  if previous is not None:
    return previous
  selected = options.get("cpu_cost_target", target)
  selected = (
    selected
    if isinstance(selected, CPUTarget)
    else resolve_target(selected, processor=options.get("cpu_cost_processor"))
  )
  overrides = options.get("cost_profiles", ())
  overrides = (overrides,) if isinstance(overrides, str) else tuple(overrides)
  operations = snapshot()
  implementations = implementation_snapshot()
  costs = build_model(operations, selected, overrides)
  key = (operations.fingerprint, implementations.fingerprint, costs.profiles, selected)
  with _lock:
    result = _cache.get(key)
    if result is None:
      result = CompilerEnvironment(operations, implementations, costs, selected)
      _cache[key] = result
    return result


def registry_fingerprint():
  """Include every profile: an outer factory may return a different target."""
  from ..._identity import callable_identity
  from .passes.manager import registered_passes
  from .pipeline import load_builtin_passes

  # The first factory lookup must see the same registry as its first lowering.
  # Otherwise lazy builtin registration invalidates an untouched factory once.
  load_builtin_passes()

  callbacks = tuple((p.name, p.stage, p.before, p.after, callable_identity(p.callback)) for p in registered_passes())
  folds = tuple((op.name, callable_identity(op.fold)) for op in snapshot().definitions.values() if op.fold is not None)
  return (snapshot().fingerprint, implementation_snapshot().fingerprint, profile_fingerprint(), callbacks, folds)
