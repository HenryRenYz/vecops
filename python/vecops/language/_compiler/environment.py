"""One immutable semantic/capability/cost snapshot for a compilation."""

import hashlib
import threading
from dataclasses import dataclass, replace
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
  vector_state: object = None
  # Actual code-generation target. Cost-profile overrides must never select
  # which architecture's instructions are emitted or remove a legal path.
  codegen_target: object = CPUTarget()

  @cached_property
  def fingerprint(self):
    identity = (
      self.operations.fingerprint,
      self.implementations.fingerprint,
      self.costs.fingerprint,
      self.target,
      self.vector_state,
      self.codegen_target,
    )
    return hashlib.sha256(repr(identity).encode()).hexdigest()


_cache = WeakValueDictionary()
_lock = threading.RLock()


def freeze(options=None, *, target="generic"):
  options = options or {}
  previous = options.get("_compiler_environment")
  if previous is not None:
    return previous
  codegen_target = target if isinstance(target, CPUTarget) else resolve_target(target)
  selected = options.get("cpu_cost_target", target)
  selected = (
    selected
    if isinstance(selected, CPUTarget)
    else resolve_target(selected, processor=options.get("cpu_cost_processor"))
  )
  from ..._arm_vector_state import VectorState, get_vector_state

  vector_state = options.get("_vector_state")
  if vector_state is None:
    # Dry/cross lowering must not infer the target's vector geometry from
    # the build host. The compile driver supplies its explicit snapshot;
    # only a native standalone lowering may query the calling thread.
    native = isinstance(target, str) and target.lower().startswith("native")
    vector_state = get_vector_state() if native else VectorState(None, None)
  if selected.architecture == "arm" and vector_state.sve_bytes is not None:
    # Cached CPU detection describes the ISA, not a thread's mutable VL.
    # Ordinary vector cost/planning uses VL; SME panel sizing uses SVL itself.
    selected = replace(selected, vector_bits=vector_state.sve_bytes * 8)
  overrides = options.get("cost_profiles", ())
  overrides = (overrides,) if isinstance(overrides, str) else tuple(overrides)
  operations = snapshot()
  implementations = implementation_snapshot()
  costs = build_model(operations, selected, overrides)
  key = (operations.fingerprint, implementations.fingerprint, costs.profiles, selected, vector_state, codegen_target)
  with _lock:
    result = _cache.get(key)
    if result is None:
      result = CompilerEnvironment(operations, implementations, costs, selected, vector_state, codegen_target)
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
