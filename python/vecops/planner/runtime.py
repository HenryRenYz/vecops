# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Runtime binding of an offline plan to ``vecops.memory`` arenas.

The core planner remains framework-neutral. ``PlanSession`` is the optional
Torch adapter used to materialize planned tensor views and read-only model-state
replicas. Ordinary execution is unchanged when no session is active.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

from vecops._dtype import framework_dtype
from vecops.graph import ExecutionGraph, TensorKind, current_capture

from .ir import MemoryPlan, MemoryTier
from .validate import validate_plan

_active_session: contextvars.ContextVar[PlanSession | None] = contextvars.ContextVar(
  "vecops_memory_plan_session", default=None
)
_active_session_count = 0
_executable_factories = contextvars.ContextVar('vecops_executable_factories', default=False)
_torch_empty: Any | None = None
_torch_empty_like: Any | None = None


@contextlib.contextmanager
def executable_factories():
  """Record real factory nodes for an executable instead of eager arena views.

  Only affects ``empty``/``empty_like`` inside ``GraphCapture.capture_torch``.
  Metadata capture and ordinary PlanSession allocation remain unchanged. The
  executable owns allocation/liveness planning; source-session temporaries are
  not frozen into external snapshot bindings. Nested scopes are supported.
  """
  token = _executable_factories.set(True)
  try:
    yield
  finally:
    _executable_factories.reset(token)


def _load_torch_factories() -> None:
  """Resolve optional Torch factories once, outside later Dynamo traces."""
  global _torch_empty, _torch_empty_like
  torch = __import__("torch")
  _torch_empty = torch.empty
  _torch_empty_like = torch.empty_like


def current_session() -> "PlanSession | None":
  # Keep the overwhelmingly common no-session path visible as a plain global
  # guard.  Dynamo can constant-fold it without attempting to trace
  # ContextVar.get(), while real planner sessions retain context-local state.
  if _active_session_count == 0:
    return None
  return _active_session.get()


def _torch_dtype(name: str):
  # Memory plans may hold framework-only dtypes, not just kernel ABI types.
  return framework_dtype(name, "torch", allow_framework_only=True)


def _canonical_file(path: str) -> str:
  normalized = path.replace("\\", "/")
  for marker in ("/csrc/", "/src/", "/scripts/"):
    if marker in normalized:
      return marker[1:] + normalized.split(marker, 1)[1]
  return normalized


def _caller_source(depth: int = 1) -> tuple[str, str]:
  frame = sys._getframe(depth)
  return _canonical_file(frame.f_code.co_filename), frame.f_code.co_name


@dataclass(frozen=True)
class _FactoryCandidate:
  tensor_id: str
  shape: tuple[int, ...]
  strides: tuple[int, ...]
  dtype: str
  storage_bytes: int


class _RuntimeLoop:
  def __init__(self, session: "PlanSession", name: str, trip_count: int | str) -> None:
    self.session = session
    self.name = name
    self.closed = False
    candidates = session._captured_loops.get((tuple(session._scope), name), ())
    templates = [loop for loop in candidates
                 if not isinstance(loop.trip_count, int) or loop.trip_count == trip_count]
    if candidates and not templates:
      raise RuntimeError(f"runtime loop {name!r} trip count differs from its memory plan")
    self.observed = {sample.iteration for loop in templates for sample in loop.samples}
    self.representatives = {}
    for loop in templates:
      for ordinal, sample in enumerate(loop.samples):
        role = sample.role
        if role is None:
          role = "prologue" if ordinal == 0 else "steady_sample"
        self.representatives.setdefault(role, sample.iteration)
    from .core import _fully_observed
    self.fully_observed = bool(templates) and all(_fully_observed(loop) for loop in templates)

  @contextlib.contextmanager
  def iteration(self, index: int | str, role: str | None):
    if self.closed:
      raise RuntimeError("cannot enter an iteration of a closed runtime loop")
    if index in self.observed:
      sample = index
    elif self.fully_observed:
      raise RuntimeError(f"runtime loop {self.name!r} has an unrecorded iteration {index!r}")
    else:
      sample = self.representatives.get(role,
          self.representatives.get("steady_sample", 0 if role == "prologue" else 1))
    with self.session._loop_iteration(self.name, index, sample):
      yield

  def close(self) -> None:
    self.closed = True


class PlanSession:
  """Owner of physical fast/slow arenas for one memory plan.

  Static factory replay caches eligible Tensor views on entry. It may coexist
  with dynamic factories, whose storage-owned leases use the remaining budget
  for retained, escaping or otherwise unselected allocations. Strict replay
  validates selected sites; expected dynamic sites are not replay failures.
  Captures must describe persistent aliases and asynchronous lifetimes before
  a storage can safely be admitted as a synchronous temporary slot.
  The ordinary context manager closes on exit. ``activation()`` instead keeps
  the same arenas and model bindings across sequential synchronous invocations;
  its owner must call ``close()`` after releasing dependent executables.
  """

  def __init__(
    self,
    graph: ExecutionGraph,
    plan: MemoryPlan,
    *,
    system: Any | None = None,
    domain: int | None = None,
    allow_fast_fallback: bool = False,
    large_pages: bool = True,
    strict_factory_replay: bool = False,
    enable_factory_replay: bool = False,
    enable_dynamic_factories: bool = False,
    validate: bool = True,
  ) -> None:
    if validate:
      validate_plan(graph, plan)
    if system is None:
      from vecops import memory

      system = memory.default_system()
    from .buffers import current_session as buffer_session
    self._buffer_session = buffer_session()
    self.graph = graph
    self.plan = plan
    self.system = system
    self.domain = domain
    self._allocation_domain: int | None = None
    self.allow_fast_fallback = allow_fast_fallback
    self.large_pages = large_pages
    self.strict_factory_replay = strict_factory_replay
    self.enable_factory_replay = enable_factory_replay
    self.enable_dynamic_factories = enable_dynamic_factories
    self._dynamic_pool = None
    self.fast_owner = None
    self.slow_owner = None
    self._fast_owner_bytes = 0
    self._slow_owner_bytes = 0
    self._token: contextvars.Token | None = None
    self._entered = False
    self._closed = False
    self._original_data: dict[int, tuple[Any, Any]] = {}
    self._views: dict[str, Any] = {}
    self._arena_bases: dict[tuple[MemoryTier, str], Any] = {}
    self._copied_storages: set[str] = set()
    self._binding_stats = {
      "copied_bytes": 0,
      "copied_storages": 0,
      "bound_objects": 0,
    }
    self._known_factory_keys: set[tuple[Any, ...]] = set()
    self._factory_sites = self._build_factory_sites()
    self._factory_signatures = {key[:4] for key in self._factory_sites}
    self._factory_cursors: dict[tuple[tuple[Any, ...], tuple[int, ...]], int] = {}
    # Cache metadata, never Tensor objects or replay slots. Scope/source and
    # epoch cursors remain invocation-specific, including retained ordinals.
    self._factory_metadata_cache: dict[tuple[Any, ...], tuple[str, bool, tuple[int, ...] | None]] = {}
    self._factory_metadata_cache_hits = 0
    self._factory_metadata_cache_misses = 0
    self._factory_cursor_keys: list[list[tuple[Any, ...]]] = []
    self._scope: list[str] = []
    self._loop_epochs: list[int] = []
    self._iteration_aliases: list[tuple[str, str]] = []
    self._next_loop_epoch = 0
    self._captured_loops = {}
    for loop in graph.loops:
      self._captured_loops.setdefault((tuple(loop.scope), loop.name), []).append(loop)
    self._activation_lock = threading.Lock()
    self._persistent_ready = False
    self._planned_factory_allocations = 0
    self._factory_fallbacks = 0
    self._factory_fallback_reasons: Counter[str] = Counter()
    self._factory_fallback_samples: list[dict[str, Any]] = []
    self._factory_allocated_bytes = 0
    self._factory_large_page_bytes = 0
    self._factory_reserved_bytes = 0
    self._factory_owner_allocations = 0

  def _state_fast_arena_bytes(self) -> int:
    return max(
      (
        placement.offset + placement.bytes
        for placement in self.plan.placements.values()
        if placement.offset is not None
        and placement.annotations.get("eager_external_replica") is True
      ),
      default=0,
    )

  def __enter__(self) -> "PlanSession":
    global _active_session_count
    if self._entered or self._closed:
      raise RuntimeError("memory plan sessions are single-use")
    if current_session() is not None:
      raise RuntimeError("memory plan sessions cannot be nested")
    self._entered = True
    try:
      self._allocation_domain = (
        self.domain if self.domain is not None else self.system.current_cpu_domain()
      )
      allocated_fast_bytes = (
        self.plan.fast_arena_bytes
        if self.enable_factory_replay
        else self._state_fast_arena_bytes()
      )
      if allocated_fast_bytes:
        self.fast_owner = (self._buffer_session or self.system).allocate(
          allocated_fast_bytes,
          placement="high_bandwidth",
          domain=self.domain,
          fallback="default" if self.allow_fast_fallback else None,
          alignment=self.plan.machine.alignment,
          large_pages=self.large_pages,
        )
        self._fast_owner_bytes = allocated_fast_bytes
        if self.enable_factory_replay:
          self._factory_reserved_bytes = allocated_fast_bytes
          self._factory_owner_allocations = 1
          self._factory_large_page_bytes = getattr(
            self.fast_owner, "large_page_bytes", 0
          )
      if self.enable_dynamic_factories:
        from ._pool import TensorPool

        capacity = max(0, self.plan.machine.fast_capacity - allocated_fast_bytes)
        if self._buffer_session is not None:
          self._dynamic_pool = self._buffer_session.tensor_pool(capacity)
        elif capacity:
          owner = self.system.allocate(
            capacity, placement="high_bandwidth", domain=self.domain,
            fallback=None, alignment=self.plan.machine.alignment,
            large_pages=self.large_pages,
          )
          self._dynamic_pool = TensorPool(owner, capacity, self.plan.machine.alignment)
      if self.enable_factory_replay:
        for candidates in self._factory_sites.values():
          for candidate in candidates:
            if candidate is None:
              continue
            view = self.tensor_view(candidate.tensor_id)
            object.__setattr__(view, "_vecops_arena_view", True)
      self._token = _active_session.set(self)
      _active_session_count += 1
    except Exception:
      if self._token is not None:
        _active_session.reset(self._token)
        self._token = None
      self.fast_owner = None
      self.slow_owner = None
      self._dynamic_pool = None
      self._fast_owner_bytes = 0
      self._slow_owner_bytes = 0
      self._closed = True
      raise
    return self

  def __exit__(self, exc_type, exc, traceback) -> None:
    self.close()

  @contextlib.contextmanager
  def activation(self):
    """Activate a persistent arena without reallocating or rebinding state.

    Activations cannot overlap or nest. Views/leases and bound module data
    remain owned while inactive; only the context-local factory interception
    ends. Close explicitly after graphs, requests and borrowed outputs finish.
    Exceptions deactivate the session while preserving it for a later call.
    """
    global _active_session_count
    if not self._activation_lock.acquire(blocking=False):
      raise RuntimeError('memory plan activations cannot overlap or nest')
    try:
      if self._closed:
        raise RuntimeError('cannot activate a closed memory plan session')
      if (current_session() is not None or self._token is not None or
          (self._entered and not self._persistent_ready)):
        raise RuntimeError('memory plan activations cannot overlap or nest')
      if self._scope or self._loop_epochs or self._factory_cursor_keys or self._iteration_aliases:
        raise RuntimeError('previous memory plan activation left an unbalanced runtime scope')
      # Factory ordinals describe one invocation, not the arena's lifetime.
      # Keep owners, state bindings and metadata caches; restart root epochs.
      self._factory_cursors.clear()
      self._next_loop_epoch = 0
      if not self._entered:
        self.__enter__()
        self._persistent_ready = True
      else:
        self._token = _active_session.set(self)
        _active_session_count += 1
      try:
        yield self
      finally:
        _active_session.reset(self._token)
        self._token = None
        _active_session_count -= 1
    finally:
      self._activation_lock.release()

  def close(self) -> None:
    global _active_session_count
    if self._closed:
      return
    for _identity, (tensor, original) in reversed(tuple(self._original_data.items())):
      tensor.data = original
    self._original_data.clear()
    self._views.clear()
    self._arena_bases.clear()
    self._copied_storages.clear()
    self._factory_cursors.clear()
    self._factory_cursor_keys.clear()
    self._scope.clear()
    self._loop_epochs.clear()
    self._iteration_aliases.clear()
    if self._token is not None:
      _active_session.reset(self._token)
      self._token = None
      _active_session_count -= 1
    self.fast_owner = None
    self.slow_owner = None
    self._dynamic_pool = None
    self._fast_owner_bytes = 0
    self._slow_owner_bytes = 0
    self._closed = True

  def stats(self) -> dict[str, int | None]:
    """Return arena, binding, replay and metadata-cache telemetry."""
    return {
      "fast_arena_bytes": self.plan.fast_arena_bytes,
      "allocated_fast_arena_bytes": (
        self._fast_owner_bytes if self.fast_owner is not None else 0
      ),
      "slow_arena_bytes": self.plan.slow_arena_bytes,
      "allocated_slow_arena_bytes": (
        self._slow_owner_bytes if self.slow_owner is not None else 0
      ),
      "fast_target_id": getattr(self.fast_owner, "target_id", None),
      "fast_target_verified": self.fast_target_verified(),
      "slow_target_id": getattr(self.slow_owner, "target_id", None),
      "fast_large_page_bytes": getattr(self.fast_owner, "large_page_bytes", 0),
      "slow_large_page_bytes": getattr(self.slow_owner, "large_page_bytes", 0),
      "factory_metadata_cache_hits": self._factory_metadata_cache_hits,
      "factory_metadata_cache_misses": self._factory_metadata_cache_misses,
      "planned_factory_allocations": self._planned_factory_allocations,
      "factory_fallbacks": self._factory_fallbacks,
      "factory_allocated_bytes": self._factory_allocated_bytes,
      "factory_large_page_bytes": self._factory_large_page_bytes,
      "factory_reserved_bytes": self._factory_reserved_bytes,
      "factory_owner_allocations": self._factory_owner_allocations,
      "dynamic_reserved_bytes": self._dynamic_pool.capacity if self._dynamic_pool else 0,
      "dynamic_live_bytes": self._dynamic_pool.live_bytes if self._dynamic_pool else 0,
      "dynamic_peak_bytes": self._dynamic_pool.peak_bytes if self._dynamic_pool else 0,
      "dynamic_allocations": self._dynamic_pool.allocations if self._dynamic_pool else 0,
      "dynamic_misses": self._dynamic_pool.misses if self._dynamic_pool else 0,
      **self._binding_stats,
    }

  def fast_target_verified(self) -> bool | None:
    """Confirm that the arena landed on the topology's fastest target tier."""
    target = getattr(self.fast_owner, "target_id", None)
    if target is None or self._allocation_domain is None:
      return None
    tiers = self.system.tiers(self._allocation_domain, objective="bandwidth")
    if not tiers:
      return None
    return target in set(tiers[0].get("targets", ()))

  def diagnostics(self) -> dict[str, Any]:
    """Return bounded factory-matching diagnostics for profiling runs."""
    return {
      "fallback_reasons": dict(sorted(self._factory_fallback_reasons.items())),
      "fallback_samples": list(self._factory_fallback_samples),
    }

  def _factory_fallback(self, reason: str, key: Any) -> None:
    self._factory_fallbacks += 1
    self._factory_fallback_reasons[reason] += 1
    if len(self._factory_fallback_samples) < 12:
      self._factory_fallback_samples.append(
        {"reason": reason, "key": repr(key)}
      )

  def _build_factory_sites(self) -> dict[tuple[Any, ...], list[_FactoryCandidate | None]]:
    result: dict[tuple[Any, ...], list[_FactoryCandidate | None]] = {}
    factories = {"aten.empty.memory_format", "aten.empty_like.default"}
    # The same physical allocation can appear twice: once self-registered
    # by the planner factory (explicit, direct-caller source) and once
    # via the dispatch recorder (stack-walk source). The explicit event
    # owns the runtime key, so it must win the dedupe regardless of the
    # graph's event ordering.
    seen_tensors: set[tuple[tuple[Any, ...], str]] = set()

    def _factory_key(event, source, tensor_id):
      metadata = self.graph.tensors[tensor_id].metadata
      return (
          tuple(event.scope),
          _canonical_file(str(source["file"])),
          str(source["function"]),
          event.name,
          tuple(metadata.shape),
          metadata.dtype,
      )

    ordered = [e for e in self.graph.events
               if e.name in factories
               and isinstance(e.attributes.get("source"), dict)]
    ordered.sort(key=lambda e: 0 if e.attributes.get("explicit_factory") is True else 1)
    for event in ordered:
      source = event.attributes.get("source")
      if event.attributes.get("explicit_factory") is True:
        candidate_ids = list(event.outputs)
      else:
        candidate_ids = [
          tensor_id
          for tensor_id in event.outputs
          if self.graph.storages[self.graph.tensors[tensor_id].storage_id].allocation_event
          == event.id
        ]
      for tensor_id in candidate_ids:
        dedupe = (tuple(event.scope), tensor_id)
        if dedupe in seen_tensors:
          continue
        seen_tensors.add(dedupe)
        tensor = self.graph.tensors[tensor_id]
        storage = self.graph.storages[tensor.storage_id]
        key = _factory_key(event, source, tensor_id)
        self._known_factory_keys.add(key)
        # Preserve allocation order even when only some calls at one source
        # site are replayable. A retained tensor must not consume the next
        # temporary's static slot.
        slots = result.setdefault(key, [])
        slots.append(None)
        placement = self.plan.placements[tensor.storage_id]
        if placement.offset is None or placement.tier != MemoryTier.fast:
          continue
        if placement.annotations.get("loop_replay_safe") is False:
          continue
        if (tensor.metadata.requires_grad or storage.persistent
            or any(self.graph.tensors[alias].escapes
                   or self.graph.tensors[alias].persistent
                   for alias in storage.tensors)):
          continue
        slots[-1] = _FactoryCandidate(
            tensor_id, tuple(tensor.metadata.shape),
            tuple(tensor.metadata.strides), tensor.metadata.dtype, storage.bytes,
        )
    return {key: slots for key, slots in result.items() if any(slots)}

  def _factory_tensor(
    self,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    source: tuple[str, str],
  ):
    if not self.enable_factory_replay and not self.enable_dynamic_factories:
      return None
    if name not in {"aten.empty.memory_format", "aten.empty_like.default"}:
      return None
    cached_metadata = None
    if name == "aten.empty_like.default":
      prototype = args[0]
      shape = tuple(prototype.shape)
      dtype = kwargs.get("dtype") or prototype.dtype
      device = kwargs.get("device") or prototype.device
      if not kwargs:
        import torch

        # Subclasses, non-strided layouts, non-CPU devices and keyword
        # overrides retain the general path, including unplanned fallbacks.
        if (type(prototype) is torch.Tensor
            and prototype.layout == torch.strided and device.type == "cpu"):
          signature = (shape, tuple(prototype.stride()), dtype, device)
          cached_metadata = self._factory_metadata_cache.get(signature)
          if cached_metadata is None:
            self._factory_metadata_cache_misses += 1
            cached_metadata = (
                str(dtype), str(device).startswith("cpu"),
                self._factory_strides(name, args, kwargs, shape),
            )
            if len(self._factory_metadata_cache) < 256:
              self._factory_metadata_cache[signature] = cached_metadata
          else:
            self._factory_metadata_cache_hits += 1
    else:
      shape = (
        tuple(args[0])
        if len(args) == 1 and not isinstance(args[0], int)
        else tuple(args)
      )
      import torch

      dtype = kwargs.get("dtype") or torch.get_default_dtype()
      device = kwargs.get("device") or torch.device("cpu")
    if cached_metadata is None:
      dtype_name = str(dtype)
      is_cpu = str(device).startswith("cpu")
    else:
      dtype_name, is_cpu, _ = cached_metadata
    if not is_cpu:
      self._factory_fallback("non_cpu", (name, shape, dtype_name, str(device)))
      return None
    key = (
      tuple(self._scope), source[0], source[1], name, shape, dtype_name,
    )
    candidates = self._factory_sites.get(key) if self.enable_factory_replay else None
    if candidates:
      cursor_key = (key, tuple(self._loop_epochs))
      cursor = self._factory_cursors.get(cursor_key, 0)
      if cursor == 0 and self._factory_cursor_keys:
        self._factory_cursor_keys[-1].append(cursor_key)
      self._factory_cursors[cursor_key] = cursor + 1
      if cursor < len(candidates):
        candidate = candidates[cursor]
        if candidate is None:
          return (self._dynamic_factory_tensor(name, args, kwargs, shape, dtype)
                  if self.enable_dynamic_factories else None)
        import torch

        if cached_metadata is not None:
          # Only keyword-free requests can reach this branch. A hit does not
          # authorize a different candidate shape/layout or reuse its cursor.
          strides = cached_metadata[2]
          supported = strides is not None and strides == candidate.strides
        else:
          strides = self._factory_strides(name, args, kwargs, shape)
          supported = (
            strides is not None and strides == candidate.strides
            and not kwargs.get("requires_grad", False)
            and not kwargs.get("pin_memory", False)
            and kwargs.get("layout", torch.strided) == torch.strided
            and not (set(kwargs) - {"dtype", "device", "layout", "pin_memory",
                                    "requires_grad", "memory_format"})
          )
        if supported:
          self._planned_factory_allocations += 1
          self._factory_allocated_bytes += candidate.storage_bytes
          return self._views[candidate.tensor_id]
        reason = "metadata_mismatch"
      else:
        reason = "epoch_exhausted"
      self._factory_fallback(reason, key)
      if self.strict_factory_replay:
        raise RuntimeError(f"planned factory site {key!r}: {reason}")
    elif (self.enable_factory_replay and key not in self._known_factory_keys
          and key[:4] in self._factory_signatures):
      self._factory_fallback("metadata_mismatch", key)
      if self.strict_factory_replay:
        raise RuntimeError(f"planned factory site {key!r}: uncaptured metadata")
    elif not self.enable_dynamic_factories:
      self._factory_fallback("unplanned_site", key)
    if self.enable_dynamic_factories:
      return self._dynamic_factory_tensor(name, args, kwargs, shape, dtype)
    return None

  @staticmethod
  def _factory_strides(name, args, kwargs, shape):
    """Resolve empty_like's dense preserve-format contract for either arena."""
    import torch

    default = (torch.preserve_format if name == "aten.empty_like.default"
               else torch.contiguous_format)
    memory_format = kwargs.get("memory_format", default)
    if memory_format not in (torch.contiguous_format, torch.preserve_format):
      return None
    if name == "aten.empty.memory_format" and memory_format != torch.contiguous_format:
      return None
    if name == "aten.empty_like.default" and memory_format == torch.preserve_format:
      prototype = args[0]
      ordered = sorted((stride, dim) for dim, stride in zip(shape, prototype.stride())
                       if dim > 1)
      expected = 1
      for stride, dim in ordered:
        if stride != expected:
          break
        expected *= dim
      else:
        return tuple(prototype.stride())
    strides = []
    stride = 1
    for dim in reversed(shape):
      strides.append(stride)
      stride *= max(1, dim)
    return tuple(reversed(strides))

  def _dynamic_factory_tensor(self, name, args, kwargs, shape, dtype):
    """Lease unplanned, retained, and escaping tensors outside static slots."""
    import torch

    if (self._dynamic_pool is None or kwargs.get("out") is not None
        or kwargs.get("pin_memory", False)
        or kwargs.get("layout", torch.strided) != torch.strided
        or set(kwargs) - {"dtype", "device", "layout", "pin_memory",
                          "requires_grad", "memory_format"}):
      return None
    strides = self._factory_strides(name, args, kwargs, shape)
    if strides is None:
      return None
    replacement = self._dynamic_pool.empty(shape, dtype, strides)
    if replacement is not None and kwargs.get("requires_grad", False):
      replacement.requires_grad_(True)
    return replacement

  def enter_scope(self, name: str) -> None:
    """Mirror one framework scope, canonicalizing sampled loop indices."""
    normalized = str(name)
    if self._iteration_aliases and normalized == self._iteration_aliases[-1][0]:
      normalized = self._iteration_aliases[-1][1]
    self._scope.append(normalized)

  def leave_scope(self) -> None:
    if not self._scope:
      raise RuntimeError("runtime memory-plan scope stack underflow")
    self._scope.pop()

  def declare_loop(self, name: str, trip_count: int | str) -> _RuntimeLoop:
    return _RuntimeLoop(self, name, trip_count)

  @contextlib.contextmanager
  def _loop_iteration(self, name: str, index: int | str, sample: int | str):
    self._next_loop_epoch += 1
    self._loop_epochs.append(self._next_loop_epoch)
    self._factory_cursor_keys.append([])
    self._iteration_aliases.append((str(index), str(sample)))
    self._scope.append(f"{name}[{sample}]")
    try:
      yield
    finally:
      for key in self._factory_cursor_keys.pop():
        self._factory_cursors.pop(key, None)
      self._scope.pop()
      self._iteration_aliases.pop()
      self._loop_epochs.pop()

  def _owner(self, tier: MemoryTier, required_end: int):
    owner = self.fast_owner if tier == MemoryTier.fast else self.slow_owner
    if owner is None:
      size = (
        self.plan.fast_arena_bytes
        if tier == MemoryTier.fast
        else self.plan.slow_arena_bytes
      )
      if not size:
        raise RuntimeError(f"{tier.value} arena has zero planned size")
      owner = (self._buffer_session or self.system).allocate(
        size,
        placement="high_bandwidth" if tier == MemoryTier.fast else "default",
        domain=self.domain,
        fallback=(
          "default"
          if tier == MemoryTier.fast and self.allow_fast_fallback
          else None
        ),
        alignment=self.plan.machine.alignment,
        large_pages=self.large_pages,
      )
      if tier == MemoryTier.fast:
        self.fast_owner = owner
        self._fast_owner_bytes = size
      else:
        self.slow_owner = owner
        self._slow_owner_bytes = size
    allocated = (
      self._fast_owner_bytes if tier == MemoryTier.fast else self._slow_owner_bytes
    )
    if required_end > allocated:
      raise RuntimeError(
        f"{tier.value} arena view ends at {required_end}, beyond allocated {allocated}; "
        "materialize shared tensor views before binding eager model state"
      )
    return owner

  def tensor_view(self, tensor_id: str):
    """Create a Torch view over the planned physical slot for one tensor."""
    return self._arena_tensor(tensor_id, cache=True)

  def _arena_tensor(self, tensor_id: str, *, cache: bool = False):
    if not self._entered or self._closed:
      raise RuntimeError("enter the memory plan session before materializing tensors")
    if cache:
      cached = self._views.get(tensor_id)
      if cached is not None:
        return cached
    tensor = self.graph.tensors[tensor_id]
    storage = self.graph.storages[tensor.storage_id]
    placement = self.plan.placements[tensor.storage_id]
    if placement.offset is None:
      raise ValueError(f"storage {storage.id} is externally owned and has no arena slot")
    if tensor.metadata.layout not in {"strided", "torch.strided"}:
      raise ValueError(f"tensor {tensor_id} has unsupported layout {tensor.metadata.layout}")
    if not isinstance(storage.bytes, int):
      raise ValueError(f"tensor {tensor_id} has no concrete storage size")
    if not isinstance(tensor.metadata.storage_offset, int):
      raise ValueError(f"tensor {tensor_id} has a symbolic storage offset")
    if not all(isinstance(item, int) for item in tensor.metadata.shape):
      raise ValueError(f"tensor {tensor_id} has a symbolic shape")
    if not all(isinstance(item, int) for item in tensor.metadata.strides):
      raise ValueError(f"tensor {tensor_id} has symbolic strides")

    import torch

    dtype = _torch_dtype(tensor.metadata.dtype)
    element_size = tensor.metadata.element_size
    if element_size is None or storage.bytes % element_size:
      raise ValueError(f"storage {storage.id} is incompatible with tensor {tensor_id} dtype")
    owner = self._owner(placement.tier, placement.offset + storage.bytes)
    base_key = (placement.tier, tensor.metadata.dtype)
    base = self._arena_bases.get(base_key)
    if base is None:
      owner_bytes = (
        self._fast_owner_bytes
        if placement.tier == MemoryTier.fast
        else self._slow_owner_bytes
      )
      if owner_bytes % element_size or placement.offset % element_size:
        raise ValueError(
          f"arena or offset for {tensor_id} is not aligned to its element size"
        )
      base = torch.frombuffer(owner, dtype=dtype, count=owner_bytes // element_size)
      self._arena_bases[base_key] = base
    view = torch.as_strided(
      base,
      size=tuple(tensor.metadata.shape),
      stride=tuple(tensor.metadata.strides),
      storage_offset=placement.offset // element_size + tensor.metadata.storage_offset,
    )
    if cache:
      self._views[tensor_id] = view
    return view

  def bind_module_state(self, module: Any) -> dict[str, int]:
    """Copy selected read-only parameters/buffers to their fast planned slots."""
    if not self._entered or self._closed:
      raise RuntimeError("enter the memory plan session before binding module state")
    named = {
      **dict(module.named_parameters(remove_duplicate=False)),
      **dict(module.named_buffers(remove_duplicate=False)),
    }
    copied_bytes = 0
    copied_storages = 0
    bound_objects = 0
    expected_storages = {
      tensor_value.storage_id
      for tensor_value in self.graph.tensors.values()
      if tensor_value.kind in {TensorKind.parameter, TensorKind.buffer}
      and self.plan.placements[tensor_value.storage_id].tier == MemoryTier.fast
      and self.plan.placements[tensor_value.storage_id].replicated
      and self.plan.placements[tensor_value.storage_id].read_only
      and self.plan.placements[tensor_value.storage_id].annotations.get(
        "eager_external_replica"
      )
      is True
    }
    with __import__("torch").no_grad():
      for tensor_value in self.graph.tensors.values():
        if tensor_value.kind not in {TensorKind.parameter, TensorKind.buffer}:
          continue
        placement = self.plan.placements[tensor_value.storage_id]
        if not (
          placement.tier == MemoryTier.fast
          and placement.replicated
          and placement.read_only
          and placement.annotations.get("eager_external_replica") is True
        ):
          continue
        names = tensor_value.annotations.get("names", ())
        if tensor_value.name is not None and tensor_value.name not in names:
          names = [tensor_value.name, *names]
        source = next((named[name] for name in names if name in named), None)
        if source is None:
          continue
        identity = id(source)
        if identity in self._original_data:
          continue
        source_bytes = source.untyped_storage().nbytes()
        if source_bytes != placement.bytes:
          raise RuntimeError(
            f"module state {names!r} has {source_bytes} storage bytes, "
            f"but plan storage {tensor_value.storage_id} has {placement.bytes}"
          )
        destination = self.tensor_view(tensor_value.id)
        destination.copy_(source)
        if tensor_value.storage_id not in self._copied_storages:
          self._copied_storages.add(tensor_value.storage_id)
          copied_bytes += placement.bytes
          copied_storages += 1
        self._original_data[identity] = (source, source.data)
        source.data = destination
        bound_objects += 1
    missing = expected_storages - self._copied_storages
    if missing:
      raise RuntimeError(
        "memory plan could not bind selected module-state storages: "
        + ", ".join(sorted(missing))
      )
    self._binding_stats = {
      "copied_bytes": copied_bytes,
      "copied_storages": copied_storages,
      "bound_objects": bound_objects,
    }
    return dict(self._binding_stats)


def _record_capture_allocation(
    name: str,
    tensor: Any,
    source: tuple[str, str],
    *,
    capture: Any | None = None,
):
  """Register one explicit factory event on the active graph capture.

  Capture-time attribution otherwise walks the whole Python stack, which
  attributes wrapper allocations to unrelated frames (e.g. the matmul
  dispatch mode); the runtime factory key uses the direct caller of the
  factory instead. Self-registering with the identical source keeps both
  key constructions in lockstep, so planned factory sites exist for
  wrapper-level outputs.
  """
  if capture is None:
    capture = current_capture()
  if capture is None:
    return
  try:
    from ..graph.torch import register_tensor
  except ImportError:
    return
  try:
    register_tensor(capture, tensor, kind="temporary")
    file_name, function = source
    capture.record_operator(
        name,
        inputs=[],
        outputs=[tensor],
        attributes={
            "source": {
                "file": file_name,
                "function": function,
                "line": 0,
            },
            "explicit_factory": True,
        },
    )
  except Exception:  # diagnostics must never break allocation
    pass


def empty(*shape: Any, **kwargs: Any):
  """Drop-in ``torch.empty`` that uses a planned arena when one is active."""
  session = current_session()
  capture = current_capture()
  if (_executable_factories.get() and capture is not None
      and getattr(capture, '_torch_replay_tape', None) is not None):
    # Executable capture needs a real dispatcher factory node. Returning an
    # eager planner view here hides the allocation from the executable tape,
    # turning every temporary into a pinned external arena snapshot. Metadata
    # capture and normal eager planner placement retain their existing paths.
    if _torch_empty is None:
      _load_torch_factories()
    return _torch_empty(*shape, **kwargs)
  if (session is None or not (session.enable_factory_replay or session.enable_dynamic_factories)) and capture is None:
    if _torch_empty is None:
      _load_torch_factories()
    return _torch_empty(*shape, **kwargs)

  source = _caller_source(2)
  if session is not None:
    replacement = session._factory_tensor(
      "aten.empty.memory_format",
      shape,
      kwargs,
      source=source,
    )
    if replacement is not None:
      return replacement
  if _torch_empty is None:
    _load_torch_factories()
  result = _torch_empty(*shape, **kwargs)
  if str(kwargs.get("device") or "cpu").startswith("cpu"):
    _record_capture_allocation(
        "aten.empty.memory_format", result, source, capture=capture
    )
  return result


def empty_like(prototype: Any, **kwargs: Any):
  """Drop-in ``torch.empty_like`` that uses a planned arena when active."""
  session = current_session()
  capture = current_capture()
  if (_executable_factories.get() and capture is not None
      and getattr(capture, '_torch_replay_tape', None) is not None):
    if _torch_empty_like is None:
      _load_torch_factories()
    return _torch_empty_like(prototype, **kwargs)
  if (session is None or not (session.enable_factory_replay or session.enable_dynamic_factories)) and capture is None:
    if _torch_empty_like is None:
      _load_torch_factories()
    return _torch_empty_like(prototype, **kwargs)

  source = _caller_source(2)
  if session is not None:
    replacement = session._factory_tensor(
      "aten.empty_like.default",
      (prototype,),
      kwargs,
      source=source,
    )
    if replacement is not None:
      return replacement
  if _torch_empty_like is None:
    _load_torch_factories()
  result = _torch_empty_like(prototype, **kwargs)
  if str(kwargs.get("device") or prototype.device).startswith("cpu"):
    _record_capture_allocation(
        "aten.empty_like.default", result, source, capture=capture
    )
  return result


def contiguous(tensor):
  """Materialize a contiguous tensor in the active dynamic fast pool if set."""
  if tensor.is_contiguous():
    return tensor
  session = current_session()
  if session is None or not session.enable_dynamic_factories:
    return tensor.contiguous()
  result = empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
  result.copy_(tensor)
  return result


__all__ = ["PlanSession", "current_session", "empty", "empty_like", "contiguous"]
