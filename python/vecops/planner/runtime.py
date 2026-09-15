"""Runtime binding of an offline plan to ``vecops.memory`` arenas.

The core planner remains framework-neutral. ``PlanSession`` is the optional
Torch adapter used to materialize planned tensor views and read-only model-state
replicas. Ordinary execution is unchanged when no session is active.
"""

from __future__ import annotations

import contextlib
import contextvars
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

from vecops.graph import ExecutionGraph, TensorKind

from .ir import MemoryPlan, MemoryTier
from .validate import validate_plan


_active_session: contextvars.ContextVar[PlanSession | None] = contextvars.ContextVar(
  "vecops_memory_plan_session", default=None
)
_active_session_count = 0
_torch_empty: Any | None = None
_torch_empty_like: Any | None = None


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
  import torch

  prefix = "torch."
  value = name[len(prefix) :] if name.startswith(prefix) else name
  dtype = getattr(torch, value, None)
  if not isinstance(dtype, torch.dtype):
    raise TypeError(f"unsupported Torch dtype in memory plan: {name!r}")
  return dtype


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


class _RuntimeLoop:
  def __init__(self, session: "PlanSession", name: str) -> None:
    self.session = session
    self.name = name
    self.closed = False

  @contextlib.contextmanager
  def iteration(self, index: int | str, role: str | None):
    if self.closed:
      raise RuntimeError("cannot enter an iteration of a closed runtime loop")
    sample = 0 if role == "prologue" else 1
    with self.session._loop_iteration(self.name, index, sample):
      yield

  def close(self) -> None:
    self.closed = True


class PlanSession:
  """Single-use owner of physical fast/slow arenas for one memory plan."""

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
    validate: bool = True,
  ) -> None:
    if validate:
      validate_plan(graph, plan)
    if system is None:
      from vecops import memory

      system = memory.default_system()
    self.graph = graph
    self.plan = plan
    self.system = system
    self.domain = domain
    self._allocation_domain: int | None = None
    self.allow_fast_fallback = allow_fast_fallback
    self.large_pages = large_pages
    self.strict_factory_replay = strict_factory_replay
    self.enable_factory_replay = enable_factory_replay
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
    self._factory_sites = self._build_factory_sites()
    self._factory_cursors: dict[tuple[tuple[Any, ...], tuple[int, ...]], int] = {}
    self._scope: list[str] = []
    self._loop_epochs: list[int] = []
    self._iteration_aliases: list[tuple[str, str]] = []
    self._next_loop_epoch = 0
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
        self.fast_owner = self.system.allocate(
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
      self._token = _active_session.set(self)
      _active_session_count += 1
    except Exception:
      if self._token is not None:
        _active_session.reset(self._token)
        self._token = None
      self.fast_owner = None
      self.slow_owner = None
      self._fast_owner_bytes = 0
      self._slow_owner_bytes = 0
      self._closed = True
      raise
    return self

  def __exit__(self, exc_type, exc, traceback) -> None:
    self.close()

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
    self._scope.clear()
    self._loop_epochs.clear()
    self._iteration_aliases.clear()
    if self._token is not None:
      _active_session.reset(self._token)
      self._token = None
      _active_session_count -= 1
    self.fast_owner = None
    self.slow_owner = None
    self._fast_owner_bytes = 0
    self._slow_owner_bytes = 0
    self._closed = True

  def stats(self) -> dict[str, int | None]:
    """Return arena backing and model-state binding telemetry."""
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
      "planned_factory_allocations": self._planned_factory_allocations,
      "factory_fallbacks": self._factory_fallbacks,
      "factory_allocated_bytes": self._factory_allocated_bytes,
      "factory_large_page_bytes": self._factory_large_page_bytes,
      "factory_reserved_bytes": self._factory_reserved_bytes,
      "factory_owner_allocations": self._factory_owner_allocations,
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

  def _build_factory_sites(self) -> dict[tuple[Any, ...], list[_FactoryCandidate]]:
    result: dict[tuple[Any, ...], list[_FactoryCandidate]] = {}
    factories = {"aten.empty.memory_format", "aten.empty_like.default"}
    seen_tensors: set[tuple[tuple[Any, ...], str]] = set()
    for event in self.graph.events:
      source = event.attributes.get("source")
      if event.name not in factories or not isinstance(source, dict):
        continue
      if event.attributes.get("explicit_factory") is True:
        # Self-registered by the planner factory: the output is the
        # allocated tensor even when stack-walk attribution (and the
        # storage allocation_event with it) went to a different event.
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
        placement = self.plan.placements[self.graph.tensors[tensor_id].storage_id]
        if placement.offset is None or placement.tier != MemoryTier.fast:
          continue
        if placement.annotations.get("loop_replay_safe") is False:
          continue
        metadata = self.graph.tensors[tensor_id].metadata
        key = (
          tuple(event.scope),
          _canonical_file(str(source["file"])),
          str(source["function"]),
          event.name,
          tuple(metadata.shape),
          metadata.dtype,
        )
        result.setdefault(key, []).append(_FactoryCandidate(tensor_id))
    return result

  def _factory_tensor(
    self,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    source: tuple[str, str],
  ):
    if not self.enable_factory_replay:
      return None
    if name not in {"aten.empty.memory_format", "aten.empty_like.default"}:
      return None
    if name == "aten.empty_like.default":
      prototype = args[0]
      shape = tuple(prototype.shape)
      dtype = kwargs.get("dtype") or prototype.dtype
      device = kwargs.get("device") or prototype.device
    else:
      shape = (
        tuple(args[0])
        if len(args) == 1 and not isinstance(args[0], int)
        else tuple(args)
      )
      import torch

      dtype = kwargs.get("dtype") or torch.get_default_dtype()
      device = kwargs.get("device") or torch.device("cpu")
    if not str(device).startswith("cpu"):
      self._factory_fallback("non_cpu", (name, shape, str(dtype), str(device)))
      return None
    key = (
      tuple(self._scope),
      source[0],
      source[1],
      name,
      shape,
      str(dtype),
    )
    candidates = self._factory_sites.get(key)
    if not candidates:
      self._factory_fallback("unplanned_site", key)
      return None
    cursor_key = (key, tuple(self._loop_epochs))
    cursor = self._factory_cursors.get(cursor_key, 0)
    if cursor < len(candidates):
      candidate = candidates[cursor]
      self._factory_cursors[cursor_key] = cursor + 1
    else:
      self._factory_fallback("epoch_exhausted", key)
      if self.strict_factory_replay:
        raise RuntimeError(
          f"planned factory site {key!r} executed too many times in one loop epoch"
        )
      return None

    expected = self.graph.tensors[candidate.tensor_id].metadata
    compatible = (
      tuple(expected.shape) == shape
      and expected.dtype == str(dtype)
      and str(device).startswith("cpu")
    )
    if not compatible:
      self._factory_fallback("metadata_mismatch", key)
      if self.strict_factory_replay:
        raise RuntimeError(
          f"planned factory site {key!r} expected shape={expected.shape}, "
          f"dtype={expected.dtype}; got shape={shape}, dtype={dtype}, device={device}"
        )
      return None
    self._planned_factory_allocations += 1
    storage = self.graph.storages[self.graph.tensors[candidate.tensor_id].storage_id]
    self._factory_allocated_bytes += storage.bytes
    # The plan proves that this physical slot is dead before its next replay.
    # Reusing the prebuilt Tensor view removes both allocator and view-building
    # overhead from the hot path.
    return self.tensor_view(candidate.tensor_id)

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
    del trip_count
    return _RuntimeLoop(self, name)

  @contextlib.contextmanager
  def _loop_iteration(self, name: str, index: int | str, sample: int):
    self._next_loop_epoch += 1
    self._loop_epochs.append(self._next_loop_epoch)
    self._iteration_aliases.append((str(index), str(sample)))
    self._scope.append(f"{name}[{sample}]")
    try:
      yield
    finally:
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
      owner = self.system.allocate(
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


def _record_capture_allocation(name: str, tensor: Any, source: tuple[str, str]):
  """Register one explicit factory event on the active graph capture.

  Capture-time attribution otherwise walks the whole Python stack, which
  attributes wrapper allocations to unrelated frames (e.g. the matmul
  dispatch mode); the runtime factory key uses the direct caller of the
  factory instead. Self-registering with the identical source keeps both
  key constructions in lockstep, so planned factory sites exist for
  wrapper-level outputs.
  """
  try:
    from ..graph import current_capture
    from ..graph.torch import register_tensor
  except ImportError:
    return
  capture = current_capture()
  if capture is None:
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
  source = _caller_source(2)
  session = current_session()
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
    _record_capture_allocation("aten.empty.memory_format", result, source)
  return result


def empty_like(prototype: Any, **kwargs: Any):
  """Drop-in ``torch.empty_like`` that uses a planned arena when active."""
  source = _caller_source(2)
  session = current_session()
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
    _record_capture_allocation("aten.empty_like.default", result, source)
  return result


__all__ = ["PlanSession", "current_session", "empty", "empty_like"]
