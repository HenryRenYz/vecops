# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Framework-neutral graph construction, scopes, loops, and annotations."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import weakref
from dataclasses import dataclass
from typing import Any, Iterable

from .ir import (
  AccessKind,
  Event,
  ExecutionGraph,
  Loop,
  LoopSample,
  StorageValue,
  TensorKind,
  TensorMetadata,
  TensorUse,
  TensorValue,
)


_active_capture: contextvars.ContextVar[GraphCapture | None] = contextvars.ContextVar(
  "vecops_graph_capture", default=None
)
_active_capture_count = 0


def _normalize_kind(value: TensorKind | str) -> TensorKind:
  return value if isinstance(value, TensorKind) else TensorKind(value)


def _normalize_access(value: AccessKind | str) -> AccessKind:
  return value if isinstance(value, AccessKind) else AccessKind(value)


def current_capture() -> GraphCapture | None:
  """Return the active capture in this context, or ``None``."""
  # Keep the normal execution path visible to Dynamo as a plain global guard.
  # ContextVar.get() is neither traceable nor free, while graph capture is a
  # short-lived precompile-only state.
  if _active_capture_count == 0:
    return None
  return _active_capture.get()


class GraphCapture:
  """Incrementally construct an :class:`ExecutionGraph`.

  Framework adapters provide opaque object and storage identities.  The
  builder retains weak references to live tensor objects, so tracing a real
  model does not extend activation lifetimes or consume additional model-sized
  memory.
  """

  def __init__(self, name: str = "execution", *, attributes: dict[str, Any] | None = None) -> None:
    if not name:
      raise ValueError("execution graph name must not be empty")
    self.name = name
    self.attributes = dict(attributes or {})
    self.events: list[Event] = []
    self.tensors: dict[str, TensorValue] = {}
    self.storages: dict[str, StorageValue] = {}
    self.loops: list[Loop] = []
    self.inputs: list[str] = []
    self.outputs: list[str] = []
    self._objects: dict[int, tuple[weakref.ReferenceType[Any] | None, str]] = {}
    self._strong_objects: dict[int, Any] = {}
    self._retained_objects: list[Any] = []
    self._storage_keys: dict[Any, str] = {}
    self._storage_live_objects: dict[str, int] = {}
    self._storage_versions: dict[str, int] = {}
    self._scope: list[str] = []
    self._event_counter = 0
    self._tensor_counter = 0
    self._storage_counter = 0
    self._loop_counter = 0
    self._active_token: contextvars.Token | None = None
    self._finished = False

  def __enter__(self) -> "GraphCapture":
    global _active_capture_count
    if self._finished:
      raise RuntimeError("a finished graph capture cannot be re-entered")
    if self._active_token is not None:
      raise RuntimeError("graph capture is already active")
    if current_capture() is not None:
      raise RuntimeError("graph captures cannot be nested")
    self._active_token = _active_capture.set(self)
    _active_capture_count += 1
    return self

  def __exit__(self, exc_type, exc, traceback) -> None:
    global _active_capture_count
    assert self._active_token is not None
    _active_capture.reset(self._active_token)
    self._active_token = None
    _active_capture_count -= 1

  @property
  def scope(self) -> tuple[str, ...]:
    return tuple(self._scope)

  def capture_torch(self, function, *args, preserve_rng_state: bool = True, **kwargs):
    """Execute once and capture a static CPU program for native replay.

    Unlike precompile's metadata-only capture, this runs real kernels and may
    mutate inputs/module state. The default CPU RNG state is restored by default
    on success and failure. Python side effects/control flow are not replayed;
    only the observed static tensor program is compiled. Use fresh captures for
    different shapes, layouts or non-Tensor arguments. Existing manual and
    precompile captures remain metadata-only and do not retain activations.
    """
    from ._replay import capture_torch
    return capture_torch(self, function, args, kwargs,
                         preserve_rng_state=preserve_rng_state)

  def compile_torch(self, *, memory_plan=None, system=None, domain=None,
                    large_pages: bool = True, native_operators=None,
                    borrow_readonly_state: bool = False, native_pointwise: bool = False,
                    scratch_allocator=None, borrow_outputs: bool = False, native_donation: bool = False):
    """Lower this capture to a guarded, synchronous C++ CPU interpreter.

    Call after ``capture_torch``. Unsupported operators/registrations fail
    explicitly, never falling back to a Python loop. ``native_operators`` maps
    qualified custom operator names to strong C++ registration/module owners;
    materialized vecops Torch registrations are discovered automatically.
    Optional memory plans must describe this exact graph and select internal
    nonescaping factory allocations. Other fast placements are rejected;
    ordinary functional ATen results keep using the native allocator and their
    unapplied slow placements are reported on the executable. By default,
    borrowed planner arenas are isolated. ``borrow_readonly_state=True`` may
    retain verified, dedicated, already-bound immutable module-state ranges;
    their contents must remain unchanged for this executable's lifetime.
    ``native_pointwise=True`` lowers supported floating ATen intermediates to
    recordable SIMD grids and owns their nonescaping scratch buffers. Escaping
    functional outputs retain their ordinary allocation/alias semantics.
    ``scratch_allocator(bytes)`` may provide a preallocated CPU uint8 arena.
    It runs only at compile time and must return enough contiguous storage.
    Sharing an arena across executables requires externally serialized calls;
    only nonescaping temporaries are placed there, never outputs or state.
    ``borrow_outputs=True`` also preallocates supported escaping factories and
    pointwise outputs in a separate private arena. Those returned tensors may
    be overwritten on the next call; clone values that must outlive a replay.
    Unsupported functional ATen results still use their native allocator.
    ``native_donation=True`` permits audited same-index pointwise grids to
    reuse a dead, internal, nonescaping temporary. Alias-group liveness is
    checked and extended through the new result. Inputs, state, broadcast
    aliases and storage owned by other planners cannot be donated. This cold
    optimization is opt-in and does not change eager operation schemas.
    """
    from ._replay import compile_torch
    return compile_torch(self, memory_plan=memory_plan, system=system, domain=domain,
                         large_pages=large_pages, native_operators=native_operators,
                         borrow_readonly_state=borrow_readonly_state,
                         native_pointwise=native_pointwise, scratch_allocator=scratch_allocator,
                         borrow_outputs=borrow_outputs, native_donation=native_donation)

  def _next_event_id(self) -> tuple[str, int]:
    self._event_counter += 1
    return f"e{self._event_counter}", self._event_counter

  def _append_event(
    self,
    kind: str,
    name: str,
    *,
    uses: list[TensorUse] | None = None,
    outputs: list[str] | None = None,
    attributes: dict[str, Any] | None = None,
  ) -> Event:
    event_id, index = self._next_event_id()
    event = Event(
      event_id,
      index,
      kind,
      name,
      self.scope,
      list(uses or ()),
      list(outputs or ()),
      dict(attributes or {}),
    )
    self.events.append(event)
    return event

  def _object_entry(self, value: Any) -> str | None:
    entry = self._objects.get(id(value))
    if entry is None:
      return None
    reference, tensor_id = entry
    if reference is None:
      return tensor_id if self._strong_objects.get(id(value)) is value else None
    return tensor_id if reference() is value else None

  def _remember_object(self, value: Any, tensor_id: str) -> None:
    identity = id(value)
    existing = self._objects.get(identity)
    if existing is not None:
      reference, existing_tensor_id = existing
      if existing_tensor_id == tensor_id and (
        (reference is None and self._strong_objects.get(identity) is value)
        or (reference is not None and reference() is value)
      ):
        return
    storage_id = self.tensors[tensor_id].storage_id
    owner = weakref.ref(self)

    def release(reference: weakref.ReferenceType[Any]) -> None:
      capture = owner()
      if capture is None:
        return
      current = capture._objects.get(identity)
      if current is None or current[0] is not reference:
        return
      capture._objects.pop(identity, None)
      capture._storage_live_objects[storage_id] -= 1

    try:
      reference = weakref.ref(value, release)
    except TypeError:
      # Framework tensor objects are normally weak-referenceable.  This
      # fallback keeps annotation adapters for simpler foreign objects usable.
      reference = None
      self._strong_objects[identity] = value
    self._objects[identity] = (reference, tensor_id)
    self._storage_live_objects[storage_id] += 1

  def tensor(
    self,
    value: Any,
    metadata: TensorMetadata,
    *,
    storage_identity: Any | None = None,
    storage_bytes: int | str | None = None,
    name: str | None = None,
    kind: TensorKind | str = TensorKind.unknown,
    persistent: bool = False,
    read_only: bool = False,
    external: bool = False,
    annotations: dict[str, Any] | None = None,
  ) -> str:
    """Register or update one framework tensor object."""
    existing = self._object_entry(value)
    if existing is not None:
      tensor = self.tensors[existing]
      if name is not None:
        names = tensor.annotations.setdefault("names", [])
        if tensor.name is None:
          tensor.name = name
        if name not in names:
          names.append(name)
      normalized_kind = _normalize_kind(kind)
      if tensor.kind in {TensorKind.unknown, TensorKind.external} or normalized_kind not in {
        TensorKind.unknown,
        TensorKind.external,
      }:
        tensor.kind = normalized_kind
      tensor.persistent |= persistent
      tensor.read_only |= read_only
      tensor.annotations.update(annotations or {})
      storage = self.storages[tensor.storage_id]
      storage.persistent |= persistent
      storage.external |= external
      return existing

    self._tensor_counter += 1
    tensor_id = f"t{self._tensor_counter}"
    known_storage = (
      self._storage_keys.get(storage_identity)
      if storage_identity is not None
      else None
    )
    if known_storage is not None and self._storage_live_objects[known_storage] > 0:
      storage_id = known_storage
      alias_of = self.storages[storage_id].tensors[0]
      alias_kind = "storage"
    else:
      self._storage_counter += 1
      storage_id = f"s{self._storage_counter}"
      alias_of = None
      alias_kind = None
      if storage_identity is not None:
        self._storage_keys[storage_identity] = storage_id
      self.storages[storage_id] = StorageValue(
        storage_id,
        storage_bytes,
        external=external,
        persistent=persistent,
      )
      self._storage_live_objects[storage_id] = 0
      self._storage_versions[storage_id] = 0

    tensor_annotations = dict(annotations or {})
    if name is not None:
      tensor_annotations.setdefault("names", [name])
    self.tensors[tensor_id] = TensorValue(
      id=tensor_id,
      metadata=metadata,
      storage_id=storage_id,
      name=name,
      kind=_normalize_kind(kind),
      alias_of=alias_of,
      alias_kind=alias_kind,
      persistent=persistent,
      read_only=read_only,
      annotations=tensor_annotations,
    )
    self.storages[storage_id].tensors.append(tensor_id)
    self._remember_object(value, tensor_id)
    return tensor_id

  def tensor_id(self, value: Any) -> str:
    tensor_id = self._object_entry(value)
    if tensor_id is None:
      raise KeyError("tensor object is not registered with the active capture")
    return tensor_id

  def retain(self, value: Any) -> None:
    """Keep a metadata/proxy object alive for adapter identity stability."""
    self._retained_objects.append(value)

  def bind_object(
    self, value: Any, tensor_id: str, *, storage_identity: Any | None = None
  ) -> None:
    """Bind an adapter proxy object to an existing logical tensor identity."""
    if tensor_id not in self.tensors:
      raise KeyError(f"cannot bind an unknown tensor ID {tensor_id}")
    self._remember_object(value, tensor_id)
    self.retain(value)
    if storage_identity is not None:
      self._storage_keys[storage_identity] = self.tensors[tensor_id].storage_id

  def mark_input(self, tensor_id: str) -> None:
    tensor = self.tensors[tensor_id]
    tensor.kind = TensorKind.input
    self.storages[tensor.storage_id].external = True
    # Preserve formal boundary arity: the same Tensor may legally be supplied
    # through multiple input slots.
    self.inputs.append(tensor_id)

  def mark_outputs(self, values: Iterable[Any]) -> None:
    for value in values:
      tensor_id = self._object_entry(value)
      if tensor_id is None:
        continue
      tensor = self.tensors[tensor_id]
      tensor.escapes = True
      if tensor.kind in {TensorKind.unknown, TensorKind.temporary}:
        tensor.kind = TensorKind.output
      # Preserve output pytree leaf order and repeated aliases.
      self.outputs.append(tensor_id)

  def mark_alias(self, alias: Any, base: Any, *, kind: str = "explicit") -> None:
    """Declare that two already registered tensor values share storage."""
    alias_id = self.tensor_id(alias)
    base_id = self.tensor_id(base)
    if alias_id == base_id:
      return
    alias_tensor = self.tensors[alias_id]
    base_tensor = self.tensors[base_id]
    if alias_tensor.storage_id == base_tensor.storage_id:
      alias_tensor.alias_of = base_id
      alias_tensor.alias_kind = kind
      return

    old_storage = self.storages[alias_tensor.storage_id]
    live_alias_objects = sum(
      1
      for reference, tensor_id in self._objects.values()
      if tensor_id == alias_id and (reference is None or reference() is not None)
    )
    self._storage_live_objects[old_storage.id] -= live_alias_objects
    self._storage_live_objects[base_tensor.storage_id] += live_alias_objects
    old_storage.tensors.remove(alias_id)
    if not old_storage.tensors:
      del self._storage_live_objects[old_storage.id]
      del self._storage_versions[old_storage.id]
      del self.storages[old_storage.id]
      for key, storage_id in list(self._storage_keys.items()):
        if storage_id == old_storage.id:
          del self._storage_keys[key]
    alias_tensor.storage_id = base_tensor.storage_id
    alias_tensor.alias_of = base_id
    alias_tensor.alias_kind = kind
    self.storages[base_tensor.storage_id].tensors.append(alias_id)

  def annotate_tensor(
    self,
    value: Any,
    *,
    name: str | None = None,
    kind: TensorKind | str | None = None,
    persistent: bool | None = None,
    read_only: bool | None = None,
    annotations: dict[str, Any] | None = None,
  ) -> None:
    tensor = self.tensors[self.tensor_id(value)]
    if name is not None:
      tensor.name = name
      names = tensor.annotations.setdefault("names", [])
      if name not in names:
        names.append(name)
    if kind is not None:
      tensor.kind = _normalize_kind(kind)
    if persistent is not None:
      tensor.persistent = persistent
      self.storages[tensor.storage_id].persistent |= persistent
    if read_only is not None:
      tensor.read_only = read_only
    tensor.annotations.update(annotations or {})

  def mark_retained(self, values: Any, *, reason: str | None = None) -> None:
    """Extend registered values to graph exit because Python state retains them."""
    for value in _tensor_leaves(values, self):
      tensor = self.tensors[self.tensor_id(value)]
      tensor.escapes = True
      tensor.annotations["retained"] = reason or True

  def record_operator(
    self,
    name: str,
    *,
    inputs: Iterable[tuple[Any, AccessKind | str, str | None]],
    outputs: Iterable[Any],
    attributes: dict[str, Any] | None = None,
  ) -> Event:
    """Record one operator and update storage versions and producers."""
    normalized_inputs = [
      (self.tensor_id(value), _normalize_access(access), argument)
      for value, access, argument in inputs
    ]
    written_storages = {
      self.tensors[tensor_id].storage_id
      for tensor_id, access, _argument in normalized_inputs
      if access.writes
    }
    versions_before = dict(self._storage_versions)
    for storage_id in written_storages:
      self._storage_versions[storage_id] += 1
    uses = [
      TensorUse(
        tensor_id,
        access,
        argument,
        versions_before[self.tensors[tensor_id].storage_id],
        self._storage_versions[self.tensors[tensor_id].storage_id],
      )
      for tensor_id, access, argument in normalized_inputs
    ]
    output_ids = [self.tensor_id(value) for value in outputs]
    event = self._append_event(
      "operator",
      name,
      uses=uses,
      outputs=output_ids,
      attributes=attributes,
    )
    input_ids = {item[0] for item in normalized_inputs}
    for tensor_id in output_ids:
      tensor = self.tensors[tensor_id]
      if tensor.producer_event is None and tensor_id not in input_ids:
        tensor.producer_event = event.id
        if tensor.kind == TensorKind.unknown:
          tensor.kind = TensorKind.temporary
        storage = self.storages[tensor.storage_id]
        if storage.allocation_event is None and not storage.external:
          storage.allocation_event = event.id
      if tensor_id not in input_ids:
        access = AccessKind.alias if tensor.alias_of is not None else AccessKind.allocate
        version = self._storage_versions[tensor.storage_id]
        event.uses.append(TensorUse(tensor_id, access, "return", version, version))
    return event

  def _replay_session(self):
    # Metadata/precompile capture never changes a runtime session. Executable
    # capture observes a real invocation inside the existing planner context.
    tape = getattr(self, "_torch_replay_tape", None)
    return None if tape is None else tape.session_owner

  @contextlib.contextmanager
  def region(
    self,
    name: str,
    inputs: Iterable[Any] = (),
    *,
    attributes: dict[str, Any] | None = None,
  ):
    input_ids = [self.tensor_id(value) for value in inputs if self._object_entry(value) is not None]
    session = self._replay_session()
    if session is not None:
      session.enter_scope(name)
    enter = self._append_event(
      "region_enter",
      name,
      uses=[TensorUse(item, AccessKind.read, "argument") for item in input_ids],
      attributes=attributes,
    )
    self._scope.append(name)
    state: dict[str, Any] = {"outputs": ()}
    try:
      yield state
    finally:
      self._scope.pop()
      if session is not None:
        session.leave_scope()
      output_ids = [
        self.tensor_id(value)
        for value in state.get("outputs", ())
        if self._object_entry(value) is not None
      ]
      self._append_event(
        "region_exit",
        name,
        uses=[TensorUse(item, AccessKind.alias, "return") for item in output_ids],
        outputs=output_ids,
        attributes={"enter_event": enter.id},
      )

  def declare_loop(
    self,
    name: str,
    trip_count: int | str,
    *,
    attributes: dict[str, Any] | None = None,
  ) -> "LoopHandle":
    self._loop_counter += 1
    loop_id = f"l{self._loop_counter}"
    begin = self._append_event(
      "loop_enter",
      name,
      attributes={"loop_id": loop_id, "trip_count": trip_count, **dict(attributes or {})},
    )
    loop = Loop(
      loop_id,
      name,
      self.scope,
      trip_count,
      begin_event=begin.id,
      attributes=dict(attributes or {}),
    )
    self.loops.append(loop)
    session = self._replay_session()
    runtime = None if session is None else session.declare_loop(name, trip_count)
    return LoopHandle(self, loop, runtime)

  def record_marker(
    self, kind: str, name: str, *, attributes: dict[str, Any] | None = None
  ) -> Event:
    return self._append_event(kind, name, attributes=attributes)

  def enter_scope(self, name: str, *, kind: str = "scope") -> None:
    session = self._replay_session()
    if session is not None:
      session.enter_scope(name)
    self._append_event(f"{kind}_enter", name)
    self._scope.append(name)

  def leave_scope(self, *, kind: str = "scope") -> None:
    if not self._scope:
      raise RuntimeError("cannot leave an empty graph scope stack")
    name = self._scope.pop()
    session = self._replay_session()
    if session is not None:
      session.leave_scope()
    self._append_event(f"{kind}_exit", name)

  def finish(self) -> ExecutionGraph:
    """Finalize semantic liveness and return a validated immutable snapshot."""
    if self._active_token is not None:
      raise RuntimeError("leave the graph capture context before finishing it")
    if self._finished:
      raise RuntimeError("graph capture can only be finished once")
    if self._scope:
      raise RuntimeError("graph capture has unclosed scopes")
    self._finished = True
    graph_end = self.events[-1].index if self.events else 0
    event_indexes = {event.id: event.index for event in self.events}
    accesses: dict[str, list[int]] = {tensor_id: [] for tensor_id in self.tensors}
    for event in self.events:
      for use in event.uses:
        accesses[use.tensor_id].append(event.index)
      for output in event.outputs:
        accesses[output].append(event.index)

    for tensor_id, tensor in self.tensors.items():
      start = event_indexes.get(tensor.producer_event, 0)
      seen = accesses[tensor_id]
      end = max(seen, default=start)
      if tensor.persistent or tensor.escapes:
        end = graph_end
      tensor.lifetime_start = start
      tensor.lifetime_end = end
    for storage in self.storages.values():
      storage.lifetime_start = min(self.tensors[item].lifetime_start for item in storage.tensors)
      storage.lifetime_end = max(self.tensors[item].lifetime_end for item in storage.tensors)
      storage.persistent |= any(self.tensors[item].persistent for item in storage.tensors)

    graph = ExecutionGraph(
      name=self.name,
      events=list(self.events),
      tensors=dict(self.tensors),
      storages=dict(self.storages),
      loops=list(self.loops),
      inputs=list(self.inputs),
      outputs=list(self.outputs),
      attributes=dict(self.attributes),
    )
    graph.validate()
    return graph


@dataclass
class LoopHandle:
  """Record sampled loop iterations without expanding the declared trip count."""

  capture: GraphCapture | None
  loop: Loop | None
  runtime: Any | None = None
  closed: bool = False

  @contextlib.contextmanager
  def iteration(
    self,
    index: int | str,
    *,
    role: str | None = None,
    inputs: Any = (),
  ):
    if self.runtime is not None and self.capture is None:
      with self.runtime.iteration(index, role):
        yield {"outputs": ()}
      return
    if self.capture is None or self.loop is None:
      yield {"outputs": ()}
      return
    if self.closed:
      raise RuntimeError("cannot enter an iteration of a closed loop")
    with contextlib.ExitStack() as contexts:
      if self.runtime is not None:
        contexts.enter_context(self.runtime.iteration(index, role))
      self.loop.observed_iterations.append(index)
      label = f"{self.loop.name}[{index}]"
      enter = self.capture._append_event(
        "loop_iteration_enter",
        label,
        attributes={"loop_id": self.loop.id, "iteration": index, "role": role},
      )
      input_ids = [
        self.capture.tensor_id(value)
        for value in _tensor_leaves(inputs, self.capture)
      ]
      sample = LoopSample(index, role, enter.id, input_tensors=input_ids)
      self.loop.samples.append(sample)
      self.capture._scope.append(label)
      state: dict[str, Any] = {"outputs": ()}
      try:
        yield state
      finally:
        self.capture._scope.pop()
        exit_event = self.capture._append_event(
          "loop_iteration_exit",
          label,
          attributes={"loop_id": self.loop.id, "iteration": index, "role": role},
        )
        sample.end_event = exit_event.id
        sample.output_tensors = [
          self.capture.tensor_id(value)
          for value in _tensor_leaves(state.get("outputs", ()), self.capture)
        ]

  def close(self) -> None:
    if self.closed:
      return
    self.closed = True
    if self.runtime is not None:
      self.runtime.close()
    if self.capture is None or self.loop is None:
      return
    event = self.capture._append_event("loop_exit", self.loop.name, attributes={"loop_id": self.loop.id})
    self.loop.end_event = event.id


def declare_loop(
  name: str,
  trip_count: int | str,
  *,
  attributes: dict[str, Any] | None = None,
) -> LoopHandle:
  capture = current_capture()
  if capture is None:
    try:
      from vecops.planner.runtime import current_session

      session = current_session()
    except ImportError:
      session = None
    runtime = None if session is None else session.declare_loop(name, trip_count)
    return LoopHandle(None, None, runtime)
  return capture.declare_loop(name, trip_count, attributes=attributes)


def enter_scope(name: str, *, kind: str = "scope") -> None:
  capture = current_capture()
  if capture is not None:
    capture.enter_scope(str(name), kind=kind)
    return
  try:
    from vecops.planner.runtime import current_session

    session = current_session()
  except ImportError:
    session = None
  if session is not None:
    session.enter_scope(str(name))


def leave_scope(*, kind: str = "scope") -> None:
  capture = current_capture()
  if capture is not None:
    capture.leave_scope(kind=kind)
    return
  try:
    from vecops.planner.runtime import current_session

    session = current_session()
  except ImportError:
    session = None
  if session is not None:
    session.leave_scope()


def mark_alias(alias: Any, base: Any, *, kind: str = "explicit") -> None:
  """Annotate aliasing during an active eager/FakeTensor trace; otherwise no-op."""
  capture = current_capture()
  if capture is not None:
    capture.mark_alias(alias, base, kind=kind)


def mark_tensor(value: Any, **properties: Any) -> None:
  """Attach lifetime/ownership properties to a registered tensor."""
  capture = current_capture()
  if capture is not None:
    capture.annotate_tensor(value, **properties)


def mark_outputs(values: Iterable[Any]) -> None:
  capture = current_capture()
  if capture is not None:
    capture.mark_outputs(_tensor_leaves(values, capture))


def mark_retained(values: Any, *, reason: str | None = None) -> None:
  """Mark values retained by a model cache until the captured execution ends."""
  capture = current_capture()
  if capture is not None:
    capture.mark_retained(values, reason=reason)


def wrap_region(
  function=None,
  /,
  *,
  name: str | None = None,
  attributes: dict[str, Any] | None = None,
):
  """Wrap capture regions and mirror their lexical scopes during factory replay.

  Replay needs the same scope keys as capture, including decorated callables.
  Ordinary eager execution and dynamic-only sessions do not add region scopes;
  ``VECOPS_MIRROR_REGION_SCOPE=1`` can also request them for diagnostics.
  """
  if function is None:
    return functools.partial(wrap_region, name=name, attributes=attributes)
  region_name = name or f"{function.__module__}.{function.__qualname__}"

  @functools.wraps(function)
  def wrapped(*args, **kwargs):
    capture = current_capture()
    if capture is not None:
      tensor_inputs = _tensor_leaves((args, kwargs), capture)
      with capture.region(region_name, tensor_inputs, attributes=attributes) as state:
        result = function(*args, **kwargs)
        state["outputs"] = _tensor_leaves(result, capture)
        return result
    # Match captured factory keys only when replay is active. Retained and
    # escaping aliases are excluded by the planner; merely dropping a lexical
    # region would silently route a different allocation into its parent scope.
    try:
      from vecops.planner.runtime import current_session

      session = current_session()
    except ImportError:
      session = None
    if session is None or (
        not session.enable_factory_replay
        and os.environ.get("VECOPS_MIRROR_REGION_SCOPE") != "1"
    ):
      return function(*args, **kwargs)
    session.enter_scope(region_name)
    try:
      return function(*args, **kwargs)
    finally:
      session.leave_scope()

  return wrapped


def _tensor_leaves(value: Any, capture: GraphCapture) -> list[Any]:
  result: list[Any] = []
  if capture._object_entry(value) is not None:
    return [value]
  if isinstance(value, dict):
    for item in value.values():
      result.extend(_tensor_leaves(item, capture))
  elif isinstance(value, (tuple, list)):
    for item in value:
      result.extend(_tensor_leaves(item, capture))
  return result


__all__ = [
  "GraphCapture",
  "LoopHandle",
  "current_capture",
  "declare_loop",
  "enter_scope",
  "leave_scope",
  "mark_alias",
  "mark_outputs",
  "mark_retained",
  "mark_tensor",
  "wrap_region",
]
