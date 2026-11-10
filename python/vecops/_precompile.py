# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Model-level collection and parallel preparation of lazy JIT artifacts."""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

from . import _C
from ._dtype import normalize_dtype
from ._schema import TensorMeta


@dataclass(frozen=True)
class CompileRequest:
  """One normalized, storage-free request for a native operator artifact."""

  operator: _C.Operator
  call: _C.KernelCall
  key: tuple[Any, ...]
  name: str
  source: str | None = None


@dataclass(frozen=True)
class PrecompileResult:
  """Preparation counts and graphs, including qualified planning-only restore.

  ``graph`` always exposes the complete captured IR; an application discovery
  provider may defer decoding until first access. ``planning_graph`` is a
  separate immutable projection and never substitutes for the full graph.
  """

  collected: int
  prepared: int
  parallelism: int
  elapsed_seconds: float
  graph: Any | None = None
  planning_graph: Any | None = None
  _graph_loader: Any | None = field(default=None,repr=False,compare=False)

  def __getattribute__(self,name):
    value=object.__getattribute__(self,name)
    if name=='graph' and value is None:
      loader=object.__getattribute__(self,'_graph_loader')
      if loader is not None:
        value=loader()
        object.__setattr__(self,'graph',value)
        object.__setattr__(self,'_graph_loader',None)
    return value


class CompileRequestCollector:
  """Deduplicated insertion-ordered collection of artifact requests."""

  def __init__(self) -> None:
    self._requests: dict[tuple[Any, ...], CompileRequest] = {}
    self.discover_graph_artifacts = False
    self.graph_kernels = {}
    self.torch_call_observer = None

  @property
  def requests(self) -> tuple[CompileRequest, ...]:
    return tuple(self._requests.values())

  def add(self, request: CompileRequest) -> None:
    self._requests.setdefault(request.key, request)

  def __len__(self) -> int:
    return len(self._requests)


_active_collector: CompileRequestCollector | None = None
_collection_lock = threading.Lock()


def is_precompiling() -> bool:
  """Return whether this process is tracing model compile requests."""
  return _active_collector is not None


def is_discovering_graph_artifacts() -> bool:
  """Return whether the current fake trace can include graph lowering jobs."""
  return _active_collector is not None and _active_collector.discover_graph_artifacts


@contextlib.contextmanager
def collect_compile_requests():
  """Collect vecops calls without executing them in the current context."""
  global _active_collector
  if not _collection_lock.acquire(blocking=False):
    raise RuntimeError("vecops compile-request collection is already active")
  collector = CompileRequestCollector()
  _active_collector = collector
  try:
    yield collector
  finally:
    _active_collector = None
    _collection_lock.release()


def _element_strides(argument: Any) -> tuple[int, ...]:
  stride = getattr(argument, "stride", None)
  if callable(stride):
    return tuple(int(value) for value in stride())
  strides = getattr(argument, "strides", None)
  if strides is None:
    shape = tuple(int(value) for value in argument.shape)
    dense = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
      dense[axis] = dense[axis + 1] * shape[axis + 1]
    return tuple(dense)
  itemsize = getattr(argument, "itemsize", None)
  if itemsize is not None:
    return tuple(int(value) // int(itemsize) for value in strides)
  return tuple(int(value) for value in strides)


def _tensor_request(argument: Any, parameter: _C.TensorDef):
  if argument is None:
    return None, ("none",)
  if isinstance(argument, TensorMeta):
    shape = argument.shape
    strides = argument.strides
    dtype = normalize_dtype(argument.dtype)
    device_type = argument.device_type
    device_index = argument.device_index
  else:
    if not hasattr(argument, "shape") or not hasattr(argument, "dtype"):
      raise TypeError(
          f"tensor parameter {parameter.name!r} cannot collect metadata from "
          f"{type(argument).__name__}"
      )
    shape = tuple(int(value) for value in argument.shape)
    strides = _element_strides(argument)
    dtype = normalize_dtype(argument.dtype)
    # Generated source kernels currently target the device contract declared
    # by KernelDef. Fake/meta tensors describe storage, not a different target.
    device_type = parameter.device_type
    device_index = 0 if parameter.device_index < 0 else parameter.device_index
  metadata = TensorMeta(
      shape,
      dtype=dtype,
      strides=strides,
      access=parameter.access,
      device_type=device_type,
      device_index=device_index,
  )
  descriptor = (
      "tensor",
      str(dtype),
      tuple(shape),
      tuple(strides),
      str(parameter.access),
      device_type,
      device_index,
  )
  return metadata._native(), descriptor


_SIGNED_DTYPES = {_C.DType.int8, _C.DType.int16, _C.DType.int32, _C.DType.int64}
_UNSIGNED_DTYPES = {
    _C.DType.bool_,
    _C.DType.uint8,
    _C.DType.uint16,
    _C.DType.uint32,
    _C.DType.uint64,
}


def _scalar_request(argument: Any, parameter: _C.ValueDef) -> _C.Scalar:
  if isinstance(argument, _C.Scalar):
    return argument
  if parameter.dtype in _SIGNED_DTYPES:
    return _C.Scalar.signed_integer(int(argument), parameter.dtype)
  if parameter.dtype in _UNSIGNED_DTYPES:
    return _C.Scalar.unsigned_integer(int(argument), parameter.dtype)
  return _C.Scalar.floating(float(argument), parameter.dtype)


def maybe_collect_call(
    operator: _C.Operator,
    definition: _C.KernelDef,
    arguments: Iterable[Any],
    specializations: dict[str, Any],
    *,
    name: str | None = None,
    source: str | None = None,
) -> bool:
  """Record one call when collection is active; return whether it was caught."""
  collector = _active_collector
  if collector is None:
    return False

  arguments = list(arguments)
  parameters = tuple(definition.inputs)
  if len(arguments) > len(parameters):
    raise TypeError(
        f"compile collection expected at most {len(parameters)} arguments, got "
        f"{len(arguments)}"
    )
  for parameter in parameters[len(arguments) :]:
    if isinstance(parameter, _C.TensorDef) and parameter.optional:
      arguments.append(None)
    elif isinstance(parameter, _C.ValueDef) and parameter.default is not None:
      arguments.append(parameter.default.value)
    else:
      raise TypeError(f"missing required argument: {parameter.name}")

  native_arguments = []
  signature = []
  for argument, parameter in zip(arguments, parameters):
    if isinstance(parameter, _C.TensorDef):
      native_argument, descriptor = _tensor_request(argument, parameter)
      native_arguments.append(native_argument)
      signature.append(descriptor)
    else:
      native_arguments.append(_scalar_request(argument, parameter))
      # Runtime scalar payloads do not specialize generated artifacts.
      signature.append(("scalar", str(parameter.dtype)))

  values = {}
  value_signature = []
  for symbol, kind in definition.values.items():
    value = specializations.get(symbol)
    if value is None:
      value_signature.append((symbol, "default"))
      continue
    if kind == _C.DTypeValue:
      value = normalize_dtype(value)
    values[symbol] = value
    value_signature.append((symbol, value))

  handle = getattr(operator, "_handle", id(operator))
  request_name = name or getattr(definition, "name", "vecops operator")
  key = (handle, tuple(signature), tuple(value_signature))
  collector.add(
      CompileRequest(
          operator=operator,
          call=_C.KernelCall(native_arguments, values),
          key=key,
          name=request_name,
          source=source,
      )
  )
  # Collection short-circuits the generated Torch dispatcher call, so a
  # TorchDispatchMode cannot observe this operator.  Emit the semantic vecops
  # call directly into an active execution-graph capture while retaining the
  # caller-allocated output tensors and their explicit access contracts.
  from .graph import AccessKind, current_capture

  graph_capture = current_capture()
  if graph_capture is not None:
    tape = getattr(graph_capture, '_torch_replay_tape', None)
    if (getattr(tape, '_region_projection', False)
        and getattr(graph_capture, '_torch_recording_dispatch_depth', 0)):
      # The outer ATen operation is the executable instruction. Its router's
      # internal native request still compiles, but is not a second model op.
      return True
    from .graph.torch import is_tensor, register_tensor

    graph_inputs = []
    graph_outputs = []
    for argument, parameter in zip(arguments, parameters):
      if (
        not isinstance(parameter, _C.TensorDef)
        or argument is None
        or not is_tensor(argument)
      ):
        continue
      try:
        graph_capture.tensor_id(argument)
      except KeyError:
        register_tensor(graph_capture, argument, kind="external", external=True)
      access_name = str(parameter.access).lower()
      if "inout" in access_name:
        access = AccessKind.read_write
      elif parameter.output:
        access = AccessKind.write
      else:
        access = AccessKind.read
      graph_inputs.append((argument, access, parameter.name))
      if parameter.output:
        graph_outputs.append(argument)
    graph_capture.record_operator(
      request_name,
      inputs=graph_inputs,
      outputs=graph_outputs,
      attributes={
        "framework": "vecops",
        "specializations": dict(value_signature),
      },
    )
  return True


def compile_batch(
    requests: Iterable[CompileRequest],
    *,
    parallelism: int | None = None,
    execution_parallelism: int | None = None,
) -> PrecompileResult:
  """Prepare requests through shared CMake build graphs."""
  from ._compiler import require_native_build_api

  require_native_build_api()
  unique = {request.key: request for request in requests}
  from ._build_cost import order_requests
  pending = order_requests(unique.values())
  from ._build_session import build_frontier, build_parallelism
  parallelism = build_parallelism(parallelism)
  if isinstance(parallelism, bool) or parallelism < 1:
    raise ValueError("parallelism must be a positive integer")
  if execution_parallelism is None:
    torch = sys.modules.get("torch")
    execution_parallelism = (
        int(torch.get_num_threads())
        if torch is not None
        else 1
    )
  if isinstance(execution_parallelism, bool) or execution_parallelism < 1:
    raise ValueError("execution_parallelism must be a positive integer")
  workers = parallelism if pending else 0
  start = time.perf_counter()
  if pending:
    try:
      # Providers invoke the input validator only for real cache misses. Avoid
      # scanning the whole toolchain for a frontier that merely binds hits.
      from ._build_session import current_build_session
      session = current_build_session()
      if session is not None:
        session._native_validation_required = True
      with build_frontier('native', validate=False, requests=len(pending), parallelism=parallelism):
        _C.prepare_batch(
            [(request.operator, request.call) for request in pending],
            parallel_jobs=parallelism,
            execution_parallelism=execution_parallelism,
        )
    except Exception as error:
      if hasattr(error, "add_note"):
        names = ", ".join(repr(request.name) for request in pending)
        error.add_note(f"while preparing vecops batch containing {names}")
      raise

  return PrecompileResult(
      collected=len(pending),
      prepared=len(pending),
      parallelism=workers,
      elapsed_seconds=time.perf_counter() - start,
  )


def _fake_tree(tree: Any, mode: Any, torch: Any) -> Any:
  from torch._subclasses.fake_tensor import FakeTensor
  from torch.utils._pytree import tree_map

  def convert(value):
    if isinstance(value, FakeTensor):
      return value
    if isinstance(value, torch.Tensor):
      return mode.from_tensor(value)
    return value

  return tree_map(convert, tree)


def precompile(
    model: Any,
    *args: Any,
    use_real_tensors: bool = False,
    parallelism: int | None = None,
    capture_graph: bool = False,
    graph_name: str | None = None,
    _graph_state_owner: Any | None = None,
    _torch_call_observer: Callable | None = None,
    prepare_owner: bool = True,
    prepare_barrier: Callable[[], None] | None = None,
    prepare_rank: int | None = None,
    prepare_world_size: int | None = None,
    prepare_exchange: Callable[[Any], list[Any]] | None = None,
    _discovery_cache: Any | None = None,
    **kwargs: Any,
) -> PrecompileResult:
  """Trace a model's vecops calls, then prepare all unique artifacts in batch.

  By default the trace runs under Torch ``FakeTensorMode``: parameters, inputs,
  and intermediate factory results carry metadata only. Set
  ``use_real_tensors=True`` to run ordinary Torch operations on the supplied
  values while vecops calls still return their caller-allocated, uninitialized
  outputs. Set ``capture_graph=True`` to additionally return a framework-neutral
  operator, storage, alias, loop, and semantic-lifetime trace in ``result.graph``.
  """
  import torch

  if not isinstance(use_real_tensors, bool):
    raise TypeError("use_real_tensors must be bool")

  if not isinstance(capture_graph, bool):
    raise TypeError("capture_graph must be bool")

  discovery_started = time.perf_counter()
  graph_capture = None
  captured_graph = None
  if capture_graph:
    from .graph import GraphCapture

    graph_capture = GraphCapture(
      graph_name or getattr(model, "__qualname__", model.__class__.__qualname__),
      attributes={
        "framework": "torch",
        "trace_mode": "real" if use_real_tensors else "fake",
        "torch_version": torch.__version__,
      },
    )

  def run_model(model_args, model_kwargs, *, fake_mode=None):
    if graph_capture is None:
      return model(*model_args, **model_kwargs)
    from .graph.torch import (
      mark_graph_outputs,
      record_torch,
      register_inputs,
      register_module_state,
    )

    with graph_capture:
      state_owner = _graph_state_owner if _graph_state_owner is not None else model
      if hasattr(state_owner, "named_parameters") and hasattr(
        state_owner, "named_buffers"
      ):
        register_module_state(graph_capture, state_owner, fake_mode=fake_mode)
      # ``model_args`` have already been converted by ``_fake_tree`` when a
      # FakeTensorMode is supplied; converting those FakeTensors again is both
      # unnecessary and rejected by some Torch releases.
      register_inputs(graph_capture, (model_args, model_kwargs))
      with record_torch(graph_capture):
        result = model(*model_args, **model_kwargs)
        observer_owner = getattr(_torch_call_observer, '__self__', None)
        finalize = getattr(observer_owner, 'finish_discovery', None)
        if fake_mode is not None and finalize is not None:
          finalize()
      mark_graph_outputs(graph_capture, result)
      return result

  from ._build_session import profile_phase
  with profile_phase("discovery", "model"), collect_compile_requests() as collector:
    collector.discover_graph_artifacts = not use_real_tensors
    collector.torch_call_observer = _torch_call_observer
    cache_hit = False
    workspace_allocations = None
    if _discovery_cache is not None:
      captured_graph = _discovery_cache.load(collector)
      cache_hit = _discovery_cache.hit
      if prepare_exchange is not None:
        cache_hit = all(prepare_exchange(cache_hit))
      if not cache_hit:
        _discovery_cache.discard(collector)
        captured_graph = None
    if not cache_hit:
      with torch.inference_mode():
        # Request collection is a Python-side eager trace. Ignoring nested
        # torch.compile directives also keeps the global collector check out of
        # Dynamo graphs used by ordinary inference.
        with torch.compiler.set_stance("force_eager"):
          if use_real_tensors:
            # Real tensors execute the kernels, so the native workspace
            # allocation observer can run; fake tensors short-circuit the
            # dispatch and would record nothing.
            try:
              from .workspace_observer import drain, tracking
            except ImportError:
              tracking_cm = None
            else:
              tracking_cm = tracking()
            if tracking_cm is not None:
              with tracking_cm:
                run_model(args, kwargs)
                workspace_allocations = drain()
            else:
              run_model(args, kwargs)
              workspace_allocations = None
          else:
            from .graph._metadata import metadata_mode
            mode = metadata_mode()
            with mode:
              fake_args = _fake_tree(args, mode, torch)
              fake_kwargs = _fake_tree(kwargs, mode, torch)
              run_model(fake_args, fake_kwargs, fake_mode=mode)
            workspace_allocations = None
  if graph_capture is not None and not cache_hit:
    if workspace_allocations:
      graph_capture.attributes["workspace_allocations"] = workspace_allocations
    captured_graph = graph_capture.finish()

  if _discovery_cache is not None and not cache_hit:
    _discovery_cache.save(collector, captured_graph)

  graph_result=dict(graph=captured_graph,
      planning_graph=getattr(_discovery_cache,'planning_graph',None),
      _graph_loader=getattr(_discovery_cache,'graph_loader',None))

  from ._build_session import current_build_session
  from ._schema_bridge import compile_pending_torch_bridges
  if session := current_build_session():
    session.events.append({'kind': 'graph-discovery', 'start_seconds': discovery_started-session._started,
                           'seconds': time.perf_counter()-discovery_started, 'kernels': len(collector.graph_kernels),
                           'requests': len(collector),
                           'discovery_cache': None if _discovery_cache is None else _discovery_cache.stats()})

  # Live distributed traces execute on all ranks; discovery-cache hits require
  # all-rank agreement. Shared recipe exchange coalesces compilation, retaining
  # local ABI bindings. The older rank-turn policy remains the fallback when
  # no application metadata transport is supplied.
  def prepare_collected(requests=None):
    requests = collector.requests if requests is None else requests
    from ._build_session import current_build_session
    session = current_build_session()
    if session is None:
      result = compile_batch(requests, parallelism=parallelism)
      compile_pending_torch_bridges(parallelism=parallelism)
      return result
    return session.run_tasks([
        ("native-ready", lambda: compile_batch(requests, parallelism=parallelism)),
        ("bridges-ready", lambda: compile_pending_torch_bridges(parallelism=parallelism)),
    ])[0]

  if prepare_rank is not None or prepare_world_size is not None:
    if prepare_barrier is None:
      raise ValueError("prepare_barrier is required with prepare_rank")
    if prepare_rank is None or prepare_world_size is None:
      raise ValueError("prepare_rank and prepare_world_size must be set together")
    if not 0 <= prepare_rank < prepare_world_size:
      raise ValueError("prepare_rank must be in [0, prepare_world_size)")

    if prepare_exchange is not None:
      from ._prepare_exchange import exchange_requests
      selected, owners, local_complete = exchange_requests(
          collector.requests, prepare_rank, prepare_exchange, int(torch.get_num_threads()))
      if session := current_build_session():
        session.events.append(dict(kind='distributed-prepare', rank=prepare_rank,
            requests=len(collector), union_requests=len(selected), owners=list(owners),
            local_complete=local_complete))
      # Usually one owner prepares the union. Unknown declarations and another
      # host/ISA/cache policy retain their own owner; no rank's requests vanish.
      # Report failures collectively before the next phase can wait forever.
      def finish_phase(error):
        statuses = prepare_exchange(None if error is None else str(error))
        failed = next((message for message in statuses if message is not None), None)
        if failed is not None:
          raise RuntimeError('distributed preparation failed: ' + failed) from error

      result = None
      for owner in owners:
        error = None
        if prepare_rank == owner:
          try:
            result = prepare_collected(selected)
          except Exception as caught:
            error = caught
        finish_phase(error)
      error = None
      try:
        if local_complete and result is not None:
          result = replace(result, collected=len(collector), prepared=len(collector))
        else:
          # All shared artifacts now exist. Native provider and Torch dispatcher
          # bindings remain local, and these hit-only frontiers run in parallel.
          result = prepare_collected()
      except Exception as caught:
        error = caught
      finish_phase(error)
      for kernel in collector.graph_kernels.values():
        kernel._graph_artifact_prepared = True
      return replace(result, **graph_result)

    result = None
    for turn in range(prepare_world_size):
      if prepare_rank == turn:
        result = prepare_collected()
      prepare_barrier()
    assert result is not None
    for kernel in collector.graph_kernels.values():
      kernel._graph_artifact_prepared = True
    return replace(result, **graph_result)

  # Backward-compatible two-phase owner mode.
  if prepare_barrier is not None and not prepare_owner:
    prepare_barrier()
  result = prepare_collected()
  for kernel in collector.graph_kernels.values():
    kernel._graph_artifact_prepared = True
  if prepare_barrier is not None and prepare_owner:
    prepare_barrier()
  return replace(result, **graph_result)


__all__ = [
    "CompileRequest",
    "CompileRequestCollector",
    "PrecompileResult",
    "collect_compile_requests",
    "compile_batch",
  "is_precompiling",
  "is_discovering_graph_artifacts",
    "precompile",
]
