"""Model-level collection and parallel preparation of lazy JIT artifacts."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class PrecompileResult:
  """Summary returned by :func:`compile_batch` and :func:`precompile`."""

  collected: int
  prepared: int
  parallelism: int
  elapsed_seconds: float
  graph: Any | None = None


class CompileRequestCollector:
  """Deduplicated insertion-ordered collection of artifact requests."""

  def __init__(self) -> None:
    self._requests: dict[tuple[Any, ...], CompileRequest] = {}

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
      )
  )
  # Collection short-circuits the generated Torch dispatcher call, so a
  # TorchDispatchMode cannot observe this operator.  Emit the semantic vecops
  # call directly into an active execution-graph capture while retaining the
  # caller-allocated output tensors and their explicit access contracts.
  from .graph import AccessKind, current_capture

  graph_capture = current_capture()
  if graph_capture is not None:
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
) -> PrecompileResult:
  """Prepare requests through shared CMake build graphs."""
  from ._compiler import require_native_build_api

  require_native_build_api()
  unique = {request.key: request for request in requests}
  pending = tuple(unique.values())
  if parallelism is None:
    affinity = getattr(os, "sched_getaffinity", None)
    available = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
    parallelism = min(8, available)
  if isinstance(parallelism, bool) or parallelism < 1:
    raise ValueError("parallelism must be a positive integer")
  workers = parallelism if pending else 0
  start = time.perf_counter()
  if pending:
    try:
      _C.prepare_batch(
          [(request.operator, request.call) for request in pending],
          parallel_jobs=parallelism,
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
    prepare_owner: bool = True,
    prepare_barrier: Callable[[], None] | None = None,
    prepare_rank: int | None = None,
    prepare_world_size: int | None = None,
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
  from torch._subclasses.fake_tensor import FakeTensorMode

  if not isinstance(use_real_tensors, bool):
    raise TypeError("use_real_tensors must be bool")

  if not isinstance(capture_graph, bool):
    raise TypeError("capture_graph must be bool")

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
      mark_graph_outputs(graph_capture, result)
      return result

  with collect_compile_requests() as collector:
    with torch.inference_mode():
      # Request collection is a Python-side eager trace. Ignoring nested
      # torch.compile directives also keeps the global collector check out of
      # Dynamo graphs used by ordinary inference.
      with torch.compiler.set_stance("force_eager"):
        if use_real_tensors:
          run_model(args, kwargs)
        else:
          mode = FakeTensorMode(allow_non_fake_inputs=True)
          with mode:
            fake_args = _fake_tree(args, mode, torch)
            fake_kwargs = _fake_tree(kwargs, mode, torch)
            run_model(fake_args, fake_kwargs, fake_mode=mode)

  if graph_capture is not None:
    captured_graph = graph_capture.finish()

  from ._schema_bridge import compile_pending_torch_bridges

  # A distributed trace must execute on every rank because model forward
  # contains collectives. Compilation itself is host-global. Serialize cache
  # preparation by rank so an uneven shard may add its shape variants without
  # competing with another compiler batch. Later ranks normally only resolve
  # artifacts already populated by an earlier rank.
  if prepare_rank is not None or prepare_world_size is not None:
    if prepare_barrier is None:
      raise ValueError("prepare_barrier is required with prepare_rank")
    if prepare_rank is None or prepare_world_size is None:
      raise ValueError("prepare_rank and prepare_world_size must be set together")
    if not 0 <= prepare_rank < prepare_world_size:
      raise ValueError("prepare_rank must be in [0, prepare_world_size)")

    result = None
    for turn in range(prepare_world_size):
      if prepare_rank == turn:
        result = compile_batch(collector.requests, parallelism=parallelism)
        compile_pending_torch_bridges(parallelism=parallelism)
      prepare_barrier()
    assert result is not None
    return replace(result, graph=captured_graph)

  # Backward-compatible two-phase owner mode.
  if prepare_barrier is not None and not prepare_owner:
    prepare_barrier()
  result = compile_batch(collector.requests, parallelism=parallelism)
  compile_pending_torch_bridges(parallelism=parallelism)
  if prepare_barrier is not None and prepare_owner:
    prepare_barrier()
  return replace(result, graph=captured_graph)


__all__ = [
    "CompileRequest",
    "CompileRequestCollector",
    "PrecompileResult",
    "collect_compile_requests",
    "compile_batch",
    "is_precompiling",
    "precompile",
]
