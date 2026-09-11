"""Model-level collection and parallel preparation of lazy JIT artifacts."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

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
    **kwargs: Any,
) -> PrecompileResult:
  """Trace a model's vecops calls, then prepare all unique artifacts in batch.

  By default the trace runs under Torch ``FakeTensorMode``: parameters, inputs,
  and intermediate factory results carry metadata only. Set
  ``use_real_tensors=True`` to run ordinary Torch operations on the supplied
  values while vecops calls still return their caller-allocated, uninitialized
  outputs.
  """
  import torch
  from torch._subclasses.fake_tensor import FakeTensorMode

  if not isinstance(use_real_tensors, bool):
    raise TypeError("use_real_tensors must be bool")

  with collect_compile_requests() as collector:
    with torch.inference_mode():
      # Request collection is a Python-side eager trace. Ignoring nested
      # torch.compile directives also keeps the global collector check out of
      # Dynamo graphs used by ordinary inference.
      with torch.compiler.set_stance("force_eager"):
        if use_real_tensors:
          model(*args, **kwargs)
        else:
          mode = FakeTensorMode(allow_non_fake_inputs=True)
          with mode:
            fake_args = _fake_tree(args, mode, torch)
            fake_kwargs = _fake_tree(kwargs, mode, torch)
            model(*fake_args, **fake_kwargs)

  result = compile_batch(collector.requests, parallelism=parallelism)
  from ._schema_bridge import compile_pending_torch_bridges

  compile_pending_torch_bridges(parallelism=parallelism)
  return result


__all__ = [
    "CompileRequest",
    "CompileRequestCollector",
    "PrecompileResult",
    "collect_compile_requests",
    "compile_batch",
    "is_precompiling",
    "precompile",
]
