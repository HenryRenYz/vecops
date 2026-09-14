"""Optional Torch adapter for :mod:`vecops.graph`.

Imports are intentionally local to this module so the serialized graph and
all framework-neutral capture APIs remain usable without Torch installed.
"""

from __future__ import annotations

import contextlib
import math
import traceback
from typing import Any, Iterable

from .capture import GraphCapture
from .ir import AccessKind, TensorKind, TensorMetadata


def _dimension(value: Any) -> int | str:
  try:
    return int(value)
  except (TypeError, ValueError, RuntimeError):
    return str(value)


def _logical_bytes(shape: tuple[int | str, ...], element_size: int | None) -> int | str | None:
  if element_size is None:
    return None
  if all(isinstance(item, int) for item in shape):
    return math.prod(shape) * element_size
  expression = "*".join(str(item) for item in shape) or "1"
  return f"({expression})*{element_size}"


def _storage_identity(tensor: Any) -> tuple[str, int] | None:
  try:
    storage = tensor.untyped_storage()
    identity = getattr(storage, "_cdata", None)
    if identity is None:
      identity = id(storage)
    return "torch-storage", int(identity)
  except (AttributeError, RuntimeError, NotImplementedError):
    return None


def _storage_bytes(tensor: Any) -> int | None:
  try:
    return int(tensor.untyped_storage().nbytes())
  except (AttributeError, RuntimeError, TypeError, NotImplementedError):
    return None


def tensor_metadata(tensor: Any) -> TensorMetadata:
  """Translate one real, Fake, or meta Tensor into framework-neutral metadata."""
  shape = tuple(_dimension(item) for item in tensor.shape)
  try:
    strides = tuple(_dimension(item) for item in tensor.stride())
  except (AttributeError, RuntimeError):
    strides = ()
  try:
    element_size = int(tensor.element_size())
  except (AttributeError, RuntimeError):
    element_size = None
  try:
    storage_offset = _dimension(tensor.storage_offset())
  except (AttributeError, RuntimeError):
    storage_offset = 0
  return TensorMetadata(
    shape=shape,
    strides=strides,
    dtype=str(tensor.dtype),
    device=str(tensor.device),
    layout=str(tensor.layout),
    element_size=element_size,
    storage_offset=storage_offset,
    logical_bytes=_logical_bytes(shape, element_size),
    requires_grad=bool(getattr(tensor, "requires_grad", False)),
  )


def is_tensor(value: Any) -> bool:
  import torch

  return isinstance(value, torch.Tensor)


def tensor_leaves(value: Any) -> list[Any]:
  """Flatten only Tensor leaves while accepting arbitrary model containers."""
  import torch
  from torch.utils._pytree import tree_flatten

  leaves, _spec = tree_flatten(value)
  return [item for item in leaves if isinstance(item, torch.Tensor)]


def register_tensor(
  capture: GraphCapture,
  tensor: Any,
  *,
  name: str | None = None,
  kind: TensorKind | str = TensorKind.unknown,
  persistent: bool = False,
  read_only: bool = False,
  external: bool = False,
  annotations: dict[str, Any] | None = None,
) -> str:
  return capture.tensor(
    tensor,
    tensor_metadata(tensor),
    storage_identity=_storage_identity(tensor),
    storage_bytes=_storage_bytes(tensor),
    name=name,
    kind=kind,
    persistent=persistent,
    read_only=read_only,
    external=external,
    annotations=annotations,
  )


def _schema_argument_values(
  schema: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[tuple[Any, Any]]:
  result = []
  for index, argument in enumerate(getattr(schema, "arguments", ())):
    if index < len(args):
      value = args[index]
    elif argument.name in kwargs:
      value = kwargs[argument.name]
    else:
      continue
    result.append((argument, value))
  return result


def _source_location() -> dict[str, Any] | None:
  """Return the nearest non-Torch/non-capture Python frame for an allocation."""
  for frame in reversed(traceback.extract_stack(limit=48)[:-1]):
    normalized = frame.filename.replace("\\", "/")
    if (
      "/site-packages/torch/" in normalized
      or "/vecops/graph/" in normalized
      or "/vecops/planner/" in normalized
    ):
      continue
    if normalized.endswith("/vecops/_precompile.py"):
      continue
    return {"file": frame.filename, "line": frame.lineno, "function": frame.name}
  return None


def _alias_set(alias_info: Any) -> set[str]:
  if alias_info is None:
    return set()
  result = set()
  for name in ("before_set", "after_set"):
    values = getattr(alias_info, name, ())
    try:
      result.update(str(item) for item in values)
    except TypeError:
      pass
  return result


class TorchDispatchRecorder:
  """A ``TorchDispatchMode`` facade recording every dispatched Torch op."""

  def __new__(cls, capture: GraphCapture):
    from torch.utils._python_dispatch import TorchDispatchMode

    class _Mode(TorchDispatchMode):
      def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        schema = getattr(function, "_schema", None)
        input_entries: list[tuple[Any, AccessKind, str | None]] = []
        input_aliases: list[tuple[Any, set[str]]] = []
        if schema is None:
          argument_values = [(None, args), (None, kwargs)]
        else:
          argument_values = _schema_argument_values(schema, args, kwargs)
        for argument, value in argument_values:
          alias_info = getattr(argument, "alias_info", None) if argument is not None else None
          access = AccessKind.read_write if getattr(alias_info, "is_write", False) else AccessKind.read
          argument_name = getattr(argument, "name", None)
          aliases = _alias_set(alias_info)
          for tensor in tensor_leaves(value):
            try:
              capture.tensor_id(tensor)
            except KeyError:
              # A value first observed as an input to this op entered from
              # outside the traced execution. Already registered values are
              # internal results or model state and must retain that ownership.
              register_tensor(
                capture,
                tensor,
                kind=TensorKind.external,
                external=True,
              )
            input_entries.append((tensor, access, argument_name))
            input_aliases.append((tensor, aliases))

        result = function(*args, **kwargs)
        outputs = tensor_leaves(result)
        new_object_ids = {id(tensor) for tensor in outputs if capture._object_entry(tensor) is None}
        for tensor in outputs:
          register_tensor(capture, tensor, kind=TensorKind.unknown)

        returns = tuple(getattr(schema, "returns", ())) if schema is not None else ()
        for index, output in enumerate(outputs):
          # Actual Storage identity is authoritative.  Schema alias sets are a
          # fallback for fake/custom kernels that conservatively manufacture a
          # fresh metadata tensor while declaring view semantics.
          output_id = capture.tensor_id(output)
          if capture.tensors[output_id].alias_of is not None:
            continue
          if _storage_identity(output) is not None:
            # Schemas such as aten::contiguous conservatively permit aliasing
            # but may return fresh storage for this invocation.  A concrete or
            # FakeTensor storage identity therefore wins over the schema.
            continue
          if len(returns) == len(outputs):
            return_schema = returns[index]
          elif len(returns) == 1:
            return_schema = returns[0]
          else:
            return_schema = None
          return_aliases = _alias_set(getattr(return_schema, "alias_info", None))
          if not return_aliases:
            continue
          for input_tensor, aliases in input_aliases:
            if return_aliases & aliases:
              capture.mark_alias(output, input_tensor, kind="schema")
              break

        produces_storage = any(
          id(tensor) in new_object_ids
          and capture.tensors[capture.tensor_id(tensor)].alias_of is None
          for tensor in outputs
        )
        attributes = {"schema": str(schema) if schema is not None else None}
        if produces_storage:
          source = _source_location()
          if source is not None:
            attributes["source"] = source
        capture.record_operator(
          str(function),
          inputs=input_entries,
          outputs=outputs,
          attributes=attributes,
        )
        return result

    return _Mode()


def register_module_state(
  capture: GraphCapture, module: Any, *, fake_mode: Any | None = None
) -> None:
  """Register named parameters and buffers before executing a module."""
  for name, tensor in module.named_parameters(remove_duplicate=False):
    tensor_id = register_tensor(
      capture,
      tensor,
      name=name,
      kind=TensorKind.parameter,
      persistent=True,
      read_only=True,
      external=True,
      annotations={"module_state": "parameter"},
    )
    if fake_mode is not None:
      observed = fake_mode.from_tensor(tensor)
      capture.bind_object(observed, tensor_id, storage_identity=_storage_identity(observed))
  for name, tensor in module.named_buffers(remove_duplicate=False):
    if tensor is None:
      continue
    tensor_id = register_tensor(
      capture,
      tensor,
      name=name,
      kind=TensorKind.buffer,
      persistent=True,
      external=True,
      annotations={"module_state": "buffer"},
    )
    if fake_mode is not None:
      observed = fake_mode.from_tensor(tensor)
      capture.bind_object(observed, tensor_id, storage_identity=_storage_identity(observed))


def register_inputs(capture: GraphCapture, values: Any, *, fake_mode: Any | None = None) -> list[Any]:
  observed = []
  for index, tensor in enumerate(tensor_leaves(values)):
    item = fake_mode.from_tensor(tensor) if fake_mode is not None else tensor
    tensor_id = register_tensor(
      capture,
      item,
      name=f"input.{index}",
      kind=TensorKind.input,
      external=True,
    )
    capture.mark_input(tensor_id)
    observed.append(item)
  return observed


def mark_graph_outputs(capture: GraphCapture, values: Any) -> None:
  capture.mark_outputs(tensor_leaves(values))


@contextlib.contextmanager
def record_torch(capture: GraphCapture):
  """Record dispatched Torch operations into an already active capture."""
  with TorchDispatchRecorder(capture):
    yield


__all__ = [
  "TorchDispatchRecorder",
  "is_tensor",
  "mark_graph_outputs",
  "record_torch",
  "register_inputs",
  "register_module_state",
  "register_tensor",
  "tensor_leaves",
  "tensor_metadata",
]
