"""Template-function annotations for the vecops language frontend."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Union, get_args, get_origin

from .expr import Const, SymbolicInt


def _dimensions(values: Iterable[Any], what: str) -> tuple[SymbolicInt, ...]:
  result = []
  for value in values:
    if isinstance(value, SymbolicInt):
      result.append(value)
    elif isinstance(value, int) and not isinstance(value, bool):
      result.append(Const(value))
    else:
      raise TypeError(f"{what} dimensions must be integers or T symbolic values, got {value!r}")
  return tuple(result)


def dense_strides(shape: tuple[SymbolicInt, ...]) -> tuple[SymbolicInt, ...]:
  if not shape:
    return ()
  result: list[SymbolicInt] = [Const(1) for _ in shape]
  for axis in range(len(shape) - 2, -1, -1):
    result[axis] = result[axis + 1] * shape[axis + 1]
  return tuple(result)


@dataclass(frozen=True)
class TensorType:
  """A tensor ABI annotation with Python-expression shape and strides."""

  shape: tuple[SymbolicInt, ...]
  strides: tuple[SymbolicInt, ...] | None
  dtype: Any
  optional: bool = False
  access: str = "input"

  def __hash__(self):
    """Support typing unions without hashing mutable symbolic metadata.

    Older supported Python versions require hashable Optional/Union members.
    Symbol names (and delayed captured values) can change after decoration,
    so a structural hash would invalidate typing's caches. A stable class
    hash preserves the existing descriptor equality contract; the few tensor
    descriptors in an annotation union can safely share this hash bucket.
    Artifact identity is still derived from TemplateIR's canonical metadata.
    """
    return hash(type(self))

  def __or__(self, other):
    if other is None or other is type(None):
      return replace(self, optional=True)
    return NotImplemented

  def __ror__(self, other):
    return self.__or__(other)

  @property
  def concrete_strides(self) -> tuple[SymbolicInt, ...]:
    # Preserve ``None`` in TemplateIR as the future backend-selected layout
    # marker, but the current C++ source backend lowers it to dense row-major.
    return dense_strides(self.shape) if self.strides is None else self.strides


class Tensor:
  """Construct a template tensor annotation.

  ``strides=None`` currently lowers to dense row-major strides while remaining
  distinguishable in TemplateIR for future backend-selected layouts.
  """

  def __new__(
    cls,
    shape: Iterable[Any],
    strides: Iterable[Any] | None = None,
    dtype: Any = None,
    *,
    access: str = "input",
  ) -> TensorType:
    if access not in {"input", "output", "inout"}:
      raise ValueError("tensor access must be 'input', 'output', or 'inout'")
    shape_value = _dimensions(shape, "shape")
    stride_value = None if strides is None else _dimensions(strides, "stride")
    if stride_value is not None and len(shape_value) != len(stride_value):
      raise ValueError("tensor shape and stride ranks differ")
    return TensorType(shape_value, stride_value, dtype, False, access)


def normalize_tensor_annotation(annotation: Any) -> Any:
  """Lower ``typing.Optional[TensorType]`` to the native TensorType marker."""
  origin = get_origin(annotation)
  if origin is Union:
    arguments = get_args(annotation)
    tensors = [item for item in arguments if isinstance(item, TensorType)]
    if len(arguments) == 2 and len(tensors) == 1 and type(None) in arguments:
      return replace(tensors[0], optional=True)
  return annotation


__all__ = ["Tensor", "TensorType", "dense_strides", "normalize_tensor_annotation"]
