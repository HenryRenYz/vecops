"""Template declaration and independent source/DSL capture."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from .expr import Const, DType, Dynamic, IntExpr
from .types import TensorType, normalize_tensor_annotation


def _bind_names(value: Any, name: str) -> None:
  if isinstance(value, (Const, Dynamic, DType)):
    value.bind_name(name)
  elif isinstance(value, tuple):
    for index, item in enumerate(value):
      _bind_names(item, f"{name}_{index}")
  elif isinstance(value, dict):
    for key, item in value.items():
      _bind_names(item, str(key))


def _expression_leaves(value: Any):
  if isinstance(value, IntExpr):
    yield from _expression_leaves(value.left)
    yield from _expression_leaves(value.right)
  elif isinstance(value, Dynamic):
    yield value
    for constraint in (value.alignment, value.lower, value.upper):
      if isinstance(constraint, Const):
        yield constraint
  elif isinstance(value, (Const, DType)):
    yield value


def _tensor_leaves(tensor: TensorType):
  for dimension in (*tensor.shape, *tensor.concrete_strides):
    yield from _expression_leaves(dimension)
  if isinstance(tensor.dtype, DType):
    yield tensor.dtype


def _evaluate_annotations(function: Callable[..., Any], local_scope: Mapping[str, Any]) -> dict[str, Any]:
  result = {}
  namespace = dict(function.__globals__)
  namespace.update(local_scope)
  for name, value in function.__annotations__.items():
    evaluated = eval(value, namespace, namespace) if isinstance(value, str) else value
    result[name] = normalize_tensor_annotation(evaluated)
  return result


def _resolve_source(
  function: Callable[..., Any], source: str | Path | None, source_root: str | Path | None
) -> Path | None:
  if source is None:
    return None
  path = Path(source).expanduser()
  if path.is_absolute():
    return path.resolve()
  if source_root is not None:
    return (Path(source_root).expanduser() / path).resolve()
  filename = inspect.getsourcefile(function) or function.__code__.co_filename
  if not filename or filename.startswith("<"):
    raise ValueError("relative template source requires a file-backed function or source_root")
  return (Path(filename).resolve().parent / path).resolve()


@dataclass(frozen=True)
class TemplateIR:
  """Immutable Python frontend description before output/access lowering."""

  name: str
  signature: inspect.Signature
  annotations: Mapping[str, Any]
  captures: tuple[Any, ...]
  source: Path | None
  dependencies: tuple[Path, ...]
  recipe_id: str | None
  origin: str
  body: Any = None

  def canonical(self):
    parameters = []
    for parameter in self.signature.parameters.values():
      annotation = self.annotations.get(parameter.name, parameter.annotation)
      if isinstance(annotation, TensorType):
        parameters.append(
          (
            parameter.name,
            "tensor",
            tuple(item.canonical() for item in annotation.shape),
            None if annotation.strides is None else tuple(item.canonical() for item in annotation.strides),
            _canonical_value(annotation.dtype),
            annotation.optional,
            annotation.access,
            parameter.kind.name,
          )
        )
      else:
        parameters.append((parameter.name, "scalar", repr(annotation), repr(parameter.default), parameter.kind.name))
    return (
      self.name,
      tuple(parameters),
      tuple(_canonical_value(value) for value in self.captures),
      str(self.source) if self.source else None,
      tuple(map(str, self.dependencies)),
      self.recipe_id,
      self.body.identity if self.body is not None else None,
    )


def _canonical_value(value: Any):
  if hasattr(value, "canonical"):
    return value.canonical()
  if value is None or isinstance(value, (str, bytes, float, bool, int)):
    return (type(value).__name__, value)
  return (type(value).__module__, type(value).__qualname__, repr(value))


class TemplateFunction:
  """A captured kernel declaration consumed by :func:`vecops.language.compile`."""

  def __init__(self, function: Callable[..., Any], ir: TemplateIR, compiler: Any = None) -> None:
    functools.update_wrapper(self, function)
    self.function = function
    self.ir = ir
    self.compiler = compiler
    self.__signature__ = ir.signature

  def __call__(self, *args, **kwargs):
    raise TypeError("TemplateFunction must be passed to T.compile or returned from a T.jit function")


def template(
  function: Callable[..., Any] | None = None,
  *,
  source: str | Path | None = None,
  source_root: str | Path | None = None,
  compiler: Any = None,
  recipe_id: str | None = None,
  dependencies=(),
):
  """Capture a kernel without executing it: C++ source or source-parsed DSL."""
  caller = inspect.currentframe().f_back
  local_scope = dict(caller.f_locals) if caller is not None else {}

  def decorate(target: Callable[..., Any]) -> TemplateFunction:
    for name, value in local_scope.items():
      _bind_names(value, name)
    annotations = _evaluate_annotations(target, local_scope)
    signature = inspect.signature(target)
    captures: list[Any] = []
    seen: set[int] = set()

    def add(value: Any) -> None:
      if isinstance(value, IntExpr):
        for leaf in _expression_leaves(value):
          add(leaf)
        return
      if isinstance(value, tuple):
        for item in value:
          add(item)
        return
      if isinstance(value, dict):
        for item in value.values():
          add(item)
        return
      if not isinstance(value, (Const, Dynamic, DType)):
        return
      identity = id(value)
      if identity not in seen:
        seen.add(identity)
        captures.append(value)

    for parameter in signature.parameters.values():
      annotation = annotations.get(parameter.name, parameter.annotation)
      if isinstance(annotation, TensorType):
        for value in _tensor_leaves(annotation):
          add(value)

    closure = inspect.getclosurevars(target)
    for name in target.__code__.co_freevars:
      if name in closure.nonlocals:
        value = closure.nonlocals[name]
        _bind_names(value, name)
        add(value)
    for name, value in closure.globals.items():
      _bind_names(value, name)
      add(value)

    resolved = _resolve_source(target, source, source_root)
    if resolved is not None and not resolved.is_file():
      raise FileNotFoundError(f"template kernel source does not exist: {resolved}")
    dependency_paths = tuple(
      (resolved.parent / Path(item)).resolve()
      if resolved is not None and not Path(item).is_absolute()
      else Path(item).expanduser().resolve()
      for item in dependencies
    )
    missing = [path for path in dependency_paths if not path.is_file()]
    if missing:
      raise FileNotFoundError(f"template dependency does not exist: {missing[0]}")
    origin = inspect.getsourcefile(target) or target.__code__.co_filename
    body = None
    if resolved is None:
      from ._compiler.frontend.capture import capture

      body = capture(target, local_scope)
    ir = TemplateIR(
      f"{target.__module__}.{target.__qualname__}",
      signature,
      MappingProxyType(annotations),
      tuple(captures),
      resolved,
      dependency_paths,
      recipe_id,
      origin,
      body,
    )
    return TemplateFunction(target, ir, compiler)

  if function is not None:
    return decorate(function)
  return decorate


__all__ = ["TemplateFunction", "TemplateIR", "template"]
