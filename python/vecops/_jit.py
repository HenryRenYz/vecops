# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Signature-driven declarations for lazy compiler-managed ``__kernel__`` recipes.

The decorated Python function is a schema declaration, not the implementation:
the actual vector kernel lives in the supplied C++ file and must define
``void __kernel__(...)``.  Decoration validates Python metadata only; artifact
lookup and native compilation happen on a later call or :meth:`compile_for`.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any, Callable

from ._compiler import Compiler
from ._dtype import Const, DType, normalize_dtype
from ._operator import Operator
from ._schema import Dynamic, KernelDef, TensorDef, ValueDef
from .typing import ScalarAnnotation, TensorAnnotation


class JitKernel:
  """A lazy source recipe whose :class:`KernelDef` comes from a signature.

  Calls merge keyword-only specialization defaults with supplied keyword
  values, then forward to a lazily-created :class:`~vecops.Operator`. Positional
  arguments remain in declaration order and include caller-allocated outputs.
  """

  def __init__(
    self,
    function: Callable[..., Any],
    source: Path,
    kernel_def: KernelDef,
    specialization_defaults: dict[str, Any],
    compiler: Compiler | None,
    recipe_id: str | None,
  ) -> None:
    functools.update_wrapper(self, function)
    self.__signature__ = inspect.signature(function)
    self.function = function
    self.source = source
    self.kernel_def = kernel_def
    self.specialization_defaults = specialization_defaults
    self.compiler = compiler
    self.recipe_id = recipe_id
    self._operator: Operator | None = None

  @property
  def operator(self) -> Operator:
    """Return the lazily-created runtime operator; constructing it does not compile."""
    if self._operator is None:
      self._operator = Operator(self.source, self.kernel_def, self.compiler, recipe_id=self.recipe_id)
    return self._operator

  def __call__(self, *args, **kwargs):
    """Resolve and invoke this recipe with runtime arguments and specializations."""
    values = dict(self.specialization_defaults)
    values.update(kwargs)
    return self.operator(*args, **values)

  def compile_for(self, *args, **kwargs) -> None:
    """Bind and prepare an artifact without executing a kernel.

    Tensor arguments may be :class:`vecops.TensorMeta`, allowing offline
    precompilation before caller-owned storage exists.
    """
    values = dict(self.specialization_defaults)
    values.update(kwargs)
    self.operator.compile_for(*args, **values)


def _resolve_source(function: Callable[..., Any], source: str | Path, source_root: str | Path | None) -> Path:
  """Resolve a relative source path against the declaration file or explicit root."""
  path = Path(source).expanduser()
  if path.is_absolute():
    return path.resolve()
  if source_root is not None:
    return (Path(source_root).expanduser() / path).resolve()
  filename = inspect.getsourcefile(function) or function.__code__.co_filename
  if not filename or filename.startswith("<"):
    raise ValueError("relative JIT source requires a file-backed function or an explicit source_root")
  return (Path(filename).resolve().parent / path).resolve()


def _definition(
  function: Callable[..., Any],
  symbols: dict[str, Dynamic] | None = None,
) -> tuple[KernelDef, dict[str, Any]]:
  """Lower an out-style declaration signature into a public kernel definition."""
  signature = inspect.signature(function)
  annotations = inspect.get_annotations(function, eval_str=True)
  return_annotation = annotations.get("return", signature.return_annotation)
  if return_annotation not in {inspect.Signature.empty, None, type(None)}:
    raise TypeError("out-style JIT declarations must return None")
  parameters = []
  specialization_kinds: dict[str, Any] = {}
  specialization_defaults: dict[str, Any] = {}
  for parameter in signature.parameters.values():
    annotation = annotations.get(parameter.name, parameter.annotation)
    if annotation is inspect.Parameter.empty:
      raise TypeError(f"JIT parameter {parameter.name!r} requires an annotation")
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
      if annotation not in {Const, DType}:
        raise TypeError(f"keyword-only JIT parameter {parameter.name!r} must be annotated as vt.Const or vt.DType")
      specialization_kinds[parameter.name] = annotation
      if parameter.default is not inspect.Parameter.empty:
        specialization_defaults[parameter.name] = (
          normalize_dtype(parameter.default) if annotation is DType else parameter.default
        )
      continue
    if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
      raise TypeError("JIT declarations do not support *args or **kwargs")
    if isinstance(annotation, TensorAnnotation):
      if parameter.default is not inspect.Parameter.empty and parameter.default is not None:
        raise TypeError(f"tensor parameter {parameter.name!r} may only default to None")
      optional = annotation.optional or parameter.default is None
      parameters.append(
        TensorDef(
          parameter.name,
          annotation.shape,
          annotation.strides,
          dtype=annotation.dtype,
          optional=optional,
          access=annotation.access,
        )
      )
    else:
      dtype = annotation.dtype if isinstance(annotation, ScalarAnnotation) else annotation
      if parameter.default is inspect.Parameter.empty:
        parameters.append(ValueDef(parameter.name, dtype))
      else:
        parameters.append(ValueDef(parameter.name, dtype, parameter.default))
  name = f"{function.__module__}.{function.__qualname__}"
  return KernelDef(
    name, parameters, specialization_kinds, symbols=symbols
  ), specialization_defaults


def jit(
  source: str | Path,
  *,
  source_root: str | Path | None = None,
  compiler: Compiler | None = None,
  recipe_id: str | None = None,
  symbols: dict[str, Dynamic] | None = None,
):
  """Declare a compiler-managed ``__kernel__`` from a Python signature.

  ``source`` is resolved relative to the decorated function's file unless an
  absolute path or ``source_root`` is given. Positional annotations use
  ``vt.In``/``vt.Out``/``vt.InOut`` or scalar types; keyword-only annotations
  must be ``vt.Const`` or ``vt.DType`` specializations. The declaration must
  return ``None`` because all output is out-style. ``symbols`` may map a
  dimension name to ``vt.Dynamic`` so repeated uses are checked at runtime
  without specializing the artifact on the observed value.
  """

  def decorate(function: Callable[..., Any]) -> JitKernel:
    kernel_def, defaults = _definition(function, symbols)
    resolved = _resolve_source(function, source, source_root)
    if not resolved.is_file():
      raise FileNotFoundError(f"JIT kernel source does not exist: {resolved}")
    return JitKernel(function, resolved, kernel_def, defaults, compiler, recipe_id)

  return decorate


__all__ = ["JitKernel", "jit"]
