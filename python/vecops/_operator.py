"""Public lazy runtime operators and ordered source-recipe groups."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import _C
from ._compiler import Compiler
from ._dtype import DType, normalize_dtype
from ._schema import KernelDef, TensorDef, TensorMeta


class Operator:
  """One declarative source kernel with cache-aware lazy artifact resolution.

  This is the lower-level public equivalent of a :class:`JitKernel`: callers
  provide a C++ source path and :class:`KernelDef` directly.  Construction
  fingerprints source and builds native dispatch state, but it does not invoke
  CMake or allocate tensor storage.
  """

  def __init__(
    self,
    source: str | Path,
    kernel_def: KernelDef,
    compiler: Compiler | None = None,
    *,
    recipe_id: str | None = None,
  ) -> None:
    self.source = Path(source).expanduser().resolve()
    self.kernel_def = kernel_def
    self.compiler = compiler or Compiler()
    source_identity = hashlib.sha256(self.source.read_bytes()).hexdigest()[:16]
    namespace = self.compiler.cache_namespace
    self._native = _C.Operator(
      self.source,
      kernel_def._native(),
      self.compiler.native,
      self.compiler.cache_mode,
      self.compiler.cache_dir,
      namespace,
      recipe_id or f"{kernel_def.name}:{source_identity}",
      source_identity,
    )

  @property
  def native(self) -> _C.Operator:
    """Return the expert-level pybind object backing this Python wrapper."""
    return self._native

  def _specializations(self, kwargs: dict[str, Any]) -> dict[str, Any]:
    result = dict(kwargs)
    for name, kind in self.kernel_def.values.items():
      if kind is DType and name in result:
        result[name] = normalize_dtype(result[name])
    return result

  def __call__(self, *args, **kwargs):
    """Bind, resolve, and synchronously invoke this operator."""
    return self._native(*args, **self._specializations(kwargs))

  def compile_for(self, *args, **kwargs) -> None:
    """Prepare a matching executable without requiring tensor storage or executing it."""
    native_arguments = []
    for index, argument in enumerate(args):
      if isinstance(argument, TensorMeta):
        native_arguments.append(argument._native())
      elif index < len(self.kernel_def.parameters) and isinstance(self.kernel_def.parameters[index], TensorDef):
        native_arguments.append(_C.TensorView(argument, writable=self.kernel_def.parameters[index].output))
      else:
        native_arguments.append(argument)
    specializations = self._specializations(kwargs)
    self._native.prepare(_C.KernelCall(native_arguments, specializations))


def _parameter_surface(parameter):
  from ._schema import TensorDef, ValueDef

  if isinstance(parameter, TensorDef):
    return ("tensor", parameter.name, parameter.optional, parameter.access, len(parameter.shape))
  if isinstance(parameter, ValueDef):
    return (
      "value",
      parameter.name,
      parameter.dtype,
      parameter.has_default,
      parameter.default if parameter.has_default else None,
    )
  raise TypeError(f"unsupported parameter {parameter!r}")


class OperatorGroup:
  """Ordered, C++-dispatched group of structurally compatible JIT recipes.

  The first applicable recipe wins. Matching, artifact resolution, and fallback
  on ``NotApplicable``/``NotFound`` happen in C++; Python is not in the hot
  dispatch path. All recipes must expose the same public runtime parameters
  and named specialization vocabulary.
  """

  def __init__(self, name: str, kernels, compiler: Compiler | None = None) -> None:
    kernels = tuple(kernels)
    if not kernels:
      raise ValueError("an operator group requires at least one JIT kernel")
    first_surface = tuple(_parameter_surface(item) for item in kernels[0].kernel_def.parameters)
    first_values = kernels[0].kernel_def.values
    for kernel in kernels[1:]:
      if tuple(_parameter_surface(item) for item in kernel.kernel_def.parameters) != first_surface:
        raise ValueError("ordered JIT recipes must have compatible public parameter structures")
      if kernel.kernel_def.values != first_values:
        raise ValueError("ordered JIT recipes must declare the same specialization values")
    configured = [kernel.compiler for kernel in kernels if kernel.compiler is not None]
    self.compiler = compiler or (configured[0] if configured else Compiler())
    if any(item is not self.compiler for item in configured):
      raise ValueError("ordered JIT recipes must use one shared Compiler")
    self.name = name
    self.kernels = kernels
    self.kernel_def = kernels[0].kernel_def
    sources = [kernel.source for kernel in kernels]
    fingerprints = [hashlib.sha256(source.read_bytes()).hexdigest()[:16] for source in sources]
    definitions = [kernel.kernel_def._native(name=name) for kernel in kernels]
    recipe_ids = [
      kernel.recipe_id or f"{kernel.__module__}.{kernel.__name__}:{digest}"
      for kernel, digest in zip(kernels, fingerprints)
    ]
    defaults = [
      {
        key: normalize_dtype(value) if kernel.kernel_def.values[key] is DType else value
        for key, value in kernel.specialization_defaults.items()
      }
      for kernel in kernels
    ]
    self._native = _C.Operator._group(
      sources,
      definitions,
      self.compiler.native,
      self.compiler.cache_mode,
      self.compiler.cache_dir,
      self.compiler.cache_namespace,
      recipe_ids,
      fingerprints,
      defaults,
    )

  @property
  def native(self):
    """Return the grouped raw native operator used by generated wrappers."""
    return self._native

  def __call__(self, *args, **kwargs):
    """Invoke the first recipe that can serve the supplied metadata."""
    for symbol, kind in self.kernel_def.values.items():
      if symbol in kwargs and kind is DType:
        kwargs[symbol] = normalize_dtype(kwargs[symbol])
    return self._native(*args, **kwargs)


__all__ = ["Operator", "OperatorGroup"]
