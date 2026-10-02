"""Public lazy runtime operators used by the language compiler."""

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

  The language compiler or expert callers provide a C++ source path and
  :class:`KernelDef` directly. Construction
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
    specializations = self._specializations(kwargs)
    from ._precompile import maybe_collect_call

    if maybe_collect_call(
      self._native,
      self.kernel_def._native(),
      args,
      specializations,
      name=self.kernel_def.name,
    ):
      outputs = tuple(
        argument
        for argument, parameter in zip(args, self.kernel_def.parameters)
        if isinstance(parameter, TensorDef) and parameter.output
      )
      return outputs[0] if len(outputs) == 1 else outputs
    return self._native(*args, **specializations)

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


__all__ = ["Operator"]
