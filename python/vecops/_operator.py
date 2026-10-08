# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Public lazy runtime operators used by the language compiler."""

from __future__ import annotations

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
    dependencies=(),
    build_identity: str | None = None,
  ) -> None:
    self.source = Path(source).expanduser().resolve()
    self.kernel_def = kernel_def
    self.compiler = (compiler or Compiler()).snapshot()
    from ._identity import (
      _source_input_diff,
      _write_build_input_trace,
      compiler_identity,
      digest,
      include_roots,
      source_inputs,
    )

    inputs = source_inputs(
      self.source,
      include_roots(self.compiler),
      dependencies,
      definition_cwd=self.compiler.definition_cwd,
      inventory_excludes=self.compiler._source_include_inventory_excludes,
    )
    source_identity = digest(inputs)
    if build_identity is None:
      try:
        build_identity = compiler_identity(self.compiler)
      except ImportError:
        if self.compiler.cache_mode != _C.ArtifactCacheMode.cache_only:
          raise
        # Preserve lazy construction in runtime-only installations. This
        # namespace cannot match any attested artifact, so invocation misses.
        build_identity = "unattested-build-inputs"
    namespace = f"{self.compiler.cache_namespace};inputs={build_identity}"
    # Consulted only by the native compiler on a cache miss, never by a
    # prepared invocation's execution path.
    source_path, include_dirs, selected = self.source, include_roots(self.compiler), self.compiler

    def validate_inputs():
      # The operator owns an immutable VL/SVL snapshot and compile definitions.
      # A batch may contain several such snapshots, and can build on a thread
      # with another current length. Validate content/policy, not that worker's
      # mutable ISA state. The generated entry adapter checks invocation state.
      current_inputs = source_inputs(
        source_path,
        include_dirs,
        dependencies,
        definition_cwd=selected.definition_cwd,
        inventory_excludes=selected._source_include_inventory_excludes,
      )
      source_equal = current_inputs == inputs
      current_identity = None
      compiler_equal = None
      if source_equal:
        current_identity = compiler_identity(selected)
        compiler_equal = current_identity == build_identity
      if not source_equal or compiler_equal is False:
        _write_build_input_trace(
          {
            "event": "operator_input_validation",
            "source": str(source_path),
            "source_inputs_equal": source_equal,
            "compiler_identity_equal": compiler_equal,
            "expected_compiler_identity": build_identity if source_equal else None,
            "current_compiler_identity": current_identity,
            "source_diff": _source_input_diff(inputs, current_inputs)
            if not source_equal
            else {"changed_count": 0, "paths": []},
          },
          compiler=selected,
        )
      return bool(source_equal and compiler_equal)

    self._native = _C.Operator(
      self.source,
      kernel_def._native(),
      self.compiler.native,
      self.compiler.cache_mode,
      self.compiler.cache_dir,
      namespace,
      recipe_id or f"{kernel_def.name}:{source_identity}",
      source_identity,
      validate_inputs,
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
