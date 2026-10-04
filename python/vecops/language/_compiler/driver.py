"""Template compilation orchestration, independent of runtime argument adaptation."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from ... import _C
from ..._compiler import Compiler
from ..._operator import Operator
from .._runtime.binding import _normalize_out_idx
from .._runtime.kernel import CompiledKernel, _CompiledCore
from ..template import TemplateFunction
from .backends.cpu.source import _build_definition
from .cache import _compile_cache, _compile_lock, _compiler_key, _source_identity


def compile(
  template_function: TemplateFunction,
  *,
  out_idx: int | list[int] | tuple[int, ...] | None = None,
  compiler: Compiler | None = None,
  pipeline=None,
  pass_options=None,
  debug=None,
) -> CompiledKernel:
  """Lower a TemplateFunction and reuse its immutable compiled-kernel core."""
  if not isinstance(template_function, TemplateFunction):
    raise TypeError("T.compile expects a T.template-decorated function")
  ir = template_function.ir
  selected = compiler or template_function.compiler or Compiler()
  indices = _normalize_out_idx(out_idx, tuple(ir.signature.parameters.values()))
  generated_source = None
  kernel_ir = None
  vector_ir = None
  backend_ir = None
  from ..._identity import compiler_identity, implementation_identity, include_roots, source_inputs
  from .._debug.report import DebugInfo, lower

  lowered = lower(
    template_function,
    target=selected.target,
    out_idx=out_idx,
    pipeline=pipeline,
    pass_options=pass_options,
    debug=debug,
  )
  if ir.source is None:
    artifact = lowered.artifact
    kernel_ir, vector_ir, generated_source = artifact.kernel, artifact.lowered, artifact.source
    backend_ir = artifact.backend_ir
    identity = hashlib.sha256(generated_source.encode()).hexdigest()
    directory = selected.cache_dir.parent / "dsl-sources" / identity
    source = directory / "kernel.cpp"
    with _compile_lock:
      if selected.cache_mode == _C.ArtifactCacheMode.cache_only:
        if not source.is_file():
          raise ImportError("cache-only DSL execution requires the precompiled dsl-sources artifact")
      else:
        directory.mkdir(parents=True, exist_ok=True)
      if not source.exists():
        descriptor, temporary = tempfile.mkstemp(prefix="kernel-", suffix=".cpp", dir=directory)
        try:
          with os.fdopen(descriptor, "w") as stream:
            stream.write(generated_source)
          os.replace(temporary, source)
        finally:
          Path(temporary).unlink(missing_ok=True)
    ir = replace(ir, source=source)
  source_identity = _source_identity(ir)
  build_identity = compiler_identity(selected)
  input_roots = include_roots(selected)
  input_snapshot = source_inputs(ir.source, input_roots, ir.dependencies)
  key = (ir.canonical(), indices, source_identity, input_snapshot, build_identity, _compiler_key(selected))
  digest = hashlib.sha256(repr(key).encode()).hexdigest()
  with _compile_lock:
    core = _compile_cache.get(key)
    cache_hit = core is not None
    if core is None:
      definition, values, order = _build_definition(ir, indices, runtime_parallelism=generated_source is not None)
      recipe = ir.recipe_id or f"{ir.name}:{source_identity[:16]}"
      recipe = f"language-v1:{recipe}:{source_identity[:16]}"
      operation = Operator(
        ir.source, definition, selected, recipe_id=recipe, dependencies=ir.dependencies, build_identity=build_identity
      )
      core = _CompiledCore(operation, definition, indices, values, order, digest)
      _compile_cache[key] = core
  result = CompiledKernel(core, template_function)
  result.source = ir.source
  result.generated_source = generated_source
  result.kernel_ir = kernel_ir
  result.vector_ir = vector_ir
  result.backend_ir = backend_ir
  result.debug = DebugInfo(
    lowered.debug._session,
    name=ir.name,
    compiler=selected,
    cache={
      "core_hit": cache_hit,
      "key": digest,
      "source_identity": source_identity,
      "build_identity": build_identity,
      "source_path": str(ir.source),
      "reason": "matching immutable core" if cache_hit else "new content/build identity",
    },
  )
  result.debug.owner = result
  result.debug.diagnostics = tuple(kernel_ir.diagnostics) if kernel_ir is not None else ()
  implementation = implementation_identity()
  configuration = _compiler_key(selected)
  result.debug._validate_cache = lambda: (
    implementation_identity() == implementation
    and _compiler_key(selected) == configuration
    and source_inputs(ir.source, input_roots, ir.dependencies) == input_snapshot
  )
  template_function.debug = result.debug
  return result
