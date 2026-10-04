"""Template compilation orchestration, independent of runtime argument adaptation."""

from __future__ import annotations
import hashlib
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from ..._compiler import Compiler
from ... import _C
from ..._operator import Operator
from ..template import TemplateFunction
from .._runtime.binding import _normalize_out_idx
from .._runtime.kernel import _CompiledCore, CompiledKernel
from .cache import _source_identity, _compiler_key, _compile_lock, _compile_cache
from .backends.cpu.source import _build_definition


def compile(
  template_function: TemplateFunction,
  *,
  out_idx: int | list[int] | tuple[int, ...] | None = None,
  compiler: Compiler | None = None,
  pipeline=None,
  pass_options=None,
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
  if ir.source is None:
    from .backends.cpu.backend import CPUBackend

    artifact = CPUBackend().lower(ir, indices, pipeline=pipeline, options=pass_options, target=selected.target)
    kernel_ir, vector_ir, generated_source = artifact.kernel, artifact.lowered, artifact.source
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
  key = (ir.canonical(), indices, source_identity, _compiler_key(selected))
  digest = hashlib.sha256(repr(key).encode()).hexdigest()
  with _compile_lock:
    core = _compile_cache.get(key)
    if core is None:
      definition, values, order = _build_definition(ir, indices, runtime_parallelism=generated_source is not None)
      recipe = ir.recipe_id or f"{ir.name}:{source_identity[:16]}"
      recipe = f"language-v1:{recipe}:{source_identity[:16]}"
      operation = Operator(ir.source, definition, selected, recipe_id=recipe)
      core = _CompiledCore(operation, definition, indices, values, order, digest)
      _compile_cache[key] = core
  result = CompiledKernel(core, template_function)
  result.source = ir.source
  result.generated_source = generated_source
  result.kernel_ir = kernel_ir
  result.vector_ir = vector_ir
  return result
