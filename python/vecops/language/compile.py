"""Public template-compilation façade; lowering and runtime are separate packages."""

from ._compiler.driver import compile
from ._runtime.kernel import CompiledKernel

# Preserve public import identities for introspection/pickling after relocation.
compile.__module__ = __name__
CompiledKernel.__module__ = __name__


def compile_pending(*, parallelism: int | None = None) -> int:
  """Build pending Torch template bridges together before model execution.

  Template compilation declares bridges lazily. This hook materializes their
  shared build graph; source-kernel specialization requests still use
  :func:`vecops.compile_batch` or :func:`vecops.precompile`.
  """
  from .._schema_bridge import compile_pending_torch_bridges

  return compile_pending_torch_bridges(parallelism=parallelism)


__all__ = ["CompiledKernel", "compile", "compile_pending"]
