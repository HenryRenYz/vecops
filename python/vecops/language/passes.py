"""Public pass-registration and pipeline API; implementation lives in the compiler."""

from ._compiler.passes.manager import (
  PassContext,
  PassManager,
  PassResult,
  PassSpec,
  register_pass,
  registered_passes,
  unregister_pass,
)

for _public_symbol in (
  PassManager,
  PassContext,
  PassResult,
  PassSpec,
  register_pass,
  unregister_pass,
  registered_passes,
):
  _public_symbol.__module__ = __name__
del _public_symbol
