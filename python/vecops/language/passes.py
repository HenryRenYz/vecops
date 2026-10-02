"""Public pass-registration and pipeline API; implementation lives in the compiler."""
from ._compiler.passes.manager import (PassResult, PassContext, PassSpec, register_pass,
                                      unregister_pass, registered_passes, PassManager)
from ._compiler.passes.manager import __all__

for _public_symbol in (PassManager, PassContext, PassResult, PassSpec,
                       register_pass, unregister_pass, registered_passes):
  _public_symbol.__module__ = __name__
del _public_symbol
