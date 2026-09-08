"""NumPy registration facade for C++-dispatched vecops source recipes.

Registered operations retain out-style tensor semantics and are retrieved from
``vecops.ops.numpy.<library>`` rather than through a global NumPy registry.
"""

from __future__ import annotations

from types import SimpleNamespace

from .._jit import JitKernel
from .._operator import Operator, OperatorGroup

_libraries: dict[str, SimpleNamespace] = {}


def register(qualified_name, dispatch, *, compiler=None):
  """Register a NumPy-facing callable backed by ordered JIT recipes.

  ``qualified_name`` follows ``"library::operator"``. ``dispatch`` can be a
  JitKernel, an ordered JitKernel sequence, or a public Operator. The returned
  callable exposes normal named specialization keyword arguments.
  """
  if qualified_name.count("::") != 1:
    raise ValueError("NumPy operator name must use the 'library::name' form")
  library, name = qualified_name.split("::")
  if isinstance(dispatch, JitKernel):
    operation = OperatorGroup(qualified_name, [dispatch], compiler)
  elif isinstance(dispatch, (list, tuple)):
    if not all(isinstance(item, JitKernel) for item in dispatch):
      raise TypeError("ordered dispatch entries must be @vecops.jit kernels")
    operation = OperatorGroup(qualified_name, dispatch, compiler)
  elif isinstance(dispatch, (Operator, OperatorGroup)):
    operation = dispatch
  else:
    raise TypeError("dispatch must be an Operator, JitKernel, or ordered JitKernel sequence")
  namespace = _libraries.setdefault(library, SimpleNamespace())
  setattr(namespace, name, operation)
  return operation


def __getattr__(name: str):
  """Return a previously registered library namespace or raise AttributeError."""
  try:
    return _libraries[name]
  except KeyError as error:
    raise AttributeError(name) from error


__all__ = ["register"]
