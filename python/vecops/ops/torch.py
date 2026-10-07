# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Torch registration facade for C++-dispatched vecops source recipes.

This module deliberately exposes a callable in ``vecops.ops.torch`` instead
of asking applications to call the generated ``torch.ops`` symbol directly:
the facade accepts named specialization kwargs and restores caller-owned Out
and InOut tensors in its return value.
"""

from __future__ import annotations

from types import SimpleNamespace

from .._jit import JitKernel
from .._operator import Operator, OperatorGroup
from .._schema_bridge import (
  compile_pending_torch_bridges as _compile_pending,
  register_torch_operator as _register,
)

_libraries: dict[str, SimpleNamespace] = {}


def _name(qualified_name: str) -> tuple[str, str]:
  """Validate and split a ``library::operator`` registration name."""
  if qualified_name.count("::") != 1:
    raise ValueError("Torch operator name must use the 'library::name' form")
  library, name = qualified_name.split("::")
  if not library or not name:
    raise ValueError("Torch operator library and name must not be empty")
  return library, name


def register(qualified_name, dispatch, *, compiler=None, **options):
  """Register one Torch CPU/Meta operator backed by ordered JIT recipes.

  ``qualified_name`` must use ``"library::operator"`` form. ``dispatch`` may
  be one recipe, an ordered recipe sequence, or an existing public Operator.
  The returned callable accepts named specialization values unlike raw
  ``torch.ops`` and leaves outputs caller-allocated/mutable. Pass
  ``return_outputs=True`` when an older TorchInductor requires the internal
  mutable operator to return Tensor values rather than ``()``; the public
  callable returns the original output objects in either mode.
  """
  library, name = _name(qualified_name)
  if isinstance(dispatch, JitKernel):
    operation = OperatorGroup(qualified_name, [dispatch], compiler)
  elif isinstance(dispatch, (list, tuple)):
    if not all(isinstance(item, JitKernel) for item in dispatch):
      raise TypeError("ordered dispatch entries must be @vecops.jit kernels")
    operation = OperatorGroup(qualified_name, dispatch, compiler)
  elif isinstance(dispatch, (Operator, OperatorGroup)):
    if compiler is not None:
      raise TypeError("compiler= is only valid when registering JIT recipes")
    operation = dispatch
  else:
    raise TypeError("dispatch must be an Operator, JitKernel, or ordered JitKernel sequence")
  function = _register(
      library,
      name,
      operation.native,
      operation.kernel_def._native(),
      compiler=operation.compiler,
      **options,
  )
  namespace = _libraries.setdefault(library, SimpleNamespace())
  setattr(namespace, name, function)
  return function


def __getattr__(name: str):
  """Return a previously registered library namespace or raise AttributeError."""
  try:
    return _libraries[name]
  except KeyError as error:
    raise AttributeError(name) from error


def compile_pending(*, parallelism: int | None = None) -> int:
  """Compile and register every pending Torch bridge in shared build graphs."""
  return _compile_pending(parallelism=parallelism)


__all__ = ["compile_pending", "register"]
