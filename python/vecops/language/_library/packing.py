# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Portable packed-ABI construction; ISA-specific raw recipes specialize it."""

from .. import dsl as T


@T.func
def pack_generic(source, destination, *, side, format):
  """Zero padding before writing logical elements, never evaluate padded DAGs.

  Physical ABI offsets are metadata primitives. Traversal and input
  transformations are source DSL. This conservative route deliberately
  favors correctness for unsupported ISA/type/layout combinations; optimized
  register/ZA recipes can replace its schedule without a native loop printer.
  The checked adapter supplies exactly the required physical prefix.
  """
  source = T.as_value(source)
  if format == "dense":
    if side == "A":
      T.store(T.reshape(destination, source.shape), source)
    else:
      T.store(T.reshape(destination, (source.shape[1], source.shape[0])), T.transpose(source))
  else:
    T.store(destination, T.zeros(destination.shape, dtype=destination.dtype))
    spatial = source.shape[0] if side == "A" else source.shape[1]
    reduction = source.shape[1] if side == "A" else source.shape[0]
    for s in T.serial(spatial, unroll=1):
      for k in T.serial(reduction, unroll=2):
        offset = T._packing_offset(s, k, spatial, reduction, destination.dtype, format, side)
        if side == "A":
          value = T.element_at(source, (s, k))
        else:
          value = T.element_at(source, (k, s))
        T.store(destination[offset], value)
