# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Unpacked FP32/FP64 MOPA algorithms in inspectable instruction-level DSL."""

from .. import dsl as T
from .. import sme


@T.func
def _matrix_tiles(a, b, output):
  """A bounded ZA tile; retain tail correlations in the source iteration."""
  width = sme.streaming_lanes(a.dtype)
  for mi in T.serial(0, a.shape[0], width, unroll=1):
    for ni in T.serial(0, b.shape[1], width, unroll=1):
      sme.zero_za()
      for ki in T.serial(a.shape[1], unroll=2):
        left = sme.load(a[mi : mi + width, ki], strided=True)
        right = sme.load(b[ki, ni : ni + width], strided=True)
        sme.mopa(0, left, right)
      for row in T.serial(T.min(width, a.shape[0] - mi), unroll=2):
        sme.store_hor(0, row, output[mi + row, ni : ni + width])


@T.func
def _direct_dot(a, b, output):
  """Source-level tile traversal, with one lexical owner."""
  if output.shape[0] > 0 and output.shape[1] > 0:
    with sme.region():
      _matrix_tiles(a, b, output)


@T.func
def direct_dot(a, b, output):
  """Protect overlapping raw producers before entering the one ZA owner.

  Copies are scalar-element storage and only occur on the overlap branch.
  They are alias preservation, not matrix-instruction packing. Neither dot
  family nor cache traversal is chosen in the native instruction printer.
  """
  T.static_assert(a.dtype == b.dtype and b.dtype == output.dtype)
  T.static_assert(a.dtype == "float32" or a.dtype == "float64")
  assert a.shape[1] == b.shape[0], "incompatible SME dot reduction extents"
  assert output.shape[0] == a.shape[0] and output.shape[1] == b.shape[1], "SME dot output shape mismatch"
  if T.view_disjoint(a, output) and T.view_disjoint(b, output):
    _direct_dot(a, b, output)
  else:
    saved_a = T.alloc(a.shape, dtype=a.dtype)
    saved_b = T.alloc(b.shape, dtype=b.dtype)
    T.copy(a, saved_a)
    T.copy(b, saved_b)
    _direct_dot(saved_a, saved_b, output)
