"""NumPy arrays backed by :mod:`vecops.memory` allocations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Sequence

  import numpy as np

  from . import System


def empty(
  shape: int | Sequence[int],
  dtype="float32",
  *,
  system: System | None = None,
  **allocation_options,
) -> np.ndarray:
  """Create a contiguous uninitialized ndarray in selected CPU memory."""
  import numpy as np

  from . import default_system

  normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
  normalized_dtype = np.dtype(dtype)
  if any(dimension < 0 for dimension in normalized_shape):
    raise ValueError("shape dimensions must be non-negative")
  count = math.prod(normalized_shape)
  owner = (system or default_system()).allocate(
    count * normalized_dtype.itemsize,
    **allocation_options,
  )
  return np.ndarray(normalized_shape, dtype=normalized_dtype, buffer=owner)


def zeros(
  shape: int | Sequence[int],
  dtype="float32",
  *,
  system: System | None = None,
  **allocation_options,
) -> np.ndarray:
  """Create a contiguous zero-initialized ndarray in selected CPU memory."""
  result = empty(shape, dtype, system=system, **allocation_options)
  result.fill(0)
  return result


__all__ = ["empty", "zeros"]
