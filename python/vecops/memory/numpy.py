# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""NumPy views whose base owner retains a native memory-pool allocation."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._dtype import framework_dtype

if TYPE_CHECKING:
  from collections.abc import Sequence
  import numpy as np
  from . import Pool


def empty(shape: int | Sequence[int], dtype='float32', *, pool: Pool,
          preferred_pool: str | None = None, allow_domain_fallback: bool = True) -> np.ndarray:
  """Allocate a contiguous array from an explicit arena-backed Pool.

  Array views keep the native allocation alive. The final base owner returns
  its pages to the arena; ``pool.trim()`` releases idle growth chunks.
  Framework-only dtypes remain valid storage even if no native kernel supports
  their arithmetic. Object dtypes require Python reference management and are
  deliberately rejected.
  """
  import numpy as np
  normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
  if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in normalized_shape):
    raise ValueError('array dimensions must be nonnegative integers')
  normalized_dtype = framework_dtype(dtype, 'numpy', allow_framework_only=True)
  if normalized_dtype.hasobject:
    raise ValueError('object-containing dtypes cannot use a raw memory pool')
  bytes = math.prod(normalized_shape) * normalized_dtype.itemsize
  if bytes == 0:
    return np.empty(normalized_shape, dtype=normalized_dtype)
  if bytes >= 1 << 64:
    raise OverflowError('array size exceeds uint64 byte range')
  owner = pool.allocate(bytes, preferred_pool=preferred_pool, allow_domain_fallback=allow_domain_fallback)
  return np.ndarray(normalized_shape, dtype=normalized_dtype, buffer=owner)


def zeros(shape: int | Sequence[int], dtype='float32', *, pool: Pool, **options) -> np.ndarray:
  """Allocate zero-initialized storage in the supplied pool."""
  result = empty(shape, dtype, pool=pool, **options)
  result.fill(0)
  return result


__all__ = ['empty', 'zeros']
