# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bounded eager allocation with storage-owned leases, including async users."""

from __future__ import annotations

import bisect
import math
import threading
import weakref

class TensorPool:
  """Suballocate one owning buffer without relying on captured lifetimes.

  Torch's frombuffer retains the memoryview until the last storage alias
  dies. Its finalizer returns the range to this pool; views, cached tensors,
  and native asynchronous work retaining a Tensor all keep the lease alive.
  The finalizer also retains the pool and its physical owner after a session
  closes, so outputs can safely escape the session.
  """

  def __init__(self, owner, capacity: int, alignment: int = 64):
    self.owner = owner
    self._buffer = memoryview(owner).cast("B")
    self.capacity = capacity
    self.alignment = max(64, alignment)
    self._free = [(0, capacity)]
    self._lock = threading.RLock()
    self.live_bytes = 0
    self.peak_bytes = 0
    self.allocations = 0
    self.misses = 0

  def capture_arenas(self):
    """Mutable backing arenas that executable capture must isolate."""
    return ((self.owner, self.capacity),)

  def _release(self, offset: int, size: int):
    with self._lock:
      self.live_bytes -= size
      # Only the immediate neighbours can coalesce with the returned range.
      # Keep address order instead of sorting and rebuilding every free span.
      index = bisect.bisect_left(self._free, (offset, 0))
      if index and self._free[index - 1][0] + self._free[index - 1][1] == offset:
        begin, length = self._free.pop(index - 1)
        offset = begin
        size += length
        index -= 1
      if index < len(self._free) and offset + size == self._free[index][0]:
        _, length = self._free.pop(index)
        size += length
      self._free.insert(index, (offset, size))

  def empty(self, shape, dtype, strides=None):
    import torch

    shape = tuple(int(dim) for dim in shape)
    elements = math.prod(shape)
    if elements <= 0 or any(dim < 0 for dim in shape):
      return None
    itemsize = dtype.itemsize
    size = elements * itemsize
    reserved = (size + self.alignment - 1) // self.alignment * self.alignment
    with self._lock:
      for index, (offset, length) in enumerate(self._free):
        if length >= reserved:
          if length == reserved:
            self._free.pop(index)
          else:
            self._free[index] = (offset + reserved, length - reserved)
          self.live_bytes += reserved
          self.peak_bytes = max(self.peak_bytes, self.live_bytes)
          self.allocations += 1
          break
      else:
        self.misses += 1
        return None
    try:
      buffer = self._buffer[offset:offset + size]
    except Exception:
      self._release(offset, reserved)
      raise
    weakref.finalize(buffer, self._release, offset, reserved)
    tensor = torch.frombuffer(buffer, dtype=dtype, count=elements)
    return tensor.reshape(shape) if strides is None else tensor.as_strided(shape, strides)
