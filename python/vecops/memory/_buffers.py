# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Common prepare-time buffer planning and storage-owned heterogeneous leases.

A reuse group is a caller assertion of serialized execution, not a memory tier
or an allocation category. Graph and kernel live ranges must use different
reuse groups when they coexist. All groups compete for the same capacities.
"""
from __future__ import annotations

import math
import weakref
import threading
from .. import _C

from ..planner.buffers import BufferRequirement, _requests, _current, current_session, plan_buffers


class BufferSession:
  """One bounded tensor/graph/workspace capacity authority.

  All native binding and allocation callbacks execute during preparation.
  Storage-owned leases may escape this context; their mappings survive until
  the final alias dies. Named reuse groups require synchronous, non-overlapping
  execution for the entire lifetime of every prepared user of that group.
  A session is bound to its construction/execution thread; independent threads
  use independent sessions. Storage aliases may be released on any thread.
  """
  def __init__(self, *, fast_capacity, slow_capacity, system=None, domain=None,
               allow_fast_fallback=False, large_pages=True):
    from . import default_system
    self._thread_id = threading.get_ident()
    self.system = system or default_system()
    self.domain = self.system.current_cpu_domain() if domain is None else int(domain)
    self._native = _C.BufferMemorySession(self.system._native, int(fast_capacity), int(slow_capacity),
                                         self.domain, bool(allow_fast_fallback), bool(large_pages))
    self._token = None
    self._closed = False
    self._allocated = 0

  @property
  def thread_id(self):
    return self._thread_id

  def _check_thread(self):
    if threading.get_ident() != self._thread_id:
      raise RuntimeError('BufferSession must be used on its preparation/execution thread')

  def __enter__(self):
    self._check_thread()
    if current_session() is not None:
      raise RuntimeError('buffer sessions cannot be nested')
    self._native.__enter__()
    self._token = _current.set(self)
    return self

  def __exit__(self, *exc):
    if current_session() is not self:
      raise RuntimeError('buffer sessions must exit in LIFO order')
    _current.reset(self._token)
    self._token = None
    self.close()
    return False

  def close(self):
    self._check_thread()
    if self._token is not None:
      if current_session() is not self:
        raise RuntimeError('buffer sessions must close in LIFO order')
      _current.reset(self._token)
      self._token = None
    self._native.close()
    self._closed = True

  def bind(self, requirements, *, reuse_group=''):
    """Bind individually placed buffers; no aggregate all-fast/all-slow fallback."""
    self._check_thread()
    return self._native.bind(_requests(requirements), str(reuse_group))

  def allocate(self, size, *, placement='high_bandwidth', fallback='default', **options):
    """MemorySystem-compatible owning allocation from the common budget.

    This is an alias-group lease, not a new physical arena. Physical domain and
    page policy are fixed when the session opens and cannot change per lease.
    """
    if placement not in {'default', 'high_bandwidth'}:
      raise ValueError('buffer session supports default and high_bandwidth placement')
    if options.get('domain') not in {None, self.domain}:
      raise ValueError('buffer session CPU domain is fixed at preparation')
    if options.get('os_numa_id') is not None or options.get('tier') is not None:
      raise ValueError('buffer session target is fixed at preparation')
    if fallback not in {None, 'none', 'default'}:
      raise ValueError('fallback must be none or default')
    policy = 'slow' if placement == 'default' else 'required' if fallback in {None, 'none'} else 'preferred'
    self._allocated += 1
    return self.bind([BufferRequirement(f'lease-{self._allocated}', int(size),
                     alignment=options.get('alignment', 64), placement=policy)])[f'lease-{self._allocated}']

  def stats(self):
    return self._native.stats()

  def manifest(self):
    """Cold placement evidence, including individual native workspace buffers."""
    return self._native.manifest()

  def current_cpu_domain(self):
    return self.domain

  def tensor_pool(self, capacity):
    return _TensorLeases(self, capacity)


class _TensorLeases:
  """Factory adapter; storage ownership and placement use BufferSession."""
  def __init__(self, session, capacity):
    self.session = session
    self.capacity = 0  # No private reservation: leases share the common budget.
    self.live_bytes = 0
    self.slow_live_bytes = 0
    self.allocations = self.misses = 0
    self.peak_bytes = 0

  def capture_arenas(self):
    # Each Storage owns its own lease. Captured Tensor references already pin
    # those leases; there is no mutable, session-wide factory arena to isolate.
    return ()

  def _release(self, size, tier):
    if tier == 'fast': self.live_bytes -= size
    else: self.slow_live_bytes -= size

  def empty(self, shape, dtype, strides=None):
    import torch
    shape = tuple(int(x) for x in shape)
    if any(x < 0 for x in shape):
      raise ValueError('negative tensor shape')
    n = math.prod(shape)
    if n == 0:
      return torch.empty(shape, dtype=dtype)
    if strides is not None:
      strides = tuple(int(x) for x in strides)
      if len(shape) != len(strides) or any(x < 0 for x in strides):
        raise ValueError('invalid tensor strides')
      n = 1 + sum((d-1)*s for d,s in zip(shape,strides))
    owner = self.session.allocate(n*dtype.itemsize)
    buffer = memoryview(owner)
    if owner.tier == 'fast': self.live_bytes += owner.size
    else:
      self.slow_live_bytes += owner.size
      self.misses += 1
    weakref.finalize(buffer, self._release, owner.size, owner.tier)
    value = torch.frombuffer(buffer, dtype=dtype, count=n)
    self.allocations += 1
    self.peak_bytes = max(self.peak_bytes, self.live_bytes)
    return value.reshape(shape) if strides is None else value.as_strided(shape, strides)
