# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Scoped controls for the optional process-local native thread pool."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Generator

from . import _C


@contextmanager
def cpu_thread_limit(threads: int) -> Generator[None]:
  """Bound direct CPU/NumPy grid execution on the calling Python thread.

  Zero restores automatic selection (environment, affinity, physical cores
  and cgroup quota). Nested scopes restore the previous budget. This does not
  change Torch's pool, process affinity or the legacy fixed native provider.
  Multiprocessing callers should use spawn/forkserver and partition CPU sets
  or provide per-process budgets; no cross-process scheduler is implied.
  """
  if isinstance(threads, bool) or not isinstance(threads, int):
    raise TypeError("CPU thread limit must be an integer")
  if not 0 <= threads <= 0xFFFFFFFF:
    raise ValueError("CPU thread limit must fit a nonnegative uint32")
  previous = _C.cpu_thread_limit(threads)
  try:
    yield
  finally:
    _C.cpu_thread_limit(previous)


def cpu_thread_pool_capacity() -> int:
  """Return the current direct CPU executor budget without creating workers."""
  return int(_C.cpu_thread_pool_capacity())


@contextmanager
def native_thread_pool_region(
  threads: int,
  *,
  enabled: bool = True,
) -> Generator[None]:
  """Keep native workers ready across several short vecops submissions.

  This region is useful around a fused model block that exclusively dispatches
  to the native vecops pool.  Do not leave it active around parallel work owned
  by Torch/OpenMP, because both teams would then compete for the same CPUs.
  """
  if not enabled:
    yield
    return
  threads = int(threads)
  if threads <= 0:
    raise ValueError("native thread-pool region requires a positive thread count")
  _C.native_thread_pool_begin_active(threads)
  try:
    yield
  finally:
    _C.native_thread_pool_end_active()
