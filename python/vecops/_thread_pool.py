"""Scoped controls for the optional process-local native thread pool."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Generator

from . import _C


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
