# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Serialize native timings on one CPU while allowing parallel compilation."""
import fcntl
import math
import os
import time
from pathlib import Path


NATIVE_WARMUP_SECONDS = 0.1


def warmup_native(calls, *, seconds=NATIVE_WARMUP_SECONDS):
  """Warm every native path on its reserved CPUs before collecting samples.

  One tiny invocation may finish before an idle powersave CPU changes its
  performance state. Round-robin calls keep compared paths under the same
  warmup workload without changing the governor or the measured repeat count.
  Call this after affinity, buffers and thread budgets are established.
  """
  paths = (calls,) if callable(calls) else tuple(calls)
  if not paths or not all(callable(path) for path in paths):
    raise ValueError("native warmup requires at least one callable")
  if not math.isfinite(seconds) or seconds < 0:
    raise ValueError("native warmup duration must be finite and nonnegative")
  begin = time.perf_counter_ns()
  rounds = 0
  while True:
    for path in paths:
      path()
    rounds += 1
    elapsed = time.perf_counter_ns() - begin
    if elapsed >= seconds * 1e9:
      return {"protocol": "native-round-robin-v1", "minimum_seconds": seconds,
              "elapsed_ns": elapsed, "rounds": rounds, "paths": len(paths)}


def isolate_compilation(cpu):
  """Keep this benchmark's build workers off the timing core and SMT siblings.

  Restrict only the current process (and its later children), never the CPU
  governor or other users' jobs. Native timing explicitly pins to cpu later.
  """
  excluded={cpu}
  path=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
  try:
    for entry in path.read_text().strip().split(','):
      endpoints=entry.split('-')
      excluded.update(range(int(endpoints[0]),int(endpoints[-1])+1))
  except OSError:
    pass
  allowed=os.sched_getaffinity(0)-excluded
  if allowed:
    os.sched_setaffinity(0,allowed)


def reserve_cpu(cpu):
  """Return a live file lease; keep it through timing, then close it.

  A separate process may compile concurrently but cannot run paired native
  benchmark loops on the same core. Closing/unwinding releases the lease.
  """
  directory=Path(__file__).resolve().parents[1]/'build-dsl-benchmarks'
  directory.mkdir(parents=True,exist_ok=True)
  lease=(directory/f'timing-cpu{cpu}.lock').open('a')
  fcntl.flock(lease,fcntl.LOCK_EX)
  return lease
