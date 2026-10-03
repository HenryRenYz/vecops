"""Serialize native timings on one CPU while allowing parallel compilation."""
import fcntl
import os
from pathlib import Path


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
