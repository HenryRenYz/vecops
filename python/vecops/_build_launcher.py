# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Standalone CMake launcher: one cross-process budget for compile and link.

This file runs without importing vecops, Torch or the native extension. File
locks are also inherited by the compiler, so killing a waiting Python parent
cannot release a live child's budget slot prematurely.

An explicit VECOPS_BUILD_CPU_LIST distributes held slots over CPUs already in
this launcher's affinity grant. Without that list the inherited mask is kept.
"""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def compiler_cpus(allowed_cpus):
  """Validate the explicit pool without widening the launcher's current grant."""
  configured = os.environ.get("VECOPS_BUILD_CPU_LIST")
  if configured is None:
    return None
  if allowed_cpus is None:
    raise ValueError("VECOPS_BUILD_CPU_LIST requires Linux CPU affinity support")
  allowed = set(allowed_cpus)
  cpus = set()
  for item in configured.split(','):
    ends = item.strip().split('-')
    if len(ends) not in (1, 2) or any(not end.isascii() or not end.isdigit() for end in ends):
      raise ValueError("invalid VECOPS_BUILD_CPU_LIST")
    first, last = int(ends[0]), int(ends[-1])
    # Bound expansion before constructing a range, including malformed huge
    # ranges. A valid range cannot contain more CPUs than the whole grant.
    if last < first or last - first + 1 > len(allowed):
      raise ValueError("VECOPS_BUILD_CPU_LIST contains CPUs outside the launcher affinity grant")
    cpus.update(range(first, last + 1))
  if not cpus or not cpus <= allowed:
    raise ValueError("VECOPS_BUILD_CPU_LIST contains CPUs outside the launcher affinity grant")
  return sorted(cpus)


def main(arguments):
  route, *command = arguments
  budget = json.loads(Path(route).read_text())
  if budget['root'] is None:
    return subprocess.call(command)
  root, jobs = Path(budget['root']), int(budget['jobs'])
  if jobs < 1 or not command or not root.is_dir():
    raise ValueError("invalid or expired vecops compiler budget")
  allowed_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
  cpus = compiler_cpus(allowed_cpus)
  started = time.perf_counter_ns()
  descriptor = None
  while descriptor is None:
    for offset in range(jobs):
      slot = (os.getpid() + offset) % jobs
      candidate = os.open(root / f"slot-{slot}", os.O_CREAT | os.O_RDWR, 0o600)
      try:
        fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
      except BlockingIOError:
        os.close(candidate)
      else:
        descriptor = candidate
        break
    if descriptor is None:
      time.sleep(0.01)
  acquired = time.perf_counter_ns()
  assigned_cpu = cpus[slot % len(cpus)] if cpus is not None else None
  bound_cpus = child_cpus = child_pid = child_affinity_error = None
  try:
    if assigned_cpu is not None:
      os.sched_setaffinity(0, {assigned_cpu})
    bound_cpus = sorted(os.sched_getaffinity(0)) if allowed_cpus is not None else None
    child = subprocess.Popen(command, pass_fds=(descriptor,))
    child_pid = child.pid
    try:
      child_cpus = sorted(os.sched_getaffinity(child.pid)) if allowed_cpus is not None else None
    except OSError as error:
      # A short-lived command may have exited already. This is an observation
      # at spawn, not a promise that an arbitrary compiler never repins itself.
      child_affinity_error = str(error)
    status = child.wait()
  finally:
    ended = time.perf_counter_ns()
    try:
      # Each launcher owns one file; no trace lock on the compiler hot path.
      (root / f"command-{os.getpid()}.json").write_text(json.dumps({
        "pid": os.getpid(), "slot": slot, "queued_ns": started,
        "started_ns": acquired, "ended_ns": ended,
        "compiler": Path(command[0]).name,
        "cpus": allowed_cpus, "allowed_cpus": allowed_cpus,
        "requested_cpus": cpus, "assigned_cpu": assigned_cpu,
        "bound_cpus": bound_cpus, "child_pid": child_pid,
        "child_cpus_after_spawn": child_cpus,
        "child_affinity_verified": child_cpus == bound_cpus if child_cpus is not None else None,
        "child_affinity_error": child_affinity_error,
        "source": next((item for item in command if item.endswith((".cpp", ".cc", ".c"))), None),
        "status": locals().get("status"),
      }) + "\n")
    finally:
      os.close(descriptor)
  return status if status >= 0 else 128 - status


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
