# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compilation policy and content snapshots shared by preparation frontiers.

This context is deliberately separate from compile-request collection: real
graph capture can execute bootstrap kernels while later requests are planned.
Independent frontiers may overlap. Compiler/linker launchers share one
cross-process budget, including the graph executor. No Tensor or executable
binding is cached here.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, copy_context
from pathlib import Path

_current = ContextVar("vecops_build_session", default=None)
_build_lock = threading.RLock()
_process_session_guard = threading.Lock()
_process_route_path = None
_route_lock = threading.Lock()


def _process_route():
  # Compilers may outlive a BuildSession. Their launcher must point to this
  # durable, process-local route, never a session's expiring token directory.
  global _process_route_path
  with _route_lock:
    if _process_route_path is None or _process_route_path.name != f"process-{os.getpid()}.json":
      base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vecops/build-routes"
      base.mkdir(parents=True, exist_ok=True)
      _process_route_path = base / f"process-{os.getpid()}.json"
      _process_route_path.write_text('{"root": null}')
  return _process_route_path


def compiler_launcher_command():
  return [sys.executable, str(Path(__file__).with_name("_build_launcher.py")), str(_process_route())]


def compiler_launcher_environment(environment=None):
  if not sys.platform.startswith("linux"):
    return {}
  command = ";".join(compiler_launcher_command())
  result = {}
  for language in ("C", "CXX"):
    for phase in ("COMPILER", "LINKER"):
      name = f"CMAKE_{language}_{phase}_LAUNCHER"
      existing = (environment or {}).get(name, os.environ.get(name, ""))
      result[name] = command + (";" + existing if existing else "")
  return result


def current_build_session():
  return _current.get()


def build_parallelism(explicit=None):
  """Resolve build jobs from explicit policy, independently of worker pinning."""
  session = current_build_session()
  if explicit is None and session is not None:
    return session.parallelism
  if explicit is None:
    cpus = os.environ.get("VECOPS_BUILD_CPU_LIST")
    if cpus:
      available = set()
      for part in cpus.split(","):
        ends = part.strip().split("-")
        first, last = int(ends[0]), int(ends[-1])
        if first < 0 or last < first:
          raise ValueError("invalid VECOPS_BUILD_CPU_LIST")
        available.update(range(first, last + 1))
      explicit = min(8, len(available))
    else:
      explicit = min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
  if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
    raise ValueError("build parallelism must be a positive integer")
  if session is not None and explicit > session.parallelism:
    raise ValueError("build parallelism exceeds the active session budget")
  return explicit


def snapshot_value(kind, key, compute, *, leaf=False):
  session = current_build_session()
  if session is None:
    return compute()
  with session._snapshot_lock:
    if session._validating:
      return compute()
    return session._value(kind, key, compute, leaf)


@contextmanager
def build_frontier(kind, *, validate=True, **attributes):
  """Run a ready build frontier, retaining real capture/binding dependencies."""
  session = current_build_session()
  with (nullcontext() if session is not None and session.concurrent else _build_lock):
    if session is not None and validate:
      session.checkpoint()
    started = time.perf_counter()
    try:
      yield
    finally:
      if session is not None:
        session.events.append(dict(kind=kind, start_seconds=started-session._started,
            seconds=time.perf_counter() - started, thread=threading.current_thread().name, **attributes))
        if validate:
          session.checkpoint()


@contextmanager
def profile_phase(kind, name):
  """Opt-in Python attribution outside executable capture's escape guard.

  Profiled samples are diagnostic and must not be used as unprofiled timing.
  Real capture retains its existing rejection of active C profilers.
  """
  configured = os.environ.get("VECOPS_PREPARE_PROFILE")
  if not configured:
    yield
    return
  import cProfile
  import pstats
  import re

  directory = Path(configured).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))[:100]
  target = directory / f"{kind}-{label}-{os.getpid()}-{time.perf_counter_ns()}"
  profiler = cProfile.Profile()
  try:
    with profiler:
      yield
  finally:
    profiler.dump_stats(str(target) + '.prof')
    with Path(str(target) + '.txt').open('w') as stream:
      pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(60)


class BuildSession:
  """One bounded preparation session with byte-validated input snapshots.

  Source/toolchain inputs must remain unchanged during preparation. Repeated
  identity lookups reuse their immutable values; actual cold builds and the preparation seal
  re-read the actual bytes and directory inventories, including new headers.
  A mismatch rejects the session. This is not an mtime or process-global cache.
  Existing artifact providers retain their publication locks and validators.
  """

  def __init__(self, *, parallelism=None, report_path=None, reuse_sdk=True, concurrent=True):
    self.parallelism = build_parallelism(parallelism)
    self.report_path = Path(report_path).expanduser().resolve() if report_path else None
    self.reuse_sdk = reuse_sdk
    self.concurrent = concurrent and sys.platform.startswith("linux")
    self._snapshot_lock = threading.RLock()
    self._thread_state = threading.local()
    self._executor = None
    self._futures = []
    self._budget_root = None
    self.validation_events = []
    self.costs = {}
    self.events = []
    self._values = {}
    self._leaves = {}
    self._validating = False
    self.hits = self.misses = 0
    self.validation_seconds = 0.0
    self._publication_check_required = False
    self._native_validation_required = False

  def validate_publication(self):
    """Recheck once after a native subprocess, before provider publication."""
    if self._publication_check_required or self._native_validation_required:
      self.checkpoint()
      self._publication_check_required = False
      self._native_validation_required = False

  def _value(self, kind, key, compute, leaf):
    token = (kind, key)
    if token in self._values:
      self.hits += 1
      return self._values[token]
    self.misses += 1
    value = compute()
    self._values[token] = value
    if leaf:
      self._leaves[token] = compute
    return value

  @property
  def _publication_check_required(self):
    return getattr(self._thread_state, "publication", False)

  @_publication_check_required.setter
  def _publication_check_required(self, value):
    self._thread_state.publication = value

  @property
  def _native_validation_required(self):
    return getattr(self._thread_state, "native", False)

  @_native_validation_required.setter
  def _native_validation_required(self, value):
    self._thread_state.native = value

  def checkpoint(self):
    with self._snapshot_lock:
      if self._validating:
        return
      started = time.perf_counter()
      rows = []
      self._validating = True
      try:
        for token, compute in tuple(self._leaves.items()):
          leaf_start = time.perf_counter()
          current = compute()
          rows.append({"kind": token[0], "key": str(token[1]), "seconds": time.perf_counter()-leaf_start})
          if current != self._values[token]:
            raise RuntimeError(f"build inputs changed during preparation: {token!r}; start a new BuildSession")
      finally:
        self._validating = False
        seconds = time.perf_counter() - started
        self.validation_seconds += seconds
        self.validation_events.append({"start_seconds": started-self._started, "seconds": seconds,
            "thread": threading.current_thread().name, "leaves": rows})

  def hash_cpus(self):
    configured = os.environ.get("VECOPS_PRECOMPILE_CPU_LIST") or os.environ.get("VECOPS_BUILD_CPU_LIST")
    if not configured:
      return os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    cpus = set()
    for part in configured.split(','):
      ends = part.strip().split('-')
      first, last = int(ends[0]), int(ends[-1])
      if first < 0 or last < first:
        raise ValueError("invalid VECOPS_BUILD_CPU_LIST")
      cpus.update(range(first, last + 1))
    return cpus

  def launcher_command(self):
    return compiler_launcher_command()

  def compiler_environment(self, environment=None):
    result = compiler_launcher_environment(environment) if self.concurrent else {}
    if os.environ.get('VECOPS_PRECOMPILE_CPU_LIST'):
      # An explicitly reserved host-wide compiler pool is independent of each
      # model rank's execution affinity and native Parallelism specialization.
      cpus = self.hash_cpus()
      if not cpus:
        raise ValueError('precompile CPU list must not be empty')
      result['VECOPS_BUILD_CPU_LIST'] = ','.join(map(str,sorted(cpus)))
    return result

  def submit(self, kind, function, *args, **kwargs):
    if self._executor is None:
      self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="vecops-build")
    context = copy_context()
    def run():
      with build_frontier(kind, validate=False):
        return function(*args, **kwargs)
    future = self._executor.submit(context.run, run)
    self._futures.append(future)
    return future

  def run_tasks(self, tasks):
    if not self.concurrent:
      return [function() for _, function in tasks]
    futures = [self.submit(kind, function) for kind, function in tasks]
    # Join every ready frontier even when one fails. No background compiler
    # may outlive a failed session or publish after its snapshot is discarded.
    error = None
    results = []
    for future in futures:
      try:
        results.append(future.result())
      except BaseException as caught:  # noqa: BLE001 - drain sibling futures before propagating interruption
        error = error or caught
    if error is not None:
      raise error
    return results

  def __enter__(self):
    if current_build_session() is not None:
      raise RuntimeError("BuildSession contexts cannot be nested")
    if not _process_session_guard.acquire(blocking=False):
      raise RuntimeError("only one independent BuildSession may be active per process")
    try:
      self._started = time.perf_counter()
      if self.concurrent:
        from ._compiler import default_cache_dir
        parent = self.report_path.parent if self.report_path else default_cache_dir() / "build-sessions"
        parent.mkdir(parents=True, exist_ok=True)
        self._budget_root = Path(tempfile.mkdtemp(prefix="build-budget-", dir=parent))
        route = _process_route()
        temporary = route.with_suffix('.new')
        temporary.write_text(json.dumps({"root": str(self._budget_root), "jobs": self.parallelism}))
        temporary.replace(route)
    except BaseException:
      _process_session_guard.release()
      raise
    self._token = _current.set(self)
    return self

  def __exit__(self, error_type, error, traceback):
    completed = False
    try:
      if self._executor is not None:
        self._executor.shutdown(wait=True, cancel_futures=error_type is not None)
        if error_type is None:
          for future in self._futures:
            future.result()
      if error_type is None:
        self.checkpoint()
        completed = True
        from ._build_cost import save_observed_costs
        save_observed_costs(self.costs)
    finally:
      _current.reset(self._token)
      try:
        if self.report_path:
          self.report_path.parent.mkdir(parents=True, exist_ok=True)
          self.report_path.write_text(
            json.dumps(
              {
                "parallelism": self.parallelism,
                "seconds": time.perf_counter() - self._started,
                "snapshot_hits": self.hits,
                "snapshot_misses": self.misses,
                "validation_seconds": self.validation_seconds,
                "validation_events": self.validation_events,
                "concurrent": self.concurrent,
                "observed_source_costs": self.costs,
                "compiler_commands": [json.loads(path.read_text()) for path in
                    sorted(self._budget_root.glob("command-*.json"))] if self._budget_root else [],
                "frontiers": self.events,
                "completed": completed,
              },
              indent=2,
            )
            + "\n"
          )
      finally:
        try:
          if self._budget_root is not None:
            _process_route().write_text('{"root": null}')
        finally:
          _process_session_guard.release()
          if self._budget_root is not None and self.report_path is None:
            shutil.rmtree(self._budget_root)
          self._values.clear()
          self._leaves.clear()
