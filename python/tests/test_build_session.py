# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Build budgets survive execution pinning; snapshots reject byte changes."""

import mmap
import os

import pytest
from vecops import BuildSession
from vecops._build_session import build_frontier, build_parallelism
from vecops._identity import file_identity, tree_identity


def test_build_cpu_list_survives_main_thread_pinning(monkeypatch):
  monkeypatch.setenv("VECOPS_BUILD_CPU_LIST", "0-55")
  monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0})
  assert build_parallelism() == 8
  with BuildSession(parallelism=16, reuse_sdk=False):
    assert build_parallelism() == 16
    assert build_parallelism(4) == 4
    with pytest.raises(ValueError, match="budget"):
      build_parallelism(32)


@pytest.mark.parametrize("change", ["mmap", "shadow", "replace"])
def test_session_revalidates_bytes_and_inventory_before_build(tmp_path, change):
  source = tmp_path / "a.h"
  source.write_bytes(b"AAAA")
  entered = []
  with pytest.raises(RuntimeError, match="build inputs changed"), BuildSession(reuse_sdk=False):
    original = file_identity(source), tree_identity(tmp_path)
    assert original == (file_identity(source), tree_identity(tmp_path))
    metadata = source.stat()
    if change == "mmap":
      with source.open("r+b") as stream, mmap.mmap(stream.fileno(), 0) as mapping:
        mapping[:] = b"BBBB"
        mapping.flush()
    elif change == "replace":
      source.unlink()
      source.write_bytes(b"BBBB")
      os.utime(source, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    else:
      (tmp_path / "shadow.h").write_bytes(b"NEW")
    with build_frontier("must-not-start"):
      entered.append(True)
  assert not entered


def test_pointwise_artifact_contract_reuses_extents_and_outer_strides():
  from vecops.graph._pointwise import kernel

  with BuildSession(reuse_sdk=False):

    def copy(shape, strides):
      return kernel(0, shape, strides, "float32", shape, strides, "float32", shape, strides, "float32", 1, ())

    assert copy((4, 8), (8, 1)) is copy((17, 32), (32, 1))
    assert copy((1, 8), (8, 1)) is copy((17, 32), (32, 1))
    assert copy((4, 8), (16, 1)) is copy((17, 32), (64, 1))
    assert copy((4, 8), (8, 1)) is not copy((4, 8), (16, 1))


def test_publication_revalidates_snapshot_before_provider_can_publish(tmp_path):
  source = tmp_path / "kernel.cpp"
  source.write_text("old")
  with pytest.raises(RuntimeError, match="build inputs changed"), BuildSession(reuse_sdk=False) as session:
    file_identity(source)
    source.write_text("new")
    session._publication_check_required = True
    session.validate_publication()


def test_pointwise_factory_validates_cached_core_before_reusing_it():
  from vecops.graph._pointwise import kernel

  with BuildSession(reuse_sdk=False):
    args = (6, (17,), (1,), "float32", (17,), (1,), "float32", (17,), (1,), "float32", 1, ())
    old = kernel(*args)
    original = old.debug._validate_cache
    try:
      old.debug._validate_cache = lambda: False
      assert kernel(*args) is not old
    finally:
      old.debug._validate_cache = original


@pytest.mark.parametrize("concurrent", [False, True])
def test_native_batch_rejects_mid_build_edit_before_publication(tmp_path, monkeypatch, concurrent):
  import vecops
  import vecops.language as T

  source = tmp_path / "kernel.cpp"
  source.write_text('#include "vecops/Kernel.h"\nvoid __kernel__(auto x, auto y) { y(0) = x(0); }\n')
  compiler = vecops.Compiler(
    target="Scalar",
    cc="/usr/bin/gcc",
    cxx="/usr/bin/g++",
    cache_dir=tmp_path / "artifacts",
    build_dir=tmp_path / "builds",
    jobs=2,
  )
  original = vecops.Compiler._observe_build
  mutated = []

  def observe(self, event):
    original(self, event)
    if event.state == "finished" and event.phase.startswith("kernels: build"):
      source.write_text(source.read_text() + "// concurrent edit\n")
      mutated.append(True)

  monkeypatch.setattr(vecops.Compiler, "_observe_build", observe)
  with pytest.raises(RuntimeError, match="build inputs changed"), BuildSession(parallelism=2, reuse_sdk=False):

    @T.jit(out_idx=[1], compiler=compiler)
    def factory():
      @T.template(source=str(source))
      def entry(x: T.Tensor((1,), (1,), T.float32), y: T.Tensor((1,), (1,), T.float32)):
        pass

      return entry

    kernel = factory()
    with vecops.collect_compile_requests() as requests:
      kernel(vecops.TensorMeta((1,), dtype="float32"), vecops.TensorMeta((1,), dtype="float32"))
    if concurrent:
      from vecops._build_session import current_build_session
      current_build_session().run_tasks([
          ('native-test', lambda: vecops.compile_batch(requests.requests, execution_parallelism=1)),
          ('independent-ready', lambda: None),
      ])
    else:
      vecops.compile_batch(requests.requests, execution_parallelism=1)
  assert mutated
  assert not list(compiler.cache_dir.rglob("*.so"))


def test_ready_frontiers_share_cross_process_budget(tmp_path):
  import json
  import subprocess
  import sys
  from concurrent.futures import ThreadPoolExecutor

  with BuildSession(parallelism=2, report_path=tmp_path / 'session.json') as session:
    def run_command():
      subprocess.run(session.launcher_command() + [sys.executable, '-c',
          'import time; time.sleep(0.08)'], check=True)
    with ThreadPoolExecutor(max_workers=7) as pool:
      list(pool.map(lambda _: run_command(), range(7)))
  report = json.loads((tmp_path / 'session.json').read_text())
  events = sorted([(c['started_ns'], 1) for c in report['compiler_commands']] +
                  [(c['ended_ns'], -1) for c in report['compiler_commands']])
  active = peak = 0
  for _, delta in events:
    active += delta
    peak = max(peak, active)
  assert peak == 2
  assert len(report['compiler_commands']) == 7


def _standalone_launcher(tmp_path, *, jobs=2, active=True):
  import json
  import sys
  from pathlib import Path

  root = tmp_path / 'slots'
  root.mkdir()
  route = tmp_path / 'route.json'
  route.write_text(json.dumps({'root': str(root) if active else None, 'jobs': jobs}))
  return [sys.executable, str(Path(__file__).parents[1] / 'vecops/_build_launcher.py'), str(route)], root


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='Linux CPU affinity required')
def test_launcher_slots_bind_real_concurrent_children_to_different_cpus(tmp_path):
  import json
  import subprocess
  import sys
  import time

  allowed = sorted(os.sched_getaffinity(0))
  if len(allowed) < 2:
    pytest.skip('two CPUs in the test process grant are required')
  command, root = _standalone_launcher(tmp_path)
  # Each child holds its slot until both children have reported their real OS
  # mask. This proves simultaneous, distinct slot assignment, without a build.
  script = ('import json,os,pathlib,sys,time; '
      'pathlib.Path(sys.argv[1]).write_text(json.dumps(sorted(os.sched_getaffinity(0)))); '
      'deadline=time.monotonic()+10\n'
      'while not pathlib.Path(sys.argv[2]).exists():\n'
      ' if time.monotonic()>deadline: raise TimeoutError("test rendezvous")\n'
      ' time.sleep(0.01)\n')
  environment = dict(os.environ, VECOPS_BUILD_CPU_LIST=','.join(map(str, allowed[:2])))
  release = tmp_path / 'release'
  outputs = [tmp_path / f'child-{index}.json' for index in range(2)]
  children = []
  try:
    for output in outputs:
      children.append(subprocess.Popen(command + [sys.executable, '-S', '-c', script,
          str(output), str(release)], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    deadline = time.monotonic() + 10
    while not all(output.exists() for output in outputs):
      assert all(child.poll() is None for child in children), 'launcher exited before rendezvous'
      assert time.monotonic() < deadline, 'launchers did not acquire two distinct slots'
      time.sleep(0.01)
    assert {tuple(json.loads(output.read_text())) for output in outputs} == {(cpu,) for cpu in allowed[:2]}
  finally:
    release.touch()
    for child in children:
      try:
        stdout, stderr = child.communicate(timeout=12)
      except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        raise
      assert child.returncode == 0, (stdout, stderr)
  traces = [json.loads(path.read_text()) for path in root.glob('command-*.json')]
  assert len(traces) == 2
  assert {trace['slot'] for trace in traces} == {0, 1}
  for trace in traces:
    assert trace['allowed_cpus'] == trace['cpus'] == allowed
    assert trace['requested_cpus'] == allowed[:2]
    assert trace['assigned_cpu'] == allowed[trace['slot']]
    assert trace['bound_cpus'] == trace['child_cpus_after_spawn'] == [trace['assigned_cpu']]
    assert trace['child_affinity_verified'] is True


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='Linux CPU affinity required')
@pytest.mark.parametrize('active', [False, True])
def test_launcher_without_cpu_list_keeps_real_child_grant(tmp_path, active):
  import json
  import subprocess
  import sys

  allowed = sorted(os.sched_getaffinity(0))
  command, root = _standalone_launcher(tmp_path, active=active)
  environment = dict(os.environ)
  environment.pop('VECOPS_BUILD_CPU_LIST', None)
  result = subprocess.run(command + [sys.executable, '-S', '-c',
      'import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))'],
      env=environment, check=True, capture_output=True, text=True, timeout=10)
  assert json.loads(result.stdout) == allowed
  traces = [json.loads(path.read_text()) for path in root.glob('command-*.json')]
  if active:
    assert len(traces) == 1
    assert traces[0]['allowed_cpus'] == traces[0]['bound_cpus'] == allowed
    assert traces[0]['assigned_cpu'] is None
    assert traces[0]['requested_cpus'] is None
  else:
    assert not traces


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='Linux CPU affinity required')
@pytest.mark.parametrize('cpu_list', ['', '-1', '1-', '1-2-3', '2-1', '1,,2', 'not-a-cpu', '0-9999999999999'])
def test_launcher_rejects_invalid_cpu_list_before_starting_child(tmp_path, cpu_list):
  import subprocess
  import sys

  command, root = _standalone_launcher(tmp_path)
  marker = tmp_path / 'must-not-run'
  result = subprocess.run(command + [sys.executable, '-S', '-c',
      'import pathlib,sys; pathlib.Path(sys.argv[1]).touch()', str(marker)],
      env=dict(os.environ, VECOPS_BUILD_CPU_LIST=cpu_list), capture_output=True, text=True, timeout=10)
  assert result.returncode != 0
  assert 'VECOPS_BUILD_CPU_LIST' in result.stderr
  assert not marker.exists()
  assert not list(root.glob('command-*.json'))


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='Linux CPU affinity required')
def test_launcher_rejects_cpu_outside_its_actual_grant(tmp_path):
  import subprocess
  import sys

  allowed = sorted(os.sched_getaffinity(0))
  if len(allowed) < 2:
    pytest.skip('two CPUs in the test process grant are required')
  command, root = _standalone_launcher(tmp_path)
  marker = tmp_path / 'must-not-run'
  # The second CPU is legal for the parent but absent from the launcher's
  # narrower grant. An environment variable cannot authorize widening it.
  wrapper = ('import os,sys; os.sched_setaffinity(0,{int(sys.argv[1])}); '
      'os.execv(sys.argv[2],sys.argv[2:])')
  result = subprocess.run([sys.executable, '-S', '-c', wrapper, str(allowed[0])] + command +
      [sys.executable, '-S', '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).touch()', str(marker)],
      env=dict(os.environ, VECOPS_BUILD_CPU_LIST=','.join(map(str, allowed[:2]))),
      capture_output=True, text=True, timeout=10)
  assert result.returncode != 0
  assert 'outside the launcher affinity grant' in result.stderr
  assert not marker.exists()
  assert not list(root.glob('command-*.json'))


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='Linux CPU affinity required')
@pytest.mark.parametrize('active', [False, True])
def test_launcher_explicit_range_requires_active_budget_for_pinning(tmp_path, active):
  import json
  import subprocess
  import sys

  allowed = sorted(os.sched_getaffinity(0))
  command, root = _standalone_launcher(tmp_path, active=active)
  cpu = allowed[0]
  result = subprocess.run(command + [sys.executable, '-S', '-c',
      'import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))'],
      env=dict(os.environ, VECOPS_BUILD_CPU_LIST=f'{cpu}-{cpu}'),
      capture_output=True, text=True, check=True, timeout=10)
  assert json.loads(result.stdout) == ([cpu] if active else allowed)
  traces = [json.loads(path.read_text()) for path in root.glob('command-*.json')]
  if active:
    assert len(traces) == 1
    assert traces[0]['requested_cpus'] == traces[0]['bound_cpus'] == [cpu]
  else:
    assert not traces


def test_parallel_frontiers_keep_context_and_join_on_error(tmp_path):
  import threading
  from vecops._build_session import current_build_session

  rendezvous = threading.Barrier(2, timeout=5)
  done = []
  with pytest.raises(ValueError, match='failed frontier'):
    with BuildSession(parallelism=2, report_path=tmp_path / 'session.json') as session:
      def fail():
        assert current_build_session() is session
        rendezvous.wait()
        raise ValueError('failed frontier')
      def succeed():
        assert build_parallelism() == 2
        rendezvous.wait()
        done.append(True)
      session.run_tasks([('fails', fail), ('succeeds', succeed)])
  assert done == [True]


def test_snapshot_is_shared_once_across_frontiers(tmp_path):
  import threading
  from vecops._build_session import snapshot_value

  calls = []
  rendezvous = threading.Barrier(2, timeout=5)
  with BuildSession(parallelism=2, report_path=tmp_path / 'session.json') as session:
    def read():
      rendezvous.wait()
      return snapshot_value('test', 'shared', lambda: calls.append(True) or 7)
    assert session.run_tasks([('one', read), ('two', read)]) == [7, 7]
  assert calls == [True]


def test_request_order_uses_cost_hints_without_dropping_requests(tmp_path, monkeypatch):
  import json
  from types import SimpleNamespace
  from vecops._build_cost import order_requests

  profile = tmp_path / 'costs.json'
  profile.write_text(json.dumps({'small.cpp': 20, 'large.cpp': 3}))
  monkeypatch.setenv('VECOPS_BUILD_COST_PROFILE', str(profile))
  requests = [SimpleNamespace(source='large.cpp'), SimpleNamespace(source='small.cpp'),
              SimpleNamespace(source='small.cpp')]
  ordered = order_requests(requests)
  assert list(ordered) == [requests[1], requests[2], requests[0]]


def test_launcher_preserves_explicit_compiler_launcher(tmp_path):
  with BuildSession(parallelism=2, report_path=tmp_path / 'session.json') as session:
    environment = session.compiler_environment({'CMAKE_CXX_COMPILER_LAUNCHER': 'ccache;--verbose'})
    assert environment['CMAKE_CXX_COMPILER_LAUNCHER'].endswith(';ccache;--verbose')


def test_cached_launcher_route_follows_next_session(tmp_path):
  import json
  import subprocess
  import sys

  with BuildSession(parallelism=1, report_path=tmp_path / 'first.json') as first:
    cached_command = first.launcher_command()
  # A retained native compiler uses the same command in another session.
  with BuildSession(parallelism=2, report_path=tmp_path / 'second.json') as second:
    assert cached_command == second.launcher_command()
    subprocess.run(cached_command + [sys.executable, '-c', 'pass'], check=True)
  assert len(json.loads((tmp_path / 'second.json').read_text())['compiler_commands']) == 1
  # The same retained compiler is also usable outside any session.
  subprocess.run(cached_command + [sys.executable, '-c', 'pass'], check=True)


def test_report_failure_releases_process_session_guard(tmp_path):
  with pytest.raises(IsADirectoryError), BuildSession(report_path=tmp_path):
    pass
  with BuildSession(reuse_sdk=False):
    pass


def test_setup_failure_releases_process_session_guard(tmp_path):
  parent = tmp_path / 'file'
  parent.write_text('not a directory')
  with pytest.raises(FileExistsError), BuildSession(report_path=parent / 'report.json'):
    pass
  with BuildSession(reuse_sdk=False):
    pass


def test_cached_include_closure_rejects_new_local_shadow_without_parent_inventory(tmp_path):
  from vecops._identity import source_inputs

  source = tmp_path / 'source' / 'kernel.cpp'
  source.parent.mkdir()
  source.write_text('#include "local.h"\n')
  includes = tmp_path / 'includes'
  includes.mkdir()
  (includes / 'local.h').write_text('original header')
  with pytest.raises(RuntimeError, match='build inputs changed'), BuildSession(reuse_sdk=False) as session:
    original = source_inputs(source, (includes,))
    # Generated files in the source directory are not an include dependency.
    (source.parent / 'artifact.o').write_bytes(b'not an input')
    session.checkpoint()
    (source.parent / 'local.h').write_text('new earlier header')
    assert source_inputs(source, (includes,)) is original
    session.checkpoint()


def test_cached_include_resolution_rejects_retargeted_header_symlink(tmp_path):
  from vecops._identity import source_inputs

  source = tmp_path / 'kernel.cpp'
  source.write_text('#include "local.h"\n')
  first, second = tmp_path / 'first.h', tmp_path / 'second.h'
  first.write_text('same content')
  second.write_text('same content')
  link = tmp_path / 'local.h'
  link.symlink_to(first)
  with pytest.raises(RuntimeError, match='build inputs changed'), BuildSession(reuse_sdk=False) as session:
    source_inputs(source)
    link.unlink()
    link.symlink_to(second)
    session.checkpoint()


def test_cached_include_structure_rechecks_mmap_bytes_at_next_checkpoint(tmp_path):
  from vecops._identity import source_inputs

  source = tmp_path / 'kernel.cpp'
  source.write_bytes(b'// AAAA\n')
  with pytest.raises(RuntimeError, match='build inputs changed'), BuildSession(reuse_sdk=False) as session:
    source_inputs(source)
    session.checkpoint()
    with source.open('r+b') as stream, mmap.mmap(stream.fileno(), 0) as mapping:
      mapping[:] = b'// BBBB\n'
      mapping.flush()
    session.checkpoint()


def test_parallel_library_hash_matches_serial_and_rechecks_mmap_bytes(tmp_path):
  from vecops._watch import tree_hash

  payload = b'a' * (17 << 20)
  (tmp_path / 'a.so').write_bytes(payload)
  (tmp_path / 'b.so').write_bytes(payload)
  cpus = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
  first = tree_hash(tmp_path, parallelism=2, cpus=cpus)
  assert first == tree_hash(tmp_path)
  with (tmp_path / 'a.so').open('r+b') as stream, mmap.mmap(stream.fileno(), 0) as mapping:
    mapping[:4] = b'bbbb'
    mapping.flush()
  assert tree_hash(tmp_path, parallelism=2, cpus=cpus) != first
  assert tree_hash(tmp_path, parallelism=2, cpus=cpus) == tree_hash(tmp_path)


def test_reserved_compiler_pool_does_not_change_execution_environment(monkeypatch):
  monkeypatch.setenv('VECOPS_BUILD_CPU_LIST','0-3')
  monkeypatch.setenv('VECOPS_PRECOMPILE_CPU_LIST','8-15')
  with BuildSession(parallelism=4,reuse_sdk=False,concurrent=False)as session:
    environment=session.compiler_environment()
    assert environment['VECOPS_BUILD_CPU_LIST']=='8,9,10,11,12,13,14,15'
    assert os.environ['VECOPS_BUILD_CPU_LIST']=='0-3'
    assert session.hash_cpus()==set(range(8,16))
