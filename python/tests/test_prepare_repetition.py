# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Regression coverage for preparation identities, aliases and frozen plans."""
from copy import copy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest
from vecops import graph
from vecops._build_session import BuildSession
from vecops.planner import graph_digest


class Tensor:
  pass


def aliased_graph():
  capture = graph.GraphCapture('alias versions')
  base, view, other, output = (Tensor() for _ in range(4))
  metadata = graph.TensorMetadata((8,), (1,), 'float32', 'cpu', element_size=4)
  with capture:
    for value, storage in ((base, 1), (view, 1), (other, 2), (output, 1)):
      capture.tensor(value, metadata, storage_identity=storage, storage_bytes=32, external=True)
    first = capture.record_operator('aliased write', inputs=[
      (base, 'read', 'read'), (view, 'read_write', 'write1'),
      (base, 'write', 'write2'), (other, 'read', 'unrelated')], outputs=[output])
    second = capture.record_operator('read next', inputs=[(view, 'read', 'view')], outputs=[])
    capture.mark_outputs([output])
  return capture.finish(), first, second


def test_alias_versions_snapshot_all_inputs_before_unique_writes():
  result, first, second = aliased_graph()
  assert [(u.argument, u.version_before, u.version_after) for u in first.uses] == [
    ('read', 0, 1), ('write1', 0, 1), ('write2', 0, 1), ('unrelated', 0, 0), ('return', 1, 1)]
  assert [(u.version_before, u.version_after) for u in second.uses] == [(1, 1)]
  result.validate()


def test_frozen_graph_detaches_nested_state_and_reuses_canonical_digest(monkeypatch):
  source, _, _ = aliased_graph()
  source.attributes['nested'] = {'list': [1, {'flag': True}]}
  source.events[0].attributes['source'] = {'path': '/kernel.cpp'}
  canonical = json.dumps(graph.ir._plain(asdict(source)), sort_keys=True)
  expected = hashlib.sha256(canonical.encode()).hexdigest()
  frozen = source.freeze()
  frozen.validate()
  assert frozen.freeze() is frozen
  assert frozen.dumps(indent=None) == canonical
  assert graph_digest(frozen) == expected
  assert graph.ExecutionGraph.loads(frozen.dumps()).to_dict() == source.to_dict()
  # Repeated plan validation must not encode a sealed graph a second time.
  with monkeypatch.context() as patch:
    patch.setattr(type(frozen), 'dumps', lambda *_args, **_kwargs: pytest.fail('duplicate encoding'))
    assert graph_digest(frozen) == expected
  source.attributes['nested']['list'][1]['flag'] = False
  source.events[0].attributes['source']['path'] = '/changed.cpp'
  source.tensors[next(iter(source.tensors))].metadata.dtype = 'int32'
  assert graph_digest(source) != expected
  assert graph_digest(source.freeze()) != expected
  detached = frozen.to_dict()
  detached['attributes']['nested']['list'][1]['flag'] = False
  assert graph_digest(frozen) == expected
  assert frozen.attributes['nested']['list'][1]['flag'] is True
  with pytest.raises(TypeError):
    frozen.name = 'changed'
  with pytest.raises(TypeError):
    del frozen.events[0].name
  with pytest.raises(TypeError):
    frozen.events[0].attributes['source']['path'] = 'changed'
  with pytest.raises(TypeError):
    frozen.tensors[next(iter(frozen.tensors))].metadata.dtype = 'int32'
  with pytest.raises(TypeError):
    frozen.attributes['nested']['list'][0] = 7


@pytest.mark.parametrize('edited', ['wrapper', 'installed'])
def test_session_probe_reuse_and_content_checkpoint(tmp_path, monkeypatch, edited):
  from vecops import _C
  from vecops import _compiler_compat as compat
  wrapper = tmp_path / 'clang-wrapper'
  wrapper.write_text('wrapper v1')
  installed = tmp_path / 'clang'
  installed.write_text('driver v1')
  monkeypatch.setattr('vecops._compiler._tool', lambda *_args: wrapper)
  calls = []

  def run(command, **kwargs):
    calls.append(command)
    return SimpleNamespace(returncode=0, stderr='', stdout=(
      f'clang version test\nInstalledDir: {tmp_path}\n' if '--version' in command else ''))

  monkeypatch.setattr(compat.subprocess, 'run', run)
  compiler = SimpleNamespace(target='Native', cxx=str(wrapper), cflags=(), environment={},
      vector_state=SimpleNamespace(sve_bytes=64, env_token='64:64'),
      cache_dir=tmp_path/'cache', definition_cwd=tmp_path, cache_mode=_C.ArtifactCacheMode.read_write)
  # Probe once, then reuse for all equivalent snapshots. Both wrapper and
  # actual driver are byte leaves: an edit rejects before publication.
  with pytest.raises(RuntimeError, match='build inputs changed'):
    with BuildSession(parallelism=1, reuse_sdk=False, concurrent=False) as session:
      for _ in range(5):
        assert compat.scalable_sink_flags(copy(compiler)) == ()
      assert sum('--version' in c for c in calls) == 1
      assert len([e for e in session.events if e['kind'] == 'toolchain-probe']) == 1
      (wrapper if edited == 'wrapper' else installed).write_text('changed bytes')
  with BuildSession(parallelism=1, reuse_sdk=False, concurrent=False):
    assert compat.scalable_sink_flags(compiler) == ()
  assert sum('--version' in c for c in calls) == 2
  # A new effective environment or flag policy never hits the old snapshot.
  with BuildSession(parallelism=1, reuse_sdk=False, concurrent=False):
    compat.scalable_sink_flags(compiler)
    compiler.environment['PROBE_TEST'] = 'other'
    compat.scalable_sink_flags(compiler)
    compiler.cflags = ('-O2',)
    compat.scalable_sink_flags(compiler)
  assert sum('--version' in c for c in calls) == 5
  compiler.cache_mode = _C.ArtifactCacheMode.cache_only
  calls.clear()
  assert compat.scalable_sink_flags(compiler) == ()
  assert calls == []


def test_frozen_planning_and_validation_preserve_source_identity():
  from vecops import memory, planner
  source, _, _ = aliased_graph()
  frozen = source.freeze()
  runtime = memory.Runtime.discover()
  view = runtime.view()
  expected_digest = planner.graph_digest(frozen)
  plan = planner.plan_memory(frozen, runtime, view, runtime.snapshot(view))
  assert planner.validate_plan(plan, runtime, view) is plan
  source.attributes['model_revision'] = 2
  assert planner.graph_digest(source.freeze()) != expected_digest
  # Graph identity is a frontend contract; recipe validation checks resources.
  assert planner.graph_digest(frozen) == expected_digest
  assert planner.validate_plan(plan, runtime, view) is plan
