# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Scalable sinking policy is selected by a bounded executable probe."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from vecops import _C
from vecops._compiler_compat import SINK_OPTION, scalable_sink_flags


def fixture(tmp_path, monkeypatch, results, *, version="test clang version 19.1.7"):
  tool = tmp_path / "clang++"
  tool.write_text("test compiler identity")
  calls = []
  monkeypatch.setattr("vecops._compiler._tool", lambda *args: tool)

  def run(command, **kwargs):
    if "--version" in command:
      return SimpleNamespace(stdout=version)
    calls.append((command, kwargs))
    return results[len(calls) - 1]

  monkeypatch.setattr("vecops._compiler_compat.subprocess.run", run)
  compiler = SimpleNamespace(target="Native", cxx=str(tool), cflags=("-O3",), environment={},
    vector_state=SimpleNamespace(sve_bytes=64, env_token=(64,64)), cache_dir=tmp_path / "cache",
    cache_mode=_C.ArtifactCacheMode.read_write, definition_cwd=tmp_path)
  return compiler, calls


def result(code=0, stderr=""):
  return SimpleNamespace(returncode=code, stderr=stderr)


def test_failure_and_success_both_required_and_probe_reused(tmp_path, monkeypatch):
  compiler, calls = fixture(tmp_path, monkeypatch,
    [result(70, "fatal: Invalid size request on a scalable vector."), result()])
  assert scalable_sink_flags(compiler) == (SINK_OPTION,)
  assert scalable_sink_flags(compiler) == (SINK_OPTION,)
  assert len(calls) == 2
  assert all(kwargs["timeout"] == 15 for _, kwargs in calls)
  assert SINK_OPTION not in calls[0][0] and SINK_OPTION in calls[1][0]
  assert all("-fno-crash-diagnostics" in command for command, _ in calls)
  assert compiler.cflags == ("-O3",)


def test_healthy_compiler_has_no_workaround(tmp_path, monkeypatch):
  compiler, calls = fixture(tmp_path, monkeypatch, [result()])
  assert scalable_sink_flags(compiler) == ()
  assert scalable_sink_flags(compiler) == ()
  assert len(calls) == 1


@pytest.mark.parametrize("version,suppressed", [
  ("clang version 19.1.7", True),
  ("BiSheng Enterprise 5.1.0.2 clang version 19.1.7", True),
  ("Apple clang version 16.0.0", True),
  ("g++ (GCC) 15.3.0", False),
])
def test_probe_diagnostics_respect_compiler_driver(tmp_path, monkeypatch, version, suppressed):
  compiler, calls = fixture(tmp_path, monkeypatch, [result()], version=version)
  assert scalable_sink_flags(compiler) == ()
  assert ("-fno-crash-diagnostics" in calls[0][0]) is suppressed
  assert compiler.cflags == ("-O3",)


@pytest.mark.parametrize("failure", [result(1,"unsupported target"), result(70,"another compiler error")])
def test_unrelated_errors_are_not_silently_hidden(tmp_path, monkeypatch, failure):
  compiler, calls = fixture(tmp_path, monkeypatch, [failure])
  with pytest.raises(RuntimeError, match="unexpectedly"):
    scalable_sink_flags(compiler)
  assert len(calls) == 1


def test_workaround_must_compile(tmp_path, monkeypatch):
  compiler, calls = fixture(tmp_path, monkeypatch,
    [result(70,"Invalid size request on a scalable vector"), result(1,"still fails")])
  with pytest.raises(RuntimeError, match="still fails"):
    scalable_sink_flags(compiler)


def test_offline_replay_never_discovers_or_invokes_tools(tmp_path, monkeypatch):
  compiler, calls = fixture(tmp_path, monkeypatch,
    [result(70,"Invalid size request on a scalable vector"), result()])
  expected = scalable_sink_flags(compiler)
  compiler.cache_mode = _C.ArtifactCacheMode.cache_only
  def forbidden(*args, **kwargs):
    raise AssertionError("offline tool invocation")
  monkeypatch.setattr("vecops._compiler._tool", forbidden)
  monkeypatch.setattr("vecops._compiler_compat.subprocess.run", forbidden)
  assert scalable_sink_flags(compiler) == expected


def test_non_arm_and_scalar_skip_tools(tmp_path, monkeypatch):
  compiler, calls = fixture(tmp_path, monkeypatch, [])
  compiler.vector_state.sve_bytes = None
  assert scalable_sink_flags(compiler) == ()
  compiler.vector_state.sve_bytes = 64
  compiler.target = "Scalar"
  assert scalable_sink_flags(compiler) == ()
  assert not calls


def test_changed_executable_reprobes_even_at_same_path(tmp_path, monkeypatch):
  compiler, calls = fixture(tmp_path, monkeypatch, [result(), result()])
  assert scalable_sink_flags(compiler) == ()
  Path(compiler.cxx).write_text("changed compiler identity")
  assert scalable_sink_flags(compiler) == ()
  assert len(calls) == 2


def test_selected_flags_are_part_of_source_and_dsl_snapshot_identity(tmp_path, monkeypatch):
  from vecops import Compiler
  from vecops._arm_vector_state import VectorState
  from vecops.language._compiler.cache import _compiler_key
  monkeypatch.setattr("vecops._arm_vector_state.get_vector_state", lambda: VectorState(64,64))
  monkeypatch.setattr("vecops._compiler_compat.scalable_sink_flags", lambda compiler: (SINK_OPTION,))
  compiler = Compiler(cache_dir=tmp_path, cflags=("-O3",))
  selected = compiler.snapshot()
  assert selected.cflags == ("-O3", SINK_OPTION)
  assert compiler.cflags == ("-O3",)
  assert _compiler_key(selected) != _compiler_key(compiler)
  assert selected.snapshot() is selected
