# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Frozen validators must not be redirected to an editable live source tree."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def entry():
  path = Path(__file__).resolve().parents[2] / "benchmarks/dsl_snapshot_entry.py"
  spec = importlib.util.spec_from_file_location("vecops_snapshot_entry_test", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_only_vecops_editable_finder_is_selected(entry):
  finder_type = type("ScikitBuildRedirectingFinder", (), {})
  ours = finder_type()
  ours.known_source_files = {"vecops.language": "/live/source/language.py"}
  other = finder_type()
  other.known_wheel_files = {"alphafold3.cpp": "/installed/af3.so"}
  assert entry.claims_vecops(ours)
  assert not entry.claims_vecops(other)
  assert not entry.claims_vecops(SimpleNamespace(known_source_files={"vecops": "ignored"}))


def test_snapshot_refuses_an_already_loaded_package(entry, tmp_path, monkeypatch):
  (tmp_path / "python/vecops").mkdir(parents=True)
  (tmp_path / "python/vecops/__init__.py").touch()
  (tmp_path / "CMakeLists.txt").touch()
  monkeypatch.setitem(sys.modules, "vecops", SimpleNamespace())
  with pytest.raises(RuntimeError, match="fresh process"):
    entry.select_snapshot(tmp_path)


def test_snapshot_requires_the_project_markers(entry, tmp_path):
  with pytest.raises(ValueError, match="snapshot needs"):
    entry.select_snapshot(tmp_path)


def test_runner_uses_frozen_tests_not_the_callers_live_tree(entry, tmp_path, monkeypatch):
  root = tmp_path / "frozen"
  root.mkdir()
  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(entry, "select_snapshot", lambda _: root)
  monkeypatch.setattr(sys, "argv", list(sys.argv))
  monkeypatch.setitem(
    sys.modules,
    "vecops",
    SimpleNamespace(__file__=str(root / "python/vecops/__init__.py"), _C=SimpleNamespace(__file__=str(root / "_C.so"))),
  )
  launched = []

  def run(path, *, run_name):
    launched.append((Path.cwd(), path, run_name, list(sys.argv)))

  monkeypatch.setattr(entry.runpy, "run_path", run)
  entry.main(["--root", str(root), "--", "python/tests/test_dsl_state.py", "--", "-q"])
  assert launched == [
    (
      root,
      str(root / "benchmarks/dsl_arm_validate.py"),
      "__main__",
      [str(root / "benchmarks/dsl_arm_validate.py"), "python/tests/test_dsl_state.py", "--", "-q"],
    ),
  ]
