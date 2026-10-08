# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Frozen validators must not be redirected to an editable live source tree."""

import importlib
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


@pytest.mark.parametrize(
  "module,arguments,metadata",
  [
    ("dsl_transpose", (128, 128, "Any", "float32", "float16", "float32", "sme"), None),
    ("dsl_matmul", (4, 4, 256, "Dynamic", "bf16", "bias_relu", "matmul"), 7),
    ("dsl_matmul_pack", (17, 65, "Dynamic", "bf16", "B", False), 2),
  ],
)
def test_benchmark_declarations_do_not_lower_unused_kernels(monkeypatch, module, arguments, metadata):
  import vecops.language as T
  from vecops.language._compiler.frontend.parser import lower

  root = Path(__file__).resolve().parents[2]
  monkeypatch.syspath_prepend(str(root / "benchmarks"))
  benchmark = importlib.import_module(module)
  calls = []

  def compile(template, **kwargs):
    calls.append((template, kwargs))
    return template

  monkeypatch.setattr(T, "compile", compile)
  declared = benchmark.templates(*arguments)
  pair = declared if metadata is None else declared[0]
  assert not calls
  assert pair[0].ir.source is None
  assert pair[1].ir.source.name.endswith("reference.cpp")
  assert pair[0].ir.signature == pair[1].ir.signature
  assert lower(pair[0].ir).body.statements
  if metadata is not None:
    assert len(declared[1]) == metadata

  compiler = object()
  result = benchmark.kernels(compiler, *arguments)
  compiled = result if metadata is None else result[0]
  assert tuple(value[0] for value in calls) == compiled
  assert len(calls) == 2
  assert all(value[1]["compiler"] is compiler for value in calls)


def test_frozen_source_replay_rejects_changed_or_missing_vector_geometry(monkeypatch):
  from vecops._arm_vector_state import VectorState

  root = Path(__file__).resolve().parents[2]
  monkeypatch.syspath_prepend(str(root / "benchmarks"))
  benchmark = importlib.import_module("dsl_register_refactor_ab")
  expected = benchmark.vector_state_identity(SimpleNamespace(vector_state=VectorState(64, 32)))
  matching = {"vector_state": expected}
  benchmark.validate_source_geometry((matching, matching), expected)
  for document in ({}, {"vector_state": {"sve_bytes": 64, "sme_bytes": 64}}):
    with pytest.raises(ValueError, match="baseline source catalog vector state"):
      benchmark.validate_source_geometry((document, matching), expected)
    with pytest.raises(ValueError, match="candidate source catalog vector state"):
      benchmark.validate_source_geometry((matching, document), expected)
