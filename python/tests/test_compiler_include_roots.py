# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import vecops
import vecops._compiler as compiler_module
import vecops._identity as identity_module


def test_include_environment_is_anchored_at_compiler_definition_cwd(tmp_path, monkeypatch):
  project = tmp_path / "project"
  project.mkdir()
  include = project / "include"
  include.mkdir()
  monkeypatch.chdir(project)
  monkeypatch.setenv("CPATH", os.pathsep + "include" + os.pathsep)
  monkeypatch.setenv("CPLUS_INCLUDE_PATH", ".")
  monkeypatch.setenv("C_INCLUDE_PATH", "")
  monkeypatch.setenv("CXXFLAGS", "-isystem include -Irelative")

  compiler = vecops.Compiler(target="Scalar", cache_dir=tmp_path / "cache")
  expected = str(project.resolve())
  assert compiler.definition_cwd == project.resolve()
  assert compiler.environment["CPATH"] == os.pathsep.join((expected, str(include.resolve()), expected))
  assert compiler.environment["CPLUS_INCLUDE_PATH"] == expected
  assert compiler.environment["C_INCLUDE_PATH"] == ""
  assert compiler.environment["CXXFLAGS"] == f"-isystem {include.resolve()} -I{(project / 'relative').resolve()}"

  monkeypatch.chdir(tmp_path)
  assert {str(path) for path in identity_module.include_roots(compiler)} == {
    expected,
    str(include.resolve()),
    str((project / "relative").resolve()),
  }


def test_empty_include_variables_do_not_create_a_current_directory_root(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  for variable in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "CXXFLAGS"):
    monkeypatch.setenv(variable, "")
  compiler = vecops.Compiler(target="Scalar", cache_dir=tmp_path / "cache")
  assert compiler.environment["CPATH"] == ""
  assert compiler.environment["CPLUS_INCLUDE_PATH"] == ""
  assert compiler.environment["C_INCLUDE_PATH"] == ""
  assert tmp_path.resolve() not in identity_module.include_roots(compiler)


@pytest.mark.parametrize("flags", [("-I",), ("-isystem",)])
def test_include_flag_without_path_has_clear_error(tmp_path, monkeypatch, flags):
  monkeypatch.chdir(tmp_path)
  with pytest.raises(ValueError, match="requires a path argument"):
    vecops.Compiler(target="Scalar", cache_dir=tmp_path / "cache", cflags=flags)


def test_local_include_root_shadowing_is_detected_without_scanning_workspace(tmp_path, monkeypatch):
  project = tmp_path / "project"
  source_dir = project / "source"
  include_dir = project / "include"
  source_dir.mkdir(parents=True)
  include_dir.mkdir()
  monkeypatch.chdir(project)
  monkeypatch.setenv("CPATH", os.pathsep + str(include_dir))
  monkeypatch.setenv("CPLUS_INCLUDE_PATH", "")
  monkeypatch.setenv("C_INCLUDE_PATH", "")
  monkeypatch.setenv("CXXFLAGS", "")

  source = source_dir / "kernel.cpp"
  source.write_text('#include "shadow.h"\n')
  selected = include_dir / "shadow.h"
  selected.write_text("int selected;\n")
  compiler = vecops.Compiler(target="Scalar", cache_dir=tmp_path / "cache")
  roots = identity_module.include_roots(compiler)
  before = identity_module.source_inputs(
    source,
    roots,
    definition_cwd=compiler.definition_cwd,
    inventory_excludes=compiler._source_include_inventory_excludes,
  )

  # An output-like file elsewhere in the cwd is not a source dependency.
  (project / "unrelated.log").write_text("noise\n")
  after_noise = identity_module.source_inputs(
    source,
    roots,
    definition_cwd=compiler.definition_cwd,
    inventory_excludes=compiler._source_include_inventory_excludes,
  )
  assert before == after_noise

  local_shadow = project / "shadow.h"
  local_shadow.write_text("int local_shadow;\n")
  after_shadow = identity_module.source_inputs(
    source,
    roots,
    definition_cwd=compiler.definition_cwd,
    inventory_excludes=compiler._source_include_inventory_excludes,
  )
  assert before != after_shadow
  assert any(path == str(local_shadow) for path, _identity in after_shadow[0])

  explicit = vecops.Compiler(target="Scalar", cache_dir=tmp_path / "explicit-cache", include_dirs=(project,))
  explicit_before = identity_module.source_inputs(
    source,
    identity_module.include_roots(explicit),
    definition_cwd=explicit.definition_cwd,
    inventory_excludes=explicit._source_include_inventory_excludes,
  )
  (project / "explicit-only.h").write_text("int explicit_inventory;\n")
  explicit_after = identity_module.source_inputs(
    source,
    identity_module.include_roots(explicit),
    definition_cwd=explicit.definition_cwd,
    inventory_excludes=explicit._source_include_inventory_excludes,
  )
  assert explicit_before != explicit_after


def test_relative_source_and_dependencies_use_captured_cwd_after_chdir(tmp_path, monkeypatch):
  project = tmp_path / "project"
  (project / "headers").mkdir(parents=True)
  (project / "kernel.cpp").write_text("void kernel() {}\n")
  dependency = project / "headers" / "extra.h"
  dependency.write_text("constexpr int extra = 1;\n")
  other_cwd = tmp_path / "other"
  other_cwd.mkdir()
  monkeypatch.chdir(project)
  compiler = vecops.Compiler(target="Scalar", cache_dir=tmp_path / "cache")

  monkeypatch.chdir(other_cwd)
  before = identity_module.source_inputs(
    "kernel.cpp", dependencies=("headers/extra.h",), definition_cwd=compiler.definition_cwd
  )
  dependency.write_text("constexpr int extra = 2;\n")
  after = identity_module.source_inputs(
    "kernel.cpp", dependencies=("headers/extra.h",), definition_cwd=compiler.definition_cwd
  )
  assert before != after
  assert (str(dependency.resolve()), identity_module.file_identity(dependency)[1]) in after[0]


@pytest.mark.parametrize("delimiter", ('"', "<"))
def test_include_closure_observes_quoted_and_angle_search_order(tmp_path, delimiter):
  source_directory = tmp_path / "source"
  source_directory.mkdir()
  source = source_directory / "kernel.cpp"
  closing = '"' if delimiter == '"' else ">"
  source.write_text(f"#include {delimiter}value.h{closing}\n")
  sibling = source_directory / "value.h"
  sibling.write_text("constexpr int value = 51;\n")
  selected_from_root = tmp_path / "value.h"
  selected_from_root.write_text("constexpr int value = 1;\n")
  expected = sibling if delimiter == '"' else selected_from_root
  ignored = selected_from_root if delimiter == '"' else sibling
  before = identity_module.source_inputs(source, (tmp_path,), inventory_excludes=(tmp_path,))
  assert str(expected) in dict(before[0])
  assert str(ignored) not in dict(before[0])
  ignored.write_text("constexpr int value = 52;\n")
  assert before == identity_module.source_inputs(source, (tmp_path,), inventory_excludes=(tmp_path,))
  expected.write_text("constexpr int value = 3;\n")
  assert before != identity_module.source_inputs(source, (tmp_path,), inventory_excludes=(tmp_path,))


def test_native_config_and_identity_probe_use_frozen_absolute_include_roots(tmp_path, monkeypatch):
  project = tmp_path / "project"
  project.mkdir()
  other_cwd = tmp_path / "other"
  other_cwd.mkdir()
  include = project / "include"
  include.mkdir()
  (project / "relative-runtime.so").write_bytes(b"runtime")
  monkeypatch.chdir(project)
  monkeypatch.setenv("CPATH", os.pathsep + "include")
  monkeypatch.setenv("CPLUS_INCLUDE_PATH", "")
  monkeypatch.setenv("C_INCLUDE_PATH", "")
  monkeypatch.setenv("CXXFLAGS", "-Irelative")

  compiler = vecops.Compiler(target="Scalar", cache_dir=tmp_path / "cache")
  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(compiler_module, "_sdk", lambda: (tmp_path, vecops._C.SdkLayout.package))
  monkeypatch.setattr(compiler_module, "_tool", lambda *_args: Path(sys.executable))
  config = compiler._native_config()
  native_environment = dict(config.toolchain.environment)
  launchers = {f"CMAKE_{language}_{phase}_LAUNCHER"
               for language in ("C", "CXX") for phase in ("COMPILER", "LINKER")}
  assert {key: value for key, value in native_environment.items() if key not in launchers} == compiler.environment
  assert set(native_environment) - set(compiler.environment) <= launchers
  assert str(project.resolve()) in compiler.environment["CPATH"].split(os.pathsep)
  assert str(include.resolve()) in compiler.environment["CPATH"].split(os.pathsep)
  assert f"-I{(project / 'relative').resolve()}" in compiler.environment["CXXFLAGS"]

  monkeypatch.setattr(identity_module, "implementation_identity", lambda: "sdk")
  hashed = []

  def fake_file_identity(path):
    hashed.append(str(path))
    return str(path), "file"

  monkeypatch.setattr(identity_module, "file_identity", fake_file_identity)
  scanned = []
  monkeypatch.setattr(identity_module, "tree_identity", lambda path, *_args: scanned.append(str(path)) or ())
  monkeypatch.setattr("vecops._compiler._tool", lambda *_args: Path(sys.executable))
  calls = []

  def fake_run(command, **kwargs):
    calls.append(kwargs)
    if "-print-file-name=" in command[-1]:
      return SimpleNamespace(stdout="relative-runtime.so", stderr="")
    roots = f"{project}\n{include}\n.\n/usr/include\n"
    return SimpleNamespace(stdout="", stderr=f"#include <...> search starts here:\n{roots}End of search list\n")

  monkeypatch.setattr(subprocess, "run", fake_run)
  first_identity = identity_module.compiler_identity(compiler)
  monkeypatch.chdir(other_cwd)
  second_identity = identity_module.compiler_identity(compiler)
  assert first_identity == second_identity
  assert calls and all(call["cwd"] == project.resolve() for call in calls)
  assert str(project.resolve()) not in scanned
  assert str(include.resolve()) not in scanned
  assert "/usr/include" in scanned
  assert str((project / "relative-runtime.so").resolve()) in hashed
