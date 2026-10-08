# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Vector-length cache/build snapshots, tested without native compilation."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import vecops
import vecops._compiler as compiler_module
import vecops._identity as identity_module
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.cache import _compiler_key


def _mock_state(monkeypatch, state):
  monkeypatch.setattr("vecops._arm_vector_state.get_vector_state", lambda: state)


@pytest.mark.parametrize(
  "first,second",
  [(VectorState(32, 64), VectorState(64, 32)), (VectorState(64, 64), VectorState(64, 32))],
)
def test_native_config_uses_frozen_sve_and_sme_widths_without_fixed_vector_flags(monkeypatch, tmp_path, first, second):
  _mock_state(monkeypatch, first)
  monkeypatch.setattr(compiler_module, "_sdk", lambda: (tmp_path, vecops._C.SdkLayout.package))
  monkeypatch.setattr(compiler_module, "_tool", lambda *_args: Path(sys.executable))

  compiler = vecops.Compiler(target="NativeFixedStreamingSVE", cache_mode="cache-only")
  frozen = compiler.snapshot()
  _mock_state(monkeypatch, second)
  config = frozen._native_config()

  assert frozen.vector_state == first
  assert "-msve-vector-bits=scalable" in tuple(config.compile_options)
  assert not any(
    flag.startswith(("-msve-vector-bits=128", "-msve-vector-bits=256", "-msve-vector-bits=512"))
    for flag in config.compile_options
  )
  definitions = set(config.compile_definitions)
  assert f"VECOPS_KERNEL_SVE_BYTES={first.sve_bytes}" in definitions
  assert f"VECOPS_KERNEL_SME_BYTES={first.sme_bytes}" in definitions
  assert f"VECOPS_TARGET_FIXED_STREAMING_SVE_BITS={first.sme_bytes * 8}" in definitions
  assert config.target_arch == "Native"


def test_compile_cache_key_freezes_state_and_distinguishes_reversed_widths(monkeypatch):
  state_a = VectorState(32, 64)
  state_b = VectorState(64, 32)
  _mock_state(monkeypatch, state_a)
  compiler_a = vecops.Compiler(target="Native", cache_mode="cache-only").snapshot()
  key_a = _compiler_key(compiler_a)

  _mock_state(monkeypatch, state_b)
  compiler_b = vecops.Compiler(target="Native", cache_mode="cache-only").snapshot()
  key_b = _compiler_key(compiler_b)

  assert key_a != key_b
  assert _compiler_key(compiler_a) == key_a
  assert compiler_a.vector_state == state_a
  assert not compiler_a.is_environment_current()
  assert compiler_b.is_environment_current()


def test_jit_factory_cache_key_tracks_current_vector_state(monkeypatch):
  states = iter((VectorState(32, 64), VectorState(32, 64), VectorState(64, 32)))
  monkeypatch.setattr("vecops._arm_vector_state.get_vector_state", lambda: next(states))
  calls = []

  @vecops.language.jit
  def make_value():
    calls.append(len(calls))
    return object()

  first = make_value()
  assert make_value() is first
  second = make_value()
  assert second is not first
  assert len(calls) == 2


def test_driver_passes_compiler_vector_snapshot_into_environment(monkeypatch):
  from vecops.language._compiler import driver
  from vecops.language._debug import report

  state = VectorState(32, 64)
  _mock_state(monkeypatch, state)
  captured = {}

  class StopAfterLower(Exception):
    pass

  def capture_lower(*_args, **kwargs):
    captured.update(kwargs)
    raise StopAfterLower

  monkeypatch.setattr(report, "lower", capture_lower)

  @vecops.language.template
  def template_function():
    pass

  with pytest.raises(StopAfterLower):
    driver.compile(template_function, compiler=vecops.Compiler(target="Native"))
  assert captured["pass_options"]["_vector_state"] == state


def test_build_identity_includes_vector_state_without_running_a_compiler(monkeypatch, tmp_path):
  compiler_a_state = VectorState(32, 64)
  compiler_b_state = VectorState(64, 32)
  compiler_a = vecops.Compiler(cache_dir=tmp_path / "cache", target="Native").snapshot()
  compiler_b = vecops.Compiler(cache_dir=tmp_path / "cache", target="Native").snapshot()
  compiler_a._vector_state = compiler_a_state
  compiler_b._vector_state = compiler_b_state

  monkeypatch.setattr(identity_module, "implementation_identity", lambda: "mock-sdk")
  monkeypatch.setattr(compiler_module, "_tool", lambda *_args: Path(sys.executable))

  def fake_run(command, **_kwargs):
    if "-print-file-name=" in command[-1]:
      return SimpleNamespace(stdout=sys.executable, stderr="")
    return SimpleNamespace(stdout="", stderr="#include <...> search starts here:\nEnd of search list\n")

  monkeypatch.setattr(subprocess, "run", fake_run)
  identity_a = identity_module.compiler_identity(compiler_a)
  identity_b = identity_module.compiler_identity(compiler_b)
  assert identity_a != identity_b


def test_operator_retains_compiler_vector_snapshot(monkeypatch, tmp_path):
  state_a = VectorState(32, 64)
  state_b = VectorState(64, 32)
  _mock_state(monkeypatch, state_a)
  monkeypatch.setattr(identity_module, "source_inputs", lambda *_args, **_kwargs: ())
  monkeypatch.setattr(identity_module, "include_roots", lambda _compiler: ())
  monkeypatch.setattr(identity_module, "compiler_identity", lambda _compiler: "mock-build")
  monkeypatch.setattr(vecops._C, "Operator", lambda *_args: object())

  class FakeDefinition:
    name = "snapshot_test"
    parameters = ()
    values: ClassVar[dict] = {}

    @staticmethod
    def _native():
      return object()

  operator = vecops.Operator(
    tmp_path / "kernel.cpp",
    FakeDefinition(),
    vecops.Compiler(target="Native", cache_mode="cache-only"),
    build_identity="mock-build",
  )
  _mock_state(monkeypatch, state_b)
  assert operator.compiler.vector_state == state_a
  assert not operator.compiler.is_environment_current()


def test_private_stdlib_flags_attest_actual_header_and_runtime_search(monkeypatch, tmp_path):
  monkeypatch.setattr(identity_module, "implementation_identity", lambda: "mock-sdk")
  monkeypatch.setattr(compiler_module, "_tool", lambda *_args: Path(sys.executable))
  monkeypatch.setenv("CXXFLAGS", "--gcc-install-dir=/ignored-global-prefix")
  compiler = vecops.Compiler(
    cache_dir=tmp_path / "cache",
    cflags=("-fno-exceptions",),
    environment={"CXXFLAGS": "--gcc-install-dir=/project/private/gcc/14 '-L/project/private library'"},
  )
  commands = []

  def fake_run(command, **kwargs):
    commands.append(command)
    assert kwargs["env"]["CXXFLAGS"] == compiler.environment["CXXFLAGS"]
    if command[-1].startswith("-print-file-name="):
      return SimpleNamespace(stdout=sys.executable, stderr="")
    return SimpleNamespace(stdout="", stderr="#include <...> search starts here:\nEnd of search list\n")

  monkeypatch.setattr(subprocess, "run", fake_run)
  identity_module.compiler_identity(compiler)
  assert len(commands) == 4
  for command in commands:
    assert command[1:4] == [
      "--gcc-install-dir=/project/private/gcc/14",
      "-L/project/private library",
      "-fno-exceptions",
    ]
    assert "--gcc-install-dir=/ignored-global-prefix" not in command
