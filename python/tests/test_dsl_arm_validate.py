# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""ARM validation runner safety and process-scoped SDK selection."""

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runner():
  path = Path(__file__).parents[2] / "benchmarks/dsl_arm_validate.py"
  spec = importlib.util.spec_from_file_location("vecops_test_arm_runner", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.fixture
def private_scratch():
  with tempfile.TemporaryDirectory(prefix="vecops-runner-test-", dir="/tmp") as directory:
    path = Path(directory)
    path.chmod(0o700)
    yield path


def mock_installed_sdk(tmp_path):
  root = tmp_path / "_sdk"
  for name in (
    "lib/cmake/Vecops/VecopsConfig.cmake",
    "lib/cmake/Vecops/VecopsTargets.cmake",
    "include/vecops/Kernel.h",
    "include/vecops/vec/Vec.h",
    "lib/libvecops.a",
  ):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test SDK contents")
  original = lambda: (tmp_path / "source", "source")
  native = SimpleNamespace(
    __file__=str(tmp_path / "_C.so"), SdkLayout=SimpleNamespace(package="package", source_tree="source")
  )
  return SimpleNamespace(_C=native, _sdk=original), root, original


def test_installed_sdk_override_is_explicit_and_restored_on_failure(runner, tmp_path):
  module, root, original = mock_installed_sdk(tmp_path)
  with pytest.raises(RuntimeError, match="test failure"), runner.sdk_selection(module, "installed") as selected:
    assert selected == (root, "package")
    assert module._sdk() == selected
    raise RuntimeError("test failure")
  assert module._sdk is original


def test_incomplete_installed_sdk_never_silently_falls_back(runner, tmp_path):
  module, root, original = mock_installed_sdk(tmp_path)
  (root / "lib/libvecops.a").unlink()
  with pytest.raises(RuntimeError, match="incomplete"), runner.sdk_selection(module, "installed"):
    pytest.fail("must reject incomplete SDK")
  assert module._sdk is original


def test_sdk_content_provenance_tracks_header_and_library_contents(runner, tmp_path):
  _, root, _ = mock_installed_sdk(tmp_path)
  first = runner.sdk_identity(root, installed=True)
  (root / "include/vecops/Kernel.h").write_bytes(b"changed header")
  second = runner.sdk_identity(root, installed=True)
  (root / "lib/libvecops.a").write_bytes(b"changed library")
  third = runner.sdk_identity(root, installed=True)
  assert len({first["sha256"], second["sha256"], third["sha256"]}) == 3


def test_basetemp_refuses_existing_results_and_broad_targets(runner, tmp_path):
  existing = tmp_path / "existing"
  existing.mkdir()
  (existing / "result").write_text("preserve", encoding="utf-8")
  with pytest.raises(ValueError, match="erase existing data"):
    runner.fresh_basetemp(existing, allow_local=True)
  with pytest.raises(ValueError, match="broad directory"):
    runner.fresh_basetemp(Path.home(), allow_local=True)
  assert (existing / "result").read_text() == "preserve"


def test_failure_events_preserve_full_diagnostics_and_ignore_passes(runner, tmp_path, capsys):
  path = tmp_path / "failures.jsonl"
  provenance = {"job_id": "unit-job", "vector_state": {"sve_bytes": 16, "sme_bytes": 64}}
  recorder = runner.FailureRecorder(path, provenance)
  passed = SimpleNamespace(failed=False, when="call")
  recorder.pytest_runtest_logreport(passed)
  assert path.read_text() == ""
  failure = SimpleNamespace(failed=True, nodeid="test_kernel[α]", when="setup", longreprtext="full\ntraceback α")
  recorder.pytest_runtest_logreport(failure)
  recorder.pytest_collectreport(failure)
  events = [json.loads(line) for line in path.read_text().splitlines()]
  assert [event["phase"] for event in events] == ["setup", "collection"]
  assert all(event["diagnostic"] == failure.longreprtext for event in events)
  assert all(event["provenance"] == provenance for event in events)
  assert "test failed (setup)" in capsys.readouterr().out
  with pytest.raises(FileExistsError):
    runner.FailureRecorder(path, provenance)
  assert len(path.read_text().splitlines()) == 2


def test_scratch_accepts_only_explicit_private_task_root(runner, private_scratch):
  assert runner.validate_scratch_root(private_scratch) == private_scratch
  basetemp = runner.fresh_basetemp(private_scratch / "pytest-run", scratch_root=private_scratch)
  assert basetemp == private_scratch / "pytest-run"
  assert runner.fresh_basetemp(None, scratch_root=private_scratch).parent == private_scratch
  assert runner.output_path_allowed(private_scratch / "results.xml", scratch_root=private_scratch)
  with pytest.raises(ValueError, match="home or the explicit"):
    runner.fresh_basetemp(private_scratch / "pytest-run")


def test_scratch_accepts_private_shared_memory_root(runner):
  if not Path("/dev/shm").is_dir() or not os.access("/dev/shm", os.W_OK):
    pytest.skip("shared-memory scratch is unavailable on this platform")
  with tempfile.TemporaryDirectory(prefix="vecops-runner-test-", dir="/dev/shm") as directory:
    path = Path(directory)
    path.chmod(0o700)
    assert runner.validate_scratch_root(path) == path
    assert runner.fresh_basetemp(path / "pytest-run", scratch_root=path) == path / "pytest-run"


@pytest.mark.parametrize("path", ["/tmp", "/dev/shm", "/tmp/other", "/tmp/vecops-", "vecops-private"])
def test_scratch_rejects_broad_unscoped_or_relative_roots(runner, path):
  with pytest.raises(ValueError, match="task-private"):
    runner.validate_scratch_root(path)


def test_scratch_rejects_home_source_nested_missing_and_traversal_roots(runner, private_scratch):
  for path in [
    Path.home(),
    runner.ROOT,
    private_scratch / "nested",
    private_scratch.with_name(private_scratch.name + "-missing"),
    private_scratch / ".." / private_scratch.name,
  ]:
    with pytest.raises(ValueError, match="task-private|existing directory"):
      runner.validate_scratch_root(path)


def test_scratch_rejects_public_permissions_and_foreign_owner(runner, private_scratch, monkeypatch):
  private_scratch.chmod(0o755)
  with pytest.raises(ValueError, match="current UID with permissions 0700"):
    runner.validate_scratch_root(private_scratch)
  private_scratch.chmod(0o700)
  uid = os.getuid()
  monkeypatch.setattr(runner.os, "getuid", lambda: uid + 1)
  with pytest.raises(ValueError, match="current UID with permissions 0700"):
    runner.validate_scratch_root(private_scratch)


def test_scratch_rejects_root_and_child_symlinks(runner, private_scratch):
  alias = private_scratch.with_name(private_scratch.name + "-alias")
  alias.symlink_to(private_scratch, target_is_directory=True)
  try:
    with pytest.raises(ValueError, match="symlinks"):
      runner.validate_scratch_root(alias)
  finally:
    alias.unlink()
  nested = private_scratch / "nested"
  nested.mkdir()
  (private_scratch / "alias").symlink_to(nested, target_is_directory=True)
  with pytest.raises(ValueError, match="symlinks"):
    runner.fresh_basetemp(private_scratch / "alias/pytest-run", scratch_root=private_scratch)


def test_scratch_basetemp_cannot_clear_root_or_existing_data(runner, private_scratch):
  with pytest.raises(ValueError, match="broad directory"):
    runner.fresh_basetemp(private_scratch, scratch_root=private_scratch)
  for broad in [Path("/tmp"), Path("/dev/shm"), Path.home(), runner.ROOT]:
    with pytest.raises(ValueError, match="broad directory"):
      runner.fresh_basetemp(broad, scratch_root=private_scratch)
  existing = private_scratch / "previous"
  existing.mkdir()
  (existing / "result.csv").write_text("preserve", encoding="utf-8")
  with pytest.raises(ValueError, match="erase existing data"):
    runner.fresh_basetemp(existing, scratch_root=private_scratch)
  assert (existing / "result.csv").read_text() == "preserve"


@pytest.mark.parametrize("fails", [False, True])
def test_validation_caches_are_private_fresh_and_restored(runner, private_scratch, monkeypatch, fails):
  monkeypatch.setenv("VECOPS_CACHE_DIR", "/application/cache")
  monkeypatch.setenv("VECOPS_TORCH_BRIDGE_DIR", "/application/bridges")
  monkeypatch.delenv("VECOPS_BUILD_DIR", raising=False)
  basetemp = private_scratch / "pytest"
  caches = {}
  try:
    with runner.validation_cache_selection(basetemp) as selected:
      caches.update(selected)
      for name, raw_path in selected.items():
        path = Path(raw_path)
        assert runner.os.environ[name] == raw_path
        assert path.is_relative_to(private_scratch)
        assert not path.is_relative_to(basetemp)
        assert path.is_dir() and path.stat().st_mode & 0o777 == 0o700
      if fails:
        raise RuntimeError("simulated compilation failure")
  except RuntimeError as error:
    assert fails and str(error) == "simulated compilation failure"
  assert os.environ["VECOPS_CACHE_DIR"] == "/application/cache"
  assert os.environ["VECOPS_TORCH_BRIDGE_DIR"] == "/application/bridges"
  assert "VECOPS_BUILD_DIR" not in os.environ
  assert all(Path(path).is_dir() for path in caches.values())
  with pytest.raises(FileExistsError), runner.validation_cache_selection(basetemp):
    pytest.fail("existing cache evidence must not be reused or overwritten")


def test_validation_cache_rejects_symlinked_output(runner, private_scratch):
  alias = private_scratch / "pytest-cache"
  alias.symlink_to(private_scratch, target_is_directory=True)
  with pytest.raises(ValueError, match="symlinks"), runner.validation_cache_selection(private_scratch / "pytest"):
    pytest.fail("must not follow an application-cache alias")


def test_scratch_scope_does_not_allow_other_shared_outputs(runner, private_scratch):
  outside = private_scratch.with_name(private_scratch.name + "-other") / "pytest-run"
  assert not runner.output_path_allowed(outside, scratch_root=private_scratch)
  with pytest.raises(ValueError, match="home or the explicit"):
    runner.fresh_basetemp(outside, scratch_root=private_scratch)


@pytest.mark.parametrize("option", ["--basetemp", "--basetemp=/tmp", "--junitxml", "--junitxml=/tmp/result.xml"])
def test_pytest_extra_options_cannot_override_validated_paths(runner, option):
  with pytest.raises(SystemExit) as error:
    runner.main(["--allow-local", "--", option])
  assert error.value.code == 2


@pytest.mark.parametrize("name", ["../test", "/usr/bin/script", "dsl_arm_validate", "not_a_benchmark"])
def test_benchmark_selector_is_scoped_to_known_local_scripts(runner, name):
  with pytest.raises(ValueError):
    runner.benchmark_path(name)


def test_benchmark_forwards_arguments_and_exit_code_and_restores_argv(runner, monkeypatch):
  previous_argv, previous_path = runner.sys.argv, list(runner.sys.path)
  script = runner.benchmark_path("dsl_norm_catalog")

  def run(path, *, run_name):
    assert path == str(script)
    assert run_name == "__main__"
    assert runner.sys.argv == [str(script), "--list", "--dtype", "float32"]
    assert runner.sys.path[0] == str(script.parent)
    raise SystemExit(7)

  monkeypatch.setattr(runner.runpy, "run_path", run)
  assert runner.run_benchmark(script, ["--list", "--dtype", "float32"]) == 7
  assert runner.sys.argv is previous_argv
  assert runner.sys.path == previous_path


def test_login_node_is_rejected_even_if_aarch64(runner, monkeypatch):
  monkeypatch.setattr(runner.platform, "machine", lambda: "aarch64")
  monkeypatch.setattr(runner.platform, "node", lambda: "login02")
  monkeypatch.setattr(runner.sys, "platform", "linux")
  with pytest.raises(RuntimeError, match="login node"):
    runner.check_machine()
  with pytest.raises(RuntimeError, match="login node"):
    runner.check_machine(standalone_arm=True)


def test_standalone_arm_requires_explicit_opt_in_and_arm_host(runner, monkeypatch):
  monkeypatch.setattr(runner.platform, "machine", lambda: "aarch64")
  monkeypatch.setattr(runner.platform, "node", lambda: "test14")
  monkeypatch.setattr(runner.sys, "platform", "linux")
  monkeypatch.delenv("CCS_JOB_ID", raising=False)
  with pytest.raises(RuntimeError, match="Donau"):
    runner.check_machine()
  runner.check_machine(standalone_arm=True)
  monkeypatch.setattr(runner.platform, "machine", lambda: "x86_64")
  with pytest.raises(RuntimeError, match="AArch64"):
    runner.check_machine(standalone_arm=True)
  with pytest.raises(ValueError, match="not both"):
    runner.check_machine(standalone_arm=True, allow_local=True)


def test_source_smoke_cannot_misreport_installed_sdk(runner, tmp_path):
  module, root, _ = mock_installed_sdk(tmp_path)
  module._sdk = lambda: (root, "package")
  with pytest.raises(RuntimeError, match="editable source SDK"), runner.sdk_selection(module, "source"):
    pytest.fail("installed SDK must not masquerade as a source SDK smoke")


@pytest.mark.parametrize("use_scratch", [False, True])
def test_main_forwards_pytest_selectors_and_failure_without_native_build(
  runner, monkeypatch, tmp_path, private_scratch, capsys, use_scratch
):
  import vecops._compiler as compiler
  import vecops._identity as identity

  source = tmp_path / "source"
  (source / "include/vecops").mkdir(parents=True)
  (source / "include/vecops/Kernel.h").write_text("header", encoding="utf-8")
  (source / "CMakeLists.txt").write_text("project(mock)", encoding="utf-8")
  monkeypatch.setattr(compiler, "_sdk", lambda: (source, compiler._C.SdkLayout.source_tree))
  monkeypatch.setattr(identity, "implementation_identity", lambda: "mock-implementation")
  monkeypatch.setattr(runner, "compiler_path", lambda name: "/tool/" + name)
  monkeypatch.setattr(runner, "check_machine", lambda **kwargs: None)
  monkeypatch.setenv("CC", "old-cc")
  monkeypatch.setenv("CXX", "old-cxx")
  monkeypatch.setenv("VECOPS_TEST_TARGET", "old-target")
  arguments = []

  def run(selected):
    arguments.extend(selected)
    assert runner.os.environ["CC"] == "/tool/clang"
    assert runner.os.environ["CXX"] == "/tool/clang++"
    assert runner.os.environ["VECOPS_TEST_TARGET"] == "Native"
    for name in ("VECOPS_CACHE_DIR", "VECOPS_TORCH_BRIDGE_DIR", "VECOPS_BUILD_DIR"):
      assert Path(runner.os.environ[name]).parent == basetemp.with_name(basetemp.name + "-cache")
    return 1

  monkeypatch.setattr(pytest, "main", run)
  output_root = private_scratch if use_scratch else tmp_path
  basetemp, junit = output_root / "pytest", output_root / "results.xml"
  scope_arguments = ["--scratch-root", str(private_scratch)] if use_scratch else ["--allow-local"]
  result = runner.main(
    [
      *scope_arguments,
      "--basetemp",
      str(basetemp),
      "--junitxml",
      str(junit),
      "python/tests/test_dsl.py::test_control_flow",
      "--",
      "-q",
      "-k",
      "control",
    ]
  )
  assert result == 1
  assert arguments == [
    "python/tests/test_dsl.py::test_control_flow",
    f"--basetemp={basetemp}",
    f"--junitxml={junit}",
    "-q",
    "-k",
    "control",
  ]
  output = capsys.readouterr().out
  assert '"source_implementation_identity": "mock-implementation"' in output
  assert ('"scratch_root": "' + str(private_scratch) + '"' if use_scratch else '"scratch_root": null') in output
  assert str(source) in output
