# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Permission failures must remain explicit content identities, never cache hits."""

from __future__ import annotations

import errno
import hashlib
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest
import vecops._watch as watch
from vecops._identity import file_identity, tree_identity


@pytest.mark.skipif(os.uname().sysname != "Linux", reason="Linux dnotify fallback")
def test_native_dnotify_budget_does_not_exhaust_process_descriptors(tmp_path):
  from dsl_test_utils import cpp_compile_command

  root = Path(__file__).parents[2]
  executable = tmp_path / "watch-budget"
  source = Path(__file__).with_name("content_watch_reference.cpp")
  subprocess.run(
    cpp_compile_command(source, executable, flags=("-std=c++20", "-O2", "-pthread", f"-I{root / 'python/bindings'}")),
    check=True,
    capture_output=True,
    text=True,
  )
  directories = tmp_path / "watched"
  directories.mkdir()
  result = subprocess.run([str(executable), str(directories)], capture_output=True, text=True)
  if result.returncode == 77:
    pytest.skip("native dnotify unavailable")
  assert result.returncode == 0, result.stdout + result.stderr


def _clear_watch_caches():
  with watch._lock:
    watch._files.clear()
    watch._trees.clear()
    watch._watches.clear()


def _identity_text(value):
  return repr(value)


def test_tree_scan_permission_is_explicit_and_not_memoized(tmp_path, monkeypatch):
  blocked = tmp_path / "unreadable"
  blocked.mkdir()
  payload = blocked / "header.h"
  payload.write_text("first version")
  readable = tmp_path / "readable.h"
  readable.write_text("visible before")
  real_scandir = os.scandir
  _clear_watch_caches()

  def denied_scandir(path):
    if Path(path).absolute() == blocked.absolute():
      raise PermissionError(errno.EACCES, "permission denied", str(path))
    return real_scandir(path)

  monkeypatch.setattr(os, "scandir", denied_scandir)
  failed = tree_identity(tmp_path)
  failure_text = _identity_text(failed)
  failed_records = dict(failed)
  assert str(blocked.absolute()) in failure_text
  assert "scandir" in failure_text or "directory" in failure_text
  assert str(errno.EACCES) in failure_text or "EACCES" in failure_text
  assert failed_records[str(blocked.absolute())].startswith("unreadable:")
  assert hashlib.sha256(b"visible before").hexdigest() in failed_records.values()

  monkeypatch.setattr(os, "scandir", real_scandir)
  payload.write_text("second version")
  readable.write_text("visible after!")
  recovered = tree_identity(tmp_path)
  assert recovered != failed
  recovered_records = dict(recovered)
  assert all(not value.startswith("unreadable:") for value in recovered_records.values())
  assert str(payload.absolute()) in recovered_records
  assert recovered_records[str(payload.absolute())] == hashlib.sha256(b"second version").hexdigest()
  assert recovered_records[str(readable.absolute())] == hashlib.sha256(b"visible after!").hexdigest()


def test_content_read_permission_is_explicit_and_not_memoized(tmp_path, monkeypatch):
  source = tmp_path / "source.h"
  source.write_text("before")
  real_open = Path.open
  _clear_watch_caches()

  def denied_open(path, *args, **kwargs):
    if path.absolute() == source.absolute():
      raise PermissionError(errno.EACCES, "permission denied", str(path))
    return real_open(path, *args, **kwargs)

  monkeypatch.setattr(Path, "open", denied_open)
  failed = file_identity(source)
  failure_text = _identity_text(failed)
  assert failed[0] == str(source.absolute())
  assert str(source.absolute()) in failure_text
  assert failed[1] == f"unreadable:open:{errno.EACCES}"

  monkeypatch.setattr(Path, "open", real_open)
  source.write_text("after!")
  recovered = file_identity(source)
  assert recovered != failed
  assert recovered[1] == hashlib.sha256(b"after!").hexdigest()


@pytest.mark.parametrize(
  "operation,sentinel",
  [("resolve", "resolve_parent"), ("is_symlink", "lstat_symlink")],
)
def test_content_metadata_permission_is_explicit_and_not_memoized(tmp_path, monkeypatch, operation, sentinel):
  source = tmp_path / "source.h"
  source.write_text("before")
  real_operation = getattr(Path, operation)
  blocked = source.parent if operation == "resolve" else source
  _clear_watch_caches()

  def denied(path, *args, **kwargs):
    if path.absolute() == blocked.absolute():
      raise PermissionError(errno.EACCES, "permission denied", str(path))
    return real_operation(path, *args, **kwargs)

  monkeypatch.setattr(Path, operation, denied)
  failed = file_identity(source)
  assert failed == (str(source.absolute()), f"unreadable:{sentinel}:{errno.EACCES}")
  assert not watch._files

  monkeypatch.setattr(Path, operation, real_operation)
  source.write_text("after!")
  assert file_identity(source)[1] == hashlib.sha256(b"after!").hexdigest()


def test_tree_root_resolution_permission_is_explicit_and_not_memoized(tmp_path, monkeypatch):
  source = tmp_path / "source.h"
  source.write_text("before")
  real_resolve = Path.resolve
  _clear_watch_caches()

  def denied(path, *args, **kwargs):
    if path.absolute() == tmp_path.absolute():
      raise PermissionError(errno.EACCES, "permission denied", str(path))
    return real_resolve(path, *args, **kwargs)

  monkeypatch.setattr(Path, "resolve", denied)
  assert tree_identity(tmp_path) == ((str(tmp_path.absolute()), f"unreadable:resolve_root:{errno.EACCES}"),)
  assert not watch._trees

  monkeypatch.setattr(Path, "resolve", real_resolve)
  assert dict(tree_identity(tmp_path))[str(source.absolute())] == hashlib.sha256(b"before").hexdigest()


def test_missing_and_unreadable_explicit_sources_have_distinct_identities(tmp_path, monkeypatch):
  missing = tmp_path / "does-not-exist.h"
  missing_identity = file_identity(missing)
  assert missing_identity[0] == str(missing.absolute())
  assert "missing" in _identity_text(missing_identity).lower()

  source = tmp_path / "protected.h"
  source.write_text("protected")
  real_open = Path.open

  def denied_open(path, *args, **kwargs):
    if path.absolute() == source.absolute():
      raise PermissionError(errno.EACCES, "permission denied", str(path))
    return real_open(path, *args, **kwargs)

  _clear_watch_caches()
  monkeypatch.setattr(Path, "open", denied_open)
  denied_identity = file_identity(source)
  assert denied_identity[0] == str(source.absolute())
  assert denied_identity[1] == f"unreadable:open:{errno.EACCES}"
  assert denied_identity != missing_identity


def test_tree_stat_permission_is_explicit_for_nested_system_header_dir(tmp_path, monkeypatch):
  opencsd = tmp_path / "opencsd"
  c_api = opencsd / "c_api"
  c_api.mkdir(parents=True)
  header = c_api / "public.h"
  header.write_text("before")
  real_scandir = os.scandir
  _clear_watch_caches()

  class Entry:
    def __init__(self, path, *, directory=False, fail_stat=False):
      self.path = str(path)
      self.name = Path(path).name
      self.directory = directory
      self.fail_stat = fail_stat

    def is_symlink(self):
      return False

    def is_dir(self, *, follow_symlinks=True):
      if self.fail_stat:
        raise PermissionError(errno.EACCES, "permission denied", self.path)
      return self.directory

    def is_file(self, *, follow_symlinks=True):
      return not self.directory

  def denied_stat_scandir(path):
    path = Path(path)
    if path == tmp_path:
      return nullcontext([Entry(opencsd, directory=True)])
    if path == opencsd:
      return nullcontext([Entry(c_api, fail_stat=True)])
    return real_scandir(path)

  monkeypatch.setattr(os, "scandir", denied_stat_scandir)
  failed = tree_identity(tmp_path)
  failed_records = dict(failed)
  assert failed_records[str(c_api.absolute())] == f"unreadable:stat:{errno.EACCES}"

  monkeypatch.setattr(os, "scandir", real_scandir)
  header.write_text("after!")
  recovered_records = dict(tree_identity(tmp_path))
  assert recovered_records[str(header.absolute())] == hashlib.sha256(b"after!").hexdigest()
  assert all(not value.startswith("unreadable:") for value in recovered_records.values())
