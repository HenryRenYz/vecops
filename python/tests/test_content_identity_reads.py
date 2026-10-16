# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Every identity call observes bytes, including unobservable filesystem writes."""

import hashlib
import mmap
import os
from pathlib import Path

import pytest
import vecops._watch as watch


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root, file):
  assert watch.content_hash(file) == digest(file)
  expected = tuple(sorted((str(p.absolute()), digest(p)) for p in root.rglob("*") if p.is_file()))
  assert watch.tree_hash(root) == expected


def create(root):
  root.mkdir(parents=True)
  file = root / "a.h"
  file.write_bytes(b"AAAA")
  check(root, file)
  return file


def test_shared_mapping_changes_without_close_events(tmp_path):
  file = create(tmp_path / "root")
  with file.open("r+b") as stream:
    with mmap.mmap(stream.fileno(), 0) as mapping:
      check(file.parent, file)
      mapping[:] = b"CCCC"
      mapping.flush()
      check(file.parent, file)
      mapping[:] = b"DDDD"
      mapping.flush()
      check(file.parent, file)


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_ancestor_rebinding_reads_current_content(tmp_path, kind):
  root = tmp_path / "container/root"
  file = create(root)
  metadata = file.stat()
  if kind == "directory":
    root.parent.rename(tmp_path / "original-container")
    root.mkdir(parents=True)
  else:
    other = tmp_path / "other/root"
    other.mkdir(parents=True)
    root.parent.rename(tmp_path / "original-container")
    root.parent.symlink_to(other.parent, target_is_directory=True)
  file.write_bytes(b"CCCC")
  os.utime(file, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
  (root / "new.h").write_bytes(b"NEW")
  check(root, file)


@pytest.mark.parametrize("transient", [False, True])
def test_outside_hardlink_writes_preserve_identity_correctness(tmp_path, transient):
  file = create(tmp_path / "root")
  metadata = file.stat()
  other = tmp_path / "outside.h"
  os.link(file, other)
  other.write_bytes(b"CCCC")
  os.utime(other, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
  if transient:
    other.unlink()
    assert file.stat().st_nlink == 1
  check(file.parent, file)


def test_same_size_mtime_rewrite_is_rehashed(tmp_path):
  file = create(tmp_path / "root")
  metadata = file.stat()
  file.write_bytes(b"CCCC")
  os.utime(file, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
  assert file.stat().st_size == metadata.st_size
  assert file.stat().st_mtime_ns == metadata.st_mtime_ns
  check(file.parent, file)


def test_symlink_and_inventory_filtering_semantics(tmp_path):
  root = tmp_path / "root"
  file = create(root)
  outside = tmp_path / "outside"
  outside.mkdir()
  target = outside / "target.h"
  target.write_bytes(b"TARGET")
  (root / "file.h").symlink_to(target)
  (root / "directory").symlink_to(outside, target_is_directory=True)
  (root / "__pycache__").mkdir()
  (root / "__pycache__/ignored.h").write_bytes(b"IGNORED")
  (root / "ignored.txt").write_bytes(b"NO")
  records = watch.tree_hash(root, suffixes={".h"})
  assert records == tuple(
    sorted([(str(file.absolute()), digest(file)), (str((root / "file.h").absolute()), digest(target))])
  )
  alias = tmp_path / "alias"
  alias.symlink_to(root, target_is_directory=True)
  assert dict(watch.tree_hash(alias, suffixes={".h"}))[str((alias / "a.h").absolute())] == digest(file)
  target.write_bytes(b"CHANGED")
  assert dict(watch.tree_hash(root, suffixes={".h"}))[str((root / "file.h").absolute())] == digest(target)


def test_identity_does_not_depend_on_native_watch_events(tmp_path, monkeypatch):
  from vecops import _C

  def unavailable(*args):
    pytest.fail("content identity consulted optional event facility")

  monkeypatch.setattr(_C, "content_watch_directory", unavailable)
  monkeypatch.setattr(_C, "content_watch_epoch", unavailable)
  file = create(tmp_path / "root")
  file.write_bytes(b"CCCC")
  check(file.parent, file)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork")
def test_child_process_reads_current_bytes(tmp_path):
  file = create(tmp_path / "root")
  file.write_bytes(b"CCCC")
  reader, writer = os.pipe()
  pid = os.fork()
  if pid == 0:
    try:
      os.close(reader)
      check(file.parent, file)
      os.write(writer, b"PASS")
    finally:
      os._exit(0)
  os.close(writer)
  result = os.read(reader, 16)
  os.close(reader)
  actual_pid, status = os.waitpid(pid, 0)
  assert actual_pid == pid and status == 0 and result == b"PASS"


def test_each_call_opens_content_again(tmp_path, monkeypatch):
  file = create(tmp_path / "root")
  opening = Path.open
  calls = []

  def open_current(path, *args, **kwargs):
    if path == file and args and args[0] == "rb":
      calls.append(path)
    return opening(path, *args, **kwargs)

  monkeypatch.setattr(Path, "open", open_current)
  watch.content_hash(file)
  watch.content_hash(file)
  watch.tree_hash(file.parent)
  assert calls == [file, file, file]
