# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Toolchain tree hashing stays fresh without per-header ancestor walks."""

import hashlib
import os
from pathlib import Path

from vecops._watch import content_hash, tree_hash


def test_tree_hash_resolves_directories_instead_of_every_header(tmp_path, monkeypatch):
  for number in range(64):
    (tmp_path / f"header-{number}.h").write_text(f"constexpr int value = {number};\n")
  nested = tmp_path / "nested"
  nested.mkdir()
  (nested / "value.h").write_text("nested header\n")
  resolve = Path.resolve
  calls = []

  def counted(path, *args, **kwargs):
    calls.append(path)
    return resolve(path, *args, **kwargs)

  monkeypatch.setattr(Path, "resolve", counted)
  actual = tree_hash(tmp_path)
  assert len(calls) <= 3  # Root validation and one check per scanned directory.
  expected = tuple(sorted(
    (str(path.absolute()), hashlib.sha256(path.read_bytes()).hexdigest())
    for path in tmp_path.rglob("*.h")
  ))
  assert actual == expected


def test_tree_hash_detects_same_size_rewrite_with_restored_mtime(tmp_path):
  header = tmp_path / "value.h"
  header.write_bytes(b"old header\n")
  timestamp = header.stat()
  before = tree_hash(tmp_path)
  header.write_bytes(b"new header\n")
  os.utime(header, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
  assert tree_hash(tmp_path) != before
  assert dict(tree_hash(tmp_path))[str(header.absolute())] == content_hash(header)


def test_tree_hash_follows_file_contents_and_excludes_directory_links(tmp_path):
  root = tmp_path / "root"
  root.mkdir()
  target = tmp_path / "target.h"
  target.write_bytes(b"first header\n")
  (root / "header.h").symlink_to(target)
  (root / "outside").symlink_to(tmp_path, target_is_directory=True)
  before = tree_hash(root)
  assert before == ((str((root / "header.h").absolute()), content_hash(target)),)
  target.write_bytes(b"other header\n")
  assert tree_hash(root) != before
