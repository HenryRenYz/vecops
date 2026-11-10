# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Content identities read current bytes on every call.

Filesystem notifications cannot observe every write (for example an existing
writable shared mapping), so file and directory content hashes are never
persistently memoized. Metadata probes preserve explicit identity diagnostics;
they are not substitutes for reading content. A scan is not an atomic snapshot
of a concurrently mutating filesystem.
"""

import os
from pathlib import Path


def content_hash(path):
  return _content_hash(path, check_parent=True)


def _content_hash(path, *, check_parent):
  import hashlib

  path = Path(path)
  if check_parent:
    try:
      path.parent.resolve()
    except FileNotFoundError:
      return "missing"
    except OSError as error:
      return f"unreadable:resolve_parent:{error.errno}"
  try:
    path.is_symlink()
  except FileNotFoundError:
    return "missing"
  except OSError as error:
    return f"unreadable:lstat_symlink:{error.errno}"
  try:
    with path.open("rb") as stream:
      hasher = hashlib.sha256()
      for chunk in iter(lambda: stream.read(1048576), b""):
        hasher.update(chunk)
      return hasher.hexdigest()
  except FileNotFoundError:
    return "missing"
  except OSError as error:
    # Unrelated inaccessible headers must remain explicit in the identity.
    return f"unreadable:open:{error.errno}"


def tree_hash(root, suffixes=None, *, parallelism=1, cpus=None):
  root = Path(root)
  try:
    root.resolve()
  except FileNotFoundError:
    return ((str(root.absolute()), "missing"),)
  except OSError as error:
    return ((str(root.absolute()), f"unreadable:resolve_root:{error.errno}"),)
  records = []
  large_files = []
  pending = [root]
  while pending:
    directory = pending.pop()
    try:
      directory.resolve()
      directory.is_symlink()
      with os.scandir(directory) as entries:
        children = sorted(entries, key=lambda entry: entry.name)
    except FileNotFoundError:
      records.append((str(directory.absolute()), "missing"))
      continue
    except OSError as error:
      records.append((str(directory.absolute()), f"unreadable:scandir:{error.errno}"))
      continue
    for entry in children:
      if entry.name == "__pycache__":
        continue
      path = Path(entry.path)
      try:
        entry.is_symlink()
        # Directory links are never followed; file links retain content semantics.
        if entry.is_dir(follow_symlinks=False):
          pending.append(path)
        elif entry.is_file() and (suffixes is None or path.suffix in suffixes):
          # This directory was resolved before scandir. Repeating that full
          # ancestor walk for every header dominates large toolchain trees.
          # Still lstat and read every leaf afresh; no timestamp/hash cache.
          # Large Torch libraries are independent SHA256 streams. Parallelize
          # their reads, retaining the exact serial identity and a fresh scan
          # on every call. Small header inventories stay on the serial path.
          size = 0
          if parallelism > 1 and path.suffix in {".so", ".a", ".dylib"}:
            try:
              size = entry.stat().st_size
            except OSError:
              pass
          if size >= 16 << 20:
            large_files.append(path)
          else:
            records.append((str(path.absolute()), _content_hash(path, check_parent=False)))
      except OSError as error:
        records.append((str(path.absolute()), f"unreadable:stat:{error.errno}"))
  if large_files:
    def read(path):
      return str(path.absolute()), _content_hash(path, check_parent=False)
    if len(large_files) == 1 or parallelism <= 1:
      records.extend(map(read, large_files))
    else:
      from concurrent.futures import ThreadPoolExecutor
      def initialize():
        if cpus is not None and hasattr(os, "sched_setaffinity"):
          os.sched_setaffinity(0, cpus)
      with ThreadPoolExecutor(max_workers=min(parallelism, len(large_files)),
                              initializer=initialize, thread_name_prefix="vecops-hash") as pool:
        records.extend(pool.map(read, large_files))
  return tuple(sorted(records))
