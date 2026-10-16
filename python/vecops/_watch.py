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
  import hashlib

  path = Path(path)
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


def tree_hash(root, suffixes=None):
  root = Path(root)
  try:
    root.resolve()
  except FileNotFoundError:
    return ((str(root.absolute()), "missing"),)
  except OSError as error:
    return ((str(root.absolute()), f"unreadable:resolve_root:{error.errno}"),)
  records = []
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
          records.append((str(path.absolute()), content_hash(path)))
      except OSError as error:
        records.append((str(path.absolute()), f"unreadable:stat:{error.errno}"))
  return tuple(sorted(records))
