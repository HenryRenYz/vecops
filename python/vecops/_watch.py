"""Optional Linux content-hash acceleration using kernel mutation events.

Not a size/mtime shortcut: inotify falls back to native Linux dnotify when
unavailable; without either facility hashes are read from content again.
A mutation invalidates all cached hashes;
the native compiler still validates before and after a cold build.
"""

import ctypes
import os
import struct
import sys
import threading
from pathlib import Path

_lock = threading.RLock()
_files, _trees, _watches = {}, {}, set()
_generation = 0
_native_generation = None
_libc, _descriptor = None, -1
if sys.platform == "linux":
  try:
    _libc = ctypes.CDLL(None, use_errno=True)
    _descriptor = _libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
  except (AttributeError, OSError):
    pass


def _after_fork():
  """A child must not consume the parent's inotify event queue."""
  global _lock, _descriptor, _native_generation
  _lock = threading.RLock()
  _files.clear()
  _trees.clear()
  _watches.clear()
  _native_generation = None
  if _descriptor >= 0:
    os.close(_descriptor)
  _descriptor = _libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC) if _libc is not None else -1


if hasattr(os, "register_at_fork"):
  os.register_at_fork(after_in_child=_after_fork)


def _drain():
  global _generation, _descriptor, _native_generation
  from . import _C

  if hasattr(_C, "content_watch_epoch"):
    epoch = _C.content_watch_epoch()
    if epoch != _native_generation:
      _files.clear()
      _trees.clear()
      _watches.clear()
      _generation += 1
      _native_generation = epoch
  if _descriptor < 0:
    return
  try:
    while True:
      events = os.read(_descriptor, 65536)
      if not events:
        break
      _files.clear()
      _trees.clear()
      _generation += 1
      cursor = 0
      while cursor < len(events):
        _, mask, _, length = struct.unpack_from("iIII", events, cursor)
        cursor += 16 + length
        if mask & (0x8000 | 0x400 | 0x800):
          _watches.clear()
        if mask & 0x4000:  # Overflow: no cached hash may be trusted.
          os.close(_descriptor)
          _descriptor = -1
          _watches.clear()
          return
  except BlockingIOError:
    pass
  except OSError:
    _files.clear()
    _trees.clear()


def _watch_directory(directory):
  directory = str(Path(directory).resolve())
  if directory not in _watches:
    mask = 0x00000FCE  # exclude read-open and close-without-write events
    if _descriptor >= 0 and _libc.inotify_add_watch(_descriptor, os.fsencode(directory), mask) >= 0:
      _watches.add(directory)
    else:
      from . import _C

      if (
        hasattr(_C, "content_watch_directory")
        and _C.content_watch_epoch() >= 0
        and _C.content_watch_directory(Path(directory))
      ):
        _watches.add(directory)
  return directory in _watches


def content_hash(path):
  import hashlib

  path = Path(path)
  # Watch the resolved parent's entries: replacing the file invalidates its
  # prior contents too. Symlink chains can change outside that directory, so
  # symlink spellings are deliberately not memoized.
  with _lock:
    _drain()
    parent = str(path.parent.resolve())
    watched = False
    if not path.is_symlink():
      watched = _watch_directory(parent)
    key = (str(path.absolute()), parent)
    if watched and key in _files:
      return _files[key]
    generation = _generation
    try:
      with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1048576), b""):
          hasher.update(chunk)
        result = hasher.hexdigest()
    except FileNotFoundError:
      return "missing"
    _drain()
    if watched and generation == _generation:
      _files[key] = result
    return result


def tree_hash(root, suffixes=None):
  root = Path(root)
  key = (str(root.absolute()), str(root.resolve()), tuple(sorted(suffixes)) if suffixes else None)
  with _lock:
    _drain()
    if key in _trees:
      return _trees[key]
    paths = sorted(root.rglob("*"))
    watched = _watch_directory(root)
    for directory in paths:
      if "__pycache__" in directory.parts:
        continue
      if directory.is_dir():
        watched = _watch_directory(directory) and watched
      # Resolving symlink inventories needs an independent ancestry watch;
      # fall back instead of claiming that link targets cannot change.
      if directory.is_symlink():
        watched = False
    generation = _generation
    results = tuple(
      (str(p.absolute()), content_hash(p))
      for p in paths
      if p.is_file() and (suffixes is None or p.suffix in suffixes) and "__pycache__" not in p.parts
    )
    _drain()
    if watched and generation == _generation:
      _trees[key] = results
    return results
