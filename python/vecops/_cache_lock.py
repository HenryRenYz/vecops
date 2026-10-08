# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cross-process writer locks for local POSIX artifact caches."""

from contextlib import contextmanager


@contextmanager
def file_lock(path):
  import fcntl

  with path.open("a+b") as stream:
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
