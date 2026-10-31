# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Preparation-only provenance checks for the optional glibc audit cache.

The loader owns redirection. This module never loads/copies a library, changes
page policy, or enters a prepared graph's execution path.
"""
from __future__ import annotations

import os
from pathlib import Path


def is_loader_copy(original: Path, mapped: Path) -> bool:
  """Prove both the v1 source association and byte identity of a mapped DSO.

  Equal filenames/SONAMEs alone are insufficient for runtime provenance. The
  caller must additionally prove all function addresses belong to this exact
  mapped object, retaining its NOLOAD handle (not open another runtime).
  """
  if not os.environ.get('VECOPS_NUMA_PLACEMENT'):
    return False
  configured = os.environ.get('VECOPS_NUMA_CACHE_DIR')
  if configured is None:
    xdg = os.environ.get('XDG_CACHE_HOME', '')
    if xdg.startswith('/'):
      configured = str(Path(xdg) / 'vecops/numa-loader')
    else:
      home = os.environ.get('HOME', '')
      if not home.startswith('/'):
        return False
      configured = str(Path(home) / '.cache/vecops/numa-loader')
  root = Path(configured)
  if not root.is_absolute():
    return False
  try:
    original, mapped, root = original.resolve(), mapped.resolve(), root.resolve()
    parts = mapped.relative_to(root).parts
    if len(parts) < 6 or parts[0] != 'v1' or parts[4] != 'root':
      return False
    if Path('/', *parts[5:]) != original:
      return False
    source, copy = original.stat(), mapped.stat()
    if (source.st_dev, source.st_ino) == (copy.st_dev, copy.st_ino):
      return False
    if copy.st_uid != os.geteuid() or copy.st_nlink != 1 or copy.st_mode & 0o022:
      return False
    from ._watch import content_hash
    digest = content_hash(original)
    return parts[3].endswith('-' + digest) and content_hash(mapped) == digest
  except (OSError, ValueError):
    return False
