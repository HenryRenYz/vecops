# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Strict, content-bound exact catalog selections for resumable benchmarks."""
import hashlib
import json
from pathlib import Path


def selection_identity(path, architecture, field):
  if path is None:
    return None
  path = Path(path).resolve()
  data = path.read_bytes()
  document = json.loads(data)
  if not isinstance(document, dict) or document.get('architecture') != architecture:
    raise ValueError('selection architecture differs from runner architecture')
  ids = document.get(field)
  if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
    raise ValueError(f'{field} must be a nonempty list of nonempty strings')
  if len(set(ids)) != len(ids):
    raise ValueError(f'duplicate {field}')
  return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'architecture': architecture, field: ids}


def select_rows(rows, identity, field, key):
  rows = tuple(rows)
  if identity is None:
    return rows
  wanted = set(identity[field])
  known = {key(row) for row in rows}
  if wanted - known:
    raise ValueError(f'unknown or out-of-scope {field}: {sorted(wanted - known)}')
  return tuple(row for row in rows if key(row) in wanted)


def verify_selection_resume(previous, current):
  if previous != current:
    raise ValueError('resume selection differs from immutable plan')
