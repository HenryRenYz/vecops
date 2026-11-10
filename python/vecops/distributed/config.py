# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Communication policy; hardware and process grants belong to MemoryRuntime."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class Policy:
  path: str | None = None
  communication: dict = field(default_factory=dict)
  fingerprint: str = 'automatic'

  @classmethod
  def load(cls, path=None):
    path = path or os.environ.get('VECOPS_COMM_CONFIG')
    if not path:
      return cls()
    try:
      import tomllib
    except ImportError:
      import tomli as tomllib
    path = Path(path).expanduser().resolve()
    with path.open('rb') as stream:
      data = tomllib.load(stream)
    if unknown := set(data) - {'schema', 'communication'}:
      raise ValueError(f'unknown communication policy sections: {sorted(unknown)}; hardware belongs to MemoryRuntime')
    if data.get('schema', 'communication-policy/0.1') != 'communication-policy/0.1':
      raise ValueError('unsupported communication policy schema')
    comm = data.get('communication', {})
    if not isinstance(comm, dict):
      raise ValueError('communication must be a TOML table')
    supported = {'algorithm', 'initial_capacity_bytes', 'relay_threshold_bytes', 'chunk_bytes',
                 'timeout_ms', 'calibration', 'copy_schedule', 'early_snapshot', 'minimum_gain',
                 'stage_us', 'sync_warmup', 'sync_rounds'}
    if unknown := set(comm) - supported:
      raise ValueError(f'unsupported communication policy fields: {sorted(unknown)}')
    if comm.get('algorithm', 'auto') not in {'auto', 'flat'}:
      raise ValueError('unknown communication algorithm; automatic candidates require resource/cost evidence')
    if comm.get('copy_schedule', 'auto') not in {'auto', 'peer'}:
      raise ValueError('copy_schedule must be auto or peer')
    if type(comm.get('early_snapshot', False)) is not bool:
      raise ValueError('early_snapshot must be a bool')
    for key in ('initial_capacity_bytes', 'chunk_bytes', 'timeout_ms'):
      if key in comm and (type(comm[key]) is not int or comm[key] <= 0):
        raise ValueError(f'{key} must be a positive integer')
    for key in ('relay_threshold_bytes', 'sync_warmup'):
      if key in comm and (type(comm[key]) is not int or comm[key] < 0):
        raise ValueError(f'{key} must be a nonnegative integer')
    if type(comm.get('sync_rounds', 7)) is not int or not 3 <= comm.get('sync_rounds', 7) <= 31:
      raise ValueError('sync_rounds must be an integer in [3,31]')
    if comm.get('sync_warmup', 3) > 31:
      raise ValueError('sync_warmup must not exceed 31')
    if 'stage_us' in comm and (type(comm['stage_us']) not in (float, int)
                             or not math.isfinite(comm['stage_us']) or comm['stage_us'] < 0):
      raise ValueError('stage_us must be finite and nonnegative')
    gain = comm.get('minimum_gain', .05)
    if type(gain) not in (float, int) or not math.isfinite(gain) or not 0 <= gain < 1:
      raise ValueError('minimum_gain must be in [0,1)')
    if 'calibration' in comm and (not isinstance(comm['calibration'], str) or not comm['calibration']):
      raise ValueError('calibration must name a profile file')
    identity = {**data, 'communication': {key: value for key, value in comm.items() if key != 'calibration'}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return cls(str(path), comm, fingerprint)

  def describe(self):
    return {'path': self.path, 'fingerprint': self.fingerprint, 'communication': self.communication}


@dataclass(frozen=True)
class Hints:
  """Explicit performance policy; never grant alias or lifetime permissions."""
  reuse: str = 'unknown'
  tag: str = ''
  copy_levels: tuple[int, ...] | None = None

  def __post_init__(self):
    if self.copy_levels is not None and (not self.copy_levels or any(type(v) is not int or v not in (0, 1) for v in self.copy_levels)):
      raise ValueError('copy_levels must be a nonempty tuple of share=0/copy=1 modes')


@dataclass
class Options:
  config: str | None = None
  threads: int = 0
  timeout_ms: int | None = None

  @classmethod
  def from_file(cls, path, **kwargs):
    Policy.load(path)
    return cls(str(Path(path).expanduser().resolve()), **kwargs)
