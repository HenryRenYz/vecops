# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Vecops-owned platform configuration, independent of model configuration.

Domain containment, OS memory targets and shared bandwidth resources have
separate identities. Unknown capacity is never interpreted as independent or
infinite bandwidth. Only implemented, validated algorithms may be selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Platform:
  path: str | None = None
  domains: tuple[dict, ...] = ()
  pools: tuple[dict, ...] = ()
  resources: tuple[dict, ...] = ()
  links: tuple[dict, ...] = ()
  communication: dict = field(default_factory=dict)
  fingerprint: str = 'automatic'
  root: str = 'host0'

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
    unknown = set(data) - {'topology', 'memory', 'resources', 'communication'}
    if unknown:
      raise ValueError(f'unknown vecops platform sections: {sorted(unknown)}')
    topology = data.get('topology', {})
    if topology.get('schema_version', 1) != 1:
      raise ValueError('unsupported vecops platform schema_version')
    domains = topology.get('domains', [])
    root = topology.get('root', 'host0')
    parents = {d['id']: d.get('parent', root) for d in domains}
    if len(parents) != len(domains) or root in parents:
      raise ValueError('duplicate/root domain identifier')
    occupied = set()
    for domain in domains:
      current, visited = domain['id'], set()
      while current != root:
        if current in visited or current not in parents:
          raise ValueError('cyclic or missing topology parent')
        visited.add(current)
        current = parents[current]
      for node in domain.get('numa_nodes', []):
        if node in occupied or not isinstance(node, int) or node < 0:
          raise ValueError('invalid or multiply owned CPU NUMA node')
        occupied.add(node)
    resources = data.get('resources', [])
    by_id = {r['id']: r for r in resources}
    if len(by_id) != len(resources):
      raise ValueError('duplicate bandwidth resource')
    for resource in resources:
      bandwidth = resource.get('capacity_GBps', 1)
      if not isinstance(bandwidth, (float, int)) or not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('bandwidth must be positive or omitted (unknown)')
      if resource.get('basis', 'payload_bytes') not in {'payload_bytes', 'read_write_bytes'}:
        raise ValueError('unknown resource byte accounting basis')
    pools = data.get('memory', {}).get('pools', [])
    owned_memory = set()
    for pool in pools:
      if pool['kind'] not in {'unknown', 'dram', 'hbm', 'cxl', 'pmem'}:
        raise ValueError('unknown memory kind')
      if pool.get('resource') and pool['resource'] not in by_id:
        raise ValueError('memory pool references missing bandwidth resource')
      for node in pool.get('os_nodes', []):
        if type(node) is not int or node<0:
          raise ValueError("OS memory nodes must be nonnegative integers")
        if node in owned_memory:
          raise ValueError('OS memory node belongs to multiple pools')
        owned_memory.add(node)
      owner = pool['owner']
      if owner != root and owner not in parents and not owner.startswith('numa:'):
        raise ValueError('memory pool references missing domain')
    for link in topology.get('links', []):
      if link['resource'] not in by_id or any(x not in parents and x != root for x in (link['a'], link['b'])):
        raise ValueError('topology link references missing domain/resource')
    comm = data.get('communication', {})
    supported = {'algorithm', 'direction', 'replica_domain', 'initial_capacity_bytes',
                 'relay_threshold_bytes', 'chunk_bytes', 'require_placement', 'timeout_ms', 'levels', 'calibration',
                 'copy_schedule'}
    if set(comm)-supported:
      raise ValueError(f'unsupported communication policy fields: {sorted(set(comm)-supported)}')
    if comm.get('algorithm', 'auto') not in {'auto', 'flat', 'hierarchical'}:
      raise ValueError('unknown communication algorithm')
    if comm.get('copy_schedule','auto') not in {'auto','peer'}:
      raise ValueError('copy_schedule must be auto or peer')
    if comm.get('direction', 'auto') != 'auto':
      raise ValueError('global direction overrides are unsupported: AllToAll uses push, AllGather uses pull')
    for key in ('initial_capacity_bytes', 'chunk_bytes', 'timeout_ms'):
      if key in comm and (type(comm[key]) is not int or comm[key] <= 0):
        raise ValueError(f'{key} must be a positive integer')
    if 'relay_threshold_bytes' in comm and (type(comm['relay_threshold_bytes']) is not int or comm['relay_threshold_bytes'] < 0):
      raise ValueError('relay_threshold_bytes must be a nonnegative integer')
    if type(comm.get('require_placement', False)) is not bool:
      raise ValueError('require_placement must be boolean')
    levels = comm.get('levels', [])
    if len(levels)>16:
      raise ValueError('at most 16 communication levels are supported')
    for level in levels:
      if set(level)-{'domain', 'mode'} or not isinstance(level.get('domain'), str) or level.get('mode') not in {'share', 'copy'}:
        raise ValueError('each communication level requires domain and share/copy mode')
    identity = {**data, 'communication': {k:v for k,v in comm.items() if k!='calibration'}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return cls(path=str(path), domains=tuple(domains), pools=tuple(pools),
               resources=tuple(resources), links=tuple(topology.get("links", [])),
               communication=comm, fingerprint=fingerprint, root=root)

  def resolve(self, cpu_node: int):
    """Return the configured replica domain and a concrete OS memory target."""
    parents = {d['id']: d for d in self.domains}
    owner = next((d for d in self.domains if cpu_node in d.get('numa_nodes', [])), None)
    level = self.communication.get('replica_domain', 'die')
    current = owner
    while current is not None and current.get('kind') != level:
      current = parents.get(current.get('parent'))
    replica = current['id'] if current else f'numa:{cpu_node}'
    target = -1
    owners = {f'numa:{cpu_node}'}
    if owner is not None:
      owners.add(owner['id'])
    candidates = [p for p in self.pools if p['owner'] in owners and p['kind'] == 'hbm']
    if candidates:
      nodes = candidates[0]['os_nodes']
      if len(nodes) != 1:
        raise ValueError('HBM endpoint requires one concrete OS target')
      target = nodes[0]
    return replica, target

  def domain_for(self, cpu_node, kind):
    if kind == 'numa':
      return f'numa:{cpu_node}'
    if kind == 'host':
      return self.root
    parents = {d['id']: d for d in self.domains}
    current = next((d for d in self.domains if cpu_node in d.get('numa_nodes', [])), None)
    while current is not None:
      if current.get('kind') == kind:
        return current['id']
      current = parents.get(current.get('parent'))
    raise ValueError(f'no {kind} domain configured for CPU NUMA {cpu_node}')

  def partitions(self, cpu_nodes):
    levels, modes = [], []
    for level in self.communication.get('levels', []):
      names = [self.domain_for(node, level['domain']) for node in cpu_nodes]
      ids = {name:i for i,name in enumerate(sorted(set(names)))}
      levels.append([ids[name] for name in names])
      modes.append(int(level['mode']=='copy'))
    for outer, inner in zip(levels, levels[1:]):
      parent = {}
      for a,b in zip(outer,inner):
        if b in parent and parent[b]!=a:
          raise ValueError('communication levels must be nested coarse-to-fine')
        parent[b]=a
    return levels, modes

  def describe(self):
    return {'path': self.path, 'fingerprint': self.fingerprint, 'root':self.root,
            'domains': list(self.domains), 'memory_pools': list(self.pools),
            'resources': list(self.resources), 'links': list(self.links), 'communication': self.communication,
            'note': 'nominal bandwidth is descriptive; it is not a measured performance proof'}


def current_cpu_node():
  allowed = os.sched_getaffinity(0)
  choices = []
  for path in Path('/sys/devices/system/node').glob('node[0-9]*'):
    nodes = set()
    for part in (path / 'cpulist').read_text().strip().split(','):
      if not part:
        continue
      bounds = part.split('-')
      nodes.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    choices.append((len(allowed & nodes), int(path.name[4:])))
  return max(choices, default=(0, 0), key=lambda x: (x[0], -x[1]))[1]


@dataclass(frozen=True)
class Hints:
  """Performance estimates only; never grant alias or lifetime permissions."""
  reuse: str = 'unknown'
  tag: str = ''
  copy_levels: tuple[int, ...] | None = None

  def __post_init__(self):
    if self.copy_levels is not None and (not self.copy_levels or any(type(v) is not int or v not in (0,1) for v in self.copy_levels)):
      raise ValueError('copy_levels must be a nonempty tuple of share=0/copy=1 modes')


@dataclass
class Options:
  config: str | None = None
  threads: int = 0
  timeout_ms: int | None = None

  @classmethod
  def from_file(cls, path, **kwargs):
    Platform.load(path)  # fail before communicator creation
    return cls(str(Path(path).expanduser().resolve()), **kwargs)
