# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cold communication planning over opaque resource facts.

The physical graph and rank grants come from MemoryRuntime.  Resource incidence,
not hardware names, supplies possible nested rank partitions.  Cost schedules
use the public memory model; they are offline projections of the participating
ranks, never allocation or mapping permissions.  An incomplete, candidate-specific
cost does not authorize replication.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math


def _enum(item):
  return item.name if hasattr(item, 'name') else item[-1] if isinstance(item, list) else item


def _value(module, name, **fields):
  item = getattr(module, name)()
  for key, value in fields.items():
    setattr(item, key, value)
  return item


def rank_facts(runtime, pool, threads):
  """Address-free public graph facts and the actual launch's worker count.

  A main thread's affinity does not describe an already-created executor team.
  Consequently ``execution_slots`` is not used as a worker-count measurement.
  Multi-agent teams require a registered worker-to-agent assignment before their
  placement costs can be treated as complete.
  """
  from vecops.memory._resource import _contract_value
  view = runtime.view()
  agents = [a.id for a in view.graph.agents
            if a.id in view.executable_agents and 'host_load_store' in a.capabilities]
  if pool not in view.allocatable_pools or not agents:
    raise ValueError('communication rank lacks an executable agent or allocatable payload pool')
  if type(threads) is not int or threads < 1:
    raise ValueError('communication worker count must be a positive integer')
  return {'agents': agents, 'workers': threads, 'payload_pool': pool,
          'accessible_pools': list(view.accessible_pools),
          'graph': {'pools': _contract_value(view.graph.pools),
                    'resources': _contract_value(view.graph.resources),
                    'access': _contract_value([edge for edge in view.graph.access if edge.agent in agents])}}


def balanced_targets(partition, rank, sizes):
  """Exactly the native executor's cold, byte-balanced ingress assignment."""
  members = [r for r, group in enumerate(partition) if group == partition[rank]]
  loads = dict.fromkeys(members, 0)
  result = [-1] * len(partition)
  for source, size in enumerate(sizes):
    if partition[source] != partition[rank]:
      selected = min(members, key=lambda member: (loads[member], member))
      loads[selected] += size
      result[source] = selected
  return result


def _canonical_partition(groups):
  labels = {}
  return tuple(labels.setdefault(group, len(labels)) for group in groups)


def _classes(relation):
  """Return equivalence classes, rejecting a merely connected projection."""
  size = len(relation)
  labels = [-1] * size
  for row in range(size):
    if not relation[row][row]:
      return None
    for col in range(size):
      if relation[row][col] != relation[col][row]:
        return None
    if labels[row] < 0:
      label = max(labels, default=-1) + 1
      for col in range(size):
        if relation[row][col]:
          if labels[col] >= 0 and labels[col] != label:
            return None
          labels[col] = label
  if any(relation[a][b] != (labels[a] == labels[b]) for a in range(size) for b in range(size)):
    return None
  return tuple(labels)


@dataclass(frozen=True)
class RoutingTopology:
  partitions: tuple[tuple[int, ...], ...]
  quality: str
  missing_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteCandidate:
  modes: tuple[int, ...]
  service_us: float
  stage_count: int
  quality: str
  comparison_complete: bool
  missing_facts: tuple[str, ...] = ()
  cancelled_common_terms: tuple[str, ...] = ()


class CommunicationCompiler:
  """Compile legal nested candidates and resource-contention cost schedules.

  Rank order is semantic; resource, rule, agent and pool identifiers are opaque.
  Distinct directional fabric resources and unequal capacities are preserved.
  Tensor output pools must be certified facts, not a budget's intended pool.
  """

  def __init__(self, members, *, stage_us=None, native_module=None, max_levels=16):
    self.members = tuple(members)
    if not self.members:
      raise ValueError('communication requires participating ranks')
    if stage_us is not None and (isinstance(stage_us, bool) or not math.isfinite(stage_us) or stage_us < 0):
      raise ValueError('stage_us must be a finite nonnegative measured or configured cost')
    self.stage_us = stage_us
    self.native = native_module
    self.max_levels = max_levels
    self.pools = {}
    self.resources = {}
    self.edges = {}
    for member in self.members:
      if type(member.get('workers')) is not int or member['workers'] < 1:
        raise ValueError('rank worker count must come from actual launch facts')
      graph = member['graph']
      for section, target in [('pools', self.pools), ('resources', self.resources)]:
        for item in graph.get(section, []):
          old = target.setdefault(item['id'], item)
          if old != item:
            raise ValueError(f'inconsistent physical {section} fact: {item["id"]}')
      for edge in graph.get('access', []):
        key = (edge['agent'], edge['pool'])
        for mode in edge['access']:
          mode = _enum(mode)
          old = self.edges.setdefault((*key, mode), edge)
          if old != edge:
            raise ValueError('ambiguous access fact; resolve access rules before communication planning')
    for member in self.members:
      if member['payload_pool'] not in self.pools:
        raise ValueError('rank payload pool is absent from public graph')
    self.topology = self._topology()

  def _edge(self, rank, pool, mode):
    member = self.members[rank]
    if pool not in member['accessible_pools']:
      return None
    agents = member['agents']
    # Do not invent the distribution of native workers among CPU domains.
    return self.edges.get((agents[0], pool, mode)) if len(agents) == 1 else None

  def _topology(self):
    size = len(self.members)
    paths = [[None] * size for _ in range(size)]
    missing = []
    for reader in range(size):
      for owner, member in enumerate(self.members):
        edge = self._edge(reader, member['payload_pool'], 'read')
        if edge is None:
          missing.append(f'unknown rank {reader} read path to payload of rank {owner}')
        else:
          paths[reader][owner] = frozenset(edge['read_fabric'])
    if missing:
      return RoutingTopology((), 'incomplete', tuple(missing))
    # One cut may only split one current branch.  Cuts are combined when they
    # refine separate branches, keeping unequal and noncontiguous groups valid.
    partition = tuple(0 for _ in range(size))
    levels = []
    while True:
      refinements = []
      deferred = False
      for resource in self.resources:
        # A resource consumed even by a rank's own-pool load is a local/shared
        # service ceiling, not a separator of rank groups. Keep it in costs.
        if any(resource in paths[rank][rank] for rank in range(size)):
          continue
        groups = []
        for parent in dict.fromkeys(partition):
          ranks = [rank for rank in range(size) if partition[rank] == parent]
          relation = [[resource not in paths[a][b] and resource not in paths[b][a]
                       for b in ranks] for a in ranks]
          classes = _classes(relation)
          if classes is None:
            # A local cut may become provable after a different resource
            # separates its enclosing branch. Recheck it on the next pass.
            deferred = True
            groups = None
            break
          groups.append((ranks, classes))
        if groups is None:
          continue
        labels = list(partition)
        for ranks, classes in groups:
          for rank, label in zip(ranks, classes):
            labels[rank] = (partition[rank], label)
        candidate = _canonical_partition(labels)
        if candidate != _canonical_partition(partition):
          refinements.append(candidate)
      if not refinements:
        if deferred:
          return RoutingTopology((), 'incomplete', ('resource incidence is not a provable nested cut',))
        break
      # Crossing cuts do not define a tree.  Taking their intersections would
      # silently turn a general fabric into an invented hierarchy.
      for a in refinements:
        sets_a = [{r for r in range(size) if a[r] == group} for group in set(a)]
        for b in refinements:
          sets_b = [{r for r in range(size) if b[r] == group} for group in set(b)]
          if any(x & y and not (x <= y or y <= x) for x in sets_a for y in sets_b):
            return RoutingTopology((), 'incomplete', ('non-laminar resource cuts',))
      def refines(fine, coarse):
        return all(fine[a] != fine[b] or coarse[a] == coarse[b]
                   for a in range(size) for b in range(size))
      # Preserve intermediate cuts: intersecting every compatible refinement
      # at once would collapse a multi-level fabric straight to its finest cut.
      # Independent branch refinements can be combined at the same depth.
      coarsest = [candidate for candidate in refinements
                  if not any(candidate != other and refines(candidate,other)
                             for other in refinements)]
      partition = _canonical_partition([tuple(candidate[rank] for candidate in coarsest)
                                        for rank in range(size)])
      if len(set(partition)) == size:
        break  # singleton relays add no useful locality before final readers
      levels.append(partition)
      if len(levels) >= self.max_levels:
        return RoutingTopology((), 'incomplete', ('resource cuts exceed supported execution levels',))
    return RoutingTopology(tuple(levels), 'reference')

  def candidates(self):
    levels = len(self.topology.partitions)
    if not levels:
      return ((),)
    # A bounded strategy search; all-share remains the correctness baseline.
    if levels > 8:
      return (tuple(0 for _ in range(levels)), *(tuple(int(i == k) for i in range(levels))
                                                for k in range(levels)))
    return tuple(product((0, 1), repeat=levels))

  def _stage(self, copies):
    """Project known copies onto the public scheduler, including path caps."""
    if self.native is None:
      from vecops.memory import native
    else:
      native = self.native
    resources = {}
    missing = []
    for key, item in self.resources.items():
      if _enum(item['kind']) != 'bandwidth':
        continue
      resources[key] = _value(native, 'Resource', id=key, rule=item['rule'],
        bandwidth=item.get('bandwidth'), read=item.get('read'), write=item.get('write'),
        sharing=getattr(native.SharingMode, _enum(item['sharing'])), kind=native.ResourceKind.bandwidth)
    activities = []
    for index, (executor, source, target, size) in enumerate(copies):
      if not size:
        continue
      demands = []
      latency = 0.0
      for endpoint, mode in ((source, 'read'), (target, 'write')):
        if endpoint is None:
          missing.append(f'unobserved {mode} pool for rank {executor}')
          continue
        weights = endpoint if isinstance(endpoint, dict) else {endpoint: 1}
        if not weights or any(type(n) is not int or n <= 0 for n in weights.values()):
          raise ValueError('observed endpoint requires positive integer per-pool byte weights')
        for pool, weight in weights.items():
          fraction = weight / sum(weights.values())
          edge = self._edge(executor, pool, mode)
          if edge is None:
            missing.append(f'unknown rank {executor} {mode} path to {pool}')
            continue
          latency = max(latency, (edge.get('latency_ns') or 0) / 1000)
          service = edge.get(mode + '_service') or self.pools[pool].get(mode + '_service')
          if service:
            demands.append(_value(native, 'ResourceDemand', resource=service,
              service=getattr(native.DemandKind, mode), amount=fraction, per_invocation_bytes=0))
          for resource in edge[mode + '_fabric']:
            demands.append(_value(native, 'ResourceDemand', resource=resource,
              service=native.DemandKind.traffic, amount=fraction, per_invocation_bytes=0))
          cap = (edge.get('bandwidth') or {}).get(mode)
          if cap is not None:
            # This derived ceiling belongs only to the offline cost projection.
            key = 'communication/path-cap/' + json.dumps([edge['id'], mode], separators=(',', ':'))
            if key in self.resources:
              raise ValueError('physical resource collides with derived query cap')
            resources[key] = _value(native, 'Resource', id=key, rule='', bandwidth=cap,
              sharing=native.SharingMode.shared, kind=native.ResourceKind.bandwidth)
            demands.append(_value(native, 'ResourceDemand', resource=key,
              service=native.DemandKind.traffic, amount=fraction, per_invocation_bytes=0))
          if not service and not edge[mode + '_fabric'] and cap is None:
            missing.append(f'unmodeled rank {executor} {mode} service to {pool}')
      activities.append(_value(native, 'Activity', id=f'copy/{index}', method='', payload_bytes=size,
        demands=demands, dependencies=[], fixed_us=latency))
    graph = _value(native, 'ResourceGraph', resources=list(resources.values()))
    model = native.ResourceModel(graph)
    # This is explicitly a union cost projection, not a process-authorized
    # Runtime view.  There are no agents, mappings or allocation grants here.
    view = _value(native, 'ResourceView', graph=graph, epoch=0)
    cost = model.estimate_schedule(view, _value(native, 'CostPlan', activities=activities),
      _value(native, 'CostContext', view_epoch=view.epoch, active=[], execution_slots={}, include_preparation=False))
    missing.extend(cost.missing_facts)
    return cost.total_us, tuple(dict.fromkeys(missing))

  def estimate(self, sizes, modes, *, input_pools=None, output_pools=None):
    size = len(self.members)
    if len(sizes) != size or any(type(n) is not int or n < 0 or n >= 1 << 64 for n in sizes):
      raise ValueError('one exact uint64 payload size per rank is required')
    if tuple(modes) not in self.candidates():
      raise ValueError('route does not belong to this resource topology')
    if input_pools is None:
      input_pools = [None] * size
    if output_pools is None:
      output_pools = [None] * size
    if len(input_pools) != size or len(output_pools) != size:
      raise ValueError('one observed input/output pool per rank is required')
    pools = [member['payload_pool'] for member in self.members]
    total, missing = self._stage([(rank, input_pools[rank], pools[rank], sizes[rank]) for rank in range(size)])
    # Publisher inputs are a truly common, separate stage.  Missing source facts
    # can be cancelled for candidate comparison; destination writes remain costed.
    cancelled = tuple(f'publisher input read for rank {r}' for r in range(size) if input_pools[r] is None)
    relevant = [item for item in missing if not item.startswith('unobserved read pool for rank ')]
    origins = [list(range(size)) for _ in range(size)]
    stages = 0
    for partition, enabled in zip(self.topology.partitions, modes):
      if not enabled:
        continue
      copies = []
      for destination in range(size):
        targets = balanced_targets(partition, destination, sizes)
        for source in range(size):
          if targets[source] >= 0:
            origin, target = origins[destination][source], targets[source]
            if destination == target and origin != target:
              copies.append((destination, pools[origin], pools[target], sizes[source]))
            origins[destination][source] = target
      service, absent = self._stage(copies)
      total += service
      relevant.extend(absent)
      stages += 1
    def output(rank, source):
      endpoint = output_pools[rank]
      if isinstance(endpoint, (list, tuple)):
        if len(endpoint) != size:
          raise ValueError('output placement list needs one endpoint per source rank')
        return endpoint[source]
      return endpoint
    service, absent = self._stage([(rank, pools[origins[rank][source]], output(rank, source), sizes[source])
                                  for rank in range(size) for source in range(size)])
    total += service
    relevant.extend(absent)
    if stages and self.stage_us is None:
      relevant.append('unknown additional-stage synchronization cost')
    elif stages:
      total += stages * self.stage_us
    relevant.extend(self.topology.missing_facts)
    return RouteCandidate(tuple(modes), total, stages, 'incomplete' if relevant or cancelled else 'reference',
                          not relevant,
                          tuple(dict.fromkeys(relevant)), cancelled)

  def select(self, sizes, *, input_pools=None, output_pools=None, minimum_gain=.05):
    if not 0 <= minimum_gain < 1:
      raise ValueError('minimum_gain must be in [0,1)')
    candidates = tuple(self.estimate(sizes, modes, input_pools=input_pools, output_pools=output_pools)
                       for modes in self.candidates())
    baseline = candidates[0]
    eligible = [candidate for candidate in candidates if candidate.comparison_complete]
    best = min(eligible, key=lambda item: item.service_us, default=baseline)
    if not baseline.comparison_complete or best.service_us >= baseline.service_us * (1-minimum_gain):
      best = baseline
    return best, candidates
