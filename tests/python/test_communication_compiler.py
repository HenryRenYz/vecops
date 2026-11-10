# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Resource-derived communication candidates, without hardware vocabulary."""
from copy import deepcopy

import pytest

from vecops.distributed.compiler import CommunicationCompiler, balanced_targets


def facts():
  resources = [{'id': f'service-{i}', 'rule': 'opaque', 'kind': 'bandwidth',
                'sharing': 'read_write', 'read': 100., 'write': 100., 'bandwidth': None}
               for i in range(4)]
  resources += [{'id': 'bridge-forward', 'rule': 'arbitrary', 'kind': 'bandwidth',
                 'sharing': 'shared', 'bandwidth': 1., 'read': None, 'write': None},
                {'id': 'bridge-reverse', 'rule': 'arbitrary', 'kind': 'bandwidth',
                 'sharing': 'shared', 'bandwidth': 2., 'read': None, 'write': None}]
  pools = [{'id': f'pool-{i}', 'read_service': f'service-{i}', 'write_service': f'service-{i}'}
           for i in range(4)]
  access = []
  for rank in range(4):
    for owner in range(4):
      fabric = [] if rank // 2 == owner // 2 else ['bridge-forward' if rank < 2 else 'bridge-reverse']
      access.append({'id': f'path-{rank}-{owner}', 'agent': f'agent-{rank}', 'pool': f'pool-{owner}',
        'access': ['read', 'write'], 'read_fabric': fabric, 'write_fabric': fabric,
        'read_service': None, 'write_service': None, 'latency_ns': 0.,
        'bandwidth': {'read': 100., 'write': 100.}})
  graph = {'pools': pools, 'resources': resources, 'access': access}
  return [{'agents': [f'agent-{rank}'], 'workers': 3, 'payload_pool': f'pool-{rank}',
           'accessible_pools': [pool['id'] for pool in pools], 'graph': deepcopy(graph)}
          for rank in range(4)]


def test_directed_resources_form_opaque_nested_partition_and_cost_real_writes():
  compiler = CommunicationCompiler(facts(), stage_us=2)
  assert compiler.topology.partitions == ((0, 0, 1, 1),)
  best, candidates = compiler.select([1 << 20] * 4,
    input_pools=[f'pool-{i}' for i in range(4)], output_pools=[f'pool-{i}' for i in range(4)])
  assert best.modes == (1,)
  assert best.service_us < candidates[0].service_us
  assert best.stage_count == 1 and best.quality == 'reference'
  # Replication reads each external source once per destination group, rather
  # than once per final reader; it also writes the actual replica pool.
  assert best.service_us > 2 * (1 << 20) / (1 << 30) * 1e6


def test_renaming_and_vector_order_do_not_change_compiled_topology_or_cost():
  original = facts()
  renamed = deepcopy(original)
  def rename(value):
    if isinstance(value, str) and value.startswith(('path-', 'pool-', 'agent-', 'service-', 'bridge-')):
      return 'cowos_superdie/infinity_fabric/' + value[::-1]
    if isinstance(value, list):
      return [rename(item) for item in value]
    if isinstance(value, dict):
      return {key: rename(item) for key, item in value.items()}
    return value
  renamed = rename(renamed)
  for member in renamed:
    for vector in member['graph'].values():
      vector.reverse()
  left = CommunicationCompiler(original, stage_us=2)
  right = CommunicationCompiler(renamed, stage_us=2)
  assert left.topology == right.topology
  a = left.estimate([1000] * 4, (1,), output_pools=[f'pool-{i}' for i in range(4)])
  b = right.estimate([1000] * 4, (1,), output_pools=[rename(f'pool-{i}') for i in range(4)])
  assert a.service_us == pytest.approx(b.service_us)


def test_unknown_consumer_writes_or_added_stage_cost_keeps_flat():
  compiler = CommunicationCompiler(facts(), stage_us=2)
  best, candidates = compiler.select([1 << 20] * 4)
  assert best.modes == (0,)
  assert all(case.quality == 'incomplete' for case in candidates)
  assert any('unobserved write' in missing for missing in best.missing_facts)
  assert best.cancelled_common_terms == tuple(f'publisher input read for rank {i}' for i in range(4))
  compiler = CommunicationCompiler(facts())
  pools = [f'pool-{i}' for i in range(4)]
  best, candidates = compiler.select([1 << 20] * 4, output_pools=pools)
  assert best.modes == (0,)
  assert 'unknown additional-stage synchronization cost' in candidates[1].missing_facts


def test_crossing_cuts_and_missing_rank_grants_do_not_invent_a_tree():
  members = facts()
  graph = members[0]['graph']
  graph['resources'].extend([{'id': 'other-cut', 'rule': '', 'kind': 'bandwidth',
    'sharing': 'shared', 'bandwidth': 1., 'read': None, 'write': None}])
  for edge in graph['access']:
    a, b = int(edge['agent'].split('-')[-1]), int(edge['pool'].split('-')[-1])
    if a % 2 != b % 2:
      edge['read_fabric'].append('other-cut')
  for member in members:
    member['graph'] = deepcopy(graph)
  compiler = CommunicationCompiler(members)
  assert not compiler.topology.partitions
  assert compiler.topology.missing_facts == ('non-laminar resource cuts',)
  members = facts()
  members[0]['accessible_pools'].remove('pool-3')
  compiler = CommunicationCompiler(members)
  assert compiler.topology.quality == 'incomplete' and compiler.candidates() == ((),)


def test_partial_executor_assignment_and_unobserved_bandwidth_are_unknown():
  members = facts()
  members[0]['agents'].append('another-agent')
  assert CommunicationCompiler(members).topology.quality == 'incomplete'
  members = facts()
  for member in members:
    member['graph']['resources'][-1]['bandwidth'] = None
  compiler = CommunicationCompiler(members, stage_us=2)
  pools = [f'pool-{i}' for i in range(4)]
  best, _ = compiler.select([10000] * 4, output_pools=pools)
  assert best.modes == (0,) and best.quality == 'incomplete'


def test_balanced_assignment_uses_real_bytes_and_noncontiguous_unequal_members():
  assert balanced_targets([3, 8, 3, 3, 8], 0, [1, 20, 1, 1, 10]) == [-1, 0, -1, -1, 2]
  assert balanced_targets([3, 8, 3, 3, 8], 1, [20, 1, 10, 5, 1]) == [1, -1, 4, 4, -1]


def test_consumer_profile_exact_contract_does_not_accept_materialization_evidence():
  from vecops.distributed._routing import consumer_profile_key
  from vecops.distributed.tuning import _profile_routes
  rows=[dict(kind='borrowed-consumer',bytes=4096,inputs=[None],outputs=[{'opaque-pool':8192}],
    consumer={'operator':'example::consumer','specs':[3],'binary_sha256':'a'*64,
              'source':{'shape':[1024],'strides':[1],'dtype':'torch.float32','device':'cpu'},
              'output':{'shape':[2,1024],'strides':[1024,1],'dtype':'torch.float32','device':'cpu'}})]
  key=consumer_profile_key(rows)
  assert key.startswith('consumer/')
  common=deepcopy(rows);common[0]['inputs']=[{'other-source-pool':4096}]
  assert consumer_profile_key(common)==key  # publication is an explicitly common stage
  for mutate in ('binary','specs','output','placement'):
    changed=deepcopy(rows)
    if mutate=='binary':changed[0]['consumer']['binary_sha256']='b'*64
    elif mutate=='specs':changed[0]['consumer']['specs']=[4]
    elif mutate=='output':changed[0]['consumer']['output']['strides']=[1025,1]
    else:changed[0]['outputs']=[{'other-pool':8192}]
    assert consumer_profile_key(changed)!=key
  unknown=deepcopy(rows);unknown[0]['outputs']=[None]
  assert consumer_profile_key(unknown) is None
  assert consumer_profile_key([]) is None
  case=dict(key=key,selected=[1],consumer_contract=rows)
  with pytest.raises(ValueError,match='consumer calibration contract'):_profile_routes({'cases':[case]})
  case['measurement']='borrowed-consumer'
  assert _profile_routes({'cases':[case]})=={key:[1]}
  case['consumer_contract'][0]['consumer']['specs']=[99]
  with pytest.raises(ValueError,match='consumer calibration contract'):_profile_routes({'cases':[case]})


def test_consumer_and_materialization_profiles_are_installed_separately(monkeypatch):
  from types import SimpleNamespace
  from vecops.distributed import _routing
  class Backend:
    def configure_routes(self,routes):self.routes=routes
  backend=Backend();planner=SimpleNamespace(consumer_routes={},_cache={'old':1})
  monkeypatch.setattr(_routing,'for_backend',lambda _:planner)
  _routing.install_profiles(backend,{'all_gather:layout':[1],'consumer/exact':[0]})
  assert backend.routes=={'all_gather:layout':[1]}
  assert planner.consumer_routes=={'consumer/exact':[0]}
  assert planner._cache=={}



def test_laminar_fabric_preserves_intermediate_levels_and_unequal_branches():
  members=facts()
  # Six participants, coarse {0,1,2,3}/{4,5}, then finer {0,1}/{2,3}/{4,5}.
  partitions=((0,0,0,0,1,1),(0,0,1,1,2,2))
  graph=deepcopy(members[0]['graph'])
  graph['pools']=[dict(id=f'p{i}',read_service=None,write_service=None) for i in range(6)]
  graph['resources']=[dict(id=f'fabric-{level}',rule='custom',kind='bandwidth',sharing='shared',
                           bandwidth=1.+level,read=None,write=None) for level in range(2)]
  graph['access']=[]
  for reader in range(6):
    for owner in range(6):
      fabric=[f'fabric-{level}' for level,part in enumerate(partitions) if part[reader]!=part[owner]]
      graph['access'].append(dict(id=f'e{reader}-{owner}',agent=f'a{reader}',pool=f'p{owner}',access=['read','write'],
        read_fabric=fabric,write_fabric=fabric,read_service=None,write_service=None,latency_ns=0.,
        bandwidth={'read':100.,'write':100.}))
  members=[dict(agents=[f'a{i}'],workers=1,payload_pool=f'p{i}',accessible_pools=[f'p{j}' for j in range(6)],graph=graph) for i in range(6)]
  compiler=CommunicationCompiler(members,stage_us=1)
  assert compiler.topology.partitions==partitions
  assert len(compiler.candidates())==4
  for modes in compiler.candidates():
    estimate=compiler.estimate([100,200,300,400,500,600],modes,output_pools=[f'p{i}' for i in range(6)])
    assert estimate.comparison_complete



def test_one_way_nontransitive_fabric_is_not_projected_into_a_hierarchy():
  members=facts()
  for member in members:
    for edge in member['graph']['access']:
      edge['read_fabric']=['bridge-forward'] if edge['agent']=='agent-0' and edge['pool']=='pool-1' else []
  compiler=CommunicationCompiler(members,stage_us=1)
  assert compiler.topology.partitions==()
  assert compiler.topology.quality=='incomplete'
  assert compiler.topology.missing_facts==('resource incidence is not a provable nested cut',)



def test_local_fabric_service_is_costed_without_becoming_a_rank_cut():
  members=facts()
  for member in members:
    graph=member['graph']
    graph['resources'].append(dict(id='front-service',rule='opaque',kind='bandwidth',sharing='shared',
                                  bandwidth=80.,read=None,write=None))
    for edge in graph['access']:edge['read_fabric'].append('front-service')
  compiler=CommunicationCompiler(members,stage_us=1)
  assert compiler.topology.partitions==((0,0,1,1),)
  best,_=compiler.select([1<<20]*4,output_pools=[f'pool-{i}' for i in range(4)])
  assert best.comparison_complete


def _fabric_members(paths):
  """Describe directed, complete read paths; labels carry no hierarchy meaning."""
  count=len(paths)
  resources=list(dict.fromkeys(resource for row in paths for path in row for resource in path))
  graph=dict(pools=[dict(id=f'p{i}',read_service=None,write_service=None)for i in range(count)],
      resources=[dict(id=name,rule='opaque',kind='bandwidth',sharing='shared',
          bandwidth=1.,read=None,write=None)for name in resources],access=[])
  for reader,row in enumerate(paths):
    assert len(row)==count
    for owner,path in enumerate(row):
      graph['access'].append(dict(id=f'e{reader}-{owner}',agent=f'a{reader}',pool=f'p{owner}',
          access=['read','write'],read_fabric=list(path),write_fabric=list(path),
          read_service=None,write_service=None,latency_ns=0.,bandwidth={'read':100.,'write':100.}))
  return [dict(agents=[f'a{i}'],workers=1,payload_pool=f'p{i}',
      accessible_pools=[f'p{j}'for j in range(count)],graph=deepcopy(graph))for i in range(count)]


def _branch_local_fabric(coarse,fine):
  # Cross-branch paths consume only the enclosing directed resource. A local
  # resource is absent, rather than implicitly extended, outside its branch.
  return _fabric_members([[[]if a==b else
      [f'outer-{coarse[b]}-{coarse[a]}']if coarse[a]!=coarse[b]else
      [f'inner-{coarse[a]}-{fine[b]}-{fine[a]}']if fine[a]!=fine[b]else []
      for b in range(len(coarse))]for a in range(len(coarse))])


def test_three_endpoint_local_cut_waits_for_proven_enclosing_partition():
  # A/B use a directional local link. C is outside that link's domain, so
  # absence alone is nontransitive until the enclosing A/B|C cut is applied.
  members=_fabric_members([[[],['d10'],['s10']],
                           [['d01'],[],['s10']],
                           [['s01'],['s01'],[]]])
  for member in members:
    member['graph']['resources'].sort(key=lambda r:('s'in r['id'],r['id']))
  compiler=CommunicationCompiler(members,stage_us=1)
  assert compiler.topology.partitions==((0,0,1),)
  assert compiler.topology.quality=='reference'and not compiler.topology.missing_facts
  assert compiler.candidates()==((0,),(1,))


@pytest.mark.parametrize('coarse,fine,expected',[
    ((0,1,0,1,0,1,1),(0,2,1,3,0,2,2),
      ((0,1,0,1,0,1,1),(0,1,2,3,0,1,1))),
    ((0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1),
     (0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3),
      ((0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1),
       (0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3))),
])
def test_independent_branch_local_cuts_preserve_nested_order_and_membership(coarse,fine,expected):
  members=_branch_local_fabric(coarse,fine)
  for reverse in (False,True):
    if reverse:
      for member in members:member['graph']['resources'].reverse()
    compiler=CommunicationCompiler(members,stage_us=1)
    assert compiler.topology.partitions==expected
    assert compiler.topology.quality=='reference'and len(compiler.candidates())==4
    assert compiler.estimate([100+i for i in range(len(members))],(1,1),
        output_pools=[f'p{i}'for i in range(len(members))]).comparison_complete


def test_single_enclosing_domain_keeps_existing_eight_rank_partition():
  compiler=CommunicationCompiler(_branch_local_fabric((0,)*8,(0,1,0,1,0,1,0,1)),stage_us=1)
  assert compiler.topology.partitions==((0,1,0,1,0,1,0,1),)
  assert compiler.candidates()==((0,),(1,))


def test_unprovable_local_resource_still_discards_already_proven_coarse_levels():
  members=_branch_local_fabric((0,0,0,1,1),(0,0,0,1,1))
  for member in members:
    member['graph']['resources'].append(dict(id='unproved',rule='opaque',kind='bandwidth',
        sharing='shared',bandwidth=1.,read=None,write=None))
    for edge in member['graph']['access']:
      if edge['agent']=='a0'and edge['pool']=='p1':edge['read_fabric'].append('unproved')
  compiler=CommunicationCompiler(members)
  assert compiler.topology.partitions==()and compiler.candidates()==((),)
  assert compiler.topology.missing_facts==('resource incidence is not a provable nested cut',)


def test_deferred_local_cuts_do_not_bypass_supported_level_limit():
  members=_branch_local_fabric((0,0,0,0,1,1),(0,0,1,1,2,2))
  compiler=CommunicationCompiler(members,max_levels=1)
  assert compiler.topology.partitions==()
  assert compiler.topology.missing_facts==('resource cuts exceed supported execution levels',)
