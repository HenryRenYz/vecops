# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Portable performance evidence never replaces live communicator authority."""
import copy
import json
from types import SimpleNamespace

import pytest

from vecops.distributed import tuning


@pytest.fixture
def evidence():
  members=[]
  facts=[]
  for rank in range(2):
    members.append({'host':'boot-A','capacity_authority':'host:boot-A:dev:inode-A',
      'threads':rank+1,'binary':'transport-dso','platform':'policy',
      'payload_pool':f'pool-{rank}','payload_page':4096,
      'resources':{'agents':[f'agent-{rank}'],'workers':rank+1,
                   'graph':{'resources':[{'id':'fabric','bandwidth':123}] }},
      'runtime':{'boot':'boot-A','frequency_policy':{'policy0':{'scaling_max_freq':'2000'}},
                 'user_tag':'clock-tag'},'calibration':False})
    facts.append({'hardware':{'architecture':'test-architecture','kernel':'test-kernel',
      'cpus':[{'id':0,'model':'test-cpu','microcode':'revision-A'}],
      'firmware':{'bios_version':'revision-A'}},'cpu_grant':[rank*2,rank*2+1],
      'worker_binding':{'provider':'native','torch_resident_team':'0'}})
  live=tuning.context(SimpleNamespace(fingerprint='policy'),members,1,'transport-dso')
  return live,facts


def _profile(path, live, facts):
  value={'schema_version':2,'context':tuning._performance_context(live,facts),
         'runtime_context':live,'cases':[{'key':'materialization','selected':[1]}]}
  path.write_text(json.dumps(value))
  return value


def test_new_job_and_boot_reuse_preserves_full_live_context(tmp_path,evidence):
  live,facts=evidence
  before=copy.deepcopy(live)
  path=tmp_path/'profile.json'
  _profile(path,live,facts)
  next_job=copy.deepcopy(live)
  for member in next_job['members']:
    member['host']='boot-B'
    member['capacity_authority']='other-host:boot-B:dev:inode-B'
  for runtime in next_job['runtime']:runtime['boot']='boot-B'
  assert next_job!=live
  routes,_=tuning.load_profile(path,tuning._performance_context(next_job,facts))
  assert routes=={'materialization':[1]}
  assert live==before  # projection must not rewrite membership/template authority
  projected=tuning._performance_context(live,facts)
  projected['members'][0]['resources']['workers']=999
  assert live==before


@pytest.mark.parametrize('change',[
  'cpu','microcode','firmware','architecture','kernel','grant','rank-order','worker-binding',
  'workers','resource','frequency','tag','binary','member-binary','policy','payload-page'])
def test_changed_performance_evidence_is_stale(tmp_path,evidence,change):
  live,facts=evidence
  path=tmp_path/'profile.json';_profile(path,live,facts)
  if change in ('cpu','microcode'):
    facts[0]['hardware']['cpus'][0]['model' if change=='cpu' else 'microcode']='different'
  elif change=='firmware':facts[0]['hardware']['firmware']['bios_version']='different'
  elif change in ('architecture','kernel'):facts[0]['hardware'][change]='different'
  elif change=='grant':facts[0]['cpu_grant']=[4,5]
  elif change=='rank-order':facts.reverse()
  elif change=='worker-binding':facts[0]['worker_binding']['provider']='torch'
  elif change=='workers':live['threads'][0]=2
  elif change=='resource':live['members'][0]['resources']['graph']['resources'][0]['bandwidth']=456
  elif change=='frequency':live['runtime'][0]['frequency_policy']['policy0']['scaling_max_freq']='3000'
  elif change=='tag':live['runtime'][1]['user_tag']='changed-clock'
  elif change=='binary':live['binary']='changed-dso'
  elif change=='member-binary':live['members'][0]['binary']='changed-dso'
  elif change=='policy':live['policy']='changed-policy'
  elif change=='payload-page':live['members'][0]['payload_page']=2<<20
  with pytest.warns(RuntimeWarning,match='stale'):
    routes,_=tuning.load_profile(path,tuning._performance_context(live,facts))
  assert not routes


@pytest.mark.parametrize('missing',['cpu','grant'])
def test_missing_stable_evidence_cannot_accept_even_equal_profile(tmp_path,evidence,missing):
  live,facts=evidence
  if missing=='cpu':facts[0]['hardware']=None
  else:facts[0]['cpu_grant']=[]
  path=tmp_path/'profile.json';value=_profile(path,live,facts)
  with pytest.warns(RuntimeWarning,match='stale'):
    routes,_=tuning.load_profile(path,value['context'])
  assert not routes


def test_legacy_profile_without_hardware_is_rejected(tmp_path,evidence):
  live,_=evidence
  path=tmp_path/'legacy.json'
  path.write_text(json.dumps({'schema_version':1,'context':live,'cases':[]}))
  with pytest.raises(ValueError,match='schema'):tuning.load_profile(path,live)


def test_stable_cpu_identity_ignores_live_clock_but_tracks_firmware(monkeypatch):
  values={'/proc/cpuinfo':'''processor : 0
vendor_id : test-vendor
cpu family : 1
model : 2
stepping : 3
flags : b a
microcode : 4
cpu MHz : 1000
''','/sys/devices/virtual/dmi/id/bios_version':'firmware-A'}
  monkeypatch.setattr(tuning,'_read_identity',lambda path:values.get(path,''))
  old=tuning._hardware_identity()
  values['/proc/cpuinfo']=values['/proc/cpuinfo'].replace('1000','2000').replace('b a','a b')
  assert tuning._hardware_identity()==old
  values['/sys/devices/virtual/dmi/id/bios_version']='firmware-B'
  assert tuning._hardware_identity()!=old
  values['/proc/cpuinfo']='processor : 0\nmodel name : vague description\n'
  assert tuning._hardware_identity() is None


def test_arm_identity_requires_real_implementer_and_revision(monkeypatch):
  text='''processor : 0
CPU implementer : 0xab
CPU architecture : 8
CPU variant : 0x1
CPU part : 0xcde
CPU revision : 2
Features : sve fp
'''
  monkeypatch.setattr(tuning,'_read_identity',lambda path:text if path=='/proc/cpuinfo' else '')
  assert tuning._hardware_identity()['cpus'][0]['CPU revision']=='2'
  text=text.replace('CPU implementer : 0xab\n','')
  assert tuning._hardware_identity() is None


def test_team_grant_precedes_openmp_narrowed_caller(monkeypatch):
  monkeypatch.setattr(tuning,'_hardware_identity',lambda:{'cpus':['cpu']})
  monkeypatch.setattr(tuning.os,'sched_getaffinity',lambda _: {0})
  binding=SimpleNamespace(object='cpu-agent',backend='linux_cpu',labels={'allowed_cpus':'2,3,4,5'})
  runtime=SimpleNamespace(view=lambda:SimpleNamespace(executable_agents=['cpu-agent']),
                          native=SimpleNamespace(describe_bindings=lambda:[binding]))
  monkeypatch.setenv('VECOPS_THREAD_POOL_CPU_LIST','7,4-5')
  assert tuning._local_performance_facts(runtime,3)['cpu_grant']==[7,4,5]
  monkeypatch.delenv('VECOPS_THREAD_POOL_CPU_LIST')
  assert tuning._local_performance_facts(runtime,3)['cpu_grant']==[2,3,4,5]
  binding.labels={}
  assert not tuning._local_performance_facts(runtime,3)['cpu_grant']


def test_authoritative_grant_allows_oversubscribed_team_but_caller_fallback_does_not(monkeypatch):
  monkeypatch.setattr(tuning,'_hardware_identity',lambda:{'cpus':['cpu']})
  monkeypatch.setattr(tuning.os,'sched_getaffinity',lambda _: {0})
  binding=SimpleNamespace(object='cpu-agent',backend='linux_cpu',labels={'allowed_cpus':'2'})
  runtime=SimpleNamespace(view=lambda:SimpleNamespace(executable_agents=['cpu-agent']),
                          native=SimpleNamespace(describe_bindings=lambda:[binding]))
  monkeypatch.setenv('VECOPS_THREAD_POOL_CPU_LIST','7')
  assert tuning._local_performance_facts(runtime,3)['cpu_grant']==[7]
  monkeypatch.delenv('VECOPS_THREAD_POOL_CPU_LIST')
  assert tuning._local_performance_facts(runtime,3)['cpu_grant']==[2]
  binding.labels={}
  assert not tuning._local_performance_facts(runtime,3)['cpu_grant']


def test_lifetime_registry_survives_wrapper_reconstruction_and_write(tmp_path,monkeypatch,evidence):
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops.distributed import _routing
  from vecops.distributed.compiler import RoutingTopology
  live,facts=evidence
  native=SimpleNamespace(_routing_id=123,tuning_context=json.dumps(live))
  other_wrapper=SimpleNamespace(_routing_id=123)
  class Store:
    rows={'vecops/performance-identity/1':json.dumps(facts[1]).encode()}
    def set(self,key,value):self.rows[key]=value.encode()
    def get(self,key):return self.rows[key]
  planner=SimpleNamespace(lifetime=SimpleNamespace(alive=True),rank=0,size=2,store=Store(),
    compiler=SimpleNamespace(topology=RoutingTopology(((0,1),),'reference')))
  monkeypatch.setattr(_routing,'_registry',{123:planner})
  monkeypatch.setattr(tuning,'_local_performance_facts',lambda *args:facts[0])
  tuning.bind_profile_context(native,live,None)
  assert tuning.profile_context(other_wrapper)==tuning.profile_context(native)
  detached=tuning.profile_context(other_wrapper);detached['members'].clear()
  assert tuning.profile_context(native)['members']==tuning._performance_context(live,facts)['members']
  monkeypatch.setattr(vd,'backend',lambda group=None:other_wrapper)
  # A reconstructed real wrapper exposes the same native tuning_context field.
  other_wrapper.tuning_context=native.tuning_context
  monkeypatch.setattr(dist,'get_rank',lambda group=None:0)
  monkeypatch.setattr(dist,'barrier',lambda group=None:None)
  installed=[]
  monkeypatch.setattr(_routing,'install_profiles',lambda backend,routes:installed.append(routes))
  path=tmp_path/'profile.json'
  record=tuning.write_profile(path,[{'key':'case','selected':[1]}])
  assert record['context']==tuning.profile_context(native)
  assert record['runtime_context']==live and json.loads(native.tuning_context)==live
  assert installed==[{'case':[1]}]
  facts[0]['hardware']=None
  planner._performance_context=json.dumps(tuning._performance_context(live,facts))
  with pytest.raises(ValueError,match='stable CPU'):tuning.write_profile(path,[])
  planner.lifetime.alive=False
  with pytest.raises(ValueError,match='unavailable'):tuning.profile_context(other_wrapper)


def test_same_level_count_with_different_cut_membership_is_stale(tmp_path,monkeypatch,evidence):
  from vecops.distributed import _routing
  from vecops.distributed.compiler import RoutingTopology
  live,facts=evidence
  # Four ranks admit two genuinely different one-level partitions.
  live=copy.deepcopy(live)
  live['members']+=copy.deepcopy(live['members'])
  live['runtime']+=copy.deepcopy(live['runtime'])
  live['threads']+=live['threads']
  facts+=copy.deepcopy(facts)
  for rank,fact in enumerate(facts):fact['cpu_grant']=[2*rank,2*rank+1]
  original=copy.deepcopy(live)
  prefix='vecops/performance-identity/'
  rows={prefix+str(rank):json.dumps(fact).encode() for rank,fact in enumerate(facts)}
  store=SimpleNamespace(set=lambda key,value:rows.__setitem__(key,value.encode()),get=rows.__getitem__)
  native=SimpleNamespace(_routing_id=321)
  compiler=SimpleNamespace(topology=RoutingTopology(((0,0,1,1),),'reference',('partial cost',)))
  planner=SimpleNamespace(lifetime=SimpleNamespace(alive=True),rank=0,size=4,store=store,compiler=compiler)
  monkeypatch.setattr(_routing,'_registry',{321:planner})
  monkeypatch.setattr(tuning,'_local_performance_facts',lambda *args:facts[0])
  tuning.bind_profile_context(native,live,None)
  old=tuning.profile_context(native)
  assert old['generated_topology']=={
    'partitions':[[0,0,1,1]],'quality':'reference','missing_facts':['partial cost']}
  path=tmp_path/'profile.json'
  path.write_text(json.dumps({'schema_version':2,'context':old,
                             'cases':[{'key':'case','selected':[1]}]}))
  assert tuning.load_profile(path,old)[0]=={'case':[1]}
  compiler.topology=RoutingTopology(((0,1,0,1),),'reference',('partial cost',))
  tuning.bind_profile_context(native,live,None)
  current=tuning.profile_context(native)
  assert len(current['generated_topology']['partitions'])==len(old['generated_topology']['partitions'])
  with pytest.warns(RuntimeWarning,match='stale'):
    assert tuning.load_profile(path,current)[0]=={}
  assert live==original
