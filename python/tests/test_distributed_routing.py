# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Nested asymmetric routes and cold, evidence-based route selection."""
import os
import pytest


def _routing(rank,size,store):
  os.environ['VECOPS_NUM_THREADS']='2'
  if lists:=os.environ.get('VECOPS_TEST_CPU_LISTS'):
    import json
    os.sched_setaffinity(0,json.loads(lists)[rank])
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops.distributed.config import Platform
  from vecops.distributed.tuning import tune_all_gather,write_profile,load_profile
  # Logical test topology: uneven outer groups 2+1, then three leaf groups.
  # No claim about hardware performance/placement is made by this fixture.
  Platform.load=classmethod(lambda cls,path=None:Platform(communication={'relay_threshold_bytes':0}))
  Platform.partitions=lambda self,nodes:([[0,0,1],[0,1,2]],[1,1])
  torch.set_num_threads(2);vd.register_backend()
  dist.init_process_group('vecops',init_method='file://'+store,rank=rank,world_size=size,
      pg_options=vd.Options(timeout_ms=10000))
  try:
    native=vd.backend()
    source=torch.arange(33,dtype=torch.float32).reshape(3,11)+rank*100
    storage=[torch.full((3,22),-1.) for _ in range(size)]
    outputs=[x[:,::2] for x in storage]
    modes=[[0,0],[1,0],[0,1],[1,1]]
    for route in modes:
      native.set_route(route)
      handle=native.prepare('all_gather',outputs,[source],[],[],0)
      for i in range(7):
        torch.ops.vecops_comm.run.default(handle,outputs,[source+i])
        for peer,out in enumerate(outputs):
          assert torch.equal(out,torch.arange(33).reshape(3,11)+peer*100+i)
          assert (storage[peer][:,1::2]==-1).all()
    native.set_route([])
    case=tune_all_gather(source,outputs,candidates=modes,iterations=3,rounds=3,warmup=1)
    assert case['selected'] in modes
    path=store+'.json'
    profile=write_profile(path,[case])
    routes,_=load_profile(path,profile['context'])
    assert routes[case['key']]==case['selected']
    import warnings
    with warnings.catch_warnings(record=True) as caught:
      stale,_=load_profile(path,{**profile['context'],'threads':999})
      assert not stale and caught
    handle=native.prepare('all_gather',outputs,[source],[],[],0)
    torch.ops.vecops_comm.run.default(handle,outputs,[source])
    dist.barrier();vd.seal()
    with pytest.raises(RuntimeError,match='after seal'):native.set_route([0,0])
    with pytest.raises(RuntimeError,match='after seal'):native.configure_routes({})
    before=vd.statistics()
    for _ in range(10):torch.ops.vecops_comm.run.default(handle,outputs,[source])
    assert vd.statistics()['plans']==before['plans']
    dist.barrier()
  finally:
    dist.destroy_process_group()


def test_nested_asymmetric_routes_and_calibration(tmp_path):
  import torch.multiprocessing as mp
  from vecops.distributed._build import load
  load();mp.spawn(_routing,args=(3,str(tmp_path/'store')),nprocs=3,join=True)


def test_platform_retains_resource_links_and_nested_levels(tmp_path):
  from vecops.distributed import Platform
  path=tmp_path/'p.toml'
  path.write_text('''[topology]
[[topology.domains]]
id="s0"
kind="socket"
parent="host0"
[[topology.domains]]
id="d0"
kind="die"
parent="s0"
numa_nodes=[0,1]
[[topology.domains]]
id="d1"
kind="die"
parent="s0"
numa_nodes=[2]
[[topology.links]]
a="d0"
b="d1"
resource="link"
[[resources]]
id="link"
capacity_GBps=128
[communication]
levels=[{domain="socket",mode="share"},{domain="die",mode="copy"},{domain="numa",mode="share"}]
''')
  platform=Platform.load(path)
  assert platform.links[0]['resource']=='link'
  assert platform.partitions([0,1,2])==([[0,0,0],[0,0,1],[0,1,2]],[0,1,0])


def test_explicit_numa_domain_owns_hbm_pool(tmp_path):
  from vecops.distributed import Platform
  path=tmp_path/'tree.toml'
  path.write_text('''[topology]
root="machine"
[[topology.domains]]
id="d0"
kind="die"
parent="machine"
[[topology.domains]]
id="n0"
kind="numa"
parent="d0"
numa_nodes=[0]
[[memory.pools]]
id="fast0"
owner="n0"
kind="hbm"
os_nodes=[16]
''')
  platform=Platform.load(path)
  assert platform.resolve(0)==('d0',16)
  assert platform.domain_for(0,'host')=='machine'


def test_copy_schedule_is_independent_of_vector_implementation(tmp_path):
  import pytest
  from vecops.distributed.config import Platform
  path=tmp_path/'copy.toml'
  for schedule in ('auto','peer'):
    path.write_text(f'[communication]\ncopy_schedule="{schedule}"\n')
    assert Platform.load(path).communication['copy_schedule']==schedule
  path.write_text('[communication]\ncopy_schedule="unsupported"\n')
  with pytest.raises(ValueError,match='copy_schedule'):Platform.load(path)
  # Instruction selection is internal. Legacy backend switches must not be
  # silently accepted after transport becomes unconditionally vec based.
  for backend in ('auto','libc','vec'):
    path.write_text(f'[communication]\nrow_copy_backend="{backend}"\n')
    with pytest.raises(ValueError,match='unsupported communication policy'):
      Platform.load(path)
