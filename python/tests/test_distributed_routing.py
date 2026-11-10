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
  from vecops.distributed.config import Policy
  from vecops.distributed.tuning import tune_all_gather,write_profile,load_profile,profile_context
  # Logical test topology: uneven outer groups 2+1, then three leaf groups.
  # No claim about hardware performance/placement is made by this fixture.
  Policy.load=classmethod(lambda cls,path=None:Policy(communication={'relay_threshold_bytes':0,'algorithm':'flat'}))
  from vecops.distributed.compiler import CommunicationCompiler,RoutingTopology
  CommunicationCompiler._topology=lambda self:RoutingTopology(((0,0,1),(0,1,2)),'reference')
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
    routes,_=load_profile(path,profile_context(native))
    assert routes[case['key']]==case['selected']
    import warnings
    with warnings.catch_warnings(record=True) as caught:
      stale,_=load_profile(path,{**profile_context(native),'threads':999})
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


def test_policy_rejects_hardware_kinds_and_manual_topology(tmp_path):
  from vecops.distributed import Policy
  path=tmp_path/'policy.toml'
  for section in ('topology','memory','resources'):
    path.write_text(f'[{section}]\nid="cowos_superdie"\n')
    with pytest.raises(ValueError,match='unknown communication policy sections'):Policy.load(path)
  path.write_text('[communication]\nlevels=[{domain="arbitrary",mode="copy"}]\n')
  with pytest.raises(ValueError,match='unsupported communication policy fields'):Policy.load(path)


def test_policy_control_cost_has_explicit_units_and_bounded_sampling(tmp_path):
  from vecops.distributed import Policy
  path=tmp_path/'policy.toml'
  path.write_text('schema="communication-policy/0.1"\n[communication]\nstage_us=1.5\nsync_rounds=7\n')
  assert Policy.load(path).communication['stage_us']==1.5
  for invalid in ('stage_us=-1','stage_us=true','sync_rounds=2','sync_rounds=true','sync_warmup=32'):
    path.write_text('[communication]\n'+invalid+'\n')
    with pytest.raises(ValueError):Policy.load(path)


def test_copy_schedule_is_independent_of_vector_implementation(tmp_path):
  import pytest
  from vecops.distributed.config import Policy
  path=tmp_path/'copy.toml'
  for schedule in ('auto','peer'):
    path.write_text(f'[communication]\ncopy_schedule="{schedule}"\n')
    assert Policy.load(path).communication['copy_schedule']==schedule
  path.write_text('[communication]\ncopy_schedule="unsupported"\n')
  with pytest.raises(ValueError,match='copy_schedule'):Policy.load(path)
  # Instruction selection is internal. Legacy backend switches must not be
  # silently accepted after transport becomes unconditionally vec based.
  for backend in ('auto','libc','vec'):
    path.write_text(f'[communication]\nrow_copy_backend="{backend}"\n')
    with pytest.raises(ValueError,match='unsupported communication policy'):
      Policy.load(path)
