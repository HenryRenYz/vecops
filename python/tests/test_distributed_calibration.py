# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Calibration is a group-wide decision, including asymmetric worker counts."""
import os
import pytest


def _calibration(rank,size,store,directory,case):
  os.environ.pop('VECOPS_COMM_TUNING_TAG',None)
  if lists:=os.environ.get('VECOPS_TEST_CPU_LISTS'):
    import json
    os.sched_setaffinity(0,json.loads(lists)[rank])
  import json
  from pathlib import Path
  import warnings
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops.distributed.tuning import write_profile,profile_context
  torch.set_num_threads(1);vd.register_backend()
  root=Path(directory)
  config=root/f'policy-{rank}.toml'
  text='''[communication]
algorithm="flat"
relay_threshold_bytes=0
'''
  if case=='enablement' and rank==0 or case=='missing':
    text+='calibration="absent.json"\n'
  config.write_text(text)
  options=vd.Options(config=str(config),threads=rank+1,timeout_ms=3000)
  try:
    if case!='stale':
      with pytest.raises(ValueError,match='enablement|calibration failed'):
        dist.init_process_group('vecops',init_method='file://'+store,rank=rank,world_size=size,pg_options=options)
      return
    dist.init_process_group('vecops',init_method='file://'+store,rank=rank,world_size=size,pg_options=options)
    native=vd.backend();signature=json.loads(native.tuning_context)
    assert signature['threads']==list(range(1,size+1))
    signatures=[None]*size
    dist.all_gather_object(signatures,signature)
    assert all(x==signature for x in signatures)
    # Registry lookup survives a fresh wrapper for the same native lifetime.
    from types import SimpleNamespace
    wrapped=SimpleNamespace(_routing_id=native._routing_id)
    assert profile_context(wrapped)==profile_context(native)
    value=torch.tensor([float(rank)])
    outputs=[torch.empty_like(value) for _ in range(size)]
    key=native.plan_key('all_gather',outputs,[value])
    profile=root/'routes.json'
    evidence=write_profile(profile,[{'key':key,'selected':[1]}])
    assert evidence['schema_version']==2 and evidence['runtime_context']==signature
    assert json.loads(native.tuning_context)==signature
    dist.barrier();dist.destroy_process_group()
    # Same policy identity, now referencing its calibration. One member's
    # changed tuning tag invalidates the group-wide workload context for ALL.
    config.write_text(text+'calibration="routes.json"\n')
    if rank==1:os.environ['VECOPS_COMM_TUNING_TAG']='frequency-changed'
    with warnings.catch_warnings(record=True) as caught:
      dist.init_process_group('vecops',init_method='file://'+store+'-2',rank=rank,world_size=size,pg_options=options)
    assert any('stale' in str(row.message) for row in caught)
    dist.all_gather(outputs,value)
    assert [x.item() for x in outputs]==list(range(size))
    dist.destroy_process_group()
    os.environ.pop('VECOPS_COMM_TUNING_TAG',None)
    with warnings.catch_warnings(record=True) as caught:
      dist.init_process_group('vecops',init_method='file://'+store+'-3',rank=rank,world_size=size,pg_options=options)
    assert not any('stale' in str(row.message) for row in caught)
    dist.all_gather(outputs,value+10)
    assert [x.item() for x in outputs]==list(range(10,10+size))
    dist.barrier()
  finally:
    if dist.is_initialized():dist.destroy_process_group()


@pytest.mark.parametrize('case',['stale','enablement','missing'])
def test_calibration_consensus_with_asymmetric_workers(tmp_path,case):
  import torch.multiprocessing as mp
  from vecops.distributed._build import load
  load();mp.spawn(_calibration,args=(3,str(tmp_path/'store'),str(tmp_path),case),nprocs=3,join=True)
