# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Local producer failure or protocol misuse must wake cross-process waiters."""
import os
import pytest


def _fail(rank,size,store,case):
  os.environ['VECOPS_NUM_THREADS']='2'
  if lists:=os.environ.get('VECOPS_TEST_CPU_LISTS'):
    import json
    os.sched_setaffinity(0,json.loads(lists)[rank])
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  torch.set_num_threads(2);vd.register_backend()
  dist.init_process_group('vecops',init_method='file://'+store,rank=rank,world_size=size,
      pg_options=vd.Options(timeout_ms=2000))
  try:
    region=vd.PageExchange((17,),torch.float32)
    if rank==0:
      if case=='cancel':region.begin();region.abort()
      else:
        with pytest.raises(RuntimeError,match='requires begin'):region.publish()
    else:
      with pytest.raises(RuntimeError,match='aborted'):
        region.begin();region.producer().fill_(rank);region.publish()
  finally:
    dist.destroy_process_group()


@pytest.mark.parametrize('case',['cancel','misuse'])
def test_failed_lease_wakes_peers(tmp_path,case):
  import torch.multiprocessing as mp
  from vecops.distributed._build import load
  load();mp.spawn(_fail,args=(3,str(tmp_path/'store'),case),nprocs=3,join=True)
