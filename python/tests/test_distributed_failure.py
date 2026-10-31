# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Invalid local buffers and wire-type mismatch must wake every peer."""
import os
import pytest


def _fail(rank, size, store, case):
  os.environ['VECOPS_NUM_THREADS']='2'
  if lists := os.environ.get('VECOPS_TEST_CPU_LISTS'):
    import json
    os.sched_setaffinity(0, json.loads(lists)[rank])
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  torch.set_num_threads(2)
  vd.register_backend()
  dist.init_process_group('vecops', init_method='file://'+store,rank=rank,world_size=size,
                          pg_options=vd.Options(timeout_ms=2000))
  try:
    source=torch.arange(9,dtype=torch.float32).reshape(3,3)
    if case=='dtype' and rank==0:
      source=source.to(torch.int32)  # same byte count, different wire type
    outputs=[torch.empty_like(source) for _ in range(size)]
    if case=='overlap' and rank==0:
      outputs[0]=torch.zeros(1,dtype=source.dtype).expand(3,3)
    with pytest.raises((RuntimeError,ValueError),match='overlap|mismatch|aborted'):
      dist.all_gather(outputs,source)
  finally:
    dist.destroy_process_group()


@pytest.mark.parametrize('case',['overlap','dtype'])
def test_collective_errors_propagate(tmp_path,case):
  from vecops.distributed._build import load
  import torch.multiprocessing as mp
  load()
  mp.spawn(_fail,args=(3,str(tmp_path/'store'),case),nprocs=3,join=True)
