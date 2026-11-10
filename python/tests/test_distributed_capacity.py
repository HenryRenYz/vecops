# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Real transport roots use one authority; imports and shutdown do not leak."""
import os
from pathlib import Path

import pytest


def _capacity_worker(rank, size, rendezvous, case):
  os.environ['VECOPS_NUM_THREADS']='1'
  import torch
  import torch.distributed as dist
  from vecops import distributed, memory
  torch.set_num_threads(1)
  options=memory.native.LinuxRuntimeOptions()
  options.participant=f'capacity-rank-{rank}'
  directory=Path(os.environ['VECOPS_MEMORY_CAPACITY_DIRECTORY'])
  if case=='different':
    directory=directory/f'rank-{rank}'
    directory.mkdir(mode=0o700)
  if case!='independent':options.capacity_directory=str(directory)
  runtime=memory.Runtime.discover(options=options)
  def committed():return sum(group.committed_bytes for group in runtime.snapshot().groups)
  assert committed()==0
  distributed.register_backend()
  with memory.use_runtime(runtime):
    if case!='shared':
      with pytest.raises(ValueError,match='one shared MemoryRuntime capacity authority'):
        dist.init_process_group('vecops',init_method='file://'+rendezvous,rank=rank,world_size=size)
      assert committed()==0
      return
    dist.init_process_group('vecops',init_method='file://'+rendezvous,rank=rank,world_size=size)
    try:
      assert committed()==1<<20  # source only; no potential relay allocation
      exchange=distributed.PageExchange((17,),torch.float32,slots=2)
      stride=exchange.producer().stride(0)*4
      assert committed()==(1<<20)+2*size*stride
      for iteration in range(3):
        exchange.begin()
        exchange.producer().fill_(rank+iteration)
        exchange.publish()
        for peer,row in enumerate(exchange.consumer()):
          assert torch.all(row==peer+iteration)
        exchange.retire()
      assert committed()==(1<<20)+2*size*stride
    finally:
      dist.destroy_process_group()
    # The PageExchange object still exists, but shutdown has certified stopped
    # access and removed its imported aliases before refunding the owned roots.
    assert committed()==0


@pytest.mark.parametrize('case',['shared','different','independent'])
def test_transport_capacity_authority_and_shutdown(tmp_path,case):
  import torch.multiprocessing as mp
  from vecops.distributed._build import load
  load()
  mp.spawn(_capacity_worker,args=(2,str(tmp_path/'store'),case),nprocs=2,join=True)
