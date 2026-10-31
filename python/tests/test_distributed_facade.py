# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""The same logical collective facade must replay on native and Gloo groups."""
import os
import pytest


def _facade(rank,size,store,backend):
  os.environ['VECOPS_NUM_THREADS']='2'
  if lists := os.environ.get('VECOPS_TEST_CPU_LISTS'):
    import json
    os.sched_setaffinity(0, json.loads(lists)[rank])
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops.graph import GraphCapture
  torch.set_num_threads(2)
  vd.register_backend()
  dist.init_process_group(backend,init_method='file://'+store,rank=rank,world_size=size)
  try:
    group=vd.group()
    class Program(torch.nn.Module):
      def forward(self,x):
        return (group.all_gather(x,dim=1,hints=vd.Hints(reuse='once',copy_levels=(0,))),
                group.reshard(x,source_axis=0,target_axis=1))
    def reference(i):
      sources=[torch.arange(2*size*3).reshape(2,size*3)+r*1000+i for r in range(size)]
      return torch.cat(sources,dim=1),torch.cat([x[:,rank*3:(rank+1)*3] for x in sources],dim=0)
    source=torch.arange(2*size*3).reshape(2,size*3)+rank*1000
    program=Program()
    expected=reference(0)
    for result,ref in zip(program(source),expected):
      assert torch.equal(result,ref)
    capture=GraphCapture('portable-collectives')
    capture.capture_torch(program,source)
    executable=capture.compile_torch(native_operators=vd.native_operators())
    for i in (1,7):
      actual=executable(source+i)
      for result,ref in zip(actual,reference(i)):
        assert torch.equal(result,ref)
    if backend=='vecops':
      pending_output=torch.empty(size)
      pending=dist.all_gather_into_tensor(pending_output,torch.tensor([float(rank)]),async_op=True)
      if rank==0:
        pending.wait()
      actual=executable(source+9)  # Other ranks must drain the earlier eager Work.
      pending.wait()
      assert torch.equal(pending_output,torch.arange(float(size)))
      for result,ref in zip(actual,reference(9)):
        assert torch.equal(result,ref)
  finally:
    dist.destroy_process_group()


@pytest.mark.parametrize('backend',['vecops','gloo'])
def test_portable_facade_graph(tmp_path,backend):
  import torch.multiprocessing as mp
  if backend=='vecops':
    from vecops.distributed._build import load
    load()
  mp.spawn(_facade,args=(3,str(tmp_path/'store'),backend),nprocs=3,join=True)
