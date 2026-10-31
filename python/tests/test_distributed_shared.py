# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Real shared leases/page aliases, including a delayed reader and graph replay."""
import os
import pytest


def _shared(rank, size, store):
  os.environ['VECOPS_NUM_THREADS']='2'
  if lists := os.environ.get('VECOPS_TEST_CPU_LISTS'):
    import json
    os.sched_setaffinity(0,json.loads(lists)[rank])
  import ctypes
  import time
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops.graph import GraphCapture
  torch.set_num_threads(2)
  vd.register_backend()
  dist.init_process_group('vecops',init_method='file://'+store,rank=rank,world_size=size,
      pg_options=vd.Options(timeout_ms=10000))
  try:
    # Synchronous consumer sees rank snapshots, not a materialized full tensor.
    library=torch.library.Library('vecops_test_consumer','DEF')
    library.define('pack(Tensor addresses, Tensor(a!) output, int? count=None) -> ()')
    def pack(addresses, output, count):
      for peer,address in enumerate(addresses.tolist()):
        values=(ctypes.c_float*count).from_address(address)
        output[peer].copy_(torch.tensor(list(values)))
      if rank==size-1:
        time.sleep(.002)  # Next publication must not overwrite this lease.
    library.impl('pack',pack,'CPU')
    source=torch.arange(7,dtype=torch.float32)+rank*100
    output=torch.empty(size,7)
    before=vd.statistics()['copy_bytes']
    vd.consume_all_gather(source,output,'vecops_test_consumer::pack',[7],owners=(library,))
    # A configured hierarchy may copy additional domain replicas, but must
    # not also materialize a private P-way gathered result for this consumer.
    copied=vd.statistics()['copy_bytes']-before
    assert source.numel()*4 <= copied <= source.numel()*4*size
    assert torch.equal(output,torch.stack([torch.arange(7)+r*100 for r in range(size)]))
    class Consume(torch.nn.Module):
      def forward(self,x):
        out=torch.empty(size,7)
        return vd.consume_all_gather(x,out,'vecops_test_consumer::pack',[7],owners=(library,))
    capture=GraphCapture('borrowed-consumer')
    capture.capture_torch(Consume(),source)
    executable=capture.compile_torch(native_operators=vd.native_operators())
    for i in range(3):
      assert torch.equal(executable(source+i),output+i)
    for placement in ('source','destination'):
      for count in (0,1,17,4097):
        exchange=vd.PageExchange((count,),torch.float32,slots=2,placement=placement)
        assert exchange.producer().stride(0)*4 % os.sysconf('SC_PAGE_SIZE')==0
        for i in range(8):
          slot=i%2
          exchange.begin(slot)
          for peer,part in enumerate(exchange.producer(slot)):
            part.fill_(rank*100+peer+i*1000)
          exchange.publish(slot)
          expected=torch.stack([torch.full((count,),r*100+rank+i*1000.) for r in range(size)])
          if rank==size-1 and i==3:time.sleep(.02)
          assert torch.equal(exchange.consumer(slot),expected)
          exchange.retire(slot)
        if count==17:
          class Program(torch.nn.Module):
            def forward(self,x):
              exchange.begin()
              exchange.producer().copy_(x)
              exchange.publish()
              out=exchange.consumer().clone()
              exchange.retire()
              return out
          x=torch.stack([torch.full((count,),rank*100+peer+9000.) for peer in range(size)])
          capture=GraphCapture('page-exchange')
          capture.capture_torch(Program(),x)
          executable=capture.compile_torch(native_operators=vd.native_operators())
          for i in range(12):
            result=executable(x+i)
            ref=torch.stack([torch.full((count,),r*100+rank+9000.+i) for r in range(size)])
            assert torch.equal(result,ref)
        dist.barrier()
    # Both slots may be live before either is retired.
    exchange=vd.PageExchange((3,),torch.int64,slots=2)
    for slot in range(2):
      exchange.begin(slot);exchange.producer(slot).fill_(rank+slot*100);exchange.publish(slot)
    for slot in (1,0):
      assert torch.equal(exchange.consumer(slot)[:,0],torch.arange(size)+slot*100)
      exchange.retire(slot)
    dist.barrier()
    vd.seal()
    before=vd.statistics()
    for i in range(1000):
      exchange.begin();exchange.producer().fill_(rank+i*100);exchange.publish()
      assert torch.equal(exchange.consumer()[:,0],torch.arange(size)+i*100)
      exchange.retire()
    after=vd.statistics()
    assert after['capacity_growths']==before['capacity_growths']
    assert after['copy_bytes']==before['copy_bytes']
    with pytest.raises(RuntimeError,match='after seal'):
      vd.PageExchange((2,),torch.float32)
    dist.barrier()
  finally:
    dist.destroy_process_group()


@pytest.mark.parametrize('size',[1,3])
def test_shared_native_and_page_aliases(tmp_path,size):
  from vecops.distributed._build import load
  import torch.multiprocessing as mp
  load()
  mp.spawn(_shared,args=(size,str(tmp_path/'store')),nprocs=size,join=True)
