# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Real native graph replay consumes resource-generated routes and pool facts."""
import os
import json
import time
import statistics
from pathlib import Path


def _worker(rank,size,store):
  os.environ.update(VECOPS_NUM_THREADS='1',VECOPS_THREAD_POOL='native',VECOPS_COMM_POOL=f'bank-{rank}')
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops import memory
  from vecops.graph import GraphCapture
  from vecops.planner import GraphMemoryCompiler
  from vecops.memory._resource import value
  m=memory.native
  torch.set_num_threads(1)
  pools=[value('MemoryPool',id=f'bank-{r}',capacity_bytes=64<<20,address_space='host',granularity_bytes=4096,
    page_sizes_bytes=[4096],capacity_group=f'capacity-{r}',read_service=f'service-{r}',write_service=f'service-{r}') for r in range(size)]
  resources=[value('Resource',id=f'service-{r}',rule='opaque',read=100.,write=100.,
    sharing=m.SharingMode.read_write,kind=m.ResourceKind.bandwidth) for r in range(size)]
  resources += [value('Resource',id=f'directed-{direction}',rule='custom fabric',bandwidth=bandwidth,
    sharing=m.SharingMode.shared,kind=m.ResourceKind.bandwidth) for direction,bandwidth in [('a',.5),('b',.8)]]
  edges=[]
  for reader in range(size):
    for owner in range(size):
      fabric=[] if reader//2==owner//2 else ['directed-a' if reader<2 else 'directed-b']
      edges.append(value('AccessConnection',id=f'path-{reader}-{owner}',rule='custom path',agent=f'agent-{reader}',
        pool=f'bank-{owner}',access=[m.AccessMode.read,m.AccessMode.write],mapping=m.MappingMode.direct,
        bandwidth={'read':100.,'write':100.},read_fabric=fabric,write_fabric=fabric))
  graph=value('ResourceGraph',pools=pools,resources=resources,access=edges,
    agents=[value('Agent',id=f'agent-{r}',capabilities=['host_load_store'],address_space='host') for r in range(size)],
    capacity_groups=[value('CapacityGroup',id=f'capacity-{r}',capacity_bytes=64<<20) for r in range(size)])
  discovery=value('LinuxDiscovery',graph=graph,
    scope=value('Scope',executable_agents=[f'agent-{rank}'],allocatable_pools=[p.id for p in pools],
      accessible_pools=[p.id for p in pools],execution_slots={f'agent-{rank}':1}),
    pool_bindings=[value('LinuxPoolBinding',pool=p.id,numa_nodes=[]) for p in pools],
    pool_available_bytes={p.id:64<<20 for p in pools})
  runtime=memory.Runtime(m.make_linux_runtime(discovery,value('LinuxRuntimeOptions',participant=f'rank-{rank}',
    capacity_directory=os.environ['VECOPS_MEMORY_CAPACITY_DIRECTORY'])))
  with memory.use_runtime(runtime):
    vd.register_backend()
    dist.init_process_group('vecops',init_method='file://'+store,rank=rank,world_size=size,
      pg_options=vd.Options.from_file(store+'.toml',threads=1,timeout_ms=15000))
    try:
      source=torch.arange(1<<18,dtype=torch.float32)+rank*1000000
      def function(x):
        output=torch.empty((size,x.numel()),dtype=x.dtype)
        vd.launch('all_gather',list(output.unbind()),[x])
        return output
      capture=GraphCapture('opaque-resource-communication')
      from vecops.graph._metadata import metadata_mode
      with metadata_mode():capture.capture_torch_metadata(function,source)
      plan=capture.prepare_torch(borrow_outputs=True,native_operators=vd.native_operators())
      view=runtime.view()
      requirements=plan.memory_requirements(allowed_pools=(f'bank-{rank}',),consumers=(f'agent-{rank}',))
      program=GraphMemoryCompiler(runtime,view,runtime.snapshot(view)).compile(requirements)
      prepared=program.prepare(runtime.native,view).materialize(external={r.id:source for r in requirements if not r.allocated})
      plan.bind_metadata((source,))
      executable=plan.compile(runtime=runtime,memory_program=prepared)
      routing=vd.statistics()['routing']
      assert routing['partitions']==[[0,0,1,1]]
      assert routing['synchronization']['source']=='cold-native-barrier'
      assert len(routing['synchronization']['max_rank_samples'])==3
      selected=routing['plans'][-1]
      assert selected['modes']==[1] and selected['selected']['comparison_complete']
      assert selected['selected']['quality']=='incomplete'  # publisher source remains a cancelled common term
      root=next(r for r in requirements if r.allocated)
      output=prepared.tensor(root.id,(size,source.numel()),torch.float32)
      outputs=list(output.unbind())
      flat=vd.backend().prepare('all_gather',outputs,[source],[],[],0)
      barrier=vd.backend().prepare('barrier',[],[],[],[],0)
      def flat_call():torch.ops.vecops_comm.run.default(flat,outputs,[source])
      def auto_call():return executable(source)
      # Measure the real borrowed-source operator and then consume its exact
      # profile through the graph frontend. Ordinary AllGather calibration is
      # deliberately present and must never substitute for consumer evidence.
      import vecops.language as T
      @T.jit(out_idx=[1])
      def pack_factory(peers,delay):
        Peers,Delay=peers,delay
        Count,Stride=T.dynamic(lower=0),T.dynamic(lower=0)
        @T.template(source=str(Path(__file__).with_name('ResidentPeerPack.cpp')))
        def kernel(addresses:T.Tensor((Peers,),(1,),T.int64),
                   out:T.Tensor((Peers,Count),(Stride,1),T.float32)):
          Delay
        return kernel
      pack=pack_factory(size,0)
      pack.explicit_torch_function()(torch.tensor([source.data_ptr()]*size),output)
      operation,specs=pack.explicit_torch_binding()
      from vecops.distributed import tuning
      case=tuning.tune_consumer(source,output,operation._schema.name,specs,candidates=[[0],[1]],
          output_placement={f'bank-{rank}':output.nbytes},iterations=2,rounds=3,warmup=1)
      copy_key=vd.backend().plan_key('all_gather',outputs,[source])
      tuning.write_profile(store+'.profile.json',[case,{'key':copy_key,'selected':[1]}])
      def borrowed(x):
        result=torch.empty((size,x.numel()),dtype=x.dtype)
        return vd.consume_all_gather(x,result,operation._schema.name,specs,owners=(pack,))
      borrowed_capture=GraphCapture('exact-consumer-profile')
      with metadata_mode():borrowed_capture.capture_torch_metadata(borrowed,source)
      inventory=tuning.consumer_requests([borrowed_capture,borrowed_capture])
      assert len(inventory)==1 and inventory[0].operator_name==operation._schema.name
      assert inventory[0].source.shape==tuple(source.shape)
      assert inventory[0].group is dist.distributed_c10d._get_default_group()
      borrowed_plan=borrowed_capture.prepare_torch(borrow_outputs=True,native_operators=vd.native_operators())
      borrowed_req=borrowed_plan.memory_requirements(allowed_pools=(f'bank-{rank}',),consumers=(f'agent-{rank}',))
      borrowed_program=GraphMemoryCompiler(runtime,view,runtime.snapshot(view)).compile(borrowed_req)
      borrowed_prepared=borrowed_program.prepare(runtime.native,view).materialize(external={r.id:source for r in borrowed_req if not r.allocated})
      borrowed_plan.bind_metadata((source,))
      borrowed_replay=borrowed_plan.compile(runtime=runtime,memory_program=borrowed_prepared)
      consumer_record=vd.statistics()['routing']['plans'][-1]
      assert consumer_record['source']=='empirical-consumer',consumer_record
      assert consumer_record['profile_key']==case['key']
      assert consumer_record['modes']==case['selected']
      vd.seal()
      for _ in range(3):flat_call();actual=auto_call()
      for peer in range(size):torch.testing.assert_close(actual[peer],torch.arange(source.numel(),dtype=torch.float32)+peer*1000000,rtol=0,atol=0)
      borrowed_result=borrowed_replay(source)
      for peer in range(size):torch.testing.assert_close(borrowed_result[peer],torch.arange(source.numel(),dtype=torch.float32)+peer*1000000,rtol=0,atol=0)
      before=vd.statistics();timings={'flat':[],'resource':[]}
      for round in range(5):
        for name,call in ([('flat',flat_call),('resource',auto_call)] if round%2==0 else [('resource',auto_call),('flat',flat_call)]):
          torch.ops.vecops_comm.run.default(barrier,[],[])
          begin=time.perf_counter_ns()
          for _ in range(8):call()
          timings[name].append((time.perf_counter_ns()-begin)/8)
      after=vd.statistics()
      assert after['plans']==before['plans'] and after['capacity_growths']==before['capacity_growths']
      assert len(after['routing']['plans'])==len(before['routing']['plans'])==2
      reports=dist.FileStore(store+'.reports',size)
      reports.set(str(rank),json.dumps(timings))
      if rank==0:
        rows=[json.loads(reports.get(str(peer)).decode()) for peer in range(size)]
        result={name:statistics.median(max(row[name][i] for row in rows) for i in range(5)) for name in timings}
        Path(os.environ.get('VECOPS_COMM_TIMING_REPORT',store+'.timing.json')).write_text(json.dumps({'maxrank_median_ns':result,'routing':routing,
          'scope':'synthetic reference topology on one x86 host; executor correctness/overhead, not hardware bandwidth validation'},indent=2)+'\n')
      del actual,output,outputs,executable
      del borrowed_result,borrowed_replay
      borrowed_prepared.close()
      prepared.close()
    finally:
      dist.destroy_process_group()


def test_resource_generated_graph_route_is_native_and_stable(tmp_path):
  import torch.multiprocessing as mp
  from vecops.distributed._build import load
  load()
  store=str(tmp_path/'store')
  Path(store+'.toml').write_text('schema="communication-policy/0.1"\n[communication]\nrelay_threshold_bytes=0\nsync_warmup=1\nsync_rounds=3\n')
  mp.spawn(_worker,args=(4,store),nprocs=4,join=True)
