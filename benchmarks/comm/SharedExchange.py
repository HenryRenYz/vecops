# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Joint producer+AllToAll+consumer comparison, not a synchronization-only claim.

Page aliases shift traffic into producer stores or consumer reads. All three
programs perform the same BF16 production and reduction, under native replay.
Allocation, mmap, capture, compilation and warmup are excluded from timing.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--cpus',required=True)
  parser.add_argument('--config')
  parser.add_argument('--output',required=True)
  parser.add_argument('--iterations',type=int,default=20)
  parser.add_argument('--local',type=int,nargs='+',default=[16,96,128])
  args=parser.parse_args()
  rank=int(os.environ['RANK']);cpus=json.loads(Path(args.cpus).read_text())[rank]
  os.sched_setaffinity(0,cpus)
  os.environ.update(VECOPS_NUM_THREADS=str(len(cpus)),VECOPS_THREAD_POOL_CPU_LIST=','.join(map(str,cpus)),
      OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',KMP_BLOCKTIME='0')
  if args.config:os.environ['VECOPS_COMM_CONFIG']=str(Path(args.config).resolve())
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops.graph import GraphCapture
  vd.register_backend();torch.set_num_threads(len(cpus))
  dist.init_process_group('vecops');world=dist.get_world_size()
  records=[]
  for local in args.local:
    shape=(local,local,128)
    seed=torch.stack([torch.full(shape,rank*10+peer+1.,dtype=torch.bfloat16) for peer in range(world)])
    private=vd.PageExchange(shape,torch.bfloat16,placement='destination')
    # The materialized case uses the same memory tier as aliases. Its private
    # tensors retain independent writable storage, unlike either shared view.
    from Memory import pool
    factory,pool_id,runtime=pool()
    def empty():return factory.empty(tuple(seed.shape),dtype=seed.dtype)
    ordinary_source,ordinary_target=empty(),empty()
    for reuse in (1,4):
      results={}
      programs={}
      for mode in ('materialized','source','destination'):
        exchange=None if mode=='materialized' else (private if mode=='destination' else vd.PageExchange(shape,seed.dtype,placement='source'))
        def program(x):
          if exchange is None:
            torch.mul(x,2,out=ordinary_source)
            vd.launch('all_to_all',list(ordinary_target.unbind(0)),list(ordinary_source.unbind(0)))
            target=ordinary_target
          else:
            exchange.begin();torch.mul(x,2,out=exchange.producer());exchange.publish()
            target=exchange.consumer()
          result=target.sum(dtype=torch.float32)
          for _ in range(reuse-1):result=result+target.sum(dtype=torch.float32)
          if exchange is not None:exchange.retire()
          return result
        capture=GraphCapture('joint-page-exchange')
        capture.capture_torch(program,seed)
        executable=capture.compile_torch(native_operators=vd.native_operators())
        programs[mode]=executable
        for _ in range(5):value=executable(seed)
        expected=sum((peer*10+rank+1)*2 for peer in range(world))*local*local*128*reuse
        assert value.item()==expected,(mode,rank,value.item(),expected)
        results[mode]=[]
      # Rotate order to reduce fixed-order temperature/frequency bias.
      names=list(programs)
      for round in range(5):
        for mode in names[round%3:]+names[:round%3]:
          dist.barrier();started=time.perf_counter_ns()
          for _ in range(args.iterations):programs[mode](seed)
          elapsed=(time.perf_counter_ns()-started)/1e9/args.iterations
          results[mode].append(elapsed)
      records.append({'local':local,'peer_bytes':local*local*128*2,'reuse':reuse,'seconds':results})
      dist.barrier()
  gathered=[None]*world
  dist.all_gather_object(gathered,{'rank':rank,'records':records})
  if rank==0:
    summary=[]
    for index,row in enumerate(records):
      medians={mode:statistics.median(max(r['records'][index]['seconds'][mode][i] for r in gathered) for i in range(5)) for mode in row['seconds']}
      summary.append({k:v for k,v in row.items() if k!='seconds'}|{'maxrank_median_seconds':medians})
    Path(args.output).write_text(json.dumps({'ranks':gathered,'summary':summary,'scope':'joint producer+transport+consumer; no whole-model claim'},indent=2)+'\n')
  dist.destroy_process_group()


if __name__=='__main__':main()
