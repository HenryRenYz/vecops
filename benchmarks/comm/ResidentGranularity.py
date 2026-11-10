# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compare real resident collective payloads under rotated chunk policies.

Run with torchrun, one process per CPU list. All input/output and shared pages
use the resource runtime's selected memory pool. Excludes capture, allocation and first touch.
This measures collective replay, not whole-model speedup or pure load/store.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--config',required=True)
  p.add_argument('--cpus',required=True)
  p.add_argument('--output',required=True)
  p.add_argument('--chunks-kib',type=int,nargs='+',default=[256,128,64,16])
  p.add_argument('--bytes',type=int,nargs='+',default=[524288,18874368])
  p.add_argument('--row-bytes',type=int,nargs='+',default=[4096,24576])
  p.add_argument('--base-pages',action='store_true',help='disable huge pages for a separate page-size sensitivity run')
  p.add_argument('--iterations',type=int,default=20)
  p.add_argument('--rounds',type=int,default=5)
  args=p.parse_args()
  if min(args.chunks_kib+args.bytes+args.row_bytes+[args.iterations,args.rounds])<=0:
    p.error('sizes and iteration counts must be positive')
  if len(args.row_bytes)!=len(args.bytes) or any(size%run for size,run in zip(args.bytes,args.row_bytes)):
    p.error('--row-bytes must divide and correspond to each input byte count')
  output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
  rank=int(os.environ['RANK']);cpus=json.loads(Path(args.cpus).read_text())[rank]
  os.sched_setaffinity(0,cpus)
  os.environ.update(VECOPS_THREAD_POOL='native',VECOPS_NUM_THREADS=str(len(cpus)),
    VECOPS_THREAD_POOL_CPU_LIST=','.join(map(str,cpus)),OMP_WAIT_POLICY='PASSIVE',
    GOMP_SPINCOUNT='0',KMP_BLOCKTIME='0')
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops import memory
  from vecops.graph import GraphCapture
  torch.set_num_threads(len(cpus));vd.register_backend()
  dist.init_process_group('gloo');world=dist.get_world_size()
  from Memory import pool
  factory,pool_id,runtime=pool(base_pages=args.base_pages)
  def empty(shape):return factory.empty(tuple(shape),dtype=torch.uint8)
  tensors=[]
  for size,run in zip(args.bytes,args.row_bytes):
    if size%world:p.error('--bytes must be divisible by world size for AllToAll')
    source=empty((size,));source.fill_(rank+1)
    gather=empty((world,size));gather.zero_()
    interleaved=empty((size//run,world,run));interleaved.zero_()
    send=empty((world,size//world));target=empty(send.shape);target.zero_()
    for peer in range(world):send[peer].fill_((rank*world+peer)%256)
    tensors.extend([('all_gather',size,[source],list(gather.unbind()),gather),
                    ('all_gather_interleaved',size,[source.reshape(size//run,run)],list(interleaved.unbind(1)),interleaved),
                    ('all_to_all',size,list(send.unbind()),list(target.unbind()),target)])
  programs={};groups=[]
  for chunk in args.chunks_kib:
    path=output.parent/f'communication-chunk-{chunk}.toml'
    if rank==0:
      import re
      text=Path(args.config).read_text()
      text,count=re.subn(r'(?m)^chunk_bytes\s*=\s*\d+\s*$',f'chunk_bytes = {chunk*1024}',text)
      if count!=1:raise ValueError('benchmark config must contain exactly one chunk_bytes setting')
      path.write_text(text)
    dist.barrier()
    group=dist.new_group(backend='vecops',pg_options=vd.Options.from_file(path,threads=len(cpus)))
    groups.append(group)
    for operation,size,inputs,outputs,target in tensors:
      def call(*values):
        vd.launch('all_gather' if operation.startswith('all_gather') else operation,outputs,list(values),group=group)
        return target
      capture=GraphCapture('resident-collective-granularity')
      capture.capture_torch(call,*inputs)
      replay=capture.compile_torch(native_pointwise=True,borrow_outputs=True,
                                  native_operators=vd.native_operators(group))
      for _ in range(3):replay(*inputs)
      for peer in range(world):
        expected=peer+1 if operation.startswith('all_gather') else (peer*world+rank)%256
        value=outputs[peer]
        assert value[(0,)*value.ndim].item()==expected and value[(-1,)*value.ndim].item()==expected
      programs[(chunk,operation,size)]=(replay,inputs)
    vd.seal(group)
  timings={key:[] for key in programs}
  keys=list(programs)
  for turn in range(args.rounds):
    # Rotate BOTH the policy and payload order, identically on all ranks.
    shift=turn%len(keys)
    for key in keys[shift:]+keys[:shift]:
      dist.barrier();replay,inputs=programs[key]
      start=time.perf_counter_ns()
      for _ in range(args.iterations):replay(*inputs)
      timings[key].append((time.perf_counter_ns()-start)/args.iterations)
  records=[dict(chunk_kib=k[0],operation=k[1],input_bytes=k[2],nanoseconds=v) for k,v in timings.items()]
  gathered=[None]*world;dist.all_gather_object(gathered,dict(rank=rank,records=records))
  if rank==0:
    summary=[]
    for i,row in enumerate(records):
      maximums=[max(r['records'][i]['nanoseconds'][j] for r in gathered) for j in range(args.rounds)]
      summary.append({k:v for k,v in row.items() if k!='nanoseconds'}|
                     dict(maxrank_median_ns=statistics.median(maximums),maxrank_round_ns=maximums))
    output.write_text(json.dumps(dict(summary=summary,ranks=gathered,
      base_pages=args.base_pages,scope='resident collective replay; fixed placements, same pool, all allocations/capture/warmup excluded'),indent=2)+'\n')
  programs.clear()
  for group in groups:dist.destroy_process_group(group)
  dist.destroy_process_group()


if __name__=='__main__':main()
