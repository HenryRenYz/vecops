# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fixed real payloads: resident pointwise/reduction, then an ATen boundary.

Run each provider/wait policy in a fresh bound process. Allocation, compilation,
first touch and validation are excluded. Htop occupancy is not a success metric.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--threads',type=int,default=36)
p.add_argument('--node',type=int)
p.add_argument('--iterations',type=int,default=50)
a=p.parse_args()
import torch
from vecops import memory
from vecops.planner import GraphMemoryCompiler
from vecops.graph import GraphCapture

torch.set_num_threads(a.threads)
records=[]
runtime=memory.current_runtime() or memory.default_runtime()
view=runtime.view()
allowed_pools=()
input_pool=None
if a.node is not None:
 matches={binding.object for binding in runtime.describe_bindings()
          if binding.labels.get('os_numa')==str(a.node)
          and binding.object in view.allocatable_pools}
 if len(matches)!=1:
  raise ValueError('--node must identify exactly one allocatable memory pool')
 allowed_pools=tuple(matches)
 input_pool=memory.Pool.create(runtime,allowed_pools[0],allowed_pools=allowed_pools,
                               retain_idle_bytes=0)
with torch.inference_mode():
 for n in (128,768):
  if a.node is not None:
   x=input_pool.empty((1,n,24,768),dtype=torch.float32,allow_domain_fallback=False)
   x.uniform_(-1,1)
  else:x=torch.rand(1,n,24,768)*2-1
  function=lambda x: ((x*.5).sum(-2)+1).relu()
  capture=GraphCapture('pool-boundary-payload')
  capture.capture_torch(function,x)
  plan=capture.prepare_torch(native_pointwise=True,borrow_outputs=True)
  requirements=tuple(req for req in plan.memory_requirements(allowed_pools=allowed_pools) if req.allocated)
  compiled=GraphMemoryCompiler(runtime,view,runtime.snapshot(view)).compile(requirements)
  prepared=compiled.prepare(runtime,view).materialize()
  program=plan.compile(runtime=runtime,view=view,memory_program=prepared)
  expected=torch.softmax(function(x),dim=-1)
  torch.testing.assert_close(torch.softmax(program(x),dim=-1),expected)
  for _ in range(10):torch.softmax(program(x),dim=-1)
  for mixed in (False,True):
   samples=[]
   cpu=time.process_time_ns();start=time.perf_counter_ns()
   for _ in range(a.iterations):
    begin=time.perf_counter_ns()
    y=program(x)
    if mixed:y=torch.softmax(y,dim=-1)
    samples.append(time.perf_counter_ns()-begin)
   wall=time.perf_counter_ns()-start;cpu=time.process_time_ns()-cpu
   records.append(dict(tokens=n,mixed_aten=mixed,samples_ns=samples,
       median_ns=statistics.median(samples),core_equivalents=cpu/wall,
       islands=program.allocator_policy['native_island_sizes']))
result=dict(torch=torch.__version__,provider=os.environ.get('VECOPS_THREAD_POOL'),
    wait=os.environ.get('OMP_WAIT_POLICY'),blocktime=os.environ.get('KMP_BLOCKTIME'),
    affinity=sorted(os.sched_getaffinity(0)),records=records)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print([(r['tokens'],r['mixed_aten'],round(r['median_ns']/1e6,3),round(r['core_equivalents'],2))for r in records],flush=True)
