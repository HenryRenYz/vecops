# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""ABBA parked/active comparison with real contiguous and interleaved vec copy.

Set CPU_LIST and process NUMA policy before startup. Separate processes select
wake policies; do not interpret this single-NUMA probe as fabric bandwidth.
"""
import argparse,ctypes,json,os,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
import torch,vecops
torch.set_num_threads(36)
cpus=os.environ['VECOPS_THREAD_POOL_CPU_LIST'];first=int(cpus.split('-')[0].split(',')[0]);os.sched_setaffinity(0,{first})
address=vecops._C.native_thread_pool_address(36)
library=ctypes.CDLL(a.library)
library.measure_copy_pool.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_uint,ctypes.c_uint,ctypes.POINTER(ctypes.c_double)]
def task_snapshot():
 result={}
 for directory in Path('/proc/self/task').glob('[0-9]*'):
  try:
   fields=dict(line.split(':',1)for line in (directory/'status').read_text().splitlines()if ':'in line)
   cpu=fields['Cpus_allowed_list'].strip()
   if not cpu.isdigit():continue
   result[directory.name]=(int(cpu),int(fields['voluntary_ctxt_switches']),int((directory/'schedstat').read_text().split()[0]))
  except (OSError,KeyError,ValueError):pass
 return result
result=[]
for label,rows,run,peers,workers in [('snapshot-512KiB',1,512<<10,1,2),('gather-4MiB',16384,32,8,36),('gather-144MiB',98304,192,8,36)]:
 for active in (False,True,True,False):
  with vecops.native_thread_pool_region(36,enabled=active):
   samples=(ctypes.c_double*50)()
   before=task_snapshot()
   status=library.measure_copy_pool(address,workers,rows,run,peers,50,100,samples)
   if status:raise RuntimeError((label,status))
   after=task_snapshot()
   idle=[(tid,cpu,v-before[tid][1],runtime-before[tid][2])for tid,(cpu,v,runtime)in after.items()if tid in before and first+workers<=cpu<first+36]
   result.append(dict(payload=label,bytes=rows*run*peers,workers=workers,active=active,gap_us=100,median_us=statistics.median(samples)/1000,samples_ns=list(samples),untargeted_sleep_switches=sum(x[2]for x in idle),untargeted_cpu_ns=sum(x[3]for x in idle),untargeted_tasks=idle))
a.output.write_text(json.dumps(dict(cpu_list=cpus,wake_policy=os.environ.get('VECOPS_THREAD_POOL_WAKE_POLICY','per-worker'),blocks=result),indent=2)+'\n')
print([(r['payload'],r['active'],round(r['median_us'],2))for r in result])
