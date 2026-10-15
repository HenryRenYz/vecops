# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bounded supplemental builders: isolated chunk frontends and one native job.

This coordinator only builds. It never reserves timing CPUs or calibrates.
Each chunk keeps its own row-status, generated-source manifest, deadline and
native cache, while completed and interrupted prior runs remain untouched.
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dsl_matmul_supplemental_catalog import manifest


def cpu_list(value):
  values=[]
  for part in value.split(','):
    if '-' in part:
      lo,hi=map(int,part.split('-'))
      values.extend(range(lo,hi+1))
    else:
      values.append(int(part))
  return sorted(set(values))


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--architecture',choices=('x86','arm'),required=True)
  parser.add_argument('--features',nargs='*',default=[])
  parser.add_argument('--work-dir',type=Path,required=True)
  parser.add_argument('--build-cpus',required=True)
  parser.add_argument('--workers',type=int,default=8)
  parser.add_argument('--chunk-size',type=int,default=16)
  parser.add_argument('--chunk-timeout',type=int,default=7200)
  parser.add_argument('--case-timeout',type=int,default=1800)
  parser.add_argument('--batch-input-attestation',action='store_true',help='enable the private real-input guard for new packing chunks')
  parser.add_argument('--suites',nargs='*',choices=('MatmulPack','MatmulWeightReuse','StatefulMatmul'),
    default=['MatmulPack','MatmulWeightReuse','StatefulMatmul'])
  parser.add_argument('--cpu',type=int,required=True,help='unused timing affinity passed through build-only CLI')
  parser.add_argument('--cc',required=True)
  parser.add_argument('--cxx',required=True)
  args=parser.parse_args()
  cpus=cpu_list(args.build_cpus)
  if not cpus or not 0<args.workers<=len(cpus) or args.chunk_size<=0 or args.chunk_timeout<=0:
    parser.error('positive workers/chunk-size/timeout and at least one distinct CPU per worker required')
  root=args.root.resolve();work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=True)
  coverage=manifest(args.architecture,args.features,root)
  (work/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
  groups=[]
  for suite in args.suites:
    rows=[r for r in coverage['registrations'] if r['suite']==suite and r['dsl_equivalent']]
    for index in range((len(rows)+args.chunk_size-1)//args.chunk_size):
      groups.append({'suite':suite,'chunk':index,'registrations':[r['registration_id']
        for r in rows[index*args.chunk_size:(index+1)*args.chunk_size]]})
  records=[]

  def lane(position):
    results=[]
    for group in groups[position::args.workers]:
      suite=group['suite'];index=group['chunk']
      path=work/suite;path.mkdir(exist_ok=True)
      command=['taskset','-c',str(cpus[position]),sys.executable,'-u',
        str(root/'benchmarks/dsl_matmul_supplemental_entry.py'),'--root',str(root),
        '--runner','pack' if suite=='MatmulPack' else 'lifecycle','--',
        '--architecture',args.architecture,'--work-dir',str(path),'--chunk-index',str(index),
        '--chunk-size',str(args.chunk_size),'--build-only','--cpu',str(args.cpu),'--jobs','1',
        '--cc',args.cc,'--cxx',args.cxx]
      if suite!='MatmulPack':command+=['--suite',suite]
      else:command+=['--case-timeout',str(args.case_timeout)]
      if suite=='MatmulPack' and args.batch_input_attestation:
        command+=['--batch-input-attestation']
      if args.features:command+=['--features',*args.features]
      log=path/f'chunk-{index:03d}.log'
      started=time.time()
      with log.open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
          env=os.environ.copy(),start_new_session=True)
        try:
          code=process.wait(timeout=args.chunk_timeout)
        except subprocess.TimeoutExpired:
          # The process group belongs only to this chunk, including its compiler.
          try:os.killpg(process.pid,signal.SIGTERM)
          except ProcessLookupError:pass
          try:process.wait(timeout=10)
          except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
          code=124
      result={**group,'exit_code':code,'build_cpu':cpus[position],'log':str(log),
        'started_unix':started,'finished_unix':time.time(),'command':command}
      results.append(result)
      (work/f'lane-{position:02d}.json').write_text(json.dumps(results,indent=2)+'\n')
      print(suite,index,code,flush=True)
    return results

  with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    for result in pool.map(lane,range(args.workers)):
      records.extend(result)
  (work/'build-status.json').write_text(json.dumps({'groups':records,
    'completed':len(records)==len(groups),'all_exit_zero':all(r['exit_code']==0 for r in records),
    'native_jobs_per_frontend':1,'frontend_workers':args.workers,'timed':False},indent=2)+'\n')


if __name__=='__main__':
  main()
