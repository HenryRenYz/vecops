# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact-key parallel pure lowering followed by guarded private native batches.

--registration-ids-file selects exact architecture-scoped registrations before
grouping. The selection path, raw content hash and IDs are immutable on resume
and propagated to every lower/native worker.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from dsl_benchmark_selection import selection_identity, select_rows, verify_selection_resume


def digest(data):
  return hashlib.sha256(data).hexdigest()


def save(path,value):
  path.parent.mkdir(parents=True,exist_ok=True)
  temporary=path.with_suffix(f'.{os.getpid()}.tmp')
  temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def make_plan(rows, suite, scenario_for, identity=None):
  """Group only selected registrations; never expand a selected shared template."""
  rows=select_rows(rows,identity,'registration_ids',lambda row:row['registration_id'])
  groups={}
  for row in rows:
    if suite=='packing':
      mode=row.get('compensation_mode','PackOnly');exact=row['metadata']=='Const' or mode!='PackOnly'
      key=repr(tuple(row[k] for k in ('native_a','native_b','native','memory','pipeline','side','selector','metadata'))
        +(mode,(row['spatial'],row['k']) if exact else None))
    else:
      scenario=scenario_for(row)
      key=repr((scenario.compilation_key,row.get('calls_per_pack',1),row['suite'],scenario.reference_compensated_pack))
    groups.setdefault(digest(key.encode()),{'key':key,'rows':[]})['rows'].append(row)
  return {'selection':identity,'groups':list(groups.values()),'registrations':len(rows),'timed':False,
    'dsl_pass_options':{},'suite':suite,'dsl_workspace':'explicit result T.workspace outside repeat/call loops' if suite=='lifecycle' else None}


def verify_plan_resume(previous, current):
  verify_selection_resume(previous.get('selection'),current['selection'])
  if previous['groups']!=current['groups'] or previous.get('suite')!=current['suite']:
    raise ValueError('resume registration scope differs from immutable plan')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--work-dir',type=Path,required=True)
  parser.add_argument('--architecture',choices=('x86','arm'),required=True)
  parser.add_argument('--registration-ids-file',type=Path,help='exact architecture/registration_ids JSON selection; content-bound on resume')
  parser.add_argument('--suite',choices=('lifecycle','packing'),default='lifecycle')
  parser.add_argument('--features',nargs='*',default=[])
  parser.add_argument('--cc',required=True)
  parser.add_argument('--cxx',required=True)
  parser.add_argument('--target',default='Native')
  parser.add_argument('--lower-workers',type=int,default=24)
  parser.add_argument('--lower-indices',type=int,nargs='*',help='explicit nonoverlapping remaining group indices when adopting earlier lowering')
  parser.add_argument('--native-workers',type=int,default=8)
  parser.add_argument('--native-jobs',type=int,default=3)
  parser.add_argument('--plan-only',action='store_true',help='validate and save exact plan without lowering, building or timing')
  parser.add_argument('--native-only',action='store_true')
  parser.add_argument('--lower-only',action='store_true',help='collect proved sources without entering header attestation or native compilation')
  parser.add_argument('--prelower-dir',type=Path)
  parser.add_argument('--lower-cpu-base',type=int,default=0)
  parser.add_argument('--native-cpu-base',type=int,default=24)
  parser.add_argument('--timeout',type=int,default=7200)
  parser.add_argument('--worker',choices=('lower','native'))
  parser.add_argument('--index',type=int)
  args=parser.parse_args()
  if args.plan_only and (args.worker or args.index is not None or args.native_only or args.lower_only or args.lower_indices is not None):
    parser.error('plan-only cannot be combined with workers or execution-only options')
  if args.registration_ids_file is not None:args.registration_ids_file=args.registration_ids_file.resolve()
  root=args.root.resolve();work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=True)
  sys.path.insert(0,str(root/'benchmarks'))
  from dsl_snapshot_entry import select_snapshot
  select_snapshot(root)
  os.chdir(root.parents[1])
  import vecops
  import vecops.language as T
  from dsl_matmul_artifact_ab import template_contract
  from dsl_matmul_repair_suite import BatchInputAttestation, compile_cpp_source, compile_prelowered
  from dsl_matmul_source_catalog import source_metadata
  from dsl_matmul_supplemental_catalog import supplemental_cases
  from dsl_matmul_supplemental_lifecycle import scenario_for, templates
  from vecops.language._compiler.driver import _sme_grid_compiler, _strict_dsl_compiler

  def template_pair(row):
    if args.suite=='packing':
      from dsl_matmul_supplemental_pack import templates as pack_templates
      return pack_templates(row)
    return templates(row)[1]

  def metadata(row):
    if args.suite!='packing':return source_metadata(scenario_for(row))
    from dsl_matmul_supplemental_pack import packed_size
    s,k=row['spatial'],row['k'];side=row['side'];size=packed_size(s,k,row['native'])
    result=(vecops.TensorMeta((s,k) if side=='A' else (k,s),strides=(k,1) if side=='A' else (1,k),dtype=getattr(vecops,row['memory'])),
      vecops.TensorMeta((size,),dtype=getattr(vecops,row['native'])))
    if row.get('compensation_mode','PackOnly')!='PackOnly':result+=(vecops.TensorMeta((s,),dtype=vecops.int32),)
    return result

  rows=[r for r in supplemental_cases(args.architecture,args.features) if
        (args.suite=='packing' and r['suite']=='MatmulPack' and r['dsl_equivalent']) or
        (args.suite=='lifecycle' and (r['suite']=='MatmulWeightReuse' or r['suite']=='StatefulMatmul' and r['name']=='Legacy'))]
  identity=selection_identity(args.registration_ids_file,args.architecture,'registration_ids')
  plan=make_plan(rows,args.suite,scenario_for,identity)
  groups=plan['groups']
  plan_path=work/'plan.json'
  if plan_path.exists():verify_plan_resume(json.loads(plan_path.read_text()),plan)
  save(plan_path,plan)
  if args.plan_only:return
  if args.worker=='lower':
    group=groups[args.index];directory=work/'groups'/digest(group['key'].encode())
    row=group['rows'][0];pair=template_pair(row)
    selected=vecops.Compiler(target=args.target,cc=args.cc,cxx=args.cxx,cache_dir=directory/'policy-cache').snapshot()
    lowered=T.lower(pair[0],target=args.target,pass_options={'_vector_state':selected.vector_state})
    effective=_sme_grid_compiler(_strict_dsl_compiler(selected),lowered.artifact)
    text=lowered.artifact.source.encode();path=directory/'kernel.cpp';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(text)
    contract,_=template_contract(pair[0])
    save(directory/'proof.json',{'source':str(path),'raw_sha256':digest(text),'contract':contract,
      'effective_cflags':list(effective.cflags),'vector_state_token':selected.vector_state.env_token,'runtime_parallelism':True})
    return
  if args.worker=='native':
    jobs=args.native_jobs
    os.sched_setaffinity(0,set(range(args.native_cpu_base+args.index*jobs,args.native_cpu_base+(args.index+1)*jobs)))
    batch=groups[args.index::args.native_workers];directory=work/'native'/f'batch-{args.index:03d}'
    directory.mkdir(parents=True,exist_ok=True)
    compiler=vecops.Compiler(target=args.target,cc=args.cc,cxx=args.cxx,jobs=jobs,cache_dir=directory/'cache',build_dir=directory/'build')
    attestation=BatchInputAttestation(directory);artifacts=[];error=None
    try:
      with vecops.collect_compile_requests() as collector:
        for group in batch:
          group_dir=(args.prelower_dir or work/'groups')/digest(group['key'].encode());proof=json.loads((group_dir/'proof.json').read_text())
          pair=template_pair(group['rows'][0])
          compiled=(compile_prelowered(pair[0],proof,compiler,attestation=attestation),
                    compile_cpp_source(pair[1],compiler,attestation=attestation))
          for row in group['rows']:
            meta=metadata(row)
            for kernel in compiled:kernel.operator(*meta,1,**kernel._core.specialization_values)
            artifacts.append({'registration_id':row['registration_id'],'row':row,'dsl_source':str(compiled[0].source),'cpp_source':str(compiled[1].source)})
      save(directory/'manifest.json',{'artifacts':artifacts,'timed':False})
      attestation.guard('pre_build')
      vecops.compile_batch(collector.requests,parallelism=jobs,execution_parallelism=1)
    except Exception as exc:  # noqa: BLE001 -- retain native diagnostics before post-build guard
      error=repr(exc)
    try:
      attestation.guard('post_build')
    except Exception as exc:
      save(directory/'build-status.json',{'admitted':False,'attestation_error':repr(exc),'error':error});raise
    checkpoint=directory/'batch-input-attestation.json'
    save(directory/'build-status.json',{'admitted':error is None,'error':error,'registrations':[r['registration_id'] for r in artifacts],
      'attestation_checkpoint':str(checkpoint),'attestation_checkpoint_sha256':digest(checkpoint.read_bytes())})
    if error is not None:raise RuntimeError(error)
    return

  base=[sys.executable,'-u',str(root/'benchmarks/dsl_matmul_supplemental_accelerated.py'),
    '--root',str(root),'--work-dir',str(work),'--architecture',args.architecture,'--cc',args.cc,'--cxx',args.cxx,
    '--target',args.target,'--native-workers',str(args.native_workers),'--native-jobs',str(args.native_jobs),
    '--native-cpu-base',str(args.native_cpu_base),'--suite',args.suite]
  if args.registration_ids_file:base+=['--registration-ids-file',str(args.registration_ids_file.resolve())]
  if args.prelower_dir:base+=['--prelower-dir',str(args.prelower_dir.resolve())]
  if args.features:base+=['--features',*args.features]
  def run(stage,index,cpu):
    log=work/'logs'/f'{stage}-{index:04d}.log';log.parent.mkdir(exist_ok=True)
    with log.open('w') as stream:
      process=subprocess.run(['taskset','-c',str(cpu),*base,'--worker',stage,'--index',str(index)],
        stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout,check=False)
    result={'stage':stage,'index':index,'cpu':cpu,'exit_code':process.returncode,'log':str(log)}
    save(work/'statuses'/f'{stage}-{index:04d}.json',result);print(stage,index,process.returncode,flush=True)
    return result
  def lower_lane(lane):
    indices=range(len(groups)) if args.lower_indices is None else args.lower_indices
    return [run('lower',i,args.lower_cpu_base+lane) for i in list(indices)[lane::args.lower_workers]]
  lower_results=[]
  if not args.native_only:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.lower_workers) as pool:
      lower_results=[r for lane in pool.map(lower_lane,range(args.lower_workers)) for r in lane]
  if any(r['exit_code'] for r in lower_results):raise RuntimeError('pure lowering failures retained in statuses')
  if args.lower_only:
    complete=True
    for group in groups:
      path=work/'groups'/digest(group['key'].encode())/'proof.json'
      if not path.exists():complete=False;continue
      proof=json.loads(path.read_text())
      if digest(Path(proof['source']).read_bytes())!=proof['raw_sha256']:
        raise ValueError('adopted lowering source changed')
    save(work/'lower-complete.json',{'groups':len(groups),'registrations':len(rows),'lower':lower_results,'completed':complete,'timed':False})
    return
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.native_workers) as pool:
    native_results=list(pool.map(lambda i:run('native',i,args.native_cpu_base+i),range(args.native_workers)))
  save(work/'complete.json',{'groups':len(groups),'registrations':len(rows),'lower':lower_results,'native':native_results,
    'completed':not any(r['exit_code'] for r in native_results),'timed':False})


if __name__=='__main__':
  main()
