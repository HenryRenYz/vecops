# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Build and audit exact equivalent MatmulPack registrations in bounded chunks.

--build-only does not time. A second invocation --time-only reuses compiled
requests, validates both implementations against an independent bitwise ABI
oracle, then performs calibrated alternating DSL/C++ timing. Missing sidecar
contracts remain explicit in the full supplemental coverage manifest.
"""

import argparse
import faulthandler
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments, is_arm, packed_size, packing_geometry
from dsl_matmul_supplemental_catalog import ROOT, SELECTORS, manifest
from dsl_timing import isolate_compilation, reserve_cpu
from dsl_transpose import aligned_empty


def templates(row):
  if row.get('compensation_mode','PackOnly')!='PackOnly':
    return compensated_templates(row)
  s,k=row['spatial'],row['k']
  S,K=(s,k) if row['metadata']=='Const' else (T.dynamic(lower=0),T.dynamic(lower=0))
  P=T.dynamic(lower=0)
  side,memory,native,pipeline=row['side'],row['memory'],row['native'],row['pipeline']
  shape,strides=((S,K),(K,1)) if side=='A' else ((K,S),(1,K))
  format_=packing_geometry(native)[0]

  @T.template
  def dsl(A:T.Tensor(shape,strides,dtype=memory),O:T.Tensor((P,),dtype=native,access='output'),repeat:T.int64):
    for iteration in T.serial(repeat,unroll=1):
      value=T.load(A)
      if pipeline=='multiply4':
        value=value*4
      elif pipeline=='add1':
        value=value+T.cast(1,native)
      T.pack(value,O,side=side,format=format_,dtype=native)
      T.compiler_fence()

  Native=T.DType(getattr(vecops,native))
  NativeA=T.DType(getattr(vecops,row['native_a']))
  NativeB=T.DType(getattr(vecops,row['native_b']))
  SideB=T.Const(side=='B')
  Pipeline=T.Const({'identity':0,'multiply4':1,'add1':2}[pipeline])
  Implementation=T.Const(SELECTORS[row['selector']])

  @T.template(source=ROOT/'benchmarks/dsl_matmul_supplemental_pack_reference.cpp')
  def cpp(A:T.Tensor(shape,strides,dtype=memory),O:T.Tensor((P,),dtype=native,access='output'),repeat:T.int64):
    Native,NativeA,NativeB,SideB,Pipeline,Implementation  # noqa: B018

  return dsl,cpp


def compensated_templates(row):
  s,k=row['spatial'],row['k']
  S,K=(s,k) if row['metadata']=='Const' else (T.dynamic(lower=0),T.dynamic(lower=0))
  P=T.dynamic(lower=0)
  DS=T.dynamic(lower=0)
  memory=row['memory']
  transform=row['pipeline']=='multiply4'
  format_=packing_geometry('int8')[0]
  CompensationMode=T.Const(1 if row['compensation_mode']=='SeparateSum' else 2)
  Transform=T.Const(transform)
  LogicalN,LogicalK=T.Const(s),T.Const(k)

  @T.template
  def dsl(A:T.Tensor((K,S),(1,K),dtype=memory),O:T.Tensor((P,),dtype='int8',access='output'),
          D:T.Tensor((DS,),dtype='int32',access='output'),repeat:T.int64):
    assert D.shape[0]==A.shape[1], 'compensation sidecar length mismatch'
    for iteration in T.serial(repeat,unroll=1):
      value=T.load(A)
      if transform:
        value=value*4
      native=T.cast(value,'int8')
      if CompensationMode == 2:
        T.library.pack_b_compensated(native,O,D,zero_point=3,format=format_)
      else:
        T.pack(native,O,side='B',format=format_,dtype='int8')
        sums=T.reduce_add(T.bitcast(T.cast(native,'int32'),'uint32'),axis=0)
        correction=T.bitcast(T.cast(0,'uint32')-sums*T.cast(3,'uint32'),'int32')
        T.store(D,correction)
      T.compiler_fence()

  @T.template(source=ROOT/'benchmarks/dsl_matmul_supplemental_compensation_reference.cpp')
  def cpp(A:T.Tensor((K,S),(1,K),dtype=memory),O:T.Tensor((P,),dtype='int8',access='output'),
          D:T.Tensor((DS,),dtype='int32',access='output'),repeat:T.int64):
    CompensationMode,Transform,LogicalN,LogicalK  # noqa: B018
  return dsl,cpp


def compensation_oracle(row,logical):
  native=logical.astype(np.float32)*np.float32(4) if row['pipeline']=='multiply4' else logical
  native=native.astype(np.int8).astype(np.int32)
  sums=native.astype(np.uint32).sum(axis=1,dtype=np.uint32)
  return (np.uint32(0)-sums*np.uint32(row['zero_point'])).view(np.int32)


def packing_arguments(row):
  """Independent bounded initialization and physical-ABI oracle for one row."""
  s,k=row['spatial'],row['k']
  logical=aligned_empty((s,k),row['memory'])
  for begin in range(0,s,256):
    count=min(256,s-begin)
    indices=np.arange(begin*k,(begin+count)*k,dtype=np.int64)
    if row['category']=='compensation':
      values=(indices*5)%63-31
      if row['memory']=='float32':values=values.astype(np.float32)*np.float32(0.25)
    elif row['memory'] in {'int8','uint8'}:
      values=(indices*13+7)%127
    else:
      values=((indices%251)-125)*np.float32(0.015625)
    logical[begin:begin+count]=values.reshape(count,k)
  source=logical if row['side']=='A' else logical.T
  output=aligned_empty((packed_size(s,k,row['native']),),row['native'])
  expected=oracle(row,logical)
  if row.get('compensation_mode','PackOnly')!='PackOnly':
    return (source,output,aligned_empty((s,),'int32')),expected,compensation_oracle(row,logical)
  return (source,output),expected,None


def oracle(row, logical):
  native=np.dtype(row['native'])
  if row['pipeline']=='multiply4':
    logical=logical.astype(np.float32)*np.float32(4)
  elif row['pipeline']=='add1':
    logical=logical.astype(np.float32 if native.itemsize<4 else native)+1
  if native.kind in {'i','u'} and logical.dtype.kind=='f':
    limits=np.iinfo(native)
    logical=np.clip(np.nan_to_num(logical,nan=0,posinf=limits.max,neginf=limits.min),limits.min,limits.max)
  logical=logical.astype(native)
  _,panel,group=packing_geometry(row['native'])
  s,k=logical.shape
  if is_arm():
    result=np.zeros(((s+panel-1)//panel,(k+group-1)//group,panel,group),dtype=native)
    for start in range(0,s,panel):
      slab=logical[start:start+panel]
      padded=np.zeros((panel,((k+group-1)//group)*group),dtype=native)
      padded[:slab.shape[0],:k]=slab
      result[start//panel]=padded.reshape(panel,-1,group).transpose(1,0,2)
    return result.reshape(-1)
  # AMX's public physical order differs for A and B.
  from dsl_benchmark_utils import packing_geometry as geometry
  _,tile,chunk=geometry(row['native'])
  result=np.zeros(((s+tile-1)//tile,(k+chunk-1)//chunk,tile,chunk),dtype=native)
  for start in range(0,s,tile):
    for begin in range(0,k,chunk):
      result[start//tile,begin//chunk,:min(tile,s-start),:min(chunk,k-begin)]=logical[start:start+tile,begin:begin+chunk]
  if row['side']=='B':
    element_group=4//native.itemsize
    result=result.reshape(result.shape[:2]+(tile,chunk//element_group,element_group)).transpose(0,1,3,2,4)
  return result.reshape(-1)


def measure_pair(calls,cpu,rounds):
  repeats={}
  for label,call in calls.items():
    repeat=1
    while True:
      start=time.perf_counter_ns();call(repeat);elapsed=time.perf_counter_ns()-start
      if elapsed>=25_000_000 or repeat>=2**30:
        break
      repeat*=max(2,min(16,25_000_000//max(1,elapsed)))
    repeats[label]=repeat
  samples={label:[] for label in calls}
  raw=[]
  for round_ in range(rounds):
    for label in (('dsl','cpp') if round_%2==0 else ('cpp','dsl')):
      start=time.perf_counter_ns();calls[label](repeats[label]);elapsed=time.perf_counter_ns()-start
      samples[label].append(elapsed/repeats[label]);raw.append({'round':round_,'label':label,'elapsed_ns':elapsed,'repeat':repeats[label]})
  medians={label:float(np.median(values)) for label,values in samples.items()}
  return {'cpu':cpu,'rounds':rounds,'median_ns':medians,'dsl_over_cpp':medians['dsl']/medians['cpp'],'samples':raw}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--architecture',choices=('x86','arm'),required=True)
  parser.add_argument('--features',nargs='*',choices=('amx_fp16','sme_f64'),default=[])
  parser.add_argument('--category',choices=('plain','fused','compensation'),help='optional source-registration category, without dropping it from coverage')
  parser.add_argument('--compensation-only',action='store_true',help='only registrations that emit a compensation sidecar')
  parser.add_argument('--work-dir',type=Path,required=True)
  parser.add_argument('--chunk-size',type=int,default=16)
  parser.add_argument('--chunk-index',type=int,required=True)
  parser.add_argument('--all',action='store_true',help='one persistent frontend collects every equivalent registration before parallel native compilation')
  parser.add_argument('--build-only',action='store_true')
  parser.add_argument('--time-only',action='store_true')
  parser.add_argument('--validate-only',action='store_true',help='bitwise oracle validation without calibration or timing')
  parser.add_argument('--cpu',type=int,required=True)
  parser.add_argument('--jobs',type=int,default=2)
  parser.add_argument('--rounds',type=int,default=12)
  parser.add_argument('--case-timeout',type=int,default=1800,help='isolated frontend deadline including real header attestation; timed-out rows stay explicit')
  parser.add_argument('--batch-input-attestation',action='store_true',help='real SDK/toolchain pre/post guards with a private cache for this chunk')
  add_toolchain_arguments(parser)
  args=parser.parse_args()
  if sum((args.build_only,args.time_only,args.validate_only))!=1 or args.rounds<=0 or args.rounds%2 or args.chunk_size<=0 or args.case_timeout<=0:
    parser.error('choose one of --build-only/--time-only/--validate-only; rounds must be positive/even and chunk-size positive')
  coverage=manifest(args.architecture,args.features)
  equivalent=[row for row in coverage['registrations'] if row['suite']=='MatmulPack' and row['dsl_equivalent']]
  if args.category:
    equivalent=[row for row in equivalent if row['category']==args.category]
  if args.compensation_only:
    equivalent=[row for row in equivalent if row.get('compensation_mode','PackOnly')!='PackOnly']
  rows=equivalent if args.all else equivalent[args.chunk_index*args.chunk_size:(args.chunk_index+1)*args.chunk_size]
  if not rows:
    parser.error('chunk-index has no equivalent packing rows')
  work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=True)
  coverage_tmp=work/f'coverage-{os.getpid()}.json'
  coverage_tmp.write_text(json.dumps(coverage,indent=2)+'\n')
  coverage_tmp.replace(work/'coverage.json')
  path=work/('all' if args.all else f'chunk-{args.chunk_index:03d}');path.mkdir(exist_ok=True)
  isolate_compilation(args.cpu)
  compiler=vecops.Compiler(target=args.target,cc=args.cc,cxx=args.cxx,jobs=args.jobs,
    cache_dir=(path/'cache' if args.batch_input_attestation else work/'cache'),build_dir=path/'build')
  attestation=None
  if args.batch_input_attestation:
    from dsl_matmul_repair_suite import BatchInputAttestation
    attestation=BatchInputAttestation(path) if args.build_only else BatchInputAttestation.restore(path)
    if not args.build_only:
      attestation.guard('supplemental_pack_recovery_pre')
  cases=[]
  failures=[]
  kernels={}
  statuses={row['registration_id']:{'registration_id':row['registration_id'],'state':'pending'} for row in rows}
  def save_status():
    temporary=path/f'status-{os.getpid()}.json'
    temporary.write_text(json.dumps({'rows':list(statuses.values()),'native_jobs':args.jobs,'frontend_processes':1},indent=2)+'\n')
    temporary.replace(path/'row-status.json')
  save_status()
  def deadline(signum,frame):
    raise TimeoutError('frontend case deadline exceeded')
  signal.signal(signal.SIGALRM,deadline)
  with vecops.collect_compile_requests() as collector:
    for row in rows:
      print('BEGIN',row['registration_id'],row['spatial'],row['k'],row['side'],row['native'],row['metadata'],row['selector'],flush=True)
      started=time.monotonic()
      statuses[row['registration_id']]['state']='lowering'
      save_status()
      signal.alarm(args.case_timeout)
      faulthandler.dump_traceback_later(max(1,args.case_timeout-10))
      try:
        mode=row.get('compensation_mode','PackOnly')
        exact_shape=row['metadata']=='Const' or mode!='PackOnly'
        key=(row['native_a'],row['native_b'],row['native'],row['memory'],row['pipeline'],row['side'],
          row['selector'],row['metadata'],mode,(row['spatial'],row['k']) if exact_shape else None)
        if key not in kernels:
          pair_templates=templates(row)
          if attestation is None:
            kernels[key]=tuple(T.compile(kernel,compiler=compiler) for kernel in pair_templates)
          else:
            from dsl_matmul_artifact_ab import template_contract
            from dsl_matmul_repair_suite import compile_cpp_source, compile_prelowered
            from vecops.language._compiler.driver import _sme_grid_compiler, _strict_dsl_compiler
            selected=compiler.snapshot()
            proof_path=path/'prelower'/hashlib.sha256(repr(key).encode()).hexdigest()
            if args.build_only:
              lowered=T.lower(pair_templates[0],target=args.target,pass_options={'_vector_state':selected.vector_state})
              effective=_sme_grid_compiler(_strict_dsl_compiler(selected),lowered.artifact)
              proof_path.mkdir(parents=True,exist_ok=True)
              source=proof_path/'kernel.cpp';source.write_text(lowered.artifact.source)
              contract,_=template_contract(pair_templates[0])
              proof={'source':str(source),'raw_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'contract':contract,'effective_cflags':list(effective.cflags),
                'vector_state_token':selected.vector_state.env_token,'runtime_parallelism':True}
              (proof_path/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
            else:
              proof=json.loads((proof_path/'proof.json').read_text())
            kernels[key]=(compile_prelowered(pair_templates[0],proof,compiler,attestation=attestation),
              compile_cpp_source(pair_templates[1],compiler,attestation=attestation))
        pair=kernels[key]
        s,k=row['spatial'],row['k'];side=row['side'];size=packed_size(s,k,row['native'])
        metadata=(vecops.TensorMeta((s,k) if side=='A' else (k,s),strides=(k,1) if side=='A' else (1,k),dtype=getattr(vecops,row['memory'])),
                  vecops.TensorMeta((size,),dtype=getattr(vecops,row['native'])))
        if row.get('compensation_mode','PackOnly')!='PackOnly':
          metadata+= (vecops.TensorMeta((s,),dtype=vecops.int32),)
        for label,kernel in zip(('dsl','cpp'),pair):
          print('LOWER',row['registration_id'],label,flush=True)
          kernel.operator(*metadata,1,**kernel._core.specialization_values)
        cases.append((row,pair,size))
        statuses[row['registration_id']].update(state='collected',frontend_seconds=time.monotonic()-started)
        print('END',row['registration_id'],time.monotonic()-started,flush=True)
      except (TimeoutError, RuntimeError, ValueError, TypeError, AssertionError, SyntaxError, AttributeError) as error:
        failures.append({'registration_id':row['registration_id'],'row':row,'error':repr(error),'phase':'frontend'})
        statuses[row['registration_id']].update(state='frontend_failed',error=repr(error))
        print('FAILED',row['registration_id'],repr(error),flush=True)
      finally:
        signal.alarm(0)
        faulthandler.cancel_dump_traceback_later()
        save_status()
  audit={'registrations':[row['registration_id'] for row in rows],'runtime_root':str(ROOT),
    'failures':failures,'source_hashes':coverage['source_hashes'],'artifacts':[{'registration_id':row['registration_id'],
      'dsl_source':str(pair[0].source),'cpp_source':str(pair[1].source),
      'dsl_sha256':hashlib.sha256(Path(pair[0].source).read_bytes()).hexdigest(),
      'cpp_sha256':hashlib.sha256(Path(pair[1].source).read_bytes()).hexdigest()} for row,pair,_ in cases]}
  (path/'manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
  if args.build_only:
    print(f'{len(rows)} registrations, {len(collector.requests)} collected requests',flush=True)
    if attestation is not None:
      attestation.guard('supplemental_pack_pre_build')
    try:
      vecops.compile_batch(collector.requests,parallelism=args.jobs,execution_parallelism=1)
    finally:
      if attestation is not None:
        attestation.guard('supplemental_pack_post_build')
    for row,_,_ in cases:
      statuses[row['registration_id']]['state']='built'
    save_status()
    (path/'build-complete.json').write_text(json.dumps({'registrations':[row['registration_id'] for row,_,_ in cases],'completed':not failures,'failures':failures,
      'batch_input_attestation':args.batch_input_attestation,
      'checkpoint_sha256':hashlib.sha256((path/'batch-input-attestation.json').read_bytes()).hexdigest() if attestation is not None else None})+'\n')
    return
  if not (path/'build-complete.json').exists():
    raise RuntimeError('time-only requires a completed build-only chunk')
  lease=reserve_cpu(args.cpu)
  os.sched_setaffinity(0,{args.cpu})
  results=[]
  try:
    for row,pair,size in cases:
      s,k=row['spatial'],row['k']
      logical=aligned_empty((s,k),row['memory'])
      # Bounded row-wise initialization avoids full-size int64/random temporaries.
      for begin in range(0,s,256):
        count=min(256,s-begin)
        indices=np.arange(begin*k,(begin+count)*k,dtype=np.int64)
        if row['category']=='compensation':
          values=(indices*5)%63-31
          if row['memory']=='float32':
            values=values.astype(np.float32)*np.float32(0.25)
        elif row['memory'] in {'int8','uint8'}:
          values=(indices*13+7)%127
        else:
          values=((indices%251)-125)*np.float32(0.015625)
        logical[begin:begin+count]=values.reshape(count,k)
      source=logical if row['side']=='A' else logical.T
      output=aligned_empty((size,),row['native'])
      expected=oracle(row,logical)
      compensated=row.get('compensation_mode','PackOnly')!='PackOnly'
      correction=aligned_empty((s,),'int32') if compensated else None
      values=(source,output,correction) if compensated else (source,output)
      calls={label:lambda repeat,kernel=kernel,values=values:kernel.operator.native(*values,repeat,**kernel._core.specialization_values)
             for label,kernel in zip(('dsl','cpp'),pair)}
      for call in calls.values():
        if compensated:
          correction.fill(0x12345678)
        output.fill(19);call(1)
        np.testing.assert_array_equal(output.view(np.uint8),expected.view(np.uint8))
        if compensated:
          np.testing.assert_array_equal(correction,compensation_oracle(row,logical))
      result={'registration':row,'validated':True,'timed':not args.validate_only}
      if not args.validate_only:
        result.update(measure_pair(calls,args.cpu,args.rounds))
      results.append(result)
      result_path=path/('pending-results.json' if attestation is not None else 'results.json')
      result_path.write_text(json.dumps({'rows':results,'failures':failures,'completed':len(results)==len(cases) and not failures},indent=2)+'\n')
      print(row['registration_id'],result.get('dsl_over_cpp','oracle passed'),flush=True)
    if attestation is not None:
      attestation.guard('supplemental_pack_recovery_post')
      (path/'pending-results.json').replace(path/'results.json')
  finally:
    lease.close()


if __name__=='__main__':
  main()
