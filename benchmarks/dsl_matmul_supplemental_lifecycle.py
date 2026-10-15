# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Paired WeightReuse and Stateful Legacy native lifecycle benchmarks.

One native repeat is one B pack followed by R batched calls for online-B
WeightReuse, or R raw calls. The C++ bound invocation/session and the DSL
workspace are each created before the native repeat loop. Prepared is excluded:
the public DSL has no counterpart to its persistent prepared operation plan.
Build-only collects without allocating benchmark payloads or timing.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments, packing_geometry
from dsl_matmul_catalog import CatalogCase
from dsl_matmul_fusion_catalog import DTYPES, INPUTS, OUTPUTS
from dsl_matmul_source_catalog import SourceScenario, source_arguments, source_metadata
from dsl_matmul_supplemental_catalog import ROOT, supplemental_cases
from dsl_matmul_supplemental_pack import measure_pair
from dsl_timing import isolate_compilation, reserve_cpu
from dsl_transpose import aligned_empty


class LifecycleScenario(SourceScenario):
  """Keep supplemental binding semantics independent of catalog adapter version."""

  @property
  def reference_bound_invocation(self):
    return self.case.catalog!='StatefulMatmul'

  @property
  def reference_compensated_pack(self):
    return (self.reference_bound_invocation and self.online and self.pack_b
            and self.case.arm_condition.startswith('x86 source')
            and self.case.input_pipeline_a=='AsymmetricQuantize4Zp3')


def lifecycle_arguments(scenario):
  values,expected=source_arguments(scenario)
  values=list(values)
  # Scenario's prior is a mutable vector with a zero batch stride. The common
  # argument helper intentionally returns a read-only broadcast view; rebuild
  # just this adapter view on independently owned writable backing storage.
  if scenario.shared_prior:
    original=values[3]
    backing=aligned_empty(original.shape[-2:],original.dtype)
    backing[...]=original[0]
    values[3]=np.lib.stride_tricks.as_strided(backing,shape=original.shape,
      strides=(0,*backing.strides),writeable=True)
  return tuple(values),expected


def scenario_for(row):
  if row['suite']=='StatefulMatmul' and row['name']=='Prepared':
    raise ValueError('Prepared operation plan has no equivalent public DSL API')
  inverse={value:key for key,value in DTYPES.items()}
  stateful=row['suite']=='StatefulMatmul'
  na,nb=('bfloat16','bfloat16') if stateful else (row['native_a'],row['native_b'])
  acc='int32' if na in {'int8','uint8'} else ('float64' if na=='float64' else 'float32')
  packing='raw' if stateful or row['strategy']=='raw' else 'online_packedb'
  m,n,k=row['shape']
  case=CatalogCase(catalog='MatmulScenarioExpanded' if not stateful else 'StatefulMatmul',
    source=row['source'],line=0,name=row['name'],m=m,n=n,k=k,
    native_a=inverse[na],native_b=inverse[nb],accumulator=inverse[acc],
    memory_a=inverse[row['memory_a']],memory_b=inverse[row['memory_b']],memory_c=inverse[row['memory_c']],
    memory_c_input=inverse[row.get('memory_c_input',row['memory_c'])],
    input_pipeline_a=row.get('input_pipeline_a','Convert'),input_pipeline_b=row.get('input_pipeline_b','Convert'),
    output_pipeline=row.get('output_pipeline','Convert'),input_modes=(packing,),metadata=(row['metadata'],),
    batch=None if stateful else str(row['batch']),broadcast=None if stateful else 'B',
    arm_condition='x86 source' if row['architecture']=='x86' else 'ARM source',dsl_supported=True)
  return LifecycleScenario(case,row['metadata'],'matmul','auto',packing)


def templates(row):
  scenario=scenario_for(row)
  case=scenario.case
  M,N,K=(case.m,case.n,case.k) if scenario.mode=='Const' else tuple(T.Dynamic(name=name,lower=1) for name in ('M','N','K'))
  Batch=scenario.batch if scenario.mode=='Const' else T.Dynamic(name='Batch',lower=1)
  PA=T.Dynamic(name='PackedCapacityA',lower=1)
  PB=T.Dynamic(name='PackedCapacityB',lower=1)
  ad,bd,cd,dd,na,nb,acc=scenario.types
  batched=scenario.batched
  DM=1 if scenario.broadcast_prior else M
  AT=T.Tensor((Batch,M,K),(M*K,K,1),ad) if batched else T.Tensor((M,K),(K,1),ad)
  BT=T.Tensor((Batch,K,N),(0,1,K),bd) if batched else T.Tensor((K,N),(1,K),bd)
  CT=T.Tensor((Batch,M,N),(M*N,N,1),cd,access='output') if batched else T.Tensor((M,N),(N,1),cd,access='output')
  DT=T.Tensor((Batch,DM,N),(0 if scenario.shared_prior else DM*N,N,1),dd,access='inout') if batched else T.Tensor((DM,N),dtype=dd,access='inout')
  RT=T.Tensor((Batch,M,1),dtype='float32') if batched else T.Tensor((M,1),dtype='float32')
  APType=T.Tensor((PA,),dtype=na,access='inout')
  BPType=T.Tensor((PB,),dtype=nb,access='inout')
  ia,ib,out=case.input_pipeline_a,case.input_pipeline_b,case.output_pipeline
  quant_a,quant_b=ia in INPUTS[1:4],ib in INPUTS[1:4]
  zp=3 if ia!='Quantize4' and na=='uint8' else 0
  packed=scenario.pack_b
  correction=scenario.reference_compensated_pack
  format_=packing_geometry(nb)[0]
  calls=row.get('calls_per_pack',1)
  prior=scenario.has_prior

  @T.func
  def input_a(A):
    if quant_a:
      return T.cast(T.load(A,dtype='float32')*4+zp,na)
    return T.load(A,dtype=na)

  @T.func
  def input_b(B):
    if quant_b:
      return T.cast(T.load(B,dtype='float32')*4,nb)
    return T.load(B,dtype=nb)

  @T.template
  def dsl(A:AT,B:BT,C:CT,D:DT,S:T.Tensor((1,1),dtype=acc),SC:T.Tensor((1,N),dtype=acc),
          SQ:T.Tensor((1,N),dtype='float32'),RQ:RT,RD:RT,W:T.Tensor((1,N),dtype='float32'),
          Z:T.Tensor((1,1),dtype='int32'),AP:APType,BP:BPType,repeat:T.int64):
    workspace=T.workspace(C.shape,dtype=acc)
    for iteration in T.serial(repeat,unroll=1):
      raw_b=B[0] if batched else B
      if packed:
        if correction:
          sidecar=T.alloc((raw_b.shape[1],),dtype='int32')
          T.library.pack_b_compensated(input_b(raw_b),BP,sidecar,zero_point=3,format=format_)
          destination=D[0] if batched else D
          T.store(destination,T.broadcast_to(T.load(sidecar),destination.shape))
        else:
          T.pack(input_b(raw_b),BP,side='B',format=format_,dtype=nb)
      for call in T.serial(calls,unroll=1):
        a=input_a(A)
        b=T.packed(BP,(raw_b.shape[0],raw_b.shape[1]),side='B',format=format_) if packed else input_b(raw_b)
        initial=T.load(D,dtype=acc) if prior else None
        if batched:
          result=T.library.batched_matmul((a,b),C,acc=initial,acc_dtype=acc,strategy='rows',workspace=workspace)
        else:
          result=T.library.matmul((a,b),C,acc=initial,acc_dtype=acc,workspace=workspace)
        if out=='BiasRelu' or out=='Relu':
          result=T.max(result,0)
        elif out=='Dequantize':
          result=T.cast(result,'float32')*0.125
        elif out=='AsymmetricDequantize':
          result=T.cast(result,'float32')*0.0625
        T.store(C,result)
        T.compiler_fence()

  NativeA,NativeB=T.DType(getattr(vecops,na)),T.DType(getattr(vecops,nb))
  InputPipelineA,InputPipelineB=T.Const(INPUTS.index(ia)),T.Const(INPUTS.index(ib))
  OutputPipeline=T.Const(OUTPUTS.index(out))
  HasPrior,BroadcastPrior,SwapAB=T.Const(prior),T.Const(scenario.broadcast_prior),T.Const(True)
  PackA,PackB,OnlinePack,OnlyPack,SharedB=map(T.Const,(False,packed,packed,False,batched))
  ReferenceFamily,ReferencePolicy=T.Const(0),T.Const(0)
  ReferenceBoundInvocation=T.Const(row['suite']!='StatefulMatmul')
  ReferenceCompensatedPack=T.Const(correction)
  CallsPerPack=T.Const(calls)

  @T.template(source=ROOT/'benchmarks/dsl_matmul_supplemental_lifecycle_reference.cpp')
  def cpp(A:AT,B:BT,C:CT,D:DT,S:T.Tensor((1,1),dtype=acc),SC:T.Tensor((1,N),dtype=acc),
          SQ:T.Tensor((1,N),dtype='float32'),RQ:RT,RD:RT,W:T.Tensor((1,N),dtype='float32'),
          Z:T.Tensor((1,1),dtype='int32'),AP:APType,BP:BPType,repeat:T.int64):
    NativeA,NativeB,InputPipelineA,InputPipelineB,OutputPipeline,HasPrior,BroadcastPrior,SwapAB  # noqa: B018
    PackA,PackB,OnlinePack,OnlyPack,SharedB,ReferenceFamily,ReferencePolicy,ReferenceBoundInvocation  # noqa: B018
    ReferenceCompensatedPack,CallsPerPack  # noqa: B018
  return scenario,(dsl,cpp)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--architecture',choices=('x86','arm'),required=True)
  parser.add_argument('--features',nargs='*',choices=('amx_fp16','sme_f64'),default=[])
  parser.add_argument('--suite',choices=('MatmulWeightReuse','StatefulMatmul'),required=True)
  parser.add_argument('--work-dir',type=Path,required=True)
  parser.add_argument('--chunk-index',type=int,default=0)
  parser.add_argument('--chunk-size',type=int,default=16)
  parser.add_argument('--all',action='store_true')
  parser.add_argument('--build-only',action='store_true')
  parser.add_argument('--time-only',action='store_true')
  parser.add_argument('--validate-only',action='store_true')
  parser.add_argument('--cpu',type=int,required=True)
  parser.add_argument('--jobs',type=int,default=8)
  parser.add_argument('--rounds',type=int,default=12)
  add_toolchain_arguments(parser)
  args=parser.parse_args()
  if sum((args.build_only,args.time_only,args.validate_only))!=1 or args.rounds<=0 or args.rounds%2 or args.chunk_size<=0:
    parser.error('choose build-only, time-only or validate-only; positive even rounds and positive chunk-size required')
  rows=[row for row in supplemental_cases(args.architecture,args.features) if row['suite']==args.suite and row.get('name')!='Prepared']
  if not args.all:
    rows=rows[args.chunk_index*args.chunk_size:(args.chunk_index+1)*args.chunk_size]
  if not rows:
    parser.error('no matching registrations')
  work=args.work_dir.resolve()/('all' if args.all else f'chunk-{args.chunk_index:03d}')
  work.mkdir(parents=True,exist_ok=True)
  isolate_compilation(args.cpu)
  compiler=vecops.Compiler(cc=args.cc,cxx=args.cxx,target=args.target,jobs=args.jobs,cache_dir=work.parent/'cache',build_dir=work/'build')
  cases=[]
  failures=[]
  kernels={}
  statuses={row['registration_id']:{'registration_id':row['registration_id'],'state':'pending'} for row in rows}
  def save_status():
    temporary=work/f'status-{os.getpid()}.json'
    temporary.write_text(json.dumps({'rows':list(statuses.values()),'frontend_processes':1,'native_jobs':args.jobs},indent=2)+'\n')
    temporary.replace(work/'row-status.json')
  save_status()
  with vecops.collect_compile_requests() as collector:
    for row in rows:
      print('BEGIN',row['registration_id'],row['suite'],row['shape'],row.get('calls_per_pack',1),flush=True)
      statuses[row['registration_id']]['state']='lowering'
      save_status()
      try:
        scenario=scenario_for(row)
        key=(scenario.compilation_key,row.get('calls_per_pack',1),row['suite'],scenario.reference_compensated_pack)
        if key not in kernels:
          _,pair=templates(row)
          kernels[key]=tuple(T.compile(kernel,compiler=compiler) for kernel in pair)
        compiled=kernels[key]
        meta=source_metadata(scenario)
        for kernel in compiled:
          kernel.operator(*meta,1,**kernel._core.specialization_values)
        cases.append((row,scenario,compiled))
        statuses[row['registration_id']]['state']='collected'
        print('END',row['registration_id'],flush=True)
      except (RuntimeError,ValueError,TypeError,AssertionError,SyntaxError,AttributeError) as error:
        failures.append({'registration_id':row['registration_id'],'phase':'frontend','error':repr(error)})
        statuses[row['registration_id']].update(state='frontend_failed',error=repr(error))
        print('FAILED',row['registration_id'],repr(error),flush=True)
      save_status()
  audit={'registrations':rows,'dsl_pass_options':{},'dsl_workspace':'explicit T.workspace result storage declared once outside repeat/call loops',
    'cpp_workspace':'ExecutionSession and storage created once outside native repeat loop',
    'cpp_binding':'WeightReuse bound invocation outside repeat; Stateful Legacy operation binding on each call',
    'warmup':'CPP and DSL oracle invocations precede measure_pair; no additional warmup is inside the timed native kernel',
    'setup_scope':'workspace/session setup occurs once per native invocation and is amortized by repeat; original GoogleBenchmark keeps setup outside its State loop',
    'dsl_public_helper_output':'public matmul/batched_matmul with explicit workspace writes C before the harness final epilogue/store; this helper behavior is retained',
    'timed_unit':'one optional B pack followed by calls_per_pack matmul invocations',
    'failures':failures,'artifacts':[{'registration_id':row['registration_id'],
    'dsl_source':str(pair[0].source),'cpp_source':str(pair[1].source),
    'dsl_sha256':hashlib.sha256(Path(pair[0].source).read_bytes()).hexdigest(),
    'cpp_sha256':hashlib.sha256(Path(pair[1].source).read_bytes()).hexdigest()} for row,_,pair in cases]}
  (work/'manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
  if args.build_only:
    vecops.compile_batch(collector.requests,parallelism=args.jobs,execution_parallelism=1)
    for row,_,_ in cases:
      statuses[row['registration_id']]['state']='built'
    save_status()
    (work/'build-complete.json').write_text(json.dumps({'completed':not failures,'failures':failures,'registrations':[row['registration_id'] for row,_,_ in cases]})+'\n')
    return
  if not (work/'build-complete.json').exists():
    raise RuntimeError('time-only requires completed build-only')
  lease=reserve_cpu(args.cpu)
  os.sched_setaffinity(0,{args.cpu})
  results=[]
  try:
    for row,scenario,pair in cases:
      values,expected=lifecycle_arguments(scenario)
      if row['suite']=='StatefulMatmul':
        values[0].fill(1);values[1].fill(1);expected.fill(row['shape'][2])
      calls={label:lambda repeat,kernel=kernel,values=values:kernel.operator.native(*values,repeat,**kernel._core.specialization_values)
             for label,kernel in zip(('dsl','cpp'),pair)}
      tolerance=0.02 if values[2].dtype.itemsize<=2 else (8e-4 if values[2].dtype.itemsize==4 else 2e-10)
      calls['cpp'](1)
      reference=values[2].copy()
      if reference.dtype.kind in 'iu':
        np.testing.assert_array_equal(reference,expected)
      else:
        np.testing.assert_allclose(reference,expected,rtol=tolerance,atol=tolerance)
      calls['dsl'](1)
      if reference.dtype.kind in 'iu':
        np.testing.assert_array_equal(values[2],reference)
      else:
        np.testing.assert_allclose(values[2],reference,rtol=tolerance,atol=tolerance)
      result={'registration':row,'validated':True,'timed':not args.validate_only}
      if not args.validate_only:
        result.update(measure_pair(calls,args.cpu,args.rounds))
      results.append(result)
      (work/'results.json').write_text(json.dumps({'rows':results,'completed':len(results)==len(cases)},indent=2)+'\n')
  finally:
    lease.close()


if __name__=='__main__':
  main()
