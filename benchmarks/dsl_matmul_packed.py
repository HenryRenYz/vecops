"""Isolate prepacked leaf codegen from online packing/cache policy differences."""
import argparse
import csv
import os
from pathlib import Path
import numpy as np
import torch
import vecops
import vecops.language as T
from dsl_matmul import PIPELINES,ROOT
from dsl_transpose import aligned_empty,measure,numpy_dtype
from dsl_timing import isolate_compilation,reserve_cpu


def kernels(compiler,m,n,k,mode,kind,pipeline,level,packing):
  M,N,K=(m,n,k) if mode=='Const' else (T.dynamic(lower=0) for _ in range(3))
  PSA,PSB=(T.dynamic(lower=0) for _ in range(2))
  ad,bd,na,nb,acc={'bf16':('bfloat16','bfloat16','bfloat16','bfloat16','float32'),
                 'convert':('float32','float32','bfloat16','bfloat16','float32'),
                 'i8':('int8','int8','int8','int8','int32'),
                 'mixed':('uint8','int8','uint8','int8','int32')}[kind]
  prologue,epilogue,input_scale=PIPELINES[pipeline]
  out=acc if not epilogue else 'float32'
  DM=1 if prologue==2 else M
  pa,pb='A' in packing,'B' in packing
  @T.template
  def pack(A:T.Tensor((M,K),(K,1),ad),B:T.Tensor((K,N),(1,K),bd),
           AP:T.Tensor((PSA,),dtype=na,access='output'),BP:T.Tensor((PSB,),dtype=nb,access='output')):
    a=T.load(A,dtype='float32')*2+1 if input_scale==2 else T.load(A)
    b=T.load(B,dtype='float32')*2+1 if input_scale==2 else T.load(B)
    T.pack(a,AP,side='A',format='amx',dtype=na)
    T.pack(b,BP,side='B',format='amx',dtype=nb)
  @T.template
  def dsl(A:T.Tensor((M,K),(K,1),ad),B:T.Tensor((K,N),(1,K),bd),
          AP:T.Tensor((PSA,),dtype=na),BP:T.Tensor((PSB,),dtype=nb),
          C:T.Tensor((M,N),dtype=out,access='output'),D:T.Tensor((DM,N),dtype=acc),repeat:int):
    for iteration in T.serial(repeat,unroll=1):
      if pa:
        a=T.packed(AP,(M,K),side='A',format='amx')
      else:
        a=T.cast(T.load(A,dtype='float32')*2+1,na) if input_scale==2 else T.load(A,dtype=na)
      if pb:
        b=T.packed(BP,(K,N),side='B',format='amx')
      else:
        b=T.cast(T.load(B,dtype='float32')*2+1,nb) if input_scale==2 else T.load(B,dtype=nb)
      if level=='dot':
        value=T.dot((a,b),C,acc=T.load(D) if prologue else None)
      else:
        value=T.library.matmul((a,b),C,acc=T.load(D) if prologue else None)
      if epilogue:
        value=T.cast(value,'float32')
        if epilogue==1:value=T.max(value,0)
        elif epilogue==2:value=T.min(T.max(value,-0.25),0.25)
        elif epilogue==3:value=value*0.5
        else:value=value/(1+T.exp(-value))
        T.store(C,value)
      T.compiler_fence()
  NativeA,NativeB=T.DType(getattr(vecops,na)),T.DType(getattr(vecops,nb))
  PackA,PackB=T.Const(pa),T.Const(pb)
  Prologue,Epilogue,InputScale=T.Const(prologue),T.Const(epilogue),T.Const(input_scale)
  @T.template(source=ROOT/'benchmarks/dsl_matmul_packed_reference.cpp')
  def cpp(A:T.Tensor((M,K),(K,1),ad),B:T.Tensor((K,N),(1,K),bd),
          AP:T.Tensor((PSA,),dtype=na),BP:T.Tensor((PSB,),dtype=nb),
          C:T.Tensor((M,N),dtype=out,access='output'),D:T.Tensor((DM,N),dtype=acc),repeat:int):
    NativeA,NativeB,PackA,PackB,Prologue,Epilogue,InputScale
  return (T.compile(pack,compiler=compiler),T.compile(dsl,compiler=compiler),T.compile(cpp,compiler=compiler)),(ad,bd,na,nb,acc,out,prologue)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--build-dir',type=Path,default=ROOT/'build-dsl-matmul-packed')
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--shape',nargs='+',default=['19x23x257','256x256x256'])
  parser.add_argument('--kind',nargs='+',default=['bf16'],choices=['bf16','convert','i8','mixed'])
  parser.add_argument('--pipeline',nargs='+',default=['plain','bias_relu'],choices=tuple(PIPELINES))
  parser.add_argument('--mode',nargs='+',default=['Const','Dynamic'])
  parser.add_argument('--level',nargs='+',default=['dot','matmul'])
  parser.add_argument('--packing',nargs='+',default=['AB'],choices=['A','B','AB'])
  parser.add_argument('--jobs',type=int,default=4)
  parser.add_argument('--rounds',type=int,default=9)
  parser.add_argument('--cpu',type=int,default=2)
  args=parser.parse_args()
  isolate_compilation(args.cpu)
  args.build_dir.mkdir(parents=True,exist_ok=True)
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  compiler=vecops.Compiler(target='Native',cc='gcc',cxx='g++',jobs=args.jobs,
    cache_dir=args.build_dir/'cache',build_dir=args.build_dir/'build',include_dirs=[ROOT/'benchmarks'])
  cases=[]
  with vecops.collect_compile_requests() as collector:
    for shape in args.shape:
      m,n,k=map(int,shape.split('x'))
      for kind in args.kind:
        for pipeline in args.pipeline:
          for mode in args.mode:
            for level in args.level:
              for packing in args.packing:
                pair,types=kernels(compiler,m,n,k,mode,kind,pipeline,level,packing)
                ad,bd,na,nb,acc,out,prologue=types
                kr=32 if na=='bfloat16' else 64
                psa=((m+15)//16)*((k+kr-1)//kr)*16*kr
                psb=((n+15)//16)*((k+kr-1)//kr)*16*kr
                meta=(vecops.TensorMeta((m,k),dtype=getattr(vecops,ad)),
                      vecops.TensorMeta((k,n),strides=(1,k),dtype=getattr(vecops,bd)),
                      vecops.TensorMeta((psa,),dtype=getattr(vecops,na)),
                      vecops.TensorMeta((psb,),dtype=getattr(vecops,nb)),
                      vecops.TensorMeta((m,n),dtype=getattr(vecops,out)),
                      vecops.TensorMeta((1 if prologue==2 else m,n),dtype=getattr(vecops,acc)))
                pair[0].operator(*meta[:4],**pair[0]._core.specialization_values)
                for kernel in pair[1:]:
                  kernel.operator(*meta,1,**kernel._core.specialization_values)
                cases.append((m,n,k,kind,pipeline,mode,level,packing,pair,types))
  print(f'{len(cases)} cases, {len(collector)} artifacts',flush=True)
  vecops.compile_batch(collector.requests,parallelism=args.jobs,execution_parallelism=1)
  timing_lease=reserve_cpu(args.cpu)
  os.sched_setaffinity(0,{args.cpu})
  rng=np.random.default_rng(19)
  with args.output.open('w') as stream:
    writer=csv.writer(stream)
    writer.writerow(('m','n','k','dtype','pipeline','meta','level','packing','dsl_us','cpp_us','dsl_over_cpp'))
    for m,n,k,kind,pipeline,mode,level,packing,pair,types in cases:
      ad,bd,na,nb,acc,out,prologue=types
      a,b=aligned_empty((m,k),ad),aligned_empty((n,k),bd).T
      a[:]=rng.integers(0 if ad=='uint8' else -3,4,size=a.shape)/(8 if 'float' in ad else 1)
      b[:]=rng.integers(-3,4,size=b.shape)/(8 if 'float' in bd else 1)
      kr=32 if na=='bfloat16' else 64
      ap=aligned_empty((((m+15)//16)*((k+kr-1)//kr)*16*kr,),na)
      bp=aligned_empty((((n+15)//16)*((k+kr-1)//kr)*16*kr,),nb)
      c=aligned_empty((m,n),out)
      d=aligned_empty((1 if prologue==2 else m,n),acc)
      d[:]=rng.integers(-3,4,size=d.shape)
      pack,ds,cp=pair
      pack.operator.native(a,b,ap,bp,**pack._core.specialization_values)
      calls=lambda repeat:tuple(lambda kernel=kernel:kernel.operator.native(a,b,ap,bp,c,d,repeat,**kernel._core.specialization_values) for kernel in (ds,cp))
      call_d,call_c=calls(1)
      call_c();reference=c.copy();call_d()
      np.testing.assert_allclose(c,reference,rtol=5e-3,atol=3e-4)
      x,y=a.astype(np.float32),b.astype(np.float32)
      if pipeline=='affine_input':x,y=x*2+1,y*2+1
      expected=x.astype(numpy_dtype(na)).astype(numpy_dtype(acc))@y.astype(numpy_dtype(nb)).astype(numpy_dtype(acc))
      if prologue:expected+=d
      epilogue=PIPELINES[pipeline][1]
      if epilogue:expected=expected.astype(np.float32)
      if epilogue==1:expected=np.maximum(expected,0)
      if epilogue==2:expected=np.clip(expected,-0.25,0.25)
      if epilogue==3:expected*=0.5
      if epilogue==4:expected=expected/(1+np.exp(-expected))
      np.testing.assert_allclose(c,expected,rtol=5e-3,atol=3e-4)
      repeat=max(2,min(4000,20_000_000//(m*n*k)))
      td,tc=measure(calls(repeat),repeat,args.rounds)
      writer.writerow((m,n,k,kind,pipeline,mode,level,packing,f'{td:.4f}',f'{tc:.4f}',f'{td/tc:.4f}'));stream.flush()
      print(f'{mode} {m}x{n}x{k} {kind}/{pipeline}/{level}/packed{packing}: {td/tc:.3f}x',flush=True)
  timing_lease.close()


if __name__=='__main__':main()
