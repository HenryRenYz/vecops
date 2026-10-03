"""Same-round MatmulBench/MatmulFusionBench shapes and transform pipelines.

No JIT in timings. Native inner loops, identical metadata and buffers,
randomized alternating order; pin timing only. C++ uses the current public
operator including its automatically selected families, not a slowed baseline.
"""
import argparse
import ast
import csv
import os
import random
import re
import statistics
from pathlib import Path
import numpy as np
import torch
import vecops
import vecops.language as T
from dsl_transpose import aligned_empty,measure,numpy_dtype
from dsl_timing import isolate_compilation,reserve_cpu

ROOT=Path(__file__).resolve().parents[1]


def catalog(integer=False):
  values={"tile":16,"k_step":64 if integer else 32}
  def evaluate(source):
    tree=ast.parse(source.strip(),mode="eval").body
    def visit(node):
      if isinstance(node,ast.Constant) and isinstance(node.value,int):
        return node.value
      if isinstance(node,ast.Name):
        return values[node.id]
      if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult)):
        a,b=visit(node.left),visit(node.right)
        return a+b if isinstance(node.op,ast.Add) else a-b if isinstance(node.op,ast.Sub) else a*b
      raise ValueError("nonconstant benchmark extent")
    return visit(tree)
  cases=[]
  for filename in ("MatmulBenchCommon.h","MatmulScenarioBenchCommon.h"):
    text=(ROOT/"benchmarks/ops"/filename).read_text()
    for group,name,m,n,k in re.findall(r'\{"([\w]+)",\s*"([\w]+)",\s*([^,]+),\s*([^,]+),\s*([^,]+),',text):
      try:
        shape=tuple(evaluate(x) for x in (m,n,k))
      except (KeyError,ValueError,SyntaxError):
        continue
      if all(x>0 for x in shape):
        cases.append((group,name,*shape))
  # Distinct catalog labels can share a shape. Compile/test/timing once per
  # shape while retaining the first exact catalog label in the CSV.
  return tuple({(m,n,k):(group,name,m,n,k) for group,name,m,n,k in reversed(cases)}.values())[::-1]


PIPELINES={"plain":(0,0,1),"accumulate":(1,0,1),"bias_relu":(2,1,1),
           "bias_clamp":(2,2,1),"scale":(0,3,1),"silu":(0,4,1),"affine_input":(0,0,2)}


def kernels(compiler,m,n,k,mode,kind,pipeline,level,pass_options=None):
  M,N,K=(m,n,k) if mode=="Const" else (T.dynamic(lower=0) for _ in range(3))
  ad,bd,na,nb,acc={"bf16":("bfloat16","bfloat16","bfloat16","bfloat16","float32"),
                   "convert":("float32","float32","bfloat16","bfloat16","float32"),
                   "i8":("int8","int8","int8","int8","int32"),
                   "mixed":("uint8","int8","uint8","int8","int32")}[kind]
  prologue,epilogue,input_scale=PIPELINES[pipeline]
  out=acc if epilogue==0 else "float32"
  DM=1 if prologue==2 else M
  @T.template
  def dsl(A:T.Tensor((M,K),(K,1),ad),B:T.Tensor((K,N),(1,K),bd),
          C:T.Tensor((M,N),(N,1),out,access="output"),D:T.Tensor((DM,N),(N,1),acc),repeat:int):
    for iteration in T.serial(repeat,unroll=1):
      if input_scale==2:
        a=T.cast(T.load(A,dtype="float32")*2+1,na)
        b=T.cast(T.load(B,dtype="float32")*2+1,nb)
      else:
        a=T.load(A,dtype=na)
        b=T.load(B,dtype=nb)
      prior=T.load(D) if prologue!=0 else None
      if level=="dot":
        value=T.dot((a,b),C,acc=prior)
      else:
        value=T.library.matmul((a,b),C,acc=prior)
      if epilogue!=0:
        value=T.cast(value,"float32")
        if epilogue==1:
          value=T.max(value,0)
        elif epilogue==2:
          value=T.min(T.max(value,-0.25),0.25)
        elif epilogue==3:
          value=value*0.5
        else:
          value=value/(1+T.exp(-value))
        T.store(C,value)
      T.compiler_fence()
  NativeA,NativeB=T.DType(getattr(vecops,na)),T.DType(getattr(vecops,nb))
  Prologue,Epilogue,InputScale=T.Const(prologue),T.Const(epilogue),T.Const(input_scale)
  @T.template(source=ROOT/"benchmarks/dsl_matmul_reference.cpp")
  def cpp(A:T.Tensor((M,K),(K,1),ad),B:T.Tensor((K,N),(1,K),bd),
          C:T.Tensor((M,N),(N,1),out,access="output"),D:T.Tensor((DM,N),(N,1),acc),repeat:int):
    NativeA,NativeB,Prologue,Epilogue,InputScale
  return (T.compile(dsl,compiler=compiler,pass_options=pass_options),T.compile(cpp,compiler=compiler)),(ad,bd,na,nb,acc,out,prologue)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--build-dir",type=Path,default=ROOT/"build-dsl-matmul")
  parser.add_argument("--output",type=Path)
  parser.add_argument("--shape",nargs="+")
  parser.add_argument("--kind",nargs="+",default=["bf16"],choices=["bf16","convert","i8","mixed"])
  parser.add_argument("--pipeline",nargs="+",default=["plain"],choices=tuple(PIPELINES))
  parser.add_argument("--mode",nargs="+",default=["Const","Dynamic"])
  parser.add_argument("--level",nargs="+",default=["dot","matmul"])
  parser.add_argument("--rounds",type=int,default=9)
  parser.add_argument("--jobs",type=int,default=4)
  parser.add_argument("--cpu",type=int,default=2)
  parser.add_argument("--max-flops",type=int,default=300_000_000,help="Skip huge cases; set 0 for the full catalog")
  parser.add_argument("--fusion",choices=["auto","always","never"],default="auto")
  parser.add_argument("--gemv-columns",type=int,default=4,choices=range(1,9))
  args=parser.parse_args()
  isolate_compilation(args.cpu)
  args.build_dir.mkdir(parents=True,exist_ok=True)
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  compiler=vecops.Compiler(target="Native",cc="gcc",cxx="g++",jobs=args.jobs,
                          cache_dir=args.build_dir/"cache",build_dir=args.build_dir/"build")
  cases=[]
  with vecops.collect_compile_requests() as collector:
    for kind in args.kind:
      for group,name,m,n,k in catalog(kind in {"i8","mixed"}):
        if args.shape and f"{m}x{n}x{k}" not in args.shape or args.max_flops and 2*m*n*k>args.max_flops:
          continue
        for mode in args.mode:
          for pipeline in args.pipeline:
            for level in args.level:
              pair,types=kernels(compiler,m,n,k,mode,kind,pipeline,level,{"cpu_matmul_fusion":args.fusion,'cpu_gemv_columns':args.gemv_columns})
              ad,bd,na,nb,acc,out,prologue=types
              meta=(vecops.TensorMeta((m,k),dtype=getattr(vecops,ad)),
                    vecops.TensorMeta((k,n),strides=(1,k),dtype=getattr(vecops,bd)),
                    vecops.TensorMeta((m,n),dtype=getattr(vecops,out)),
                    vecops.TensorMeta((1 if prologue==2 else m,n),dtype=getattr(vecops,acc)))
              for kernel in pair:
                kernel.operator(*meta,1,**kernel._core.specialization_values)
              cases.append((group,name,m,n,k,mode,kind,pipeline,level,pair,types))
  print(f"{len(cases)} cases, {len(collector)} artifacts",flush=True)
  vecops.compile_batch(collector.requests,parallelism=args.jobs,execution_parallelism=1)
  timing_lease=reserve_cpu(args.cpu)
  os.sched_setaffinity(0,{args.cpu})
  stream=args.output.open("w") if args.output else __import__("sys").stdout
  writer=csv.writer(stream)
  writer.writerow(("group","name","m","n","k","meta","dtype","pipeline","level","dsl_us","cpp_us","dsl_over_cpp"))
  rng=np.random.default_rng(17)
  for group,name,m,n,k,mode,kind,pipeline,level,pair,types in cases:
    ad,bd,na,nb,acc,out,prologue=types
    a=aligned_empty((m,k),ad)
    b=aligned_empty((n,k),bd).T
    c=aligned_empty((m,n),out)
    d=aligned_empty((1 if prologue==2 else m,n),acc)
    a[:]=rng.integers(0 if ad=="uint8" else -3,4,size=a.shape)/ (8 if "float" in ad else 1)
    b[:]=rng.integers(-3,4,size=b.shape)/ (8 if "float" in bd else 1)
    d[:]=rng.integers(-3,4,size=d.shape)
    calls=lambda repeat:tuple(lambda kernel=kernel:kernel.operator.native(a,b,c,d,repeat,**kernel._core.specialization_values) for kernel in pair)
    ds,cp=calls(1)
    cp();reference=c.copy();ds()
    np.testing.assert_allclose(c,reference,rtol=5e-3,atol=3e-4)
    x,y=a.astype(np.float32),b.astype(np.float32)
    if pipeline=="affine_input":
      x,y=x*2+1,y*2+1
    expected=x.astype(numpy_dtype(na)).astype(numpy_dtype(acc))@y.astype(numpy_dtype(nb)).astype(numpy_dtype(acc))
    if prologue:
      expected=expected+d
    epilogue=PIPELINES[pipeline][1]
    if epilogue:
      expected=expected.astype(np.float32)
    if epilogue==1:expected=np.maximum(expected,0)
    if epilogue==2:expected=np.clip(expected,-0.25,0.25)
    if epilogue==3:expected=expected*0.5
    if epilogue==4:expected=expected/(1+np.exp(-expected))
    np.testing.assert_allclose(c,expected.astype(c.dtype),rtol=5e-3,atol=3e-4)
    repeat=max(2,min(4000,20_000_000//(m*n*k)))
    td,tc=measure(calls(repeat),repeat,args.rounds)
    writer.writerow((group,name,m,n,k,mode,kind,pipeline,level,f"{td:.4f}",f"{tc:.4f}",f"{td/tc:.4f}"));stream.flush()
    if args.output:print(f"{mode} {m}x{n}x{k} {kind}/{pipeline}/{level}: {td/tc:.3f}x",flush=True)
  if args.output:stream.close()
  timing_lease.close()


if __name__=="__main__":
  main()
