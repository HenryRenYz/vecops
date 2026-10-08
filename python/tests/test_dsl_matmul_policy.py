# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fusion is a DAG/cost decision, including inlined helper and packing paths."""
import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.pipeline import kernel_pipeline, cpu_vector_pipeline
from vecops.language._compiler.lowering.tile_to_vector import lower_kernel
from vecops.language._compiler.backends.cpu.planning import prepare
from vecops.language._compiler.ir.traversal import walk


@T.func
def affine(x):
  return x * 2 + 1


@T.func
def expensive_tail(x):
  return T.exp(affine(x))


def plans(template, policy='auto',**options):
  options={'cpu_matmul_fusion': policy,**options}
  kernel=kernel_pipeline(options=options).run(lower(template.ir))
  vectors=lower_kernel(kernel, manager=cpu_vector_pipeline(options=options))
  prepared=prepare(vectors)
  return tuple(prepared.matmuls.values())


@pytest.mark.parametrize('columns,expensive,materialized',[
    (1,False,False), (1,True,False), (64,False,False),
    (64,True,True), (4096,False,True)])
def test_reload_cost_across_helper_calls(columns,expensive,materialized):
  @T.template
  def kernel(A:T.Tensor((17,33),dtype='float32'),
             B:T.Tensor((33,columns),dtype='float32'),
             C:T.Tensor((17,columns),dtype='float32',access='output')):
    a=expensive_tail(T.load(A)) if expensive else affine(T.load(A))
    T.dot((a,B),C,implementation='vector')
  plan,=plans(kernel)
  operand=plan.operands[0]
  assert (operand.materialization is not None)==materialized
  assert operand.operation_cost >= (16 if expensive else 2)
  if materialized:
    assert I.constant(operand.condition) is True


def test_packed_b_tail_produced_outside_mc_loop():
  @T.template
  def kernel(A:T.Tensor((4096,33),dtype='float32'),
             B:T.Tensor((33,65),dtype='float32'),
             C:T.Tensor((4096,65),dtype='float32',access='output')):
    T.library.matmul((A,expensive_tail(T.load(B))),C,packing='B',BM=16,BN=32)
  plan,=plans(kernel)
  assert plan.operands[1].materialization is None
  assert 'outside MC' in plan.operands[1].reason
  never,=plans(kernel,'never')
  assert never.operands[1].materialization is not None


def test_dynamic_reload_cost_remains_runtime_predicate():
  N=T.dynamic(lower=0)
  @T.template
  def kernel(A:T.Tensor((17,33),dtype='float32'),
             B:T.Tensor((33,N),dtype='float32'),
             C:T.Tensor((17,N),dtype='float32',access='output')):
    T.dot((expensive_tail(T.load(A)),B),C)
  plan,=plans(kernel)
  operand=plan.operands[0]
  assert operand.materialization is not None
  assert I.constant(operand.condition) is None
  always,=plans(kernel,'always')
  assert always.operands[0].materialization is None
  tuned,=plans(kernel,cpu_gemv_columns=8)
  assert tuned.vector_columns==8
  for invalid in (0,9,True):
    with pytest.raises(ValueError,match='cpu_gemv_columns'):
      plans(kernel,cpu_gemv_columns=invalid)


def test_load_conversion_participates_in_recompute_cost():
  @T.template
  def kernel(A:T.Tensor((17,33),dtype='float32'),
             B:T.Tensor((33,4096),dtype='bfloat16'),
             C:T.Tensor((17,4096),dtype='float32',access='output')):
    T.dot((T.load(A,dtype='bfloat16'),B),C)
  plan,=plans(kernel)
  assert plan.operands[0].operation_cost==1
  assert plan.operands[0].materialization is not None
  assert plan.operands[1].operation_cost==0
  assert plan.operands[1].materialization is None


def test_layernorm_statistics_not_recomputed_in_dot_operand():
  @T.template
  def kernel(A:T.Tensor((17,33),dtype='float32'),
             B:T.Tensor((33,65),dtype='float32'),
             C:T.Tensor((17,65),dtype='float32',access='output')):
    a=affine(T.library.layernorm(T.load(A)))
    T.library.matmul((a,B),C,packing='A')
  plan,=plans(kernel)
  assert any(e.op in {'mul','add','sub'} for e in walk(plan.operands[0].expression))
  assert not any(e.op.startswith('reduce_') or e.op in {'axis_reduce','normalize'}
                 for e in walk(plan.operands[0].expression))


def test_dynamic_packing_bypass_uses_gemv_recompute_policy():
  M,N,K=(T.dynamic(lower=0) for _ in range(3))
  @T.template
  def kernel(A:T.Tensor((M,K),dtype='float32'),
             B:T.Tensor((K,N),dtype='float32'),
             C:T.Tensor((M,N),dtype='float32',access='output')):
    T.library.matmul((expensive_tail(T.load(A)),expensive_tail(T.load(B))),C)
  plan,=plans(kernel)
  def evaluate(e,m,n):
    for dimension,value in zip(plan.expression.type.shape,(m,n)):
      if I.same_expr(e,dimension):
        return value
    if e.op=='literal':
      return e.data
    if e.op=='where':
      return evaluate(e.args[1 if evaluate(e.args[0],m,n) else 2],m,n)
    a,b=(evaluate(arg,m,n) for arg in e.args)
    return {'and':lambda:a and b,'gt':lambda:a>b,'add':lambda:a+b,
            'floordiv':lambda:a//b}[e.op]()
  a,b=plan.operands
  assert a.materialization is not None and b.materialization is not None
  assert evaluate(a.condition,1,256) is True
  assert evaluate(b.condition,1,256) is False
  assert evaluate(a.condition,256,1) is False
  assert evaluate(b.condition,256,1) is True
  assert evaluate(a.condition,256,64) is False
  assert evaluate(b.condition,256,64) is False
