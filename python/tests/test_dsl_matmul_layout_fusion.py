# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Transpose/packing/broadcast fusion uses global, not local panel coordinates."""
import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_source_algorithms_lowered, make_compiler


@pytest.fixture(scope='module')
def compiler(tmp_path_factory):
  root=tmp_path_factory.mktemp('dsl-matmul-layout')
  return make_compiler(target='Native',jobs=4,
                       cache_dir=root/'cache',build_dir=root/'build')


def test_dense_pack_uses_automatic_transpose(compiler):
  @T.template
  def kernel(A:T.Tensor((17,65),dtype='float32'),B:T.Tensor((65,45),dtype='float32'),
             C:T.Tensor((17,45),dtype='float32',access='output')):
    b=T.pack(T.load(B)+T.cast(T.arange(45),'float32'),side='B',format='dense')
    value=T.library.matmul((A,b),C,BM=16,BN=16,BK=32)
    T.store(C,value*0.5)
  compiled=T.compile(kernel,compiler=compiler)
  rng=np.random.default_rng(9)
  a=rng.normal(size=(17,65)).astype(np.float32)
  b=rng.normal(size=(65,45)).astype(np.float32)
  c=np.empty((17,45),np.float32)
  compiled.operator.native(a,b,c)
  np.testing.assert_allclose(c,(a@(b+np.arange(45,dtype=np.float32)))*0.5,rtol=2e-4,atol=2e-4)
  # arange is not an affine memory permutation; it deliberately retains the
  # general vector scheduler while still using the transposed global index.


def test_dense_pack_overlap_and_plain_transpose_plan(compiler):
  @T.template
  def kernel(B:T.Tensor((19,23),dtype='float32'),
             P:T.Tensor((19*23,),dtype='float32',access='output')):
    T.pack(B,P,side='B',format='dense')
  compiled=T.compile(kernel,compiler=compiler)
  assert_source_algorithms_lowered(compiled.backend_ir)
  b=np.arange(19*23,dtype=np.float32).reshape(19,23)
  expected=b.T.ravel().copy()
  compiled.operator.native(b,b.ravel())
  np.testing.assert_array_equal(b.ravel(),expected)


@T.func
def normalized_affine(x,gamma,bias):
  return T.cast(T.library.layernorm(x)*T.load(gamma)+T.load(bias),'bfloat16')


@pytest.mark.parametrize('column_output',[False,True])
def test_rank_one_normalization_tail_and_output_orientation(compiler,column_output):
  stride=(1,17) if column_output else (23,1)
  @T.template
  def kernel(A:T.Tensor((17,65),dtype='float32'),B:T.Tensor((65,23),dtype='bfloat16'),
             G:T.Tensor((65,),dtype='float32'),H:T.Tensor((65,),dtype='float32'),
             C:T.Tensor((17,23),stride,'float32',access='output')):
    a=normalized_affine(T.load(A)+T.load(H),G,H)
    value=T.library.matmul((a,B),C,BM=16,BN=16,BK=32)
    T.store(C,value+T.cast(T.arange(23),'float32'))
  compiled=T.compile(kernel,compiler=compiler)
  import ml_dtypes
  rng=np.random.default_rng(9)
  a=rng.normal(size=(17,65)).astype(np.float32)
  b=rng.normal(size=(65,23)).astype(ml_dtypes.bfloat16)
  g=rng.normal(size=65).astype(np.float32)
  h=rng.normal(size=65).astype(np.float32)
  c=np.empty((23,17),np.float32).T if column_output else np.empty((17,23),np.float32)
  compiled.operator.native(a,b,g,h,c)
  x=a+h
  centered=x-x.mean(axis=1,keepdims=True)
  normalized=(centered/np.sqrt(np.mean(centered*centered,axis=1,keepdims=True)+1e-5)*g+h).astype(ml_dtypes.bfloat16).astype(np.float32)
  expected=normalized@b.astype(np.float32)+np.arange(23,dtype=np.float32)
  np.testing.assert_allclose(c,expected,rtol=2e-4,atol=2e-4)
