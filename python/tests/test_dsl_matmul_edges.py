# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Product boundaries, packed panel origins and AMX compiler-ordering edges."""
import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import amx_available_for_test, make_compiler


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root=tmp_path_factory.mktemp('dsl-matmul-edges')
  return make_compiler(target='Native',jobs=4,
                       cache_dir=root/'cache',build_dir=root/'build')


@pytest.mark.parametrize('dtype',['bfloat16','float16'])
def test_const_native_width_and_configured_tails(compiler,dtype):
  import ml_dtypes
  memory=ml_dtypes.bfloat16 if dtype=='bfloat16' else np.float16
  rng=np.random.default_rng(31)
  requests=[]
  with vecops.collect_compile_requests() as collector:
    for m,n,k in ((16,48,128),(8,128,128),(19,23,64),(17,19,65)):
      @T.template
      def kernel(A:T.Tensor((m,k),dtype=dtype),B:T.Tensor((k,n),(1,k),dtype),
                 C:T.Tensor((m,n),dtype='float32',access='output')):
        value=T.dot((T.load(A),T.load(B)),C)
      requests.append((m,n,k,T.compile(kernel,compiler=compiler)))
  vecops.compile_batch(collector.requests,parallelism=4,execution_parallelism=1)
  for m,n,k,kernel in requests:
    a=rng.normal(size=(m,k)).astype(memory)
    b=rng.normal(size=(n,k)).astype(memory).T
    c=np.empty((m,n),np.float32)
    kernel.operator.native(a,b,c)
    np.testing.assert_allclose(c,a.astype(np.float32)@b.astype(np.float32),rtol=2e-4,atol=2e-4)


@pytest.mark.parametrize('side',['A','B'])
def test_input_output_identical_mapping_snapshots(compiler,side):
  @T.template
  def kernel(A:T.Tensor((17,17),dtype='float32'),B:T.Tensor((17,17),dtype='float32'),
             C:T.Tensor((17,17),dtype='float32',access='output')):
    T.dot((A,B),C,implementation='vector')
  compiled=T.compile(kernel,compiler=compiler)
  rng=np.random.default_rng(19)
  a=rng.normal(size=(17,17)).astype(np.float32)
  b=rng.normal(size=(17,17)).astype(np.float32)
  expected=a@b
  c=a if side=='A' else b
  compiled.operator.native(a,b,c)
  np.testing.assert_allclose(c,expected,rtol=2e-5,atol=2e-5)


def test_sibling_products_and_pack_product_boundaries(compiler):
  @T.template
  def kernel(A:T.Tensor((7,9),dtype='float32'),B:T.Tensor((9,11),dtype='float32'),
             C:T.Tensor((7,11),dtype='float32',access='output'),
             O:T.Tensor((7,9),dtype='float32',access='output')):
    first=T.dot(A,B)
    second=T.dot(T.load(A)*2,B)
    T.store(C,T.max(first+second,0))
    packed=T.pack(first,side='A',format='dense')
    T.dot((packed,T.transpose(B)),O)
  compiled=T.compile(kernel,compiler=compiler)
  rng=np.random.default_rng(19)
  a=rng.normal(size=(7,9)).astype(np.float32)
  b=rng.normal(size=(9,11)).astype(np.float32)
  c,o=np.empty((7,11),np.float32),np.empty((7,9),np.float32)
  compiled.operator.native(a,b,c,o)
  np.testing.assert_allclose(c,np.maximum(3*(a@b),0),rtol=2e-5,atol=2e-5)
  np.testing.assert_allclose(o,(a@b)@b.T,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize('dtype,format_,implementation',[
  ('float32','dense','vector'),
  pytest.param('bfloat16','amx','amx', marks=pytest.mark.skipif(not amx_available_for_test(), reason="AMX packing requires an x86 AMX target")),
  pytest.param('float16','amx','vector', marks=pytest.mark.skipif(not amx_available_for_test(), reason="AMX packing requires an x86 AMX target")),
])
def test_prepacked_cache_panels_and_fused_final_store(compiler,dtype,format_,implementation):
  import ml_dtypes
  memory={'float32':np.float32,'bfloat16':ml_dtypes.bfloat16,'float16':np.float16}[dtype]
  @T.template
  def kernel(A:T.Tensor((37,65),dtype=dtype),B:T.Tensor((65,45),(1,65),dtype),
             D:T.Tensor((1,45),dtype='float32'),C:T.Tensor((37,45),dtype='float64',access='output')):
    a=T.pack(A,side='A',format=format_)
    b=T.pack(B,side='B',format=format_)
    value=T.library.matmul((a,b),C,acc=T.load(D),BM=32,BN=32,BK=32,implementation=implementation)
    T.store(C,T.cast(value,'float64')*0.5+1)
  compiled=T.compile(kernel,compiler=compiler)
  rng=np.random.default_rng(21)
  a=rng.normal(size=(37,65)).astype(memory)
  b=rng.normal(size=(45,65)).astype(memory).T
  d=rng.normal(size=(1,45)).astype(np.float32)
  c=np.empty((37,45),np.float64)
  compiled.operator.native(a,b,d,c)
  expected=(a.astype(np.float32)@b.astype(np.float32)+d).astype(np.float64)*0.5+1
  np.testing.assert_allclose(c,expected,rtol=3e-5,atol=3e-5)


def test_zero_k_with_explicit_phases(compiler):
  @T.template
  def kernel(A:T.Tensor((17,0),dtype='bfloat16'),B:T.Tensor((0,19),dtype='bfloat16'),
             D:T.Tensor((1,19),dtype='float32'),C:T.Tensor((17,19),dtype='float32',access='output')):
    value=T.library.matmul((A,B),C,acc=T.load(D),BM=32,BN=32,BK=64)
    T.store(C,T.max(value+1,0))
  compiled=T.compile(kernel,compiler=compiler)
  import ml_dtypes
  a=np.empty((17,0),ml_dtypes.bfloat16)
  b=np.empty((0,19),ml_dtypes.bfloat16)
  d=np.arange(19,dtype=np.float32).reshape(1,19)-5
  c=np.empty((17,19),np.float32)
  compiled.operator.native(a,b,d,c)
  np.testing.assert_array_equal(c,np.broadcast_to(np.maximum(d+1,0),c.shape))
