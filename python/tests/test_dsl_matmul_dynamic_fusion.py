"""Actual runtime GEMV bypass chooses the non-packing recompute policy."""
import numpy as np
import vecops
import vecops.language as T


@T.func
def produced(x):
  return T.cast(T.exp(T.load(x,dtype='float32')*2+1),'bfloat16')


def test_expensive_inputs_row_col_and_packing(tmp_path):
  M,N,K=(T.dynamic(lower=0) for _ in range(3))
  @T.template
  def kernel(A:T.Tensor((M,K),dtype='float32'),B:T.Tensor((K,N),dtype='float32'),
             C:T.Tensor((M,N),dtype='float32',access='output')):
    value=T.library.matmul((produced(A),produced(B)),C,BM=16,BN=16,BK=32)
    T.store(C,value*0.5)
  compiler=vecops.Compiler(target='Native',cc='gcc',cxx='g++',jobs=6,
                          cache_dir=tmp_path/'cache',build_dir=tmp_path/'build')
  compiled=T.compile(kernel,compiler=compiler)
  import ml_dtypes
  rng=np.random.default_rng(31)
  for m,n,k in ((1,257,65),(65,1,129),(17,19,33)):
    a=rng.uniform(-0.25,0.25,(m,k)).astype(np.float32)
    b=rng.uniform(-0.25,0.25,(k,n)).astype(np.float32)
    c=np.empty((m,n),np.float32)
    compiled.operator.native(a,b,c,**compiled._core.specialization_values)
    x=np.exp(a*2+1).astype(ml_dtypes.bfloat16).astype(np.float32)
    y=np.exp(b*2+1).astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_allclose(c,(x@y)*0.5,rtol=2e-4,atol=2e-4)
