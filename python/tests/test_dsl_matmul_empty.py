"""Empty dynamic matrices must not enter an invalid small AMX configuration."""
import numpy as np
import vecops
import vecops.language as T


def test_empty_dynamic_amx_and_zero_k_bias(tmp_path):
  M,N,K=(T.dynamic(lower=0) for _ in range(3))
  @T.template
  def kernel(A:T.Tensor((M,K),(K,1),'bfloat16'),B:T.Tensor((K,N),(N,1),'bfloat16'),
             D:T.Tensor((1,N),dtype='float32'),C:T.Tensor((M,N),dtype='float32',access='output')):
    value=T.dot((A,B),C,acc=T.load(D),implementation='amx')
    T.store(C,value+1)
  compiler=vecops.Compiler(target='Native',cc='gcc',cxx='g++',jobs=6,
                          cache_dir=tmp_path/'cache',build_dir=tmp_path/'build')
  compiled=T.compile(kernel,compiler=compiler)
  import ml_dtypes
  for m,n,k in ((0,5,7),(5,0,7),(0,0,0),(1,5,0),(5,1,0),(17,19,0)):
    a=np.empty((m,k),ml_dtypes.bfloat16)
    # Empty NumPy arrays use canonical dense ABI strides, including when a
    # transposed empty ndarray reports both C/F-contiguous. Declare that layout.
    b=np.empty((k,n),ml_dtypes.bfloat16)
    d=np.arange(n,dtype=np.float32).reshape(1,n)
    c=np.empty((m,n),np.float32)
    compiled.operator.native(a,b,d,c,**compiled._core.specialization_values)
    np.testing.assert_array_equal(c,np.broadcast_to(d+1,c.shape))
  assert 'empty products need no resources' in compiled.generated_source
