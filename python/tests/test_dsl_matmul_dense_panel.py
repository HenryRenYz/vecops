# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Runtime cache packing also binds the prepared register-transpose path."""
import numpy as np
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler


def test_runtime_dense_panel_transpose_and_global_arithmetic(tmp_path):
  M,N,K=(T.dynamic(lower=0) for _ in range(3))
  @T.template
  def kernel(A:T.Tensor((M,K),dtype='float32'),B:T.Tensor((K,N),dtype='float32'),
             C:T.Tensor((M,N),dtype='float32',access='output')):
    value=T.library.matmul((T.load(A)*2+1,T.load(B)*0.5-1),C,
                           BM=16,BN=32,BK=32,implementation='vector')
    T.store(C,T.max(value*0.25+1,0))
  compiler=make_compiler(target='Native',jobs=6,
                         cache_dir=tmp_path/'cache',build_dir=tmp_path/'build')
  compiled=T.compile(kernel,compiler=compiler)
  rng=np.random.default_rng(17)
  a=rng.normal(size=(37,65)).astype(np.float32)
  b=rng.normal(size=(65,45)).astype(np.float32)
  c=np.empty((37,45),np.float32)
  compiled.operator.native(a,b,c,**compiled._core.specialization_values)
  np.testing.assert_allclose(c,np.maximum(((a*2+1)@(b*0.5-1))*0.25+1,0),rtol=3e-5,atol=3e-5)
  assert 'transpose_vec' in compiled.generated_source
