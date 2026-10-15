# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Non-unit-K SME staging preserves floating storage bits and padding."""
import platform
import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import get_vector_state


def template(dtype, side):
  rows,cols,capacity = (T.dynamic(lower=0) for _ in range(3))
  rs,cs = T.dynamic(),T.dynamic()
  @T.template
  def kernel(X:T.Tensor((rows,cols),(rs,cs),dtype=dtype),
             P:T.Tensor((capacity,),dtype=dtype,access="output")):
    T.pack(T.load(X),P,side=side,format="sme")
  return kernel


@pytest.mark.skipif(platform.machine().lower() not in {"aarch64","arm64"},reason="requires native SME")
@pytest.mark.parametrize("dtype",["float16","float32","float64"])
@pytest.mark.parametrize("side",["A","B"])
def test_native_sme_raw_pack_float_payloads(dtype,side,tmp_path):
  dt=np.dtype(dtype);bits=np.dtype(f"uint{dt.itemsize*8}")
  patterns={2:[0,0x8000,0x3c00,0x7c00,0xfc00,0x7e01,0x7d23,1],
            4:[0,0x80000000,0x3f800000,0x7f800000,0xff800000,0x7fc00001,0x7fa01234,1],
            8:[0,0x8000000000000000,0x3ff0000000000000,0x7ff0000000000000,
               0xfff0000000000000,0x7ff8000000000001,0x7ff0123456789abc,1]}[dt.itemsize]
  shape=(17,19) if side=="A" else (19,17)
  source=np.resize(np.array(patterns,dtype=bits),np.prod(shape)).reshape(shape).view(dt)
  compiler=make_compiler(target="Native",jobs=2,cache_dir=tmp_path/"cache",build_dir=tmp_path/"build")
  compiled=T.compile(template(dtype,side),compiler=compiler)
  word=max(4,dt.itemsize);group=word//dt.itemsize
  span=2*get_vector_state().sme_bytes//word
  for x in [np.array(source,order="C"),np.array(source,order="F"),source[::-1,::-1]]:
    logical=x if side=="A" else x.T;s,k=logical.shape
    expected=np.zeros(((s+span-1)//span,(k+group-1)//group,span,group),dtype=bits)
    logical_bits=logical.view(bits)
    for i in range(s):
      for j in range(k):expected[i//span,j//group,i%span,j%group]=logical_bits[i,j]
    output_bits=np.full(expected.size+7,85,dtype=bits)
    compiled.operator.native(x,output_bits.view(dt),**compiled._core.specialization_values)
    np.testing.assert_array_equal(output_bits[:expected.size],expected.reshape(-1))
    np.testing.assert_array_equal(output_bits[expected.size:],85)
