# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bounded floating sum lowering; other axes/dtypes keep their ATen semantics."""
from functools import lru_cache


@lru_cache(None)
def kernel(a_shape,a_strides,a_dtype,output_shape,output_strides,output_dtype,parallelism,vector_token):
  import vecops.language as T
  @T.jit(out_idx=[2])
  def factory():
    @T.template(source='_Reduction.cpp',recipe_id='graph-short-axis-sum-v1')
    def entry(a:T.Tensor(a_shape,a_strides,a_dtype), b:T.Tensor(a_shape,a_strides,a_dtype),
              output:T.Tensor(output_shape,output_strides,output_dtype), alpha:T.float64):
      pass
    return entry
  compiled=factory()
  compiled.explicit_torch_function()
  return compiled


def select(schema,arguments,result,metadata):
  if schema.name!='aten::sum' or schema.overload_name!='dim_IntList':return None
  if not hasattr(result,'id') or not hasattr(arguments[0],'id'):return None
  a=metadata(arguments[0]);out=metadata(result)
  floating={'torch.float32','torch.bfloat16','torch.float16'}
  if a.dtype not in floating or out.dtype!=a.dtype or len(a.shape)<2:return None
  axes=arguments[1]
  if not isinstance(axes,(list,tuple)) or len(axes)!=1:return None
  if not isinstance(axes[0],int) or axes[0]%len(a.shape)!=len(a.shape)-2:return None
  if not 0<=a.shape[-2]<=32 or a.shape[-1]%64 or a.strides[-1]!=1:return None
  keep=arguments[2] if len(arguments)>2 else False
  if type(keep) is not bool:return None
  expected=(*a.shape[:-2],1,a.shape[-1]) if keep else (*a.shape[:-2],a.shape[-1])
  from ._pointwise import contiguous_layout
  if tuple(out.shape)!=expected or not contiguous_layout(out.shape,out.strides):return None
  return arguments[0]
