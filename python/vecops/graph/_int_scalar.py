# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact one-element int64 offset addition for graph control/seed dataflow."""
from functools import lru_cache


@lru_cache(None)
def kernel(a_shape,a_strides,output_shape,output_strides,parallelism,vector_token):
  import vecops.language as T
  @T.jit(out_idx=[2])
  def factory():
    @T.template(source='_IntScalar.cpp',recipe_id='graph-integer-scalar-add-v1')
    def entry(a:T.Tensor(a_shape,a_strides,T.int64),b:T.Tensor(a_shape,a_strides,T.int64),
              output:T.Tensor(output_shape,output_strides,T.int64),offset:T.int64):
      pass
    return entry
  compiled=factory()
  compiled.explicit_torch_function()
  return compiled


def select(schema,arguments,result,metadata):
  if schema.name!='aten::add' or schema.overload_name not in {'Tensor','Scalar'}:return None
  if not hasattr(result,'id') or len(arguments)<2 or not hasattr(arguments[0],'id'):return None
  offset=arguments[1]
  if type(offset) is not int or not -(1<<63)<=offset<(1<<63):return None
  if len(arguments)>2 and (type(arguments[2]) is not int or arguments[2]!=1):return None
  a,out=metadata(arguments[0]),metadata(result)
  if a.dtype!='torch.int64' or out.dtype!=a.dtype or tuple(a.shape)!=(1,) or tuple(out.shape)!=(1,):return None
  return arguments[0],offset
