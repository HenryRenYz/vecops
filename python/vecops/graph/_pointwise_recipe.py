# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Relocatable pointwise lowering decisions; native bindings remain local."""
import hashlib
import os


def contract(tape, execution, parallelism, vector_token, borrow_outputs, native_donation):
  from vecops._identity import implementation_identity
  from vecops.language._compiler.environment import registry_fingerprint
  def arguments(instruction):
    name = instruction.function._schema.name
    if name in {'vecops_comm::run','vecops_comm::consume'}:
      from vecops.distributed._template import describe
      return (describe(name,instruction.arguments[0]),*instruction.arguments[1:])
    return instruction.arguments
  instructions=[(str(x.function._schema),arguments(x),x.result,x.matmul_precision) for x in tape.instructions]
  tensors=[(key,x.metadata.shape,x.metadata.strides,x.metadata.dtype,x.metadata.device,
            x.metadata.layout,x.escapes)for key,x in execution.tensors.items()]
  return hashlib.sha256(repr((1,instructions,tensors,parallelism,vector_token,
      borrow_outputs,native_donation,os.environ.get('VECOPS_GRAPH_SHORT_REDUCTION','1'),
      os.environ.get('VECOPS_GRAPH_INTEGER_SCALARS','1'),implementation_identity(),
      registry_fingerprint())).encode()).hexdigest()


class _Binding:
  _graph_artifact_prepared=True

  def __init__(self, function, specs):
    self.function,self.specs=function,specs

  def explicit_torch_binding(self):
    return self.function,self.specs


def export(key, pointwise, inplace):
  from ._template import encode
  rows=[]
  for ordinal,(compiled,a,b,output,alpha) in pointwise.items():
    function,specs=compiled.explicit_torch_binding()
    schema=function._schema
    rows.append([ordinal,schema.name,schema.overload_name,str(schema),encode(specs),
                 encode(a),encode(b),encode(output),alpha,
                 encode(inplace.get(ordinal))])
  return dict(version=1,contract=key,rows=rows)


def restore(payload, key, instruction_count):
  if payload is None or payload.get('version')!=1 or payload.get('contract')!=key:
    return None
  from ._template import decode
  from ._replay import _native_owners
  import torch
  pointwise,inplace={},{}
  try:
    for ordinal,name,overload,schema,specs,a,b,output,alpha,alias in payload['rows']:
      if type(ordinal)is not int or not 0<=ordinal<instruction_count or ordinal in pointwise:
        raise ValueError('invalid pointwise recipe ordinal')
      namespace,op=name.split('::')
      if namespace!='vecops_language':raise ValueError('untracked pointwise namespace')
      function=getattr(getattr(getattr(torch.ops,namespace),op),overload or 'default')
      if str(function._schema)!=schema:raise ValueError('pointwise schema changed')
      pointwise[ordinal]=(_Binding(function,decode(specs)),decode(a),decode(b),decode(output),alpha)
      if alias is not None:inplace[ordinal]=decode(alias)
    # Require published, current native registration owners before accepting
    # a recipe; an op name by itself is never evidence of a native binding.
    _native_owners((x[0].function for x in pointwise.values()),{})
  except (KeyError,ValueError,TypeError,AttributeError,RuntimeError):
    return None
  return pointwise,inplace
