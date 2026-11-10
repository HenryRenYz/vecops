# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Conservative JSON recipes for an application's validated discovery cache.

This module does not decide whether model tracing can be skipped. Its caller
must attest model code, state structure, inputs and execution configuration.
Restoration independently validates the original operator/compiler byte inputs.
"""
from types import SimpleNamespace

from . import _C
from ._compiler import Compiler, _CACHE_MODES
from ._identity import compiler_identity
from ._operator import Operator, _declarations
from ._prepare_exchange import _definition_key, _encode_argument, _decode_argument


_COMPILER_FIELDS = ('target','cc','cxx','cmake','generator','jobs','output','environment',
                    'cache_namespace','include_dirs','cflags','library_dirs','libraries','ldflags')


def _compiler_record(compiler):
  result = {key:getattr(compiler,key) for key in _COMPILER_FIELDS}
  result.update(cache_dir=str(compiler.cache_dir),build_dir=str(compiler.build_dir),
                cache_mode=next(key for key,value in _CACHE_MODES.items() if value==compiler.cache_mode))
  for key in ('cc','cxx','cmake'):
    if result[key] is not None:result[key]=str(result[key])
  for key in ('include_dirs','library_dirs'):
    result[key]=list(map(str,result[key]))
  return dict(kwargs=result,cwd=str(compiler.definition_cwd),identity=compiler_identity(compiler),
              vector_state=compiler.vector_state.env_token,
              inventory_excludes=list(map(str,compiler._source_include_inventory_excludes)))


def export_requests(collector):
  from . import _schema_bridge as bridge
  operators,handles,requests = [],{},[]
  for request in collector.requests:
    handle=request.operator._handle
    if handle not in handles:
      declaration=_declarations.get(handle)
      if declaration is None:
        raise ValueError('raw operator declaration is not persistently relocatable')
      record=dict(source=str(declaration.source),definition=_definition_key(declaration.kernel_def),
          compiler=_compiler_record(declaration.compiler),recipe_id=declaration._recipe_id,
          dependencies=list(map(str,declaration._dependencies)),source_identity=declaration._source_identity,
          bridges=[])
      for registration in bridge._registrations.values():
        if registration.operator._handle!=handle:continue
        # Every entry here is attached to the validated, persistently
        # reconstructible source Operator above. Namespace spelling is not an
        # ownership proof (AF3 peer kernels have their own namespace).
        record['bridges'].append(dict(library=registration.library,name=registration.name,
            source=registration.source,build_directory=None if registration.build_directory is None
              else str(registration.build_directory),verbose=registration.verbose,
            extra_cflags=registration.extra_cflags,return_outputs=registration.return_outputs,
            spec_order=registration.spec_order))
      handles[handle]=len(operators);operators.append(record)
    values=[(name,('dtype',int(value)) if isinstance(value,_C.DType) else ('int',value))
            for name,value in request.call.values.items()]
    requests.append(dict(operator=handles[handle],name=request.name,source=request.source,
        signature=request.key[1],values=values,
        arguments=[_encode_argument(x) for x in request.call.arguments.values]))
  return dict(version=1,operators=operators,requests=requests)


def _restore_definition(record):
  name,parameters,values,runtime_parallelism=record
  def bound(value):return value[1]
  def expression(value):
    kind=value[0]
    if kind=='const':
      return _C.DimensionExpr.const_symbol(value[1]) if value[1] else _C.DimensionExpr.literal(value[3])
    if kind=='dynamic':
      if not value[1]:raise ValueError('persisted runtime expression needs a named symbol')
      return _C.DimensionExpr.runtime_symbol(value[1])
    method={'add':'add','sub':'sub','mul':'mul','floordiv':'floordiv','mod':'mod','min':'minimum','max':'maximum'}[kind]
    return getattr(_C.DimensionExpr,method)(expression(value[1]),expression(value[2]))
  def dimension(value):
    if value[0]=='expression':return _C.DimensionDef.expression(expression(value[1]))
    kind,symbol,fixed,alignment,low,high=value
    kind=kind.split('.')[-1]
    if kind=='fixed':return _C.DimensionDef(fixed)
    if kind=='symbol':return _C.DimensionDef(symbol)
    if kind=='const':return _C.DimensionDef.constant()
    if kind=='dynamic':
      return (_C.DimensionDef.named_dynamic(symbol,bound(alignment),bound(low),bound(high)) if symbol else
              _C.DimensionDef.dynamic(bound(alignment),bound(low),bound(high)))
    raise ValueError('unsupported persisted dimension expression')
  inputs=[]
  for parameter in parameters:
    if parameter[0]=='tensor':
      _,key,shape,strides,symbol,dtype,anonymous,access,optional,device,index=parameter
      inputs.append(_C.TensorDef(key,list(map(dimension,shape)),list(map(dimension,strides)),
          dtype=None if anonymous else symbol or _C.DType(dtype),optional=optional,
          access=getattr(_C.TensorAccess,access.split('.')[-1]),device_type=device,device_index=index))
    else:
      _,key,dtype,default=parameter
      inputs.append(_C.ValueDef(key,_C.DType(dtype),None if default is None else default[1]))
  definition=_C.KernelDef(name,inputs,{key:getattr(_C.SpecializationType,value.split('.')[-1])for key,value in values})
  definition.runtime_parallelism=runtime_parallelism
  definition.validate()
  return definition


def restore_requests(collector,payload):
  """Recreate operators under the current byte attestation, without execution."""
  from pathlib import Path
  from ._precompile import CompileRequest
  from ._schema_bridge import register_torch_operator
  if payload['version']!=1:raise ValueError('unsupported discovery recipe version')
  import json
  declarations=[]
  compilers={}
  cwd=Path.cwd().resolve()
  for record in payload['operators']:
    policy=record['compiler']
    if cwd!=Path(policy['cwd']):
      raise ValueError('discovery compiler definition directory changed')
    # A restored declaration owns an immutable snapshot. Only byte-identical
    # complete policies share it, within this restoration (never process-wide).
    policy_key=json.dumps(policy,sort_keys=True,separators=(',',':'),allow_nan=False)
    compiler=compilers.get(policy_key)
    if compiler is None:
      compiler=Compiler(**policy['kwargs']).snapshot()
      compiler._source_include_inventory_excludes=tuple(Path(x)for x in policy['inventory_excludes'])
      if compiler.vector_state.env_token!=policy['vector_state'] or compiler_identity(compiler)!=policy['identity']:
        raise ValueError('discovery compiler input identity changed')
      compilers[policy_key]=compiler
    native=_restore_definition(record['definition'])
    definition=SimpleNamespace(name=native.name,parameters=native.inputs,
        _native=lambda native=native:native,_discovery_key=record['definition'])
    declaration=Operator(record['source'],definition,compiler,recipe_id=record['recipe_id'],
                         dependencies=record['dependencies'],build_identity=policy['identity'])
    if declaration._source_identity!=record['source_identity']:
      raise ValueError('discovery operator source inputs changed')
    declarations.append(declaration)
    for registration in record['bridges']:
      register_torch_operator(operator=declaration.native,kernel_def=native,compiler=declaration.compiler,
                              **registration)
  pending=[]
  for record in payload['requests']:
    declaration=declarations[record['operator']]
    values={key:_C.DType(value) if kind=='dtype' else value for key,(kind,value)in record['values']}
    def tuples(value):return tuple(map(tuples,value)) if isinstance(value,(list,tuple)) else value
    signature=tuples(record['signature'])
    # Runtime scalar payloads are not specialization keys, exactly as in live
    # collection; tensor argument geometry remains part of the request key.
    key=(declaration.native._handle,signature,tuple((key,str(value))for key,value in values.items()))
    call=_C.KernelCall([_decode_argument(x,s)for x,s in zip(record['arguments'],signature)],values)
    pending.append(CompileRequest(declaration.native,call,key,record['name'],record['source']))
  collector._restored_declarations=declarations
  for request in pending:collector.add(request)
