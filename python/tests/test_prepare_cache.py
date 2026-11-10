# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Discovery recipes round-trip schemas and reject modified byte dependencies."""
import json
from pathlib import Path

import pytest
import vecops
import vecops.language as T
from vecops import _prepare_cache as cache
from vecops._prepare_exchange import _definition_key,operator_key
from vecops._precompile import CompileRequestCollector,maybe_collect_call,collect_compile_requests


def test_schema_expression_roundtrip_is_address_free():
  n=T.dynamic(lower=0);n.bind_name('N')
  definition=vecops.KernelDef('expression',[vecops.TensorDef('x',(n,3),(n*3,1),dtype='Type')])
  key=_definition_key(definition)
  restored=cache._restore_definition(json.loads(json.dumps(key)))
  assert restored.inputs[0].strides[0].kind==vecops._C.DimensionKind.expression
  assert restored.values==definition._native().values


@pytest.mark.parametrize('custom_namespace',[False,True])
def test_request_restore_preserves_metadata_and_rejects_header_edit(tmp_path,monkeypatch,custom_namespace):
  monkeypatch.setattr(cache,'compiler_identity',lambda _:'fixture-build')
  header=tmp_path/'dependency.h';header.write_text('before')
  source=tmp_path/'kernel.cpp';source.write_text('#include "dependency.h"\n')
  compiler=vecops.Compiler(target='Scalar',cache_mode='cache-only',cache_dir=tmp_path/'cache')
  definition=vecops.KernelDef('copy',[
      vecops.TensorDef('x',[vecops.Dynamic(1,0)],[1],dtype=vecops.float32),
      vecops.TensorDef('y',[vecops.Dynamic(1,0)],[1],dtype=vecops.float32,output=True)])
  operator=vecops.Operator(source,definition,compiler,recipe_id='copy',build_identity='fixture-build')
  if custom_namespace:
    from vecops import _schema_bridge
    monkeypatch.setattr(_schema_bridge,'_registrations',{})
    _schema_bridge.register_torch_operator('vecops_test_restore_namespace','copy',
        operator.native,definition._native(),compiler=operator.compiler,source=str(source))
  with collect_compile_requests()as collector:
    maybe_collect_call(operator.native,definition._native(),
        [vecops.TensorMeta((7,),dtype='float32'),vecops.TensorMeta((7,),dtype='float32')],{},source=str(source))
  payload=json.loads(json.dumps(cache.export_requests(collector)))
  # Distinct operators can use one complete frozen compiler policy.
  second=json.loads(json.dumps(payload['operators'][0]))
  second['recipe_id']='copy-second'
  payload['operators'].append(second)
  request=json.loads(json.dumps(payload['requests'][0]));request['operator']=1
  payload['requests'].append(request)
  constructors=[]
  original=cache.Compiler
  def construct(**kwargs):
    constructors.append(kwargs)
    return original(**kwargs)
  monkeypatch.setattr(cache,'Compiler',construct)
  restored=CompileRequestCollector();cache.restore_requests(restored,payload)
  assert len(constructors)==1
  assert restored._restored_declarations[0].compiler is restored._restored_declarations[1].compiler
  request=restored.requests[0]
  assert tuple(request.call.arguments.values[0].sizes)==(7,)
  assert request.call.arguments.values[1].writable
  assert operator_key(request.operator)==operator_key(operator.native)
  payload['operators'][1]['compiler']['kwargs']['cflags']=['-O1']
  constructors.clear()
  cache.restore_requests(CompileRequestCollector(),payload)
  assert len(constructors)==2
  header.write_text('changed same dependency')
  with pytest.raises(ValueError,match='source inputs changed'):
    cache.restore_requests(CompileRequestCollector(),payload)
