# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Union request ownership preserves uneven shapes, policies and private state."""
import json
from types import SimpleNamespace

import pytest
import vecops
from vecops import _C
from vecops._precompile import CompileRequest
from vecops import _prepare_exchange as exchange


def declaration(name='copy', policy='toolchain'):
  definition=vecops.KernelDef(name,[
      vecops.TensorDef('x',[vecops.Dynamic(1,0)],[1],dtype=vecops.float32),
      vecops.TensorDef('out',[vecops.Dynamic(1,0)],[1],dtype=vecops.float32,
                       access=vecops.TensorAccess.inout),
      vecops.ValueDef('alpha',vecops.float32)],values={'Compute':vecops.DType})
  return SimpleNamespace(source='/source.cpp',kernel_def=definition,
      _build_identity=policy,_source_identity='source-content',_recipe_id='test',
      compiler=SimpleNamespace(cache_dir='/cache',cache_mode='read-write'))


def request(handle,n):
  operation=SimpleNamespace(_handle=handle)
  signature=(('tensor','float32',(n,),(1,),'TensorAccess.input',1,0),
             ('tensor','float32',(n,),(1,),'TensorAccess.inout',1,0),('scalar','float32'))
  call=_C.KernelCall([vecops.TensorMeta((n,),dtype='float32')._native(),
      vecops.TensorMeta((n,),dtype='float32',access=vecops.TensorAccess.inout)._native(),
      _C.Scalar.floating(2.5,_C.DType.float32)],{'Compute':_C.DType.float32})
  return CompileRequest(operation,call,(handle,signature,(('Compute',_C.DType.float32),)), 'copy','/source.cpp')


def payload(requests,rank,parallelism):
  result=[]
  def capture(value):
    result.append(json.loads(json.dumps(value)))
    raise InterruptedError('collected')
  with pytest.raises(InterruptedError,match='collected'):
    exchange.exchange_requests(requests,rank,capture,parallelism)
  return result[0]


def test_union_moves_uneven_shapes_to_equivalent_local_operator(monkeypatch):
  monkeypatch.setattr(exchange,'_declarations',{10:declaration(),20:declaration()})
  a,b=request(10,8),request(20,3)
  gathered=[payload([a],0,36),payload([b],1,36)]
  first,owners,complete=exchange.exchange_requests([a],0,lambda _:gathered,36)
  second,other_owners,other_complete=exchange.exchange_requests([b],1,lambda _:gathered,36)
  assert owners==other_owners==(0,)
  assert complete and not other_complete and not second
  assert {tuple(x.call.arguments.values[0].sizes)for x in first}=={(8,),(3,)}
  assert all(x.operator is a.operator for x in first)
  assert all(x.call.arguments.values[1].writable for x in first)
  assert all(x.call.arguments.values[2].value==2.5 for x in first)
  assert all(x.call.values=={'Compute':_C.DType.float32}for x in first)
  # Process handles and tensor addresses never appear in the transport record.
  assert 'handle' not in json.dumps(gathered) and 'data_ptr' not in json.dumps(gathered)


@pytest.mark.parametrize('difference',['policy','parallelism','schema','private'])
def test_union_retains_incompatible_or_unknown_owner(monkeypatch,difference):
  first,second=declaration(),declaration()
  if difference=='policy':second._build_identity='another-toolchain'
  if difference=='schema':second.kernel_def=declaration('different').kernel_def
  registry={10:first,20:second}
  if difference=='private':registry.pop(20)
  monkeypatch.setattr(exchange,'_declarations',registry)
  a,b=request(10,8),request(20,8)
  threads=18 if difference=='parallelism'else 36
  gathered=[payload([a],0,36),payload([b],1,threads)]
  for rank,requests,nthreads in [(0,[a],36),(1,[b],threads)]:
    selected,owners,complete=exchange.exchange_requests(requests,rank,lambda _:gathered,nthreads)
    assert owners==(0,1) and len(selected)==1 and complete
    assert selected[0].operator is requests[0].operator


def test_union_deduplicates_equal_shapes_across_all_ranks(monkeypatch):
  monkeypatch.setattr(exchange,'_declarations',{h:declaration()for h in range(1,5)})
  requests=[[request(h,8)]for h in range(1,5)]
  gathered=[payload(r,i,36)for i,r in enumerate(requests)]
  selected,owners,_=exchange.exchange_requests(requests[0],0,lambda _:gathered,36)
  assert owners==(0,)and len(selected)==1
