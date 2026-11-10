# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compile discovery can expose Torch metadata before a bridge exists."""
from types import SimpleNamespace

import pytest
import torch
import vecops
from vecops import _precompile, _schema_bridge
from vecops.graph.torch import tensor_metadata


def test_storage_free_observer_preserves_memory_graph_and_restores_on_failure(monkeypatch):
  definition=vecops.KernelDef('observer::run',[
      vecops.TensorDef('A',[4],[1],dtype=vecops.float32),
      vecops.TensorDef('C',[4],[1],dtype=vecops.float32,access=vecops.TensorAccess.output),
  ])._native()
  registration=SimpleNamespace(source=None,name='run')
  wrapper=_schema_bridge._public_wrapper('observer','run',definition,
      SimpleNamespace(_handle=123),None,registration,[False],[None])
  def compile_batch(requests,**kwargs):
    return _precompile.PrecompileResult(len(requests),len(requests),1,0.)
  monkeypatch.setattr(_precompile,'compile_batch',compile_batch)
  monkeypatch.setattr(_schema_bridge,'compile_pending_torch_bridges',lambda **kwargs:0)
  def model(x):
    out=torch.empty_like(x)
    return wrapper(x,out)
  x=torch.ones(4)
  baseline=vecops.precompile(model,x,capture_graph=True,graph_name='observer')
  seen=[]
  def observe(registration,arguments):
    seen.append((registration.name,tuple(tensor_metadata(x)for x in arguments)))
  result=vecops.precompile(model,x,capture_graph=True,graph_name='observer',
                          _torch_call_observer=observe)
  assert len(seen)==1 and seen[0][0]=='run'
  assert all(m.shape==(4,)for m in seen[0][1])
  assert result.graph.dumps()==baseline.graph.dumps()
  assert result.collected==result.prepared==1
  def reject(*args):raise RuntimeError('observer failed')
  with pytest.raises(RuntimeError,match='observer failed'):
    vecops.precompile(model,x,_torch_call_observer=reject)
  assert not vecops.is_precompiling()
  assert vecops.precompile(model,x).prepared==1


def test_discovery_hit_requires_all_rank_agreement(monkeypatch):
  from torch._subclasses.fake_tensor import is_fake
  observed=[]
  def model(x):
    observed.append(is_fake(x));return x+1
  class Cache:
    hit=False
    discarded=False
    saved=False
    def load(self,collector):self.hit=True;return None
    def discard(self,collector):self.hit=False;self.discarded=True
    def save(self,collector,graph):self.saved=True
    def stats(self):return dict(hit=self.hit)
  cache=Cache()
  def exchange(value):return [True,False]if isinstance(value,bool)else[value,value]
  monkeypatch.setattr(_precompile,'compile_batch',lambda requests,**kwargs:
      _precompile.PrecompileResult(len(requests),len(requests),1,0.))
  monkeypatch.setattr(_schema_bridge,'compile_pending_torch_bridges',lambda **kwargs:0)
  result=vecops.precompile(model,torch.ones(2),capture_graph=True,
      prepare_rank=0,prepare_world_size=2,prepare_barrier=lambda:None,
      prepare_exchange=exchange,_discovery_cache=cache)
  assert cache.discarded and cache.saved and not cache.hit
  assert observed==[True]and result.graph is not None
