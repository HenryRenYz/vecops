# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Borrowed resident plans preserve the consumer's output access contract."""
from types import SimpleNamespace
import pytest
import torch
from vecops import _schema_bridge
from vecops.distributed import _resident


@pytest.mark.parametrize('access',[2,3])
def test_borrowed_output_preserves_declared_access(monkeypatch,access):
  monkeypatch.delenv('VECOPS_GRAPH_RESIDENT_COMM',raising=False)
  providers={}
  monkeypatch.setattr(_schema_bridge,'register_replay_plan_provider',lambda name,fn:providers.update({name:fn}))
  lib=torch.library.Library('vecops_test_resident_access_'+str(access),'DEF')
  lib.define('consumer(Tensor addresses, Tensor(a!) output) -> ()')
  addresses=torch.empty(2,dtype=torch.int64)
  nested=SimpleNamespace(abi_address=10)
  monkeypatch.setattr(_schema_bridge,'prepare_replay_plan',lambda *args,**kwargs:(nested,[1,access],[0,0],[]))
  module=SimpleNamespace(resident_consumer_info=lambda identity:(lib.ns+'::consumer',addresses,[]),
                         resident_plan=lambda *args:SimpleNamespace(abi_address=11))
  _resident.install(module)
  metadata=SimpleNamespace(shape=(2,3),strides=(3,1),dtype='torch.bfloat16',device='cpu')
  lease,flags,_,_=providers['vecops_comm::consume'](
      SimpleNamespace(_schema=SimpleNamespace(name='vecops_comm::consume')),
      (17,object(),object()),lambda ref:metadata)
  assert flags==[access,1]
  assert lease.consumer[0] is nested
