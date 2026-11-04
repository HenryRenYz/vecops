# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fixed-layout cache promotion: ownership, aliases, effects and no hot allocation."""
import threading
import pytest
import torch
from vecops.graph import GraphCapture
from vecops.planner.buffers import _current


class Buffer(bytearray):
  @property
  def size(self):return len(self)


class Session:
  domain=0
  thread_id=threading.get_ident()
  def __init__(self):self.requests=[];self.available=1<<20
  def stats(self):return dict(fast_capacity=1<<20,largest_fast_free=self.available)
  def bind(self,requirements,**kwargs):
    result={}
    for request in requirements:
      buffer=Buffer(request.bytes)
      buffer.tier='fast' if request.placement=='required' else 'slow'
      buffer.offset=len(self.requests)*64
      if buffer.tier=='fast':self.available-=request.bytes
      self.requests.append(request)
      result[request.name]=buffer
    return result


@pytest.mark.parametrize('mode',['readonly','write','unused','external_alias','no_capacity'])
def test_cache_transfer_preserves_layout_and_is_prepared_once(monkeypatch,mode):
  monkeypatch.setenv('VECOPS_THREAD_POOL','native')
  torch.set_num_threads(4)
  session=Session();token=_current.set(session)
  try:
    with torch.inference_mode():
      x=torch.arange(37.)
      capture=GraphCapture('cache-producer')
      capture.capture_torch(lambda x:(x+1,(x+2).view(37)),x)
      producer=capture.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
      values=producer(x)
      original=tuple(v.data_ptr() for v in values)
      steady=GraphCapture('cache-consumer')
      def function(x,a,b):
        if mode=='write':a.add_(1)
        if mode=='unused':return x+3
        return x+a+b
      arguments=(values[0] if mode=='external_alias' else x,*values)
      steady.capture_torch(function,*arguments)
      consumer=steady.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
      if mode=='no_capacity':session.available=0
      transfer=producer.prepare_output_transfer(producer.fixed_output_values(values),consumer,
          input_offset=1,expected_replays=200)
      if mode in {'unused','no_capacity'}:
        assert transfer is None
        return
      assert transfer is not None
      assert set(transfer.replacements)==({1} if mode in {'write','external_alias'} else {0,1})
      count=len(session.requests)
      for repeat in range(3):
        values=producer(x+repeat)
        moved=transfer.replay(values)
        assert tuple(v.data_ptr() for v in values)==original
        for old,new in zip(values,moved):
          assert old.untyped_storage().nbytes()==new.untyped_storage().nbytes()
          assert old.storage_offset()==new.storage_offset()
          torch.testing.assert_close(old,new,rtol=0,atol=0)
        if mode=='readonly':
          torch.testing.assert_close(consumer(x,*moved),x+2*(x+repeat)+3,rtol=0,atol=0)
      assert len(session.requests)==count
      assert transfer.calls==3
  finally:_current.reset(token)


def test_cache_transfer_keeps_complete_storage_and_nonzero_offset(monkeypatch):
  monkeypatch.setenv('VECOPS_THREAD_POOL','native')
  session=Session();token=_current.set(session)
  try:
    with torch.inference_mode():
      def initialize(x):
        cache=x+1
        return cache,cache[5:]
      x=torch.arange(131109.)  # Multiple 64 KiB tasks plus a vector tail.
      capture=GraphCapture('cache-alias-producer')
      capture.capture_torch(initialize,x)
      producer=capture.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
      values=producer(x)
      capture=GraphCapture('cache-alias-consumer')
      capture.capture_torch(lambda a,b:a[5:]+b,*values)
      consumer=capture.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
      assert producer.prepare_output_transfer(values,consumer,input_offset=0,expected_replays=1) is None
      transfer=producer.prepare_output_transfer(values,consumer,input_offset=0,expected_replays=200)
      assert transfer is not None and len(transfer.rows)==1
      moved=transfer.replay(values)
      assert moved[0].untyped_storage()._cdata==moved[1].untyped_storage()._cdata
      assert moved[1].storage_offset()==values[1].storage_offset()==5
      torch.testing.assert_close(consumer(*moved),2*(x[5:]+1),rtol=0,atol=0)
      with pytest.raises(RuntimeError,match='source address changed'):
        transfer.replay(tuple(v.clone() for v in values))
  finally:_current.reset(token)


def test_transfer_kernel_cache_tracks_fresh_team_configuration(monkeypatch):
  from vecops._arm_vector_state import get_vector_state
  from vecops.graph._output_transfer import _copy_kernel
  monkeypatch.setenv('VECOPS_THREAD_POOL','native')
  monkeypatch.setenv('VECOPS_NUM_THREADS','4')
  torch.set_num_threads(4)
  session=Session();token=_current.set(session)
  try:
    with torch.inference_mode():
      x=torch.arange(131109.)
      capture=GraphCapture('cache-width-producer')
      capture.capture_torch(lambda x:x+1,x)
      producer=capture.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
      values=(producer(x),)
      capture=GraphCapture('cache-width-consumer')
      capture.capture_torch(lambda x:x+1,*values)
      consumer=capture.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
      vector=get_vector_state().env_token
      four=_copy_kernel(4,vector)
      monkeypatch.setenv('VECOPS_NUM_THREADS','2')
      # Execution width is independent of the Torch/compiler job count.
      assert torch.get_num_threads()==4
      transfer=producer.prepare_output_transfer(values,consumer,input_offset=0,expected_replays=200)
      assert _copy_kernel(2,vector) is not four
      leases=[owner for owner in transfer.executable._owners
              if hasattr(owner,'workers') and hasattr(owner,'grids')]
      assert leases and {owner.workers for owner in leases}=={2}
      moved=transfer.replay(values)
      torch.testing.assert_close(consumer(*moved),x+2,rtol=0,atol=0)
  finally:_current.reset(token)


@pytest.mark.parametrize('bound_alias',[False,True])
def test_overlapping_distinct_storages_and_bound_state_are_not_partially_moved(monkeypatch,bound_alias):
  monkeypatch.setenv('VECOPS_THREAD_POOL','native')
  monkeypatch.setenv('VECOPS_NUM_THREADS','4')
  torch.set_num_threads(4)
  session=Session();token=_current.set(session)
  try:
    with torch.inference_mode():
      x=torch.arange(37.)
      capture=GraphCapture('cache-overlap-producer')
      capture.capture_torch(lambda x:x+1,x)
      producer=capture.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
      cache=producer(x)
      owner=next(iter(producer._retained_output_owner.buffers.values()))
      alias=torch.frombuffer(owner,dtype=torch.float32,count=cache.numel())
      assert alias.untyped_storage()._cdata!=cache.untyped_storage()._cdata
      assert alias.data_ptr()==cache.data_ptr()
      capture=GraphCapture('cache-overlap-consumer')
      if bound_alias:
        class Consumer(torch.nn.Module):
          def __init__(self):
            super().__init__();self.register_buffer('alias',alias)
          def forward(self,a):return a+self.alias
        values=(cache,)
        capture.capture_torch(Consumer(),*values)
      else:
        values=(cache,alias)
        capture.capture_torch(lambda a,b:a+b,*values)
      consumer=capture.compile_torch(native_pointwise=True,native_donation=True,
          borrow_outputs=True,borrow_readonly_state=True)
      assert producer.prepare_output_transfer(producer.fixed_output_values(values),consumer,
          input_offset=0,expected_replays=200) is None
      torch.testing.assert_close(consumer(*values),2*(x+1),rtol=0,atol=0)
  finally:_current.reset(token)
