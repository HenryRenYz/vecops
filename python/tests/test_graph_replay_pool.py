# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Pool regions are native tape events, including unwind on a kernel error."""
import ctypes
import os
import shlex
import subprocess

import pytest
import torch
import vecops
from vecops.graph import GraphCapture


@pytest.fixture(scope='module')
def pool_provider(tmp_path_factory):
  root=tmp_path_factory.mktemp('graph-native-pool')
  source=root/'provider.cpp';library=root/'provider.so'
  source.write_text(r'''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <cstdint>
#include <atomic>
#include <chrono>
#include <thread>
struct Pool { uint32_t size;uint16_t major,minor;uint64_t identity;void* context;
  void* capacity;void* inside;void* parallel_for; };
static int depth=0,begins=0,ends=0;
static std::atomic<bool> armed{false},waiting{false},released{false};
extern "C" void provider_parallel_for() {}
static Pool pool{sizeof(Pool),1,0,123,nullptr,nullptr,nullptr,(void*)&provider_parallel_for};
extern "C" const Pool* vecops_native_thread_pool_v1(uint32_t) {return &pool;}
extern "C" void vecops_native_thread_pool_begin_active(uint32_t) {
  ++depth;++begins;
  if(armed.exchange(false)) {
    waiting.store(true);
    const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(!released.load() && std::chrono::steady_clock::now()<limit)std::this_thread::yield();
    waiting.store(false);
  }
}
extern "C" void vecops_native_thread_pool_end_active() {--depth;++ends;}
extern "C" int pool_depth() {return depth;}
extern "C" int pool_begins() {return begins;}
extern "C" int pool_ends() {return ends;}
extern "C" void pool_arm() {released.store(false);armed.store(true);}
extern "C" int pool_waiting() {return waiting.load();}
extern "C" void pool_release() {released.store(true);}
''')
  compiler=shlex.split(os.environ.get('VECOPS_GRAPH_REPLAY_CXX')or os.environ.get('CXX')or'g++')
  subprocess.run(compiler+['-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
  owner=ctypes.CDLL(str(library))
  owner.vecops_native_thread_pool_v1.argtypes=[ctypes.c_uint32]
  owner.vecops_native_thread_pool_v1.restype=ctypes.c_void_p
  return owner


def test_native_pool_regions_replay_without_python_and_unwind_after_error(pool_provider,monkeypatch):
  from vecops import _C,_schema_bridge
  from vecops.graph.capture import current_capture
  owner=pool_provider
  monkeypatch.setattr(_C,'native_thread_pool_address',owner.vecops_native_thread_pool_v1)
  monkeypatch.setattr(_schema_bridge,'_native_thread_pool',None)
  def forbidden(*args):raise AssertionError('Python pool activation during replay')
  monkeypatch.setattr(_C,'native_thread_pool_begin_active',forbidden)
  monkeypatch.setattr(_C,'native_thread_pool_end_active',forbidden)
  def function(x,index):
    with vecops.native_thread_pool_region(2):
      temporary=x+1
      with vecops.native_thread_pool_region(2):
        return torch.index_select(temporary,0,index)
  capture=GraphCapture('native-pool-regions')
  capture.capture_torch(function,torch.arange(8.),torch.tensor([0,2]))
  assert current_capture()is None
  assert owner.pool_depth()==0 and owner.pool_begins()==2 and owner.pool_ends()==2
  assert sum('::pool_begin' in str(i.function._schema)for i in capture._torch_replay_tape.instructions)==2
  replay=capture.compile_torch()
  expected=torch.tensor([3.,5.])
  assert torch.equal(replay(torch.arange(8.)+2,torch.tensor([0,2])),expected)
  assert owner.pool_depth()==0 and owner.pool_begins()==4 and owner.pool_ends()==4
  with pytest.raises((RuntimeError,IndexError),match='range|bounds'):
    replay(torch.arange(8.),torch.tensor([0,100]))
  assert owner.pool_depth()==0 and owner.pool_begins()==6 and owner.pool_ends()==6
  assert torch.equal(replay(torch.arange(8.)+2,torch.tensor([0,2])),expected)
  assert owner.pool_depth()==0 and owner.pool_begins()==8 and owner.pool_ends()==8


def test_capture_failure_balances_pool_and_restores_capture_context(pool_provider,monkeypatch):
  from vecops import _C,_schema_bridge
  from vecops.graph.capture import current_capture
  monkeypatch.setattr(_C,'native_thread_pool_address',pool_provider.vecops_native_thread_pool_v1)
  monkeypatch.setattr(_schema_bridge,'_native_thread_pool',None)
  before=pool_provider.pool_begins()
  def function(x):
    with vecops.native_thread_pool_region(2):
      raise ValueError('intentional capture error')
  capture=GraphCapture('failed-pool-region')
  with pytest.raises(ValueError,match='intentional'):
    capture.capture_torch(function,torch.ones(8))
  assert current_capture()is None and pool_provider.pool_depth()==0
  assert pool_provider.pool_begins()==before+1
  assert pool_provider.pool_ends()==before+1
  with pytest.raises(RuntimeError,match='successfully'):
    capture.compile_torch()


def test_actual_native_pool_dso_can_be_bound_and_owned():
  from vecops.graph._replay_pool import _resolve_provider
  owner,begin,end=_resolve_provider(2)
  assert owner is not None and begin and end
  def function(x):
    with vecops.native_thread_pool_region(2):return x+1
  capture=GraphCapture('actual-native-pool')
  capture.capture_torch(function,torch.ones(8))
  replay=capture.compile_torch()
  assert torch.equal(replay(torch.ones(8)*2),torch.ones(8)*3)


def test_concurrent_same_and_different_graphs_are_rejected_and_recover(pool_provider,monkeypatch):
  import concurrent.futures
  import time
  from vecops import _C,_schema_bridge
  monkeypatch.setattr(_C,'native_thread_pool_address',pool_provider.vecops_native_thread_pool_v1)
  monkeypatch.setattr(_schema_bridge,'_native_thread_pool',None)
  def function(x):
    with vecops.native_thread_pool_region(2):return x+1
  first=GraphCapture('concurrent-first');first.capture_torch(function,torch.ones(8));first=first.compile_torch()
  second=GraphCapture('concurrent-second');second.capture_torch(function,torch.ones(8));second=second.compile_torch()
  pool_provider.pool_arm()
  with concurrent.futures.ThreadPoolExecutor(max_workers=1)as workers:
    future=workers.submit(first,torch.ones(8))
    try:
      deadline=time.monotonic()+2
      while not pool_provider.pool_waiting()and time.monotonic()<deadline:time.sleep(.001)
      assert pool_provider.pool_waiting(),'native interpreter must release the GIL while executing'
      with pytest.raises(RuntimeError,match='concurrent|reentrant'):first(torch.ones(8))
      with pytest.raises(RuntimeError,match='concurrent|reentrant'):first._executor.run([torch.ones(8)])
      with pytest.raises(RuntimeError,match='concurrent'):second(torch.ones(8))
    finally:
      pool_provider.pool_release()
    assert torch.equal(future.result(timeout=3),torch.ones(8)*2)
  assert pool_provider.pool_depth()==0
  assert torch.equal(first(torch.ones(8)),torch.ones(8)*2)
  assert torch.equal(second(torch.ones(8)),torch.ones(8)*2)


def test_capture_and_replay_share_native_gate_and_release_on_capture_error(pool_provider,monkeypatch):
  import concurrent.futures
  import time
  from vecops import _C,_schema_bridge
  from vecops.graph.capture import current_capture
  monkeypatch.setattr(_C,'native_thread_pool_address',pool_provider.vecops_native_thread_pool_v1)
  monkeypatch.setattr(_schema_bridge,'_native_thread_pool',None)
  def function(x):
    with vecops.native_thread_pool_region(2):return x+1
  compiled=GraphCapture('preexisting-graph');compiled.capture_torch(function,torch.ones(8));compiled=compiled.compile_torch()
  capture=GraphCapture('held-capture');another=GraphCapture('rejected-capture')
  pool_provider.pool_arm()
  with concurrent.futures.ThreadPoolExecutor(max_workers=1)as workers:
    future=workers.submit(capture.capture_torch,function,torch.ones(8))
    try:
      deadline=time.monotonic()+2
      while not pool_provider.pool_waiting()and time.monotonic()<deadline:time.sleep(.001)
      assert pool_provider.pool_waiting()
      with pytest.raises(RuntimeError,match='concurrent'):compiled(torch.ones(8))
      with pytest.raises(RuntimeError,match='concurrent'):compiled._executor.run([torch.ones(8)])
      with pytest.raises(RuntimeError,match='concurrent'):another.capture_torch(function,torch.ones(8))
    finally:pool_provider.pool_release()
    assert torch.equal(future.result(timeout=3),torch.ones(8)*2)
  assert current_capture()is None
  another.capture_torch(function,torch.ones(8))
  assert torch.equal(another.compile_torch()(torch.ones(8)),torch.ones(8)*2)
  def fail(x):raise ValueError('capture fails while holding native gate')
  with pytest.raises(ValueError,match='native gate')as retained_exception:
    GraphCapture('intentional-error').capture_torch(fail,torch.ones(8))
  assert retained_exception.value.__traceback__ is not None
  assert torch.equal(compiled(torch.ones(8)),torch.ones(8)*2)
