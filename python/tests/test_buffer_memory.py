# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared capacity, phase reuse and storage alias ownership contracts."""
import gc
import pytest
from vecops.memory import BufferRequirement as B, BufferSession, System, plan_buffers


def session(fast=0, slow=4096):
  return BufferSession(system=System(backend='system'), fast_capacity=fast,
                       slow_capacity=slow, large_pages=False)


def test_common_plan_splits_buffers_counts_worker_padding_and_half_open_lifetimes():
  p = plan_buffers([B('packed-a', 65, replicas=3, traffic_bytes=10000),
                    B('packed-b', 256), B('later', 384, lifetimes=((1,2),))],
                   fast_capacity=384, slow_capacity=1024)
  assert p['buffers']['packed-a']['size'] == 384
  assert p['buffers']['packed-a']['tier'] == 'fast'
  assert p['buffers']['packed-b']['tier'] == 'slow'
  assert p['buffers']['later']['offset'] == 0
  assert p['buffers']['later']['tier'] == 'fast'


def test_unique_leases_and_serial_groups_share_one_capacity_without_aliasing():
  with session(slow=1024) as s:
    keep = s.bind([B('output',128)])
    first = s.bind([B('temporary',256)],reuse_group='programs')
    second = s.bind([B('larger',512)],reuse_group='programs')
    assert s.stats()['slow_reserved'] == 128+512
    import numpy as np
    out=np.frombuffer(keep['output'],dtype='uint8');out.fill(31)
    tmp=np.frombuffer(second['larger'],dtype='uint8');tmp.fill(17)
    assert (out==31).all()
    assert np.frombuffer(first['temporary'],dtype='uint8')[0] == 17
    other=s.bind([B('kernel-workspace',128)],reuse_group='kernels')
    np.frombuffer(other['kernel-workspace'],dtype='uint8').fill(99)
    assert (out==31).all() and (tmp==17).all()


def test_alias_keeps_lease_and_mapping_after_session_close():
  torch=pytest.importorskip('torch')
  s=session(slow=1024)
  with s:
    owner=s.allocate(128,placement='default')
    value=torch.frombuffer(owner,dtype=torch.float32).fill_(7)
    alias=value[1:]
    del value,owner
    gc.collect()
    assert s.stats()['slow_reserved']==128
  assert torch.equal(alias,torch.full_like(alias,7))
  with pytest.raises(RuntimeError,match='closed'):
    s.allocate(64)


def test_dead_storage_lease_releases_capacity():
  with session(slow=128) as s:
    owner=s.allocate(128,placement='default')
    assert s.stats()['slow_reserved']==128
    del owner
    gc.collect()
    assert s.stats()['slow_reserved']==0
    assert s.allocate(128,placement='default').size==128


def test_invalid_specs_do_not_corrupt_session():
  with session() as s:
    for spec in [B('bad',-1),B('bad',1,alignment=3),B('bad',1,replicas=0),B('bad',1,lifetimes=((1,1),))]:
      with pytest.raises((ValueError, RuntimeError)):
        s.bind([spec])
    with pytest.raises(ValueError,match='unique'):
      s.bind([B('same',64),B('same',64)])
    assert s.stats()['slow_reserved']==0
    assert s.allocate(64,placement='default').size==64


def test_graph_internal_and_retained_buffers_use_common_budget_without_hot_planning():
  torch=pytest.importorskip('torch')
  from vecops.graph import GraphCapture
  x=torch.arange(16,dtype=torch.float32)
  def fn(x):
    out=torch.empty_like(x)
    out.copy_(x+1)
    return out
  with session(slow=1<<20) as s, torch.inference_mode():
    capture=GraphCapture('common-buffer-graph')
    capture.capture_torch(fn,x)
    exe=capture.compile_torch(native_pointwise=True,borrow_outputs=True)
    before=s.stats()['plans']
    for i in range(3):
      torch.testing.assert_close(exe(x+i),x+i+1)
    assert s.stats()['plans']==before
    assert exe.allocator_policy['buffer_placement']['retained']['slow']>0
  with torch.inference_mode(), pytest.raises(RuntimeError,match='prepared BufferSession'):
    exe(x)


def test_releasing_larger_serial_plan_shrinks_reservation():
  with session(slow=1024) as s:
    small=s.bind([B('small',128)],reuse_group='serial')
    large=s.bind([B('large',512)],reuse_group='serial')
    assert s.stats()['slow_reserved']==512
    del large
    gc.collect()
    assert s.stats()['slow_reserved']==128
    assert memoryview(small['small']).nbytes==128


def test_exhaustion_is_checked_in_release_build_and_is_transactional():
  with session(slow=128) as s:
    owner=s.allocate(128,placement='default')
    before=s.stats()
    with pytest.raises(RuntimeError,match='does not fit'):
      s.allocate(64,placement='default')
    assert s.stats()==before
    with pytest.raises(RuntimeError,match='required-fast'):
      s.allocate(64,fallback=None)
    assert memoryview(owner).nbytes==128


def test_foreign_thread_cannot_bind_or_close_a_live_session():
  from concurrent.futures import ThreadPoolExecutor
  with session(slow=256) as s:
    owner=s.allocate(128,placement='default')
    before=s.stats()
    def foreign_use():
      with pytest.raises(RuntimeError,match='execution thread'):
        s.bind([B('foreign',64)])
      with pytest.raises(RuntimeError,match='execution thread'):
        s.close()
    with ThreadPoolExecutor(max_workers=1) as pool:
      pool.submit(foreign_use).result()
    assert s.stats()==before
    assert memoryview(owner).nbytes==128


def test_tensor_planner_dynamic_leases_and_graph_capture_share_one_session():
  torch=pytest.importorskip('torch')
  import vecops
  from vecops import planner
  from vecops.graph import GraphCapture
  value=torch.arange(16.)
  def factory(x):
    out=planner.empty_like(x);out.copy_(x+1)
    return out*2
  execution=vecops.precompile(factory,value,capture_graph=True).graph
  machine=planner.MachineProfile(fast_capacity=0,slow_capacity=1<<20,
      fast_bandwidth_mib_s=100,slow_bandwidth_mib_s=10,transfer_bandwidth_mib_s=100)
  plan=planner.plan_memory(execution,machine)
  with session(slow=4<<20) as memory,torch.inference_mode():
    with planner.PlanSession(execution,plan,system=memory.system,large_pages=False,
                            enable_dynamic_factories=True) as runtime:
      held=planner.empty_like(value);held.fill_(7)
      assert runtime._dynamic_pool.allocations>0
      capture=GraphCapture('common-dynamic-factory-capture')
      capture.capture_torch(lambda x:x+held,value)
      replay=capture.compile_torch()
    # Closing the model planner and deleting the original alias must not free
    # a dynamic Storage still referenced by the executable's captured inputs.
    del held
    gc.collect()
    for step in range(3):
      torch.testing.assert_close(replay(value+step),value+step+7,rtol=0,atol=0)


def test_graph_arena_snapshots_cannot_bypass_the_shared_capacity():
  torch=pytest.importorskip('torch')
  from types import SimpleNamespace
  from vecops.graph._replay_arena import capture_leases,isolate_bindings
  with session(slow=256) as memory,torch.inference_mode():
    owner=memory.allocate(128,placement='default')
    value=torch.frombuffer(owner,dtype=torch.float32).fill_(3)
    model_session=SimpleNamespace(fast_owner=None,_fast_owner_bytes=0,
        slow_owner=owner,_slow_owner_bytes=128,_dynamic_pool=memory.tensor_pool(0),
        _buffer_session=memory,system=memory.system,domain=memory.domain,
        allow_fast_fallback=False,large_pages=False,
        plan=SimpleNamespace(machine=SimpleNamespace(alignment=64)))
    leases=capture_leases(model_session)
    isolated,owners,*_=isolate_bindings({0:value},[],leases)
    assert memory.stats()['slow_reserved']==256
    value.fill_(9)
    torch.testing.assert_close(isolated[0],torch.full_like(value,3),rtol=0,atol=0)
    with pytest.raises(RuntimeError,match='does not fit'):
      isolate_bindings({0:value},[],leases)
