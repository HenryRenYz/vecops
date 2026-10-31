# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Replay arenas stay private while the original model PlanSession is active."""
import pytest
import torch
import vecops
from vecops import graph,planner
from test_graph_replay import _factory_model,_factory_plan,_System


def test_active_session_arena_is_isolated_and_previous_state_can_feed_next_step():
  value=torch.arange(8.)
  original=vecops.precompile(_factory_model,value,capture_graph=True).graph
  plan=_factory_plan(original);system=_System()
  with planner.PlanSession(original,plan,system=system,large_pages=False,
      enable_factory_replay=True,strict_factory_replay=True) as session:
    capture=graph.GraphCapture('isolated-active-session')
    expected=capture.capture_torch(_factory_model,value)
    replay=capture.compile_torch()
    assert planner.current_session() is session
    assert replay.allocator_policy['captured_arena_copy_bytes'] > 0
    # Overwrite the old live session slot, including all its escaped aliases.
    for view in session._views.values():view.fill_(999)
    assert torch.equal(replay(value),expected)
    state=value
    for step in range(4):
      state=replay(state).clone()  # Explicit retention of static/borrowed outputs.
      assert torch.equal(state,(value+1)*(3**(step+1))+sum(3**i for i in range(1,step+1)))
    saved=replay(value).clone()
  # Closing the original session restores its parameters and destroys caches;
  # the graph's independent owner remains valid, even with old views alive.
  assert torch.equal(replay(value),saved)
  second=capture.compile_torch()
  assert torch.equal(second(value),saved)
  assert replay._bound[0].data_ptr()!=second._bound[0].data_ptr()


def test_isolated_arena_rejects_absolute_offset_through_alias_chain():
  def model(x):
    temporary=planner.empty_like(x)
    temporary.copy_(x)
    view=temporary.view(-1)
    return torch.as_strided(view,(4,),(1,),storage_offset=0) * 2
  value=torch.arange(8.)
  original=vecops.precompile(model,value,capture_graph=True).graph
  plan=_factory_plan(original)
  with planner.PlanSession(original,plan,system=_System(),large_pages=False,
      enable_factory_replay=True):
    capture=graph.GraphCapture('arena-absolute-offset')
    capture.capture_torch(model,value)
    with pytest.raises(ValueError,match='storage-offset'):
      capture.compile_torch()


def test_executable_capture_mirrors_scope_region_and_loop_without_factory_fallbacks():
  @graph.wrap_region(name='block')
  def block(x):
    temporary=planner.empty_like(x);temporary.copy_(x+1)
    return temporary*2
  def model(x):
    graph.enter_scope('outer')
    try:
      loop=graph.declare_loop('layers',2)
      try:
        for index in range(2):
          with loop.iteration(index,role='prologue' if index==0 else 'steady'):
            x=block(x)
      finally:loop.close()
      return x
    finally:graph.leave_scope()
  value=torch.ones(8)
  original=vecops.precompile(model,value,capture_graph=True).graph
  plan=_factory_plan(original)
  with planner.PlanSession(original,plan,system=_System(),large_pages=False,
      enable_factory_replay=True,strict_factory_replay=True) as session:
    capture=graph.GraphCapture('nested-active')
    expected=capture.capture_torch(model,value)
    assert session.stats()['planned_factory_allocations']==2
    assert session.stats()['factory_fallbacks']==0
    assert not session._scope and not session._loop_epochs
    replay=capture.compile_torch()
    assert torch.equal(replay(value),expected)
    assert replay.allocator_policy['captured_arena_copy_bytes']>0


def test_failed_executable_capture_restores_original_session_scope():
  value=torch.ones(8)
  original=vecops.precompile(_factory_model,value,capture_graph=True).graph
  plan=_factory_plan(original)
  with planner.PlanSession(original,plan,system=_System(),large_pages=False,
      enable_factory_replay=True) as session:
    session.enter_scope('outside')
    def fail(x):
      graph.enter_scope('unbalanced')
      raise ValueError('failure after scope enter')
    with pytest.raises(ValueError,match='failure after'):
      graph.GraphCapture('failure-scope').capture_torch(fail,value)
    assert session._scope==['outside'] and graph.current_capture() is None
    session.leave_scope()


def _state_session_model():
  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__();self.weight=torch.nn.Parameter(torch.arange(8.),requires_grad=False)
      self.mutate=False
    def forward(self,x):
      if self.mutate:self.weight.view(2,4).add_(1)
      temporary=planner.empty_like(x);temporary.copy_(x+self.weight)
      return temporary*2
  model=Model();value=torch.ones(8)
  execution=vecops.precompile(model,value,capture_graph=True).graph
  from test_graph_replay import _machine
  plan=planner.plan_memory(execution,_machine(),config=planner.PlannerConfig(
    minimum_fast_bytes=1,prefer_all_fast_when_fit=True,allow_input_fast=False,
    allow_persistent_fast=True,eager_external_replicas=True,allow_temporary_fast=True,
    only_explicit_factory_temporaries=True,require_replayable_temporaries=True))
  return model,value,execution,plan


def test_verified_immutable_state_borrow_survives_session_close_and_data_restore():
  import gc,weakref
  model,value,execution,plan=_state_session_model();system=_System()
  original=model.weight.data_ptr()
  with planner.PlanSession(execution,plan,system=system,large_pages=False,
      enable_factory_replay=True)as session:
    session.bind_module_state(model)
    bound=model.weight.data_ptr();assert bound!=original
    owner=weakref.ref(session.fast_owner)
    capture=graph.GraphCapture('immutable-borrow')
    expected=capture.capture_torch(model,value)
    replay=capture.compile_torch(borrow_readonly_state=True)
    policy=replay.allocator_policy
    assert policy['captured_arena_borrowed_state_bytes']>0
    assert 'pinned dedicated immutable planner state' in policy['captured_arena_weights']
    assert policy['captured_arena_copy_bytes']>0
    # Only temporary source slots may change while the immutable lease lives.
    state_slots={p.offset for p in plan.placements.values()if p.read_only}
    for identity,view in session._views.items():
      if plan.placements[execution.tensors[identity].storage_id].offset not in state_slots:view.fill_(999)
    assert torch.equal(replay(value),expected)
  assert model.weight.data_ptr()==original
  del capture,session,model,view;gc.collect()
  assert owner()is not None
  assert torch.equal(replay(value+1),expected+2)
  del replay;gc.collect();assert owner()is None


def test_default_state_isolation_remains_snapshot_and_alias_write_borrow_is_rejected():
  model,value,execution,plan=_state_session_model()
  with planner.PlanSession(execution,plan,system=_System(),large_pages=False,
      enable_factory_replay=True)as session:
    session.bind_module_state(model)
    capture=graph.GraphCapture('default-state-copy');expected=capture.capture_torch(model,value)
    replay=capture.compile_torch()
    model.weight.fill_(99)
    assert torch.equal(replay(value),expected)
  model,value,execution,plan=_state_session_model()
  with planner.PlanSession(execution,plan,system=_System(),large_pages=False,
      enable_factory_replay=True)as session:
    session.bind_module_state(model);model.mutate=True
    capture=graph.GraphCapture('write-immutable-alias');capture.capture_torch(model,value)
    with pytest.raises(ValueError,match='writes through an alias'):
      capture.compile_torch(borrow_readonly_state=True)


@pytest.mark.parametrize('distinct_overlapping_storage',[False,True])
def test_compaction_preserves_derived_views_and_physical_overlap(distinct_overlapping_storage):
  from vecops.graph._replay_arena import ArenaLease
  from test_graph_replay import _Owner
  owner=_Owner(4096);base=torch.frombuffer(owner,dtype=torch.float32)
  base.copy_(torch.arange(base.numel()))
  if distinct_overlapping_storage:
    first=torch.frombuffer(owner,dtype=torch.float32,count=8,offset=512)
    second=torch.frombuffer(owner,dtype=torch.float32,count=8,offset=528)
  else:first=base[:8];second=base[256:264]
  def model(x):
    first.view(2,4).add_(x)
    return second.view(2,4).t().contiguous()+3
  value=torch.tensor(1.)
  capture=graph.GraphCapture('compact-view-overlap');capture.capture_torch(model,value)
  capture._torch_replay_tape.arena_leases=[ArenaLease(owner,base.view(torch.uint8),_System(),0,'default',None,64,False,'fast')]
  replay=capture.compile_torch()
  expected=model(value).clone()
  base.fill_(999)
  assert torch.equal(replay(value),expected)
  assert replay.allocator_policy['captured_arena_copy_bytes']<=256


def test_static_views_choose_their_own_relocated_arena_component():
  from vecops.graph._replay_arena import ArenaLease
  from test_graph_replay import _Owner
  owner = _Owner(4096)
  base = torch.frombuffer(owner, dtype=torch.float32)
  base.copy_(torch.arange(base.numel()))
  first, second, third = base[:4], base[32:36], base[64:72]
  def model(x):
    return first.view(2, 2) * x + second.view(2, 2) + third.sum()
  with torch.inference_mode():
    x = torch.tensor(2.)
    capture = graph.GraphCapture('relocated-view-components')
    expected = capture.capture_torch(model, x).clone()
    capture._torch_replay_tape.arena_leases = [ArenaLease(
        owner, base.view(torch.uint8), _System(), 0, 'default', None, 64, False, 'fast')]
    program = capture.compile_torch(native_pointwise=True)
    base.fill_(999)
    torch.testing.assert_close(program(x), expected, rtol=0, atol=0)
    assert program.allocator_policy['hoisted_static_views'] == 2
    components = program.allocator_policy['captured_arena_regions'][0]['components']
    assert len(components) == 3
    assert any(row['source_offset_bytes'] != row['private_offset_bytes'] for row in components)


@pytest.mark.parametrize('observe',[
  lambda x:x.untyped_storage(),lambda x:x.storage(),lambda x:x.data_ptr(),
  lambda x:torch._C._is_alias_of(x,x)])
def test_storage_capacity_pointer_and_identity_observations_are_rejected(observe):
  def model(x):observe(x);return x+1
  with pytest.raises(ValueError,match='tensor data escape'):
    graph.GraphCapture('storage-observation').capture_torch(model,torch.ones(4))


def test_capture_profile_guard_restores_on_escape_and_chains_python_profile():
  import sys,cProfile
  previous=sys.getprofile();events=[]
  def callback(frame,event,argument):events.append(event)
  try:
    sys.setprofile(callback)
    capture=graph.GraphCapture('chain-profile');capture.capture_torch(lambda x:x+1,torch.ones(4))
    assert sys.getprofile()is callback and events
    with pytest.raises(ValueError,match='storage identity'):
      graph.GraphCapture('escape-profile').capture_torch(lambda x:(torch._C._is_alias_of(x,x),x+1)[1],torch.ones(4))
    assert sys.getprofile()is callback
  finally:sys.setprofile(previous)
  profiler=cProfile.Profile();profiler.enable()
  active=sys.getprofile()
  try:
    with pytest.raises(RuntimeError,match='active C profiler'):
      graph.GraphCapture('c-profiler').capture_torch(lambda x:x+1,torch.ones(4))
    assert sys.getprofile()is active
  finally:profiler.disable()
  assert graph.current_capture()is None


def test_storage_observation_guard_does_not_hide_explicit_factory_metadata():
  capture=graph.GraphCapture('factory-source-metadata')
  capture.capture_torch(_factory_model,torch.ones(8))
  factories=[event for event in capture._torch_execution_graph.events
             if event.attributes.get('explicit_factory')]
  assert len(factories)==1
  assert factories[0].attributes['source']['function']=='_factory_model'


@pytest.mark.parametrize('empty', [False, True])
def test_relative_arena_view_preserves_nonzero_offset_alias_and_source_lifetime(empty):
  value=torch.arange(12.)
  def model(x):
    temporary=planner.empty_like(x);temporary.copy_(x+1)
    source=temporary[2:10]
    shape=(0,2)if empty else(4,2)
    result=source.as_strided(shape,(1,4))
    result.add_(3)
    return result,temporary
  execution=vecops.precompile(model,value,capture_graph=True).graph
  with planner.PlanSession(execution,_factory_plan(execution),system=_System(),
      large_pages=False,enable_factory_replay=True,enable_dynamic_factories=True)as session:
    capture=graph.GraphCapture('relative-nonzero-arena-offset')
    expected=tuple(t.clone()for t in capture.capture_torch(model,value))
    replay=capture.compile_torch()
    assert replay.allocator_policy['captured_arena_copy_bytes']>0
    assert session._dynamic_pool is not None
    torch.frombuffer(session._dynamic_pool.owner,dtype=torch.uint8).fill_(239)
    actual=replay(value)
    assert torch.equal(actual[0],expected[0])and torch.equal(actual[1],expected[1])
    assert torch._C._is_alias_of(*actual)
    saved=tuple(t.clone()for t in actual)
  assert all(torch.equal(a,b)for a,b in zip(replay(value),saved))
  for step in range(3):
    actual=replay(value+step)
    # Re-entering the original function without a session is a valid reference.
    reference=model(value+step)
    assert all(torch.equal(a,b)for a,b in zip(actual,reference))


@pytest.mark.parametrize('transpose', [False, True])
def test_dynamic_tensor_pool_relative_view_has_complete_retained_extent(transpose):
  value=torch.arange(8.).reshape(2,4)
  if transpose:value=value.t()
  def model(x):
    temporary=planner.empty_like(x);temporary.copy_(x+1);return temporary*2
  execution=vecops.precompile(model,value,capture_graph=True).graph
  with planner.PlanSession(execution,_factory_plan(execution),system=_System(),
      large_pages=False,enable_dynamic_factories=True)as session:
    capture=graph.GraphCapture('pool-relative-view')
    expected=capture.capture_torch(model,value)
    replay=capture.compile_torch()
    assert session.stats()['dynamic_allocations']==1
    assert session.stats()['dynamic_misses']==0
    for delta in (0,1,2):assert torch.equal(replay(value+delta),expected+2*delta)
  assert torch.equal(replay(value),expected)


def test_relative_view_cannot_extend_beyond_retained_source_span():
  value=torch.arange(8.)
  def model(x):
    temporary=planner.empty_like(x);temporary.copy_(x)
    return temporary[:4].as_strided((8,),(1,))
  execution=vecops.precompile(model,value,capture_graph=True).graph
  with planner.PlanSession(execution,_factory_plan(execution),system=_System(),
      large_pages=False,enable_factory_replay=True,enable_dynamic_factories=True):
    capture=graph.GraphCapture('relative-span-escape')
    capture.capture_torch(model,value)
    with pytest.raises(ValueError,match='storage-offset.*node'):
      capture.compile_torch()


def test_empty_relative_alias_cannot_be_expanded_to_read_unretained_bytes():
  value=torch.arange(8.)
  def model(x):
    temporary=planner.empty_like(x);temporary.copy_(x)
    empty=temporary.as_strided((0,),(1,))
    return empty.as_strided((4,),(1,))
  execution=vecops.precompile(model,value,capture_graph=True).graph
  with planner.PlanSession(execution,_factory_plan(execution),system=_System(),
      large_pages=False,enable_dynamic_factories=True):
    capture=graph.GraphCapture('empty-relative-expand')
    capture.capture_torch(model,value)
    with pytest.raises(ValueError,match='storage-offset'):
      capture.compile_torch()


def test_relative_view_lowered_with_new_explicit_memory_plan():
  value=torch.arange(12.)
  def model(x):
    temporary=planner.empty_like(x);temporary.copy_(x+1)
    result=temporary[2:10].as_strided((4,2),(1,4));result.add_(3)
    return result*2
  capture=graph.GraphCapture('new-plan-relative-view')
  expected=capture.capture_torch(model,value)
  plan=_factory_plan(capture._torch_execution_graph)
  replay=capture.compile_torch(memory_plan=plan,system=_System(),large_pages=False)
  assert replay.planned_factory_count==1
  for delta in (0,1,2):assert torch.equal(replay(value+delta),expected+2*delta)


def test_inplace_relative_view_metadata_change_stays_rejected():
  def model(x):
    temporary=x.clone();temporary.as_strided_((4,2),(1,4));return temporary
  with pytest.raises(ValueError,match='metadata-mutating'):
    graph.GraphCapture('inplace-relative-view').capture_torch(model,torch.arange(8.))
