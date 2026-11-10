# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Explicit resource-backed cache copies: aliases, ownership and cold preparation."""
import gc
import weakref

import pytest
import torch

from vecops.graph import GraphCapture
from vecops.memory import Runtime, native as m, use_runtime
from vecops.planner import BufferRequirement, GraphMemoryCompiler


def value(name, **fields):
  result = getattr(m, name)()
  for key, item in fields.items():
    setattr(result, key, item)
  return result


@pytest.fixture
def runtime(monkeypatch):
  graph = value('ResourceGraph',
      agents=[value('Agent', id='executor', capabilities=['host_load_store'], address_space='host')],
      pools=[value('MemoryPool', id='opaque-bank', capacity_bytes=16<<20, address_space='host',
          granularity_bytes=4096, page_sizes_bytes=[4096], capacity_group='capacity')],
      capacity_groups=[value('CapacityGroup', id='capacity', capacity_bytes=16<<20)],
      access=[value('AccessConnection', id='access', rule='opaque', agent='executor', pool='opaque-bank',
          access=[m.AccessMode.read, m.AccessMode.write], mapping=m.MappingMode.direct)])
  discovery = value('LinuxDiscovery', graph=graph,
      scope=value('Scope', executable_agents=['executor'], allocatable_pools=['opaque-bank'],
          accessible_pools=['opaque-bank'], execution_slots={'executor':4}),
      pool_bindings=[value('LinuxPoolBinding', pool='opaque-bank', numa_nodes=[])],
      pool_available_bytes={'opaque-bank':16<<20})
  runtime = m.make_linux_runtime(discovery, value('LinuxRuntimeOptions', participant='copy-test'))
  threads = torch.get_num_threads()
  torch.set_num_threads(1)
  monkeypatch.setenv('VECOPS_NUM_THREADS', '1')
  monkeypatch.setenv('VECOPS_THREAD_POOL', 'native')
  try:
    with use_runtime(Runtime(runtime)):
      yield runtime
  finally:
    torch.set_num_threads(threads)


def committed(runtime):
  gc.collect()
  view = runtime.view(runtime.current_scope())
  return sum(group.committed_bytes for group in runtime.snapshot(view).groups)


def capture(function, arguments, runtime, name):
  graph = GraphCapture(name)
  graph.capture_torch(function, *arguments)
  return graph.compile_torch(runtime=runtime, native_pointwise=True,
      native_donation=True, borrow_outputs=True, borrow_readonly_state=True)


def targets(runtime, values, *, materialize=True):
  """Plan complete destination storages, with one identity for all aliases."""
  identities, bindings, requirements = {}, {}, []
  for index, tensor in enumerate(values):
    storage = tensor.untyped_storage()
    if storage._cdata not in identities:
      key = 'target:' + str(len(identities))
      identities[storage._cdata] = key
      requirements.append(BufferRequirement(key, storage.nbytes(), retained=True,
          allowed_pools=('opaque-bank',), independent_mapping=True))
    bindings[index] = identities[storage._cdata]
  view = runtime.view(runtime.current_scope())
  program = GraphMemoryCompiler(runtime, view, runtime.snapshot(view), native_module=m).compile(requirements)
  prepared = program.prepare(runtime, view)
  if materialize:
    prepared.materialize()
  return prepared, bindings


def guard_program(runtime, *, reused=False, remapped=False, read_only=False):
  """Tiny real native recipes; remapped uses two VAs for the same backing page."""
  from dataclasses import replace
  from types import MappingProxyType
  view = runtime.view(runtime.current_scope())
  requirements = [BufferRequirement(key,4096,allowed_pools=('opaque-bank',),
      independent_mapping=True,lifetimes=((index,index+1),) if reused else ((0,2),))
      for index,key in enumerate(('a','b'))]
  compiled = GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=m).compile(requirements)
  if remapped:
    recipe = compiled.memory
    first = compiled.bindings['a']
    slot = next(slot for slot in recipe.addresses.slots if slot.id == first.slot)
    mapping = next(item for item in recipe.mappings if item.region == slot.region)
    placement = runtime.resolve_placement(view,value('PlacementPlan',address_space='host',
        stripe_bytes=4096,weights={'opaque-bank':1}),4096,m.BackingMode.shareable_remappable,
        [m.AccessMode.read,m.AccessMode.write])
    layout = recipe.addresses
    layout.regions = [*layout.regions,value('VirtualRegion',id='alias',address_space='host',
        bytes=4096,page_bytes=4096)]
    layout.slots = [*layout.slots,value('AddressSlot',id='alias',region='alias',offset_bytes=0,
        size_bytes=4096,placement=placement)]
    recipe.addresses = layout
    recipe.mappings = [*recipe.mappings,value('PageMapping',region='alias',virtual_offset_bytes=0,
        backing=mapping.backing,backing_offset_bytes=mapping.backing_offset_bytes+first.view_offset_bytes,
        size_bytes=4096,access=[m.AccessMode.read,m.AccessMode.write])]
    bindings = dict(compiled.bindings)
    bindings['b'] = replace(bindings['b'],slot='alias',view_offset_bytes=0)
    compiled = replace(compiled,_memory_toml=m.recipe_to_toml(recipe),bindings=MappingProxyType(bindings))
  if read_only:
    recipe = compiled.memory
    mappings = recipe.mappings
    for mapping in mappings:
      mapping.access = [m.AccessMode.read]
    recipe.mappings = mappings
    compiled = replace(compiled,_memory_toml=m.recipe_to_toml(recipe))
  prepared = compiled.prepare(runtime,view)
  prepared.materialize()
  return prepared


def guard_endpoints(program, *, extra=(), bound=()):
  """Exercise copy policy with actual resource views, without compiling an op."""
  from types import SimpleNamespace as NS
  buffers = {key:program.tensor(key,(4096,),torch.uint8) for key in ('a','b')}
  values = tuple(buffer.view(torch.float32) for buffer in buffers.values())
  producer = NS(_retained_output_owner=NS(buffers=buffers,prepared=program),_protected_storages=set())
  def consumer(selected):
    inputs = (*selected,*extra)
    return NS(graph=NS(events=[],tensors={str(index):NS(storage_id=str(index)) for index in range(len(inputs))}),
        _input_refs=tuple(NS(id=str(index)) for index in range(len(inputs))),_protected_storages=set(),
        _input_storage_spans=tuple((x.untyped_storage().data_ptr(),x.untyped_storage().nbytes()) for x in inputs),
        _bound_storage_spans=tuple((x.untyped_storage().data_ptr(),x.untyped_storage().nbytes()) for x in bound))
  return producer,values,consumer


@pytest.mark.parametrize('mode',['same_key','reused_slot','remapped_backing'])
def test_copy_guard_rejects_overlapping_targets_before_jit(runtime,monkeypatch,mode):
  from vecops.graph import _output_transfer as transfer
  source = guard_program(runtime)
  destination = guard_program(runtime,reused=mode=='reused_slot',remapped=mode=='remapped_backing')
  producer,values,consumer = guard_endpoints(source)
  bindings = {0:'a',1:'a' if mode=='same_key' else 'b'}
  if mode=='remapped_backing':
    assert destination.tensor('a',(4096,),torch.uint8).data_ptr() != destination.tensor('b',(4096,),torch.uint8).data_ptr()
  def forbidden(*args):
    raise AssertionError('invalid copy reached JIT')
  monkeypatch.setattr(transfer,'_copy_kernel',forbidden)
  with pytest.raises(ValueError,match='physically independent targets'):
    transfer.prepare(producer,values,consumer(values),input_offset=0,
        target_program=destination,target_bindings=bindings)


def test_copy_guard_rejects_target_remapped_over_source(runtime,monkeypatch):
  from vecops.graph import _output_transfer as transfer
  program = guard_program(runtime,remapped=True)
  producer,values,consumer = guard_endpoints(program)
  assert values[0].data_ptr() != values[1].data_ptr()
  monkeypatch.setattr(transfer,'_copy_kernel',lambda *args:pytest.fail('invalid copy reached JIT'))
  with pytest.raises(ValueError,match='aliases a live source'):
    transfer.prepare(producer,values[:1],consumer(values[:1]),input_offset=0,
        target_program=program,target_bindings={0:'b'})


def test_copy_guard_rejects_readonly_target_before_jit(runtime,monkeypatch):
  from vecops.graph import _output_transfer as transfer
  source = guard_program(runtime)
  destination = guard_program(runtime,read_only=True)
  producer,values,consumer = guard_endpoints(source)
  monkeypatch.setattr(transfer,'_copy_kernel',lambda *args:pytest.fail('invalid copy reached JIT'))
  with pytest.raises(ValueError,match='writable physical binding'):
    transfer.prepare(producer,values[:1],consumer(values[:1]),input_offset=0,
        target_program=destination,target_bindings={0:'a'})


def test_copy_guard_protects_unselected_canonical_outputs(runtime,monkeypatch):
  from vecops.graph import _output_transfer as transfer
  program = guard_program(runtime)
  producer,values,consumer = guard_endpoints(program)
  monkeypatch.setattr(transfer,'_copy_kernel',lambda *args:pytest.fail('invalid copy reached JIT'))
  with pytest.raises(ValueError,match='aliases a live source'):
    transfer.prepare(producer,values,consumer(values[:1]),input_offset=0,
        target_program=program,target_bindings={0:'b'})


def test_copy_guard_protects_producer_bound_state(runtime,monkeypatch):
  from vecops.graph import _output_transfer as transfer
  source = guard_program(runtime)
  destination = guard_program(runtime,remapped=True)
  producer,values,consumer = guard_endpoints(source)
  state = destination.tensor('b',(4096,),torch.uint8)
  producer._bound_storage_spans = ((state.untyped_storage().data_ptr(),state.untyped_storage().nbytes()),)
  monkeypatch.setattr(transfer,'_copy_kernel',lambda *args:pytest.fail('invalid copy reached JIT'))
  with pytest.raises(ValueError,match='aliases a live source'):
    transfer.prepare(producer,values[:1],consumer(values[:1]),input_offset=0,
        target_program=destination,target_bindings={0:'a'})


@pytest.mark.parametrize('bound',[False,True])
def test_copy_guard_rejects_target_remapped_over_other_consumer_operand(runtime,monkeypatch,bound):
  from vecops.graph import _output_transfer as transfer
  source = guard_program(runtime)
  destination = guard_program(runtime,remapped=True)
  alias = destination.tensor('b',(4096,),torch.uint8)
  assert alias.data_ptr() != destination.tensor('a',(4096,),torch.uint8).data_ptr()
  producer,values,consumer = guard_endpoints(source,extra=() if bound else (alias,),bound=(alias,) if bound else ())
  monkeypatch.setattr(transfer,'_copy_kernel',lambda *args:pytest.fail('invalid copy reached JIT'))
  with pytest.raises(ValueError,match='aliases a live source or consumer operand'):
    transfer.prepare(producer,values[:1],consumer(values[:1]),input_offset=0,
        target_program=destination,target_bindings={0:'a'})


def test_copy_guard_accepts_complete_disjoint_receipts_and_shared_source_aliases(runtime):
  from vecops.graph._output_transfer import _check_targets, _physical_ranges
  source = guard_program(runtime)
  destination = guard_program(runtime)
  producer,values,consumer = guard_endpoints(source)
  storage = values[0].untyped_storage()
  groups = [dict(storage=storage,key='a',indices=[0,1])]
  aliases = (values[0],values[0][5:])
  _check_targets(destination,groups,[destination.tensor('a',(4096,),torch.uint8)],consumer(aliases),0,
      values=aliases,producer=producer)
  # Unknown backend identity cannot serve as an independence certificate.
  descriptor = value('MemoryDescriptor',spans=[value('ResolvedMemorySpan',offset_bytes=0,
      size_bytes=4096,physical_identity=None,physical_offset_bytes=0)])
  with pytest.raises(ValueError,match='physical independence receipts'):
    _physical_ranges(descriptor,0,4096)


def test_explicit_readonly_copy_preserves_layout_without_hot_allocation(runtime, monkeypatch):
  with torch.inference_mode():
    x = torch.arange(37.)
    producer = capture(lambda x:(x+1, (x+2).view(37)), (x,), runtime, 'copy-producer')
    values = producer(x)
    original = tuple(t.data_ptr() for t in values)
    consumer = capture(lambda x,a,b:x+a+b, (x,*values), runtime, 'copy-consumer')
    destination, bindings = targets(runtime, values)
    transfer = producer.prepare_output_transfer(producer.fixed_output_values(values), consumer,
        input_offset=1, target_program=destination, target_bindings=bindings)
    assert set(transfer.replacements) == {0,1}
    before = committed(runtime)
    def forbidden(*args, **kwargs):
      raise AssertionError('copy replay attempted preparation or tensor allocation')
    with monkeypatch.context() as guard:
      guard.setattr(GraphCapture, 'capture_torch', forbidden)
      guard.setattr(GraphCapture, 'compile_torch', forbidden)
      guard.setattr(GraphMemoryCompiler, 'compile', forbidden)
      guard.setattr(destination, 'tensor', forbidden)
      for repeat in range(3):
        values = producer(x+repeat)
        moved = transfer.replay(values)
        assert tuple(t.data_ptr() for t in values) == original
        for index, (old,new) in enumerate(zip(values,moved)):
          assert old.untyped_storage().nbytes() == new.untyped_storage().nbytes()
          assert old.storage_offset() == new.storage_offset()
          assert new.data_ptr() == transfer.replacements[index].data_ptr()
          torch.testing.assert_close(old,new,rtol=0,atol=0)
        torch.testing.assert_close(consumer(x,*moved), x+2*(x+repeat)+3,rtol=0,atol=0)
    assert committed(runtime) == before
    assert transfer.calls == 3


@pytest.mark.parametrize('mode', ['write', 'external_alias'])
def test_explicit_copy_rejects_consumer_writes_or_unredirected_alias(runtime, mode):
  with torch.inference_mode():
    x = torch.arange(37.)
    producer = capture(lambda x:(x+1,x+2), (x,), runtime, 'protected-copy-producer')
    values = producer(x)
    def consume(x,a,b):
      if mode == 'write':
        a.add_(1)
      return x+a+b
    consumer = capture(consume, (values[0] if mode=='external_alias' else x,*values),
        runtime, 'protected-copy-consumer')
    destination, bindings = targets(runtime, values)
    with pytest.raises(ValueError, match='modify|retain|alias'):
      producer.prepare_output_transfer(values, consumer, input_offset=1,
          target_program=destination, target_bindings=bindings)


def test_copy_keeps_complete_storage_and_nonzero_alias_offset(runtime):
  with torch.inference_mode():
    def initialize(x):
      cached = x+1
      return cached, cached[5:]
    x = torch.arange(131109.)
    producer = capture(initialize, (x,), runtime, 'offset-copy-producer')
    values = producer(x)
    consumer = capture(lambda a,b:a[5:]+b, values, runtime, 'offset-copy-consumer')
    destination, bindings = targets(runtime, values)
    with pytest.raises(ValueError, match='every cache input alias'):
      producer.prepare_output_transfer(values, consumer, input_offset=0,
          target_program=destination, target_bindings={0:bindings[0]})
    transfer = producer.prepare_output_transfer(values, consumer, input_offset=0,
        target_program=destination, target_bindings=bindings)
    assert all(tensor.device.type == 'cpu' for tensor in transfer.replacements.values())
    assert len(transfer.rows) == 1
    moved = transfer.replay(values)
    assert moved[0].untyped_storage()._cdata == moved[1].untyped_storage()._cdata
    assert moved[1].storage_offset() == values[1].storage_offset() == 5
    torch.testing.assert_close(consumer(*moved),2*(x[5:]+1),rtol=0,atol=0)
    with pytest.raises(RuntimeError, match='source address changed'):
      transfer.replay(tuple(t.clone() for t in values))


def test_copy_requires_explicit_materialized_target_and_bindings(runtime):
  with torch.inference_mode():
    x = torch.arange(7.)
    producer = capture(lambda x:x+1, (x,), runtime, 'target-required-producer')
    values = (producer(x),)
    consumer = capture(lambda x:x+1, values, runtime, 'target-required-consumer')
    with pytest.raises(TypeError, match='target_program'):
      producer.prepare_output_transfer(values,consumer,input_offset=0)
    with pytest.raises(ValueError, match='explicit materialized target program'):
      producer.prepare_output_transfer(values,consumer,input_offset=0,
          target_program=None,target_bindings={0:'target'})
    destination, bindings = targets(runtime, values, materialize=False)
    with pytest.raises(ValueError, match='must be materialized'):
      producer.prepare_output_transfer(values,consumer,input_offset=0,
          target_program=destination,target_bindings=bindings)
    destination.materialize()
    with pytest.raises(ValueError, match='explicit target buffer bindings'):
      producer.prepare_output_transfer(values,consumer,input_offset=0,
          target_program=destination,target_bindings={})
    with pytest.raises(TypeError, match='expected_replays'):
      producer.prepare_output_transfer(values,consumer,input_offset=0,
          target_program=destination,target_bindings=bindings,expected_replays=200)


def test_copy_and_escaped_alias_retain_actual_target_binding(runtime):
  with torch.inference_mode():
    x = torch.arange(37.)
    producer = capture(lambda x:x+1,(x,),runtime,'owner-copy-producer')
    values = (producer(x),)
    consumer = capture(lambda x:x+1,values,runtime,'owner-copy-consumer')
    before = committed(runtime)
    destination, bindings = targets(runtime,values)
    target_bytes = sum(root.bytes for root in destination.recipe.backings)
    reference = weakref.ref(destination)
    transfer = producer.prepare_output_transfer(values,consumer,input_offset=0,
        target_program=destination,target_bindings=bindings)
    escaped = transfer.replay(values)[0][5:]
    # Closing the Python Program cannot unmap the actual binding pinned by
    # Tensor Storage. The transfer also retains the destination Program itself.
    destination.close()
    del destination
    gc.collect()
    assert reference() is not None
    del transfer
    gc.collect()
    assert reference() is None
    assert committed(runtime) >= before + target_bytes
    torch.testing.assert_close(escaped,x[5:]+1,rtol=0,atol=0)
    del escaped
    assert committed(runtime) == before


@pytest.mark.parametrize('bound_alias', [False,True])
def test_copy_rejects_overlapping_distinct_storage_or_bound_state(runtime,bound_alias):
  with torch.inference_mode():
    x = torch.arange(37.)
    producer = capture(lambda x:x+1,(x,),runtime,'physical-alias-copy-producer')
    cached = producer(x)
    owner = next(iter(producer._retained_output_owner.buffers.values()))
    alias = torch.frombuffer(owner.numpy(),dtype=torch.float32,count=cached.numel())
    assert alias.untyped_storage()._cdata != cached.untyped_storage()._cdata
    assert alias.data_ptr() == cached.data_ptr()
    if bound_alias:
      class Consumer(torch.nn.Module):
        def __init__(self):
          super().__init__();self.register_buffer('alias',alias)
        def forward(self,value):
          return value+self.alias
      values = (cached,)
      consumer = capture(Consumer(),values,runtime,'bound-alias-copy-consumer')
    else:
      values = (cached,alias)
      consumer = capture(lambda a,b:a+b,values,runtime,'physical-alias-copy-consumer')
    destination, bindings = targets(runtime,(cached,))
    with pytest.raises(ValueError,match='alias|retain'):
      producer.prepare_output_transfer(values,consumer,input_offset=0,
          target_program=destination,target_bindings=bindings)


@pytest.mark.parametrize('provider_workers', [1,2])
def test_copy_kernel_cache_tracks_fresh_execution_width(runtime,monkeypatch,provider_workers):
  import ctypes
  from vecops import _C, _schema_bridge
  from vecops._arm_vector_state import get_vector_state
  from vecops.graph._output_transfer import _copy_kernel
  previous_threads = torch.get_num_threads()
  previous_budget = _C.cpu_thread_limit(provider_workers)
  try:
    torch.set_num_threads(provider_workers)
    # The ordinary native bridge factory is first-call-fixed. Select the public
    # adaptive provider explicitly, and certify its actual capacity independently
    # of TransferWorkers=2. A one-lane provider must clamp this request to one.
    monkeypatch.setattr(_schema_bridge,'_native_thread_pool',
        ctypes.c_void_p(_C.cpu_thread_pool_address()))
    monkeypatch.setenv('VECOPS_NUM_THREADS','2')
    assert torch.get_num_threads() == provider_workers
    assert _C.cpu_thread_pool_capacity() == provider_workers
    with torch.inference_mode():
      x = torch.arange(131109.)
      producer = capture(lambda x:x+1,(x,),runtime,'width-copy-producer')
      values = (producer(x),)
      consumer = capture(lambda x:x+1,values,runtime,'width-copy-consumer')
      destination, bindings = targets(runtime,values)
      vector = get_vector_state().env_token
      four = _copy_kernel(4,vector)
      transfer = producer.prepare_output_transfer(values,consumer,input_offset=0,
          target_program=destination,target_bindings=bindings)
      assert _copy_kernel(2,vector) is not four
      leases = [owner for owner in transfer.executable._owners
                if hasattr(owner,'workers') and hasattr(owner,'grids')]
      assert leases and {owner.workers for owner in leases} == {provider_workers}
      torch.testing.assert_close(consumer(*transfer.replay(values)),x+2,rtol=0,atol=0)
  finally:
    _C.cpu_thread_limit(previous_budget)
    torch.set_num_threads(previous_threads)
