# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Resource compiler tests, runnable with the standalone native runtime module."""
import gc
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

native = importlib.import_module(os.environ.get('VECOPS_MEMORY_TEST_MODULE', 'vecops._C.memory'))
path = Path(__file__).resolve().parents[2] / 'python/vecops/planner/compiler.py'
spec = importlib.util.spec_from_file_location('_vecops_resource_graph_compiler', path)
compiler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compiler
spec.loader.exec_module(compiler)
Buffer = compiler.BufferRequirement
Q = 4096


def obj(name, **fields):
    value = getattr(native, name)()
    for key, item in fields.items():
        setattr(value, key, item)
    return value


def packing_runtime(*, page_bytes=(Q,Q), capacity_bytes=8*Q):
    """Logical Linux pools; compilation tests never acquire their large backing."""
    pools = ['uma', 'cowos_superdie']
    graph = obj('ResourceGraph', agents=[obj('Agent', id='compute', capabilities=['host_load_store'], address_space='host')])
    graph.capacity_groups = [obj('CapacityGroup', id=f'capacity:{pool}', capacity_bytes=capacity_bytes) for pool in pools]
    graph.pools = [obj('MemoryPool', id=pool, capacity_bytes=capacity_bytes, address_space='host',
                      granularity_bytes=quantum, page_sizes_bytes=[quantum], capacity_group=f'capacity:{pool}',
                      read_service=f'infinity_fabric:{pool}', write_service=f'infinity_fabric:{pool}')
                   for pool,quantum in zip(pools,page_bytes)]
    graph.resources = [obj('Resource', id=f'infinity_fabric:{pool}', rule='custom-service',
                          read=100.0 if index == 0 else 20.0, write=50.0,
                          sharing=native.SharingMode.read_write, kind=native.ResourceKind.bandwidth)
                       for index, pool in enumerate(pools)]
    graph.access = [obj('AccessConnection', id=f'direct:{pool}', rule='opaque', agent='compute', pool=pool,
                        access=[native.AccessMode.read, native.AccessMode.write], mapping=native.MappingMode.direct)
                    for pool in pools]
    discovery = obj('LinuxDiscovery', graph=graph,
                    scope=obj('Scope', executable_agents=['compute'], allocatable_pools=pools,
                              accessible_pools=pools, execution_slots={'compute':1}),
                    pool_bindings=[obj('LinuxPoolBinding', pool=pool, numa_nodes=[], address_space='host') for pool in pools],
                    pool_available_bytes={pool:capacity_bytes for pool in pools})
    options = obj('LinuxRuntimeOptions', participant='p0')
    value = native.make_linux_runtime(discovery, options)
    view = value.view(value.current_scope())
    return value, view


@pytest.fixture
def runtime():
    return packing_runtime()


def compile_buffers(runtime, requirements):
    value, view = runtime
    return compiler.GraphMemoryCompiler(value, view, value.snapshot(view), native_module=native).compile(requirements)


def committed(runtime):
    value, view = runtime
    return sum(group.committed_bytes for group in value.snapshot(view).groups)


def test_subpage_slabs_reuse_only_nonoverlapping_lifetimes(runtime):
    requirements = [Buffer('a', 256, lifetimes=((0,1),), allowed_pools=('uma',)),
                    Buffer('b', 128, lifetimes=((1,2),), allowed_pools=('uma',)),
                    Buffer('workspace', 128, lifetimes=((0,2),), allowed_pools=('uma',))]
    program = compile_buffers(runtime, requirements)
    assert program.bindings['a'].view_offset_bytes == program.bindings['b'].view_offset_bytes == 0
    assert program.bindings['workspace'].view_offset_bytes == 256
    assert sum(program.peak_bytes_by_capacity_group.values()) == Q
    assert len(program.memory.backings) == 1
    assert committed(runtime) == 0
    mutated = program.memory
    mutated.backings = []
    assert len(program.memory.backings) == 1


def test_private_slab_uses_movable_root_only_when_the_backend_supports_it(runtime):
    value, view = runtime
    capability = value.mapping_capabilities(view, ['uma'], native.BackingMode.private_movable)
    program = compile_buffers(runtime, [Buffer('local', Q, allowed_pools=('uma',))])
    expected = (native.BackingMode.private_movable if capability.reserve_without_backing
                else native.BackingMode.shareable_remappable)
    assert program.memory.backings[0].mode == expected
    assert program.memory.backings[0].bytes == Q
    assert committed(runtime) == 0


def test_explicit_modes_and_independent_ranges_keep_their_mapping_contract(runtime):
    for mode in (native.BackingMode.private_memory, native.BackingMode.shareable_remappable):
        program = compile_buffers(runtime, [Buffer('explicit', Q, allowed_pools=('uma',), backing=mode)])
        assert program.memory.backings[0].mode == mode
    contract = obj('SharingContract', id='shared', participants=['p0', 'p1'], producers=['p0'],
                   consumers=['p1'], protocol='publish-acquire-retire')
    for fields in ({'independent_mapping': True}, {'sharing_contract': contract, 'export_to': ('p1',)}):
        program = compile_buffers(runtime, [Buffer('fd', Q, allowed_pools=('uma',), **fields)])
        assert program.memory.backings[0].mode == native.BackingMode.shareable_remappable
    mixed = obj('PlacementPlan', address_space='host', stripe_bytes=Q, weights={'uma': 1, 'cowos_superdie': 1})
    program = compile_buffers(runtime, [Buffer('stripe', 2*Q, placement=mixed)])
    assert all(root.mode == native.BackingMode.shareable_remappable for root in program.memory.backings)


@pytest.mark.parametrize('quantum',[4096,65536])
@pytest.mark.parametrize('size,alignment',[(1<<20,64),((1<<20)-1,64),(1<<20,1<<18)])
def test_large_page_preference_preserves_threshold_and_alignment_minima(quantum,size,alignment):
    runtime=packing_runtime(page_bytes=(quantum,quantum),capacity_bytes=4<<20)
    request=Buffer('large',size,alignment=alignment,allowed_pools=('uma',))
    borrowed=Buffer('borrowed',1<<20,boundary='external',external_role='input',ownership='borrowed')
    program=compile_buffers(runtime,[Buffer('prefix',64,read_bytes=1,allowed_pools=('uma',)),request,borrowed])
    expected=max(alignment,quantum if size>=1<<20 else 1)
    assert program.bindings['large'].view_offset_bytes==expected
    assert request.alignment==program.bindings['large'].alignment==alignment
    assert next(r for r in program.requirements if r.id=='large')==request
    assert program.bindings['borrowed'].slot is None
    assert committed(runtime)==0


def test_page_preference_compacts_same_placement_before_considering_another_pool():
    runtime=packing_runtime(capacity_bytes=(1<<20)+Q)
    value,view=runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    program=engine.compile([Buffer('prefix',64,read_bytes=1),Buffer('large',(1<<20)+1)])
    assert program.bindings['large'].view_offset_bytes==64
    assert {root.pool for root in program.memory.backings}=={'uma'}
    assert program.peak_bytes_by_capacity_group=={'capacity:uma':(1<<20)+Q}
    assert engine.query_counts['packing_retries']==0 and committed(runtime)==0


def test_page_preference_retries_complete_compact_layout_after_later_fragmentation():
    capacity=2*(1<<20)+2*Q
    runtime=packing_runtime(capacity_bytes=capacity)
    value,view=runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    requests=[Buffer('a',(1<<20)+64,read_bytes=1,allowed_pools=('uma',)),
              Buffer('b',(1<<20)+4080,allowed_pools=('uma',)),
              Buffer('c',4040,alignment=16,allowed_pools=('uma',))]
    yielded=[]
    def once():
        for request in requests:
            yielded.append(request.id)
            yield request
    program=engine.compile(once())
    # The aligned b leaves4032 bytes before it and16 after it: c needs4040.
    assert [program.bindings[r.id].view_offset_bytes for r in requests]==[0,1048640,2101296]
    assert yielded==['a','b','c'] and len(program.requirements)==3
    assert program.peak_bytes_by_capacity_group=={'capacity:uma':capacity}
    assert engine.query_counts['packing_retries']==1 and committed(runtime)==0


def test_page_preference_retries_compact_layout_after_final_physical_estimate_rejection():
    runtime=packing_runtime(capacity_bytes=4<<20)
    value,view=runtime
    class Queries:
        estimates=0
        def __getattr__(self,name):return getattr(value,name)
        def estimate(self,view,recipe,caches):
            self.estimates+=1
            result=value.estimate(view,recipe,caches)
            if sum(root.bytes for root in recipe.backings)>(1<<20)+Q:result.feasible=False
            return result
    queries=Queries();engine=compiler.GraphMemoryCompiler(queries,view,value.snapshot(view),native_module=native)
    program=engine.compile([Buffer('prefix',64,read_bytes=1,allowed_pools=('uma',)),
                            Buffer('large',(1<<20)+1,allowed_pools=('uma',))])
    assert queries.estimates==2 and engine.query_counts['packing_retries']==1
    assert program.bindings['large'].view_offset_bytes==64
    assert program.peak_bytes_by_capacity_group=={'capacity:uma':(1<<20)+Q}
    assert committed(runtime)==0


def test_final_estimate_rejection_retains_exact_capacity_shortfall(runtime):
    value, view = runtime
    class Queries:
        def __getattr__(self, name):return getattr(value, name)
        def estimate(self, current_view, recipe, caches):
            # A provider observes less capacity than the earlier planning snapshot.
            return SimpleNamespace(feasible=False,
                new_backing_bytes_by_capacity_group={'capacity:uma':2*Q},
                shortfalls=[SimpleNamespace(kind=SimpleNamespace(name='pool'),
                    id='uma', requested_bytes=2*Q, available_bytes=Q)])
    engine=compiler.GraphMemoryCompiler(Queries(),view,value.snapshot(view),native_module=native)
    with pytest.raises(MemoryError,match='pool uma: requested=') as failure:
        engine.compile([Buffer('buffer',2*Q,allowed_pools=('uma',))])
    assert failure.value.shortfalls == (('pool','uma',2*Q,Q),)
    assert committed(runtime)==0


def test_native_estimate_reports_exact_shortfalls_and_clears_after_refund(runtime):
    value, view = runtime
    def recipe(pages):
        return obj('MemoryRecipe',participant='p0',layout_generation=1,
            backings=[obj('BackingRequest',id='pending',owner='p0',pool='uma',
                bytes=pages*Q,page_bytes=Q,mode=native.BackingMode.shareable_remappable)])
    lease=value.reserve(view,recipe(7),value.snapshot(view).epoch,[])
    rejected=value.estimate(view,recipe(2),[])
    assert not rejected.feasible
    group,=[row for row in rejected.shortfalls
            if row.kind==native.CapacityLimitKind.capacity_group and row.id=='capacity:uma']
    assert (group.requested_bytes,group.available_bytes)==(2*Q,Q)
    del lease
    gc.collect()
    accepted=value.estimate(view,recipe(2),[])
    assert accepted.feasible and not accepted.shortfalls
    assert committed(runtime)==0


@pytest.mark.parametrize('failure',[RuntimeError('provider permission rejected'),
                                   ValueError('recipe validation rejected'),MemoryError('host allocation failed')])
def test_page_preference_does_not_retry_nonplanning_failures(failure):
    runtime=packing_runtime(capacity_bytes=4<<20)
    value,view=runtime
    class Queries:
        estimates=0
        def __getattr__(self,name):return getattr(value,name)
        def estimate(self,*args):
            self.estimates+=1
            raise failure
    queries=Queries();engine=compiler.GraphMemoryCompiler(queries,view,value.snapshot(view),native_module=native)
    with pytest.raises(type(failure),match=str(failure)):
        engine.compile([Buffer('large',1<<20)])
    assert queries.estimates==1 and engine.query_counts['packing_retries']==0
    assert committed(runtime)==0


def test_page_preference_uses_each_resolved_pool_page_size():
    runtime=packing_runtime(page_bytes=(Q,16*Q),capacity_bytes=4<<20)
    requests=[]
    for pool in ('uma','cowos_superdie'):
        requests.extend([Buffer(pool+'/prefix',64,read_bytes=1,allowed_pools=(pool,)),
                         Buffer(pool+'/large',1<<20,allowed_pools=(pool,))])
    program=compile_buffers(runtime,requests)
    assert program.bindings['uma/large'].view_offset_bytes==Q
    assert program.bindings['cowos_superdie/large'].view_offset_bytes==16*Q
    assert committed(runtime)==0


def test_page_preference_preserves_interleaved_period_and_phase():
    runtime=packing_runtime(capacity_bytes=4<<20)
    value,view=runtime
    plan=obj('PlacementPlan',address_space='host',stripe_bytes=Q,weights={'uma':1,'cowos_superdie':2},
             cycle=['cowos_superdie','uma','cowos_superdie'],phase=1)
    program=compile_buffers(runtime,[Buffer('prefix',64,read_bytes=1,placement=plan),
                                    Buffer('large',1<<20,placement=plan)])
    assert program.bindings['large'].view_offset_bytes==3*Q
    placement=program.memory.addresses.slots[0].placement
    size=compiler._align(3*Q+(1<<20),Q)
    expected=value.resolve_placement(view,plan,size,native.BackingMode.shareable_remappable,
                                     [native.AccessMode.read,native.AccessMode.write])
    assert placement.key==expected.key
    assert [(s.pool,s.offset_bytes,s.size_bytes)for s in placement.segments]==[
        (s.pool,s.offset_bytes,s.size_bytes)for s in expected.segments]
    assert placement.segments[0].pool=='uma'
    assert sum(program.peak_bytes_by_capacity_group.values())==size
    assert committed(runtime)==0


def test_movable_mode_does_not_accept_independent_pages_or_stripes(runtime):
    with pytest.raises(ValueError, match='private_movable'):
        compile_buffers(runtime, [Buffer('invalid', Q, independent_mapping=True,
                                        backing=native.BackingMode.private_movable)])
    mixed = obj('PlacementPlan', address_space='host', stripe_bytes=Q, weights={'uma': 1, 'cowos_superdie': 1})
    with pytest.raises(ValueError, match='private_movable'):
        compile_buffers(runtime, [Buffer('invalid', 2*Q, placement=mixed,
                                        backing=native.BackingMode.private_movable)])
    assert committed(runtime) == 0


def test_unsupported_movable_capability_falls_back_but_query_errors_propagate(runtime):
    value, view = runtime

    class Queries:
        fail = False

        def __getattr__(self, name):
            return getattr(value, name)

        def mapping_capabilities(self, current_view, pools, mode):
            if mode == native.BackingMode.private_movable:
                if self.fail:
                    raise RuntimeError('capability permission denied')
                return SimpleNamespace(reserve_without_backing=False)
            return value.mapping_capabilities(current_view, pools, mode)

    queries = Queries()
    engine = compiler.GraphMemoryCompiler(queries, view, value.snapshot(view), native_module=native)
    program = engine.compile([Buffer('fallback', Q, allowed_pools=('uma',))])
    assert program.memory.backings[0].mode == native.BackingMode.shareable_remappable
    queries.fail = True
    with pytest.raises(RuntimeError, match='capability permission denied'):
        engine.compile([Buffer('failure', Q, allowed_pools=('uma',))])
    assert committed(runtime) == 0


def test_single_mapping_capability_keeps_distinct_regions_in_distinct_roots(runtime):
    value, view = runtime

    class Queries:
        def __getattr__(self, name):
            return getattr(value, name)

        def mapping_capabilities(self, current_view, pools, mode):
            result = value.mapping_capabilities(current_view, pools, mode)
            result.multiple_mappings = False
            return result

    requirements = []
    for name in ('first', 'second'):
        contract = obj('SharingContract', id=name, participants=['p0', 'p1'], producers=['p0'],
                       consumers=['p1'], protocol='publish-acquire-retire')
        requirements.append(Buffer(name, Q, allowed_pools=('uma',), sharing_contract=contract, export_to=('p1',)))
    program = compiler.GraphMemoryCompiler(Queries(), view, value.snapshot(view), native_module=native).compile(requirements)
    assert len(program.memory.addresses.regions) == len(program.memory.backings) == 2
    assert len({mapping.backing for mapping in program.memory.mappings}) == 2
    assert sum(program.peak_bytes_by_capacity_group.values()) == 2*Q
    assert committed(runtime) == 0


def test_different_placements_have_distinct_slabs_even_at_disjoint_times(runtime):
    program = compile_buffers(runtime, [Buffer('a',128,lifetimes=((0,1),),allowed_pools=('uma',)),
                                        Buffer('b',128,lifetimes=((1,2),),allowed_pools=('cowos_superdie',))])
    assert program.bindings['a'].slot != program.bindings['b'].slot
    assert len(program.memory.addresses.regions) == 2


def test_weighted_placement_charges_actual_pages_and_roundtrips(runtime):
    plan = obj('PlacementPlan', address_space='host', stripe_bytes=Q, weights={'uma':3,'cowos_superdie':7})
    program = compile_buffers(runtime, [Buffer('mixed',10*Q,placement=plan)])
    charges = {backing.pool:backing.bytes for backing in program.memory.backings}
    assert charges == {'uma':3*Q,'cowos_superdie':7*Q}
    assert native.recipe_to_toml(native.recipe_from_toml(program.dumps())) == program.dumps()


def test_shared_or_independent_ranges_get_exclusive_page_envelopes(runtime):
    contract = obj('SharingContract', id='shared', participants=['p0','p1'], producers=['p0'],
                   consumers=['p1'], protocol='publish-acquire-retire')
    program = compile_buffers(runtime, [Buffer(name,64,allowed_pools=('uma',),independent_mapping=True,
        sharing_contract=contract,export_to=('p1',)) for name in ('a','b')])
    a,b = program.bindings['a'],program.bindings['b']
    assert a.view_offset_bytes % Q == b.view_offset_bytes % Q == 0
    assert abs(a.view_offset_bytes-b.view_offset_bytes) >= Q
    assert sum(program.peak_bytes_by_capacity_group.values()) == 2*Q


def test_real_cost_queries_choose_pool_without_memory_tier_names(runtime):
    program = compile_buffers(runtime, [Buffer('traffic',Q,read_bytes=100*Q,consumers=('compute',))])
    assert program.memory.backings[0].pool == 'uma'
    assert program.used_conservative_fallback is False


def test_infeasible_request_is_pure_and_does_not_consume_runtime_budget(runtime):
    with pytest.raises(MemoryError):
        compile_buffers(runtime,[Buffer('too-big',9*Q,allowed_pools=('uma',))])
    assert committed(runtime)==0


def test_model_serial_transients_use_maximum_not_sum(runtime):
    value, view = runtime
    engine = compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    programs = {'x':[Buffer('temporary',8*Q,allowed_pools=('uma',))],
                'y':[Buffer('workspace',8*Q,allowed_pools=('uma',))]}
    calls = [compiler.ProgramCall('first','x'),compiler.ProgramCall('second','y')]
    program = engine.compile_model(programs,calls,serialized=True)
    assert sum(program.peak_bytes_by_capacity_group.values())==8*Q
    first = program.bindings[program.call_bindings['first']['temporary']]
    second = program.bindings[program.call_bindings['second']['workspace']]
    assert (first.slot,first.view_offset_bytes)==(second.slot,second.view_offset_bytes)
    with pytest.raises(MemoryError):
        engine.compile_model(programs,calls,serialized=False)


def test_model_retained_outputs_extend_through_later_consumers(runtime):
    value,view=runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    programs={'producer':[Buffer('output',2*Q,boundary='external',external_role='output',allowed_pools=('uma',)),
                          Buffer('workspace',4*Q,allowed_pools=('uma',))],
              'consumer':[Buffer('input',2*Q,boundary='external',external_role='input',ownership='borrowed'),
                          Buffer('workspace',4*Q,allowed_pools=('uma',))]}
    program=engine.compile_model(programs,[compiler.ProgramCall('a','producer',{'output':'cache'}),
                                           compiler.ProgramCall('b','consumer',{'input':'cache'})],serialized=True)
    assert program.call_bindings['a']['output']==program.call_bindings['b']['input']=='cache'
    assert sum(program.peak_bytes_by_capacity_group.values())==6*Q
    assert program.bindings['cache'].view_offset_bytes != program.bindings['b/workspace'].view_offset_bytes


def test_deferred_va_materialization_and_escaped_tensor_binding_lifetime(runtime):
    torch=pytest.importorskip('torch')
    value,view=runtime
    program=compile_buffers(runtime,[Buffer('output',64,boundary='external',external_role='output',allowed_pools=('uma',))])
    prepared=program.prepare(value,view)
    tensor=prepared.tensor('output',(16,),torch.float32)
    address=tensor.data_ptr()  # address inspection is permitted before backing exists
    assert committed(runtime)==0
    prepared.materialize()
    assert tensor.data_ptr()==address and committed(runtime)==Q
    tensor.fill_(7)
    alias=tensor[2:]
    token=prepared.completed()
    prepared.record_use(token)
    value.events.release([token])
    prepared.close()
    del prepared,tensor
    gc.collect()
    assert committed(runtime)==Q
    assert torch.all(alias==7)
    del alias
    gc.collect()
    assert committed(runtime)==0


def test_explicit_readonly_state_redirect_initializes_once(runtime):
    torch=pytest.importorskip('torch')
    value,view=runtime
    placement=obj('PlacementPlan',address_space='host',stripe_bytes=Q,weights={'uma':1})
    program=compile_buffers(runtime,[Buffer('weight',64,boundary='external',external_role='state',
        ownership='borrowed',read_only=True,may_redirect=True,placement=placement)])
    source=torch.arange(16,dtype=torch.float32)
    prepared=program.prepare(value,view)
    with torch.device('meta'):
        prepared.materialize(external={'weight':source})
        relocated=prepared.tensor('weight',(16,),torch.float32)
    assert relocated.device.type == 'cpu'
    assert torch.equal(relocated,source) and relocated.data_ptr()!=source.data_ptr()
    source.fill_(-1)
    assert relocated[3].item()==3
    prepared.close()


def test_invalid_requirements_and_cross_call_pool_constraints_are_rejected(runtime):
    for requirement in [Buffer('x',-1),Buffer('x',1,alignment=3),Buffer('x',1,lifetimes=((1,1),)),
                        Buffer('x',1,boundary='external'),Buffer('x',1,read_only=True,write_bytes=1)]:
        with pytest.raises(ValueError):
            compile_buffers(runtime,[requirement])
    value,view=runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    programs={'a':[Buffer('x',Q,boundary='external',external_role='output',allowed_pools=('uma',))],
              'b':[Buffer('x',Q,boundary='external',external_role='input',ownership='borrowed',allowed_pools=('cowos_superdie',))]}
    with pytest.raises(ValueError,match='incompatible pool'):
        engine.compile_model(programs,[compiler.ProgramCall('a','a',{'x':'shared'}),
                                       compiler.ProgramCall('b','b',{'x':'shared'})],serialized=True)


def test_retained_native_state_is_live_across_all_model_calls(runtime):
    programs={'x':[Buffer('state',Q,retained=True,allowed_pools=('uma',))],
              'y':[Buffer('scratch',Q,allowed_pools=('uma',))]}
    value,view=runtime
    program=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native).compile_model(
        programs,[compiler.ProgramCall('a','x'),compiler.ProgramCall('b','y')],serialized=True)
    a=program.bindings[program.call_bindings['a']['state']]
    b=program.bindings[program.call_bindings['b']['scratch']]
    assert a.view_offset_bytes!=b.view_offset_bytes
    assert program.peak_bytes_by_capacity_group['capacity:uma']==2*Q


def test_one_canonical_storage_preserves_cross_dtype_aliases_and_final_owner(runtime):
    torch=pytest.importorskip('torch')
    program=compile_buffers(runtime,[Buffer('bits',64)])
    value,view=runtime
    prepared=program.prepare(value,view).materialize()
    floats=prepared.tensor('bits',(16,),torch.float32)
    bits=prepared.tensor('bits',(32,),torch.int16)
    assert floats.untyped_storage()._cdata==bits.untyped_storage()._cdata
    floats.fill_(1)
    assert torch.equal(bits,floats.view(torch.int16))
    alias=bits[2:]
    del floats,bits
    assert prepared.tensor('bits',(64,),torch.uint8).untyped_storage()._cdata==alias.untyped_storage()._cdata
    prepared.close()
    assert committed(runtime)==Q
    assert alias.numel()==30
    del alias
    gc.collect()
    assert committed(runtime)==0


def test_incremental_packing_handles_out_of_order_lifetimes_without_live_aliases(runtime):
    requirements=[Buffer(f'buffer-{index:04d}',64,lifetimes=(((index*79)%300,(index*79)%300+3),))
                  for index in range(300)]
    program=compile_buffers(runtime,requirements)
    for index,left in enumerate(requirements):
        a=program.bindings[left.id]
        for right in requirements[index+1:]:
            if any(begin<other_end and other_begin<end for begin,end in left.lifetimes
                   for other_begin,other_end in right.lifetimes):
                b=program.bindings[right.id]
                assert a.slot!=b.slot or a.view_offset_bytes+a.size_bytes<=b.view_offset_bytes or \
                    b.view_offset_bytes+b.size_bytes<=a.view_offset_bytes
    assert sum(program.peak_bytes_by_capacity_group.values())==Q
    assert committed(runtime)==0


def test_zero_traffic_does_not_authorize_an_unavailable_consumer(runtime):
    with pytest.raises(ValueError,match='consumer'):
        compile_buffers(runtime,[Buffer('buffer',64,consumers=('unknown-engine',))])


class _ReserveRacer:
    def __init__(self, runtime, errors):
        self.runtime = runtime
        self.errors = iter(errors)
        self.snapshots = self.reservations = 0

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def snapshot(self, *args):
        self.snapshots += 1
        return self.runtime.snapshot(*args)

    def reserve(self, *args):
        self.reservations += 1
        error = next(self.errors, None)
        if error is not None:
            raise RuntimeError(error)
        return self.runtime.reserve(*args)


def test_materialize_retries_only_stale_admission_with_fresh_snapshots(runtime, monkeypatch):
    monkeypatch.setattr(compiler.time, 'sleep', lambda _seconds: None)
    value, view = runtime
    racer = _ReserveRacer(value, ['invalid_argument: stale budget snapshot',
        'shared_capacity.stale: stale shared capacity snapshot'])
    program = compile_buffers(runtime, [Buffer('buffer',64,allowed_pools=('uma',))])
    prepared = program.prepare(racer,view).materialize()
    assert racer.snapshots == racer.reservations == 3
    assert committed(runtime) == Q
    prepared.close()
    assert committed(runtime) == 0


@pytest.mark.parametrize('error', [
    'out_of_memory: memory allocation failed',
    'shared_capacity.permissions: stale shared capacity snapshot',
    'invalid_argument: invalid recipe: stale budget snapshot',
    'shared_capacity.stale: invalid recipe',
])
def test_nonstale_reservation_errors_are_never_retried(runtime, error):
    value, view = runtime
    racer = _ReserveRacer(value,[error])
    recipe = compile_buffers(runtime,[Buffer('buffer',64)]).memory
    with pytest.raises(RuntimeError) as failure:
        compiler.reserve_recipe_latest(racer,view,recipe)
    assert str(failure.value) == error
    assert racer.snapshots == racer.reservations == 1
    assert committed(runtime) == 0


def test_stale_reservation_retry_is_bounded(runtime, monkeypatch):
    monkeypatch.setattr(compiler.time, 'sleep', lambda _seconds: None)
    value, view = runtime
    racer = _ReserveRacer(value,['invalid_argument: stale budget snapshot'] * 65)
    recipe = compile_buffers(runtime,[Buffer('buffer',64)]).memory
    with pytest.raises(RuntimeError,match='^invalid_argument: stale budget snapshot$'):
        compiler.reserve_recipe_latest(racer,view,recipe)
    assert racer.snapshots == racer.reservations == 64
    assert committed(runtime) == 0


def test_sharing_contract_cannot_be_overwritten_by_slab_or_link_merging(runtime):
    first = obj('SharingContract',id='opaque',participants=['p0','p1'],producers=['p0'],
                consumers=['p1'],protocol='publish-acquire-retire')
    second = obj('SharingContract',id='opaque',participants=['p0','p2'],producers=['p0'],
                 consumers=['p2'],protocol='publish-acquire-retire')
    with pytest.raises(ValueError,match='conflicting sharing contract'):
        compile_buffers(runtime,[Buffer('a',64,sharing_contract=first),Buffer('b',64,sharing_contract=second)])
    value,view = runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    programs={'producer':[Buffer('out',64,boundary='external',external_role='output',sharing_contract=first)],
              'consumer':[Buffer('in',64,boundary='external',external_role='input',ownership='borrowed',
                                 sharing_contract=second)]}
    with pytest.raises(ValueError,match='incompatible sharing contracts'):
        engine.compile_model(programs,[compiler.ProgramCall('a','producer',{'out':'shared'}),
                                       compiler.ProgramCall('b','consumer',{'in':'shared'})],serialized=True)
    assert committed(runtime) == 0


def test_duplicate_native_slot_labels_cannot_silently_merge_distinct_buffers(runtime):
    descriptors=[dict(name='normalized',bytes=size,replicas=1,replica_stride=0,
        alignment=64,domain='global') for size in (1835008,1376256)]
    with pytest.raises(ValueError,match='duplicate native workspace identity: normalized'):
        compiler.native_workspace_requirements(SimpleNamespace(buffer_requirements=descriptors),prefix='kernel')
    value,view=runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    with pytest.raises(ValueError,match='duplicate local buffer identities'):
        engine.compile_model({'program':[Buffer('same',64),Buffer('same',128)]},
            [compiler.ProgramCall('call','program')],serialized=True)
    assert committed(runtime)==0


def test_native_call_count_weights_placement_without_multiplying_storage():
    lease=SimpleNamespace(buffer_requirements=[dict(name='scratch',bytes=Q,alignment=64,
        replicas=1,replica_stride=Q,domain='global',traffic_bytes=Q,retained=False)])
    fixed=Buffer('fixed',Q,read_bytes=2*Q,consumers=('compute',))
    def placed(count):
        runtime=packing_runtime(capacity_bytes=Q)
        native_requests=compiler.native_workspace_requirements(lease,prefix='native',
            consumers=('compute',),execution_count=count)
        program=compile_buffers(runtime,[fixed,*native_requests])
        slots={slot.id:slot for slot in program.memory.addresses.slots}
        pools={key:slots[binding.slot].placement.segments[0].pool
               for key,binding in program.bindings.items()}
        assert sum(program.peak_bytes_by_capacity_group.values())==2*Q
        assert committed(runtime)==0
        return native_requests[0],pools
    once,initial=placed(1)
    repeated,weighted=placed(13)
    assert initial=={'fixed':'uma','native/scratch':'cowos_superdie'}
    assert weighted=={'fixed':'cowos_superdie','native/scratch':'uma'}
    assert repeated.read_bytes==13*once.read_bytes
    assert repeated.bytes==once.bytes==Q and repeated.lifetimes==once.lifetimes


def test_native_call_count_keeps_replica_geometry_and_fallback_provenance(runtime):
    descriptor=dict(name='scratch',bytes=128,alignment=64,replicas=3,replica_stride=256,
                    domain='worker_local',retained=True)
    lease=SimpleNamespace(buffer_requirements=[descriptor],suspend_buffers=lambda:None)
    for traffic,per_call,assumed in ((0,640,True),(12.25,13,False)):
        descriptor['traffic_bytes']=traffic
        scope=compiler.ScopedNativeLease(lease,prefix='kernel',lifetime=(3,7),execution_count=13)
        requirement,=scope.requirements()
        assert requirement.read_bytes==13*per_call and requirement.write_bytes==0
        assert requirement.bytes==640 and requirement.alignment==64
        assert requirement.lifetimes==((3,7),) and not requirement.retained
        assert requirement.traffic_assumed is assumed
    for count in (0,-1,True,1.5,None,1<<64):
        with pytest.raises(ValueError,match='native execution count'):
            compiler.native_workspace_requirements(lease,prefix='bad',execution_count=count)
        with pytest.raises(ValueError,match='native execution count'):
            compiler.ScopedNativeLease(lease,prefix='bad',lifetime=(0,1),execution_count=count)
    descriptor['traffic_bytes']=1<<63
    with pytest.raises(ValueError,match='native workspace read traffic'):
        compiler.native_workspace_requirements(lease,prefix='overflow',execution_count=2)
    assert committed(runtime)==0


def test_repeated_calls_fold_internal_time_but_preserve_external_backedge(runtime):
    value,view=runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    requirements=[Buffer('tail-output',Q,lifetimes=((8,9),),boundary='external',external_role='output',
                         allowed_pools=('uma',),write_bytes=Q),
                  Buffer('head-temporary',Q,lifetimes=((0,1),),allowed_pools=('uma',),read_bytes=2*Q)]
    program=engine.compile_model({'loop':requirements},[compiler.ProgramCall('loop','loop',repeat_count=2000)],
                                 serialized=True)
    assert len(program.requirements[0].lifetimes)==1
    assert {r.id:r for r in program.requirements}['loop/tail-output'].lifetimes==((0,9),)
    assert program.bindings['loop/tail-output'].view_offset_bytes != program.bindings['loop/head-temporary'].view_offset_bytes
    assert {r.id:r for r in program.requirements}['loop/head-temporary'].read_bytes==4000*Q
    assert sum(program.peak_bytes_by_capacity_group.values())==2*Q


def test_explicit_input_output_carry_identity_lives_across_whole_repeated_call(runtime):
    value,view=runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    requirements=[Buffer('read-head',Q,lifetimes=((0,1),),boundary='external',external_role='input',
                         ownership='borrowed',allowed_pools=('uma',)),
                  Buffer('write-tail',Q,lifetimes=((8,9),),boundary='external',external_role='output',
                         allowed_pools=('uma',)),
                  Buffer('middle',Q,lifetimes=((3,4),),allowed_pools=('uma',))]
    program=engine.compile_model({'loop':requirements},[compiler.ProgramCall('loop','loop',
      bindings={'read-head':'carry','write-tail':'carry'},repeat_count=200)],serialized=True)
    assert program.bindings['carry'].view_offset_bytes != program.bindings['loop/middle'].view_offset_bytes
    assert {r.id:r.lifetimes for r in program.requirements}['carry']==((0,9),)


def test_unordered_calls_cannot_reuse_different_local_phases(runtime):
    value,view=runtime
    program=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native).compile_model(
      {'a':[Buffer('first',Q,lifetimes=((0,1),),allowed_pools=('uma',))],
       'b':[Buffer('later',Q,lifetimes=((100,101),),allowed_pools=('uma',))]},
      [compiler.ProgramCall('a','a',repeat_count=10),compiler.ProgramCall('b','b')],serialized=False)
    assert sum(program.peak_bytes_by_capacity_group.values())==2*Q


def test_retained_union_does_not_take_placement_priority_from_busy_transient(runtime):
    program=compile_buffers(runtime,[Buffer('hot',8*Q,read_bytes=100*Q,consumers=('compute',)),
      Buffer('retained',Q,read_bytes=Q,consumers=('compute',),retained=True)])
    slots={slot.id:slot for slot in program.memory.addresses.slots}
    assert {s.pool for s in slots[program.bindings['hot'].slot].placement.segments}=={'uma'}
    assert {s.pool for s in slots[program.bindings['retained'].slot].placement.segments}=={'cowos_superdie'}


def test_retained_union_blocks_all_times_without_disabling_other_reuse(runtime):
    program=compile_buffers(runtime,[Buffer('first',128,lifetimes=((0,1),),read_bytes=300),
      Buffer('retained',128,lifetimes=((3,4),),retained=True,read_bytes=200),
      Buffer('second',128,lifetimes=((1,2),),read_bytes=100)])
    assert program.bindings['first'].view_offset_bytes==program.bindings['second'].view_offset_bytes
    assert program.bindings['retained'].view_offset_bytes != program.bindings['first'].view_offset_bytes


def test_folded_plan_is_safe_against_expanded_random_loop_lifetimes(runtime):
    import random
    rng=random.Random(811)
    value,view=runtime
    for _ in range(12):
        programs={}
        calls=[]
        expanded={}
        cursor=0
        for call in range(3):
            local=[]
            for index in range(8):
                begin=rng.randrange(8)
                end=rng.randrange(begin+1,10)
                local.append(Buffer(f'b{index}',rng.choice((64,128,256)),lifetimes=((begin,end),),
                    allowed_pools=('uma',),retained=rng.random()<.1))
            name=f'c{call}'
            repeats=rng.randrange(1,8)
            programs[name]=local
            calls.append(compiler.ProgramCall(name,name,repeat_count=repeats))
            duration=max(b for requirement in local for a,b in requirement.lifetimes)
            for requirement in local:
                expanded[f'{name}/{requirement.id}']=tuple((cursor+i*duration+a,cursor+i*duration+b)
                    for i in range(repeats) for a,b in requirement.lifetimes)
            cursor+=duration*repeats
        program=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native).compile_model(
            programs,calls,serialized=True)
        for requirement in program.requirements:
            if requirement.retained:expanded[requirement.id]=((0,cursor),)
        for i,left in enumerate(program.requirements):
            a=program.bindings[left.id]
            for right in program.requirements[i+1:]:
                b=program.bindings[right.id]
                if a.slot!=b.slot:continue
                memory_overlap=(a.view_offset_bytes < b.view_offset_bytes+right.bytes
                    and b.view_offset_bytes < a.view_offset_bytes+left.bytes)
                live_overlap=compiler._overlap(expanded[left.id],expanded[right.id])
                assert not (memory_overlap and live_overlap),(left.id,right.id)


def test_scoped_native_state_initializes_only_when_active_and_reuses_after_stop(runtime):
    import ctypes
    class Lease:
        buffer_requirements=[dict(name='state',bytes=128,alignment=64,replicas=1,replica_stride=128,
                                  domain='global',traffic_bytes=128,retained=True)]
        def __init__(self,number):self.number,self.owner,self.address=number,None,None
        def activate_buffers(self,mapping,owner):
            assert self.owner is None
            self.owner,self.address=owner,mapping['state']['address']
            ctypes.c_ubyte.from_address(self.address).value=self.number
        def suspend_buffers(self):self.owner=None
    a,b=Lease(11),Lease(22)
    first=compiler.ScopedNativeLease(a,prefix='a',lifetime=(0,1))
    second=compiler.ScopedNativeLease(b,prefix='b',lifetime=(1,2))
    value,view=runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    program=engine.compile_model({'anchor':()},[compiler.ProgramCall('p','anchor')],serialized=True,
      scoped_requirements=first.requirements()+second.requirements())
    assert not any(r.retained for r in program.requirements)
    assert program.bindings['a/state'].view_offset_bytes==program.bindings['b/state'].view_offset_bytes
    prepared=program.prepare(value,view).materialize()
    assert a.owner is b.owner is None
    first.activate(prepared)
    assert ctypes.c_ubyte.from_address(a.address).value==11
    first.suspend()
    second.activate(prepared)
    assert a.address==b.address and ctypes.c_ubyte.from_address(b.address).value==22
    second.suspend()
    first.activate(prepared)
    assert ctypes.c_ubyte.from_address(a.address).value==11
    first.suspend()
    prepared.close()
    assert committed(runtime)==0


def test_call_domains_and_explicit_scoped_id_collisions_are_strict(runtime):
    programs={'x':[Buffer('tmp',64,lifetimes=((2,4),))]}
    calls=[compiler.ProgramCall('a','x',repeat_count=200),compiler.ProgramCall('b','x')]
    assert dict(compiler.call_time_domains(programs,calls,serialized=True))=={'a':(0,4),'b':(4,8)}
    assert dict(compiler.call_time_domains(programs,calls,serialized=False))=={'a':(0,1),'b':(0,1)}
    value,view=runtime
    with pytest.raises(ValueError,match='scoped buffer collides'):
        compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native).compile_model(
            programs,calls,serialized=True,scoped_requirements=[Buffer('a/tmp',64)])


def test_exact_cold_query_cache_preserves_payload_mode_and_compile_scope(runtime):
    value,view=runtime
    class Costs:
        def __init__(self):self.calls=[]
        def estimate_access(self,view,query,context):
            self.calls.append((str(query.access),query.payload_bytes,
              tuple((s.pool,s.offset_bytes,s.size_bytes) for s in query.memory.segments)))
            return value.costs.estimate_access(view,query,context)
    class Queries:
        def __init__(self):self.costs=Costs()
        def __getattr__(self,name):return getattr(value,name)
    queries=Queries()
    engine=compiler.GraphMemoryCompiler(queries,view,value.snapshot(view),native_module=native)
    requests=[Buffer('a',Q,read_bytes=Q,write_bytes=Q,consumers=('compute',)),
              Buffer('b',Q,read_bytes=Q,write_bytes=Q,consumers=('compute',)),
              Buffer('c',2*Q,read_bytes=2*Q,consumers=('compute',))]
    engine.compile(requests)
    assert engine.query_counts['access_hits']>=4
    assert len(queries.costs.calls)==6  # read/write Q + read 2Q, each of two pools
    first=list(queries.costs.calls)
    engine.compile(requests)
    assert queries.costs.calls==first+first  # caches cannot outlive a compilation


def test_explicit_admission_reserves_capacity_without_backing_and_releases_on_close(runtime):
    value, view = runtime
    prepared = compile_buffers(runtime, [Buffer('held', Q, allowed_pools=('uma',))]).prepare(value, view)
    before = value.snapshot(view)
    prepared.admit().admit()
    pending = value.snapshot(view)
    assert sum(group.unbacked_reserved_bytes for group in pending.groups) == Q
    assert committed(runtime) == 0 and prepared.binding is None and prepared.backings is None
    prepared.close()
    after = value.snapshot(view)
    assert sum(group.unbacked_reserved_bytes for group in after.groups) == 0
    assert [(p.pool,p.allocatable_bytes) for p in after.pools] == [(p.pool,p.allocatable_bytes) for p in before.pools]


@pytest.mark.parametrize('rejection', ['native-stale', 'physical-sample'])
def test_model_admission_replans_after_a_competitor_changes_pool_capacity(runtime, rejection):
    value, view = runtime
    competitor = compile_buffers(runtime, [Buffer('other', Q, allowed_pools=('uma',))]).prepare(value, view)
    class RacingRuntime:
        def __init__(self):
            self.once = False
        def __getattr__(self, name):
            return getattr(value, name)
        def reserve(self, *args):
            if not self.once:
                self.once = True
                competitor.admit()
                if rejection == 'physical-sample':
                    raise RuntimeError('shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: '
                        'group=capacity:uma, requested_bytes=32768, sampled_free_bytes=32768, pending_bytes=4096')
            return value.reserve(*args)
    racer = RacingRuntime()
    planner = compiler.GraphMemoryCompiler(racer, view, value.snapshot(view), native_module=native)
    programs = {'region': [Buffer('tensor', 8*Q, read_bytes=8*Q)]}
    prepared = planner.prepare_model(programs, [compiler.ProgramCall('call', 'region')], serialized=True)
    assert set(prepared.program.peak_bytes_by_capacity_group) == {'capacity:cowos_superdie'}
    assert committed(runtime) == 0
    assert sum(group.unbacked_reserved_bytes for group in value.snapshot(view).groups) == 9*Q
    prepared.materialize()
    assert committed(runtime) == 8*Q
    prepared.close(); competitor.close()
    assert committed(runtime) == 0
    assert sum(group.unbacked_reserved_bytes for group in value.snapshot(view).groups) == 0


@pytest.mark.parametrize('message', [
    'invalid_argument: hard capacity limit exceeded',
    'shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: '
    'group=capacity:uma, requested_bytes=4096, sampled_free_bytes=0, pending_bytes=0',
])
def test_model_admission_does_not_retry_capacity_failure_without_budget_change(runtime, message):
    value, view = runtime
    racer = _ReserveRacer(value, [message])
    planner = compiler.GraphMemoryCompiler(racer, view, value.snapshot(view), native_module=native)
    with pytest.raises(RuntimeError, match='hard capacity limit exceeded|physical_free'):
        planner.prepare_model({'region':[Buffer('tensor', Q)]}, [compiler.ProgramCall('call','region')], serialized=True)
    assert racer.reservations == 1
    assert sum(group.unbacked_reserved_bytes for group in value.snapshot(view).groups) == 0


def test_model_replans_when_competitor_changes_capacity_at_final_estimate(runtime):
    value,view=runtime
    competitor=compile_buffers(runtime,[Buffer('other',Q,allowed_pools=('uma',))]).prepare(value,view)
    class RacingRuntime:
        estimates=0
        def __getattr__(self,name):return getattr(value,name)
        def estimate(self,*args):
            self.estimates+=1
            if self.estimates==1:competitor.admit()
            return value.estimate(*args)
    racer=RacingRuntime()
    engine=compiler.GraphMemoryCompiler(racer,view,value.snapshot(view),native_module=native)
    prepared=engine.prepare_model({'region':[Buffer('tensor',8*Q,read_bytes=8*Q)]},
        [compiler.ProgramCall('call','region')],serialized=True)
    try:
        assert racer.estimates==2
        assert set(prepared.program.peak_bytes_by_capacity_group)=={'capacity:cowos_superdie'}
        assert committed(runtime)==0
        prepared.materialize()
        assert committed(runtime)==8*Q
    finally:prepared.close();competitor.close()
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def test_model_replans_initial_infeasibility_only_after_capacity_released(runtime):
    value,view=runtime
    held=compile_buffers(runtime,[Buffer('a',8*Q,allowed_pools=('uma',)),
        Buffer('b',8*Q,allowed_pools=('cowos_superdie',))]).prepare(value,view).admit()
    stale=value.snapshot(view)
    class ReleasingRuntime:
        snapshots=0
        def __getattr__(self,name):return getattr(value,name)
        def snapshot(self,*args):
            self.snapshots+=1
            held.close()
            return value.snapshot(*args)
    racer=ReleasingRuntime()
    engine=compiler.GraphMemoryCompiler(racer,view,stale,native_module=native)
    prepared=engine.prepare_model({'region':[Buffer('tensor',Q)]},
        [compiler.ProgramCall('call','region')],serialized=True)
    try:
        assert racer.snapshots>=1 and committed(runtime)==0
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==Q
    finally:prepared.close();held.close()


@pytest.mark.parametrize('failure',['estimate-rejection','memory-allocation-error'])
def test_model_compile_failures_do_not_retry_unchanged_capacity_or_heap_errors(runtime,failure):
    value,view=runtime
    class RejectingRuntime:
        estimates=0
        def __getattr__(self,name):return getattr(value,name)
        def estimate(self,*args):
            self.estimates+=1
            if failure=='memory-allocation-error':raise MemoryError('host allocation failed')
            result=value.estimate(*args);result.feasible=False;return result
    racer=RejectingRuntime()
    engine=compiler.GraphMemoryCompiler(racer,view,value.snapshot(view),native_module=native)
    with pytest.raises(MemoryError,match='host allocation failed|final physical backing estimate'):
        engine.prepare_model({'region':[Buffer('tensor',Q)]},
            [compiler.ProgramCall('call','region')],serialized=True)
    assert racer.estimates==1
    assert committed(runtime)==0
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def test_external_resource_finalization_precedes_admission_and_replans(runtime):
    value,view=runtime
    competitor=compile_buffers(runtime,[Buffer('transport',Q,allowed_pools=('uma',))]).prepare(value,view)
    seen=[]
    def finalize(proposed):
        # The graph has addresses/placement only, with no physical charge or
        # pending graph grant that could prevent a transport root admission.
        assert proposed.binding is None and proposed._lease is None
        binding=proposed.program.call_bindings['call']['tensor']
        seen.append(proposed.planned_placement(binding).segments[0].pool)
        if len(seen)==1:
            competitor.admit()
            return True
        return False
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    prepared=engine.prepare_model({'region':[Buffer('tensor',8*Q,read_bytes=8*Q)]},
        [compiler.ProgramCall('call','region')],serialized=True,finalize_resources=finalize)
    try:
        assert seen==['uma','cowos_superdie']
        assert committed(runtime)==0
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==9*Q
        prepared.materialize()
        assert committed(runtime)==8*Q
    finally:prepared.close();competitor.close()
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


@pytest.mark.parametrize('failure',['exception','nonbool','unstable'])
def test_external_resource_finalizer_failures_leave_no_graph_grant(runtime,failure):
    value,view=runtime;seen=[]
    def finalize(proposed):
        seen.append(proposed)
        assert proposed.binding is None and proposed._lease is None
        if failure=='exception':raise RuntimeError('transport registration failed')
        return 1 if failure=='nonbool' else True
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    error=TypeError if failure=='nonbool' else RuntimeError
    message='return bool' if failure=='nonbool' else 'stabilize' if failure=='unstable' else 'registration failed'
    with pytest.raises(error,match=message):
        engine.prepare_model({'region':[Buffer('tensor',Q)]},[compiler.ProgramCall('call','region')],
            serialized=True,admission_attempts=2,finalize_resources=finalize)
    assert len(seen)==(2 if failure=='unstable' else 1)
    assert all(p._closed for p in seen)
    assert committed(runtime)==0
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def test_peer_admission_retry_retains_proposal_and_withdraws_before_finalizer(runtime):
    value,view=runtime
    # The first local grant succeeds while a peer needs to replan. Keep the
    # local proposal, withdrawing its grant only at the finalization barrier.
    outcomes=iter([True,True,True,True,False,True,False,True,True,True,True,True])
    calls=[]
    def consensus(local):
        result=next(outcomes);assert local or not result
        calls.append((local,result));return result
    finalized=[]
    def finalize(proposed):
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0
        finalized.append(proposed);return False
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    prepared=engine.prepare_model({'region':[Buffer('tensor',Q)]},[compiler.ProgramCall('call','region')],
        serialized=True,finalize_resources=finalize,preparation_consensus=consensus)
    try:
        assert len(finalized)==2 and finalized[0] is finalized[1] and not finalized[1]._closed
        assert len(calls)==12
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==Q
    finally:prepared.close()
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def test_peer_planning_failure_prevents_collective_resource_finalizer(runtime):
    value,view=runtime;outcomes=iter([False,False]);finalized=[]
    def consensus(local):return next(outcomes)
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    with pytest.raises(RuntimeError,match='peer model preparation failed'):
        engine.prepare_model({'region':[Buffer('tensor',Q)]},[compiler.ProgramCall('call','region')],
            serialized=True,finalize_resources=lambda p:finalized.append(p),preparation_consensus=consensus)
    assert finalized==[]
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


@pytest.mark.parametrize('alignment',[1<<16,1<<22])
def test_small_strong_alignment_reaches_actual_reserved_va_without_extra_backing(alignment):
    runtime,view=packing_runtime(capacity_bytes=8*Q)
    request=Buffer('strong',64,alignment=alignment,allowed_pools=('uma',),
        backing=native.BackingMode.shareable_remappable)
    program=compiler.GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=native).compile([request])
    region,=program.memory.addresses.regions
    assert region.bytes==region.page_bytes==Q
    assert region.base_alignment_bytes==alignment
    assert sum(program.peak_bytes_by_capacity_group.values())==Q
    decoded=native.recipe_from_toml(program.dumps())
    assert decoded.addresses.regions[0].base_alignment_bytes==alignment
    prepared=program.prepare(runtime,view)
    try:
        binding=program.bindings['strong']
        before=prepared.addresses.slot(binding.slot).value+binding.view_offset_bytes
        assert before%alignment==0
        assert sum(g.committed_bytes+g.unbacked_reserved_bytes for g in runtime.snapshot(view).groups)==0
        prepared.materialize()
        assert prepared.addresses.slot(binding.slot).value+binding.view_offset_bytes==before
        assert sum(g.committed_bytes for g in runtime.snapshot(view).groups)==Q
    finally:
        prepared.close()
    assert sum(g.committed_bytes+g.unbacked_reserved_bytes for g in runtime.snapshot(view).groups)==0


def test_region_alignment_uses_strongest_requirement_without_raising_page_quantum():
    runtime,view=packing_runtime(capacity_bytes=8*Q)
    program=compiler.GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=native).compile([
        Buffer('first',64,alignment=1<<16,lifetimes=((0,1),),allowed_pools=('uma',)),
        Buffer('second',64,alignment=1<<22,lifetimes=((1,2),),allowed_pools=('uma',))])
    region,=program.memory.addresses.regions
    assert region.base_alignment_bytes==1<<22
    assert region.page_bytes==region.bytes==Q
    prepared=program.prepare(runtime,view)
    try:
        for name,alignment in [('first',1<<16),('second',1<<22)]:
            binding=program.bindings[name]
            assert (prepared.addresses.slot(binding.slot).value+binding.view_offset_bytes)%alignment==0
    finally:
        prepared.close()


def test_soft_base_alignment_survives_compact_offset_fallback_without_extra_charge():
    capacity=(1<<20)+Q
    runtime,view=packing_runtime(capacity_bytes=capacity)
    program=compiler.GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=native,
        preferred_alignment_bytes=2<<20).compile([
            Buffer('prefix',64,read_bytes=1,allowed_pools=('uma',)),
            Buffer('large',1<<20,allowed_pools=('uma',))])
    region,=program.memory.addresses.regions
    assert program.bindings['large'].view_offset_bytes==64  # compact trial fits
    assert region.bytes==sum(program.peak_bytes_by_capacity_group.values())==capacity
    assert region.page_bytes==Q and region.base_alignment_bytes==2<<20
    prepared=program.prepare(runtime,view)
    try:
        assert prepared.addresses.slot(program.bindings['large'].slot).value%(2<<20)==0
        assert sum(g.committed_bytes+g.unbacked_reserved_bytes for g in runtime.snapshot(view).groups)==0
    finally:
        prepared.close()


def test_soft_base_alignment_does_not_apply_to_small_only_slab():
    runtime,view=packing_runtime()
    program=compiler.GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=native,
        preferred_alignment_bytes=2<<20).compile([Buffer('small',64,allowed_pools=('uma',))])
    region,=program.memory.addresses.regions
    assert region.page_bytes==region.bytes==region.base_alignment_bytes==Q


def _budget_domain(groups, pools, constraints=(), *, idle=0):
    """Explicit opaque facts, independently inspectable pending recipes."""
    objects=[SimpleNamespace(id=p,capacity_group=g,capacity_bytes=cap,
        granularity_bytes=q,page_sizes_bytes=[q]) for p,g,cap,q in pools]
    view=SimpleNamespace(graph=SimpleNamespace(pools=objects),allocatable_pools=[p.id for p in objects])
    snapshot=SimpleNamespace(epoch=1,groups=[SimpleNamespace(capacity_group=g,hard_limit_bytes=hard,
        committed_bytes=used,unbacked_reserved_bytes=pending,system_available_bytes=free,
        idle_available_bytes=idle,shared_remaining_bytes=shared)for g,hard,used,pending,free,shared in groups],
        constraints=[SimpleNamespace(id=n,capacity_groups=members,remaining_bytes=cap)for n,members,cap in constraints],
        pools=[SimpleNamespace(pool=p.id,capacity_group=p.capacity_group,allocatable_bytes=p.capacity_bytes,
            shared_constraints=[n for n,members,_cap in constraints if p.capacity_group in members])for p in objects])
    return view,snapshot


class _BudgetRuntime:
    participant = 'p0'
    def __init__(self,snapshots,errors=()):
        self.states=list(snapshots);self.errors=list(errors);self.snapshots=0;self.recipes=[]
    def snapshot(self,view,caches):
        assert caches==[]
        result=self.states[min(self.snapshots,len(self.states)-1)];self.snapshots+=1
        return result
    def reserve(self,view,recipe,epoch,caches):
        assert caches==[]
        self.recipes.append(recipe)
        if self.errors:raise RuntimeError(self.errors.pop(0))
        return object()


def test_pending_budget_rounds_pool_pages_and_returns_actual_recipe():
    view,snapshot=_budget_domain([('a',8*Q,0,0,8*Q,None),('b',16*Q,0,0,16*Q,None)],
        [('first','a',Q,Q),('second','b',16*Q,4*Q)])
    lease,recipe=compiler.reserve_budget_latest(_BudgetRuntime([snapshot]),view,Q+1,
        (x for x in ('first','second')),native_module=native)
    assert lease is not None and [(b.pool,b.bytes,b.page_bytes)for b in recipe.backings]==[('first',Q,Q),('second',4*Q,4*Q)]
    assert recipe.addresses.regions==[] and recipe.addresses.slots==[] and recipe.mappings==[]


def test_pending_budget_alias_groups_and_idle_cache_are_not_double_counted():
    view,snapshot=_budget_domain([('shared',8*Q,2*Q,Q,3*Q,2*Q),('other',8*Q,0,0,8*Q,None)],
        [('alias-a','shared',8*Q,Q),('alias-b','shared',8*Q,Q),('tail','other',8*Q,Q)],idle=2*Q)
    _lease,recipe=compiler.reserve_budget_latest(_BudgetRuntime([snapshot]),view,4*Q,
        ['alias-a','alias-b','tail'],native_module=native)
    assert [(b.pool,b.bytes)for b in recipe.backings]==[('alias-a',2*Q),('tail',2*Q)]


def test_pending_budget_charges_every_overlapping_constraint():
    view,snapshot=_budget_domain([(g,8*Q,0,0,8*Q,None)for g in ('a','b','c')],
        [('one','a',8*Q,Q),('two','b',8*Q,Q),('three','c',8*Q,Q)],
        [('ab',['a','b'],3*Q),('bc',['b','c'],2*Q)])
    _lease,recipe=compiler.reserve_budget_latest(_BudgetRuntime([snapshot]),view,5*Q,
        ['one','two','three'],native_module=native)
    assert [(b.pool,b.bytes)for b in recipe.backings]==[('one',3*Q),('three',2*Q)]
    with pytest.raises(MemoryError,match='ordered pending budget'):
        compiler.reserve_budget_latest(_BudgetRuntime([snapshot]),view,5*Q+1,
            ['one','two','three'],native_module=native)


def test_pending_budget_ordered_failure_is_not_global_infeasibility():
    view,snapshot=_budget_domain([(g,4*Q,0,0,4*Q,None)for g in ('a','b','c')],
        [('one','a',4*Q,Q),('two','b',4*Q,Q),('three','c',4*Q,Q)],
        [('ab',['a','b'],4*Q),('ac',['a','c'],4*Q)])
    value=_BudgetRuntime([snapshot])
    with pytest.raises(MemoryError,match='ordered pending budget'):
        compiler.reserve_budget_latest(value,view,8*Q,['one','two','three'],native_module=native)
    assert not value.recipes
    _lease,recipe=compiler.reserve_budget_latest(value,view,8*Q,['two','three','one'],native_module=native)
    assert [(b.pool,b.bytes)for b in recipe.backings]==[('two',4*Q),('three',4*Q)]


@pytest.mark.parametrize('error',[
    'invalid_argument: stale budget snapshot',
    'shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: group=opaque, requested_bytes=8192, sampled_free_bytes=4096, pending_bytes=0',
])
def test_pending_budget_rebuilds_distribution_after_rejected_sample(error,monkeypatch):
    monkeypatch.setattr(compiler.time,'sleep',lambda _seconds:None)
    pools=[('first','a',4*Q,Q),('next','b',4*Q,Q)]
    view,first=_budget_domain([('a',4*Q,0,0,4*Q,None),('b',4*Q,0,0,4*Q,None)],pools)
    _,second=_budget_domain([('a',4*Q,0,0,Q,None),('b',4*Q,0,0,4*Q,None)],pools)
    second.epoch=2;value=_BudgetRuntime([first,second],[error])
    _lease,recipe=compiler.reserve_budget_latest(value,view,3*Q,['first','next'],native_module=native)
    assert [(b.pool,b.bytes)for b in value.recipes[0].backings]==[('first',3*Q)]
    assert [(b.pool,b.bytes)for b in recipe.backings]==[('first',Q),('next',2*Q)]
    assert value.snapshots==len(value.recipes)==2


@pytest.mark.parametrize('error',[
    'shared_capacity.permissions: permission denied',
    'shared_capacity.commit_uncertain: durability failure',
    'invalid_argument: invalid recipe: stale budget snapshot',
    'out_of_memory: memory allocation failed',
    'shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: '
    'group=opaque, requested_bytes=8192, sampled_free_bytes=unknown, pending_bytes=0',
])
def test_pending_budget_never_retries_unproven_rejection(error,runtime):
    value,view=runtime;racer=_ReserveRacer(value,[error])
    with pytest.raises(RuntimeError)as failure:
        compiler.reserve_budget_latest(racer,view,Q,['uma'],native_module=native)
    assert str(failure.value)==error and racer.snapshots==racer.reservations==1
    assert sum(g.unbacked_reserved_bytes+g.committed_bytes for g in value.snapshot(view).groups)==0


def test_pending_budget_real_native_stale_replan_and_refund(runtime,monkeypatch):
    monkeypatch.setattr(compiler.time,'sleep',lambda _seconds:None)
    value,view=runtime
    competitor=compile_buffers(runtime,[Buffer('other',7*Q,allowed_pools=('uma',))]).prepare(value,view)
    class Racing(_ReserveRacer):
        def reserve(self,*args):
            if not self.reservations:competitor.admit()
            return super().reserve(*args)
    racer=Racing(value,[])
    lease,recipe=compiler.reserve_budget_latest(racer,view,4*Q,['uma','cowos_superdie'],native_module=native)
    try:
        assert racer.reservations==2
        assert [(b.pool,b.bytes)for b in recipe.backings]==[('uma',Q),('cowos_superdie',3*Q)]
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==11*Q
        assert committed(runtime)==0
    finally:
        del lease;gc.collect();competitor.close()
    assert sum(g.unbacked_reserved_bytes+g.committed_bytes for g in value.snapshot(view).groups)==0


@pytest.mark.parametrize('byte_count,attempts',[(True,1),(-1,1),(1<<64,1),(1,0),(1,65),(1,True)])
def test_pending_budget_validates_counts_before_runtime_use(byte_count,attempts,runtime):
    value,view=runtime
    with pytest.raises(ValueError):
        compiler.reserve_budget_latest(value,view,byte_count,['uma'],native_module=native,attempts=attempts)


def test_pending_budget_bad_domain_or_insufficient_quantized_capacity_leaves_no_pending(runtime):
    value,view=runtime
    for order in ([],['uma','uma'],['unknown'],'uma'):
        with pytest.raises(ValueError):compiler.reserve_budget_latest(value,view,Q,order,native_module=native)
    with pytest.raises(MemoryError,match='ordered pending budget'):
        compiler.reserve_budget_latest(value,view,16*Q+1,['uma','cowos_superdie'],native_module=native)
    assert sum(g.unbacked_reserved_bytes+g.committed_bytes for g in value.snapshot(view).groups)==0


def test_pending_budget_snapshot_physical_already_excludes_pending_and_idle_is_unusable():
    view,snapshot=_budget_domain([('g',8*Q,2*Q,2*Q,3*Q,None)],[('p','g',8*Q,Q)],idle=2*Q)
    _lease,recipe=compiler.reserve_budget_latest(_BudgetRuntime([snapshot]),view,3*Q,['p'],native_module=native)
    assert recipe.backings[0].bytes==3*Q
    with pytest.raises(MemoryError):
        compiler.reserve_budget_latest(_BudgetRuntime([snapshot]),view,3*Q+1,['p'],native_module=native)


def test_pending_budget_native_zero_and_bounded_rejection_refund(runtime,monkeypatch):
    value,view=runtime
    lease,recipe=compiler.reserve_budget_latest(value,view,0,['uma'],native_module=native)
    assert recipe.backings==[] and committed(runtime)==0
    del lease;gc.collect()
    monkeypatch.setattr(compiler.time,'sleep',lambda _seconds:None)
    racer=_ReserveRacer(value,['invalid_argument: stale budget snapshot']*3)
    with pytest.raises(RuntimeError,match='stale budget snapshot'):
        compiler.reserve_budget_latest(racer,view,Q,['uma'],native_module=native,attempts=3)
    assert racer.snapshots==racer.reservations==3
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def test_pending_budget_unknown_capacity_or_constraint_identity_fails_closed():
    view,snapshot=_budget_domain([('g',Q,0,0,Q,None)],[('p','g',Q,Q)])
    view.graph.pools[0].capacity_bytes=None
    with pytest.raises(ValueError,match='pool capacity'):
        compiler.reserve_budget_latest(_BudgetRuntime([snapshot]),view,1,['p'],native_module=native)
    view.graph.pools[0].capacity_bytes=Q;snapshot.pools[0].shared_constraints=['missing']
    with pytest.raises(ValueError,match='constraint identity'):
        compiler.reserve_budget_latest(_BudgetRuntime([snapshot]),view,1,['p'],native_module=native)


def test_pending_budget_rejects_invalid_quantum_and_uint64_rounding_overflow():
    view,snapshot=_budget_domain([('g',8*Q,0,0,8*Q,None)],[('p','g',8*Q,3)])
    with pytest.raises(ValueError,match='power of two'):
        compiler.reserve_budget_latest(_BudgetRuntime([snapshot]),view,1,['p'],native_module=native)
    limit=(1<<64)-1
    view,snapshot=_budget_domain([(g,limit,0,0,limit,None)for g in ('a','b')],
        [('first','a',limit,Q),('tail','b',limit,Q)])
    value=_BudgetRuntime([snapshot])
    with pytest.raises(ValueError,match='rounded budget charge'):
        compiler.reserve_budget_latest(value,view,limit,['first','tail'],native_module=native)
    assert not value.recipes


@pytest.mark.parametrize('finalizer',['none','stable','growth','probe-stable','probe-growth'])
def test_concurrent_native_admission_holds_winner_and_spills_loser(runtime,finalizer):
    import threading
    value,view=runtime
    barrier=threading.Barrier(2,timeout=10)
    votes=[None,None];results=[None,None];errors=[None,None];plans=[[],[]]
    def agree(rank,ok):
        votes[rank]=ok;barrier.wait();result=all(votes);barrier.wait();return result
    class Engine(compiler.GraphMemoryCompiler):
        def compile_model(self,*args,**kwargs):
            result=super().compile_model(*args,**kwargs)
            plans[self.rank].append(result)
            return result
    engines=[Engine(value,view,value.snapshot(view),native_module=native) for _ in range(2)]
    finalized=[[],[]];probed=[[],[]]
    def work(rank):
        engines[rank].rank=rank
        def finalize(prepared):
            # All pending graph grants were withdrawn before this collective
            # callback. Neither addresses nor the offset plan were discarded.
            assert prepared._lease is None and prepared.binding is None
            assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0
            finalized[rank].append(prepared)
            agree(rank,True)
            return (finalizer=='growth' and len(finalized[rank])==2) or finalizer=='probe-growth'
        def probe(prepared):
            pending=sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)
            probed[rank].append(pending)
            return finalizer=='probe-growth' and len(probed[rank])==2 and rank==1
        try:
            results[rank]=engines[rank].prepare_model({'region':[Buffer('tensor',6*Q,read_bytes=6*Q)]},
                [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4,
                preparation_consensus=lambda ok:agree(rank,ok),
                probe_resources=probe if finalizer.startswith('probe-') else None,
                finalize_resources=finalize if finalizer!='none' else None)
        except BaseException as error:
            errors[rank]=error;barrier.abort()
    threads=[threading.Thread(target=work,args=(rank,),daemon=True) for rank in range(2)]
    try:
        for thread in threads:thread.start()
        for thread in threads:thread.join(20)
        assert not any(thread.is_alive() for thread in threads)
        assert errors==[None,None],errors
        growth=finalizer in ('growth','probe-growth')
        assert (sum(map(len,plans))==6 if growth else sorted(len(items) for items in plans)==[1,2])
        assert sorted(next(iter(result.program.peak_bytes_by_capacity_group)) for result in results)==[
            'capacity:cowos_superdie','capacity:uma']
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==12*Q
        assert committed(runtime)==0
        expected={'none':0,'stable':2,'growth':4,'probe-stable':0,'probe-growth':1}
        assert [len(items) for items in finalized]==[expected[finalizer]]*2
        if finalizer.startswith('probe-'):
            expected_pending=[0,6*Q,0,6*Q] if growth else [0,6*Q]
            assert probed==[expected_pending,expected_pending]
        # Grants independently own accepted capacity until the stopped plans
        # close; real backing is not required to test this protocol.
    finally:
        barrier.abort()
        for result in results:
            if result is not None:result.close()
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


@pytest.mark.parametrize('failure_stage',['replan','configuration','probe','finalizer'])
def test_concurrent_peer_fatal_failure_refunds_retained_native_grants(runtime,failure_stage):
    import threading
    value,view=runtime
    barrier=threading.Barrier(2,timeout=10);votes=[None,None]
    errors=[None,None];results=[None,None];counts=[0,0];finals=[0,0]
    configs=[0,0];probes=[0,0]
    def agree(rank,ok):
        votes[rank]=ok;barrier.wait();result=all(votes);barrier.wait();return result
    class Engine(compiler.GraphMemoryCompiler):
        def compile_model(self,*args,**kwargs):
            counts[self.rank]+=1
            if failure_stage=='replan' and counts[self.rank]==2:
                raise RuntimeError('fatal peer compilation')
            return super().compile_model(*args,**kwargs)
    engines=[Engine(value,view,value.snapshot(view),native_module=native) for _ in range(2)]
    def work(rank):
        engines[rank].rank=rank
        def finalize(prepared):
            finals[rank]+=1
            assert prepared._lease is None
            if finals[rank]==2 and rank==0:raise RuntimeError('fatal peer finalizer')
            return False
        def configure(_prepared):
            configs[rank]+=1
            if configs[rank]==2 and rank==0 and failure_stage=='configuration':
                raise RuntimeError('fatal peer resource configuration')
        def probe(_prepared):
            probes[rank]+=1
            if probes[rank]==2 and rank==0 and failure_stage=='probe':
                raise RuntimeError('fatal peer resource probe')
            return False
        try:
            results[rank]=engines[rank].prepare_model({'region':[Buffer('tensor',6*Q,read_bytes=6*Q)]},
                [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4,
                preparation_consensus=lambda ok:agree(rank,ok),
                configure_resources=configure if failure_stage=='configuration' else None,
                probe_resources=probe if failure_stage in ('configuration','probe') else None,
                finalize_resources=finalize if failure_stage!='replan' else None)
        except BaseException as error:errors[rank]=error
    threads=[threading.Thread(target=work,args=(rank,),daemon=True) for rank in range(2)]
    try:
        for thread in threads:thread.start()
        for thread in threads:thread.join(20)
        assert not any(thread.is_alive() for thread in threads)
        assert all(isinstance(error,RuntimeError) for error in errors),errors
        assert all('fatal peer' in str(error) or 'peer model preparation' in str(error)
            or 'peer cold resource' in str(error) for error in errors)
        assert results==[None,None] and sorted(counts)==[1,2]
        if failure_stage=='configuration':
            assert probes==[1,1] and finals==[0,0]
        if failure_stage=='probe':
            assert probes==[2,2] and finals==[0,0]
    finally:
        barrier.abort()
        for result in results:
            if result is not None:result.close()
    assert committed(runtime)==0
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def _sample_headroom_snapshot(value, view, caches, free, *, pool_quota=None):
    """Fresh native records with an injected OS sample; no ledger mutation."""
    result=value.snapshot(view,caches)
    groups=result.groups
    group=next(g for g in groups if g.capacity_group=='capacity:uma')
    group.system_available_bytes=min(group.system_available_bytes,free)
    result.groups=groups
    pools=result.pools
    for pool in pools:
        if pool.pool=='uma':
            pool.allocatable_bytes=min(pool.allocatable_bytes,
                group.idle_available_bytes+group.system_available_bytes)
            if pool_quota is not None:pool.allocatable_bytes=min(pool.allocatable_bytes,pool_quota)
    result.pools=pools
    return result


def _headroom_reject(value,view,recipe,caches,free):
    """Real recipe validation/estimate, then a typed lower physical sample."""
    result=value.estimate(view,recipe,caches)
    requested=result.new_backing_bytes_by_capacity_group.get('capacity:uma',0)
    if requested>free:
        result.feasible=False
        result.shortfalls=list(result.shortfalls)+[obj('CapacityShortfall',
            kind=native.CapacityLimitKind.capacity_group,id='capacity:uma',
            requested_bytes=requested,available_bytes=free)]
    return result


def test_model_headroom_continuous_sample_loss_fills_and_spills_real_runtime(runtime):
    value,view=runtime
    class Samples:
        free=8*Q
        estimates=0
        def __getattr__(self,name):return getattr(value,name)
        def snapshot(self,v,caches):
            return _sample_headroom_snapshot(value,v,caches,self.free)
        def estimate(self,v,recipe,caches):
            self.estimates+=1;self.free-=Q
            return _headroom_reject(value,v,recipe,caches,self.free)
    samples=Samples()
    engine=compiler.GraphMemoryCompiler(samples,view,samples.snapshot(view,[]),native_module=native)
    prepared=engine.prepare_model({'region':[Buffer(f'tensor:{i}',Q,read_bytes=8*Q)for i in range(8)]},
        [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4)
    try:
        assert samples.estimates==2
        assert dict(prepared.program.peak_bytes_by_capacity_group)=={'capacity:uma':6*Q,'capacity:cowos_superdie':2*Q}
        assert not hasattr(engine,'_cold_physical_headroom')
        assert committed(runtime)==0
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==8*Q
        prepared.materialize()
        assert committed(runtime)==8*Q
    finally:prepared.close()
    gc.collect()
    assert committed(runtime)==0
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def test_model_headroom_independent_pool_quota_does_not_learn(runtime):
    value,view=runtime
    class Samples:
        estimates=0
        def __getattr__(self,name):return getattr(value,name)
        def snapshot(self,v,caches):
            return _sample_headroom_snapshot(value,v,caches,7*Q,pool_quota=4*Q)
        def estimate(self,v,recipe,caches):
            self.estimates+=1
            return _headroom_reject(value,v,recipe,caches,3*Q)
    samples=Samples()
    initial=_sample_headroom_snapshot(value,view,[],8*Q,pool_quota=4*Q)
    engine=compiler.GraphMemoryCompiler(samples,view,initial,native_module=native)
    with pytest.raises(compiler._PlanningCapacityError,match='final physical backing estimate'):
        engine.prepare_model({'region':[Buffer(f'tensor:{i}',Q,read_bytes=8*Q)for i in range(8)]},
            [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4)
    assert samples.estimates==1
    assert not hasattr(engine,'_cold_physical_headroom')
    assert engine._budget_groups['capacity:uma']==(8*Q,0)
    assert committed(runtime)==0
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def test_model_headroom_actual_new_grant_is_counted_once(runtime):
    value,view=runtime
    competitor=compile_buffers(runtime,[Buffer('competing',2*Q,allowed_pools=('uma',))]).prepare(value,view)
    class Samples:
        estimates=0
        def __getattr__(self,name):return getattr(value,name)
        def estimate(self,*args):
            self.estimates+=1
            if self.estimates==1:competitor.admit()
            return value.estimate(*args)
    samples=Samples()
    engine=compiler.GraphMemoryCompiler(samples,view,value.snapshot(view),native_module=native)
    prepared=None
    try:
        prepared=engine.prepare_model({'region':[Buffer(f'tensor:{i}',Q,read_bytes=8*Q)for i in range(8)]},
            [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4)
        assert samples.estimates==2
        assert dict(prepared.program.peak_bytes_by_capacity_group)=={'capacity:uma':6*Q,'capacity:cowos_superdie':2*Q}
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==10*Q
        assert not hasattr(engine,'_cold_physical_headroom')
    finally:
        if prepared is not None:prepared.close()
        competitor.close()
    assert committed(runtime)==0
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def test_model_headroom_fatal_finalizer_restores_policy_and_refunds_retained_grant(runtime):
    value,view=runtime
    engine=compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native)
    previous=object();engine._cold_physical_headroom=previous
    decisions=iter([True,True,True,True,False,True,False,True,True,False])
    def consensus(ok):
        result=next(decisions)
        assert not result or ok
        return result
    proposals=[];failure=PermissionError('fatal finalizer retains original error')
    def finalize(prepared):
        proposals.append(prepared)
        assert prepared._lease is None
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0
        if len(proposals)==2:raise failure
        return False
    with pytest.raises(PermissionError)as caught:
        engine.prepare_model({'region':[Buffer('tensor',8*Q,read_bytes=8*Q)]},
            [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4,
            preparation_consensus=consensus,finalize_resources=finalize)
    assert caught.value is failure
    assert len(proposals)==2 and proposals[0]is proposals[1]and proposals[0]._closed
    assert engine._cold_physical_headroom is previous
    assert committed(runtime)==0
    assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==0


def _three_pool_headroom_runtime():
    """Three real resource pools; every test root remains a few pages."""
    names=('alpha','beta','gamma')
    graph=obj('ResourceGraph',agents=[obj('Agent',id='compute',capabilities=['host_load_store'],address_space='host')])
    graph.capacity_groups=[obj('CapacityGroup',id='capacity:'+p,capacity_bytes=8*Q)for p in names]
    graph.pools=[obj('MemoryPool',id=p,capacity_bytes=8*Q,address_space='host',granularity_bytes=Q,
        page_sizes_bytes=[Q],capacity_group='capacity:'+p,read_service='service:'+p,write_service='service:'+p)for p in names]
    graph.resources=[obj('Resource',id='service:'+p,rule='custom-service',read=rate,write=rate,
        sharing=native.SharingMode.read_write,kind=native.ResourceKind.bandwidth)for p,rate in zip(names,(100.,50.,10.))]
    graph.access=[obj('AccessConnection',id='access:'+p,rule='opaque',agent='compute',pool=p,
        access=[native.AccessMode.read,native.AccessMode.write],mapping=native.MappingMode.direct)for p in names]
    discovery=obj('LinuxDiscovery',graph=graph,scope=obj('Scope',executable_agents=['compute'],allocatable_pools=list(names),
        accessible_pools=list(names),execution_slots={'compute':1}),pool_bindings=[obj('LinuxPoolBinding',pool=p,numa_nodes=[],address_space='host')for p in names],
        pool_available_bytes={p:8*Q for p in names})
    value=native.make_linux_runtime(discovery,obj('LinuxRuntimeOptions',participant='p0'))
    return value,value.view(value.current_scope())


def test_model_headroom_observes_all_declining_groups_before_next_spill():
    value,view=_three_pool_headroom_runtime()
    class Samples:
        free={'alpha':8*Q,'beta':8*Q,'gamma':8*Q}
        estimates=0
        def __getattr__(self,name):return getattr(value,name)
        def snapshot(self,current,caches):
            result=value.snapshot(current,caches)
            groups=result.groups
            for g in groups:g.system_available_bytes=min(g.system_available_bytes,self.free[g.capacity_group.split(':')[1]])
            result.groups=groups;pools=result.pools
            for p in pools:p.allocatable_bytes=min(p.allocatable_bytes,self.free[p.pool])
            result.pools=pools;return result
        def estimate(self,current,recipe,caches):
            self.estimates+=1
            if self.estimates==1:self.free={'alpha':6*Q,'beta':6*Q,'gamma':8*Q}
            else:self.free['beta']=5*Q
            estimate=value.estimate(current,recipe,caches)
            shortfalls=[]
            for group,requested in estimate.new_backing_bytes_by_capacity_group.items():
                available=self.free[group.split(':')[1]]
                if requested>available:shortfalls.append(obj('CapacityShortfall',kind=native.CapacityLimitKind.capacity_group,
                    id=group,requested_bytes=requested,available_bytes=available))
            if shortfalls:estimate.feasible=False;estimate.shortfalls=list(estimate.shortfalls)+shortfalls
            return estimate
    samples=Samples();engine=compiler.GraphMemoryCompiler(samples,view,samples.snapshot(view,[]),native_module=native)
    prepared=None
    try:
        prepared=engine.prepare_model({'region':[Buffer('tensor:'+str(i),Q,read_bytes=8*Q)for i in range(10)]},
            [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4)
        # beta declined alongside alpha before it became the next full group.
        # Its per-group guard moves two pages to gamma before another rejection.
        assert samples.estimates==2
        assert dict(prepared.program.peak_bytes_by_capacity_group)=={'capacity:alpha':4*Q,'capacity:beta':4*Q,'capacity:gamma':2*Q}
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==10*Q
        assert sum(g.committed_bytes for g in value.snapshot(view).groups)==0
        assert not hasattr(engine,'_cold_physical_headroom')
        prepared.materialize()
        assert sum(g.committed_bytes for g in value.snapshot(view).groups)==10*Q
    finally:
        if prepared is not None:prepared.close()
    gc.collect()
    assert all(g.committed_bytes==g.unbacked_reserved_bytes==0 for g in value.snapshot(view).groups)


def _all_group_headroom_snapshot(free,*,idle=0,quota=None,epoch=1,shared=None):
    groups=[SimpleNamespace(capacity_group=name,hard_limit_bytes=32*Q,committed_bytes=0,live_bytes=0,
        idle_available_bytes=idle,idle_reserved_bytes=0,retired_bytes=0,unbacked_reserved_bytes=0,
        largest_cached_block_bytes=idle,shared_remaining_bytes=shared,system_available_bytes=value)for name,value in free.items()]
    pools=[SimpleNamespace(pool=g.capacity_group,capacity_group=g.capacity_group,
        largest_cached_block_bytes=idle,shared_constraints=[],allocatable_bytes=min(idle+compiler._ColdPhysicalHeadroom.fresh(g),
            quota[g.capacity_group]if quota and g.capacity_group in quota else 1<<64))for g in groups]
    return SimpleNamespace(epoch=epoch,groups=groups,pools=pools,constraints=[])


@pytest.mark.parametrize('boundary',['idle','quota','peer','constraint_only','max_not_sum'])
def test_model_all_group_headroom_boundaries(boundary):
    H=compiler._ColdPhysicalHeadroom;policy=H();idle=3*Q if boundary=='idle'else 0
    quota={'b':Q}if boundary=='quota'else None
    before=_all_group_headroom_snapshot({'a':8*Q,'b':8*Q},idle=idle,quota=quota)
    after=_all_group_headroom_snapshot({'a':6*Q,'b':6*Q},idle=idle,quota=quota)
    shortfalls=(('capacity_group','a',8*Q,6*Q),)
    if boundary=='peer':after.epoch+=1;after.groups[0].shared_remaining_bytes=30*Q
    if boundary=='constraint_only':shortfalls=(('constraint','cut',8*Q,6*Q),)
    policy.observe(shortfalls,before,after)
    if boundary in ('peer','constraint_only'):assert policy.losses=={};return
    expected={'a':2*Q}if boundary=='quota'else{'a':2*Q,'b':2*Q}
    assert policy.losses==expected
    budgets={g.capacity_group:(idle+H.fresh(g),idle)for g in after.groups}
    policy.apply(after,budgets)
    assert budgets['a']==(idle+4*Q,idle)
    assert budgets['b']==((idle+6*Q,idle)if boundary=='quota'else(idle+4*Q,idle))
    if boundary=='max_not_sum':
        later=_all_group_headroom_snapshot({'a':5*Q,'b':5*Q})
        policy.observe((('capacity_group','a',6*Q,5*Q),),after,later)
        assert policy.losses=={'a':2*Q,'b':2*Q}


@pytest.mark.parametrize('compact_failure',[False,True])
def test_compact_drops_preferred_metadata_before_second_compile_without_gc(runtime,compact_failure):
    import weakref
    from collections import Counter
    value,view=runtime
    class MetadataOwner:
        def __init__(self):self.large_metadata=bytearray(2<<20)
    failure=PermissionError('compact provider failure remains original')
    result=object()
    class Engine(compiler.GraphMemoryCompiler):
        def _compile(self,requirements,*,prefer_page_alignment):
            if prefer_page_alignment:
                owner=MetadataOwner();self.failed_metadata=weakref.ref(owner)
                self.query_counts=Counter(preferred_marker=3)
                raise compiler._PlanningCapacityError('force compact trial')
            # No gc.collect here: merely leaving the old handler must drop its
            # exception/traceback, including the first _compile frame's owner.
            assert self.failed_metadata()is None
            self.query_counts=Counter(compact_marker=2)
            if compact_failure:raise failure
            return result
    engine=Engine(value,view,value.snapshot(view),native_module=native)
    if compact_failure:
        with pytest.raises(PermissionError)as caught:engine.compile([Buffer('large',1<<20)])
        assert caught.value is failure and caught.value.__context__ is None
    else:assert engine.compile([Buffer('large',1<<20)])is result
    assert engine.query_counts==Counter(preferred_marker=3,compact_marker=2,packing_retries=1)
    assert committed(runtime)==0


@pytest.mark.parametrize('drop_before_last_snapshot',[False,True])
def test_model_reserve_drift_rollback_epoch_learns_before_next_full_plan(runtime,drop_before_last_snapshot):
    """Real native pending/VA rollback; deterministic physical samples, no backing."""
    value,view=runtime
    class Samples:
        free=8*Q
        estimates=0
        rejections=0
        rollback_epochs=[]
        def __getattr__(self,name):return getattr(value,name)
        def snapshot(self,v,caches):return _sample_headroom_snapshot(value,v,caches,self.free)
        def estimate(self,v,recipe,caches):
            self.estimates+=1
            if drop_before_last_snapshot:self.free-=Q
            return value.estimate(v,recipe,caches)
        def reserve(self,v,recipe,epoch,caches):
            self.free-=Q
            requested=sum(r.bytes for r in recipe.backings if r.owner==value.participant and r.pool=='uma')
            if requested>self.free:
                # Exercise actual C++ pending grant create/release and the public
                # epoch transition. The capacity error is a deterministic OS
                # sampler stand-in; no real backing is acquired or mapped.
                before=value.snapshot(v,caches)
                lease=value.reserve(v,recipe,before.epoch,caches)
                assert sum(g.unbacked_reserved_bytes for g in value.snapshot(v,caches).groups)>0
                lease=None
                after=value.snapshot(v,caches)
                self.rollback_epochs.append((before.epoch,after.epoch))
                assert after.epoch>before.epoch
                assert all(g.unbacked_reserved_bytes==g.committed_bytes==0 for g in after.groups)
                self.rejections+=1
                raise RuntimeError('shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: '
                    f'group=capacity:uma, requested_bytes={requested}, sampled_free_bytes={self.free}, pending_bytes=0')
            return value.reserve(v,recipe,value.snapshot(v,caches).epoch,caches)
    samples=Samples();engine=compiler.GraphMemoryCompiler(samples,view,samples.snapshot(view,[]),native_module=native)
    prepared=None
    try:
        prepared=engine.prepare_model({'region':[Buffer(f'page:{i}',Q,read_bytes=8*Q)for i in range(8)]},
            [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4)
        assert samples.estimates==2 and samples.rejections==1
        expected=4*Q if drop_before_last_snapshot else 6*Q
        assert dict(prepared.program.peak_bytes_by_capacity_group)=={'capacity:uma':expected,'capacity:cowos_superdie':8*Q-expected}
        assert prepared.backings is None and prepared.binding is None
        assert committed(runtime)==0 and sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==8*Q
        assert not hasattr(engine,'_cold_physical_headroom')
    finally:
        if prepared is not None:prepared.close()
    assert all(g.unbacked_reserved_bytes==g.committed_bytes==0 for g in value.snapshot(view).groups)


def test_model_reserve_drift_fatal_after_learning_restores_policy_and_refunds(runtime):
    value,view=runtime;failure=PermissionError('provider failure after a proved rollback')
    class Samples:
        free=8*Q
        reservations=0
        def __getattr__(self,name):return getattr(value,name)
        def snapshot(self,v,caches):return _sample_headroom_snapshot(value,v,caches,self.free)
        def reserve(self,v,recipe,epoch,caches):
            self.reservations+=1
            if self.reservations>1:raise failure
            lease=value.reserve(v,recipe,value.snapshot(v,caches).epoch,caches);lease=None
            self.free-=Q
            raise RuntimeError('shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: '
                f'group=capacity:uma, requested_bytes={8*Q}, sampled_free_bytes={self.free}, pending_bytes=0')
    samples=Samples();engine=compiler.GraphMemoryCompiler(samples,view,samples.snapshot(view,[]),native_module=native)
    previous=object();engine._cold_physical_headroom=previous
    with pytest.raises(PermissionError)as got:
        engine.prepare_model({'region':[Buffer(f'page:{i}',Q,read_bytes=8*Q)for i in range(8)]},
            [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4)
    assert got.value is failure and engine._cold_physical_headroom is previous
    assert all(g.unbacked_reserved_bytes==g.committed_bytes==0 for g in value.snapshot(view).groups)


def test_model_reserve_physical_message_with_new_peer_grant_is_not_extra_headroom(runtime):
    value,view=runtime
    competitor=compile_buffers(runtime,[Buffer('peer',2*Q,allowed_pools=('uma',))]).prepare(value,view)
    class Peer:
        once=False
        def __getattr__(self,name):return getattr(value,name)
        def reserve(self,v,recipe,epoch,caches):
            if not self.once:
                self.once=True;competitor.admit()
                raise RuntimeError('shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: '
                    f'group=capacity:uma, requested_bytes={8*Q}, sampled_free_bytes={8*Q}, pending_bytes={2*Q}')
            return value.reserve(v,recipe,epoch,caches)
    peer=Peer();engine=compiler.GraphMemoryCompiler(peer,view,value.snapshot(view),native_module=native);prepared=None
    try:
        prepared=engine.prepare_model({'region':[Buffer(f'page:{i}',Q,read_bytes=8*Q)for i in range(8)]},
            [compiler.ProgramCall('call','region')],serialized=True,admission_attempts=4)
        assert dict(prepared.program.peak_bytes_by_capacity_group)=={'capacity:uma':6*Q,'capacity:cowos_superdie':2*Q}
        assert committed(runtime)==0 and sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups)==10*Q
    finally:
        if prepared is not None:prepared.close()
        competitor.close()
    assert all(g.unbacked_reserved_bytes==g.committed_bytes==0 for g in value.snapshot(view).groups)


@pytest.mark.parametrize('boundary',['same_epoch','peer','quota','overlap','idle'])
def test_model_reserve_headroom_requires_complete_rollback_accounting(boundary):
    H=compiler._ColdPhysicalHeadroom;policy=H()
    idle=3*Q if boundary=='idle'else 0
    quota={'a':4*Q}if boundary=='quota'else None
    before=_all_group_headroom_snapshot({'a':8*Q,'b':8*Q},idle=idle,quota=quota,epoch=1)
    after=_all_group_headroom_snapshot({'a':7*Q,'b':8*Q},idle=idle,quota=quota,epoch=2)
    pending=2*Q if boundary=='idle'else 0
    if pending:
        before.groups[0].unbacked_reserved_bytes=after.groups[0].unbacked_reserved_bytes=pending
    if boundary=='same_epoch':after.epoch=1
    if boundary=='peer':after.groups[1].shared_remaining_bytes=31*Q
    if boundary=='overlap':
        before.constraints=[SimpleNamespace(id='opaque-cut',capacity_groups=['a','b'],remaining_bytes=16*Q)]
        after.constraints=[SimpleNamespace(id='opaque-cut',capacity_groups=['a','b'],remaining_bytes=15*Q)]
    message=('shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: '
        f'group=a, requested_bytes={8*Q}, sampled_free_bytes={7*Q+pending}, pending_bytes={pending}')
    proof=compiler._reserve_physical_evidence(message,before)
    rejection=compiler._ReservePhysicalCapacityError(message,proof)
    changed=policy.observe_reserve(rejection,before,after)
    if boundary!='idle':assert not changed and policy.losses=={};return
    assert changed and policy.losses=={'a':Q}
    available={'a':(idle+7*Q,idle),'b':(idle+8*Q,idle)};policy.apply(after,available)
    assert available=={'a':(idle+6*Q,idle),'b':(idle+8*Q,idle)}


def test_reserve_physical_proof_uses_last_actual_snapshot_and_no_exception_context():
    class Runtime:
        count=0
        def snapshot(self,*args):
            self.count+=1
            return _all_group_headroom_snapshot({'a':8*Q},epoch=self.count)
        def reserve(self,*args):
            if self.count==1:raise RuntimeError('shared_capacity.stale: stale shared capacity snapshot')
            raise RuntimeError('shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: '
                f'group=a, requested_bytes={8*Q}, sampled_free_bytes={7*Q}, pending_bytes=0')
    with pytest.raises(compiler._ReservePhysicalCapacityError)as got:
        compiler.reserve_recipe_latest(Runtime(),None,None)
    assert got.value.evidence[0][0]==2 and got.value.__context__ is None
    def primitive(value):
        return type(value)in(str,int,bool,type(None))or type(value)is tuple and all(primitive(v)for v in value)
    assert primitive(got.value.evidence)


@pytest.mark.parametrize('incomplete,missing', [(False, False), (True, False), (False, True)])
def test_access_cache_freezes_values_and_view_membership_per_compile(runtime, incomplete, missing):
    """Cache hits need no native estimate owner or repeated binding getters."""
    import weakref
    from collections import Counter
    value, view = runtime
    reads, owners = Counter(), []
    class View:
        agents = ['compute']
        def __getattr__(self, name):return getattr(view, name)
        @property
        def executable_agents(self):
            reads['agents'] += 1
            return list(self.agents)
    current = View()
    class Estimate:
        def __init__(self, estimate):self.estimate = estimate
        @property
        def quality(self):
            reads['quality'] += 1
            return native.CostQuality.incomplete if incomplete else self.estimate.quality
        @property
        def missing_facts(self):
            reads['missing'] += 1
            return ['unmeasured'] if missing else self.estimate.missing_facts
        @property
        def total_us(self):
            reads['total'] += 1
            return self.estimate.total_us
    class Costs:
        def estimate_access(self, unused, query, context):
            reads['query'] += 1
            estimate = Estimate(value.costs.estimate_access(view, query, context))
            owners.append(weakref.ref(estimate))
            return estimate
    class Queries:
        costs = Costs()
        def __getattr__(self, name):return getattr(value, name)
    engine = compiler.GraphMemoryCompiler(Queries(), current, value.snapshot(view), native_module=native)
    requirement = Buffer('traffic', Q, read_bytes=Q, write_bytes=2*Q, consumers=('compute',))
    plan = obj('PlacementPlan', address_space='host', stripe_bytes=Q, weights={'uma':1})
    resolved = value.resolve_placement(view, plan, Q, native.BackingMode.shareable_remappable,
                                       [native.AccessMode.read, native.AccessMode.write])
    expected = engine._score(requirement, resolved)
    assert expected[0] is (incomplete or missing)
    for _ in range(12):assert engine._score(requirement, resolved) == expected
    assert reads == dict(agents=1, query=2, quality=2, total=2,
                         **({'missing':2} if not incomplete else {}))
    assert all(owner() is None for owner in owners)
    engine._reset_queries()
    assert engine._score(requirement, resolved) == expected
    assert reads['agents'] == 2 and reads['query'] == 4
    # A later compilation must revalidate membership, even at the same epoch.
    current.agents = []
    with pytest.raises(ValueError, match='outside the executable resource view'):
        engine.compile([requirement])
    assert reads['agents'] == 3


def test_access_cache_preserves_query_and_property_error_boundaries(runtime):
    value, view = runtime
    class Costs:
        calls = 0
        error = 'query'
        def estimate_access(self, unused, query, context):
            self.calls += 1
            if self.error == 'query':raise RuntimeError('query unavailable')
            class BrokenEstimate:
                @property
                def quality(self):raise RuntimeError('estimate property unavailable')
            return BrokenEstimate()
    costs = Costs()
    class Queries:
        def __getattr__(self, name):return getattr(value, name)
    queries = Queries();queries.costs = costs
    engine = compiler.GraphMemoryCompiler(queries, view, value.snapshot(view), native_module=native)
    requirement = Buffer('traffic', Q, read_bytes=Q, consumers=('compute',))
    plan = obj('PlacementPlan', address_space='host', stripe_bytes=Q, weights={'uma':1})
    resolved = value.resolve_placement(view, plan, Q, native.BackingMode.shareable_remappable,
                                       [native.AccessMode.read, native.AccessMode.write])
    for _ in range(2):
        with pytest.raises(ValueError, match='no usable access cost for compute: query unavailable'):
            engine._score(requirement, resolved)
    assert costs.calls == 1
    costs.error = 'property';engine._reset_queries()
    with pytest.raises(RuntimeError, match='estimate property unavailable'):
        engine._score(requirement, resolved)
    assert costs.calls == 2


def _whole_score_fixture(runtime, *, ordered_costs=None):
    """Native query values with observable enum lookups and deterministic costs."""
    from collections import Counter
    value, view = runtime
    counts = Counter()
    class Native:
        def __getattr__(self, name):return getattr(native, name)
        @property
        def AccessMode(self):
            counts['mode_lookup'] += 1
            return native.AccessMode
    agents = ('compute', 'one', 'other', 'large')
    current = SimpleNamespace(graph=view.graph, allocatable_pools=view.allocatable_pools,
        executable_agents=agents, epoch=view.epoch, execution_slots={agent:1 for agent in agents})
    class Costs:
        def estimate_access(self, unused, query, context):
            counts['queries'] += 1
            if ordered_costs is not None:
                total = ordered_costs[query.agent]
            else:
                rate = {'compute':1., 'one':2., 'other':4.}[query.agent]
                pool = 1 if query.memory.segments[0].pool == 'uma' else 3
                mode = 1 if query.access == native.AccessMode.read else 2
                total = (rate*pool*mode + query.payload_bytes/Q + context.view_epoch
                         + sum(context.execution_slots.values()) + 16*context.include_preparation)
            return obj('CostEstimate', total_us=total, quality=native.CostQuality.reference)
    class Queries:
        costs = Costs()
        def __getattr__(self, name):return getattr(value, name)
    engine = compiler.GraphMemoryCompiler(Queries(), current, value.snapshot(view), native_module=Native())
    endpoints = []
    for pool in ('uma', 'cowos_superdie'):
        plan = obj('PlacementPlan', address_space='host', stripe_bytes=Q, weights={pool:1})
        endpoints.append(value.resolve_placement(view, plan, 2*Q, native.BackingMode.shareable_remappable,
                                                [native.AccessMode.read, native.AccessMode.write]))
    return engine, endpoints, counts


def test_whole_score_cache_skips_repeated_agent_mode_work(runtime):
    engine, endpoints, counts = _whole_score_fixture(runtime)
    request = Buffer('x', Q, consumers=('compute','one'), read_bytes=3*Q, write_bytes=2*Q)
    expected = engine._score(request, endpoints[0])
    before = dict(counts)
    for _ in range(5):assert engine._score(request, endpoints[0]) == expected
    assert dict(counts) == before
    assert engine.query_counts == dict(access_queries=4, access_hits=20)
    engine._reset_queries()
    assert engine._score(request, endpoints[0]) == expected
    assert counts['queries'] == 8


@pytest.mark.parametrize('change', ['consumer', 'read', 'write', 'extent', 'endpoint',
                                    'epoch', 'slots', 'preparation'])
def test_whole_score_keys_preserve_values_and_access_counters(runtime, change):
    from dataclasses import replace
    class Uncached(dict):
        def get(self, *args):return None
        def __setitem__(self, *args):pass
    engine, endpoints, _ = _whole_score_fixture(runtime)
    reference, reference_endpoints, _ = _whole_score_fixture(runtime)
    reference._score_cache = Uncached()
    request = Buffer('x', Q, consumers=('compute','one'), read_bytes=3*Q, write_bytes=2*Q)
    original = engine._score(request, endpoints[0])
    assert original == reference._score(request, reference_endpoints[0])
    endpoint_index = 0
    if change == 'consumer':request = replace(request, consumers=('compute','other'))
    if change == 'read':request = replace(request, read_bytes=4*Q)
    if change == 'write':request = replace(request, write_bytes=3*Q)
    if change == 'extent':request = replace(request, bytes=2*Q)
    if change == 'endpoint':endpoint_index = 1
    if change in ('epoch', 'slots', 'preparation'):
        epoch, slots, preparation = engine._score_context_key
        if change == 'epoch':epoch += 1
        if change == 'slots':slots = tuple((agent,n+1)for agent,n in slots)
        if change == 'preparation':preparation = True
        for target in (engine, reference):
            target._score_context_key = epoch,slots,preparation
            target._score_context = None
    changed = reference._score(request, reference_endpoints[endpoint_index])
    assert changed != original
    assert engine._score(request, endpoints[endpoint_index]) == changed
    assert engine._score(request, endpoints[endpoint_index]) == reference._score(request, reference_endpoints[endpoint_index])
    assert engine.query_counts == reference.query_counts


def test_whole_score_consumer_order_preserves_float_accumulation(runtime):
    from dataclasses import replace
    engine, endpoints, _ = _whole_score_fixture(runtime,
        ordered_costs={'large':float(2**53), 'one':1., 'other':1.})
    request = Buffer('ordered', Q, consumers=('large','one','other'), read_bytes=Q)
    request.validate()
    assert engine._score(request, endpoints[0]) == (False, float(2**53))
    reordered = replace(request, consumers=('one','other','large'))
    assert engine._score(reordered, endpoints[0]) == (False, float(2**53+2))
    assert engine._score(request, endpoints[0]) == (False, float(2**53))
    assert engine.query_counts == dict(access_queries=3, access_hits=6)


def _component_margin_fixture(*, edges=(), missing_pool_constraint=False, idle=0):
    """Prove one loss with unchanged complete accounting, then vary dependencies."""
    before = _all_group_headroom_snapshot(dict(a=8*Q, b=8*Q, c=8*Q, d=8*Q), idle=idle)
    after = _all_group_headroom_snapshot(dict(a=6*Q, b=8*Q, c=8*Q, d=8*Q), idle=idle)
    for sample in (before, after):
        sample.constraints = [SimpleNamespace(id=name, capacity_groups=list(members), remaining_bytes=24*Q)
                              for name, members in edges]
        if missing_pool_constraint:
            sample.pools[0].shared_constraints = ['unavailable-cut']
    policy = compiler._ColdPhysicalHeadroom()
    assert policy.observe((('capacity_group', 'a', 8*Q, 6*Q),), before, after)
    assert policy.losses == {'a': 2*Q}
    return policy, before, after


def _component_apply(policy, sample):
    H = compiler._ColdPhysicalHeadroom
    available = {g.capacity_group: (g.idle_available_bytes + H.fresh(g), g.idle_available_bytes)
                 for g in sample.groups}
    policy.apply(sample, available)
    return available


@pytest.mark.parametrize('foreign', ['epoch_only', 'pending', 'cache'])
def test_component_headroom_retains_proved_margin_across_foreign_change(foreign):
    policy, _, sample = _component_margin_fixture(idle=3*Q)
    sample.epoch += 1
    if foreign == 'pending':
        sample.groups[1].unbacked_reserved_bytes = Q
        sample.groups[1].shared_remaining_bytes = 31*Q
    if foreign == 'cache':
        sample.pools[1].largest_cached_block_bytes = Q
    available = _component_apply(policy, sample)
    assert policy.losses == {'a': 2*Q}
    assert available['a'] == (7*Q, 3*Q)  # Only fresh bytes are guarded; idle stays reusable.
    assert policy.accounting == policy.fingerprint(sample)


@pytest.mark.parametrize('change', ['pending', 'quota', 'cache', 'idle', 'pool', 'constraint', 'hard'])
def test_component_headroom_related_change_invalidates_margin(change):
    policy, _, sample = _component_margin_fixture()
    sample.epoch += 1
    if change == 'pending': sample.groups[0].unbacked_reserved_bytes = Q
    if change == 'quota': sample.pools[0].allocatable_bytes = 5*Q
    if change == 'cache': sample.pools[0].largest_cached_block_bytes = Q
    if change == 'idle':
        sample.groups[0].idle_available_bytes = Q
        sample.pools[0].allocatable_bytes = 7*Q
    if change == 'pool':
        sample.pools.append(SimpleNamespace(pool='alias-a', capacity_group='a',
            largest_cached_block_bytes=0, shared_constraints=[], allocatable_bytes=6*Q))
    if change == 'constraint':
        sample.constraints = [SimpleNamespace(id='new-cut', capacity_groups=['a'], remaining_bytes=24*Q)]
    if change == 'hard': sample.groups[0].hard_limit_bytes -= Q
    available = _component_apply(policy, sample)
    assert policy.losses == {}
    expected = (7*Q, Q) if change == 'idle' else (6*Q, 0)
    assert available['a'] == expected


@pytest.mark.parametrize('changed_group,retained', [('b', False), ('c', False), ('d', True)])
def test_component_headroom_overlap_transitive_closure(changed_group, retained):
    policy, _, sample = _component_margin_fixture(edges=(('ab', ('a', 'b')), ('bc', ('b', 'c'))))
    sample.epoch += 1
    next(g for g in sample.groups if g.capacity_group == changed_group).unbacked_reserved_bytes = Q
    available = _component_apply(policy, sample)
    assert policy.losses == ({'a': 2*Q} if retained else {})
    assert available['a'] == ((4*Q, 0) if retained else (6*Q, 0))


def test_component_headroom_overlap_remaining_and_membership_invalidate():
    for change in ('remaining', 'membership'):
        policy, _, sample = _component_margin_fixture(edges=(('ab', ('a', 'b')),))
        sample.epoch += 1
        if change == 'remaining': sample.constraints[0].remaining_bytes -= Q
        else: sample.constraints[0].capacity_groups.append('c')
        _component_apply(policy, sample)
        assert policy.losses == {}


@pytest.mark.parametrize('partial', ['missing_group', 'missing_constraint'])
def test_component_headroom_partial_closure_retains_epoch_protection(partial):
    policy, _, sample = _component_margin_fixture(
        edges=(('outside', ('a', 'unseen')),) if partial == 'missing_group' else (),
        missing_pool_constraint=partial == 'missing_constraint')
    assert _component_apply(policy, sample)['a'] == (4*Q, 0)
    sample.epoch += 1
    assert _component_apply(policy, sample)['a'] == (6*Q, 0)
    assert policy.losses == {}


@pytest.mark.parametrize('already_proved', [False, True])
def test_component_headroom_epoch_aba_never_teaches_new_typed_loss(already_proved):
    proved, _, before = _component_margin_fixture()
    policy = proved if already_proved else compiler._ColdPhysicalHeadroom()
    after = _all_group_headroom_snapshot(dict(a=5*Q, b=8*Q, c=8*Q, d=8*Q), epoch=2)
    # End balances are equal, but peer pending may have appeared during estimate
    # and disappeared before this sample. Component equality cannot disprove ABA.
    assert not policy.observe((('capacity_group', 'a', 6*Q, 5*Q),), before, after)
    assert policy.losses == ({'a': 2*Q} if already_proved else {})


@pytest.mark.parametrize('changed_peer', [False, True])
def test_component_headroom_reserve_keeps_strict_proof_and_unrelated_margin(changed_peer):
    policy, _, before = _component_margin_fixture()
    after = _all_group_headroom_snapshot(dict(a=6*Q, b=7*Q, c=8*Q, d=8*Q), epoch=2)
    if changed_peer: after.groups[1].unbacked_reserved_bytes = Q
    message = ('shared_capacity.physical_free: physical free sample minus pending reservations is exhausted: '
               f'group=b, requested_bytes={8*Q}, sampled_free_bytes={7*Q}, pending_bytes=0')
    failure = compiler._ReservePhysicalCapacityError(message, compiler._reserve_physical_evidence(message, before))
    assert policy.observe_reserve(failure, before, after) is (not changed_peer)
    assert policy.losses == ({'a': 2*Q} if changed_peer else {'a': 2*Q, 'b': Q})
    assert _component_apply(policy, after)['a'] == (4*Q, 0)


def test_component_headroom_keeps_only_primitive_keys_without_snapshot_owners():
    import weakref
    class Owner: pass
    class Snapshot(SimpleNamespace): pass
    def observe():
        policy, before, after = _component_margin_fixture(edges=(('ab', ('a', 'b')),))
        owner = Owner()
        snapshot = Snapshot(**after.__dict__, owner=owner)
        refs = (weakref.ref(snapshot), weakref.ref(owner))
        _component_apply(policy, snapshot)
        return policy, refs
    policy, refs = observe()
    assert all(ref() is None for ref in refs)  # No gc.collect or traceback cleanup needed.
    def primitive(value):
        return type(value) in (str, int, bool, type(None)) or (
            type(value) is tuple and all(primitive(v) for v in value)) or (
            type(value) is dict and all(primitive(k) and primitive(v) for k, v in value.items()))
    assert all(primitive(value) for value in vars(policy).values())
    assert policy.losses == {'a': 2*Q}


def test_component_headroom_foreign_finalizer_growth_avoids_relearning_real_runtime(runtime):
    value, view = runtime
    peer = compile_buffers(runtime, [Buffer('foreign-resource', Q, allowed_pools=('cowos_superdie',))]).prepare(value, view)
    class Samples:
        free = 8*Q
        estimates = 0
        finalizations = 0
        def __getattr__(self, name): return getattr(value, name)
        def snapshot(self, v, caches): return _sample_headroom_snapshot(value, v, caches, self.free)
        def estimate(self, v, recipe, caches):
            self.estimates += 1
            self.free -= 2*Q if self.estimates == 1 else Q
            return _headroom_reject(value, v, recipe, caches, self.free)
        def finalize(self, prepared):
            self.finalizations += 1
            if self.finalizations == 1:
                peer.admit()  # Actual unrelated native pending grant advances epoch.
                return True
            return False
    samples = Samples()
    engine = compiler.GraphMemoryCompiler(samples, view, samples.snapshot(view, []), native_module=native)
    prepared = None
    try:
        prepared = engine.prepare_model({'region': [Buffer(f'page:{i}', Q, read_bytes=8*Q) for i in range(8)]},
            [compiler.ProgramCall('call', 'region')], serialized=True,
            finalize_resources=samples.finalize, admission_attempts=4)
        assert samples.estimates == 3 and samples.finalizations == 2
        assert dict(prepared.program.peak_bytes_by_capacity_group) == {
            'capacity:uma': 3*Q, 'capacity:cowos_superdie': 5*Q}
        assert prepared.binding is None and prepared.backings is None
        assert not hasattr(engine, '_cold_physical_headroom')
        assert sum(g.unbacked_reserved_bytes for g in value.snapshot(view).groups) == 9*Q
    finally:
        if prepared is not None: prepared.close()
        peer.close()
    assert all(g.committed_bytes == g.unbacked_reserved_bytes == 0 for g in value.snapshot(view).groups)


@pytest.mark.parametrize('changed_group,retained', [('c', False), ('d', True)])
def test_component_headroom_pool_constraint_reference_joins_omitted_group(changed_group, retained):
    # A pool can supply a dependency even if its group is not in the constraint's
    # explicit member list. Retention must conservatively join both sources.
    before = _all_group_headroom_snapshot(dict(a=8*Q, b=8*Q, c=8*Q, d=8*Q))
    after = _all_group_headroom_snapshot(dict(a=6*Q, b=8*Q, c=8*Q, d=8*Q))
    for sample in (before, after):
        sample.constraints = [SimpleNamespace(id='bc', capacity_groups=['b', 'c'], remaining_bytes=24*Q)]
        sample.pools[0].shared_constraints = ['bc']
    policy = compiler._ColdPhysicalHeadroom()
    assert policy.observe((('capacity_group', 'a', 8*Q, 6*Q),), before, after)
    after.epoch += 1
    next(g for g in after.groups if g.capacity_group == changed_group).unbacked_reserved_bytes = Q
    _component_apply(policy, after)
    assert policy.losses == ({'a': 2*Q} if retained else {})
