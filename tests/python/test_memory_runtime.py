# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Native Python ownership, mapping, invocation and GIL contracts.

Synthetic bindings use ordinary Linux pages without NUMA policy; live NUMA
placement is qualified separately by LinuxRuntimeSmoke on the actual host.
"""
import ctypes
import gc
import importlib
import os
import threading
import weakref

import pytest

m = importlib.import_module(os.environ.get("VECOPS_MEMORY_TEST_MODULE", "vecops._C.memory"))
Q = os.sysconf("SC_PAGESIZE")


def make(name, **fields):
    value = getattr(m, name)()
    for name, item in fields.items():
        setattr(value, name, item)
    return value


@pytest.fixture
def runtime():
    pools = [make("MemoryPool", id=p, capacity_bytes=32 * Q, address_space="host",
                  granularity_bytes=Q, page_sizes_bytes=[Q], capacity_group=f"cap-{p}")
             for p in ("a", "b")]
    graph = make("ResourceGraph", agents=[make("Agent", id="executor", capabilities=["host_load_store"],
                                             address_space="host")], pools=pools,
                 capacity_groups=[make("CapacityGroup", id=f"cap-{p}", capacity_bytes=32 * Q)
                                  for p in ("a", "b")],
                 domains=[make("AllocationDomain", id="domain", rule="opaque", members=["a", "b"])],
                 access=[make("AccessConnection", id=f"direct-{p}", rule="opaque", agent="executor",
                              pool=p, access=[m.AccessMode.read, m.AccessMode.write], mapping=m.MappingMode.direct)
                         for p in ("a", "b")],
                 methods=[make("Method", id="copy", request_executor=True, bandwidth=10.0, startup_us=0.0,
                               routes=[make("Route", id=a+b, source=a, target=b) for a in ("a", "b")
                                       for b in ("a", "b")])])
    scope = make("Scope", executable_agents=["executor"], allocatable_pools=["a", "b"],
                 accessible_pools=["a", "b"], execution_slots={"executor": 1})
    discovery = make("LinuxDiscovery", graph=graph, scope=scope,
                     pool_bindings=[make("LinuxPoolBinding", pool=p, numa_nodes=[]) for p in ("a", "b")],
                     pool_available_bytes={"a": 32 * Q, "b": 32 * Q})
    options = make("LinuxRuntimeOptions", participant="p0", cpu_queues={"main": "executor"},
                   cpu_copy_methods=["copy"])
    return m.make_linux_runtime(discovery, options)


def view(runtime):
    return runtime.view(runtime.current_scope())


def committed(runtime):
    gc.collect()
    return sum(group.committed_bytes for group in runtime.snapshot(view(runtime)).groups)


def test_residency_query_ignores_pool_claim_and_preserves_capacity(runtime):
    owner = ctypes.create_string_buffer(b'\x5d' * Q)
    address = make('VirtualAddress', address_space='host', value=ctypes.addressof(owner))
    external = m.ExternalMemoryView('sample', address, Q, owner, 'a')
    first = runtime.observe_external(view(runtime), external)
    second = runtime.observe_external(view(runtime), external)
    # Synthetic pools have no physical NUMA binding. The caller's claim is not
    # a placement observation, even when the range is completely resident.
    assert not first.complete
    assert not first.observed_placement
    assert first.resident_bytes in (None, Q)
    assert second.sample_epoch > first.sample_epoch
    assert committed(runtime) == 0
    assert owner.raw[:Q] == b'\x5d' * Q


def backing(bytes=Q, name="root"):
    return make("BackingRequest", id=name, owner="p0", pool="a", bytes=bytes, page_bytes=Q,
                mode=m.BackingMode.shareable_remappable)


def recipe(runtime):
    placement = runtime.resolve_placement(view(runtime), make("PlacementPlan", address_space="host",
        stripe_bytes=Q, weights={"a": 1}), Q, m.BackingMode.shareable_remappable,
        [m.AccessMode.read, m.AccessMode.write])
    return make("MemoryRecipe", participant="p0", layout_generation=1,
        addresses=make("AddressLayout", regions=[make("VirtualRegion", id="r", address_space="host", bytes=Q,
            page_bytes=Q)], slots=[make("AddressSlot", id="slot", region="r", offset_bytes=0, size_bytes=Q,
                placement=placement)]), backings=[backing()],
        mappings=[make("PageMapping", region="r", virtual_offset_bytes=0, backing="root", backing_offset_bytes=0,
            size_bytes=Q, access=[m.AccessMode.read, m.AccessMode.write])])


def bind(runtime, plan, prepared):
    v = view(runtime)
    lease = runtime.reserve(v, plan, runtime.snapshot(v).epoch)
    roots = runtime.acquire_backing(v, plan, lease)
    return runtime.bind(v, plan, prepared, roots)


def arena(runtime, pages=4):
    return runtime.create_arena(view(runtime), make("ArenaOptions", domain="domain",
        hard_limit=make("Limit", bytes_or_capacity_fraction=pages * Q),
        initial=make("Preallocate", bytes_or_limit_fraction=pages * Q), grow_by_bytes=pages * Q,
        retain_idle_bytes=pages * Q, page_bytes=Q, backing=m.BackingMode.shareable_remappable,
        allow_growth=True))


def status(state=m.EventState.succeeded, stopped=True):
    return make("EventStatus", state=state, accesses_stopped=stopped)


def context(name, generation=0):
    return make("InvocationContext", stamp=make("InvocationStamp", id=name, binding_generation=generation))


def test_python_buffer_retains_physical_ownership(runtime):
    allocation = runtime.allocate(view(runtime), backing())
    assert allocation.address % Q == 0
    buffer = memoryview(allocation)
    buffer[0] = 71
    del allocation
    assert committed(runtime) == Q and buffer[0] == 71
    del buffer
    assert committed(runtime) == 0


def test_exported_slice_retains_root_after_arena_allocation_dies(runtime):
    pool = arena(runtime)
    a = pool.allocate(make("AllocateRequest", bytes=Q, preferred_pool="a", allow_domain_fallback=False))
    b = pool.allocate(make("AllocateRequest", bytes=Q, preferred_pool="a", allow_domain_fallback=False))
    assert b.address == a.address + Q
    root = b.export_backing()
    memoryview(b)[0] = 29
    reservation = runtime.reserve_address("host", Q, Q)
    mapped = runtime.map(reservation, root, make("MapRequest", target_offset_bytes=0, backing_offset_bytes=0,
                                               size_bytes=Q))
    assert ctypes.c_ubyte.from_address(mapped.address).value == 29
    del a, b, root
    gc.collect()
    assert pool.trim() == 0
    del mapped
    gc.collect()
    assert pool.trim() == 4 * Q
    assert committed(runtime) == 0


def test_offset_only_prepare_bind_and_stable_address_switch(runtime):
    plan = recipe(runtime)
    prepared = runtime.prepare_addresses(view(runtime), plan.addresses)
    buffer = memoryview(prepared.slot_buffer("slot"))
    address = prepared.slot("slot").value
    assert committed(runtime) == 0  # no buffer access before physical bind
    first = bind(runtime, plan, prepared)
    assert first.slot("slot") == address
    buffer[0] = 81
    first_generation = first.generation
    proof = runtime.quiesce(first, [])
    runtime.unbind(first, proof)
    second = bind(runtime, plan, prepared)
    assert second.generation != first_generation
    assert second.slot("slot") == address
    buffer[0] = 23
    assert ctypes.c_ubyte.from_address(second.slot("slot")).value == 23
    runtime.unbind(second, runtime.quiesce(second, []))


def test_invocation_local_external_owner_and_physical_alias_validation(runtime):
    plan = recipe(runtime)
    prepared = runtime.prepare_addresses(view(runtime), plan.addresses)
    binding = bind(runtime, plan, prepared)
    source = (ctypes.c_ubyte * Q)(*[73] * Q)
    address = make("VirtualAddress", address_space="host", value=ctypes.addressof(source))
    imported = runtime.import_memory(view(runtime), m.ExternalMemoryView("source", address, Q, source, "b"))
    ctx = context("copy-one", binding.generation)
    ctx.external = [make("ExternalBinding", symbol="input", memory=imported)]
    command = make("MemoryCommand", id="copy", opcode=m.MemoryOpcode.copy,
        source=make("ByteRangeRef", kind=m.MemoryRefKind.external, memory="input", size_bytes=Q),
        target=make("ByteRangeRef", kind=m.MemoryRefKind.slot, memory="slot", size_bytes=Q),
        method="copy", queue="main")
    submission = runtime.submit(view(runtime), make("MemoryCommandPlan", commands=[command]), binding, ctx)
    assert submission.submission_error is None
    assert memoryview(prepared.slot_buffer("slot"))[0] == 73
    assert runtime.events.wait(submission.finished).state == m.EventState.succeeded
    # Multiple ports can name the same native completion; the Python release
    # facade intentionally deduplicates complete authenticated token tuples.
    runtime.events.release([submission.finished, submission.finished])
    alias = m.ExternalMemoryView("alias", make("VirtualAddress", address_space="host", value=binding.slot("slot")),
                                 Q, binding, "a")
    ctx = context("copy-alias", binding.generation)
    ctx.external = [make("ExternalBinding", symbol="input", memory=runtime.import_memory(view(runtime), alias))]
    with pytest.raises(RuntimeError, match="overlap"):
        runtime.submit(view(runtime), make("MemoryCommandPlan", commands=[command]), binding, ctx)


def test_async_failed_but_accessing_keeps_arena_pages_out_of_reuse(runtime):
    pool = arena(runtime, 1)
    allocation = pool.allocate(make("AllocateRequest", bytes=Q, allow_domain_fallback=True))
    done = threading.Event()
    callback = lambda: status(m.EventState.failed, done.is_set())
    token = runtime.events.import_host_event(callback, callback, None, done,
                                            context("active-failure", allocation.info.generation))
    pool.release(allocation, token)
    with pytest.raises(RuntimeError):
        pool.allocate(make("AllocateRequest", bytes=Q, allow_domain_fallback=True))
    with pytest.raises(RuntimeError):
        runtime.events.release([token])
    done.set()
    assert runtime.events.wait(token).accesses_stopped
    replacement = pool.allocate(make("AllocateRequest", bytes=Q, allow_domain_fallback=True))
    assert replacement.info.generation != token.binding_generation
    runtime.events.release([token])


def test_wait_releases_gil_and_callback_owner_until_stopped(runtime):
    class Owner:
        pass
    lifetime = Owner()
    alive = weakref.ref(lifetime)
    ready = threading.Event()
    entered = threading.Event()
    def query():
        return status(m.EventState.succeeded if ready.is_set() else m.EventState.pending, ready.is_set())
    def wait():
        entered.set()
        assert ready.wait(3), "producer Python thread could not run"
        return status()
    token = runtime.events.import_host_event(query, wait, None, lifetime, context("thread"))
    del lifetime
    assert alive() is not None
    def produce():
        if entered.wait(3):
            ready.set()
    producer = threading.Thread(target=produce)
    producer.start()
    assert runtime.events.wait(token).accesses_stopped
    producer.join(3)
    runtime.events.release([token])
    gc.collect()
    assert alive() is None


def test_callback_exception_becomes_native_error(runtime):
    def broken():
        raise ValueError("producer failed")
    token = runtime.events.import_host_event(broken, lambda: status(m.EventState.failed), None, object(),
                                            context("callback-error"))
    with pytest.raises(RuntimeError, match="python.event.*producer failed"):
        runtime.events.query(token)
    assert runtime.events.wait(token).accesses_stopped
    runtime.events.release([token])
