# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""C++ resource-model binding contracts; missing bindings are a test failure.

Use VECOPS_MEMORY_TEST_MODULE only to exercise the same bindings in an isolated
extension before they are linked into vecops._C.memory. No hardware probe or
physical memory allocation is involved in these tests.
"""

from collections import Counter
import importlib
import os

import pytest


MODULE_NAME = os.environ.get("VECOPS_MEMORY_TEST_MODULE", "vecops._C.memory")
memory = importlib.import_module(MODULE_NAME)
PAGE = 4096
UINT64_MAX = (1 << 64) - 1


def make(name, **fields):
    value = getattr(memory, name)()
    for field, item in fields.items():
        setattr(value, field, item)
    return value


@pytest.fixture(scope="module", autouse=True)
def report_binding_origin():
    print(f"memory binding module: {MODULE_NAME}")
    print(f"memory binding origin: {getattr(memory, '__file__', '<extension submodule>')}")


@pytest.fixture
def hardware():
    """Opaque identifiers deliberately have no CPU hierarchy interpretation."""
    pool_ids = ["uma/bank:0", "cowos_superdie/bank:1"]
    agent = make(
        "Agent", id="opaque-worker", capabilities=["host_load_store"], address_space="host"
    )
    pools = []
    groups = []
    resources = []
    access = []
    for index, pool_id in enumerate(pool_ids):
        group = f"capacity@{index}"
        service = f"infinity_fabric/service:{index}"
        pools.append(
            make(
                "MemoryPool", id=pool_id, capacity_bytes=16 * PAGE, address_space="host",
                granularity_bytes=PAGE, page_sizes_bytes=[PAGE], read_service=service,
                write_service=service, capacity_group=group,
            )
        )
        groups.append(make("CapacityGroup", id=group, capacity_bytes=16 * PAGE))
        resources.append(
            make(
                "Resource", id=service, rule="arbitrary-service", read=100.0, write=50.0,
                sharing=memory.SharingMode.read_write, kind=memory.ResourceKind.bandwidth,
            )
        )
        access.append(
            make(
                "AccessConnection", id=f"direct:{index}", rule="arbitrary-path",
                agent=agent.id, pool=pool_id,
                access=[memory.AccessMode.read, memory.AccessMode.write],
                mapping=memory.MappingMode.direct,
            )
        )
    routes = [
        make("Route", id=f"route:{a}:{b}", source=a, target=b, demands=[])
        for a in pool_ids for b in pool_ids
    ]
    graph = make(
        "ResourceGraph", agents=[agent], pools=pools, capacity_groups=groups,
        resources=resources, access=access,
        methods=[make("Method", id="opaque-copy", request_executor=True,
                      bandwidth=20.0, startup_us=2.0, routes=routes)],
        domains=[make("AllocationDomain", id="some-domain", rule="user-rule", members=pool_ids)],
    )
    model = memory.ResourceModel(graph)
    scope = make(
        "Scope", executable_agents=[agent.id], allocatable_pools=pool_ids,
        accessible_pools=pool_ids, execution_slots={agent.id: 1},
    )
    return model, model.view(scope, 7), pool_ids


def placement(model, view, pools, pages=1, weights=None):
    plan = make(
        "PlacementPlan", address_space="host", stripe_bytes=PAGE,
        weights=weights or {pools[0]: 1}, cycle=[], phase=0,
    )
    return model.resolve_placement(view, plan, pages * PAGE)


def recipe_for_placement(resolved):
    """Map one auditable VA offset region onto the resolved physical pools."""
    size = sum(segment.size_bytes for segment in resolved.segments)
    sizes = Counter()
    for segment in resolved.segments:
        sizes[segment.pool] += segment.size_bytes
    backing_ids = {pool: f"backing:{index}" for index, pool in enumerate(sorted(sizes))}
    backings = [
        make("BackingRequest", id=backing_ids[pool], owner="p0", pool=pool, bytes=count,
             page_bytes=PAGE, mode=memory.BackingMode.shareable_remappable, export_to=[])
        for pool, count in sorted(sizes.items())
    ]
    next_offset = Counter()
    mappings = []
    for segment in resolved.segments:
        mappings.append(
            make(
                "PageMapping", region="region", virtual_offset_bytes=segment.offset_bytes,
                backing=backing_ids[segment.pool], backing_offset_bytes=next_offset[segment.pool],
                size_bytes=segment.size_bytes,
                access=[memory.AccessMode.read, memory.AccessMode.write],
            )
        )
        next_offset[segment.pool] += segment.size_bytes
    addresses = make(
        "AddressLayout",
        regions=[make("VirtualRegion", id="region", address_space="host", bytes=size, page_bytes=PAGE)],
        slots=[make("AddressSlot", id="buffer", region="region", offset_bytes=0,
                    size_bytes=size, placement=resolved)],
    )
    return make(
        "MemoryRecipe", participant="p0", layout_generation=1, sharing_contracts=[],
        addresses=addresses, backings=backings, mappings=mappings,
    )


@pytest.mark.parametrize("value", [0, (1 << 53) + 1, (1 << 63) + 17, UINT64_MAX])
def test_uint64_fields_and_variants_are_exact(value):
    segment = make("PlacedSegment", offset_bytes=value, size_bytes=value, pool="opaque")
    mapping = make("PageMapping", virtual_offset_bytes=value, backing_offset_bytes=value)
    token = make("CompletionToken", binding_generation=value)
    limit = make("Limit", bytes_or_capacity_fraction=value)
    initial = make("Preallocate", bytes_or_limit_fraction=value)
    assert segment.offset_bytes == segment.size_bytes == value
    assert mapping.virtual_offset_bytes == mapping.backing_offset_bytes == value
    assert token.binding_generation == value
    assert limit.bytes_or_capacity_fraction == value
    assert initial.bytes_or_limit_fraction == value
    limit.bytes_or_capacity_fraction = 0.75
    initial.bytes_or_limit_fraction = 0.25
    assert limit.bytes_or_capacity_fraction == 0.75
    assert initial.bytes_or_limit_fraction == 0.25


@pytest.mark.parametrize("invalid", [-1, 1 << 64, 1.5])
def test_uint64_offsets_reject_out_of_range_or_fractional_values(invalid):
    segment = memory.PlacedSegment()
    with pytest.raises((TypeError, OverflowError)):
        segment.offset_bytes = invalid


def test_fact_booleans_are_not_coerced_to_integer_variant():
    facts = {"yes": True, "no": False, "count": UINT64_MAX, "kind": "opaque"}
    inventory = make("HardwareInventory", facts=facts)
    expansion = make("HardwareExpansion", requirements=facts)
    for result in [inventory.facts, expansion.requirements]:
        assert type(result["yes"]) is bool
        assert type(result["no"]) is bool
        assert result["yes"] is True and result["no"] is False
        assert type(result["count"]) is int and result["count"] == UINT64_MAX
    with pytest.raises((TypeError, OverflowError, RuntimeError)):
        inventory.facts = {"negative_count": -1}
    with pytest.raises(TypeError):
        inventory.facts = {"floating_count": 1.5}


def test_graph_snapshot_cannot_mutate_immutable_resource_model(hardware):
    model, view, pools = hardware
    assert isinstance(model, memory.MemoryCostModel)
    snapshot = model.graph
    snapshot.pools = []
    snapshot.agents = []
    snapshot.capacity_groups = []
    assert [pool.id for pool in model.graph.pools] == pools
    assert model.graph.agents[0].id == "opaque-worker"
    assert placement(model, view, pools).segments[0].pool == pools[0]


def test_weighted_placement_and_recipe_layout_toml_roundtrip(hardware):
    model, view, pools = hardware
    resolved = placement(model, view, pools, pages=10, weights={pools[0]: 3, pools[1]: 7})
    totals = Counter()
    cursor = 0
    for segment in resolved.segments:
        assert segment.offset_bytes == cursor
        assert segment.offset_bytes % PAGE == segment.size_bytes % PAGE == 0
        cursor += segment.size_bytes
        totals[segment.pool] += segment.size_bytes
    assert cursor == 10 * PAGE
    assert totals == {pools[0]: 3 * PAGE, pools[1]: 7 * PAGE}
    assert model.compatible_overlap(resolved, 0, resolved, 0, cursor)
    recipe = recipe_for_placement(resolved)
    model.validate_recipe(view, recipe)
    encoded = memory.recipe_to_toml(recipe)
    decoded = memory.recipe_from_toml(encoded)
    model.validate_recipe(view, decoded)
    assert memory.recipe_to_toml(decoded) == encoded
    assert [(item.virtual_offset_bytes, item.backing_offset_bytes) for item in decoded.mappings] == [
        (item.virtual_offset_bytes, item.backing_offset_bytes) for item in recipe.mappings
    ]
    layout = make(
        "DistributedLayout", id="opaque-layout", generation=1, backings=recipe.backings,
        participants=[make("ParticipantLayout", participant="p0", addresses=recipe.addresses,
                           mappings=recipe.mappings)],
    )
    text = memory.layout_to_toml(layout)
    assert memory.layout_to_toml(memory.layout_from_toml(text)) == text


def test_invalid_recipe_and_aliasing_command_report_cpp_errors(hardware):
    model, view, pools = hardware
    recipe = recipe_for_placement(placement(model, view, pools))
    mappings = recipe.mappings
    mappings[0].backing = "missing-backing"
    recipe.mappings = mappings
    with pytest.raises(RuntimeError, match=r"^[\w.]+: .+"):
        model.validate_recipe(view, recipe)
    recipe = recipe_for_placement(placement(model, view, pools))
    environment = model.describe_operands(view, recipe)
    reference = make("ByteRangeRef", kind=memory.MemoryRefKind.slot, memory="buffer",
                     offset_bytes=0, size_bytes=PAGE)
    command = make(
        "MemoryCommand", id="overlapping-copy", opcode=memory.MemoryOpcode.copy,
        source=reference, target=reference, method="opaque-copy", queue="opaque-worker",
    )
    with pytest.raises(RuntimeError, match=r"^[\w.]+: .*overlap"):
        model.validate_commands(view, make("MemoryCommandPlan", commands=[command]), environment)
    with pytest.raises(RuntimeError, match=r"^[\w.]+: .+"):
        memory.recipe_from_toml("invalid = [")


def test_opaque_hardware_template_names_and_boolean_requirement_binding():
    text = '''schema = "memory-model/0.3"
name = "user-defined topology"
[units]
bandwidth = "GiB/s"
[require]
coherent = true
[groups]
uma = { count = "runtime" }
cowos_superdie = { per = "uma", count = 2 }
[agents]
custom_compute = { per = "cowos_superdie", bind = "custom-executor" }
[pools]
custom_memory = { per = "custom_compute", capacity = "discover", bind = "custom-pool", service = "infinity_fabric" }
[resources]
infinity_fabric = { per = "cowos_superdie", read = 100, write = 80 }
[paths]
custom_rule = { agent = "custom_compute", pool = "custom_memory", within = "uma", where = "any" }
'''
    expansion = memory.expand_hardware_template(text, {"uma": 1})
    assert len(expansion.graph.pools) == 2
    assert {edge.rule for edge in expansion.graph.access} == {"custom_rule"}
    inventory = make("HardwareInventory", facts={"coherent": True})
    inventory.bindings = {
        name: make("HardwareBinding", address_space="host", capabilities=["host_load_store"])
        if role == "custom-executor" else
        make("HardwareBinding", capacity_bytes=16 * PAGE, address_space="host",
             granularity_bytes=PAGE, page_sizes_bytes=[PAGE])
        for name, role in expansion.binding_requests.items()
    }
    assert len(memory.bind_hardware_inventory(expansion, inventory).pools) == 2
    assert len(memory.compile_hardware_template(text, {"uma": 1}, inventory).pools) == 2
    with pytest.raises(RuntimeError, match=r"^[\w.]+: .+"):
        memory.expand_hardware_template(text, {"socket": 1})


def test_pure_transfer_and_access_cost_queries(hardware):
    model, view, pools = hardware
    source = make("TransferEndpoint", address_space="host",
                  segments=placement(model, view, pools).segments)
    context = make("CostContext", view_epoch=view.epoch, execution_slots={"opaque-worker": 1},
                   include_preparation=False)
    candidates = model.candidates(view, source, source)
    assert candidates[0].method == "opaque-copy"
    query = make("TransferQuery", source=source, target=source, method="opaque-copy",
                 executor="opaque-worker", payload_bytes=PAGE)
    assert model.estimate(view, query, context).total_us > 0
    access = make("AccessQuery", agent="opaque-worker", memory=source,
                  access=memory.AccessMode.read, payload_bytes=PAGE)
    assert model.estimate_access(view, access, context).total_us >= 0
    assert memory.parse_memory_size("18446744073709551615 B") == UINT64_MAX
