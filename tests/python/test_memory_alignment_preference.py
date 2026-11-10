# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Soft alignment remains a cold layout policy under real capacity admission."""
import pytest
from test_memory_graph_compiler import Buffer, Q, compiler, native, packing_runtime


def plan(runtime, requirements, preference=None):
    value,view=runtime
    return compiler.GraphMemoryCompiler(value,view,value.snapshot(view),native_module=native,
        preferred_alignment_bytes=preference).compile(requirements)


@pytest.mark.parametrize('preference',[True,False,0,-1,3,1.5,'2097152',1<<64])
def test_preference_requires_a_positive_uint64_power_of_two(preference):
    with pytest.raises(ValueError,match='preferred alignment'):
        compiler.GraphMemoryCompiler(None,None,None,preferred_alignment_bytes=preference)


def requests():
    return [Buffer('prefix',64,read_bytes=1,allowed_pools=('uma',)),
            Buffer('large',1<<20,allowed_pools=('uma',))]


def test_none_preserves_logical_page_policy_and_hint_changes_only_cold_layout():
    runtime=packing_runtime(capacity_bytes=8<<20)
    original=plan(runtime,requests())
    explicit_page=plan(runtime,requests(),Q)
    larger=plan(runtime,requests(),2<<20)
    assert original.dumps()==explicit_page.dumps()
    assert original.bindings['large'].view_offset_bytes==Q
    assert larger.bindings['large'].view_offset_bytes==2<<20
    assert larger.bindings['large'].alignment==64
    assert larger.memory.addresses.regions[0].page_bytes==Q
    assert larger.memory.backings[0].page_bytes==Q
    assert larger.memory.addresses.regions[0].base_alignment_bytes==2<<20
    assert sum(g.committed_bytes for g in runtime[0].snapshot(runtime[1]).groups)==0


def test_soft_padding_compacts_in_preferred_pool_before_spilling():
    program=plan(packing_runtime(capacity_bytes=(1<<20)+Q),requests(),2<<20)
    assert program.bindings['large'].view_offset_bytes==64
    assert {root.pool for root in program.memory.backings}=={'uma'}
    assert sum(program.peak_bytes_by_capacity_group.values())==(1<<20)+Q
    assert program.memory.addresses.regions[0].base_alignment_bytes==2<<20


def test_small_buffers_remain_compact_and_hint_never_weakens_mapping_quantum():
    runtime=packing_runtime(page_bytes=(65536,65536),capacity_bytes=8<<20)
    small=plan(runtime,[Buffer('prefix',64,read_bytes=1,allowed_pools=('uma',)),
        Buffer('small',(1<<20)-1,allowed_pools=('uma',))],2<<20)
    assert small.bindings['small'].view_offset_bytes==64
    assert small.memory.addresses.regions[0].base_alignment_bytes==65536
    large=plan(runtime,requests(),Q)
    assert large.bindings['large'].view_offset_bytes==65536
