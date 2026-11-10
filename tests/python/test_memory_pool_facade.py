# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Eager pool ownership shares the same native root and capacity authority."""
import gc

import pytest
import torch

from test_memory_runtime import runtime, arena, committed, Q
from vecops.memory import Pool, Runtime, current_runtime, tensor_view, use_runtime


def test_pool_tensor_aliases_retain_the_native_block_until_last_storage_owner(runtime):
    pool = Pool(arena(runtime,4))
    tensor = pool.empty((16,),dtype=torch.float32,preferred_pool='a')
    pointer = tensor.data_ptr()
    assert pointer % pool.page_bytes == 0
    tensor.fill_(17)
    alias = tensor[2:]
    del tensor
    gc.collect()
    assert pool.trim() == 0
    assert alias[0].item() == 17
    del alias
    gc.collect()
    assert pool.trim() == 4*Q
    assert committed(runtime) == 0


def test_eager_cached_block_is_reused_and_strided_views_have_the_required_span(runtime):
    pool = Pool(arena(runtime,4))
    with torch.device('meta'):
        first = pool.empty((3,4),dtype=torch.float32,strides=(8,1))
    assert first.device.type == 'cpu'
    pointer = first.data_ptr()
    first.fill_(23)
    assert first.stride() == (8,1)
    del first
    gc.collect()
    second = pool.empty((12,),dtype=torch.float32)
    assert second.data_ptr() == pointer
    assert second.shape == (12,)


def test_active_runtime_context_restores_even_after_an_exception(runtime):
    wrapped = Runtime(runtime)
    before = current_runtime()
    with pytest.raises(ValueError):
        with use_runtime(wrapped) as selected:
            assert selected is current_runtime() is wrapped
            assert wrapped.snapshot().groups
            raise ValueError('unwind')
    assert current_runtime() is before


def test_zero_tensor_and_invalid_shapes_do_not_consume_new_capacity(runtime):
    pool = Pool(arena(runtime,4))
    initial = committed(runtime)
    with torch.device('meta'):
        empty = pool.empty((0,16))
        strided = pool.empty((0,16),strides=(32,1))
        view = tensor_view(None,shape=(0,16),dtype=torch.float32)
        strided_view = tensor_view(None,shape=(0,16),dtype=torch.float32,strides=(32,1))
    assert all(t.device.type == 'cpu' and t.numel() == 0
               for t in (empty,strided,view,strided_view))
    assert strided.stride() == strided_view.stride() == (32,1)
    for shape in [(True,1),(-1,), (1.5,)]:
        with pytest.raises(ValueError): pool.empty(shape)
    with pytest.raises(ValueError): pool.empty((3,4),strides=(1,))
    with pytest.raises(OverflowError): pool.empty((1<<63,),dtype=torch.float32)
    assert committed(runtime) == initial


def test_runtime_contract_excludes_live_capacity_and_preserves_opaque_resources(runtime):
    import json
    wrapped = Runtime(runtime)
    before = wrapped.contract()
    json.dumps(before, allow_nan=False)
    pool = Pool(arena(runtime,4))
    tensor = pool.empty((64,),dtype=torch.float32)
    assert wrapped.contract() == before
    assert [item['id'] for item in before['graph']['pools']] == ['a','b']
    assert before['limits'] == [('cap-a',32*Q),('cap-b',32*Q)]
    del tensor
    pool.trim()
    assert wrapped.contract() == before


def test_numpy_alias_retains_native_pool_root_and_returns_capacity(runtime):
    import numpy as np
    from vecops.memory import numpy as memory_numpy
    pool = Pool(arena(runtime,4))
    array = memory_numpy.zeros((7,9),dtype=np.float32,pool=pool,preferred_pool='a')
    assert array.ctypes.data % pool.page_bytes == 0
    assert np.count_nonzero(array) == 0
    alias = array[2:,1:]
    alias.fill(19)
    del array
    gc.collect()
    assert pool.trim() == 0
    np.testing.assert_array_equal(alias,19)
    del alias
    gc.collect()
    assert pool.trim() == 4*Q
    assert committed(runtime) == 0


def test_numpy_invalid_and_zero_requests_do_not_acquire_capacity(runtime):
    from vecops.memory import numpy as memory_numpy
    pool = Pool(arena(runtime,4))
    initial = committed(runtime)
    assert memory_numpy.empty((0,16),pool=pool).size == 0
    for shape in [(True,1),(-1,), (1.5,)]:
        with pytest.raises(ValueError): memory_numpy.empty(shape,pool=pool)
    with pytest.raises(ValueError,match='object-containing'):
        memory_numpy.empty((3,),dtype='object',pool=pool)
    with pytest.raises(OverflowError): memory_numpy.empty((1<<63,),pool=pool)
    assert committed(runtime) == initial
