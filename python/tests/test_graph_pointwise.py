# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Static broadcast lowering, dtype boundaries, aliases and native islands."""
import torch
import pytest
from vecops.graph import GraphCapture


@pytest.fixture(autouse=True)
def native_pool(monkeypatch):
  monkeypatch.setenv('VECOPS_THREAD_POOL', 'native')


def test_shared_preallocated_scratch_is_compile_only_and_outputs_are_independent():
  arena = torch.empty(4096, dtype=torch.uint8)
  requests = []
  def allocate(size):
    requests.append(size)
    return arena
  functions = (lambda x: ((x + 1).relu() * 2).sum(-1),
               lambda x: ((x - 3).relu() * .5).sum(-1))
  programs = []
  with torch.inference_mode():
    for function in functions:
      capture = GraphCapture('shared-static-arena')
      capture.capture_torch(function, torch.ones(3, 17))
      programs.append(capture.compile_torch(native_pointwise=True, scratch_allocator=allocate))
    assert len(requests) == 2 and all(size <= arena.numel() for size in requests)
    previous = []
    for index in range(4):
      for function, program in zip(functions, programs):
        x = torch.ones(3, 17) * index
        result = program(x)
        torch.testing.assert_close(result, function(x))
        previous.append((result, result.clone()))
    assert len(requests) == 2
    for result, saved in previous:
      torch.testing.assert_close(result, saved)
    assert all(p.allocator_policy['static_scratch_arena_bytes'] == 4096 for p in programs)


def test_preallocated_scratch_capacity_is_checked_before_executor_construction():
  with torch.inference_mode():
    capture = GraphCapture('short-static-arena')
    capture.capture_torch(lambda x: (x + 1).sum(-1), torch.ones(3, 17))
    with pytest.raises(ValueError, match='sufficient capacity'):
      capture.compile_torch(native_pointwise=True,
          scratch_allocator=lambda _: torch.empty(1, dtype=torch.uint8))


def test_borrowed_outputs_preserve_aliases_and_avoid_hot_factories():
  def function(x):
    temporary = torch.empty_like(x)
    temporary.copy_(x)
    output = (temporary + 1).relu()
    return output, output.view(3, 17)
  with torch.inference_mode():
    x = torch.ones(3, 17)
    capture = GraphCapture('borrowed-output-alias')
    capture.capture_torch(function, x)
    program = capture.compile_torch(native_pointwise=True, borrow_outputs=True)
    first, alias = program(x)
    saved, address = first.clone(), first.data_ptr()
    assert first.data_ptr() == alias.data_ptr()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
      second, _ = program(x)
    assert second.data_ptr() == address
    assert not {'aten::empty_like', 'aten::copy_', 'aten::relu'}.intersection(
        item.key for item in profile.key_averages())
    last, _ = program(x * 3)
    torch.testing.assert_close(last, x * 4)
    torch.testing.assert_close(first, last)
    torch.testing.assert_close(saved, x * 2)
    assert program.allocator_policy['retained_output_arena_bytes'] > 0
    # Reusing the previous output as input changes metadata/alias contracts.
    # Reject it before execution instead of allowing an unsafe overwrite.
    with pytest.raises(RuntimeError, match='guard'):
      program(last)


def test_borrowed_outputs_are_separate_from_shared_temporary_scratch():
  arena = torch.empty(4096, dtype=torch.uint8)
  with torch.inference_mode():
    programs = []
    for offset in (1, 3):
      capture = GraphCapture('private-borrowed-output')
      capture.capture_torch(lambda x: (x + offset).relu(), torch.ones(3, 17))
      programs.append(capture.compile_torch(native_pointwise=True, borrow_outputs=True,
          scratch_allocator=lambda _: arena))
    first = programs[0](torch.ones(3, 17))
    saved = first.clone()
    for index in range(3): programs[1](torch.ones(3, 17) * index)
    torch.testing.assert_close(first, saved)
    assert first.untyped_storage().data_ptr() != arena.untyped_storage().data_ptr()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_pointwise_broadcast_rebinding_and_escaping_outputs(dtype):
  torch.set_num_threads(4)
  x = torch.randn(3, 2, 37).to(dtype)
  b = torch.randn(37).to(dtype)
  def function(x, b):
    return ((x + b).relu().float() * .25 - 1).sum(-1)
  with torch.inference_mode():
    capture = GraphCapture('broadcast-dtype')
    capture.capture_torch(function, x, b)
    replay = capture.compile_torch(native_pointwise=True)
    assert replay.allocator_policy['native_pointwise_nodes'] >= 3
    assert replay._executor.native_islands() >= 1
    first = replay(x, b)
    saved = first.clone()
    for i in range(3):
      actual = replay(x + i, b - i * .1)
      torch.testing.assert_close(actual, function(x + i, b - i * .1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(first, saved, rtol=0, atol=0)


def test_static_views_join_island_and_dynamic_aliases_follow_new_storage():
  torch.set_num_threads(4)
  def function(x):
    a = (x + 1).view(8, 17)
    return (a.relu() * x.view(8, 17)).sum()
  with torch.inference_mode():
    capture = GraphCapture('view-join')
    capture.capture_torch(function, torch.randn(4, 34))
    replay = capture.compile_torch(native_pointwise=True)
    assert replay.allocator_policy['hoisted_static_views'] == 2
    assert replay._executor.native_islands() == 1
    for i in range(3):
      x = torch.randn(4, 34) + i
      torch.testing.assert_close(replay(x), function(x))


def test_scalar_input_views_hoist_typed_empty_geometry_and_rebind():
  def function(x):
    first, second = x.unbind(0)
    return first, second, first+second
  with torch.inference_mode():
    capture = GraphCapture('scalar-input-alias')
    capture.capture_torch(function, torch.tensor([1., 3.]))
    replay = capture.compile_torch(native_pointwise=True)
    assert replay.allocator_policy['hoisted_static_views'] == 1
    assert 'aten::unbind' not in replay.ir
    for x in (torch.tensor([2., 7.]), torch.tensor([-3., .5])):
      first, second, total = replay(x)
      torch.testing.assert_close(total, x.sum(), rtol=0, atol=0)
      assert first.shape == second.shape == ()
      assert first.data_ptr() == x.data_ptr()
      assert second.data_ptr() == x.data_ptr()+x.element_size()
      x.add_(1)
      torch.testing.assert_close(first, x[0], rtol=0, atol=0)
      torch.testing.assert_close(second, x[1], rtol=0, atol=0)


def test_multi_output_and_composite_views_use_captured_alias_geometry():
  def function(x):
    a = x + 1
    broadcast = a.unflatten(-1, (2, 17)).movedim(0, 1).broadcast_to(3, 2, 4, 17)
    _, tail = a.split((1, 3), dim=0)
    first, second, third = tail.unbind(0)
    return broadcast.relu().sum() + first.relu().sum() + second.relu().sum() + third.relu().sum()
  with torch.inference_mode():
    capture = GraphCapture('composite-multi-views')
    capture.capture_torch(function, torch.ones(4, 34))
    program = capture.compile_torch(native_pointwise=True)
    assert program.allocator_policy['hoisted_static_views'] == 5
    assert all(name not in program.ir for name in
        ('aten::broadcast_to', 'aten::unflatten', 'aten::movedim', 'aten::split_with_sizes', 'aten::unbind'))
    for index in range(3):
      x = torch.randn(4, 34) * index
      torch.testing.assert_close(program(x), function(x))


def test_noncontiguous_flatten_keeps_its_real_copy():
  def function(x):
    a = x + 1
    return a.transpose(0, 1).flatten().relu().sum()
  with torch.inference_mode():
    capture = GraphCapture('flatten-real-copy')
    capture.capture_torch(function, torch.ones(4, 34))
    program = capture.compile_torch(native_pointwise=True)
    assert 'aten::flatten' in program.ir
    for index in range(3):
      x = torch.randn(4, 34) * index
      torch.testing.assert_close(program(x), function(x))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_strided_outer_copy_joins_native_consumer_without_losing_rows(dtype):
  def function(x):
    owner = torch.empty((2, 3, 17), dtype=dtype)
    destination = owner.transpose(0, 1)
    destination.copy_(x)
    return (owner + 1).sum(-1)
  with torch.inference_mode():
    capture = GraphCapture('strided-destination-copy')
    capture.capture_torch(function, torch.ones(3, 2, 17, dtype=dtype))
    program = capture.compile_torch(native_pointwise=True)
    assert program.allocator_policy['native_pointwise_inplace_nodes'] == 1
    assert 'aten::copy_' not in program.ir
    assert program.allocator_policy['native_island_sizes'] == [2]
    for index in range(3):
      x = (torch.randn(3, 2, 17) + index).to(dtype)
      torch.testing.assert_close(program(x), function(x), rtol=0, atol=0)


def test_to_dtype_conversions_join_native_island_without_losing_rounding():
  def function(x):
    return ((x + 1).float().contiguous().bfloat16().float() * .25).sum(-1)
  with torch.inference_mode():
    capture = GraphCapture('to-dtype')
    capture.capture_torch(function, torch.randn(3, 37))
    replay = capture.compile_torch(native_pointwise=True)
    assert replay.allocator_policy['native_pointwise_nodes'] == 4
    assert replay.allocator_policy['eliminated_static_noops'] >= 1
    assert replay._executor.native_islands() == 1
    for _ in range(3):
      x = torch.randn(3, 37)
      torch.testing.assert_close(replay(x), function(x), rtol=0, atol=0)


def test_relative_as_strided_of_fixed_scratch_does_not_split_native_team():
  def function(x):
    return ((x + 1).as_strided((8, 17), (17, 1)) * 2).sum()
  with torch.inference_mode():
    capture = GraphCapture('as-strided-prologue')
    capture.capture_torch(function, torch.randn(4, 34))
    replay = capture.compile_torch(native_pointwise=True)
    assert replay._executor.prologue_views() >= 1
    assert replay._executor.native_islands() == 1
    for _ in range(3):
      x = torch.randn(4, 34)
      torch.testing.assert_close(replay(x), function(x), rtol=0, atol=0)


def test_relu_keeps_nan_and_negative_zero_and_noncontiguous_falls_back():
  torch.set_num_threads(4)
  x = torch.tensor([-0., float('nan'), -1., float('inf')])
  def function(x):
    return (x.clone().relu(), x.t().clone())
  with torch.inference_mode():
    capture = GraphCapture('special-values')
    capture.capture_torch(lambda x: x.relu().view(2, 2).clone(), x)
    replay = capture.compile_torch(native_pointwise=True)
    actual = replay(x)
    assert actual[0, 0].signbit() and actual[0, 1].isnan()
    torch.testing.assert_close(actual, x.relu().view(2, 2), equal_nan=True)


def test_static_factories_keep_live_aliases_and_reuse_dead_storage():
  def function(x):
    a = torch.empty_like(x)
    a.copy_(x)
    alias = a.view(-1)
    result = (alias + 2).sum()
    b = torch.empty_like(x)
    b.copy_(x * 3)
    return result + b.sum()
  with torch.inference_mode():
    capture = GraphCapture('static-factory-liveness')
    capture.capture_torch(function, torch.randn(4, 64))
    replay = capture.compile_torch(native_pointwise=True)
    assert replay.allocator_policy['static_factory_count'] == 2
    assert replay.allocator_policy['static_scratch_arena_bytes'] < 4 * 4 * 64 * 4
    for i in range(3):
      x = torch.randn(4, 64) + i
      torch.testing.assert_close(replay(x), function(x))


def test_recapture_reuses_artifacts_but_owns_independent_scratch(monkeypatch):
  import vecops
  calls = []
  original = vecops.compile_batch
  def compile_batch(*args, **kwargs):
    calls.append(1)
    return original(*args, **kwargs)
  monkeypatch.setattr(vecops, 'compile_batch', compile_batch)
  def function(x, scalar):
    return (x * scalar + 1).sum()
  with torch.inference_mode():
    a = GraphCapture('first-inference')
    a.capture_torch(function, torch.ones(7, 29).bfloat16(), torch.tensor(1.0031))
    first = a.compile_torch(native_pointwise=True)
    count = len(calls)
    b = GraphCapture('second-inference')
    b.capture_torch(function, torch.ones(7, 29).bfloat16(), torch.tensor(1.0031))
    second = b.compile_torch(native_pointwise=True)
    assert len(calls) == count
    x = torch.randn(7, 29).bfloat16()
    for scalar in (1.0031, .0171, -3.1):
      s = torch.tensor(scalar)
      torch.testing.assert_close(first(x, s), function(x, s), rtol=0, atol=0)
      torch.testing.assert_close(second(x + 1, s), function(x + 1, s), rtol=0, atol=0)


def test_inplace_copy_broadcast_and_arithmetic_keep_output_and_input_aliases():
  def function(x, row, scalar):
    out = torch.empty_like(x)
    out.copy_(row)
    out.add_(x)
    out.mul_(.25)
    out.div_(scalar)
    return out
  with torch.inference_mode():
    x, row, scalar = torch.randn(3, 37), torch.randn(37), torch.tensor(2.)
    capture = GraphCapture('inplace')
    capture.capture_torch(function, x, row, scalar)
    replay = capture.compile_torch(native_pointwise=True)
    assert replay.allocator_policy['native_pointwise_inplace_nodes'] == 4
    assert replay._executor.native_islands() == 1
    first = replay(x, row, scalar)
    saved = first.clone()
    for i in range(3):
      torch.testing.assert_close(replay(x+i, row-i, scalar), function(x+i, row-i, scalar))
    torch.testing.assert_close(first, saved)

    def mutate(x, row):
      x.add_(row)
      return x.view(3, 37)
    capture = GraphCapture('input-mutation')
    capture.capture_torch(mutate, x.clone(), row)
    replay = capture.compile_torch(native_pointwise=True)
    new_input = x.clone()
    actual = replay(new_input, row)
    assert actual.data_ptr() == new_input.data_ptr()
    torch.testing.assert_close(actual, x + row)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_boolean_feature_broadcast_preserves_mul_and_conversion_semantics(dtype):
  torch.set_num_threads(4)
  mask = torch.tensor([[True, False, True], [False, True, False]])[..., None]
  x = torch.randn(2, 3, 17).to(dtype)
  x[0, 1, :3] = torch.tensor([float('nan'), -0., -float('inf')]).to(dtype)
  def function(value, keep):
    # Both operand orders plus an explicit bool -> floating boundary. Keeping
    # the mask as bool prevents an extra full-size converted-mask temporary.
    return keep * value, value * keep, keep.to(dtype) + 1
  with torch.inference_mode():
    capture = GraphCapture('boolean-feature-broadcast')
    capture.capture_torch(function, x, mask)
    program = capture.compile_torch(native_pointwise=True, borrow_outputs=True)
    assert program.allocator_policy['native_pointwise_nodes'] == 4
    for keep in (mask, ~mask):
      expected = function(x, keep)
      actual = program(x, keep)
      for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(torch.signbit(got[got == 0]), torch.signbit(want[want == 0]))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
      program(x, mask)
    assert 'aten::mul' not in {item.key for item in profile.key_averages()}


def test_boolean_vector_input_keeps_general_aten_path():
  with torch.inference_mode():
    x = torch.randn(3, 17)
    mask = torch.ones_like(x, dtype=torch.bool)
    capture = GraphCapture('boolean-vector-fallback')
    capture.capture_torch(lambda a, b: a * b, x, mask)
    program = capture.compile_torch(native_pointwise=True, borrow_outputs=True)
    assert program.allocator_policy['native_pointwise_nodes'] == 0
    torch.testing.assert_close(program(x, mask), x)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('broadcast', [False, True])
def test_same_dtype_copy_preserves_raw_nan_payloads(dtype, broadcast):
  # Include signaling NaNs: widening/narrowing a copy can quiet/canonicalize
  # them even though no arithmetic or dtype conversion was requested.
  words = ([0x7f81, 0xff81, 0x8000] if dtype == torch.bfloat16 else
           [0x7c01, 0xfc01, 0x8000])
  x = torch.tensor([w if w < 0x8000 else w - 0x10000 for w in words],
                   dtype=torch.int16).view(dtype).reshape(3, 1)
  if not broadcast:
    x = x.expand(3, 17).clone()
  def function(value):
    result = torch.empty((3, 17), dtype=dtype)
    result.copy_(value)
    return result
  with torch.inference_mode():
    capture = GraphCapture('bitwise-copy-payload')
    capture.capture_torch(function, x)
    program = capture.compile_torch(native_pointwise=True, borrow_outputs=True)
    assert program.allocator_policy['native_pointwise_inplace_nodes'] == 1
    assert torch.equal(program(x).view(torch.int16), function(x).view(torch.int16))
