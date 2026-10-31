# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Storage lifetimes must outlast Python Tensor wrappers and plan sessions."""

import gc

import pytest

torch = pytest.importorskip("torch")
from vecops import graph, planner
from vecops.planner._pool import TensorPool


def test_persistent_activation_reuses_arena_and_deactivates_after_exception():
  class System:
    allocations = 0
    def current_cpu_domain(self): return 0
    def allocate(self, size, **kwargs):
      self.allocations += 1
      return bytearray(size)
  system = System()
  execution = graph.GraphCapture('persistent-activation').finish()
  machine = planner.MachineProfile(fast_capacity=128, slow_capacity=1024,
      fast_bandwidth_mib_s=100, slow_bandwidth_mib_s=10,
      transfer_bandwidth_mib_s=10, alignment=64)
  plan = planner.plan_memory(execution, machine)
  session = planner.PlanSession(execution, plan, system=system, enable_dynamic_factories=True)
  with session.activation():
    original = session._dynamic_pool.owner
    value = planner.empty(16, dtype=torch.float32).fill_(7)
    with pytest.raises(RuntimeError, match='overlap'):
      with session.activation(): pass
  assert planner.current_session() is None
  with pytest.raises(ValueError, match='failure'):
    with session.activation():
      assert session._dynamic_pool.owner is original
      raise ValueError('failure')
  assert planner.current_session() is None and not session._closed
  with session.activation():
    assert system.allocations == 1
    assert torch.equal(value, torch.full((16,), 7.))
  session.close()
  assert torch.equal(value, torch.full((16,), 7.))
  with pytest.raises(RuntimeError, match='closed'):
    with session.activation(): pass


def test_executable_capture_records_factories_instead_of_snapshotting_eager_pool():
  class System:
    def current_cpu_domain(self): return 0
    def allocate(self, size, **kwargs): return bytearray(size)
  def function(x):
    first = planner.empty_like(x)
    first.copy_(x)
    second = planner.empty_like(x)
    second.copy_(first + 1)
    return second * 2
  import vecops
  x = torch.ones(8)
  execution = vecops.precompile(function, x, capture_graph=True).graph
  machine = planner.MachineProfile(fast_capacity=1024, slow_capacity=4096,
      fast_bandwidth_mib_s=100, slow_bandwidth_mib_s=10,
      transfer_bandwidth_mib_s=10, alignment=64)
  plan = planner.plan_memory(execution, machine)
  with torch.inference_mode(), planner.PlanSession(execution, plan, system=System(),
      enable_dynamic_factories=True) as session:
    before = session.stats()['dynamic_allocations']
    capture = graph.GraphCapture('explicit-executable-factories')
    from vecops.planner.runtime import executable_factories
    with executable_factories():
      capture.capture_torch(function, x)
    replay = capture.compile_torch(native_pointwise=True)
    assert session.stats()['dynamic_allocations'] == before
    assert replay.allocator_policy['captured_arena_copy_bytes'] == 0
    assert replay.allocator_policy['static_factory_count'] == 2
    for index in range(3):
      torch.testing.assert_close(replay(x + index), (x + index + 1) * 2)


def test_persistent_activation_restarts_factory_ordinals_without_reallocating():
  import vecops
  class System:
    allocations = 0
    def current_cpu_domain(self): return 0
    def allocate(self, size, **kwargs):
      self.allocations += 1
      return bytearray(size)
  def function(x):
    temporary = planner.empty_like(x)
    temporary.copy_(x)
    return temporary + 1
  x = torch.ones(16)
  execution = vecops.precompile(function, x, capture_graph=True).graph
  machine = planner.MachineProfile(fast_capacity=4096, slow_capacity=4096,
      fast_bandwidth_mib_s=100, slow_bandwidth_mib_s=10,
      transfer_bandwidth_mib_s=10, alignment=64)
  plan = planner.plan_memory(execution, machine, config=planner.PlannerConfig(
      minimum_fast_bytes=1, require_replayable_temporaries=True))
  system = System()
  session = planner.PlanSession(execution, plan, system=system,
      enable_factory_replay=True, strict_factory_replay=True)
  try:
    for index in range(3):
      with session.activation():
        torch.testing.assert_close(function(x * index), x * index + 1)
        with pytest.raises(RuntimeError, match='epoch_exhausted'):
          function(x)
    assert system.allocations == 1
    assert session.stats()['planned_factory_allocations'] == 3
  finally:
    session.close()


def test_alias_keeps_lease_until_storage_dies():
  pool = TensorPool(bytearray(128), 128)
  first = pool.empty((16,), torch.float32)
  first.fill_(7)
  alias = first[1:]
  del first
  gc.collect()
  second = pool.empty((16,), torch.float32)
  second.fill_(3)
  assert torch.all(alias == 7)
  assert pool.empty((1,), torch.float32) is None
  del alias
  gc.collect()
  assert pool.live_bytes == 64
  third = pool.empty((16,), torch.float32)
  third.fill_(9)
  assert torch.all(second == 3)
  del second, third
  gc.collect()
  assert pool.live_bytes == 0
  assert pool.empty((32,), torch.float32) is not None


def test_dynamic_session_spills_and_output_survives_close():
  class System:
    def current_cpu_domain(self):
      return 0

    def allocate(self, size, **kwargs):
      return bytearray(size)

  capture = graph.GraphCapture("dynamic")
  machine = planner.MachineProfile(
    fast_capacity=128, slow_capacity=1024, fast_bandwidth_mib_s=100,
    slow_bandwidth_mib_s=10, transfer_bandwidth_mib_s=10, alignment=64,
  )
  execution = capture.finish()
  plan = planner.plan_memory(execution, machine)
  with planner.PlanSession(execution, plan, system=System(), enable_dynamic_factories=True) as session:
    first = planner.empty(16, dtype=torch.float32)
    first.fill_(7)
    alias = first.view(4, 4).t()
    del first
    second = planner.empty_like(alias)
    assert second.stride() == alias.stride()
    second.fill_(3)
    spilled = planner.empty(1, dtype=torch.float32)
    spilled.fill_(11)
    assert session.stats()["dynamic_peak_bytes"] == 128
    assert session.stats()["dynamic_misses"] == 1
  gc.collect()
  assert torch.all(alias == 7)
  assert torch.all(second == 3)
  assert spilled.item() == 11


def test_fragmented_ranges_coalesce_after_out_of_order_storage_release():
  pool = TensorPool(bytearray(6 * 64), 6 * 64)
  tensors = [pool.empty((16,), torch.float32) for _ in range(6)]
  aliases = [tensor.view(4, 4).t() for tensor in tensors]
  del tensors
  for index in (1, 3, 2, 5, 0):
    aliases[index] = None
    gc.collect()
  # Neither disjoint free range can serve a five-block request yet.
  assert pool.empty((80,), torch.float32) is None
  aliases[4] = None
  gc.collect()
  assert pool.live_bytes == 0
  assert pool._free == [(0, 6 * 64)]
  assert pool.empty((96,), torch.float32) is not None


def test_tensor_storage_owns_lease_after_pool_reference_is_dropped():
  import weakref

  pool = TensorPool(bytearray(128), 128)
  reference = weakref.ref(pool)
  tensor = pool.empty((16,), torch.float32)
  tensor.fill_(9)
  alias = tensor.view(4, 4)
  del pool, tensor
  gc.collect()
  assert reference() is not None
  assert torch.all(alias == 9)
  del alias
  gc.collect()
  assert reference() is None
