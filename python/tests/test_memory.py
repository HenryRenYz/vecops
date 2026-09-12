from __future__ import annotations

import gc

import pytest
from vecops import memory


def test_system_buffer_is_writable_and_accounted() -> None:
  system = memory.System(backend="system")
  owner = system.allocate(4097, alignment=4096)
  view = memoryview(owner)

  assert len(view) == 4097
  view[0] = 17
  view[-1] = 29
  assert view[0] == 17
  assert view[-1] == 29
  assert system.stats()[0]["managed_bytes"] == 4097

  del view
  del owner
  gc.collect()
  assert system.stats()[0]["managed_bytes"] == 0


def test_topology_and_tiers_are_machine_readable() -> None:
  system = memory.System(backend="system")
  topology = system.topology()

  assert topology["backend"] == "system"
  assert len(topology["cpu_domains"]) == 1
  assert len(topology["memory_targets"]) == 1
  assert system.tiers(objective="bandwidth") == [{"rank": 0, "targets": [0], "value": None}]


def test_invalid_public_choice_is_clear() -> None:
  system = memory.System(backend="system")
  with pytest.raises(ValueError, match="placement must be one of"):
    system.allocate(64, placement="quickish")
  with pytest.raises(ValueError, match="tier requires"):
    system.allocate(64, tier=0)


def test_numpy_array_retains_allocation() -> None:
  np = pytest.importorskip("numpy")
  system = memory.System(backend="system")
  array = memory.numpy.zeros((7, 9), dtype=np.float32, system=system)

  assert array.shape == (7, 9)
  assert array.flags.c_contiguous
  assert np.count_nonzero(array) == 0
  assert system.stats()[0]["managed_bytes"] == array.nbytes

  del array
  assert system.stats()[0]["managed_bytes"] == 0


def test_workspace_scope_is_reentrant_across_sequential_uses() -> None:
  system = memory.System(backend="system")
  scope = system.workspace_scope(fast_capacity=4096)

  with scope:
    pass
  with scope:
    pass
