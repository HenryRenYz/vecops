# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import gc

import pytest
from vecops import memory


def test_system_buffer_is_writable_and_accounted() -> None:
  system = memory.System()
  owner = system.allocate(4097, alignment=4096, large_pages=False)
  view = memoryview(owner)

  assert len(view) == 4097
  view[0] = 17
  view[-1] = 29
  assert view[0] == 17
  assert view[-1] == 29
  assert owner.large_page_bytes == 0
  assert owner.regular_page_bytes == owner.size
  assert sum(item["managed_bytes"] for item in system.stats()) == 4097
  assert sum(item["managed_large_page_bytes"] for item in system.stats()) == 0

  del view
  del owner
  gc.collect()
  assert sum(item["managed_bytes"] for item in system.stats()) == 0


def test_topology_and_tiers_are_machine_readable() -> None:
  try:
    system = memory.System(backend="system")
  except RuntimeError as error:
    if "cannot represent a multi-NUMA machine" in str(error):
      pytest.skip("portable backend intentionally supports only one NUMA node")
    raise
  topology = system.topology()

  assert topology["backend"] == "system"
  assert len(topology["cpu_domains"]) == 1
  assert len(topology["memory_targets"]) == 1
  assert system.tiers(objective="bandwidth") == [{"rank": 0, "targets": [0], "value": None}]


def test_invalid_public_choice_is_clear() -> None:
  system = memory.System()
  with pytest.raises(ValueError, match="placement must be one of"):
    system.allocate(64, placement="quickish")
  with pytest.raises(ValueError, match="tier requires"):
    system.allocate(64, tier=0)


def test_numpy_array_retains_allocation() -> None:
  np = pytest.importorskip("numpy")
  system = memory.System()
  array = memory.numpy.zeros((7, 9), dtype=np.float32, system=system)

  assert array.shape == (7, 9)
  assert array.flags.c_contiguous
  assert np.count_nonzero(array) == 0
  assert sum(item["managed_bytes"] for item in system.stats()) == array.nbytes

  del array
  assert sum(item["managed_bytes"] for item in system.stats()) == 0


def test_workspace_session_is_single_use_and_releases_arenas() -> None:
  system = memory.System()
  session = system.workspace_session(fast_capacity=4096, slow_capacity=4096)

  assert sum(item["managed_bytes"] for item in system.stats()) == 8192
  with session:
    pass
  assert session.closed
  assert sum(item["managed_bytes"] for item in system.stats()) == 0
  with pytest.raises(RuntimeError, match="cannot be re-entered"), session:
    pass
