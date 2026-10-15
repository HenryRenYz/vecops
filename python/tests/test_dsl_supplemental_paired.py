# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Current-frequency supplemental timing balances all four implementations."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import dsl_matmul_supplemental_paired as paired


def test_fresh_frontend_uses_original_guards_and_cross_snapshot_requires_proof(tmp_path):
  root, suite = tmp_path / "frontend", tmp_path / "native"
  assert paired.source_admission(None, root, root, suite) == (None, {})
  with pytest.raises(ValueError, match="different frontend snapshot"):
    paired.source_admission(None, root, tmp_path / "another-frontend", suite)
  paired.verify_reference_root(root / "benchmarks/reference.cpp", root)
  with pytest.raises(ValueError, match="original frontend snapshot"):
    paired.verify_reference_root(tmp_path / "old-frontend/benchmarks/reference.cpp", root)


def test_checkpoint_environment_handles_native_enum_repr_and_rejects_mixed_policies():
  def policy(path):
    external = (("CC", None), ("CXX", "/tools/clang++"), ("PATH", path), ("CXXFLAGS", ""))
    # The enclosing key contains an enum repr and is deliberately not a Python
    # literal. Only its recorded external environment is parsed as data.
    return {"key": "((<ArtifactCacheMode.read_write: 1>,), '/project', (), " + repr(external) + ")"}

  expected = {"CC": None, "CXX": "/tools/clang++", "PATH": "/original/bin", "CXXFLAGS": ""}
  assert paired.checkpoint_environment({"policies": [policy("/original/bin")]}) == expected
  with pytest.raises(ValueError, match="different external environments"):
    paired.checkpoint_environment({"policies": [policy("/old/bin"), policy("/new/bin")]})


def test_four_way_timing_balances_positions_and_retains_raw_measurements(monkeypatch):
  ticks = iter(range(0, 10_000_000_000, 25_000_000))
  monkeypatch.setattr(paired.time, "perf_counter_ns", lambda: next(ticks))
  calls = {label: lambda repeat: None for label in ("old_dsl", "old_cpp", "new_dsl", "new_cpp")}
  result = paired.measure(calls, 12)
  assert len(result["samples"]) == 48
  positions = {label: Counter() for label in calls}
  for index, sample in enumerate(result["samples"]):
    positions[sample["label"]][index % 4] += 1
    assert sample["elapsed_ns"] > 0 and sample["repeat"] > 0
  assert all(counts == Counter({0: 3, 1: 3, 2: 3, 3: 3}) for counts in positions.values())
  assert result["median_ns"] == dict.fromkeys(calls, 25_000_000)
