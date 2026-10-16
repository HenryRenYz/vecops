# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compared native paths share complete warmup rounds before timing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import dsl_timing


def test_warmup_finishes_a_round_and_records_actual_elapsed_time(monkeypatch):
  clock = iter((1_000, 60_001_000, 120_001_000))
  monkeypatch.setattr(dsl_timing.time, "perf_counter_ns", lambda: next(clock))
  visited = []
  record = dsl_timing.warmup_native(
    (lambda: visited.append("baseline"), lambda: visited.append("candidate")),
  )
  assert visited == ["baseline", "candidate", "baseline", "candidate"]
  assert record == {"protocol": "native-round-robin-v1", "minimum_seconds": 0.1,
                    "elapsed_ns": 120_000_000, "rounds": 2, "paths": 2}


def test_zero_duration_still_invokes_the_native_path(monkeypatch):
  monkeypatch.setattr(dsl_timing.time, "perf_counter_ns", lambda: 100)
  visited = []
  record = dsl_timing.warmup_native(lambda: visited.append(True), seconds=0)
  assert visited == [True]
  assert record["rounds"] == record["paths"] == 1


@pytest.mark.parametrize("calls,seconds", [
  ((), 0.1), ((None,), 0.1), ((lambda: None,), -1),
  ((lambda: None,), float("nan")), ((lambda: None,), float("inf")),
])
def test_invalid_warmup_policy_is_rejected(calls, seconds):
  with pytest.raises(ValueError):
    dsl_timing.warmup_native(calls, seconds=seconds)
