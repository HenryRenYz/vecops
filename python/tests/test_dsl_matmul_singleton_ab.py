# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Singleton diagnostics retain original keys and a balanced, unfiltered order."""

import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
  path = Path(__file__).parents[2] / "benchmarks/dsl_matmul_singleton_ab.py"
  sys.path.insert(0, str(path.parent))
  spec = importlib.util.spec_from_file_location("vecops_singleton_benchmark_test", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  try:
    yield module
  finally:
    sys.path.remove(str(path.parent))


def test_balanced_permutations(benchmark):
  rows = [benchmark.balanced_order(index) for index in range(6)]
  assert len(set(rows)) == 6 and all(set(row) == {0, 1, 2} for row in rows)
  positions = Counter((value, position) for row in rows for position, value in enumerate(row))
  neighbors = Counter(pair for row in rows for pair in zip(row, row[1:]))
  assert len(positions) == 9 and set(positions.values()) == {2}
  assert len(neighbors) == 6 and set(neighbors.values()) == {2}
  assert benchmark.balanced_order(6) == rows[0]


def test_original_slow_keys_are_preserved_without_compilation(benchmark, monkeypatch, capsys):
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **_: pytest.fail("listing must not compile"))
  benchmark.main(["--svl-bytes", "64", "--source-archive-sha256", "0" * 64, "--list"])
  rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
  assert len(rows) == 6
  assert {row["execution_id"] for row in rows} == set(benchmark.DEFAULT_KEYS)
  assert {row["profile"] for row in rows} == set(benchmark.LABELS)
  assert all("packedb" in row["key"] and "/Const/" in row["key"] for row in rows)


@pytest.mark.parametrize("keys", ((), ("missing",), ("e6209a02fb9b668c8ff1e15a",) * 2))
def test_unavailable_or_duplicate_keys_are_not_silently_filtered(benchmark, keys):
  with pytest.raises(ValueError):
    benchmark.select_cases(keys, svl_bytes=64)


@pytest.mark.parametrize("extra", (("--rounds", "7"), ("--jobs", "0")))
def test_invalid_budget_precedes_compilation(benchmark, monkeypatch, extra):
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **_: pytest.fail("invalid budget must not compile"))
  with pytest.raises(SystemExit) as error:
    benchmark.main(["--svl-bytes", "64", "--source-archive-sha256", "0" * 64, "--list", *extra])
  assert error.value.code == 2
