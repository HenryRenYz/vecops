# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Schedule diagnostics cannot silently change coverage or FP semantics."""

import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
  path = Path(__file__).parents[2] / "benchmarks/dsl_norm_schedule_ab.py"
  sys.path.insert(0, str(path.parent))
  spec = importlib.util.spec_from_file_location("vecops_norm_schedule_test", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  try:
    yield module
  finally:
    sys.path.remove(str(path.parent))


def test_balanced_positions_and_directed_neighbors(benchmark):
  rows = tuple(benchmark.balanced_order(round_) for round_ in range(6))
  assert all(set(row) == set(range(6)) for row in rows)
  assert all({row[position] for row in rows} == set(range(6)) for position in range(6))
  neighbors = Counter((a, b) for row in rows for a, b in zip(row, row[1:]))
  assert len(neighbors) == 30 and set(neighbors.values()) == {1}


def test_listing_has_exact_maintained_metadata_and_no_compile(benchmark, monkeypatch, capsys):
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **_: pytest.fail("list must not compile"))
  benchmark.main(["--source-archive-sha256", "0" * 64, "--shape", "4096x4096", "--list"])
  rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
  assert len(rows) == 2 * 4 * 6
  assert {row["variance"] for row in rows} == {"moments"}
  assert {row["accuracy"] for row in rows} == {"strict"}
  assert {row["profile"] for row in rows} == set(benchmark.LABELS)


@pytest.mark.parametrize("extra", (("--rounds", "7"), ("--jobs", "0"), ("--shape", "not-a-shape")))
def test_invalid_schedule_and_metadata_fail_before_compile(benchmark, monkeypatch, extra):
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **_: pytest.fail("invalid input must not compile"))
  with pytest.raises(SystemExit) as error:
    benchmark.main(["--source-archive-sha256", "0" * 64, "--list", *extra])
  assert error.value.code == 2


def test_profiles_keep_strict_arithmetic(benchmark):
  for _, unroll, policy in benchmark.PROFILES:
    assert unroll in {2, 4}
    options = benchmark.pass_options(policy)
    assert "fp_contract" not in options
    assert options["cpu_math_chunk_pow"] == 1
  assert benchmark.pass_options("force")["cpu_pointwise_register_layout"] is True
  with pytest.raises(ValueError):
    benchmark.pass_options("unsafe")
