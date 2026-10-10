# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cache experiments preserve exact catalog coverage and arithmetic oracles."""

import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
  path = Path(__file__).parents[2] / "benchmarks/dsl_matmul_cache_ab.py"
  sys.path.insert(0, str(path.parent))
  spec = importlib.util.spec_from_file_location("vecops_matmul_cache_test", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  try:
    yield module
  finally:
    sys.path.remove(str(path.parent))


def test_position_and_directed_neighbor_balance(benchmark):
  rows = tuple(benchmark.balanced_order(round_) for round_ in range(6))
  assert all(set(row) == set(range(6)) for row in rows)
  assert all({row[position] for row in rows} == set(range(6)) for position in range(6))
  neighbors = Counter((a, b) for row in rows for a, b in zip(row, row[1:]))
  assert len(neighbors) == 30 and set(neighbors.values()) == {1}


def test_listing_retains_actual_slow_key_and_never_compiles(benchmark, monkeypatch, capsys):
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **_: pytest.fail("listing must not compile"))
  benchmark.main(["--svl-bytes", "64", "--source-archive-sha256", "0" * 64, "--list"])
  rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
  assert len(rows) == 12
  assert {row["profile"] for row in rows} == set(benchmark.LABELS)
  assert sorted(Counter(row["execution_id"] for row in rows).values()) == [6, 6]
  assert all("packedab" in row["key"] and "/128/4096/4096/" in row["key"] for row in rows)


def test_opt_in_whole_problem_diagnostic_changes_only_one_profile(benchmark, capsys):
  selected = benchmark.diagnostic_profiles(packed_ab_bypass=True)
  assert selected[0] == benchmark.PROFILES[0] and selected[2:] == benchmark.PROFILES[2:]
  assert selected[1] == ("whole-problem-library", (None, None, None), "matmul")
  assert benchmark.profile_options("whole-problem-library") == {}
  assert all(benchmark.profile_options(label) == {} for label, _, _ in benchmark.PROFILES)
  benchmark.main(["--svl-bytes", "64", "--source-archive-sha256", "0" * 64, "--sme-packed-ab-bypass", "--list"])
  rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
  assert len(rows) == 12
  assert sum(row["profile"] == "whole-problem-library" for row in rows) == 2
  assert sorted(Counter(row["execution_id"] for row in rows).values()) == [6, 6]


@pytest.mark.parametrize("extra", (("--rounds", "7"), ("--jobs", "0")))
def test_invalid_budget_fails_before_compilation(benchmark, monkeypatch, extra):
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **_: pytest.fail("invalid input must not compile"))
  with pytest.raises(SystemExit) as error:
    benchmark.main(["--svl-bytes", "64", "--source-archive-sha256", "0" * 64, "--list", *extra])
  assert error.value.code == 2


def test_block_overrides_are_explicit_and_reference_is_unchanged(benchmark, monkeypatch):
  import dsl_matmul_source_catalog as source
  from vecops.language._compiler.frontend.parser import lower
  from vecops.language._compiler.ir.traversal import nodes

  monkeypatch.setattr(source, "packing_geometry", lambda dtype: ("sme", 32, 1))
  case = next(
    case
    for case in benchmark.scenarios(benchmark.load_cases(svl_bytes=64), levels=("matmul",))
    if case.packing == "packedab"
    and case.mode == "Const"
    and (case.case.m, case.case.n, case.case.k) == (128, 4096, 4096)
  )
  default, default_packing = benchmark.source_templates(case)
  blocked, blocked_packing = benchmark.source_templates(case, blocks=(128, 128, 512))
  products = [value for value in nodes(lower(blocked[0].ir).body) if getattr(value, "op", None) == "dot"]
  assert products
  assert all((value.data.block_m, value.data.block_n, value.data.block_k) == (128, 128, 512) for value in products)
  assert blocked[1].ir.canonical() == default[1].ir.canonical()
  assert blocked_packing[1].ir.canonical() == default_packing[1].ir.canonical()
  for invalid in ((0, 1, 1), (True, 1, 1), [1, 1, 1], (1, 1)):
    with pytest.raises(ValueError, match="blocks"):
      benchmark.source_templates(case, blocks=invalid)
