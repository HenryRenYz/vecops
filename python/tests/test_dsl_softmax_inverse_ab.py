# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Frozen normalization experiments keep schedules and provenance separate."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
  path = Path(__file__).parents[2] / "benchmarks/dsl_softmax_inverse_ab.py"
  spec = importlib.util.spec_from_file_location("vecops_softmax_inverse_ab_test", path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  try:
    spec.loader.exec_module(module)
    yield module
  finally:
    sys.modules.pop(spec.name, None)


def test_schedule_listing_uses_only_public_old_snapshot_algorithms(benchmark, capsys, monkeypatch):
  monkeypatch.setattr(benchmark, "snapshot_modules", lambda root: pytest.fail("listing must not import a snapshot"))
  benchmark.main(["--list", "--shape", "8x128", "31", "--dtype", "float32", "--meta", "Any"])
  rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
  assert len(rows) == 6
  assert {row["unroll"] for row in rows} == {"1", "2"}
  assert {row["accuracy"] for row in rows} == {"strict"}
  assert {row["scope"] for row in rows if row["shape"] == "31"} == {"row-rank1"}
  assert {row["scope"] for row in rows if row["shape"] == "8x128"} == {"row-rank1", "direct-rank2"}


@pytest.mark.parametrize(
  "arguments",
  (
    ["--list", "--shape", "0"],
    ["--list", "--shape", "1x2x3"],
    ["--list", "--shape", "31", "--scope", "direct-rank2"],
    ["--list", "--unroll", "1", "1"],
    ["--source-archive-sha256", "short"],
    ["--source-archive-sha256", "0" * 64, "--namespace", "../escape"],
  ),
)
def test_invalid_or_unlabeled_experiment_is_rejected_before_compilation(benchmark, monkeypatch, arguments):
  monkeypatch.setattr(benchmark, "snapshot_modules", lambda root: pytest.fail("invalid experiment must not compile"))
  with pytest.raises(SystemExit) as error:
    benchmark.main(arguments)
  assert error.value.code == 2


def test_csv_identifies_snapshot_runner_reference_and_schedule(benchmark):
  assert benchmark.FIELDS[:4] == (
    "source_archive_sha256",
    "runner_sha256",
    "cpp_source_sha256",
    "namespace",
  )
  assert "scope" in benchmark.FIELDS and "unroll" in benchmark.FIELDS
  source = Path(benchmark.__file__).read_text()
  assert "softmax_inverse" not in source.replace("dsl_softmax_inverse_ab", "")
  assert "vecops.compile_batch" in source
  assert "namespace build directory is not empty" in source
