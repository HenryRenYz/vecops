# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Resident packed-AB diagnostics retain the complete original geometry cohort."""

import csv
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def benchmark():
  path = Path(__file__).parents[2] / "benchmarks/dsl_matmul_packed_ab_ab.py"
  sys.path.insert(0, str(path.parent))
  spec = importlib.util.spec_from_file_location("vecops_resident_packed_ab_benchmark_test", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  try:
    yield module
  finally:
    sys.path.remove(str(path.parent))


def test_original_geometry_cohort_preserves_policies_fusion_and_cache_rejects(benchmark):
  cases = benchmark.geometry_cases(svl_bytes=64)
  assert len(cases) == 44 and len({case.execution_id for case in cases}) == 44
  assert Counter(case.mode for case in cases) == {"Const": 22, "Dynamic": 22}
  assert Counter(case.case.catalog for case in cases) == {
    "MatmulBaseExpanded": 28,
    "MatmulPolicyExpanded": 12,
    "MatmulPackedExpanded": 4,
  }
  assert any(case.case.m == 1 for case in cases)
  assert any(case.case.k == 11008 for case in cases)  # Must retain runtime cache-budget rejects.
  assert {case.case.output_pipeline for case in cases} >= {"Convert", "BiasRelu", "Relu"}
  assert {case.execution_id for case in cases} >= {"f08417cde68500540d1a32bd", "84e258c6e4fc904a541c1d04"}
  assert all(case.packing == "packedab" and not case.online and not case.batched for case in cases)


@pytest.mark.parametrize("svl", (16, 32, 64, 128, 256))
def test_shards_are_exact_disjoint_cover_for_each_streaming_width(benchmark, svl):
  cases = benchmark.geometry_cases(svl_bytes=svl)
  shards = [benchmark.shard_cases(cases, index=index, count=4) for index in range(4)]
  ids = [case.execution_id for shard in shards for case in shard]
  assert len(ids) == len(set(ids)) == len(cases)
  assert set(ids) == {case.execution_id for case in cases}
  assert all(shard == cases[index::4] for index, shard in enumerate(shards))
  assert all(case.case.svl_bytes == svl for case in cases)


@pytest.mark.parametrize("svl", (None, True, 0, 15, 24, 512))
def test_invalid_or_unknown_svl_never_uses_a_fixed_width_fallback(benchmark, svl):
  with pytest.raises(ValueError, match="SVL"):
    benchmark.geometry_cases(svl_bytes=svl)


@pytest.mark.parametrize("index,count", ((0, 0), (-1, 4), (4, 4), (True, 4), (0, True), (1, 2)))
def test_invalid_and_empty_shards_are_not_silently_dropped(benchmark, index, count):
  with pytest.raises(ValueError):
    benchmark.shard_cases((None,), index=index, count=count)


def test_listing_is_complete_and_does_not_construct_a_compiler(benchmark, monkeypatch, capsys):
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **_: pytest.fail("listing must not compile"))
  benchmark.main(["--svl-bytes", "64", "--source-archive-sha256", "0" * 64, "--list"])
  rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
  assert len(rows) == 132
  assert set(Counter(row["execution_id"] for row in rows).values()) == {3}
  assert {row["profile"] for row in rows} == set(benchmark.LABELS)


def test_all_original_cohort_sources_expose_only_the_candidate_policy_guard(benchmark, monkeypatch):
  import dsl_benchmark_utils
  from vecops._arm_vector_state import VectorState
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  backend = CPUBackend()
  options = {"_vector_state": VectorState(64, 64)}
  # These are pure target-source checks on an x86 host. The benchmark factory
  # normally queries the executing machine's packing ABI, independently of
  # CPUBackend.lower(target=...). Explicitly provide the same SME geometry.
  monkeypatch.setattr(dsl_benchmark_utils, "is_arm", lambda: True)
  monkeypatch.setattr(dsl_benchmark_utils, "get_vector_state", lambda: VectorState(64, 64))
  changed = []
  for case in benchmark.geometry_cases(svl_bytes=64):
    template = benchmark.source_templates(case)[0][0]
    before = backend.lower(template.ir, (), target="SME", options=options | benchmark.CONTROL_OPTIONS).source
    after = backend.lower(template.ir, (), target="SME", options=options | benchmark.OPTIONS).source
    default = backend.lower(template.ir, (), target="SME", options=options).source
    assert default == after, case.execution_id
    assert benchmark.MARKER not in before, case.execution_id
    assert (before != after) == (benchmark.MARKER in after), case.execution_id
    if before != after:
      assert "complete padded two-by-two micro-panel stays in private L2" in after
      changed.append(case.execution_id)
  assert len(changed) == 44


def test_shared_timing_preserves_all_clocks_repeats_and_balanced_positions(benchmark, monkeypatch):
  import dsl_matmul_ab_utils as utilities

  calls, repeat_values = [], []

  def native_calls(compiled, values, repeat):
    repeat_values.append(repeat)
    return tuple(lambda label=label: calls.append(label) for label in benchmark.LABELS)

  clock = iter(range(1, 200, 2))
  monkeypatch.setattr(utilities, "native_calls", native_calls)
  monkeypatch.setattr(utilities.time, "perf_counter_ns", lambda: next(clock))
  monkeypatch.setattr(utilities, "result_metadata", lambda _: {"execution_id": "original"})
  scenario = SimpleNamespace(case=SimpleNamespace(m=1, n=128, k=512), execution_id="original")
  stream = io.StringIO()
  report = utilities.time_case(scenario, (None,) * 3, (), 6, stream, labels=benchmark.LABELS)
  records = [json.loads(line) for line in stream.getvalue().splitlines()]
  repeat = 8_000_000 // (128 * 512)
  assert repeat_values == [repeat] and len(records) == 18 and len(calls) == 21
  assert all(record["end_ns"] - record["begin_ns"] == 2 for record in records)
  assert all(record["ns"] == 2 / repeat and record["repeat"] == repeat for record in records)
  positions = Counter((record["label"], record["position"]) for record in records)
  assert len(positions) == 9 and set(positions.values()) == {2}
  assert set(report["median_ns"].values()) == {2 / repeat}
