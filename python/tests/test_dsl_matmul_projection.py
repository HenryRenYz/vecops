# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""The projection experiment preserves every real reference/meta observation."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from dsl_test_utils import assert_no_vec_mask_arrays
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import dsl_matmul_projection as runner
import dsl_matmul_source_catalog as catalog


@pytest.mark.parametrize("svl", (16, 32, 64))
def test_projection_complete_thirty_metas(svl):
  rows = runner.projection_scenarios(svl)
  assert len(rows) == len({s.execution_id for s in rows}) == 30
  assert Counter(s.mode for s in rows) == {"Const": 15, "Dynamic": 15}
  assert Counter(s.reference[1] for s in rows) == {"ExactCover": 10, "FourRegions": 10, "Automatic": 10}
  assert Counter(s.packing for s in rows) == {"raw": 12, "packedb": 6, "packedab": 12}
  assert len({f"{s.execution_id}:{v}" for s in rows for v in runner.VARIANTS}) == 60


@pytest.mark.parametrize(
  "mode,packing", (("Const", "raw"), ("Dynamic", "raw"), ("Dynamic", "packedb"), ("Const", "packedab"))
)
def test_projection_ab_only_changes_dsl_packing_request(monkeypatch, mode, packing):
  import dsl_benchmark_utils

  geometry = lambda _dtype: ("sme", 32, 2)
  monkeypatch.setattr(catalog, "packing_geometry", geometry)
  monkeypatch.setattr(dsl_benchmark_utils, "packing_geometry", geometry)
  scenario = next(s for s in runner.projection_scenarios(64) if s.mode == mode and s.packing == packing)
  (dsl, cpp), _ = catalog.source_templates(scenario)
  before = cpp.ir.canonical()
  variant = runner.online_ab_template(scenario, dsl)
  assert variant.ir.annotations == dsl.ir.annotations
  assert cpp.ir.canonical() == before
  assert cpp.ir.source == ROOT / "benchmarks/dsl_matmul_source_reference.cpp"
  backend = CPUBackend().lower(variant.ir, (), target="SME", options={"_vector_state": VectorState(32, 64)})
  assert backend.backend_ir.matmuls
  assert all(plan.options.packing == "AB" for plan in backend.backend_ir.matmuls.values())
  assert_no_vec_mask_arrays(backend.source)


def test_projection_manifest_no_silent_empty_missing_duplicate_or_stale(tmp_path):
  identity = {
    "source_label": "source18",
    "source_archive_sha256": "1" * 64,
    "source_hashes": {"cpp": "A"},
    "runner_sha256": "B",
    "runner_hashes": {"py": "B"},
    "pass_options": {},
    "implementation_identity": "C",
  }
  expected = ("first:auto", "first:online_ab")
  path = tmp_path / "part.json"
  doc = {
    **identity,
    "selected_observation_ids": expected,
    "results": {key: {"observation_id": key, "status": "passed"} for key in expected},
  }
  path.write_text(json.dumps(doc))
  assert runner.verify_documents((path,), identity, expected)["complete"]
  assert not runner.verify_documents((path, path), identity, expected)["complete"]
  assert not runner.verify_documents((path,), identity, (*expected, "missing:auto"))["complete"]
  for key, changed in (
    ("source_label", "source17"),
    ("source_archive_sha256", "2" * 64),
    ("source_hashes", {"cpp": "changed"}),
    ("runner_sha256", "changed"),
    ("runner_hashes", {"py": "changed"}),
    ("pass_options", {"cpu_matmul_pipeline": "off"}),
    ("implementation_identity", "changed"),
  ):
    assert not runner.verify_documents((path,), {**identity, key: changed}, expected)["complete"], key
  doc["results"][expected[0]]["status"] = "validation_failed"
  path.write_text(json.dumps(doc))
  assert runner.verify_documents((path,), identity, expected)["failed"] == [expected[0]]
  extra = "unexpected:auto"
  doc["selected_observation_ids"] = (*expected, extra)
  doc["results"][extra] = {"observation_id": extra, "status": "passed"}
  path.write_text(json.dumps(doc))
  assert runner.verify_documents((path,), identity, expected)["unexpected"] == [extra]
  with pytest.raises(ValueError, match="zero"):
    runner.verify_documents((path,), identity, ())


def test_projection_cache_model_no_invented_physical_l3():
  cache = {
    "line_bytes": 64,
    "l1d_bytes": 64 * 1024,
    "l2_bytes": 1024 * 1024,
    "l3_bytes": 8 * 1024 * 1024,
    "llc_bytes": 0,
    "llc_level": 0,
    "l3_present": False,
    "fallback": False,
  }
  absent = runner.modeled_blocks(cache, 768, 64)
  present = runner.modeled_blocks({**cache, "llc_bytes": 256 * 1024, "llc_level": 2}, 768, 64)
  assert absent["status"] == "model_only_not_executed_branch"
  assert absent["automatic_block_k"] == present["automatic_block_k"] == 768
  assert (
    present["automatic_block_m"] * present["automatic_block_n"]
    <= absent["automatic_block_m"] * absent["automatic_block_n"]
  )
