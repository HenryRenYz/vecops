# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Native replay requires source, ABI, specialization and policy equality."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from dsl_matmul_source_equivalence import compare_documents


def document():
  return {
    "completed": True,
    "native_environment": {"sdk_sha256": "sdk", "target": "Native"},
    "group_ids": ["g"],
    "groups": [
      {
        "id": "g",
        "execution_ids": ["row"],
        "vector_state": {"sve_bytes": 64, "sme_bytes": 64},
        "programs": {
          "dsl": {
            "source_sha256": "program",
            "contract": ["float32", "dynamic"],
            "captures": {"K": 33},
            "effective_cflags": ["-ffp-contract=off"],
          }
        },
      }
    ],
  }


@pytest.mark.parametrize("field", ["source_sha256", "contract", "captures", "effective_cflags"])
def test_source_replay_rejects_any_changed_native_program_input(field):
  old = document()
  new = copy.deepcopy(old)
  assert compare_documents(old, new)["all_programs_identical"]
  new["groups"][0]["programs"]["dsl"][field] = "changed"
  result = compare_documents(old, new)
  assert not result["all_programs_identical"] and result["changed_groups"] == ["g"]


@pytest.mark.parametrize("field", ["completed", "native_environment", "group_ids", "groups"])
def test_source_replay_requires_complete_exact_scope_and_native_environment(field):
  old, new = document(), document()
  new[field] = [] if field in {"group_ids", "groups"} else False if field == "completed" else {"sdk_sha256": "changed"}
  with pytest.raises(ValueError):
    compare_documents(old, new)
