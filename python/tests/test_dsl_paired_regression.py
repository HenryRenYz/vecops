# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Paired remeasurement rejects missing or changed native evidence."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import dsl_matmul_paired_regression as paired


def fixture(tmp_path):
  directory = tmp_path / "groups/g"
  directory.mkdir(parents=True)
  binary = directory / "kernel.so"
  binary.write_bytes(b"attested native bytes")
  manifest = directory / "manifest.json"
  manifest.write_text("{}")
  archive = directory / "evidence.tar.gz"
  archive.write_bytes(b"retained build evidence")
  (directory / "build.json").write_text(json.dumps({"status": "built"}))
  (directory / "index.json").write_text(
    json.dumps(
      {
        "manifest": str(manifest),
        "evidence_archive": str(archive),
        "evidence_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "artifacts": {"row": {"dsl": str(binary), "cpp": str(binary)}},
        "retained_binary_sha256": {str(binary): hashlib.sha256(binary.read_bytes()).hexdigest()},
      }
    )
  )
  return {"groups": [{"id": "g", "execution_ids": ["row"]}]}, binary, archive


@pytest.mark.parametrize("changed", ["binary", "archive", "build", "missing"])
def test_paired_admission_requires_exact_built_and_unchanged_inputs(tmp_path, changed):
  plan, binary, archive = fixture(tmp_path)
  indexes, inputs = paired.suite_indexes(tmp_path, plan, {"row"})
  assert indexes.keys() == {"row"}
  assert str(binary) in inputs and str(archive) in inputs
  wanted = {"row"}
  if changed == "binary":
    binary.write_bytes(b"changed")
  elif changed == "archive":
    archive.write_bytes(b"changed")
  elif changed == "build":
    (binary.parent / "build.json").write_text(json.dumps({"status": "failed"}))
  else:
    wanted.add("absent")
  with pytest.raises(ValueError):
    paired.suite_indexes(tmp_path, plan, wanted)
