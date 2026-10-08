# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""License/header checks do not infer ownership or rewrite imported grants."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def checker():
  path = Path(__file__).parents[2] / "scripts/check_licenses.py"
  spec = importlib.util.spec_from_file_location("vecops_license_check", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_missing_fields_and_near_top_requirement(checker, tmp_path):
  path = tmp_path / "kernel.py"
  path.write_text("pass\n")
  assert len(checker.notice_errors(path, tmp_path)) == 2
  path.write_text("\n" * 26 + "# SPDX-FileCopyrightText: owner\n# SPDX-License-Identifier: MIT\n")
  assert len(checker.notice_errors(path, tmp_path)) == 2
  path.write_text("# SPDX-FileCopyrightText: owner\n# SPDX-License-Identifier: MIT\npass\n")
  assert checker.notice_errors(path, tmp_path) == []


def test_exact_sidecar_for_uncommentable_material(checker, tmp_path):
  path = tmp_path / "data.bin"
  path.write_bytes(b"\xff\x00")
  assert checker.notice_errors(path, tmp_path)
  Path(str(path) + ".license").write_text("SPDX-FileCopyrightText: owner\nSPDX-License-Identifier: MIT\n")
  assert checker.notice_errors(path, tmp_path) == []


def test_upstream_exception_does_not_cover_arbitrary_new_files(checker, tmp_path):
  directory = tmp_path / "LICENSES"
  directory.mkdir()
  upstream = directory / "Arm-optimized-routines.txt"
  upstream.write_text("original upstream grant\n")
  assert checker.notice_errors(upstream, tmp_path) == []
  unknown = directory / "unreviewed.txt"
  unknown.write_text("unknown\n")
  assert len(checker.notice_errors(unknown, tmp_path)) == 2


def test_referenced_full_license_must_exist(checker, tmp_path):
  (tmp_path / "THIRD_PARTY_NOTICES.md").write_text("[full grant](LICENSES/upstream.txt)\n")
  assert checker.third_party_errors(tmp_path) == ["missing referenced full license: LICENSES/upstream.txt"]


def test_sdk_and_wheel_keep_full_license_materials():
  root = Path(__file__).parents[2]
  cmake = (root / "CMakeLists.txt").read_text()
  packaging = (root / "pyproject.toml").read_text()
  assert "DESTINATION vecops/_sdk/share/licenses/vecops" in cmake
  assert 'DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/vecops"' in cmake
  assert 'wheel.license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md", "LICENSES/*.txt"]' in packaging
