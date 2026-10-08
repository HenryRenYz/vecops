# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""A benchmark must not report success when its requested cases do not exist."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
  directory = Path(__file__).parents[2] / "benchmarks"
  sys.path.insert(0, str(directory))
  try:
    spec = importlib.util.spec_from_file_location("vecops_test_raw_matmul_filters", directory / "dsl_matmul.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
  finally:
    sys.path.remove(str(directory))


@pytest.mark.parametrize("shapes", [("19x23x17",), ("16x16x128", "19x23x17")])
def test_unknown_shapes_fail_before_building_or_changing_affinity(benchmark, monkeypatch, capsys, shapes):
  monkeypatch.setattr(benchmark, "catalog", lambda **_: (("micro", "known", 16, 16, 128),))
  monkeypatch.setattr(benchmark, "isolate_compilation", lambda *_: pytest.fail("invalid selection must not pin CPUs"))
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **_: pytest.fail("invalid selection must not build"))
  monkeypatch.setattr(sys, "argv", ["dsl_matmul.py", "--shape", *shapes])
  with pytest.raises(SystemExit) as error:
    benchmark.main()
  assert error.value.code == 2
  assert "shapes absent" in capsys.readouterr().err


def test_flop_filter_cannot_silently_turn_a_valid_selection_into_zero_tests(benchmark, monkeypatch, capsys):
  monkeypatch.setattr(benchmark, "catalog", lambda **_: (("micro", "known", 16, 16, 128),))
  monkeypatch.setattr(benchmark, "isolate_compilation", lambda *_: pytest.fail("empty selection must not pin CPUs"))
  monkeypatch.setattr(sys, "argv", ["dsl_matmul.py", "--shape", "16x16x128", "--max-flops", "1"])
  with pytest.raises(SystemExit) as error:
    benchmark.main()
  assert error.value.code == 2
  assert "no matching benchmark cases" in capsys.readouterr().err


@pytest.mark.parametrize("option, value", [("--kind", "unknown"), ("--mode", "typo"), ("--level", "typo")])
def test_unknown_execution_modes_are_not_treated_as_dynamic_or_library(benchmark, monkeypatch, option, value):
  monkeypatch.setattr(sys, "argv", ["dsl_matmul.py", option, value])
  with pytest.raises(SystemExit) as error:
    benchmark.main()
  assert error.value.code == 2
