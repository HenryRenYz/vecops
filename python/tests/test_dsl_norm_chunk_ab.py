# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Diagnostic extensions retain original normalization factories and contracts."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import vecops.language as T
from dsl_test_utils import parameter_snapshot_name
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend


@pytest.fixture(scope="module")
def benchmark():
  path = Path(__file__).parents[2] / "benchmarks/dsl_norm_chunk_ab.py"
  sys.path.insert(0, str(path.parent))
  spec = importlib.util.spec_from_file_location("vecops_norm_chunk_ab_test", path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  try:
    spec.loader.exec_module(module)
    yield module
  finally:
    sys.modules.pop(spec.name, None)
    sys.path.remove(str(path.parent))


def test_exact_six_keys_are_explicit_diagnostic_metadata_not_original_catalog(benchmark):
  cases = benchmark.cases()
  assert len(cases) == len({case.key for case in cases}) == 6
  assert {case.shape for case in cases} == {(8, 128)}
  assert {case.dtype for case in cases} == {"bfloat16"}
  assert {case.accuracy for case in cases} == {"strict"}
  assert {case.meta for case in cases} == {"Const", "Any"}
  groups = {(case.operation, case.affine, case.variance) for case in cases}
  assert groups == {
    ("layernorm", "none", "moments"),
    ("layernorm", "gamma_beta", "moments"),
    ("softmax", "none", "n/a"),
  }
  assert "diagnostic metadata" in benchmark.__doc__
  assert "does\nnot contain 8x128" in benchmark.__doc__


def test_listing_exact_eighteen_rows_does_not_construct_a_compiler(benchmark, monkeypatch, capsys):
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **kwargs: pytest.fail("listing must not compile"))
  benchmark.main(["--list", "--source-archive-sha256", "0" * 64])
  rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
  assert len(rows) == 18
  assert {row["math_chunk_pow"] for row in rows} == {"0", "1", "2"}
  assert len({tuple(row[name] for name in benchmark.KEY_FIELDS) for row in rows}) == 6


@pytest.mark.parametrize(
  "arguments",
  (
    ["--source-archive-sha256", "unverified", "--list"],
    ["--source-archive-sha256", "0" * 64, "--chunk-pow", "1", "1", "--list"],
    ["--source-archive-sha256", "0" * 64, "--rounds", "0", "--list"],
    ["--source-archive-sha256", "0" * 64, "--jobs", "0", "--list"],
    ["--source-archive-sha256", "0" * 64, "--chunk-pow", "3", "--list"],
  ),
)
def test_invalid_diagnostic_identity_or_schedule_rejected_before_compilation(benchmark, monkeypatch, arguments):
  monkeypatch.setattr(benchmark.vecops, "Compiler", lambda **kwargs: pytest.fail("invalid CLI must not compile"))
  with pytest.raises(SystemExit) as error:
    benchmark.main(arguments)
  assert error.value.code == 2


@pytest.mark.parametrize("index", range(6))
def test_math_chunk_uses_original_public_factories_unroll_and_strict_contract(benchmark, monkeypatch, index):
  case = benchmark.cases()[index]
  monkeypatch.setattr(T, "compile", lambda template, **kwargs: template)
  for chunk in (0, 1, 2):
    row, cpp = benchmark.kernels(None, case, unroll=2, pass_options=benchmark.pass_options(chunk))
    direct = benchmark.library_kernel(None, case, unroll=2, pass_options=benchmark.pass_options(chunk))
    assert cpp.ir.source.name == "dsl_norm_catalog_reference.cpp"
    for template in (row, direct):
      artifact = CPUBackend().lower(
        template.ir,
        (),
        target="SVE",
        options={
          "_vector_state": VectorState(64, 64),
          **benchmark.pass_options(chunk),
        },
      )
      assert not artifact.backend_ir.register_layouts
      regions = [region for region in artifact.lowered.regions if region.initializations]
      assert regions and all(region.unroll == 2 for region in regions)
      assert "vecops::vec::cvt::unordered" not in artifact.source
      if case.meta == "Const":
        assert tuple(int(extent) for extent in artifact.kernel.parameters["A"].shape) == (8, 128)
        assert "std::int64_t{128}" in artifact.source
      else:
        assert all(not T.is_value(extent) for extent in artifact.kernel.parameters["A"].shape)
        parameter_snapshot_name(artifact.source, "A.size(1)")
      assert "for (vecops::nint_t dsl_f_iteration_1" in artifact.source
      if case.operation == "softmax":
        assert "vecops::vec::opt::math::strict" in artifact.source
        assert "vecops::vec::rcp(" not in artifact.source
      else:
        assert "vecops::vec::rsqrt(" in artifact.source
        assert "vecops::vec::opt::math::strict" in artifact.source
