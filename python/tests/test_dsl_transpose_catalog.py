"""Complete transpose metadata, equivalent-schema reuse, and safe resuming."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import vecops
from dsl_test_utils import make_compiler


@pytest.fixture(scope="module")
def catalog_module():
  path = Path(__file__).parents[2] / "benchmarks/dsl_transpose.py"
  spec = importlib.util.spec_from_file_location("vecops_test_transpose_catalog", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_complete_catalog_preserves_every_metadata_row(catalog_module):
  cases = [(*case, mode) for case in catalog_module.catalog() for mode in catalog_module.MODES]
  assert len(cases) == len({catalog_module.case_key(case) for case in cases}) == 399
  keys = {catalog_module.schema_key(*case) for case in cases}
  assert len(keys) == 194
  constant = [case for case in cases if case[-1] == "Const"]
  assert len({catalog_module.schema_key(*case) for case in constant}) == len(constant)


def test_dynamic_pair_reuse_preserves_alignment_constraints(catalog_module):
  key = catalog_module.schema_key
  types = ("float16", "float32", "float32")
  assert key(16, 16, *types, "Any") == key(1024, 1024, *types, "Any")
  assert key(3, 5, *types, "Aligned") == key(17, 19, *types, "Aligned")
  assert key(3, 5, *types, "Aligned") == key(1024, 1024, *types, "Any")
  assert key(16, 32, *types, "Aligned") == key(1024, 1024, *types, "Aligned")
  assert key(16, 16, *types, "Aligned") != key(16, 32, *types, "Aligned")
  assert key(16, 32, *types, "Aligned") != key(16, 32, *types, "Any")
  assert key(16, 32, *types, "Const") != key(1024, 1024, *types, "Const")
  with pytest.raises(ValueError, match="metadata mode"):
    key(16, 32, *types, "unknown")


def test_shared_any_pair_collects_only_two_real_artifacts(catalog_module, tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  with vecops.collect_compile_requests() as collector:
    prepared = catalog_module.collect_kernel_pairs(
      compiler,
      [(3, 5, "float32", "float32", "float32", "Any"), (1024, 1024, "float32", "float32", "float32", "Any")],
    )
  assert len(collector) == 2
  assert len(prepared) == 2
  assert prepared[0][-1] is prepared[1][-1]


def test_runtime_extents_share_actual_native_binding_identity(catalog_module, tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  with vecops.collect_compile_requests() as collector:
    pair = catalog_module.kernels(compiler, 3, 5, "Any", "float32", "float32", "float32")
    for m, n in [(3, 5), (1024, 1024)]:
      for kernel in pair:
        kernel.operator(
          vecops.TensorMeta((m, n), dtype="float32"),
          vecops.TensorMeta((n, m), dtype="float32"),
          1,
          **kernel._core.specialization_values,
        )
  keys = [
    vecops._C.bind_kernel_call(pair[index % 2].operator.kernel_def._native(), request.call)["specialization_key"]
    for index, request in enumerate(collector.requests)
  ]
  assert keys[0] == keys[2]
  assert keys[1] == keys[3]


def test_resume_uses_complete_metadata_key_and_rejects_other_formats(catalog_module, tmp_path):
  path = tmp_path / "transpose.csv"
  assert catalog_module.load_completed(path) == set()
  case = (*catalog_module.catalog()[0], "Const")
  with path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(catalog_module.FIELDS)
    writer.writerow((*case, "1.0", "2.0", "0.5"))
  assert catalog_module.load_completed(path) == {catalog_module.case_key(case)}
  path.write_text("shape,dtype\n16x16,float32\n", encoding="utf-8")
  with pytest.raises(ValueError, match="header does not match"):
    catalog_module.load_completed(path)


def test_list_and_partitions_never_start_compilation(catalog_module, monkeypatch, capsys):
  monkeypatch.syspath_prepend(str(catalog_module.ROOT / "benchmarks"))

  def unexpected_compile(*args, **kwargs):
    pytest.fail("metadata-only listing must not start compiler construction")

  monkeypatch.setattr(catalog_module.vecops, "Compiler", unexpected_compile)
  selected = []
  for index in range(3):
    monkeypatch.setattr(
      sys,
      "argv",
      ["dsl_transpose.py", "--list", "--partition", f"{index}/3"],
    )
    catalog_module.main()
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert tuple(rows[0]) == catalog_module.FIELDS[:6]
    assert len(rows[1:]) == 133
    selected.extend(map(tuple, rows[1:]))
  assert len(selected) == len(set(selected)) == 399


@pytest.mark.parametrize(
  "arguments",
  [
    ["--shape", "16x16", "31x47"],
    ["--list", "--shape", "16x16", "31x47"],
    ["--shape", "256x256", "--dtype", "float32", "bfloat16"],
    ["--partition", "2/2"],
    ["--partition", "bad"],
    ["--limit", "-1"],
  ],
)
def test_catalog_rejects_unknown_and_partly_ignored_filter_choices(catalog_module, monkeypatch, arguments):
  monkeypatch.syspath_prepend(str(catalog_module.ROOT / "benchmarks"))
  monkeypatch.setattr(sys, "argv", ["dsl_transpose.py", *arguments])
  with pytest.raises(SystemExit) as error:
    catalog_module.main()
  assert error.value.code == 2


def test_catalog_empty_listing_and_explicit_zero_work_are_not_execution(catalog_module, monkeypatch, capsys):
  monkeypatch.syspath_prepend(str(catalog_module.ROOT / "benchmarks"))

  def unexpected_compile(**kwargs):
    pytest.fail("empty listing/partition must not construct a compiler")

  monkeypatch.setattr(catalog_module.vecops, "Compiler", unexpected_compile)
  monkeypatch.setattr(sys, "argv", ["dsl_transpose.py", "--list", "--shape", "256x256", "--dtype", "bfloat16"])
  catalog_module.main()
  assert len(capsys.readouterr().out.splitlines()) == 1
  monkeypatch.setattr(sys, "argv", ["dsl_transpose.py", "--limit", "0"])
  catalog_module.main()
  assert "0 executions" in capsys.readouterr().out
