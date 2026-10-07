"""The tiny transpose diagnostic changes only compiler loop hints."""

import sys
from pathlib import Path

import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import dsl_transpose_codegen_ab
from dsl_transpose import catalog
from dsl_transpose_codegen_ab import HINTS, row_hint_source, selected_cases


def test_diagnostic_keeps_all_original_catalog_tiny_conversion_pairs():
  expected = {(*case, "Const") for case in catalog() if case[:2] == (16, 16) and len(set(case[2:])) != 1}
  assert len(selected_cases()) == len(expected) and set(selected_cases()) == expected
  assert len(expected) > 7


def test_diagnostic_translates_catalog_order_to_kernel_factory_order(monkeypatch):
  calls = []
  pair = object(), object()

  def factory(*args):
    calls.append(args)
    return pair

  monkeypatch.setattr(dsl_transpose_codegen_ab, "kernels", factory)
  compiler = object()
  for case in selected_cases():
    assert dsl_transpose_codegen_ab.original_pair(compiler, case) is pair
    m, n, input_dtype, output_dtype, compute_dtype, mode = case
    assert calls[-1] == (compiler, m, n, mode, input_dtype, output_dtype, compute_dtype)


def test_native_derived_transpose_sources_keep_signature_and_alias_contract(tmp_path):
  case = (16, 16, "float16", "float32", "float32", "Const")
  compiler = make_compiler(build_dir=tmp_path / "build", cache_dir=tmp_path / "cache", jobs=4)
  candidates = []
  with vecops.collect_compile_requests() as collector:
    original, cpp = dsl_transpose_codegen_ab.original_pair(compiler, case)
    for hint in ("unchanged", "two"):
      source, _ = row_hint_source(original.get_kernel_source(), hint)
      path = tmp_path / f"{hint}.cpp"
      path.write_text(source)
      compiled = T.compile(dsl_transpose_codegen_ab.source_template(path, case), compiler=compiler)
      candidates.append(compiled)
    for compiled in (*candidates, cpp):
      compiled.operator(
        vecops.TensorMeta((16, 16), dtype="float16"),
        vecops.TensorMeta((16, 16), dtype="float32"),
        1,
        **compiled._core.specialization_values,
      )
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  dsl_transpose_codegen_ab.validate(case, (*candidates, cpp))


@pytest.mark.parametrize("hint", HINTS)
def test_row_hint_transformation_preserves_all_other_source_and_only_changes_entry_rows(hint):
  source = "for (vecops::nint_t dsl_c_axis_99 = 0; i < 2; ++i) {}\nvoid __kernel__(auto A) {\n  for (vecops::nint_t dsl_c_axis_1 = 0; i < 16; ++i) {}\n}\n"
  transformed, count = row_hint_source(source, hint)
  assert count == (0 if hint == "unchanged" else 1)
  original_rows = [line for line in transformed.splitlines() if not line.strip().startswith("#")]
  assert "\n".join(original_rows) + "\n" == source
  assert transformed.partition("void __kernel__(")[0] == source.partition("void __kernel__(")[0]


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("old_hint", ("unroll(disable)", "unroll_count(2)", "unroll_count(4)"))
@pytest.mark.parametrize("with_gcc", (False, True))
def test_row_hint_replaces_an_attached_pragma_without_conflicting_directives(hint, old_hint, with_gcc):
  loop = "  for (vecops::nint_t dsl_c_axis_1 = 0; i < 16; ++i) {}\n"
  gcc = "  #elif defined(__GNUC__)\n  #pragma GCC unroll 1\n" if with_gcc else ""
  prefix = "// outside entry stays untouched\n"
  original = f"{prefix}void __kernel__(auto A) {{\n  #if defined(__clang__)\n  #pragma clang loop {old_hint}\n{gcc}  #endif\n{loop}}}\n"
  transformed, count = row_hint_source(original, hint)
  if hint == "unchanged":
    assert (transformed, count) == (original, 0)
    return
  expected = "unroll(disable)" if hint == "disable" else f"unroll_count({2 if hint == 'two' else 4})"
  assert count == 1
  assert transformed.count("#pragma clang loop") == 1
  assert f"#pragma clang loop {expected}\n" in transformed
  assert transformed == original.replace(f"#pragma clang loop {old_hint}\n", f"#pragma clang loop {expected}\n")


@pytest.mark.parametrize("source", ("// no entry", "void __kernel__(auto A) {}"))
def test_row_hint_diagnostic_rejects_unknown_source_shape(source):
  with pytest.raises(ValueError):
    row_hint_source(source, "two")
