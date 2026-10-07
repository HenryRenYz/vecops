from __future__ import annotations

import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks" / "dsl_matmul_catalog.py"
SPEC = importlib.util.spec_from_file_location("dsl_matmul_catalog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = catalog
SPEC.loader.exec_module(catalog)


def test_dtype_names_match_complete_tokens():
  assert catalog._normalize_dtype("uint8_t") == "uint8"
  assert catalog._normalize_dtype("const uint8_t&") == "uint8"
  assert catalog._normalize_dtype("int8_t") == "int8"
  assert catalog._normalize_dtype("NotAnInt8Type") is None


def test_manifest_keeps_arm_sources_and_registration_families():
  cases = catalog.load_cases(ROOT, svl_bytes=64)
  sources = {case.source for case in cases}
  assert "benchmarks/ops/MatmulBench.cpp" in sources
  assert "benchmarks/ops/MatmulFusionBench.cpp" in sources
  assert "benchmarks/ops/MatmulPackedBench.cpp" in sources
  assert "benchmarks/ops/MatmulOrientationBench.cpp" in sources
  assert any(case.catalog == "MatmulBaseRegistration" for case in cases)
  assert sum(case.catalog == "MatmulBaseRegistration" for case in cases) == 18
  assert sum(case.catalog == "MatmulFamilyExpanded" for case in cases) == 23
  assert sum(case.catalog == "MatmulPolicyExpanded" for case in cases) == 36
  assert sum(case.catalog == "MatmulBaseExpanded" for case in cases) == 194
  integer_base_cases = [
    case for case in cases if case.catalog == "MatmulBaseExpanded" and "NativeI8" in (case.configuration or "")
  ]
  assert integer_base_cases
  assert all("KStep=64" not in (case.configuration or "") for case in integer_base_cases)
  assert sum(case.catalog == "MatmulFusionExpanded" for case in cases) == 540
  assert sum(case.catalog == "MatmulOrientation" for case in cases) >= 46
  assert len(cases) > 900


def test_vector_length_is_an_explicit_catalog_input():
  narrow = catalog.load_cases(ROOT, svl_bytes=32)
  wide = catalog.load_cases(ROOT, svl_bytes=128)
  assert {case.svl_bytes for case in narrow} == {32}
  assert {case.svl_bytes for case in wide} == {128}
  assert {case.lane for case in narrow} == {8}
  assert {case.lane for case in wide} == {32}


def test_orientation_cases_retain_rank2_strides_and_dtype_overrides():
  cases = [case for case in catalog.load_cases(ROOT, svl_bytes=64) if case.catalog == "MatmulOrientation"]
  assert len(cases) >= 46
  assert all(case.dsl_supported for case in cases)
  assert all(case.memory_c == "fp32" and case.input_modes == ("raw",) for case in cases)
  assert all(case.orientation.count(",") == 2 for case in cases)
  conversion = [case for case in cases if "convert" in case.name]
  assert conversion
  assert all(case.memory_a == case.dtype_a and case.memory_b == case.dtype_b for case in conversion)


def test_packed_scenarios_expand_all_arm_modes_and_batch_shapes():
  cases = catalog.load_cases(ROOT, svl_bytes=32)
  packed = [case for case in cases if case.catalog == "MatmulPackedExpanded"]
  unresolved = [case for case in cases if case.catalog == "MatmulPackedUnresolved"]
  assert not unresolved
  assert len(packed) == 459
  by_registration = defaultdict(list)
  for case in packed:
    by_registration[case.line].append(case)
    assert case.svl_bytes == 32
    assert case.lane == 8
    assert case.metadata in (("Const", "Dynamic"), ("Dynamic",))
    assert all(
      getattr(case, field) is not None
      for field in (
        "dtype_a",
        "dtype_b",
        "accumulator",
        "memory_a",
        "memory_b",
        "memory_c",
        "memory_c_input",
        "input_pipeline_a",
        "input_pipeline_b",
        "output_pipeline",
      )
    )
  assert Counter(map(len, by_registration.values())) == {28: 8, 14: 11, 27: 3}
  assert {case.input_modes[0] for case in packed} >= {
    "raw",
    "packeda",
    "packedb",
    "packedab",
    "onlinepackeda",
    "onlinepackedb",
    "onlinepackedab",
  }
  assert any(case.dtype_b == "uint8" and case.memory_b == "uint8" for case in packed)
  probe_modes = {case.input_modes[0] for case in packed if "packed_dtype_probe_cases" in (case.configuration or "")}
  assert probe_modes == {
    "raw",
    "packeda",
    "packedb",
    "packedab",
    "onlinepackeda",
    "onlinepackedb",
    "onlinepackedab",
  }
  batched = [case for case in packed if case.batch is not None]
  assert len(batched) == 81
  assert {case.broadcast for case in batched} == {"B"}
  assert all(case.dsl_supported for case in batched)


def test_family_selector_uses_registered_selection_not_atom_alias():
  cases = [case for case in catalog.load_cases(ROOT, svl_bytes=64) if case.catalog == "MatmulFamilyExpanded"]
  assert cases
  assert all(case.dsl_supported for case in cases)
  assert all(case.configuration not in {None, "FamilyI8S8S8", "FamilyI8U8U8"} for case in cases)
  assert any(case.configuration == "kernel_family::SmallVector" for case in cases)
  assert any(case.configuration == "family_selection::Automatic" for case in cases)
  assert any("Require<kernel_family::RuntimeQuantInt8>" in case.configuration for case in cases)
  runtime_quant = [case for case in cases if case.name.startswith(("decode_", "generalized_", "medium_"))]
  assert len(runtime_quant) == 5
  assert all(case.input_modes == ("packedb",) for case in runtime_quant)
  assert all(case.memory_a == "fp32" and case.native_a == "uint8" for case in runtime_quant)
  assert all(case.memory_b == "int8" and case.native_b == "int8" for case in runtime_quant)
  assert all(case.memory_c == "fp32" and case.memory_c_input == "int32" for case in runtime_quant)
  assert all(case.input_pipeline_a == "RuntimePerRowAsymmetricQuantize" for case in runtime_quant)
  assert all(case.input_pipeline_b == "Identity" for case in runtime_quant)
  assert all(case.output_pipeline == "RuntimePerRowColumnDequantize" for case in runtime_quant)
  for case in cases:
    assert all(
      getattr(case, field) is not None
      for field in (
        "m",
        "n",
        "k",
        "dtype_a",
        "dtype_b",
        "accumulator",
        "native_a",
        "native_b",
        "memory_a",
        "memory_b",
        "memory_c",
        "memory_c_input",
        "input_pipeline_a",
        "input_pipeline_b",
        "output_pipeline",
      )
    )


def test_conversion_orientation_aliases_fall_back_to_native_bf16():
  cases = [
    case
    for case in catalog.load_cases(ROOT, svl_bytes=128)
    if case.catalog == "MatmulOrientation" and "convert" in case.name
  ]
  assert len(cases) == 4
  assert all(case.native_a == case.native_b == "bf16" for case in cases)
  assert all(case.dtype_a is not None and case.dtype_b is not None for case in cases)
  assert all(case.memory_a is not None and case.memory_b is not None for case in cases)
  assert all(case.svl_bytes == 128 and case.lane == 32 for case in cases)
