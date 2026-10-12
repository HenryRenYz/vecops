# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Generated fixed-rank shape checks retain the complete overflow contract."""

import ctypes
import subprocess
from pathlib import Path

import pytest
import vecops.language as T
from dsl_test_utils import cpp_compile_command
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.support import _PREAMBLE


@pytest.fixture(scope="module")
def shape_checker(tmp_path_factory):
  root = tmp_path_factory.mktemp("metadata-shape")
  source, library = root / "shape.cpp", root / "shape.so"
  source.write_text(_PREAMBLE + """
extern "C" int check_shape(int64_t a, int64_t b, int64_t c, size_t* result) {
  try { *result = vecops_dsl::checked_shape(a, b, c); return 0; }
  catch (const std::overflow_error&) { return 1; }
}
extern "C" size_t check_scalar_shape() { return vecops_dsl::checked_shape(); }
template<int64_t Divisor> int64_t divide(int64_t value, int operation) {
  if (operation == 0) return vecops_dsl::floordiv_positive<Divisor>(value);
  if (operation == 1) return vecops_dsl::mod_positive<Divisor>(value);
  return vecops_dsl::ceildiv_positive<Divisor>(value);
}
extern "C" int64_t check_division(int64_t value, int operation, int divisor) {
  switch (divisor) {
    case 0: return divide<1>(value, operation);
    case 1: return divide<2>(value, operation);
    case 2: return divide<3>(value, operation);
    case 3: return divide<32>(value, operation);
    default: return divide<INT64_MAX>(value, operation);
  }
}
""")
  include = Path(__file__).resolve().parents[2] / "include"
  subprocess.run(cpp_compile_command(source, library, flags=(
    "-std=c++20", "-O3", "-shared", "-fPIC", "-march=native",
    f"-I{include}", f"-I{include / 'vecops'}",
  )), check=True, capture_output=True)
  loaded = ctypes.CDLL(str(library))
  loaded.check_shape.argtypes = [ctypes.c_int64] * 3 + [ctypes.POINTER(ctypes.c_size_t)]
  loaded.check_shape.restype = ctypes.c_int
  loaded.check_scalar_shape.restype = ctypes.c_size_t
  loaded.check_division.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int]
  loaded.check_division.restype = ctypes.c_int64
  return loaded


@pytest.mark.parametrize("axes, result", [
  ((2, 3, 7), 42), ((0, 3, 7), 0), (((1 << 63) - 1, 0, (1 << 63) - 1), 0),
  (((1 << 63) - 1, 1, 1), (1 << 63) - 1),
  ((-1, 3, 7), None), ((0, -1, 7), None),
  (((1 << 63) - 1, 2, 0), None), ((1 << 32, 1 << 32, 1), None),
])
def test_fixed_rank_shape_checks_order_zero_and_overflow(shape_checker, axes, result):
  out = ctypes.c_size_t(123)
  failed = shape_checker.check_shape(*axes, ctypes.byref(out))
  if result is None:
    assert failed == 1 and out.value == 123
  else:
    assert failed == 0 and out.value == result


def test_rank_zero_shape_has_one_element(shape_checker):
  assert shape_checker.check_scalar_shape() == 1


@pytest.mark.parametrize("slot, divisor", list(enumerate([1, 2, 3, 32, (1 << 63) - 1])))
def test_constant_divisor_helpers_signed_boundaries(shape_checker, slot, divisor):
  for value in [-(1 << 63), -(1 << 63) + 1, -33, -32, -31, -1, 0, 1, 31, 32, 33, (1 << 63) - 1]:
    expected = (value // divisor, value % divisor, -((-value) // divisor))
    assert tuple(shape_checker.check_division(value, operation, slot) for operation in range(3)) == expected


@pytest.mark.parametrize("target, cost_target, specialized", [
  ("SME", "AVX512", True), ("AVX512", "SME", False),
])
def test_metadata_codegen_uses_actual_architecture_not_cost_profile(target, cost_target, specialized):
  @T.template
  def kernel(C: T.Tensor((2, 2), dtype="int64", access="output"), value: int):
    T.store(C, value // 32)

  source = CPUBackend().lower(kernel.ir, (), target=target, options={"cpu_cost_target": cost_target}).source
  body = source.split("void __kernel__", 1)[1]
  assert ("floordiv_positive<32>" in body) == specialized
  assert ("checked_shape({" not in body) == specialized


@pytest.mark.parametrize("svl", [None, 16, 32, 64, 256])
def test_derived_divisors_keep_runtime_form_without_recipe_rewrite(svl):
  @T.template
  def kernel(C: T.Tensor((2,), dtype="int64", access="output"), value: int):
    width = T.sme.streaming_lanes("float32")
    divisor = (2 * width) // 2
    T.store(C, value // divisor)

  artifact = CPUBackend().lower(kernel.ir, (), target="SME", options={"_vector_state": VectorState(None, svl)})
  body = artifact.source.split("void __kernel__", 1)[1]
  assert "vecops_dsl::floordiv(" in body, "ordinary code must keep the measured literal-only placement"


def test_divisor_specialization_does_not_follow_mutated_width():
  @T.template
  def kernel(C: T.Tensor((2,), dtype="int64", access="output"), value: int, change: bool):
    width = T.sme.streaming_lanes("float32")
    if change:
      width = 7
    T.store(C, value // width)

  artifact = CPUBackend().lower(kernel.ir, (), target="SME", options={"_vector_state": VectorState(16, 64)})
  body = artifact.source.split("void __kernel__", 1)[1]
  assert "vecops_dsl::floordiv(" in body
