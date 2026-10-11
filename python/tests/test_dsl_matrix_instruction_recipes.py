# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Instruction recipes preserve producer, accumulator and sink semantics."""

import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend

BYTE_PAIRS = [(a, b, "int32") for a in ("int8", "uint8") for b in ("int8", "uint8")]
SME_FAMILIES = BYTE_PAIRS + [("int16", "int16", "int64"), ("int16", "int16", "int32")] + [
  (dtype, dtype, acc) for dtype, acc in [("float16", "float32"), ("bfloat16", "float32"),
                                        ("float32", "float32"), ("float64", "float64")]
]
AMX_FAMILIES = BYTE_PAIRS + [(dtype, dtype, "float32") for dtype in ("bfloat16", "float16")]


def matrix_template(a_dtype, b_dtype, acc_dtype, *, implementation="auto", fusion=True):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype=a_dtype), B: T.Tensor((k, n), dtype=b_dtype),
             Prior: T.Tensor((m, n), dtype=acc_dtype),
             C: T.Tensor((m, n), dtype=acc_dtype, access="output")):
    left, right = T.load(A), T.load(B)
    if fusion:
      left = T.cast(left * 2, a_dtype)
      right = T.cast(right + 1, b_dtype)
      product = T.dot(left, right, acc=T.load(Prior), acc_dtype=acc_dtype, implementation=implementation)
      T.store(C, product * 2 + 1)
    else:
      T.store(C, T.dot(left, right, acc_dtype=acc_dtype, implementation=implementation))

  return kernel


@pytest.mark.parametrize("family", SME_FAMILIES)
def test_sme_families_are_source_instruction_programs_with_fusion(family):
  artifact = CPUBackend().lower(matrix_template(*family, implementation="sme").ir, (), target="SME")
  assert artifact.backend_ir.hardware
  assert not artifact.backend_ir.matmuls
  assert "vecops::vec::sme::mopa<0" in artifact.source
  assert "vecops::vec::sme::load_hor<0>" in artifact.source
  assert "staged SME matrix instruction recipe" in "\n".join(artifact.kernel.diagnostics)


@pytest.mark.parametrize("family", AMX_FAMILIES)
def test_amx_families_are_source_instruction_programs_with_fusion(family):
  artifact = CPUBackend().lower(matrix_template(*family, implementation="amx").ir, (), target="AVX512")
  assert artifact.backend_ir.hardware
  assert not artifact.backend_ir.matmuls
  assert "vecops::vec::amx::dot<" in artifact.source
  assert "vecops::vec::amx::load<0>" in artifact.source
  assert "staged AMX matrix instruction recipe" in "\n".join(artifact.kernel.diagnostics)


NATIVE_FAMILIES = AMX_FAMILIES[:-1] if platform.machine() == "x86_64" else SME_FAMILIES


@pytest.mark.parametrize("family", NATIVE_FAMILIES)
@pytest.mark.parametrize("fusion", [False, True])
def test_native_recipes_empty_tails_and_fusion(family, fusion, tmp_path):
  import ml_dtypes

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(matrix_template(*family, fusion=fusion), compiler=compiler)
  dtypes = [ml_dtypes.bfloat16 if d == "bfloat16" else np.dtype(d) for d in family]
  for m, n, k in [(0, 19, 17), (19, 0, 17), (7, 19, 0), (1, 1, 1), (7, 19, 17),
                   (32, 32, 64), (64, 64, 96), (33, 35, 65)]:
    a = (np.arange(m * k).reshape(m, k) % 7).astype(dtypes[0])
    b = (np.arange(k * n).reshape(k, n) % 5).astype(dtypes[1])
    prior = (np.arange(m * n).reshape(m, n) % 3).astype(dtypes[2])
    out = np.empty((m, n), dtype=dtypes[2])
    compute = np.float64 if family[2].startswith("float") else np.int64
    aa = (a.astype(compute) * 2).astype(dtypes[0]) if fusion else a
    bb = (b.astype(compute) + 1).astype(dtypes[1]) if fusion else b
    expected = (aa.astype(compute) @ bb.astype(compute)).astype(dtypes[2])
    if fusion:
      expected = ((expected + prior) * 2 + 1).astype(dtypes[2])
    kernel.operator.native(a, b, prior, out, **kernel._core.specialization_values)
    np.testing.assert_array_equal(out, expected)


def blocked_template(implementation="auto"):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), "int8"),
             B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), "uint8"),
             Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), "int32"),
             C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), "int32", access="output")):
    result = T.library.matmul(T.cast(T.load(A) * 2, "int8"), T.cast(T.load(B) + 1, "uint8"),
                              acc=T.load(Prior), acc_dtype="int32", implementation=implementation,
                              BM=13, BN=17, BK=3)
    T.store(C, result * 2 + 1)

  return kernel


@pytest.mark.parametrize("target,implementation", [("AVX512", "amx"), ("SME", "sme")])
def test_cache_phases_stage_instruction_leaves_and_final_epilogue(target, implementation):
  artifact = CPUBackend().lower(blocked_template(implementation).ir, (), target=target)
  assert artifact.backend_ir.hardware
  assert not artifact.backend_ir.matmuls
  assert f"vecops::vec::{implementation}::" in artifact.source


def test_native_cache_phases_signed_strides_broadcast_and_output_alias(tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(blocked_template(), compiler=compiler)
  for m, n, k in [(7, 19, 0), (7, 19, 17), (33, 35, 65)]:
    a = (np.arange(m * k).reshape(m, k) % 7).astype(np.int8)[::-1, ::-1]
    b = (np.arange(k * n).reshape(k, n) % 5).astype(np.uint8)[::-1, ::-1]
    storage = np.full((m + 2, 2 * n + 3), -777, dtype=np.int32)
    output = storage[1:m + 1, 1:2 * n + 1:2][::-1, ::-1]
    # The row initializer aliases the first output row. Every later row must
    # see its captured initial values, rather than the first row's epilogue.
    prior = output[:1, :]
    prior[:] = np.arange(n)
    expected = ((a.astype(np.int32) * 2) @ (b.astype(np.int32) + 1) + prior.copy()) * 2 + 1
    kernel.operator.native(a, b, prior, output, **kernel._core.specialization_values)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(storage[[0, -1]], -777)
    np.testing.assert_array_equal(storage[:, 2:2 * n + 1:2], -777)


def packed_b_template():
  m, n, k, capacity = (T.dynamic(lower=0) for _ in range(4))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="int8"), PackedB: T.Tensor((capacity,), dtype="uint8"),
             C: T.Tensor((m, n), dtype="int32", access="output")):
    b = T.packed(PackedB, (k, n), side="B", format="amx")
    T.store(C, T.dot(T.load(A), b, acc_dtype="int32", implementation="amx"))

  return kernel


def test_packed_b_reuse_is_staged_as_a_checked_source_view():
  artifact = CPUBackend().lower(packed_b_template().ir, (), target="AVX512")
  assert artifact.backend_ir.hardware
  assert "packed input buffer too small" in artifact.source
  assert "packed input and matrix output may not overlap" in artifact.source
  assert "vecops::vec::amx::dot<" in artifact.source


@pytest.mark.skipif(platform.machine() != "x86_64", reason="AMX target required")
def test_native_amx_external_b_panels_direct_and_fractional_k(tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(packed_b_template(), compiler=compiler)
  for m, n, k in [(32, 32, 64), (33, 35, 65), (7, 19, 0)]:
    a = (np.arange(m * k).reshape(m, k) % 13 - 6).astype(np.int8)
    b = (np.arange(k * n).reshape(k, n) % 17).astype(np.uint8)
    npad, kpad = (n + 15) // 16 * 16, (k + 63) // 64 * 64
    dense = np.zeros((kpad, npad), dtype=np.uint8)
    dense[:k, :n] = b
    packed = dense.reshape(kpad // 64, 16, 4, npad // 16, 16).transpose(3, 0, 1, 4, 2).copy().ravel()
    out = np.empty((m, n), dtype=np.int32)
    kernel.operator.native(a, packed, out, **kernel._core.specialization_values)
    np.testing.assert_array_equal(out, a.astype(np.int32) @ b.astype(np.int32))


@pytest.mark.parametrize("family", BYTE_PAIRS)
def test_native_signedness_and_modular_accumulator(family, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(matrix_template(*family), compiler=compiler)
  for m, n, k in [(32, 32, 64), (7, 19, 17)]:
    a = (np.arange(m * k).reshape(m, k) % 13 - 6 if family[0] == "int8" else
         np.arange(m * k).reshape(m, k) % 15 + 240).astype(family[0])
    b = (np.arange(k * n).reshape(k, n) % 9 - 4 if family[1] == "int8" else
         np.arange(k * n).reshape(k, n) % 13 + 240).astype(family[1])
    prior = np.full((m, n), np.iinfo(np.int32).max - 2, dtype=np.int32)
    out = np.empty_like(prior)
    aa = (a.astype(np.int64) * 2).astype(family[0])
    bb = (b.astype(np.int64) + 1).astype(family[1])
    expected = ((aa.astype(np.int64) @ bb.astype(np.int64) + prior.astype(np.int64)) * 2 + 1).astype(np.int32)
    kernel.operator.native(a, b, prior, out, **kernel._core.specialization_values)
    np.testing.assert_array_equal(out, expected)


def test_sme_i16_optional_sign_pairs_and_geometry_guards():
  for a in ("int16", "uint16"):
    for b in ("int16", "uint16"):
      artifact = CPUBackend().lower(matrix_template(a, b, "int64", implementation="sme").ir, (), target="SME")
      assert "defined(HAS_SME_I16I64)" in artifact.source
      assert "FIXED_SVE_BITS == FIXED_STREAMING_SVE_BITS" in artifact.source
  with pytest.raises(TypeError, match="unsupported sme dot dtype"):
    CPUBackend().lower(matrix_template("int16", "uint16", "int32", implementation="sme").ir, (), target="SME")


def test_native_historical_input_and_broadcast_epilogue_alias(tmp_path):
  import ml_dtypes

  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def template(A: T.Tensor((m, k), dtype="bfloat16", access="inout"),
               B: T.Tensor((k, n), dtype="bfloat16"), Bias: T.Tensor((1, n), dtype="float32"),
               C: T.Tensor((m, n), dtype="float32", access="output")):
    captured = T.load(A)
    initial_bias = T.load(Bias)
    T.store(A, T.zeros(A.shape, dtype="bfloat16"))
    product = T.dot(T.cast(captured + 2, "bfloat16"), B, acc=1.25)
    T.store(C, product * 2 + initial_bias)

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(template, compiler=compiler)
  for rows, columns, reduction in [(32, 32, 64), (7, 19, 17)]:
    a = (np.arange(rows * reduction).reshape(rows, reduction) % 7 - 3).astype(ml_dtypes.bfloat16)
    b = (np.arange(reduction * columns).reshape(reduction, columns) % 5 - 2).astype(ml_dtypes.bfloat16)
    out = np.empty((rows, columns), np.float32)
    bias = out[:1]
    bias[:] = np.arange(columns)
    expected = ((a.astype(np.float32) + 2) @ b.astype(np.float32) + 1.25) * 2 + bias.copy()
    kernel.operator.native(a, b, bias, out, **kernel._core.specialization_values)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(a.astype(np.float32), 0)


def test_native_direct_rows_accept_negative_input_and_output_stride(tmp_path):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def template(A: T.Tensor((m, k), (T.dynamic(), 1), "int8"),
               B: T.Tensor((k, n), dtype="uint8"),
               C: T.Tensor((m, n), (T.dynamic(), 1), "int32", access="output")):
    T.store(C, T.dot(A, B))

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(template, compiler=compiler)
  a = (np.arange(32 * 64).reshape(32, 64) % 13 - 6).astype(np.int8)[::-1]
  b = (np.arange(64 * 32).reshape(64, 32) % 17).astype(np.uint8)
  out = np.empty((32, 32), np.int32)[::-1]
  kernel.operator.native(a, b, out, **kernel._core.specialization_values)
  np.testing.assert_array_equal(out, a.astype(np.int32) @ b.astype(np.int32))


def test_native_i16_vector_fallback_wraps_int64_accumulator(tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(matrix_template("int16", "uint16", "int64", implementation="vector"), compiler=compiler)
  a = (np.arange(7 * 17).reshape(7, 17) % 7 * 5000 - 15000).astype(np.int16)
  b = (np.arange(17 * 19).reshape(17, 19) % 9 * 7000).astype(np.uint16)
  prior = np.full((7, 19), np.iinfo(np.int64).max - 2, dtype=np.int64)
  aa = (a.astype(np.int64) * 2).astype(np.int16).astype(np.int64)
  bb = (b.astype(np.int64) + 1).astype(np.uint16).astype(np.int64)
  expected = (aa @ bb + prior) * 2 + 1
  out = np.empty_like(prior)
  kernel.operator.native(a, b, prior, out, **kernel._core.specialization_values)
  np.testing.assert_array_equal(out, expected)
