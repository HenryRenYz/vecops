# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Instruction recipes preserve producer, accumulator and sink semantics."""

import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_source_algorithms_lowered, make_compiler
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
  assert_source_algorithms_lowered(artifact.backend_ir)
  assert "vecops::vec::sme::mopa<0" in artifact.source
  assert "vecops::vec::sme::load_hor<0>" in artifact.source
  assert "staged SME matrix instruction recipe" in "\n".join(artifact.kernel.diagnostics)


@pytest.mark.parametrize("family", AMX_FAMILIES)
def test_amx_families_are_source_instruction_programs_with_fusion(family):
  artifact = CPUBackend().lower(matrix_template(*family, implementation="amx").ir, (), target="AVX512")
  assert artifact.backend_ir.hardware
  assert_source_algorithms_lowered(artifact.backend_ir)
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
  assert_source_algorithms_lowered(artifact.backend_ir)
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


@pytest.mark.skipif(platform.machine() != "x86_64", reason="native AMX target required")
@pytest.mark.parametrize("converting", [False, True])
def test_native_amx_full_k_pack_preserves_column_stride_and_output_alias(converting, tmp_path):
  import ml_dtypes

  native_a, native_b = ("bfloat16", "bfloat16") if converting else ("int8", "uint8")
  memory_b, accumulator = ("float32", "float32") if converting else ("uint8", "int32")
  bs, cs0, cs1 = T.dynamic(), T.dynamic(), T.dynamic()

  @T.template
  def kernel(A: T.Tensor((32, 64), dtype=native_a), B: T.Tensor((64, 32), (1, bs), memory_b),
             C: T.Tensor((32, 32), (cs0, cs1), accumulator, access="output")):
    value = T.dot(T.load(A), T.load(B, dtype=native_b), acc_dtype=accumulator, implementation="amx")
    T.store(C, value)

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  for reversed_columns in (False, True):
    for alias in (False, True):
      a = (np.arange(32 * 64).reshape(32, 64) % 7 - 3).astype(ml_dtypes.bfloat16 if converting else np.int8)
      if converting:
        b = np.asfortranarray((np.arange(64 * 32).reshape(64, 32) % 13 - 6).astype(np.float32) / 3)
      else:
        backing = np.empty(32 * 32 * 4, np.uint8)
        b = np.ndarray((64, 32), np.uint8, backing, strides=(1, 64))
        b[:] = (np.arange(64 * 32).reshape(64, 32) % 19 + 230).astype(np.uint8)
      if reversed_columns:
        b = b[:, ::-1]
      compute_b = b.astype(ml_dtypes.bfloat16).astype(np.float32) if converting else b.astype(np.int64)
      expected = a.astype(np.float32 if converting else np.int64) @ compute_b
      if alias:
        output = b.T[:, :32] if converting else np.ndarray((32, 32), np.int32, backing)
      else:
        output = np.empty((32, 32), dtype=accumulator)
      compiled.operator.native(a, b, output, **compiled._core.specialization_values)
      np.testing.assert_array_equal(output, expected.astype(accumulator))
  assert "vecops::vec::amx::dot<" in compiled.generated_source


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


def external_ab_template(implementation, library=False, a_dtype="int8", b_dtype="uint8", acc_dtype="int32"):
  m, n, k, ca, cb = (T.dynamic(lower=0) for _ in range(5))

  @T.template
  def kernel(AP: T.Tensor((ca,), dtype=a_dtype), BP: T.Tensor((cb,), dtype=b_dtype),
             Shape: T.Tensor((m, k), dtype=a_dtype), Prior: T.Tensor((1, n), dtype=acc_dtype), C: T.Tensor((m, n), dtype=acc_dtype, access="output")):
    a = T.packed(AP, (m, k), side="A", format=implementation)
    b = T.packed(BP, (k, n), side="B", format=implementation)
    if library:
      value = T.library.matmul(a, b, acc=T.load(Prior), implementation=implementation)
    else:
      value = T.dot(a, b, acc=T.load(Prior), implementation=implementation)
    T.store(C, value * 2 + 1)

  return kernel


@pytest.mark.parametrize("library", [False, True])
def test_native_external_ab_accumulator_epilogue_and_invalid_capacity(library, tmp_path):
  implementation = "amx" if platform.machine() == "x86_64" else "sme"
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(external_ab_template(implementation, library), compiler=compiler)
  for m, n, k in [(32, 64, 64), (33, 35, 65), (7, 19, 0)]:
    a = (np.arange(m * k).reshape(m, k) % 17 - 8).astype(np.int8)
    b = (np.arange(k * n).reshape(k, n) % 19 + 230).astype(np.uint8)
    if implementation == "amx":
      mp, npad, kp = (m + 15) // 16 * 16, (n + 15) // 16 * 16, (k + 63) // 64 * 64
      aa, bb = np.zeros((mp, kp), np.int8), np.zeros((npad, kp), np.uint8)
      aa[:m, :k], bb[:n, :k] = a, b.T
      ap = aa.reshape(mp // 16, 16, kp // 64, 64).transpose(0, 2, 1, 3).copy().ravel()
      bp = bb.reshape(npad // 16, 16, kp // 64, 16, 4).transpose(0, 2, 3, 1, 4).copy().ravel()
    else:
      from vecops._arm_vector_state import get_vector_state
      panel = 2 * get_vector_state().sme_bytes // 4
      kp = (k + 3) // 4 * 4
      aa, bb = np.zeros(((m + panel - 1) // panel * panel, kp), np.int8), np.zeros(((n + panel - 1) // panel * panel, kp), np.uint8)
      aa[:m, :k], bb[:n, :k] = a, b.T
      ap = aa.reshape(-1, panel, kp // 4, 4).transpose(0, 2, 1, 3).copy().ravel() if kp else np.empty(0, np.int8)
      bp = bb.reshape(-1, panel, kp // 4, 4).transpose(0, 2, 1, 3).copy().ravel() if kp else np.empty(0, np.uint8)
    prior = (np.arange(n) - 10).astype(np.int32).reshape(1, n)
    out = np.empty((m, n), np.int32)
    kernel.operator.native(ap, bp, a, prior, out, **kernel._core.specialization_values)
    np.testing.assert_array_equal(out, (a.astype(np.int32) @ b.astype(np.int32) + prior) * 2 + 1)
    if k and kp > k:
      pad_a, pad_b = np.zeros(aa.shape, bool), np.zeros(bb.shape, bool)
      pad_a[:, k:], pad_b[:, k:] = True, True
      if implementation == "amx":
        pa = pad_a.reshape(mp // 16, 16, kp // 64, 64).transpose(0, 2, 1, 3).copy().ravel()
        pb = pad_b.reshape(npad // 16, 16, kp // 64, 16, 4).transpose(0, 2, 3, 1, 4).copy().ravel()
      else:
        pa = pad_a.reshape(-1, panel, kp // 4, 4).transpose(0, 2, 1, 3).copy().ravel()
        pb = pad_b.reshape(-1, panel, kp // 4, 4).transpose(0, 2, 1, 3).copy().ravel()
      ap[pa], bp[pb] = 13, 251
      kernel.operator.native(ap, bp, a, prior, out, **kernel._core.specialization_values)
      np.testing.assert_array_equal(out, (a.astype(np.int32) @ b.astype(np.int32) + prior) * 2 + 1)
    if ap.size:
      with pytest.raises(RuntimeError, match="packed input buffer too small"):
        kernel.operator.native(ap[:-1], bp, a, prior, out, **kernel._core.specialization_values)


@pytest.mark.parametrize("library", [False, True])
def test_native_external_bf16_nan_k_padding_is_ignored(library, tmp_path):
  import ml_dtypes

  implementation = "amx" if platform.machine() == "x86_64" else "sme"
  dtype = ml_dtypes.bfloat16
  if implementation == "sme":
    from vecops._arm_vector_state import get_vector_state
    panel = 2 * get_vector_state().sme_bytes // 4
  spatial = 32 if implementation == "amx" else panel
  a = (np.arange(spatial * 17).reshape(spatial, 17) % 5 - 2).astype(dtype)
  b = (np.arange(17 * spatial).reshape(17, spatial) % 7 - 3).astype(dtype)
  kp = 32 if implementation == "amx" else 18
  aa, bb = np.full((spatial, kp), np.nan, dtype), np.full((spatial, kp), np.nan, dtype)
  aa[:, :17], bb[:, :17] = a, b.T
  if implementation == "amx":
    ap = aa.reshape(2, 16, 1, 32).transpose(0, 2, 1, 3).copy().ravel()
    bp = bb.reshape(2, 16, 1, 16, 2).transpose(0, 2, 3, 1, 4).copy().ravel()
  else:
    ap = aa.reshape(1, panel, kp // 2, 2).transpose(0, 2, 1, 3).copy().ravel()
    bp = bb.reshape(1, panel, kp // 2, 2).transpose(0, 2, 1, 3).copy().ravel()
  prior = (np.arange(spatial) - 10).astype(np.float32).reshape(1, spatial)
  out = np.empty((spatial, spatial), np.float32)
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(external_ab_template(implementation, library, "bfloat16", "bfloat16", "float32"), compiler=compiler)
  kernel.operator.native(ap, bp, a, prior, out, **kernel._core.specialization_values)
  np.testing.assert_array_equal(out, (a.astype(np.float32) @ b.astype(np.float32) + prior) * 2 + 1)


def test_native_converting_scalar_pack_preserves_overlapping_capture(tmp_path):
  import ml_dtypes
  m, k, capacity = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="float32", access="inout"),
             P: T.Tensor((capacity,), dtype="bfloat16", access="output"),
             C: T.Tensor((m, k), dtype="bfloat16", access="output")):
    original = T.load(A, dtype="bfloat16")
    T.pack(original, P, side="A", format="auto")
    T.store(A, T.zeros(A.shape, dtype="float32"))
    T.store(C, original)

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  # An odd K exercises the scalar remainder, with both alias decisions.
  for overlap in (False, True):
    storage = np.empty(8192, np.float32)
    a = storage[:13 * 17].reshape(13, 17)
    a[:] = np.arange(a.size).reshape(a.shape) / 7 - 10
    expected = a.astype(ml_dtypes.bfloat16)
    p = storage.view(ml_dtypes.bfloat16) if overlap else np.empty(16384, ml_dtypes.bfloat16)
    c = np.empty(a.shape, ml_dtypes.bfloat16)
    compiled.operator.native(a, p, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, expected)
    np.testing.assert_array_equal(a, 0)


@pytest.mark.parametrize("library", [False, True])
def test_native_inline_epilogue_broadcast_preserves_output_alias(library, tmp_path):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="int8"), B: T.Tensor((k, n), dtype="uint8"),
             Prior: T.Tensor((1, n), dtype="int32"), C: T.Tensor((m, n), dtype="int32", access="output")):
    if library:
      product = T.library.matmul(T.load(A), T.load(B), acc=T.load(Prior))
    else:
      product = T.dot(T.load(A), T.load(B), acc=T.load(Prior))
    # This fresh load is intentionally inline: expanding the sink into row
    # stores must capture it before the first row overwrites Prior.
    T.store(C, product * 2 + T.load(Prior))

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  a = (np.arange(32 * 64).reshape(32, 64) % 17 - 8).astype(np.int8)
  b = (np.arange(64 * 32).reshape(64, 32) % 19 + 230).astype(np.uint8)
  out = np.empty((32, 32), np.int32)
  out[0] = np.arange(32) - 10
  prior = out[:1]
  expected = (a.astype(np.int32) @ b.astype(np.int32) + prior.copy()) * 2 + prior.copy()
  compiled.operator.native(a, b, prior, out, **compiled._core.specialization_values)
  np.testing.assert_array_equal(out, expected)


@pytest.mark.skipif(platform.machine() == "x86_64", reason="SME FP64 target required")
def test_native_default_fp64_cache_producers_prior_and_epilogue_alias(tmp_path):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="float64"), B: T.Tensor((k, n), dtype="float64"),
             Prior: T.Tensor((1, n), dtype="float64"), C: T.Tensor((m, n), dtype="float64", access="output")):
    product = T.library.matmul(T.load(A) * 2, T.load(B) + 1, acc=T.load(Prior), implementation="sme")
    T.store(C, product * 0.25 + T.load(Prior))

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  # MatmulBase's medium shape exceeds the default whole-problem input budget
  # on 920f-4. Prior aliases the first output row, so final stores must retain
  # its old value across all rows; initialization and epilogue each occur once.
  a = (np.arange(64 * 256).reshape(64, 256) % 7 - 3).astype(np.float64)
  b = (np.arange(256 * 256).reshape(256, 256) % 5 - 2).astype(np.float64)
  out = np.empty((64, 256), np.float64)
  out[0] = np.arange(256) % 11 - 5
  prior = out[:1]
  old_prior = prior.copy()
  expected = ((a * 2) @ (b + 1) + old_prior) * 0.25 + old_prior
  compiled.operator.native(a, b, prior, out, **compiled._core.specialization_values)
  np.testing.assert_array_equal(out, expected)


@pytest.mark.skipif(platform.machine() == "x86_64", reason="SME FP64 target required")
@pytest.mark.parametrize("library", [False, True])
def test_native_broadcast_za_seed_preserves_float_bits_and_alias(library, tmp_path):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="float64"), B: T.Tensor((k, n), dtype="float64"),
             Prior: T.Tensor((1, n), dtype="float64"), C: T.Tensor((m, n), dtype="float64", access="output")):
    if library:
      product = T.library.matmul(T.load(A), T.load(B), acc=T.load(Prior), implementation="sme")
    else:
      product = T.dot(T.load(A), T.load(B), acc=T.load(Prior), implementation="sme")
    T.store(C, product)

  compiled = T.compile(kernel, compiler=make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  # An arithmetic seed (e.g. MOPA with ones) changes -0 and can quiet NaNs.
  # K=0 isolates initialization from multiply/accumulate semantics. Exercise
  # complete register blocks, spatial fringes and a bias alias of the output.
  bits = np.array([0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
                   0x7ff8000000001234, 0x7ff0000000004321], dtype=np.uint64)
  for rows, columns in [(32, 32), (17, 19)]:
    a, b = np.empty((rows, 0), np.float64), np.empty((0, columns), np.float64)
    out = np.empty((rows, columns), np.float64)
    out.view(np.uint64)[0] = np.resize(bits, columns)
    prior = out[:1]
    expected = np.broadcast_to(prior.view(np.uint64).copy(), (rows, columns))
    compiled.operator.native(a, b, prior, out, **compiled._core.specialization_values)
    np.testing.assert_array_equal(out.view(np.uint64), expected)


@pytest.mark.parametrize("family", BYTE_PAIRS + [("bfloat16", "bfloat16", "float32")])
def test_native_external_packed_k_prefix_and_poisoned_terminal_group(family, tmp_path):
  """Borrowed prefixes and private suffixes preserve every signedness pair.

  The zero-prefix, full-K and long-prefix cases exercise the instruction
  boundary independently of packing code. Padding is deliberately invalid;
  borrowed input bytes and guard elements must survive the invocation.
  """
  import ml_dtypes

  implementation = "amx" if platform.machine() == "x86_64" else "sme"
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(external_ab_template(implementation, False, *family), compiler=compiler)
  dtypes = [ml_dtypes.bfloat16 if d == "bfloat16" else np.dtype(d) for d in family]
  if implementation == "amx":
    panel, quantum = 16, 64 // np.dtype(dtypes[0]).itemsize
  else:
    from vecops._arm_vector_state import get_vector_state
    panel = 2 * get_vector_state().sme_bytes // 4
    quantum = 4 // np.dtype(dtypes[0]).itemsize
  group = 4 // np.dtype(dtypes[0]).itemsize

  def physical(rows, spatial, kpad, side):
    if implementation == "amx":
      view = rows.reshape(spatial // 16, 16, kpad // quantum, quantum)
      if side == "A":
        return view.transpose(0, 2, 1, 3).copy().ravel()
      return view.reshape(spatial // 16, 16, kpad // quantum, 16, group).transpose(0, 2, 3, 1, 4).copy().ravel()
    return rows.reshape(spatial // panel, panel, kpad // group, group).transpose(0, 2, 1, 3).copy().ravel()

  for m, n, k in [(32, 32, 0), (32, 32, 1), (32, 32, 64), (32, 32, 257),
                   (19, 23, 65), (33, 35, 129)]:
    a = (np.arange(m * k).reshape(m, k) % 7 + (249 if family[0] == "uint8" else -3)).astype(dtypes[0])
    b = (np.arange(k * n).reshape(k, n) % 5 + (251 if family[1] == "uint8" else -2)).astype(dtypes[1])
    mp, npad, kp = (m + panel - 1) // panel * panel, (n + panel - 1) // panel * panel, (k + quantum - 1) // quantum * quantum
    aa, bb = np.empty((mp, kp), dtypes[0]), np.empty((npad, kp), dtypes[1])
    if family[0] == "bfloat16":
      # Includes signalling/quiet NaN payloads and both infinities. Repair
      # selects storage bits; no arithmetic may touch invalid K members.
      poison = np.array([0x7f81, 0x7fc3, 0x7f80, 0xff80], np.uint16)
      aa.view(np.uint16)[:] = np.resize(poison, aa.shape)
      bb.view(np.uint16)[:] = np.resize(poison, bb.shape)
    else:
      aa[:], bb[:] = 13, 251 if family[1] == "uint8" else -5
    aa[:m, :k], bb[:n, :k] = a, b.T
    ap, bp = physical(aa, mp, kp, "A"), physical(bb, npad, kp, "B")
    before_a, before_b = ap.view(np.uint8).copy(), bp.view(np.uint8).copy()
    backing = np.full(m * n + 8, -777, dtypes[2])
    output = backing[4:-4].reshape(m, n)
    output[0] = np.arange(n) - 10 if family[2] == "float32" else np.iinfo(np.int32).max - np.arange(n) % 3
    prior = output[:1]
    compute = np.float32 if family[2] == "float32" else np.int64
    expected = ((a.astype(compute) @ b.astype(compute) + prior.astype(compute).copy()) * 2 + 1).astype(dtypes[2])
    compiled.operator.native(ap, bp, a, prior, output, **compiled._core.specialization_values)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(ap.view(np.uint8), before_a)
    np.testing.assert_array_equal(bp.view(np.uint8), before_b)
    np.testing.assert_array_equal(backing[:4], -777)
    np.testing.assert_array_equal(backing[-4:], -777)


@pytest.mark.parametrize("side", ["A", "B"])
@pytest.mark.parametrize("family", [("int8", "uint8", "int32"), ("bfloat16", "bfloat16", "float32")])
@pytest.mark.skipif(platform.machine() == "x86_64", reason="SME terminal-group consumer required")
def test_native_sme_one_sided_packed_prefix_and_strided_epilogue(side, family, tmp_path):
  import ml_dtypes
  from vecops._arm_vector_state import get_vector_state

  m, n, k, capacity = (T.dynamic(lower=0) for _ in range(4))
  sd = family[0] if side == "A" else family[1]

  @T.template
  def template(A: T.Tensor((m, k), dtype=family[0]), B: T.Tensor((k, n), dtype=family[1]),
               P: T.Tensor((capacity,), dtype=sd), Prior: T.Tensor((1, n), dtype=family[2]),
               C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), family[2], access="output")):
    a = T.packed(P, (m, k), side="A", format="sme") if side == "A" else T.load(A)
    b = T.packed(P, (k, n), side="B", format="sme") if side == "B" else T.load(B)
    product = T.dot(a, b, acc=T.load(Prior), implementation="sme")
    T.store(C, product * 2 + T.load(Prior))

  compiled = T.compile(template, compiler=make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  dtypes = [ml_dtypes.bfloat16 if d == "bfloat16" else np.dtype(d) for d in family]
  panel, group = 2 * get_vector_state().sme_bytes // 4, 4 // np.dtype(dtypes[0]).itemsize
  for rows, columns, reduction in [(32, 32, 1), (19, 23, 65), (33, 35, 257)]:
    a = (np.arange(rows * reduction).reshape(rows, reduction) % 7).astype(dtypes[0])
    b = (np.arange(reduction * columns).reshape(reduction, columns) % 5).astype(dtypes[1])
    source = a if side == "A" else b.T
    spatial, kp = (source.shape[0] + panel - 1) // panel * panel, (reduction + group - 1) // group * group
    dense = np.full((spatial, kp), np.nan if family[0] == "bfloat16" else 13, dtype=source.dtype)
    dense[:source.shape[0], :reduction] = source
    packed = dense.reshape(-1, panel, kp // group, group).transpose(0, 2, 1, 3).copy().ravel()
    original = packed.view(np.uint8).copy()
    backing = np.full((rows + 2, 2 * columns + 3), -777, dtypes[2])
    output = backing[1:rows + 1, 1:2 * columns + 1:2][::-1, ::-1]
    output[0] = np.arange(columns) - 10
    prior = output[:1].copy()
    compute = np.float32 if family[2] == "float32" else np.int64
    expected = ((a.astype(compute) @ b.astype(compute) + prior) * 2 + prior).astype(dtypes[2])
    compiled.operator.native(a, b, packed, prior, output, **compiled._core.specialization_values)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(packed.view(np.uint8), original)
    np.testing.assert_array_equal(backing[[0, -1]], -777)
    np.testing.assert_array_equal(backing[:, 2:2 * columns + 1:2], -777)
