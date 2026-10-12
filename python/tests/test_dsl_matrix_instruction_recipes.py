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
