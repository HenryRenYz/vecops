# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Typed grouped-dot lowering, signedness and zero-padded K tails."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def grouped_template(a_dtype="bfloat16", b_dtype="bfloat16", dtype="float32", *, transformed=False):
  k = T.dynamic(lower=0)
  group = 2 if dtype == "float32" else 4

  @T.template
  def kernel(
    A: T.Tensor((k,), (T.dynamic(),), a_dtype),
    B: T.Tensor((k,), (T.dynamic(),), b_dtype),
    C: T.Tensor(((k + group - 1) // group,), dtype=dtype, access="output"),
  ):
    a, b = T.load(A), T.load(B)
    if transformed:
      a = T.cast(T.cast(a, "float32") + 1, "bfloat16")
      b = T.cast(T.cast(b, "float32") + 2, "bfloat16")
    T.store(C, T.widening_dot(a, b, dtype=dtype))

  return kernel


def test_grouped_dot_codegen_uses_public_typed_primitive():
  for target in ("AVX512", "SME"):
    artifact = CPUBackend().lower(grouped_template().ir, (), target=target)
    assert "vecops::vec::widening_dot" in artifact.source
    assert "vecops::vec::ViewAs<vecops::bfloat16_t" in artifact.source
    assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("transformed", (False, True))
def test_bf16_grouped_dot_odd_empty_and_strided_tails(transformed, tmp_path):
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(grouped_template(transformed=transformed), compiler=compiler)
  for k in (0, 1, 2, 3, 17, 31, 32, 33, 65):
    backing_a = np.full(k + 32, np.nan, dtype=bf16)
    backing_a[16 : 16 + k] = np.asarray(np.arange(k) % 7 / 16, dtype=bf16)
    a = backing_a[16 : 16 + k][::-1]
    backing_b = np.full(2 * k + 32, np.nan, dtype=bf16)
    backing_b[16 : 16 + 2 * k : 2] = np.asarray(np.arange(k) % 11 / 16, dtype=bf16)
    b = backing_b[16 : 16 + 2 * k : 2]
    aa, bb = a.astype(np.float32), b.astype(np.float32)
    if transformed:
      aa, bb = (aa + 1).astype(bf16).astype(np.float32), (bb + 2).astype(bf16).astype(np.float32)
    expected = np.pad(aa * bb, (0, k % 2)).reshape(-1, 2).sum(axis=1)
    raw = np.full((k + 1) // 2 + 2, -99, dtype=np.float32)
    output = raw[1:-1]
    compiled.operator.native(a, b, output, **compiled._core.specialization_values)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(raw[[0, -1]], -99)
  cpuinfo = Path("/proc/cpuinfo")
  if cpuinfo.is_file() and "avx512_bf16" in cpuinfo.read_text() and not transformed:
    modules = tuple((tmp_path / "build").rglob("*.so"))
    assert modules
    assembly = "\n".join(subprocess.check_output(["objdump", "-d", str(module)], text=True) for module in modules)
    assert "vdpbf16ps" in assembly


@pytest.mark.parametrize(
  "a_dtype,b_dtype", (("int8", "int8"), ("uint8", "uint8"), ("int8", "uint8"), ("uint8", "int8"))
)
@pytest.mark.parametrize("dtype", ("int32", "uint32"))
def test_byte_grouped_dot_signedness_and_padding(a_dtype, b_dtype, dtype, tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(grouped_template(a_dtype, b_dtype, dtype), compiler=compiler)
  for k in (0, 1, 3, 4, 5, 17, 63, 65):
    a = (np.arange(k, dtype=np.int64) * 131 + 127).astype(a_dtype)[::-1]
    b = (np.arange(k * 2, dtype=np.int64) * 127 + 255).astype(b_dtype)[::2]
    sums = np.pad(a.astype(np.int64) * b.astype(np.int64), (0, (-k) % 4)).reshape(-1, 4).sum(axis=1)
    expected = (sums & 0xFFFFFFFF).astype(np.uint32).view(dtype)
    output = np.empty((k + 3) // 4, dtype=dtype)
    compiled.operator.native(a, b, output, **compiled._core.specialization_values)
    np.testing.assert_array_equal(output, expected)
  cpuinfo = Path("/proc/cpuinfo")
  if (
    cpuinfo.is_file()
    and "avx512_vnni" in cpuinfo.read_text()
    and (a_dtype, b_dtype, dtype) == ("uint8", "int8", "int32")
  ):
    modules = tuple((tmp_path / "build").rglob("*.so"))
    assert modules
    assembly = "\n".join(subprocess.check_output(["objdump", "-d", str(module)], text=True) for module in modules)
    assert "vpdpbusd" in assembly


@pytest.mark.parametrize(
  "a_dtype,b_dtype,dtype",
  (("float16", "float16", "float32"), ("bfloat16", "bfloat16", "int32"), ("int8", "uint8", "float32")),
)
def test_grouped_dot_rejects_unsupported_type_combinations(a_dtype, b_dtype, dtype):
  with pytest.raises((TypeError, ValueError), match="widening_dot supports"):
    CPUBackend().lower(grouped_template(a_dtype, b_dtype, dtype).ir, (), target="AVX512")


def test_integer_inner_dot_reduction_wraps_without_signed_overflow(tmp_path):
  k = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((1, k), dtype="uint8"),
    B: T.Tensor((k, 1), dtype="uint8"),
    D: T.Tensor((1, 1), dtype="int32"),
    C: T.Tensor((1, 1), dtype="int32", access="output"),
  ):
    T.dot((A, B), C, acc=T.load(D), acc_dtype="int32", implementation="vector")

  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  seed = np.asarray([[2147483647]], dtype=np.int32)
  for count in (0, 1, 3, 17, 33029, 66053):
    a = np.full((1, count), 255, dtype=np.uint8)
    b = np.full((count, 1), 255, dtype=np.uint8)
    output = np.empty((1, 1), dtype=np.int32)
    compiled.operator.native(a, b, seed, output, **compiled._core.specialization_values)
    expected = np.asarray([(count * 255 * 255 + 2147483647) & 0xFFFFFFFF], dtype=np.uint32).view(np.int32)
    np.testing.assert_array_equal(output.ravel(), expected)


def test_independent_dynamic_lengths_are_checked_in_native_and_debug(tmp_path):
  k, l = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((k,), dtype="uint8"),
    B: T.Tensor((l,), dtype="int8"),
    C: T.Tensor(((k + 3) // 4,), dtype="int32", access="output"),
  ):
    T.store(C, T.widening_dot(T.load(A), T.load(B), dtype="int32"))

  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  for count_a, count_b in ((3, 1), (1, 3), (0, 1), (1, 0)):
    a = np.full(count_a, 3, dtype=np.uint8)
    b = np.full(count_b, -2, dtype=np.int8)
    output = np.full((count_a + 3) // 4, 99, dtype=np.int32)
    with pytest.raises((ValueError, RuntimeError), match="length mismatch"):
      compiled.operator.native(a, b, output, **compiled._core.specialization_values)
    np.testing.assert_array_equal(output, 99)
    with pytest.raises((ValueError, AssertionError), match="length mismatch"):
      compiled.interpret(a, b, output)
  for count in (0, 1, 3, 4, 5):
    a, b = np.full(count, 3, dtype=np.uint8), np.full(count, -2, dtype=np.int8)
    output = np.empty((count + 3) // 4, dtype=np.int32)
    compiled.operator.native(a, b, output, **compiled._core.specialization_values)
    native = output.copy()
    compiled.interpret(a, b, output)
    np.testing.assert_array_equal(output, native)
