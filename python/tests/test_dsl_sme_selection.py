# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""SME selection epilogues preserve ordinary vector and captured-read contracts."""
import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import nodes


@T.func
def select_and_convert(value, dtype):
  return T.cast(T.max(value, 0), dtype)


def selection_kernel(dtype, basis="float32"):
  extent = T.dynamic(lower=0, upper=8 if basis == "float64" else 16)

  @T.template
  def kernel(A: T.Tensor((extent,), dtype=basis),
             C: T.Tensor((extent,), dtype=dtype, access="output"),
             Reference: T.Tensor((extent,), dtype=dtype, access="output")):
    T.store(Reference, T.cast(T.max(T.load(A), 0), dtype))
    with T.sme.region():
      value = T.sme.load(A)
      selected = T.sme.apply(select_and_convert, value, dtype)
      T.sme.store(selected, C)

  return kernel


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_sme_apply_source_is_one_owner_without_memory_staging(dtype):
  artifact = CPUBackend().lower(selection_kernel(dtype).ir, (), target="SME",
                                options={"_vector_state": VectorState(64, 64)})
  assert artifact.source.count(".with_resources(") == 1
  assert any(isinstance(s, I.ArchitectureCall) and s.operation == "sme.apply" for s in nodes(artifact.kernel.body))
  owner = artifact.source.split(".with_resources(", 1)[1]
  assert "vecops::vec::max(" in owner and "vecops::vec::store_convert(" in owner
  assert "throw " not in owner.split("#else", 1)[0]


@pytest.mark.parametrize("kind, fused", [("relu", True), ("rounding", True), ("arithmetic", False),
                                           ("memory", False), ("wide", False)])
def test_dot_selection_fuses_only_closed_narrow_dags(kind, fused):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))
  dtype = "float64" if kind == "wide" else "float32"

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="bfloat16"), B: T.Tensor((k, n), dtype="bfloat16"),
             Prior: T.Tensor((1, n), dtype="float32"), C: T.Tensor((m, n), dtype=dtype, access="output")):
    value = T.dot(T.load(A), T.load(B), acc=T.load(Prior), implementation="sme")
    if kind == "relu":
      value = T.max(value, 0)
    elif kind == "rounding":
      value = T.cast(T.cast(value, "float16"), "float32")
    elif kind == "arithmetic":
      value = value * 2
    elif kind == "memory":
      value = T.max(value + T.load(Prior), 0)
    else:
      value = T.max(T.cast(value, "float64"), 0)
    T.store(C, value)

  artifact = CPUBackend().lower(kernel.ir, (), target="SME", options={"_vector_state": VectorState(64, 64)})
  calls = [s for s in nodes(artifact.kernel.body) if isinstance(s, I.ArchitectureCall) and s.operation == "sme.apply"]
  assert bool(calls) == fused


def test_fp64_full_schedule_uses_all_eight_tiles():
  @T.template
  def kernel(A: T.Tensor((16, 64), dtype="float64"), B: T.Tensor((64, 32), dtype="float64"),
             C: T.Tensor((16, 32), dtype="float64", access="output")):
    T.store(C, T.dot(T.load(A), T.load(B), implementation="sme"))

  artifact = CPUBackend().lower(kernel.ir, (), target="SME", options={"_vector_state": VectorState(64, 64)})
  assert {s.tile for s in nodes(artifact.kernel.body) if isinstance(s, I.ArchitectureCall) and s.operation == "sme.mopa"} >= set(range(8))


@pytest.mark.skipif(platform.machine() != "aarch64", reason="requires native SME")
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_native_selection_matches_nonstreaming_bits_for_special_values(dtype, tmp_path):
  import ml_dtypes

  compiled = T.compile(selection_kernel(dtype), compiler=make_compiler(
    jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  bits = np.array([0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000,
                   0x7f800000, 0xff800000, 0x7fc01234, 0x7f801234,
                   0x3f808000, 0x3f818000, 0x33800000, 0xb3800000,
                   0x38800000, 0x7f7fffff], dtype=np.uint32)
  native_dtype = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
  for extent in [0, 1, 7, 16]:
    a = bits[:extent].view(np.float32).copy()
    out, expected = np.empty(extent, native_dtype), np.empty(extent, native_dtype)
    compiled.operator.native(a, out, expected, **compiled._core.specialization_values)
    np.testing.assert_array_equal(out.view(np.uint8), expected.view(np.uint8))


@pytest.mark.skipif(platform.machine() != "aarch64", reason="requires native SME")
def test_native_fused_prior_alias_matches_nonstreaming_selection(tmp_path):
  import ml_dtypes

  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="bfloat16"), B: T.Tensor((k, n), dtype="bfloat16"),
             Prior: T.Tensor((1, n), dtype="float32"), C: T.Tensor((m, n), dtype="float32", access="output"),
             Reference: T.Tensor((m, n), dtype="float32", access="output")):
    T.store(Reference, T.max(T.broadcast_to(T.load(Prior), C.shape), 0))
    value = T.dot(T.load(A), T.load(B), acc=T.load(Prior), implementation="sme")
    T.store(C, T.max(value, 0))

  compiled = T.compile(kernel, compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  bits = np.array([0, 0x80000000, 1, 0x80000001, 0x00800000, 0x7f800000,
                   0xff800000, 0x7fc01234, 0x7f801234], dtype=np.uint32)
  for rows, columns in [(32, 32), (19, 23)]:
    a, b = np.empty((rows, 0), ml_dtypes.bfloat16), np.empty((0, columns), ml_dtypes.bfloat16)
    out, expected = np.empty((rows, columns), np.float32), np.empty((rows, columns), np.float32)
    out.view(np.uint32)[0] = np.resize(bits, columns)
    prior = out[:1]
    compiled.operator.native(a, b, prior, out, expected, **compiled._core.specialization_values)
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


@T.func
def select_scalar_limit(value, limit):
  return T.min(value, T.cast(limit, "float16"))


@pytest.mark.skipif(platform.machine() != "aarch64", reason="requires native SME")
def test_native_selection_scalar_conversion_uses_public_contract(tmp_path):
  @T.template
  def kernel(A: T.Tensor((16,), dtype="float32"), C: T.Tensor((16,), dtype="float32", access="output"),
             Reference: T.Tensor((16,), dtype="float32", access="output"), limit: T.float32):
    T.store(Reference, T.min(T.load(A), T.cast(limit, "float16")))
    with T.sme.region():
      value = T.sme.load(A)
      selected = T.sme.apply(select_scalar_limit, value, limit)
      T.sme.store(selected, C)

  compiled = T.compile(kernel, compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  a = np.linspace(-2, 2, 16, dtype=np.float32)
  out, expected = np.empty_like(a), np.empty_like(a)
  for limit in [0., -0., 2.**-24, -(2.**-24), 65520., float("inf"), float("nan")]:
    compiled.operator.native(a, out, expected, limit, **compiled._core.specialization_values)
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


@T.func
def select_uint16(value):
  return T.cast(value, "uint16")


@pytest.mark.skipif(platform.machine() != "aarch64", reason="requires native SME")
def test_native_chained_selection_retains_narrowed_geometry(tmp_path):
  @T.template
  def kernel(A: T.Tensor((16,), dtype="float32"), C: T.Tensor((16,), dtype="uint16", access="output"),
             Reference: T.Tensor((16,), dtype="uint16", access="output")):
    T.store(Reference, T.cast(T.cast(T.max(T.load(A), 0), "bfloat16"), "uint16"))
    with T.sme.region():
      value = T.sme.load(A)
      narrowed = T.sme.apply(select_and_convert, value, "bfloat16")
      selected = T.sme.apply(select_uint16, narrowed)
      T.sme.store(selected, C)

  compiled = T.compile(kernel, compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  a = np.array([-1., 0., 1., 1.5, 2.5, 3.75, 7., 31.9, 128., 257., 511., 1024., 4095., 65504.,
                float("inf"), float("nan")], dtype=np.float32)
  out, expected = np.empty(16, np.uint16), np.empty(16, np.uint16)
  compiled.operator.native(a, out, expected, **compiled._core.specialization_values)
  np.testing.assert_array_equal(out, expected)


@T.func
def arithmetic_selection(value):
  return value + 1


@T.func
def widening_selection(value):
  return T.cast(value, "float64")


@pytest.mark.parametrize("helper", [arithmetic_selection, widening_selection])
def test_explicit_sme_apply_rejects_arithmetic_and_widening(helper):
  @T.template
  def kernel(A: T.Tensor((16,), dtype="float32"),
             C: T.Tensor((16,), dtype="float32", access="output")):
    with T.sme.region():
      value = T.sme.load(A)
      result = T.sme.apply(helper, value)
      T.sme.store(result, C)

  with pytest.raises(Exception, match="memory-free conversion, selection and comparison"):
    CPUBackend().lower(kernel.ir, (), target="SME", options={"_vector_state": VectorState(64, 64)})


@pytest.mark.skipif(platform.machine() != "aarch64", reason="requires native SME FA64")
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
def test_native_fp64_selection_matches_nonstreaming_special_values(dtype, tmp_path):
  import ml_dtypes

  compiled = T.compile(selection_kernel(dtype, basis="float64"), compiler=make_compiler(
    jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  native_dtype = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
  groups = [np.array([0., -0., 2.**-1074, -(2.**-1074), 2.**-149, -(2.**-149),
                      float("inf"), -float("inf")], np.float64),
            np.array([0x7ff0000000000001, 0x7ff8000000001234, 0x3ff0020000000000,
                      0x3ff0060000000000, 0x3800000000000000, 0x3810000000000000,
                      0x47efffffe0000000, 0x7fefffffffffffff], np.uint64).view(np.float64)]
  for data in groups:
    for extent in [0, 1, 7, 8]:
      out, expected = np.empty(extent, native_dtype), np.empty(extent, native_dtype)
      compiled.operator.native(data[:extent].copy(), out, expected, **compiled._core.specialization_values)
      np.testing.assert_array_equal(out.view(np.uint8), expected.view(np.uint8))


@T.func
def select_integer_limit(value, limit):
  return T.min(value, T.cast(limit, "uint8"))


def integer_limit_kernel():
  @T.template
  def kernel(A: T.Tensor((16,), dtype="float32"), C: T.Tensor((16,), dtype="float32", access="output"),
             Reference: T.Tensor((16,), dtype="float32", access="output"), limit: T.int64):
    T.store(Reference, T.min(T.load(A), T.cast(limit, "uint8")))
    with T.sme.region():
      selected = T.sme.apply(select_integer_limit, T.sme.load(A), limit)
      T.sme.store(selected, C)

  return kernel


def test_sme_selection_integer_scalar_cast_uses_saturation():
  artifact = CPUBackend().lower(integer_limit_kernel().ir, (), target="SME",
                                options={"_vector_state": VectorState(64, 64)})
  owner = artifact.source.split(".with_resources(", 1)[1]
  assert "vecops::convert<std::uint8_t>" in owner


@pytest.mark.skipif(platform.machine() != "aarch64", reason="requires native SME")
def test_native_selection_integer_scalar_cast_matches_public_saturation(tmp_path):
  compiled = T.compile(integer_limit_kernel(), compiler=make_compiler(
    jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  a = np.linspace(-2, 400, 16, dtype=np.float32)
  out, expected = np.empty_like(a), np.empty_like(a)
  for limit in [-257, -1, 0, 255, 256, 4096, (1 << 63) - 1]:
    compiled.operator.native(a, out, expected, limit, **compiled._core.specialization_values)
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))
