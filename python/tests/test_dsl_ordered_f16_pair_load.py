# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ordered full-word FP16 loads retain precise masks and logical lane order."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import (
  arm_cross_cxx_include_flags,
  assert_full_sve_memory_words,
  assert_no_vec_mask_arrays,
  cpp_compile_command,
  make_compiler,
)
from test_dsl_ordered_f16_pair_store import _body
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend

CPP_WRAPPER = r"""
#include "vecops/vec/Vec.h"
#include "vecops/vec/ConversionMemory.h"
namespace vec = vecops::vec;
using Half = vecops::float16_t;
template <typename Tag>
concept SplitLoad = vec::VectorTag<Tag> && requires(Tag tag, const Half* pointer) {
  vec::load_convert(tag, pointer, vec::mem::split);
};
static_assert(!SplitLoad<vec::ScalableTag<float, 1>>);
extern "C" void ordered_pair(const Half* input, float* output) {
  const auto tag = vec::ScalableTag<float, 1>{};
  vec::store(tag, output, vec::load_convert(tag, input));
}
extern "C" void ordered_four_words(const Half* input, float* output) {
  const auto tag = vec::ScalableTag<float, 2>{};
  vec::store(tag, output, vec::load_convert(tag, input));
}
extern "C" void ordered_word(const Half* input, float* output) {
  const auto tag = vec::ScalableTag<float, 0>{};
  vec::store(tag, output, vec::load_convert(tag, input));
}
extern "C" void ordered_subword(const Half* input, float* output) {
  const auto tag = vec::ScalableTag<float, -1>{};
  vec::store(tag, output, vec::load_convert(tag, input));
}
extern "C" void ordered_pair_first(const Half* input, float* output,
                                   vecops::nint_t count) {
  const auto tag = vec::ScalableTag<float, 1>{};
  vec::store(tag, output, vec::load_convert(
      tag, input, vec::opt::first(count), vec::opt::merge(-17.25f)));
}
extern "C" void ordered_pair_masked(const Half* input, const float* valid,
                                    const float* inactive, float* output) {
  const auto tag = vec::ScalableTag<float, 1>{};
  const auto mask = vec::cmpne(tag, vec::load(tag, valid), vec::zeros(tag));
  const auto merge = vec::load(tag, inactive);
  vec::store(tag, output, vec::load_convert(
      tag, input, vec::opt::masked(mask),
      vec::opt::merge(merge)));
}
extern "C" void ordered_non_temporal(const Half* input, float* output) {
  const auto tag = vec::ScalableTag<float, 1>{};
  vec::store(tag, output, vec::load_convert(tag, input, vec::mem::non_temporal));
}
extern "C" void ordered_fixed_one(const Half* input, float* output) {
  const auto tag = vec::FixedTag<float, 1>{};
  vec::store(tag, output, vec::load_convert(tag, input));
}
"""


def _cross(tmp_path, source, *, bits="scalable", arch="armv9-a+sve2+bf16"):
  clang, sysroot = (os.environ.get(name) for name in ("VECOPS_TEST_ARM_CROSS_CLANG", "VECOPS_TEST_ARM_CROSS_SYSROOT"))
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for actual ARM cross compilation")
  root = Path(__file__).resolve().parents[2]
  cpp, assembly = tmp_path / "pair.cpp", tmp_path / "pair.s"
  cpp.write_text(source, encoding="utf-8")
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    f"-march={arch}",
    f"-msve-vector-bits={bits}",
    "-std=c++20",
    "-O3",
    "-ffp-contract=off",
    "-DNDEBUG",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
    "-S",
    str(cpp),
    "-o",
    str(assembly),
  ]
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".command.json").write_text(json.dumps(command, indent=2))
  cpp.with_suffix(".diagnostics").write_text(result.stdout + result.stderr)
  assert result.returncode == 0, result.stderr[-8000:]
  return assembly.read_text()


@pytest.mark.parametrize("bits", ("scalable", "128", "256", "512"))
def test_ordered_f16_pair_load_real_cross_is_full_width_and_inline(tmp_path, bits):
  assembly = _cross(tmp_path, CPP_WRAPPER, bits=bits)
  for symbol, pairs in (("ordered_pair", 1), ("ordered_four_words", 2)):
    body = _body(assembly, symbol)
    assert_full_sve_memory_words(body, kind="ld1", words=pairs, bits=bits)
    assert_full_sve_memory_words(body, kind="st1", words=pairs * 2, bits=bits)
    assert len(re.findall(r"\buunpklo\b", body)) == pairs
    assert len(re.findall(r"\buunpkhi\b", body)) == pairs
    assert len(re.findall(r"\bfcvt\s+z\d+\.s", body)) == pairs * 2
    assert not re.search(r"\b(?:fcvtlt|bl|smstart|smstop)\b", body)
    assert "zip1" not in body
  for symbol in ("ordered_word", "ordered_subword", "ordered_fixed_one"):
    body = _body(assembly, symbol)
    assert len(re.findall(r"\bld1h\b", body)) == 1
    assert "unpkhi" not in body
  assert len(re.findall(r"\bld1h\b", _body(assembly, "ordered_non_temporal"))) == 2
  for symbol in ("ordered_pair_first", "ordered_pair_masked"):
    body = _body(assembly, symbol)
    assert len(re.findall(r"\bld1h\b", body)) == 1
    assert "sve_load_ordered_f16_pair" not in body
    assert "sve_widen_ordered_f16_half" not in body


def test_ordered_f16_pair_load_base_sve_real_cross(tmp_path):
  assembly = _cross(tmp_path, CPP_WRAPPER, arch="armv8-a+sve")
  body = _body(assembly, "ordered_pair")
  assert len(re.findall(r"\bld1h\b", body)) == 1
  assert "unpklo" in body and "unpkhi" in body
  assert not re.search(r"\b(?:fcvtlt|bl)\b", body)


@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32)))
def test_ordered_f16_pair_load_real_cross_separates_ordinary_and_streaming_geometry(tmp_path, vl, svl):
  source = (
    f"#define VECOPS_KERNEL_SVE_BYTES {vl}\n"
    f"#define VECOPS_KERNEL_SME_BYTES {svl}\n"
    f"#define VECOPS_TARGET_FIXED_STREAMING_SVE_BITS {svl * 8}\n" + CPP_WRAPPER
  )
  assembly = _cross(tmp_path, source, arch="armv9-a+sve2+bf16+sme+sme-fa64")
  body = _body(assembly, "ordered_pair")
  assert len(re.findall(r"\bld1h\b", body)) == 1
  assert "uunpklo" in body and "uunpkhi" in body
  assert not re.search(r"\b(?:smstart|smstop|bl)\b", body)


@pytest.mark.parametrize("bits", ("scalable", "128", "512"))
def test_public_ordered_f16_load_guard_fixture_real_cross(tmp_path, bits):
  root = Path(__file__).resolve().parents[2]
  _cross(tmp_path, (root / "tests/vec/OrderedF16PairLoadTest.cpp").read_text(), bits=bits)


def test_public_ordered_f16_load_native_scalar_oracle_and_guard_pages(tmp_path):
  root = Path(__file__).resolve().parents[2]
  executable = tmp_path / "ordered-pair-load"
  command = cpp_compile_command(
    root / "tests/vec/OrderedF16PairLoadTest.cpp",
    executable,
    flags=(
      "-std=c++20",
      "-O2",
      "-march=native",
      "-ffp-contract=off",
      "-Wno-ignored-attributes",
      "-I",
      str(root / "include"),
      "-I",
      str(root / "include/vecops"),
    ),
  )
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  (tmp_path / "native-command.json").write_text(json.dumps(command, indent=2))
  (tmp_path / "native-build.log").write_text(result.stdout + result.stderr)
  assert result.returncode == 0, result.stderr[-8000:]
  result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
  (tmp_path / "native-execution.log").write_text(result.stdout + result.stderr)
  assert result.returncode == 0, result.stderr
  assert "scalar-bit oracle, inactive lanes and guard pages passed" in result.stdout


def load_template(*, masked=False):
  N, AS, MS, CS = T.dynamic(lower=0), T.dynamic(), T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((N,), (AS,), "float16"),
    Valid: T.Tensor((N,), (MS,), "int8"),
    C: T.Tensor((N,), (CS,), "float32", access="output"),
  ):
    x = T.load(A, dtype="float32")
    if masked:
      T.store(C, x + T.cast(0.0, "float32"), mask=T.load(Valid) != 0)
    else:
      T.store(C, x)

  return kernel


@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32)))
def test_ordered_f16_load_dsl_codegen_keeps_geometry_and_capture(masked, vl, svl):
  source = (
    CPUBackend()
    .lower(load_template(masked=masked).ir, (), target="SVE2", options={"_vector_state": VectorState(vl, svl)})
    .source
  )
  assert_no_vec_mask_arrays(source)
  assert "ScalableTag<float, 1>" in source
  assert "load_convert(" in source and "store_convert(" in source
  assert "vecops_dsl::overlap(" in source
  assert "cvt::unordered" not in source and "math::fast" not in source
  assert "native-word masked remainder" in source
  if masked:
    assert "opt::masked(" in source


@pytest.fixture(scope="module")
def native_kernels(tmp_path_factory):
  directory = tmp_path_factory.mktemp("ordered-f16-loads")
  compiler = make_compiler(
    target="Native", jobs=4, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build"
  )
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for masked in (False, True):
      kernel = T.compile(load_template(masked=masked), compiler=compiler)
      kernel.operator(
        vecops.TensorMeta((129,), strides=(1,), dtype="float16"),
        vecops.TensorMeta((129,), strides=(1,), dtype="int8"),
        vecops.TensorMeta((129,), strides=(1,), dtype="float32", access=vecops.TensorAccess.output),
        **kernel._core.specialization_values,
      )
      kernels[masked] = kernel
  assert len(collector.requests) == 2
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("masked", (False, True))
def test_ordered_f16_load_dsl_native_exact_lanes_tails_signed_strides_and_alias(native_kernels, masked):
  for size in (0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129, 257):
    for layout in ("dense", "strided", "negative", "inplace-bytes", "shifted-bytes"):
      stride = {"dense": 1, "strided": 2, "negative": -2, "inplace-bytes": 1, "shifted-bytes": 1}[layout]
      # Extra room allows widening aliases to overlap the narrow source in
      # bytes, while each output still has an individually writable layout.
      backing = np.full(max(16, size * abs(stride) * 2 + 16), -99.25, dtype=np.float32)
      half = backing.view(np.float16)
      a = half[8 : 8 + size * abs(stride) : abs(stride)]
      if stride < 0:
        a = a[::-1]
      lane = np.arange(size)
      a[:] = ((lane % 127 + 1) / 128 * np.where(lane % 2 == 0, 1, -1)).astype(np.float16)
      specials = (0.0, -0.0, np.inf, -np.inf, np.nan, 2.0**-24, -(2.0**-24), 65504.0)
      for residue, value in enumerate(specials):
        a[lane % 29 == residue] = value
      expected_value = a.copy().astype(np.float32)
      valid_storage = np.ones(max(8, size * abs(stride) + 8), dtype=np.int8)
      valid = valid_storage[4 : 4 + size * abs(stride) : abs(stride)]
      if stride < 0:
        valid = valid[::-1]
      valid[:] = (lane % 4 != 1).astype(np.int8)
      if "bytes" in layout:
        c_storage = backing
        offset = 4 + int(layout == "shifted-bytes")
        c = c_storage[offset : offset + size]
      else:
        c_storage = np.full(max(8, size * abs(stride) + 8), -31.25, dtype=np.float32)
        c = c_storage[4 : 4 + size * abs(stride) : abs(stride)]
        if stride < 0:
          c = c[::-1]
      expected = c_storage.copy()
      expected_c = np.ndarray(
        c.shape, dtype=c.dtype, buffer=expected, strides=c.strides, offset=c.ctypes.data - c_storage.ctypes.data
      )
      enabled = valid != 0 if masked else np.ones(size, dtype=bool)
      # The masked fixture deliberately adds typed zero; preserve IEEE signed
      # zero rather than treating this as a bitwise identity optimization.
      if masked:
        expected_value = expected_value + np.float32(0.0)
      expected_c[enabled] = expected_value[enabled]
      native_kernels[masked](a, valid, c)
      context = f"masked={masked} size={size} layout={layout} strides={a.strides}/{c.strides}"
      np.testing.assert_array_equal(np.isnan(c_storage), np.isnan(expected), err_msg=context)
      finite = ~np.isnan(expected)
      np.testing.assert_array_equal(
        c_storage[finite].view(np.uint32), expected[finite].view(np.uint32), err_msg=context
      )
