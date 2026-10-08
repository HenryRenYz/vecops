# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ordered FP32-pair stores preserve logical lanes, rounding and addressing.

The cross tests inspect genuine SVE instructions; local Native execution is
an independent portability/numerical gate, not an ARM performance claim.
"""

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
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def pair_store_template(*, masked=False):
  N, AS, MS, CS, DS = T.dynamic(lower=0), T.dynamic(), T.dynamic(), T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((N,), (AS,), "float32"),
    Valid: T.Tensor((N,), (MS,), "int8"),
    C: T.Tensor((N,), (CS,), "float16", access="output"),
    D: T.Tensor((N,), (DS,), "float32", access="output"),
  ):
    value = T.load(A)
    if masked:
      valid = T.load(Valid) != 0
      T.store(C, value, mask=valid)
      T.store(D, T.cast(T.cast(value, "float16"), "float32"), mask=valid)
    else:
      T.store(C, value)
      T.store(D, T.cast(T.cast(value, "float16"), "float32"))

  return kernel


def _cpp_arguments(value):
  """Split generated call arguments without splitting a template or address."""
  result, start, depth = [], 0, 0
  for index, character in enumerate(value):
    if character in "<({[":
      depth += 1
    elif character in ">)}]":
      depth -= 1
    elif character == "," and depth == 0:
      result.append(value[start:index].strip())
      start = index + 1
  return [*result, value[start:].strip()]


def _half_rounding_pipeline(source):
  """Find F32 A -> F16 value -> F32 D, including generated Tag/SSA aliases.

  The spelling of a Tag is not its dtype: a direct ScalableTag<F16> and
  Rebind<F16, SomeF32Tag> have the same compute-element contract. Require the
  actual F16 producer to reach the store into D, not just two conversion
  tokens somewhere in a source containing unrelated snapshot operations.
  """
  tags = dict(re.findall(r"using (\w+)\s*=\s*([^;]+);", source))
  aliases = dict(re.findall(r"\bauto (\w+)\s*=\s*(\w+)\s*;", source))

  def dtype(tag):
    tag = tag.removesuffix("{}").strip()
    visited = set()
    while tag in tags:
      assert tag not in visited, f"recursive generated Tag alias: {tag}"
      visited.add(tag)
      tag = tags[tag].strip()
    match = re.fullmatch(r"vecops::vec::(?:ScalableTag|FixedTag|Rebind)<(.+)>", tag)
    assert match, f"unrecognized generated conversion Tag: {tag}"
    return _cpp_arguments(match[1])[0]

  def calls(line, operation):
    prefix = f"vecops::vec::{operation}("
    for match in re.finditer(re.escape(prefix), line):
      start, depth = match.end(), 1
      for index in range(start, len(line)):
        depth += (line[index] == "(") - (line[index] == ")")
        if depth == 0:
          yield _cpp_arguments(line[start:index])
          break

  half_loads, stores = set(), []
  for line in source.splitlines():
    assignment = re.match(r"\s*auto (\w+)\s*=", line)
    if assignment:
      for arguments in calls(line, "load_convert"):
        if dtype(arguments[0]) == "vecops::float16_t":
          assert "dsl_c_parameter_C_data_" not in arguments[1]
          if "dsl_c_parameter_A_data_" in arguments[1]:
            half_loads.add(assignment[1])
    for arguments in calls(line, "store_convert"):
      if dtype(arguments[0]) == "vecops::float16_t" and "dsl_c_parameter_D_data_" in arguments[1]:
        stores.append(arguments[2])
  assert half_loads, "no typed F16 conversion of captured F32 input A"
  assert stores, "no typed F16 -> F32 store into D"
  for value in stores:
    visited = set()
    while value in aliases:
      assert value not in visited, f"recursive generated SSA alias: {value}"
      visited.add(value)
      value = aliases[value]
    if value in half_loads:
      return
  raise AssertionError("F16-producing load does not reach the widening output store into D")


@pytest.mark.parametrize(
  "tag_declaration,tag",
  (
    ("using HalfTag = vecops::vec::ScalableTag<vecops::float16_t, 0>;", "HalfTag"),
    ("using Tag = vecops::vec::ScalableTag<float, 1>;", "vecops::vec::Rebind<vecops::float16_t, Tag>"),
  ),
)
def test_half_rounding_pipeline_checks_dtype_and_connected_value(tag_declaration, tag):
  source = f"""
  {tag_declaration}
  auto half = vecops::vec::load_convert({tag}{{}}, dsl_c_parameter_A_data_1);
  auto forwarded = half;
  vecops::vec::store_convert({tag}{{}}, dsl_c_parameter_D_data_10, forwarded);
  """
  _half_rounding_pipeline(source)
  with pytest.raises(AssertionError, match="does not reach"):
    _half_rounding_pipeline(source.replace(", forwarded)", ", unrelated)"))
  with pytest.raises(AssertionError, match="no typed F16"):
    _half_rounding_pipeline(source.replace("vecops::float16_t", "float"))


@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32)))
def test_ordered_f16_pair_store_lowering_keeps_math_and_alias_contract(masked, vl, svl):
  source = (
    CPUBackend()
    .lower(
      pair_store_template(masked=masked).ir,
      (),
      target="SVE2",
      options={"_vector_state": VectorState(vl, svl)},
    )
    .source
  )
  assert_no_vec_mask_arrays(source)
  assert "vecops::vec::store_convert(" in source
  # The compiler may fuse the intermediate cast into the memory operation,
  # but must still demote to FP16 before the FP32 output conversion.
  _half_rounding_pipeline(source)
  assert "cvt::unordered" not in source and "math::fast" not in source
  assert "vecops_dsl::overlap(" in source  # Preserve before-write captures.
  assert "ScalableTag<float, 1>" in source  # Existing default two-word schedule.
  if masked:
    assert "opt::masked(" in source


CPP_WRAPPER = r"""
#include "vecops/vec/Vec.h"
#include "vecops/vec/ConversionMemory.h"
namespace vec = vecops::vec;
using Half = vecops::float16_t;
extern "C" void ordered_pair(const float* input, Half* output) {
  const auto tag = vec::ScalableTag<float, 1>{};
  const auto value = vec::load(tag, input);
  vec::store_convert(tag, output, value);
}
extern "C" void ordered_pair_masked(const float* input, const float* valid,
                                    Half* output) {
  const auto tag = vec::ScalableTag<float, 1>{};
  const auto value = vec::load(tag, input);
  const auto mask = vec::cmpne(tag, vec::load(tag, valid), vec::zeros(tag));
  vec::store_convert(tag, output, value, vec::opt::masked(mask));
}
extern "C" void ordered_pair_first(const float* input, Half* output,
                                   vecops::nint_t active) {
  const auto tag = vec::ScalableTag<float, 1>{};
  const auto value = vec::load(tag, input);
  vec::store_convert(tag, output, value, vec::opt::first(active));
}
extern "C" void ordered_four_words(const float* input, Half* output) {
  const auto tag = vec::ScalableTag<float, 2>{};
  const auto value = vec::load(tag, input);
  vec::store_convert(tag, output, value);
}
extern "C" void ordered_split_pair(const float* input, Half* output) {
  const auto tag = vec::ScalableTag<float, 1>{};
  const auto value = vec::load(tag, input);
  vec::store_convert(tag, output, value, vec::mem::split);
}
extern "C" void ordered_fixed_one(const float* input, Half* output) {
  const auto tag = vec::FixedTag<float, 1>{};
  const auto value = vec::load(tag, input);
  vec::store_convert(tag, output, value);
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
  cpp.with_suffix(".command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
  cpp.with_suffix(".diagnostics").write_text(result.stdout + result.stderr, encoding="utf-8")
  assert result.returncode == 0, result.stderr[-8000:]
  return assembly.read_text(encoding="utf-8")


def _body(assembly, symbol):
  match = re.search(rf"^{symbol}:.*?^\s*\.size\s+{symbol}\b", assembly, re.MULTILINE | re.DOTALL)
  assert match, f"missing complete assembly body for {symbol}"
  return match.group()


@pytest.mark.parametrize("bits", ("scalable", "128", "256", "512"))
def test_sve_pair_store_real_cross_assembly_is_full_width_and_inline(tmp_path, bits):
  assembly = _cross(tmp_path, CPP_WRAPPER, bits=bits)
  pair = _body(assembly, "ordered_pair")
  assert len(re.findall(r"\bfcvt\s+z\d+\.h", pair)) == 2
  assert len(re.findall(r"\buzp1\s+z\d+\.h", pair)) == 1
  assert len(re.findall(r"\bst1h\b", pair)) == 1
  assert not re.search(r"\bbl\b", pair)
  larger = _body(assembly, "ordered_four_words")
  assert len(re.findall(r"\bfcvt\s+z\d+\.h", larger)) == 4
  assert_full_sve_memory_words(larger, kind="ld1", words=4, bits=bits)
  assert_full_sve_memory_words(larger, kind="st1", words=2, bits=bits)
  assert not re.search(r"\bbl\b", larger)
  split = _body(assembly, "ordered_split_pair")
  assert len(re.findall(r"\bst1h\b", split)) == 2
  for name in ("ordered_pair_masked", "ordered_pair_first"):
    body = _body(assembly, name)
    assert len(re.findall(r"\bst1h\b", body)) == 1
    assert "sve_pack_ordered_f32_pair_to_f16" not in body
  assert len(re.findall(r"\bst1h\b", _body(assembly, "ordered_fixed_one"))) == 1


def test_sve_pair_store_is_legal_on_base_sve(tmp_path):
  assembly = _cross(tmp_path, CPP_WRAPPER, arch="armv8-a+sve")
  body = _body(assembly, "ordered_pair")
  assert "fcvt" in body and "uzp1" in body and "st1h" in body
  assert not re.search(r"\b(?:fcvtnt|bl)\b", body)


@pytest.mark.parametrize("bits", ("scalable", "128", "512"))
def test_public_ordered_pair_canary_and_guard_fixture_cross_compiles(tmp_path, bits):
  root = Path(__file__).resolve().parents[2]
  _cross(tmp_path, (root / "tests/vec/OrderedF16PairStoreTest.cpp").read_text(encoding="utf-8"), bits=bits)


def test_public_ordered_pair_store_native_scalar_oracle_and_guard_pages(tmp_path):
  root = Path(__file__).resolve().parents[2]
  source = root / "tests/vec/OrderedF16PairStoreTest.cpp"
  executable = tmp_path / "ordered-pair"
  command = cpp_compile_command(
    source,
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
  (tmp_path / "native-command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
  (tmp_path / "native-build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
  assert result.returncode == 0, result.stderr[-8000:]
  result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
  (tmp_path / "native-execution.log").write_text(result.stdout + result.stderr, encoding="utf-8")
  assert result.returncode == 0, result.stderr
  assert "scalar-bit oracle, canaries and guard pages passed" in result.stdout


@pytest.fixture(scope="module")
def native_kernels(tmp_path_factory):
  directory = tmp_path_factory.mktemp("ordered-f16-pairs")
  compiler = make_compiler(
    target="Native", jobs=4, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build"
  )
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for masked in (False, True):
      kernel = T.compile(pair_store_template(masked=masked), compiler=compiler)
      kernel.operator(
        vecops.TensorMeta((129,), strides=(1,), dtype="float32"),
        vecops.TensorMeta((129,), strides=(1,), dtype="int8"),
        vecops.TensorMeta((129,), strides=(1,), dtype="float16", access=vecops.TensorAccess.output),
        vecops.TensorMeta((129,), strides=(1,), dtype="float32", access=vecops.TensorAccess.output),
        **kernel._core.specialization_values,
      )
      kernels[masked] = kernel
  assert len(collector.requests) == len(kernels) == 2
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


def _values(size):
  lane = np.arange(size)
  midpoint = np.float32(1) + (lane % 23).astype(np.float32) / np.float32(64) + np.float32(1 / 2048)
  direction = np.where(lane % 3 == 0, np.float32(0), np.float32(2))
  value = np.where(lane % 3 == 2, midpoint, np.nextafter(midpoint, direction))
  value[lane % 2 == 1] *= -1
  special = (0.0, -0.0, np.inf, -np.inf, np.nan, 2.0**-24, -(2.0**-24), 2.0**-25, -(2.0**-25), 65504.0, 65520.0)
  for residue, item in enumerate(special):
    value[lane % 29 == residue] = item
  return value.astype(np.float32)


def _view(size, dtype, stride, fill):
  storage = np.full(max(8, size * abs(stride) + 8), fill, dtype=dtype)
  view = storage[4 : 4 + size * abs(stride) : abs(stride)]
  if stride < 0:
    view = view[::-1]
  assert view.size == size
  return storage, view


@pytest.mark.parametrize("masked", (False, True))
def test_ordered_pair_dsl_native_ties_specials_tails_strides_and_capture(native_kernels, masked):
  for size in (0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129, 257):
    for layout in ("dense", "strided", "negative", "inplace-bytes", "shifted-bytes"):
      stride = {"dense": 1, "strided": 2, "negative": -2, "inplace-bytes": 1, "shifted-bytes": 1}[layout]
      a_storage, a = _view(size, np.float32, stride, 123.5)
      a[:] = _values(size)
      _, valid = _view(size, np.int8, stride, 1)
      valid[:] = (np.arange(size) % 4 != 1).astype(np.int8)
      if "bytes" in layout:
        c_storage = a_storage.view(np.float16)
        offset = 8 + int(layout == "shifted-bytes")
        c = c_storage[offset : offset + size]
      else:
        c_storage, c = _view(size, np.float16, stride, -31.25)
      d_storage, d = _view(size, np.float32, stride, -91.5)
      c_expected, d_expected = c_storage.copy(), d_storage.copy()
      c_view = np.ndarray(
        c.shape, dtype=c.dtype, buffer=c_expected, offset=c.ctypes.data - c_storage.ctypes.data, strides=c.strides
      )
      d_view = np.ndarray(
        d.shape, dtype=d.dtype, buffer=d_expected, offset=d.ctypes.data - d_storage.ctypes.data, strides=d.strides
      )
      with np.errstate(over="ignore", invalid="ignore"):
        rounded = a.copy().astype(np.float16)
      enabled = valid != 0 if masked else np.ones(size, dtype=bool)
      c_view[enabled] = rounded[enabled]
      d_view[enabled] = rounded[enabled].astype(np.float32)
      native_kernels[masked](a, valid, c, d)
      context = f"size={size} layout={layout} masked={masked}"
      for actual, expected in ((c_storage, c_expected), (d_storage, d_expected)):
        nan = np.isnan(expected)
        np.testing.assert_array_equal(np.isnan(actual), nan, err_msg=context)
        np.testing.assert_array_equal(
          actual[~nan].view(np.dtype(f"u{actual.dtype.itemsize}")),
          expected[~nan].view(np.dtype(f"u{expected.dtype.itemsize}")),
          err_msg=context,
        )
