"""Math exceptional-path placement follows emitted ownership, not ISA flags."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import arm_cross_cxx_include_flags, assert_no_vec_mask_arrays, make_compiler
from test_dsl_transpose_sme_metadata import transpose_template
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES

POLICY = "#define VECOPS_SVE_INLINE_MATH_SPECIALS 1"


def ordinary_template(dtype="float32"):
  n = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((n,), dtype=dtype), C: T.Tensor((n,), dtype=dtype, access="output")):
    T.store(C, T.rsqrt(T.load(A), accuracy="strict"))

  return kernel


def dot_template():
  @T.template
  def kernel(
    A: T.Tensor((25, 32), dtype="bfloat16"),
    B: T.Tensor((32, 41), (1, 32), "bfloat16"),
    C: T.Tensor((25, 41), dtype="float32", access="output"),
  ):
    T.dot((A, B), C)

  return kernel


def explicit_template():
  @T.template
  def kernel():
    with T.sme.region():
      T.sme.zero_za()

  return kernel


def lower(template, *, target="SME", vl=32, svl=64):
  return CPUBackend().lower(template.ir, (), target=target, options={"_vector_state": VectorState(vl, svl)})


@pytest.mark.parametrize("target", ("SVE", "SME", "generic", "AVX512"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_ordinary_vector_kernel_keeps_cold_math_outlined_on_sme_capable_target(target, dtype):
  source = lower(ordinary_template(dtype), target=target).source
  assert POLICY not in source
  assert "vecops::vec::rsqrt" in source
  assert "execution::sme::Region{}" not in source
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("factory", (dot_template, transpose_template, explicit_template))
@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (32, 32)))
def test_actual_streaming_owners_select_policy_before_every_header(factory, vl, svl):
  source = lower(factory(), vl=vl, svl=svl).source
  assert source.count(POLICY) == 1
  assert source.index(POLICY) < source.index("#include")
  assert "execution::sme::Region{}" in source
  assert "VECOPS_INLINE_LAMBDA_NOEXCEPT" in source
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_native_ordinary_math_keeps_exceptional_values_and_tails(tmp_path, dtype):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(ordinary_template(dtype), compiler=compiler)
  meta = vecops.TensorMeta((37,), dtype=dtype)
  with vecops.collect_compile_requests() as collector:
    kernel.operator(meta, meta, **kernel._core.specialization_values)
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  assert POLICY not in kernel.get_kernel_source()
  assert_no_vec_mask_arrays(kernel.get_kernel_source())
  for length in (0, 1, 7, 31, 37, 127):
    values = np.resize(np.array((0.0, -0.0, np.inf, -np.inf, np.nan, -1.0, 1.0, 2.0, 0.125), dtype=dtype), length)
    output = np.empty(length, dtype=dtype)
    kernel.operator.native(values, output, **kernel._core.specialization_values)
    with np.errstate(divide="ignore", invalid="ignore"):
      expected = np.array(1, dtype=dtype) / np.sqrt(values)
    # The public vec rsqrt contract intentionally maps both signed zeros to
    # +infinity (the estimate-instruction convention), unlike 1/sqrt(-0).
    expected[values == 0] = np.inf
    np.testing.assert_allclose(output, expected, rtol=4 * np.finfo(dtype).eps, atol=0, equal_nan=True)
    np.testing.assert_array_equal(np.signbit(output[np.isinf(expected)]), np.signbit(expected[np.isinf(expected)]))
  artifacts = kernel.debug.artifacts()
  assert artifacts
  assert all("-ffp-contract=off" in item["command"]["command"] for item in artifacts)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16)))
def test_real_cross_ordinary_math_has_outlined_special_helpers(tmp_path, dtype, vl, svl):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set ARM cross compiler/sysroot for actual code generation")
  source = lower(ordinary_template(dtype), vl=vl, svl=svl).source
  assert POLICY not in source
  element = CPP_TYPES[dtype]
  wrapper = "\n".join(
    (
      source,
      f'extern "C" void ordinary({element}* ap,{element}* cp,vecops::nint_t n) {{',
      "auto shape=vecops::tensor::make_shape(vecops::meta::Any{n});",
      "auto strides=vecops::tensor::make_strides(vecops::meta::cint<1>);",
      "auto A=vecops::tensor::make_tensor(ap,shape,strides);",
      "auto C=vecops::tensor::make_tensor(cp,shape,strides);",
      "__kernel__(A,C);",
      "}",
    )
  )
  cpp, asm, obj = tmp_path / "ordinary.cpp", tmp_path / "ordinary.s", tmp_path / "ordinary.o"
  cpp.write_text(wrapper)
  root = Path(__file__).resolve().parents[2]
  flags = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sve2+bf16+sme+sme-fa64+sme-f64f64",
    "-msve-vector-bits=scalable",
    "-std=c++20",
    "-O3",
    "-ffp-contract=off",
    "-DNDEBUG",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
  ]
  for output, mode in ((asm, "-S"), (obj, "-c")):
    subprocess.run([*flags, mode, str(cpp), "-o", str(output)], capture_output=True, text=True, check=True)
  text = asm.read_text()
  suffix = "f32" if dtype == "float32" else "f64"
  assert re.search(rf"\b(?:bl|b)\s+[^\n]*sve_rsqrt_{suffix}_special", text)
  assert "smstart" not in text and "smstop" not in text
  assert obj.stat().st_size
