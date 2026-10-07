"""SME's constant-panel proof uses its real K quantum, not generic 32."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import vecops.language as T
from dsl_test_utils import arm_cross_cxx_include_flags, assert_no_vec_mask_arrays, make_compiler
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES

DTYPES = ("float32", "float64", "float16", "bfloat16", "int8", "uint8")


def _template(dtype, k, packed):
  M, N, K = T.Const(5), T.Const(19), T.Const(k)
  acc = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  PSA, PSB = T.dynamic(lower=0), T.dynamic(lower=0)
  D0, D1 = T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype=dtype),
    B: T.Tensor((K, N), (1, K), dtype),
    AP: T.Tensor((PSA,), dtype=dtype),
    BP: T.Tensor((PSB,), dtype=dtype),
    D: T.Tensor((M, N), (D0, D1), acc),
    C: T.Tensor((M, N), dtype=acc, access="output"),
    Bias: T.Tensor((1, N), dtype=acc),
  ):
    a = T.packed(AP, (M, K), side="A", format="sme") if packed else T.load(A)
    b = T.packed(BP, (K, N), side="B", format="sme") if packed else T.load(B)
    value = T.library.matmul((a, b), C, acc=T.load(D), swap=False, BK=3, packing="none")
    T.store(C, T.max(value + T.load(Bias), 0))

  return kernel


def _lower(dtype, k, packed, vl=64, svl=64):
  return (
    CPUBackend()
    .lower(
      _template(dtype, k, packed).ir,
      (),
      target="SME",
      options={"_vector_state": VectorState(vl, svl), "cpu_matmul_pipeline": "off"},
    )
    .source
  )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", (5, 9))
@pytest.mark.parametrize("packed", (False, True))
def test_sme_const_k_split_leaf_is_the_actual_panel(dtype, k, packed):
  source = _lower(dtype, k, packed)
  extents = re.findall(r"const vecops::nint_t dsl_c_sme_extent_k_\d+=\(([^;]+)\);", source)
  assert extents
  assert all("panel_k" in value for value in extents), extents
  assert "epilogue exactly once" in source
  assert "block_k" in source
  assert_no_vec_mask_arrays(source)


def _wrapper(source, dtype, k):
  acc = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  lines = [
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    'extern "C" void split_k(void* a,void* b,void* ap,void* bp,void* d,void* c,void* bias,vecops::nint_t psa,vecops::nint_t psb) {',
  ]
  for name, pointer, element, shape, stride in (
    (
      "A",
      "a",
      dtype,
      f"vecops::meta::cint<5>,vecops::meta::cint<{k}>",
      f"vecops::meta::cint<{k}>,vecops::meta::cint<1>",
    ),
    (
      "B",
      "b",
      dtype,
      f"vecops::meta::cint<{k}>,vecops::meta::cint<19>",
      f"vecops::meta::cint<1>,vecops::meta::cint<{k}>",
    ),
    ("AP", "ap", dtype, "vecops::meta::Any{psa}", "vecops::meta::cint<1>"),
    ("BP", "bp", dtype, "vecops::meta::Any{psb}", "vecops::meta::cint<1>"),
    ("D", "d", acc, "vecops::meta::cint<5>,vecops::meta::cint<19>", "vecops::meta::Any{0},vecops::meta::Any{1}"),
    ("C", "c", acc, "vecops::meta::cint<5>,vecops::meta::cint<19>", "vecops::meta::cint<19>,vecops::meta::cint<1>"),
    (
      "Bias",
      "bias",
      acc,
      "vecops::meta::cint<1>,vecops::meta::cint<19>",
      "vecops::meta::cint<19>,vecops::meta::cint<1>",
    ),
  ):
    lines.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[element]}*>({pointer}),vecops::tensor::make_shape({shape}),vecops::tensor::make_strides({stride}));"
    )
  lines.extend(("__kernel__(A,B,AP,BP,D,C,Bias);", "}"))
  return "\n".join(lines)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("k", (5, 9))
def test_sme_const_split_k_actual_arm_codegen(dtype, packed, k, tmp_path):
  clang, sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG"), os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for real ARM codegen checks")
  root = Path(__file__).resolve().parents[2]
  cpp, assembly = tmp_path / "split_k.cpp", tmp_path / "split_k.s"
  cpp.write_text(_wrapper(_lower(dtype, k, packed), dtype, k), encoding="utf-8")
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    "-std=c++20",
    "-O3",
    "-S",
    "-DNDEBUG",
    "-DVECOPS_TARGET_SME_FA64=1",
    "-DVECOPS_TARGET_SME_F64F64=1",
    "-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS=512",
    "-DVECOPS_KERNEL_SVE_BYTES=64",
    "-DVECOPS_KERNEL_SME_BYTES=64",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
    str(cpp),
    "-o",
    str(assembly),
  ]
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
  assert result.returncode == 0, result.stderr[-6000:]
  text = assembly.read_text(encoding="utf-8")
  assert re.search(r"^\s*(?:smopa|umopa|fmopa|bfmopa)\b", text, re.MULTILINE)
  assert re.search(r"^\s*smstart\b", text, re.MULTILINE)


def _sme_available():
  return (
    os.uname().machine.lower() in {"aarch64", "arm64"}
    and "sme" in Path("/proc/cpuinfo").read_text(encoding="utf-8").split()
  )


@pytest.mark.skipif(not _sme_available(), reason="actual SME target required")
@pytest.mark.parametrize("dtype", DTYPES)
def test_sme_const_split_k_arm_native_raw_packed_prior_and_epilogue(dtype, tmp_path):
  import numpy as np
  import vecops

  if dtype == "bfloat16":
    pytest.importorskip("ml_dtypes")
  state = get_vector_state()
  assert state.sme_bytes is not None
  acc = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  kp, panel = max(1, 4 // np.dtype(dtype).itemsize), 2 * state.sme_bytes // np.dtype(acc).itemsize
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  templates = [(k, packed, _template(dtype, k, packed)) for k in (5, 9) for packed in (False, True)]
  # One collection/batch: no separate cold SDK build per Const shape.
  compiled = {(k, packed): T.compile(template, compiler=compiler) for k, packed, template in templates}

  def pack(array):
    s, k = array.shape
    result = np.zeros(((s + panel - 1) // panel, (k + kp - 1) // kp, panel, kp), dtype=dtype)
    for row in range(s):
      for kk in range(k):
        result[row // panel, kk // kp, row % panel, kk % kp] = array[row, kk]
    return result.reshape(-1)

  rng = np.random.default_rng(63)
  cases = []
  for (k, packed), kernel in compiled.items():
    lo = 0 if dtype == "uint8" else -3
    a = rng.integers(lo, 4, size=(5, k)).astype(dtype)
    b = np.asfortranarray(rng.integers(lo, 4, size=(k, 19)).astype(dtype))
    ap, bp = pack(a), pack(b.T)
    bias = np.full((1, 19), -3, dtype=acc)
    cases.append((k, packed, kernel, a, b, ap, bp, bias))
  with vecops.collect_compile_requests() as collector:
    for _, _, kernel, a, b, ap, bp, bias in cases:
      d, c = np.zeros((5, 19), dtype=acc), np.empty((5, 19), dtype=acc)
      kernel.operator.native(a, b, ap, bp, d, c, bias, **kernel._core.specialization_values)
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  for k, packed, kernel, a, b, ap, bp, bias in cases:
    for broadcast in (False, True):
      prior = rng.integers(-4, 5, size=(1 if broadcast else 5, 19)).astype(acc)
      d = np.broadcast_to(prior, (5, 19)) if broadcast else prior
      c = np.empty((5, 19), dtype=acc)
      expected = np.maximum(a.astype(acc) @ b.astype(acc) + d + bias, 0)
      kernel.operator.native(a, b, ap, bp, d, c, bias, **kernel._core.specialization_values)
      np.testing.assert_array_equal(c, expected, err_msg=f"{dtype}/K={k}/packed={packed}/broadcast={broadcast}")
