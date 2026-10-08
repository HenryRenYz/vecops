# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ordinary-SVE MMLA consumes SME-packed input without owning ZA.

Cross compilation checks real ISA routes; target execution remains required
for correctness/performance conclusions. Register values are never arrays.
"""

import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import vecops.language as T
from dsl_test_utils import arm_cross_cxx_include_flags, assert_no_vec_mask_arrays, make_compiler
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES
from vecops.language._compiler.backends.cpu.cpp.matmul import MatmulEmitter


def _template(
  dtype="bfloat16", output=None, *, shape=None, seed=True, pipeline="bias_relu", scope=False, bm=None, bn=None, bk=None
):
  output = output or ("float32" if dtype in {"bfloat16", "float16"} else "int32")
  acc = "float32" if dtype in {"bfloat16", "float16"} else "int32"
  M, N, K = tuple(T.Const(v) for v in shape) if shape else tuple(T.dynamic(lower=0) for _ in range(3))
  PSA, PSB = T.dynamic(lower=0), T.dynamic(lower=0)
  C0, C1, D0, D1 = (T.dynamic() for _ in range(4))
  blocked = any(v is not None for v in (bm, bn, bk))

  @T.func
  def compute(AP, BP, D, C, Bias):
    a = T.packed(AP, (M, K), side="A", format="sme")
    b = T.packed(BP, (K, N), side="B", format="sme")
    initial = T.load(D) if seed else None
    result = (
      T.library.matmul((a, b), C, acc=initial, swap=False, BM=bm, BN=bn, BK=bk)
      if blocked
      else T.dot((a, b), C, acc=initial)
    )
    if pipeline == "bias_relu":
      T.store(C, T.max(result + T.load(Bias), 0))
    elif pipeline == "affine":
      T.store(C, result * 0.5 + 1)
    else:
      T.store(C, result)

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype=dtype),
    AP: T.Tensor((PSA,), dtype=dtype),
    BP: T.Tensor((PSB,), dtype=dtype),
    D: T.Tensor((M, N), (D0, D1), acc),
    C: T.Tensor((M, N), (C0, C1), output, access="output"),
    Bias: T.Tensor((1, N), dtype=acc),
  ):
    if scope:
      with T.sme.region():
        compute(AP, BP, D, C, Bias)
    else:
      compute(AP, BP, D, C, Bias)

  return kernel


def _lower(*, vl=None, svl=None, **kwargs):
  return (
    CPUBackend()
    .lower(
      _template(**kwargs).ir,
      (),
      target="SME",
      options={"cpu_matmul_pipeline": "off", "_vector_state": VectorState(vl, svl)},
    )
    .source
  )


def _body(source, marker):
  start = source.index("{", source.index(marker) + len(marker))
  depth = 1
  for position in range(start + 1, len(source)):
    depth += (source[position] == "{") - (source[position] == "}")
    if depth == 0:
      return source[start + 1 : position]
  raise AssertionError(f"unclosed block following {marker}")


@pytest.mark.parametrize("dtype", ("bfloat16", "int8", "uint8"))
def test_mmla_named_registers_paired_groups_and_epilogue(dtype):
  source = _lower(dtype=dtype)
  body = _body(source, "/* ordinary-SVE PackedDot MMLA */")
  assert {int(v) for v in re.findall(r"/\* MMLA (\d+)-row family \*/", body)} == {2, 4, 8}
  for rows in (2, 4, 8):
    family = _body(body, f"/* MMLA {rows}-row family */")
    complete = _body(family, "/* MMLA two complete K groups */")
    tail = _body(family, "/* MMLA odd group: no second-group read */")
    assert complete.count("vecops::vec::load(") == 4
    assert tail.count("vecops::vec::load(") == 2
    assert "+1)*" not in tail
    assert complete.count("sve::mmla(") == (2 if rows == 4 else 1)
    assert family.count("sve::row_pair<") == rows
    assert "vecops::vec::add" in family and "vecops::vec::max" in family
    assert "vecops::vec::store_convert" in family
  assert "sme::mopa<" in source  # General matrices still have a ZA fallback.
  assert "vecops::vec::widening_dot" in source  # Unsupported geometry remains supported.
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("vl,svl,rows", ((16, 16, (2,)), (32, 32, (2, 4)), (64, 64, (2, 4, 8))))
def test_mmla_capacity_is_separate_from_vl_svl_equality(vl, svl, rows):
  source = _lower(vl=vl, svl=svl)
  body = _body(source, "/* ordinary-SVE PackedDot MMLA */")
  assert tuple(sorted({int(v) for v in re.findall(r"/\* MMLA (\d+)-row family \*/", body)})) == rows
  gate = source[: source.index("/* ordinary-SVE PackedDot MMLA */")].splitlines()[-1]
  assert f">={min(64, vl)}" in gate
  assert "==2*(vecops_dsl::pack_panel" in gate


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (32, 64)))
def test_mmla_unequal_vector_domains_keep_widening_fallback(vl, svl):
  source = _lower(vl=vl, svl=svl)
  assert "sve::mmla(" not in source
  assert "vecops::vec::widening_dot" in source


def test_mmla_never_enters_an_explicit_streaming_scope():
  # The current frontend rejects matrix operators in a raw hardware region.
  # Independently verify the emitter domain guard for internal/shared owners.
  with pytest.raises(TypeError, match="hardware region supports only"):
    _lower(scope=True)
  operand = SimpleNamespace(type=SimpleNamespace(dtype="bfloat16"))
  emitter = SimpleNamespace(hardware_scope=("sme", "scope"), matrix_operands=lambda _: (operand, operand))
  assert MatmulEmitter.sve_packed_mmla(emitter, None, None) is False


@pytest.mark.parametrize("dtype", ("bfloat16", "int8", "uint8"))
def test_mmla_partial_cache_panels_guard_k_and_spatial_origins(dtype):
  source = _lower(dtype=dtype, bk=3)
  gates = [line for line in source.splitlines() if "/* ordinary-SVE PackedDot MMLA */" in line]
  gate = next(line for line in gates if "block_k" in line)
  kp = 2 if dtype == "bfloat16" else 4
  assert f"%{kp}==0" in gate
  assert "block_k" in gate
  assert "block_m" in gate or "block_n" in gate
  # A partial K window may consume padded elements only at the original end.
  assert f")%{kp}==0 ||" in gate
  assert "epilogue exactly once" in source


def test_mmla_identity_8x2_has_contiguous_store_without_scalar_staging():
  source = _lower(shape=(8, 2, 5), seed=False, pipeline="plain", vl=64, svl=64)
  direct = _body(source, "/* contiguous 8x2 MMLA output */")
  assert "vecops::vec::store(" in direct and "opt::first(16)" in direct
  assert "store_convert" not in direct
  assert "Buffer<" not in _body(source, "/* ordinary-SVE PackedDot MMLA */")


def _cross_wrapper(source, dtype, output):
  acc = "float32" if dtype == "bfloat16" else "int32"
  definitions = [
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    (
      'extern "C" void packed_mmla(void* raw_a,void* ap,void* bp,void* dp,void* cp,void* biasp, '
      "vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t psa,vecops::nint_t psb,"
      "vecops::nint_t d0,vecops::nint_t d1,vecops::nint_t c0,vecops::nint_t c1) {"
    ),
  ]
  for name, pointer, element, shape, stride in (
    ("A", "raw_a", dtype, "vecops::meta::Any{m},vecops::meta::Any{k}", "vecops::meta::Any{k},vecops::meta::cint<1>"),
    ("AP", "ap", dtype, "vecops::meta::Any{psa}", "vecops::meta::cint<1>"),
    ("BP", "bp", dtype, "vecops::meta::Any{psb}", "vecops::meta::cint<1>"),
    ("D", "dp", acc, "vecops::meta::Any{m},vecops::meta::Any{n}", "vecops::meta::Any{d0},vecops::meta::Any{d1}"),
    ("C", "cp", output, "vecops::meta::Any{m},vecops::meta::Any{n}", "vecops::meta::Any{c0},vecops::meta::Any{c1}"),
    ("Bias", "biasp", acc, "vecops::meta::cint<1>,vecops::meta::Any{n}", "vecops::meta::Any{n},vecops::meta::cint<1>"),
  ):
    definitions.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[element]}*>({pointer}),"
      f"vecops::tensor::make_shape({shape}),vecops::tensor::make_strides({stride}));"
    )
  definitions.extend(("__kernel__(A,AP,BP,D,C,Bias);", "}"))
  return "\n".join(definitions)


@pytest.mark.parametrize(
  "dtype,output,mnemonic", (("bfloat16", "float32", "bfmmla"), ("int8", "int32", "smmla"), ("uint8", "int32", "ummla"))
)
@pytest.mark.parametrize(
  "vl,svl,shape",
  (
    (16, 16, (2, 2, 5)),
    (16, 16, (2, 8, 5)),
    (32, 32, (4, 4, 5)),
    (32, 32, (2, 8, 5)),
    (64, 64, (2, 8, 5)),
    (64, 64, (8, 2, 5)),
    (16, 64, (2, 8, 5)),
    (64, 16, (2, 8, 5)),
  ),
)
def test_mmla_arm_cross_actual_instructions(tmp_path, dtype, output, mnemonic, vl, svl, shape):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for real ARM codegen checks")
  root = Path(__file__).resolve().parents[2]
  # Check real minimal capacities as well as the 2x8 insufficient/equal-VL
  # case: it must use BFDOT/SDOT, not return a truncated MMLA accumulator.
  source = _lower(dtype=dtype, output=output, shape=shape, vl=vl, svl=svl)
  cpp = tmp_path / "mmla.cpp"
  cpp.write_text(_cross_wrapper(source, dtype, output), encoding="utf-8")
  assembly = cpp.with_suffix(".s")
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
    f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={svl * 8}",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
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
  present = bool(re.search(rf"^\s*{mnemonic}\b", text, re.MULTILINE))
  required = 64 if 8 in shape[:2] else 32 if 4 in shape[:2] else 16
  assert present == (vl == svl and vl >= required)
  assert not re.search(r"^\s*smstart\b", text, re.MULTILINE)
  if present:
    assert re.search(r"^\s*zip1\b", text, re.MULTILINE)
    assert re.search(r"^\s*uzp[12]\b", text, re.MULTILINE)
    assert not re.search(r"^\s*bl\b[^\n]*(?:join_k_groups|broadcast_segment|row_pair|4mmla)", text, re.MULTILINE)


def _sme_available():
  if os.uname().machine.lower() not in {"aarch64", "arm64"}:
    return False
  try:
    return "sme" in Path("/proc/cpuinfo").read_text(encoding="utf-8").split()
  except OSError:
    return False


@pytest.mark.skipif(not _sme_available(), reason="target ordinary-SVE/SME packing geometry required")
@pytest.mark.parametrize(
  "dtype,output", (("bfloat16", "float32"), ("bfloat16", "bfloat16"), ("int8", "int32"), ("uint8", "int32"))
)
def test_mmla_arm_native_shapes_tails_fusion_and_strides(dtype, output, tmp_path):
  import numpy as np

  if dtype == "bfloat16":
    pytest.importorskip("ml_dtypes")
  state = get_vector_state()
  assert state.sme_bytes is not None
  acc = "float32" if dtype == "bfloat16" else "int32"
  kp, panel = (2 if dtype == "bfloat16" else 4), state.sme_bytes // 2
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(_template(dtype, output), compiler=compiler)
  rng = np.random.default_rng(194)

  def pack(array):
    spatial, k = array.shape
    packed = np.zeros(((spatial + panel - 1) // panel, (k + kp - 1) // kp, panel, kp), dtype=dtype)
    for row in range(spatial):
      for col in range(k):
        packed[row // panel, col // kp, row % panel, col % kp] = array[row, col]
    return packed.reshape(-1)

  def output_array(m, n, layout):
    itemsize = np.dtype(output).itemsize
    strides = (n, 1) if layout == "row" else (1, m) if layout == "column" else (2 * n + 3, 2)
    backing = np.empty(max(1, (m - 1) * strides[0] + (n - 1) * strides[1] + 1), dtype=output)
    positive = np.ndarray((m, n), dtype=output, buffer=backing, strides=tuple(s * itemsize for s in strides))
    return positive[::-1, ::-1] if layout == "negative" else positive

  maximum = 1025 if dtype == "bfloat16" else 513
  for m, n in ((2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (8, 2)):
    for k in (0, 1, 2, 3, 4, 5, 257, maximum, maximum + 1):
      lo = 0 if dtype == "uint8" else -3
      a, b = rng.integers(lo, 4, size=(m, k)), rng.integers(lo, 4, size=(k, n))
      a = (a / 8 if dtype == "bfloat16" else a).astype(dtype)
      b = (b / 8 if dtype == "bfloat16" else b).astype(dtype)
      ap, bp = pack(a), pack(b.T)
      bias = rng.integers(-2, 3, size=(1, n)).astype(acc)
      for layout in ("row", "column", "strided", "negative"):
        c = output_array(m, n, layout)
        for prior_mode in ("full", "broadcast", "alias", "broadcast_alias"):
          if prior_mode in {"alias", "broadcast_alias"} and output != acc:
            continue
          if prior_mode in {"alias", "broadcast_alias"}:
            c[:] = rng.integers(-3, 4, size=(m, n))
            d = np.broadcast_to(c[:1], (m, n)) if prior_mode == "broadcast_alias" else c
          elif prior_mode == "broadcast":
            d = np.broadcast_to(rng.integers(-3, 4, size=(1, n)).astype(acc), (m, n))
          else:
            d = rng.integers(-3, 4, size=(m, n)).astype(acc)
          expected = np.maximum(a.astype(acc) @ b.astype(acc) + d.copy() + bias, 0).astype(output)
          kernel.operator.native(a, ap, bp, d, c, bias, **kernel._core.specialization_values)
          np.testing.assert_array_equal(c, expected, err_msg=f"{dtype}/{output}, {(m, n, k)}, {layout}/{prior_mode}")
