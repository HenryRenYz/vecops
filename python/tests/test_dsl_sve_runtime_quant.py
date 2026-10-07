"""Experimental ordinary-SVE quantize/packed-B dot family.

The opt-in family must preserve the user DAG's floating-point contraction
policy, quantize only valid K lanes, and retain normal matmul alias/phase
contracts. Cross assembly is syntax/code-generation evidence, not performance.
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

OPTION = "cpu_sve_runtime_quant"
MARKER = "/* ordinary-SVE runtime-quantized packed-B dot */"


def _template(
  *,
  shape=None,
  level="dot",
  bk=None,
  fma=False,
  seed=True,
  output="float32",
  pipeline="dequant",
  raw_b=False,
  intermediate=None,
):
  M, N, K = tuple(T.Const(v) for v in shape) if shape else tuple(T.dynamic(lower=0) for _ in range(3))
  PS = T.dynamic(lower=0)
  A0, A1, C0, C1, D0, D1 = (T.dynamic() for _ in range(6))

  @T.func
  def transform(A, RQ, Z):
    x, scale, zp = T.load(A), T.load(RQ), T.load(Z, dtype="float32")
    if intermediate is not None:
      return T.cast(T.cast(T.cast(x, intermediate) * T.cast(scale, intermediate), "float32") + zp, "uint8")
    return T.cast(T.fma(x, scale, zp) if fma else x * scale + zp, "uint8")

  @T.func
  def helper(A, BP, B, D, C, RQ, RD, W, Z):
    a = transform(A, RQ, Z)
    b = T.load(B) if raw_b else T.packed(BP, (K, N), side="B", format="sme")
    initial = T.load(D) if seed else None
    result = (
      T.library.matmul((a, b), C, acc=initial, acc_dtype="int32", swap=False, packing="none", BK=bk)
      if level == "library"
      else T.dot((a, b), C, acc=initial, acc_dtype="int32")
    )
    if pipeline == "dequant":
      result = T.cast(result, "float32") * T.load(RD) * T.load(W)
    elif pipeline == "relu":
      result = T.max(result, 0)
    T.store(C, result)

  @T.template
  def kernel(
    A: T.Tensor((M, K), (A0, A1), "float32"),
    BP: T.Tensor((PS,), dtype="int8"),
    B: T.Tensor((K, N), dtype="int8"),
    D: T.Tensor((M, N), (D0, D1), "int32"),
    C: T.Tensor((M, N), (C0, C1), output, access="output"),
    RQ: T.Tensor((M, 1), dtype="float32"),
    RD: T.Tensor((M, 1), dtype="float32"),
    W: T.Tensor((1, N), dtype="float32"),
    Z: T.Tensor((1,), dtype="int32"),
  ):
    helper(A, BP, B, D, C, RQ, RD, W, Z)

  return kernel


def _lower(*, enabled=True, vl=64, svl=64, **kwargs):
  return (
    CPUBackend()
    .lower(
      _template(**kwargs).ir,
      (),
      target="SME",
      options={
        OPTION: enabled,
        "cpu_matmul_pipeline": "off",
        "cpu_matmul_fusion": "always",
        "_vector_state": VectorState(vl, svl),
      },
    )
    .source
  )


def _body(source, marker=MARKER):
  start = source.index("{", source.index(marker) + len(marker))
  depth = 1
  for position in range(start + 1, len(source)):
    depth += (source[position] == "{") - (source[position] == "}")
    if not depth:
      return source[start + 1 : position]
  raise AssertionError(f"unclosed block following {marker}")


def test_runtime_quant_option_default_off_and_strict_bool():
  disabled = _lower(enabled=False)
  assert MARKER not in disabled
  default = CPUBackend().lower(_template().ir, (), target="SME").source
  assert MARKER not in default
  for invalid in (1, 0, "true", None):
    with pytest.raises(ValueError, match=OPTION):
      _lower(enabled=invalid)


@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32), (16, 16), (64, 64)))
def test_runtime_quant_named_vectors_independent_lengths_and_post_quant_tail(vl, svl):
  source = _lower(vl=vl, svl=svl)
  body = _body(source)
  assert "vecops::vec::widening_dot" in body
  assert f"FixedTag<{CPP_TYPES['uint8']}, 4>" in body
  assert "vecops::vec::bitcast" in body and "vecops::vec::get" in body
  assert "runtime_quant_tail_zero" in body
  assert (
    max(
      body.rfind("vecops::vec::convert", 0, body.index("runtime_quant_tail_zero")),
      body.rfind("vecops::vec::load_convert", 0, body.index("runtime_quant_tail_zero")),
    )
    >= 0
  )
  assert "vecops::vec::mul" in body and "vecops::vec::add" in body
  assert "vecops::vec::fmadd" not in body
  assert "sme::" not in body and "with_resources" not in body
  assert "pack_panel" in body
  assert "store_convert" in body
  assert_no_vec_mask_arrays(source)


def test_runtime_quant_explicit_fma_is_not_decontracted():
  body = _body(_lower(fma=True))
  assert "vecops::vec::fmadd" in body


@pytest.mark.parametrize("intermediate", ("float64", "float16", "bfloat16"))
def test_runtime_quant_intermediate_dtype_boundary_retains_general_path(intermediate):
  source = _lower(intermediate=intermediate)
  assert MARKER not in source
  assert "sme::mopa" in source


def test_runtime_quant_raw_b_and_streaming_domain_retain_fallback():
  assert MARKER not in _lower(raw_b=True)
  emitter = SimpleNamespace(active_matmul_plan=SimpleNamespace(sve_runtime_quant=True), hardware_scope=("sme", "scope"))
  assert MatmulEmitter.sve_runtime_quant(emitter, None, None) is False


def test_runtime_quant_split_k_retains_prior_and_final_phase_guards():
  source = _lower(level="library", bk=3)
  body = _body(source)
  assert "block_k" in body
  assert "epilogue exactly once" in body
  assert "runtime_quant_prior" in body
  assert "pack_offset" in body  # Original packed geometry/offset, not a repacked K window.


def _cross_wrapper(source):
  definitions = [
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    (
      'extern "C" void runtime_quant(void* ap,void* bp,void* rawb,void* dp,void* cp,void* rq,void* rd,void* wp,void* zp, '
      "vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t ps,vecops::nint_t a0,vecops::nint_t a1,"
      "vecops::nint_t d0,vecops::nint_t d1,vecops::nint_t c0,vecops::nint_t c1) {"
    ),
  ]
  for name, pointer, dtype, shape, stride in (
    ("A", "ap", "float32", "vecops::meta::Any{m},vecops::meta::Any{k}", "vecops::meta::Any{a0},vecops::meta::Any{a1}"),
    ("BP", "bp", "int8", "vecops::meta::Any{ps}", "vecops::meta::cint<1>"),
    ("B", "rawb", "int8", "vecops::meta::Any{k},vecops::meta::Any{n}", "vecops::meta::Any{n},vecops::meta::cint<1>"),
    ("D", "dp", "int32", "vecops::meta::Any{m},vecops::meta::Any{n}", "vecops::meta::Any{d0},vecops::meta::Any{d1}"),
    ("C", "cp", "float32", "vecops::meta::Any{m},vecops::meta::Any{n}", "vecops::meta::Any{c0},vecops::meta::Any{c1}"),
    (
      "RQ",
      "rq",
      "float32",
      "vecops::meta::Any{m},vecops::meta::cint<1>",
      "vecops::meta::cint<1>,vecops::meta::cint<1>",
    ),
    (
      "RD",
      "rd",
      "float32",
      "vecops::meta::Any{m},vecops::meta::cint<1>",
      "vecops::meta::cint<1>,vecops::meta::cint<1>",
    ),
    ("W", "wp", "float32", "vecops::meta::cint<1>,vecops::meta::Any{n}", "vecops::meta::Any{n},vecops::meta::cint<1>"),
    ("Z", "zp", "int32", "vecops::meta::cint<1>", "vecops::meta::cint<1>"),
  ):
    definitions.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[dtype]}*>({pointer}),"
      f"vecops::tensor::make_shape({shape}),vecops::tensor::make_strides({stride}));"
    )
  definitions.extend(("__kernel__(A,BP,B,D,C,RQ,RD,W,Z);", "}"))
  return "\n".join(definitions)


@pytest.mark.parametrize(
  "enabled,fma,vl,svl,level,bk,shape",
  tuple(
    (True, fma, vl, svl, "dot", None, None)
    for fma in (False, True)
    for vl, svl in ((16, 64), (32, 16), (64, 32), (64, 64))
  )
  + (
    (True, False, 64, 16, "library", 3, None),
    (True, False, 16, 64, "library", 3, (5, 19, 5)),
    (True, False, 64, 64, "library", 3, (5, 19, 9)),
    (False, False, 64, 64, "dot", None, None),
    (False, True, 64, 64, "dot", None, None),
  ),
)
def test_runtime_quant_actual_arm_instructions(enabled, vl, svl, fma, level, bk, shape, tmp_path):
  clang, sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG"), os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for real ARM codegen checks")
  root = Path(__file__).resolve().parents[2]
  cpp, assembly = tmp_path / "runtime_quant.cpp", tmp_path / "runtime_quant.s"
  cpp.write_text(
    _cross_wrapper(_lower(enabled=enabled, vl=vl, svl=svl, fma=fma, level=level, bk=bk, shape=shape)),
    encoding="utf-8",
  )
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    "-std=c++20",
    "-O3",
    "-ffp-contract=off",  # Matches the generated-DSL target, including vendor compilers.
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
  assert re.search(r"^\s*(?:usdot|sudot)\b" if enabled else r"^\s*usmopa\b", text, re.MULTILINE)
  # A Dynamic whole-problem budget has a legal generic SME fallback in the
  # same DSO; only a proven accepted Const dot may eliminate it entirely.
  if enabled and level == "dot" and shape is not None:
    assert not re.search(r"^\s*(?:smstart|[a-z]*mopa)\b", text, re.MULTILINE)
  contracted = bool(re.search(r"^\s*(?:fmla|fmad|fmadd)\b", text, re.MULTILINE))
  assert contracted == fma
  assert not re.search(r"^\s*bl\b[^\n]*(?:WideningDot|widening_dot)", text, re.MULTILINE)


def _sme_available():
  if os.uname().machine.lower() not in {"aarch64", "arm64"}:
    return False
  try:
    return "sme" in Path("/proc/cpuinfo").read_text(encoding="utf-8").split()
  except OSError:
    return False


@pytest.mark.skipif(not _sme_available(), reason="actual ordinary-SVE and SME packing required")
@pytest.mark.parametrize(
  "level,bk,output,pipeline",
  (("dot", None, "float32", "dequant"), ("library", 3, "float32", "dequant"), ("dot", None, "int32", "relu")),
)
def test_runtime_quant_arm_native_catalog_shapes_tails_and_strides(level, bk, output, pipeline, tmp_path):
  import numpy as np

  state = get_vector_state()
  assert state.sve_bytes is not None and state.sme_bytes is not None
  panel = state.sme_bytes // 2
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(
    _template(level=level, bk=bk, output=output, pipeline=pipeline),
    compiler=compiler,
    pass_options={OPTION: True, "cpu_matmul_fusion": "always"},
  )
  rng = np.random.default_rng(375)
  for m, n, k in (
    (1, 256, 256),
    (5, 19, 67),
    (64, 256, 256),
    (1, panel + 3, 0),
    (3, panel + 1, 1),
    (2, 3, 3),
    (1, 1, 4),
    (3, 7, 5),
  ):
    a = rng.integers(-8, 80, size=(m, k)).astype("float32") / 4
    b = rng.integers(-4, 5, size=(k, n), dtype="int8")
    packed = np.zeros(((n + panel - 1) // panel, (k + 3) // 4, panel, 4), dtype="int8")
    for kk in range(k):
      for col in range(n):
        packed[col // panel, kk // 4, col % panel, kk % 4] = b[kk, col]
    bp = packed.reshape(-1)
    rq = (2 + 2 * (np.arange(m) % 2)).astype("float32").reshape(m, 1)
    rd = (1 / rq).astype("float32")
    w = (0.03125 + (np.arange(n) % 5) * 0.0078125).astype("float32").reshape(1, n)
    z = np.array([7], dtype="int32")
    correction = (-7 * b.astype("int32").sum(axis=0)).astype("int32").reshape(1, n)
    for layout in ("row", "column", "strided", "negative"):
      strides = (n, 1) if layout == "row" else (1, m) if layout == "column" else (2 * n + 3, 2)
      backing = np.empty(max(1, (m - 1) * strides[0] + (n - 1) * strides[1] + 1), dtype=output)
      c = np.ndarray(
        (m, n), dtype=output, buffer=backing, strides=tuple(s * np.dtype(output).itemsize for s in strides)
      )
      if layout == "negative":
        c = c[::-1, ::-1]
      for reverse_a in (False, True):
        aa = a[:, ::-1] if reverse_a else a
        q = np.clip(np.trunc(np.float32(np.float32(aa * rq) + np.float32(z[0]))), 0, 255).astype("uint8")
        d = np.broadcast_to(correction, (m, n))
        expected = q.astype("int32") @ b.astype("int32") + d
        expected = (
          np.float32(np.float32(expected.astype("float32") * rd) * w)
          if pipeline == "dequant"
          else np.maximum(expected, 0)
        )
        compiled.operator.native(aa, bp, b, d, c, rq, rd, w, z, **compiled._core.specialization_values)
        np.testing.assert_array_equal(c, expected, err_msg=f"{level}/{bk}, {(m, n, k)}, {layout}/reverseA={reverse_a}")


@pytest.mark.skipif(not _sme_available(), reason="actual ordinary-SVE and SME packing required")
@pytest.mark.parametrize("fma", (False, True))
def test_runtime_quant_arm_native_conversion_policy_and_alias_prior(fma, tmp_path):
  import numpy as np

  state = get_vector_state()
  assert state.sme_bytes is not None
  panel = state.sme_bytes // 2
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(
    _template(fma=fma, output="int32", pipeline="plain"), compiler=compiler, pass_options={OPTION: True}
  )
  # This first value separates strict rounded mul+add (q=1) from FMA (q=0).
  # Conversion of exceptional values is specified by Vector convert, never
  # copied from the native scalar leaf's float-to-int32 narrowing branch.
  a = np.array([[-1.0000001192092896, np.nan, np.inf, -np.inf, 0, 126, 127, 128, -4]], dtype="float32")
  a = np.concatenate((a, a[:, ::-1]), axis=0)
  m, k, n = 2, a.shape[1], panel + 1
  rq = np.full((m, 1), 1.9999998807907104, dtype="float32")
  rd, w, z = np.ones((m, 1), dtype="float32"), np.ones((1, n), dtype="float32"), np.array([3], dtype="int32")
  b = np.tile((np.arange(n) % 5 - 2).astype("int8"), (k, 1))
  packed = np.zeros(((n + panel - 1) // panel, (k + 3) // 4, panel, 4), dtype="int8")
  for kk in range(k):
    for col in range(n):
      packed[col // panel, kk // 4, col % panel, kk % 4] = b[kk, col]
  with np.errstate(invalid="ignore", over="ignore"):
    affine = (
      np.float32(a.astype("float64") * rq.astype("float64") + float(z[0]))
      if fma
      else np.float32(np.float32(a * rq) + np.float32(z[0]))
    )
    q = np.clip(np.trunc(np.nan_to_num(affine, nan=0, posinf=255, neginf=0)), 0, 255).astype("uint8")
  assert q[0, 0] == (0 if fma else 1)
  expected_product = q.astype("int32") @ b.astype("int32")
  for broadcast in (False, True):
    c = np.arange(m * n, dtype="int32").reshape(m, n).copy()
    d = np.broadcast_to(c[:1], c.shape) if broadcast else c
    expected = expected_product + d.copy()
    compiled.operator.native(a, packed.reshape(-1), b, d, c, rq, rd, w, z, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, expected, err_msg=f"fma={fma}/broadcast_alias={broadcast}")
