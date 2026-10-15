# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Experimental ordinary-SVE quantize/packed-B dot family.

The opt-in family must preserve the user DAG's floating-point contraction
policy, quantize only valid K lanes, and retain normal matmul alias/phase
contracts. Cross assembly is syntax/code-generation evidence, not performance.
"""

import os
from pathlib import Path

import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES

OPTION = "cpu_runtime_quant_int8"
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
