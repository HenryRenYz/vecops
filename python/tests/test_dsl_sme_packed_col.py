# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""A single packed B column uses SME spatial lanes, not scalar K MACs.

The source/cross gates check the narrow automatic route without changing raw
dot's no-auto-packing contract. Native uses the target's public automatic
packing ABI; x86 checks the same strict DAG/aliases but does not execute SME.
"""

import platform

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES

_COLUMN_GATE = "/* single-packed B column uses ZA spatial lanes */"


def _template(
  dtype="bfloat16",
  *,
  shape=None,
  packing="B",
  format_="sme",
  transformed=False,
  level="dot",
  implementation="auto",
  epilogue=True,
):
  M, N, K = tuple(T.Const(v) for v in shape) if shape else tuple(T.dynamic(lower=0) for _ in range(3))
  PS = T.dynamic(lower=0)
  A0, A1, D0, D1, C0, C1 = (T.dynamic() for _ in range(6))
  memory = "float32" if transformed else dtype

  @T.template
  def kernel(
    A: T.Tensor((M, K), (A0, A1), memory),
    B: T.Tensor((K, N), (1, K), dtype),
    BP: T.Tensor((PS,), dtype=dtype),
    D: T.Tensor((M, N), (D0, D1), "float32"),
    C: T.Tensor((M, N), (C0, C1), "float32", access="output"),
    Bias: T.Tensor((1, N), dtype="float32"),
  ):
    a = T.cast(T.load(A) * 0.5 + 0.125, dtype) if transformed else T.load(A)
    if packing == "AB":
      a = T.pack(a, side="A", format=format_)
    b = T.packed(BP, (K, N), side="B", format=format_) if packing != "none" else T.load(B)
    value = (
      T.library.matmul((a, b), C, acc=T.load(D), packing="none", swap=False, BK=3, implementation=implementation)
      if level == "blocked"
      else T.library.matmul((a, b), C, acc=T.load(D), packing="none", swap=False, implementation=implementation)
      if level == "matmul"
      else T.dot((a, b), C, acc=T.load(D), implementation=implementation)
    )
    T.store(C, T.max((value + T.load(Bias)) * 0.5 + 0.125, 0) if epilogue else value)

  return kernel


def _lower(*, vl=64, svl=64, **kwargs):
  return (
    CPUBackend()
    .lower(
      _template(**kwargs).ir,
      (),
      target="SME",
      options={"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always", "_vector_state": VectorState(vl, svl)},
    )
    .source
  )


def _wrapper(source, dtype, *, shape=None, transformed=False):
  memory = "float32" if transformed else dtype
  M, N, K = (
    (f"vecops::meta::cint<{v}>" for v in shape)
    if shape
    else (
      "vecops::meta::Any{m}",
      "vecops::meta::Any{n}",
      "vecops::meta::Any{k}",
    )
  )
  lines = [
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    (
      'extern "C" void packed_col(void* ap,void* bp,void* pp,void* dp,void* cp,void* biasp,'
      "vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t ps,"
      "vecops::nint_t a0,vecops::nint_t a1,vecops::nint_t d0,vecops::nint_t d1,vecops::nint_t c0,vecops::nint_t c1) {"
    ),
  ]
  for name, pointer, dtype_, shape_, strides in (
    ("A", "ap", memory, f"{M},{K}", "vecops::meta::Any{a0},vecops::meta::Any{a1}"),
    ("B", "bp", dtype, f"{K},{N}", f"vecops::meta::cint<1>,{K}"),
    ("BP", "pp", dtype, "vecops::meta::Any{ps}", "vecops::meta::cint<1>"),
    ("D", "dp", "float32", f"{M},{N}", "vecops::meta::Any{d0},vecops::meta::Any{d1}"),
    ("C", "cp", "float32", f"{M},{N}", "vecops::meta::Any{c0},vecops::meta::Any{c1}"),
    ("Bias", "biasp", "float32", f"vecops::meta::cint<1>,{N}", f"{N},vecops::meta::cint<1>"),
  ):
    lines.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[dtype_]}*>({pointer}),"
      f"vecops::tensor::make_shape({shape_}),vecops::tensor::make_strides({strides}));"
    )
  return "\n".join((*lines, "__kernel__(A,B,BP,D,C,Bias);", "}"))


def _pack_template(dtype):
  K, N, PS = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(B: T.Tensor((K, N), (1, K), dtype), BP: T.Tensor((PS,), dtype=dtype, access="output")):
    T.pack(B, BP, side="B", format="auto")

  return kernel


def _column(shape, dtype, *, step=1, negative=False):
  m, n = shape
  assert n == 1
  backing = np.empty(max(1, m * abs(step)), dtype=dtype)
  value = np.ndarray(
    shape, dtype=dtype, buffer=backing, strides=(step * np.dtype(dtype).itemsize, np.dtype(dtype).itemsize)
  )
  return value[::-1] if negative else value


def _packed_count(*, machine=None, vector_state=None):
  """Maximum fixture K=257, using the current target's public pack ABI."""
  machine = (machine or platform.machine()).lower()
  if machine in {"aarch64", "arm64"}:
    svl = (vector_state or get_vector_state()).sme_bytes
    if svl is None:
      pytest.skip("single-packed-B SME fixture requires a valid current SME vector length")
    panel, group = 2 * svl // 4, 2
  else:
    panel, group = 16, 32
  return panel * ((257 + group - 1) // group) * group


@pytest.mark.parametrize("svl", (16, 32, 64, 256))
def test_packed_column_fixture_capacity_uses_actual_svl(svl):
  count = _packed_count(machine="aarch64", vector_state=VectorState(16, svl))
  assert count == (2 * svl // 4) * 258
  assert count >= (2 * svl // 4) * 257


def test_packed_column_fixture_x86_capacity_uses_amx_geometry():
  assert _packed_count(machine="x86_64", vector_state=VectorState(None, None)) == 16 * 288


def test_packed_column_fixture_without_sme_skips_instead_of_guessing_svl():
  with pytest.raises(pytest.skip.Exception, match="valid current SME vector length"):
    _packed_count(machine="aarch64", vector_state=VectorState(16, None))


@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_native_single_packed_b_column_strict_fusion_acc_alias_and_tails(dtype, tmp_path):
  if dtype == "bfloat16":
    pytest.importorskip("ml_dtypes")
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  options = {"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"}
  constant_shape = (16 if dtype == "float16" else 64, 1, 257)
  packed_count = _packed_count()
  pack = T.compile(_pack_template(dtype), compiler=compiler, pass_options=options)
  kernels = {
    (dynamic, level): T.compile(
      _template(dtype, shape=None if dynamic else constant_shape, format_="auto", transformed=True, level=level),
      compiler=compiler,
      pass_options=options,
    )
    for dynamic in (False, True)
    for level in ("dot", "matmul", "blocked")
  }
  with vecops.collect_compile_requests() as collector:
    pack.operator(
      vecops.TensorMeta((257, 1), strides=(1, 257), dtype=dtype),
      vecops.TensorMeta((packed_count,), dtype=dtype),
      **pack._core.specialization_values,
    )
    for (dynamic, _), compiled in kernels.items():
      m = 64 if dynamic else constant_shape[0]
      compiled.operator(
        vecops.TensorMeta((m, 257), strides=(519, 2), dtype="float32"),
        vecops.TensorMeta((257, 1), strides=(1, 257), dtype=dtype),
        vecops.TensorMeta((packed_count,), dtype=dtype),
        vecops.TensorMeta((m, 1), strides=(2, 1), dtype="float32"),
        vecops.TensorMeta((m, 1), strides=(2, 1), dtype="float32"),
        vecops.TensorMeta((1, 1), dtype="float32"),
        **compiled._core.specialization_values,
      )
  assert len(collector.requests) == 7
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  rng = np.random.default_rng(361)
  for dynamic in (False, True):
    shapes = (
      (constant_shape,) if not dynamic else ((0, 1, 7), (1, 1, 0), (16, 1, 1), (63, 1, 5), (64, 1, 257), (65, 1, 9))
    )
    for m, n, k in shapes:
      # Explicit strides survive n==1/empty NumPy canonicalization and make
      # the unused logical B descriptor obey the declared column layout.
      b = np.ndarray(
        (k, n),
        dtype=dtype,
        buffer=np.empty(max(1, k), dtype=dtype),
        strides=(np.dtype(dtype).itemsize, k * np.dtype(dtype).itemsize),
      )
      b[:] = rng.integers(-3, 4, size=b.shape) / 8
      bp = np.empty(packed_count, dtype=dtype)
      pack.operator.native(b, bp, **pack._core.specialization_values)
      for negative in (False, True):
        a = rng.integers(-4, 5, size=(m, 2 * k + 5)).astype("float32")
        a /= np.float32(16)
        a = a[:, : 2 * k : 2]
        if negative:
          a = a[::-1, ::-1]
        transformed = (a.copy() * np.float32(0.5) + np.float32(0.125)).astype(dtype).astype("float32")
        product = transformed @ b.astype("float32")
        for alias in (False, True):
          for level in ("dot", "matmul", "blocked"):
            d = _column((m, n), "float32", step=2, negative=negative)
            d[:] = rng.integers(-2, 3, size=d.shape) / 8
            c = d if alias else _column((m, n), "float32", step=2, negative=negative)
            bias = np.full((1, n), np.float32(0.125), dtype="float32")
            prior = d.copy()
            expected = np.maximum(((product + prior) + bias) * np.float32(0.5) + np.float32(0.125), 0)
            compiled = kernels[dynamic, level]
            compiled.operator.native(a, b, bp, d, c, bias, **compiled._core.specialization_values)
            np.testing.assert_array_equal(
              c, expected, err_msg=f"{dtype}/{level} {(m, n, k)} negative={negative} alias={alias}"
            )
