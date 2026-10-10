# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Whole BF16 packed-AB four-M-block columns reuse the native ZA family.

This is deliberately not a general PackedDot replacement: windows, ambiguous
packing roles, other dtypes, explicit vector requests and the existing tiny
ladder keep their previous dispatch. Runtime capacity follows SVL, not VL.
"""

import platform
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES
from vecops.language._compiler.ir.traversal import rewrite_expressions

_GATE = "/* packed AB four-M-block column uses ZA */"
_OPTIONS = {"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"}


def _template(*, shape=None, dtype="bfloat16", level="dot", implementation="auto"):
  M, N, K = tuple(T.Const(v) for v in shape) if shape else tuple(T.dynamic(lower=0) for _ in range(3))
  PSA, PSB, DM = (T.dynamic(lower=0) for _ in range(3))
  A0, A1, D0, D1, C0, C1 = (T.dynamic() for _ in range(6))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (A0, A1), "float32"),
    AP: T.Tensor((PSA,), dtype=dtype),
    BP: T.Tensor((PSB,), dtype=dtype),
    D: T.Tensor((DM, N), (D0, D1), "float32"),
    C: T.Tensor((M, N), (C0, C1), "float32", access="output"),
    Bias: T.Tensor((1, N), dtype="float32"),
  ):
    a = T.packed(AP, (M, K), side="A", format="auto")
    b = T.packed(BP, (K, N), side="B", format="auto")
    product = (
      T.library.matmul((a, b), C, acc=T.load(D), packing="none", swap=True, BK=3, implementation=implementation)
      if level == "blocked"
      else T.library.matmul((a, b), C, acc=T.load(D), packing="none", swap=True, implementation=implementation)
      if level == "matmul"
      else T.dot((a, b), C, acc=T.load(D), implementation=implementation)
    )
    T.store(C, T.max((product + T.load(Bias)) * 0.5 + 0.125, 0))

  return kernel


@contextmanager
def _packing_metadata(template, origins, sides):
  """Use formal immutable packed-window IR, without mocking dispatch."""
  name = "test-packed-ab-column-metadata"

  def transform(ir, context):
    if ir.name != template.ir.name:
      return T.passes.PassResult(ir, changed=False)

    def rewrite(value):
      if value.op != "packed_load":
        return value
      role = 0 if value.data.side == "A" else 1
      data = replace(value.data, side=sides[role] if sides else value.data.side)
      if origins:
        data = replace(data, origins=tuple(I.literal(v) for v in origins))
      return replace(value, data=data)

    return rewrite_expressions(ir, rewrite)

  T.passes.register_pass(name, before="canonicalize-memory")(transform)
  try:
    yield
  finally:
    T.passes.unregister_pass(name)


def _lower(*, vl=64, svl=64, origins=(), sides=(), **kwargs):
  template = _template(**kwargs)
  with _packing_metadata(template, origins, sides):
    return (
      CPUBackend()
      .lower(template.ir, (), target="SME", options={**_OPTIONS, "_vector_state": VectorState(vl, svl)})
      .source
    )


def _wrapper(source, *, shape=None):
  m, n, k = (
    (f"vecops::meta::cint<{v}>" for v in shape)
    if shape
    else ("vecops::meta::Any{m}", "vecops::meta::Any{n}", "vecops::meta::Any{k}")
  )
  lines = [
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    (
      'extern "C" void packed_ab_col(void* rawp,void* ap,void* bp,void* dp,void* cp,void* biasp,'
      "vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t psa,vecops::nint_t psb,"
      "vecops::nint_t dm,vecops::nint_t d0,vecops::nint_t d1,vecops::nint_t c0,vecops::nint_t c1) {"
    ),
  ]
  for name, pointer, dtype, shape_, strides in (
    ("A", "rawp", "float32", f"{m},{k}", f"{k},vecops::meta::cint<1>"),
    ("AP", "ap", "bfloat16", "vecops::meta::Any{psa}", "vecops::meta::cint<1>"),
    ("BP", "bp", "bfloat16", "vecops::meta::Any{psb}", "vecops::meta::cint<1>"),
    ("D", "dp", "float32", f"vecops::meta::Any{{dm}},{n}", "vecops::meta::Any{d0},vecops::meta::Any{d1}"),
    ("C", "cp", "float32", f"{m},{n}", "vecops::meta::Any{c0},vecops::meta::Any{c1}"),
    ("Bias", "biasp", "float32", f"vecops::meta::cint<1>,{n}", f"{n},vecops::meta::cint<1>"),
  ):
    lines.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[dtype]}*>({pointer}),"
      f"vecops::tensor::make_shape({shape_}),vecops::tensor::make_strides({strides}));"
    )
  return "\n".join((*lines, "__kernel__(A,AP,BP,D,C,Bias);", "}"))


def _geometry(*, machine=None, state=None):
  if (machine or platform.machine()).lower() in {"aarch64", "arm64"}:
    svl = (state or get_vector_state()).sme_bytes
    if svl is None:
      pytest.skip("packed-AB column Native fixture requires current SME SVL")
    return svl // 4, svl // 2, 2
  return 16, 16, 32  # Public automatic BF16 AMX pack ABI, not an ARM guess.


def _count(spatial, k, panel, group):
  return ((spatial + panel - 1) // panel) * panel * ((k + group - 1) // group) * group


@pytest.mark.parametrize("svl", (16, 32, 64, 256))
def test_capacity_and_four_block_boundary_proof_are_svl_relative(svl):
  lanes, panel, group = _geometry(machine="aarch64", state=VectorState(64, svl))
  assert (lanes, panel, group) == (svl // 4, svl // 2, 2)
  for m in (3 * lanes + 1, 4 * lanes):
    assert _count(m, 257, panel, group) == 4 * lanes * 258
    assert 3 * lanes < m <= 4 * lanes
  assert not 3 * lanes < 3 * lanes <= 4 * lanes
  assert not 3 * lanes < 4 * lanes + 1 <= 4 * lanes


def test_fixture_geometry_is_explicit_without_sme_and_metadata_plugin_cleans_up():
  with pytest.raises(pytest.skip.Exception, match="current SME SVL"):
    _geometry(machine="aarch64", state=VectorState(16, None))
  assert _geometry(machine="x86_64", state=VectorState(None, None)) == (16, 16, 32)
  with pytest.raises(RuntimeError, match="deliberate cleanup"), _packing_metadata(_template(), (), ()):
    raise RuntimeError("deliberate cleanup")
  assert "test-packed-ab-column-metadata" not in {entry.name for entry in T.passes.registered_passes()}


def _pack_template():
  M, N, K, PSA, PSB = (T.dynamic(lower=0) for _ in range(5))
  A0, A1, B0, B1 = (T.dynamic() for _ in range(4))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (A0, A1), "float32"),
    B: T.Tensor((K, N), (B0, B1), "bfloat16"),
    AP: T.Tensor((PSA,), dtype="bfloat16", access="output"),
    BP: T.Tensor((PSB,), dtype="bfloat16", access="output"),
  ):
    T.pack(T.cast(T.load(A) * 0.5 + 0.125, "bfloat16"), AP, side="A", format="auto")
    T.pack(B, BP, side="B", format="auto")

  return kernel


def _output(m, n, *, negative=False):
  backing = np.full(max(1, m * (2 * n + 3)), np.float32(-77), dtype="float32")
  value = np.ndarray((m, n), dtype="float32", buffer=backing, strides=((2 * n + 3) * 4, 8))
  return value[::-1] if negative else value


def _run_native_column_cases(tmp_path, *, options):
  pytest.importorskip("ml_dtypes")
  lanes, panel, group = _geometry()
  maximum = 4 * lanes + 1
  psa, psb = _count(maximum, 257, panel, group), _count(1, 257, panel, group)
  shape = (4 * lanes, 1, 257)
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  pack = T.compile(_pack_template(), compiler=compiler, pass_options=options)
  kernels = {
    (dynamic, level): T.compile(
      _template(shape=None if dynamic else shape, level=level), compiler=compiler, pass_options=options
    )
    for dynamic in (False, True)
    for level in ("dot", "matmul", "blocked")
  }
  with vecops.collect_compile_requests() as collector:
    pack.operator(
      vecops.TensorMeta((maximum, 257), strides=(519, 2), dtype="float32"),
      vecops.TensorMeta((257, 1), strides=(2, 1), dtype="bfloat16"),
      vecops.TensorMeta((psa,), dtype="bfloat16"),
      vecops.TensorMeta((psb,), dtype="bfloat16"),
      **pack._core.specialization_values,
    )
    for kernel in kernels.values():
      kernel.operator(
        vecops.TensorMeta((4 * lanes, 257), strides=(519, 2), dtype="float32"),
        vecops.TensorMeta((psa,), dtype="bfloat16"),
        vecops.TensorMeta((psb,), dtype="bfloat16"),
        vecops.TensorMeta((4 * lanes, 1), strides=(5, 2), dtype="float32"),
        vecops.TensorMeta((4 * lanes, 1), strides=(5, 2), dtype="float32"),
        vecops.TensorMeta((1, 1), dtype="float32"),
        **kernel._core.specialization_values,
      )
  assert len(collector.requests) == 7
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  rng = np.random.default_rng(491)
  calls = 0
  for dynamic in (False, True):
    shapes = (
      (shape,)
      if not dynamic
      else (
        (3 * lanes, 1, 5),
        (3 * lanes + 1, 1, 5),
        shape,
        (maximum, 1, 9),
        (4 * lanes, 1, 0),
        (4 * lanes, 0, 7),
        (0, 1, 7),
      )
    )
    for m, n, k in shapes:
      for negative in (False, True):
        a = rng.integers(-4, 5, size=(m, 2 * k + 5)).astype("float32")
        a /= np.float32(16)
        a = a[:, : 2 * k : 2]
        a = a[:, ::-1] if negative else a
        b = np.ndarray((k, n), dtype="bfloat16", buffer=np.empty(max(1, 2 * k), dtype="bfloat16"), strides=(4, 2))
        b[:] = rng.integers(-3, 4, size=(k, n)) / 8
        b = b[::-1] if negative else b
        ap, bp = np.full(psa, 7, dtype="bfloat16"), np.full(psb, 7, dtype="bfloat16")
        pack.operator.native(a, b, ap, bp, **pack._core.specialization_values)
        transformed = (a.copy() * np.float32(0.5) + np.float32(0.125)).astype("bfloat16").astype("float32")
        product = transformed @ b.astype("float32")
        for prior_mode in ("disjoint", "alias", "broadcast"):
          for level in ("dot", "matmul", "blocked"):
            d = _output(1 if prior_mode == "broadcast" else m, n, negative=negative)
            d[:] = rng.integers(-2, 3, size=d.shape) / 8
            c = d if prior_mode == "alias" else _output(m, n, negative=negative)
            bias = np.full((1, n), np.float32(0.125))
            prior = d.copy()
            physical = c.base.copy()
            expected = np.maximum(((product + prior) + bias) * np.float32(0.5) + np.float32(0.125), 0)
            kernel = kernels[dynamic, level]
            kernel.operator.native(a, ap, bp, d, c, bias, **kernel._core.specialization_values)
            np.testing.assert_array_equal(c, expected, err_msg=f"{dynamic}/{level}/{(m, n, k)}/{negative}/{prior_mode}")
            active = np.zeros(c.base.size, dtype=bool)
            active[[i * (2 * n + 3) + 2 * j for i in range(m) for j in range(n)]] = True
            np.testing.assert_array_equal(c.base[~active], physical[~active])
            calls += 1
  assert calls == 144


def test_native_whole_packed_ab_column_strict_boundaries_fusion_alias_and_split_k(tmp_path):
  _run_native_column_cases(tmp_path, options=_OPTIONS)
