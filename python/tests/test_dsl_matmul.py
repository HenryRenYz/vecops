# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Matrix-product semantics, layout normalization and fusion contracts."""

import re
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import amx_available_for_test, assert_source_algorithms_lowered, make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import nodes


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-matmul")
  return make_compiler(target="Native", jobs=4, cache_dir=root / "cache", build_dir=root / "build")


def template(dtype="float32", *, library=False, fused=False, packed=False):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  AS0, AS1, BS0, BS1, CS0, CS1 = (T.dynamic() for _ in range(6))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (AS0, AS1), dtype),
    B: T.Tensor((K, N), (BS0, BS1), dtype),
    C: T.Tensor((M, N), (CS0, CS1), "float32", access="inout"),
  ):
    if packed:
      a = T.pack(A, side="A", format="dense")
      b = T.pack(B, side="B", format="dense")
    else:
      a = T.load(A)
      b = T.load(B)
    if library:
      value = T.matmul((a, b), C, acc=T.load(C) if fused else None)
    else:
      value = T.dot((a, b), C, acc=T.load(C) if fused else None)
    if fused:
      T.store(C, T.max(value * 0.75 + 1, 0))

  return kernel


@pytest.mark.parametrize(
  "library,fused,packed", [(False, False, False), (False, True, False), (True, True, False), (False, False, True)]
)
def test_product_layouts_and_fusion(compiler, library, fused, packed):
  kernel = T.compile(template(library=library, fused=fused, packed=packed), compiler=compiler)
  rng = np.random.default_rng(17)
  for m, n, k in ((0, 5, 7), (5, 0, 7), (5, 7, 0), (1, 19, 13), (19, 1, 13), (3, 5, 7), (17, 19, 33), (32, 64, 64)):
    a = rng.normal(size=(m, k)).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float32)
    for transposed in (False, True):
      x = np.asfortranarray(a) if transposed else a
      y = np.asfortranarray(b) if transposed else b
      c = np.asfortranarray(np.full((m, n), 0.25, np.float32)) if transposed else np.full((m, n), 0.25, np.float32)
      expected = x @ y + (c if fused else 0)
      if fused:
        expected = np.maximum(expected * 0.75 + 1, 0)
      kernel.operator.native(x, y, c, **kernel._core.specialization_values)
      np.testing.assert_allclose(c, expected, rtol=2e-5, atol=2e-5)
  assert "vecops::ops::" not in kernel.generated_source
  assert not re.search(r"(?:array|vector)<\s*vecops::vec::(?:Vec|Mask)", kernel.generated_source)


def test_reduction_layout_uses_automatic_transpose(compiler):
  @T.template
  def kernel(A: T.Tensor((17, 19), (1, 17), "float32"), C: T.Tensor((17,), dtype="float32", access="output")):
    T.store(C, T.reduce_add(T.load(A), axis=1))

  compiled = T.compile(kernel, compiler=compiler)
  a = np.asfortranarray(np.arange(17 * 19, dtype=np.float32).reshape(17, 19))
  c = np.empty(17, np.float32)
  compiled.operator.native(a, c)
  np.testing.assert_array_equal(c, a.sum(axis=1))
  assert "normalize-reduction-layout" in compiled.kernel_ir.pass_trace
  # Normalize the column-contiguous input into row-contiguous scratch. A
  # target may execute that copy with a register butterfly or a ZA plan.
  assert any(
    isinstance(node, (I.Allocate, I.WorkspaceAllocate))
    and tuple(I.constant(value) for value in node.view.shape) == (17, 19)
    and tuple(I.constant(value) for value in node.view.strides) == (19, 1)
    for node in nodes(compiled.kernel_ir.body)
  )
  assert_source_algorithms_lowered(compiled.backend_ir)


def test_invalid_shapes_and_packing():
  @T.template
  def bad(
    A: T.Tensor((3, 7), dtype="float32"),
    B: T.Tensor((6, 5), dtype="float32"),
    C: T.Tensor((3, 5), dtype="float32", access="output"),
  ):
    T.dot((A, B), C)

  source = CPUBackend().lower(bad.ir, ()).source
  assert "incompatible dot matrix extents" in source

  @T.template
  def bad_pack(A: T.Tensor((3, 7), dtype="float32")):
    T.pack(A, format="amx")

  with pytest.raises(TypeError, match="AMX packing supports"):
    CPUBackend().lower(bad_pack.ir, ())


def test_dynamic_source_cache_packing_has_one_algorithm_without_legacy_dispatch():
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((m, k), (k, 1), "bfloat16"),
    B: T.Tensor((k, n), (1, k), "bfloat16"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    a = T.cast(T.load(A, dtype="float32") * 2 + 1, "bfloat16")
    b = T.cast(T.load(B, dtype="float32") * 2 + 1, "bfloat16")
    T.library.matmul((a, b), C)

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512", options={"cpu_amx_shared_cache_dispatch": True})
  source = artifact.source
  assert "dsl_recipe_dot_" in source
  assert "packed-layout recipe" in "\n".join(artifact.kernel.diagnostics)
  assert "matmul_blocks(" not in source
  # Bound code growth for a fully dynamic transformed-input problem. This
  # is a structural budget, not a timing shortcut or a compiler flag change.
  assert len(source) < 16_000_000

  automatic = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  assert source == automatic  # Retired knob cannot resurrect another route.


@pytest.mark.parametrize("ad,bd", [("bfloat16", "bfloat16"), ("int8", "int8"), ("uint8", "int8")])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.skipif(not amx_available_for_test(), reason="explicit AMX matmul requires an x86 AMX target")
def test_amx_tails_packed_fusion(compiler, ad, bd, packed):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  out = "float32" if ad == "bfloat16" else "int32"

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), ad), B: T.Tensor((K, N), (1, K), bd), C: T.Tensor((M, N), (N, 1), out, access="inout")
  ):
    if packed:
      a = T.pack(A, side="A", format="amx")
      b = T.pack(B, side="B", format="amx")
    else:
      a = T.load(A)
      b = T.load(B)
    value = T.dot((a, b), C, acc=T.load(C))
    T.store(C, T.max(value, 0))

  compiled = T.compile(kernel, compiler=compiler)

  def dtype(name):
    if name == "bfloat16":
      return np.dtype(pytest.importorskip("ml_dtypes").bfloat16)
    return np.dtype(name)

  rng = np.random.default_rng(41)
  for m, n, k in ((1, 8, 65), (8, 1, 65), (3, 5, 7), (16, 16, 32), (17, 19, 33), (31, 47, 65), (64, 64, 64)):
    a = rng.integers(0 if ad == "uint8" else -3, 4, size=(m, k)).astype(dtype(ad))
    b = rng.integers(-3, 4, size=(n, k)).astype(dtype(bd)).T
    c = np.full((m, n), 1, dtype(out))
    expected = np.maximum(a.astype(dtype(out)) @ b.astype(dtype(out)) + 1, 0)
    compiled.operator.native(a, b, c, **compiled._core.specialization_values)
    np.testing.assert_allclose(c, expected, rtol=2e-5, atol=2e-5)
  assert "dsl_recipe_dot_" in compiled.generated_source


@T.func
def nested_normalize(x):
  return T.library.layernorm(x, eps=1e-5)


@T.func
def nested_epilogue(x):
  return T.max(x * 0.75 + 1, 0)


@pytest.mark.parametrize("packing", ["none", "auto"])
def test_nested_func_normalization_fusion(compiler, packing):
  @T.template
  def kernel(
    A: T.Tensor((17, 33), dtype="float32"),
    B: T.Tensor((33, 19), dtype="float32"),
    C: T.Tensor((17, 19), dtype="float32", access="output"),
  ):
    a = nested_normalize(A)
    result = T.library.matmul((a, T.load(B)), C, packing=packing)
    T.store(C, nested_epilogue(result))

  compiled = T.compile(kernel, compiler=compiler)
  rng = np.random.default_rng(51)
  a = rng.normal(size=(17, 33)).astype(np.float32)
  b = rng.normal(size=(33, 19)).astype(np.float32)
  c = np.empty((17, 19), np.float32)
  expected = (a - a.mean(axis=1, keepdims=True)) / np.sqrt(a.var(axis=1, keepdims=True) + 1e-5)
  expected = np.maximum((expected @ b) * 0.75 + 1, 0)
  compiled.operator.native(a, b, c)
  np.testing.assert_allclose(c, expected, rtol=3e-5, atol=3e-5)
  assert "decompose-row-maps" in compiled.kernel_ir.pass_trace
  from vecops.language._compiler import ir as I
  from vecops.language._compiler.ir.traversal import nodes

  statistics = {
    statement.view.storage
    for statement in nodes(compiled.kernel_ir.body)
    if isinstance(statement, I.Allocate) and tuple(I.constant(n) for n in statement.view.shape) == (17,)
  }
  assert len(statistics) == 2  # mean and inverse stddev, not a full normalized tile
  # Products have become executable source recipes by this boundary. Verify
  # that actual consumers still read the two row statistics, without requiring
  # an obsolete opaque dot node to survive lowering.
  assert not any(isinstance(node, I.Expr) and node.op == "dot" for node in nodes(compiled.kernel_ir.body))
  assert statistics <= {
    load.data.storage
    for statement in nodes(compiled.kernel_ir.body)
    if isinstance(statement, (I.Assign, I.Store, I.DeclareState, I.UpdateState))
    for load in I.loads_unique(statement.value)
  }


def test_pure_dot_return(compiler):
  @T.template
  def kernel(
    A: T.Tensor((3, 7), dtype="float32"),
    B: T.Tensor((7, 5), dtype="float32"),
    C: T.Tensor((3, 5), dtype="float32", access="output"),
  ):
    value = T.dot(T.load(A) * 2, T.load(B) + 1)
    T.store(C, T.max(value, 0))

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(21, dtype=np.float32).reshape(3, 7)
  b = np.arange(35, dtype=np.float32).reshape(7, 5)
  c = np.empty((3, 5), np.float32)
  compiled.operator.native(a, b, c)
  np.testing.assert_allclose(c, (a * 2) @ (b + 1), rtol=1e-6)


@pytest.mark.parametrize("ad,bd", [("bfloat16", "bfloat16"), ("uint8", "int8")])
def test_panel_phases_conversions_and_padding(compiler, ad, bd):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  acc = "float32" if ad == "bfloat16" else "int32"

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), ad),
    B: T.Tensor((K, N), (1, K), bd),
    D: T.Tensor((1, N), dtype=acc),
    C: T.Tensor((M, N), dtype="float64", access="output"),
  ):
    a = T.cast(T.load(A, dtype="float32") * 2 + 1, ad)
    b = T.cast(T.load(B, dtype="float32") * 2 + 1, bd)
    value = T.library.matmul((a, b), C, acc=T.load(D), BM=32, BN=64, BK=64)
    T.store(C, T.cast(value, "float64") * 0.25 + 2)

  compiled = T.compile(kernel, compiler=compiler)
  dtype = lambda name: np.dtype(__import__("ml_dtypes").bfloat16) if name == "bfloat16" else np.dtype(name)
  rng = np.random.default_rng(57)
  for m, n, k in ((17, 19, 1), (33, 65, 129), (65, 97, 257)):
    a = rng.integers(0 if ad == "uint8" else -3, 4, size=(m, k)).astype(dtype(ad))
    b = rng.integers(-3, 4, size=(n, k)).astype(dtype(bd)).T
    d = rng.integers(-3, 4, size=(1, n)).astype(acc)
    c = np.empty((m, n), np.float64)
    x = (a.astype(np.float32) * 2 + 1).astype(dtype(ad)).astype(acc)
    y = (b.astype(np.float32) * 2 + 1).astype(dtype(bd)).astype(acc)
    expected = (x @ y + d).astype(np.float64) * 0.25 + 2
    compiled.operator.native(a, b, d, c, **compiled._core.specialization_values)
    np.testing.assert_allclose(c, expected, rtol=1e-6, atol=1e-6)
  assert "dsl_recipe_dot_" in compiled.generated_source
  assert "matmul_blocks(" not in compiled.generated_source


def test_column_output_swaps_full_fused_graph(compiler):
  @T.template
  def kernel(
    A: T.Tensor((17, 33), dtype="float32"),
    B: T.Tensor((33, 19), dtype="float32"),
    C: T.Tensor((17, 19), (1, 17), dtype="float32", access="output"),
  ):
    result = T.library.matmul((T.load(A) * 2 + 1, T.load(B) - 1), C)
    T.store(C, T.max(result * 0.5, 0))

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(17 * 33, dtype=np.float32).reshape(17, 33) / 100
  b = np.arange(33 * 19, dtype=np.float32).reshape(33, 19) / 100
  c = np.empty((19, 17), np.float32).T
  compiled.operator.native(a, b, c)
  np.testing.assert_allclose(c, np.maximum(((a * 2 + 1) @ (b - 1)) * 0.5, 0), rtol=2e-5)
  assert "orient-matmul" in compiled.kernel_ir.pass_trace


def test_nested_module_capture_is_frozen(monkeypatch):
  @T.func
  def original(x):
    return x * 2

  @T.func
  def replacement(x):
    return x * 9

  monkeypatch.setattr(T.library, "captured_test", original, raising=False)

  @T.template
  def kernel(A: T.Tensor((3,), dtype="float32"), C: T.Tensor((3,), dtype="float32", access="output")):
    T.store(C, T.library.captured_test(T.load(A)))

  old = kernel.ir.body.identity
  expected_source = CPUBackend().lower(kernel.ir, ()).source
  monkeypatch.setattr(T.library, "captured_test", replacement)
  source = CPUBackend().lower(kernel.ir, ()).source
  assert kernel.ir.body.identity == old
  assert source == expected_source


def test_dependent_functional_dots(compiler):
  @T.template
  def kernel(
    A: T.Tensor((3, 7), dtype="float32"),
    B: T.Tensor((7, 5), dtype="float32"),
    D: T.Tensor((5, 4), dtype="float32"),
    C: T.Tensor((3, 4), dtype="float32", access="output"),
  ):
    result = T.dot(T.dot(T.load(A), T.load(B)) + 1, T.load(D))
    T.store(C, T.max(result, 0))

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(21, dtype=np.float32).reshape(3, 7) / 10
  b = np.arange(35, dtype=np.float32).reshape(7, 5) / 10
  d = np.arange(20, dtype=np.float32).reshape(5, 4) / 10
  c = np.empty((3, 4), np.float32)
  compiled.operator.native(a, b, d, c)
  np.testing.assert_allclose(c, ((a @ b) + 1) @ d, rtol=2e-5)


@pytest.mark.parametrize("fusion", ["always", "never", "auto"])
def test_noncontiguous_layernorm_tail_fuses_through_func(compiler, fusion):
  @T.template
  def kernel(
    A: T.Tensor((17, 33), (1, 17), dtype="float32"),
    B: T.Tensor((33, 19), dtype="float32"),
    C: T.Tensor((17, 19), dtype="float32", access="output"),
  ):
    a = T.cast(nested_normalize(A) * 2 + 1, "bfloat16")
    b = T.load(B, dtype="bfloat16")
    result = T.library.matmul((a, b), C, BM=32, BN=32, BK=32)
    T.store(C, nested_epilogue(result))

  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_matmul_fusion": fusion})
  bf = __import__("ml_dtypes").bfloat16
  rng = np.random.default_rng(81)
  a = np.asfortranarray(rng.normal(size=(17, 33)).astype(np.float32))
  b = rng.normal(size=(33, 19)).astype(np.float32)
  c = np.empty((17, 19), np.float32)
  norm = (a - a.mean(axis=1, keepdims=True)) / np.sqrt(a.var(axis=1, keepdims=True) + 1e-5)
  expected = np.maximum(((norm * 2 + 1).astype(bf).astype(np.float32) @ b.astype(bf).astype(np.float32)) * 0.75 + 1, 0)
  compiled.operator.native(a, b, c)
  np.testing.assert_allclose(c, expected, rtol=2e-4, atol=2e-4)
  # The complete normalization/matmul/helper pipeline must reach executable
  # IR; the chosen target need not spell its transpose as local_interleave.
  assert_source_algorithms_lowered(compiled.backend_ir)


@pytest.mark.parametrize("dtype", ["bfloat16", "int8"])
@pytest.mark.parametrize("side", ["A", "B"])
@pytest.mark.skipif(not amx_available_for_test(), reason="AMX packing requires an x86 AMX target")
def test_packed_abi_matches_public_cpp(compiler, dtype, side):
  S, K, F = (T.dynamic(lower=0) for _ in range(3))
  shape = (S, K) if side == "A" else (K, S)

  @T.template
  def dsl(A: T.Tensor(shape, dtype=dtype), P: T.Tensor((F,), dtype=dtype, access="output")):
    T.pack(A, P, side=side, format="amx")

  Native = T.DType(getattr(vecops, dtype))
  SideB = T.Const(side == "B")

  @T.template(source=Path(__file__).with_name("dsl_matmul_pack_reference.cpp"))
  def cpp(A: T.Tensor(shape, dtype=dtype), P: T.Tensor((F,), dtype=dtype, access="output")):
    Native, SideB  # noqa: B018 -- source-kernel constexpr captures

  ds, cp = (T.compile(kernel, compiler=compiler) for kernel in (dsl, cpp))
  dt = __import__("ml_dtypes").bfloat16 if dtype == "bfloat16" else np.int8
  kr = 32 if dtype == "bfloat16" else 64
  for spatial, k in ((1, 1), (7, 31), (17, 65), (33, 129), (64, 256)):
    sh = (spatial, k) if side == "A" else (k, spatial)
    a = ((np.arange(spatial * k) % 7) - 3).reshape(sh).astype(dt)
    count = ((spatial + 15) // 16) * 16 * ((k + kr - 1) // kr) * kr
    p = np.zeros(count, dtype=dt)
    q = np.zeros_like(p)
    ds.operator.native(a, p, **ds._core.specialization_values)
    cp.operator.native(a, q, **cp._core.specialization_values)
    np.testing.assert_array_equal(p, q)
