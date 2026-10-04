"""Common iteration contracts, ordered recurrences and alias-versioned fusion."""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.iteration import FoldLoop, AxisResult
from vecops.language._compiler.ir.traversal import nodes


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-iteration")
  return vecops.Compiler(
    target="Native", cc="gcc", cxx="g++", jobs=4, build_dir=root / "build", cache_dir=root / "cache"
  )


@pytest.mark.parametrize("associative", [False, True])
def test_custom_moments_order_and_native_loop(compiler, associative):
  @T.reducer(identity=(0.0, 0.0), associative=associative)
  def sums(a, b):
    return a[0] + b[0], a[1] + b[1]

  n = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((n,), dtype="float32"), C: T.Tensor((2,), dtype="float32", access="output")):
    x = T.load(A)
    result = T.reduce((x, x * x), reducer=sums, unroll=7)
    C[0] = result[0] * 2
    C[1] = result[1] + 3

  compiled = T.compile(kernel, compiler=compiler)
  for size in (0, 1, 17, 257):
    a = np.linspace(-1, 1, size, dtype=np.float32)
    out = np.empty(2, dtype=np.float32)
    compiled(a, out)
    np.testing.assert_allclose(out, [a.sum() * 2, np.square(a).sum() + 3], rtol=3e-6, atol=3e-5)
  folds = [s for s in nodes(compiled.kernel_ir.body) if isinstance(s, FoldLoop)]
  assert bool(folds) == associative
  if associative:
    assert any(len(s.assignments) == 2 for s in folds)
    assert "fold_index" not in compiled.generated_source
    assert "vecops::vec::reduce_add" in compiled.generated_source


def test_nested_reductions_expose_one_shared_fold():
  @T.template
  def kernel(A: T.Tensor((257,), dtype="float32"), C: T.Tensor((1,), dtype="float32", access="output")):
    x = T.load(A)
    C[0] = T.reduce_add(x) / 257 + T.reduce_add(x * x) / 257

  artifact = CPUBackend().lower(kernel.ir, ())
  folds = [s for s in nodes(artifact.kernel.body) if isinstance(s, FoldLoop)]
  assert len(folds) == 1 and len(folds[0].assignments) == 2


def test_non_neutral_custom_initial_value_is_not_discarded(compiler):
  @T.reducer(identity=7.0, associative=True)
  def sum_(a, b):
    return a + b

  @T.template
  def kernel(A: T.Tensor((17,), dtype="float32"), C: T.Tensor((1,), dtype="float32", access="output")):
    C[0] = T.reduce(T.load(A), reducer=sum_)

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(17, dtype=np.float32)
  out = np.empty(1, dtype=np.float32)
  compiled(a, out)
  np.testing.assert_array_equal(out, [a.sum() + 7])
  assert not any(isinstance(s, FoldLoop) for s in nodes(compiled.kernel_ir.body))


def test_fused_axis_reductions(compiler):
  @T.template
  def kernel(A: T.Tensor((19, 37), dtype="float16"), C: T.Tensor((19,), dtype="float32", access="output")):
    x = T.load(A, dtype="float32")
    T.store(C, T.reduce_add(x, axis=1, unroll=7) + T.reduce_max(x, axis=1, unroll=7))

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(19 * 37, dtype=np.float16).reshape(19, 37) / 64
  c = np.empty(19, dtype=np.float32)
  compiled(a, c)
  np.testing.assert_allclose(c, a.astype(np.float32).sum(1) + a.max(1), rtol=2e-6)
  assert any(isinstance(e, AxisResult) and len(e.fold.sources) == 2 for e in nodes(compiled.kernel_ir.body))
  assert "fused axis fold" in compiled.generated_source


@pytest.mark.parametrize("fixed", [False, True])
def test_simd_unroll_and_loop_fusion(compiler, fixed):
  vector = T.Vector("float32", N=16) if fixed else T.Vector("float32")

  @T.template
  def kernel(
    A: T.Tensor((37,), dtype="float32"),
    B: T.Tensor((37,), dtype="float32", access="inout"),
    C: T.Tensor((37,), dtype="float32", access="output"),
  ):
    for i in T.simd(37, vector=vector, unroll=3):
      T.scatter(B, i, T.gather(A, i) * 2, unique=True)
    for j in T.simd(37, vector=vector, unroll=3):
      T.scatter(C, j, T.gather(B, j) + 1, unique=True)

  compiled = T.compile(kernel, compiler=compiler)
  assert any("loop fusion:" in s for s in compiled.kernel_ir.diagnostics)
  assert "operation-major serial unroll" in compiled.generated_source
  a = np.arange(37, dtype=np.float32)
  b, c = np.empty_like(a), np.empty_like(a)
  compiled(a, b, c)
  np.testing.assert_array_equal(c, a * 2 + 1)
  # Cross-iteration physical overlap must select the original two-loop order.
  base = np.arange(38, dtype=np.float32)
  original = base.copy()
  reference = np.empty_like(c)
  unfused = T.compile(kernel, compiler=compiler, pass_options={"loop_fusion": False})
  unfused(original[:-1], original[1:], reference)
  compiled(base[:-1], base[1:], c)
  np.testing.assert_array_equal(c, reference)
  assert not re.search(r"(?:Vec|Mask)<[^;\n]+>\s+\w+\s*\[", compiled.generated_source)


@pytest.mark.parametrize("exclusive", [False, True])
def test_scan_consumer_fusion_and_alias_fallback(compiler, exclusive):
  @T.template
  def kernel(A: T.Tensor((37,), dtype="float32"), C: T.Tensor((37,), dtype="float32", access="output")):
    T.store(C, T.scan(T.load(A) * 2, exclusive=exclusive) * 3 + 1)

  compiled = T.compile(kernel, compiler=compiler)
  assert any("loop consumer fusion:" in s for s in compiled.kernel_ir.diagnostics)
  a = np.arange(37, dtype=np.float32)
  expected = np.cumsum(a * 2)
  if exclusive:
    expected = np.concatenate(([0], expected[:-1]))
  expected = expected * 3 + 1
  c = np.empty_like(a)
  compiled(a, c)
  np.testing.assert_array_equal(c, expected)
  compiled(a, a)
  np.testing.assert_array_equal(a, expected)
