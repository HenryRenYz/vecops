# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Four-output K-inner GEMV keeps grouped dot, tails and captured aliases."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.frontend.parser import lower
from vecops.language._library.matmul import _inner_grouped_product, _inner_sum_four, vector_dot_into


@T.func
def _epilogue(value, row, column):
  return value * 2 + T.cast(row + column, value.dtype)


def template(dtype, vary_rows, *, seed=True):
  extent, k = T.dynamic(lower=0), T.dynamic(lower=0)
  m, n = (extent, 1) if vary_rows else (1, extent)
  ad, bd, cd = ("bfloat16", "bfloat16", "float32") if dtype == "bf16" else ("uint8", "int8", "int32")

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), ad),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), bd),
    Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), cd),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), cd, access="output"),
  ):
    vector_dot_into(
      (T.load(A), T.load(B)), C, acc=T.load(Prior) if seed else None, acc_dtype=cd, epilogue=_epilogue, prefer_k=True
    )

  return kernel


@pytest.fixture(scope="module", params=[("bf16", False), ("bf16", True), ("byte", False), ("byte", True)])
def compiled(request, tmp_path_factory):
  dtype, vary_rows = request.param
  root = tmp_path_factory.mktemp("shared-k-gemv")
  return (
    T.compile(
      template(dtype, vary_rows), compiler=make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build")
    ),
    dtype,
    vary_rows,
  )


def arguments(dtype, vary_rows, extent, k):
  m, n = (extent, 1) if vary_rows else (1, extent)
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  ad, bd, cd = (bf16, bf16, np.float32) if dtype == "bf16" else (np.uint8, np.int8, np.int32)
  a = np.full((m, k * 2), np.nan if dtype == "bf16" else 255, dtype=ad)[:, ::-2]
  b = np.full((k * 2, n), np.nan if dtype == "bf16" else 127, dtype=bd)[::-2]
  a[:] = (np.arange(m * k).reshape(m, k) % 11) / (8 if dtype == "bf16" else 1)
  b[:] = (np.arange(k * n).reshape(k, n) % 13 - 6) / (8 if dtype == "bf16" else 1)
  backing = np.full((m + 2, n + 2), 111, dtype=cd)
  c = backing[1 : m + 1, 1 : n + 1]
  prior = c[:1] if m else np.zeros((1, n), dtype=cd)
  prior[:] = np.arange(n) + (3 if dtype == "bf16" else 2147483600)
  oracle = (
    (
      a.astype(np.float64 if dtype == "bf16" else np.int64) @ b.astype(np.float64 if dtype == "bf16" else np.int64)
      + prior.copy()
    )
    * 2
    + np.arange(m)[:, None]
    + np.arange(n)[None, :]
  )
  return (a, b, prior, c), oracle.astype(cd), backing


@pytest.mark.parametrize("dtype,vary_rows", [("bf16", False), ("bf16", True), ("byte", False), ("byte", True)])
@pytest.mark.parametrize("k", [0, 1, 3, 17, 65, 129])
def test_debug_four_grouped_reductions(dtype, vary_rows, k):
  ad, bd, cd = ("bfloat16", "bfloat16", "float32") if dtype == "bf16" else ("uint8", "int8", "uint32")
  m, n = (4, 1) if vary_rows else (1, 4)

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype=ad), B: T.Tensor((k, n), dtype=bd), C: T.Tensor((4,), dtype=cd, access="output")
  ):
    a, b = T.load(A), T.load(B)
    if vary_rows:
      p0 = _inner_grouped_product(a[0, :], b[:, 0], cd)
      p1 = _inner_grouped_product(a[1, :], b[:, 0], cd)
      p2 = _inner_grouped_product(a[2, :], b[:, 0], cd)
      p3 = _inner_grouped_product(a[3, :], b[:, 0], cd)
    else:
      p0 = _inner_grouped_product(a[0, :], b[:, 0], cd)
      p1 = _inner_grouped_product(a[0, :], b[:, 1], cd)
      p2 = _inner_grouped_product(a[0, :], b[:, 2], cd)
      p3 = _inner_grouped_product(a[0, :], b[:, 3], cd)
    sums = T.reduce((p0, p1, p2, p3), reducer=_inner_sum_four)
    for lane in T.unroll(4):
      C[lane] = sums[lane]

  values, _, _ = arguments(dtype, vary_rows, 4, k)
  a, b = (np.ascontiguousarray(x) for x in values[:2])
  result = np.empty(4, dtype=np.float32 if dtype == "bf16" else np.uint32)
  T.compile(kernel, compiler=make_compiler(jobs=2)).interpret(a, b, result)
  expected = (
    a.astype(np.float64 if dtype == "bf16" else np.int64) @ b.astype(np.float64 if dtype == "bf16" else np.int64)
  ).reshape(-1)
  np.testing.assert_array_equal(result, expected.astype(result.dtype))


@pytest.mark.parametrize("extent,k", [(4, 3), (5, 17), (8, 65), (11, 129)])
def test_native_shared_k_tails_seed_alias_and_epilogue(compiled, extent, k):
  kernel, dtype, vary_rows = compiled
  values, oracle, backing = arguments(dtype, vary_rows, extent, k)
  kernel.operator.native(*values, **kernel._core.specialization_values)
  np.testing.assert_allclose(values[-1], oracle, rtol=2e-6, atol=1e-6)
  np.testing.assert_array_equal(backing[[0, -1]], 111)
  np.testing.assert_array_equal(backing[:, [0, -1]], 111)


@pytest.mark.parametrize("dtype,vary_rows", [("bf16", False), ("bf16", True), ("byte", False), ("byte", True)])
def test_shared_k_template_lowers(dtype, vary_rows):
  lower(template(dtype, vary_rows).ir)


@pytest.mark.parametrize("vary_rows", [False, True])
def test_explicit_four_fold_keeps_one_traversal_above_default_fusion_limit(vary_rows):
  from vecops.language._compiler.backends.cpu.backend import CPUBackend
  from vecops.language._compiler.ir.iteration import FoldLoop
  from vecops.language._compiler.ir.traversal import children

  artifact = CPUBackend().lower(template("bf16", vary_rows).ir, (), target="AVX512")

  def folds(block):
    for statement in block.statements:
      if isinstance(statement, FoldLoop):
        yield statement
      for child in children(statement):
        yield from folds(child)

  groups = [fold for fold in folds(artifact.kernel.body) if fold.explicit_tuple]
  assert groups and all(len(fold.assignments) == 4 for fold in groups)
  assert "std::tuple<float, float, float, float>" in artifact.source


def test_shared_grouped_operands_are_loaded_once_per_full_vector_iteration():
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  source = CPUBackend().lower(template("bf16", False).ir, (), target="AVX512").source
  group = source.index("std::tuple<float, float, float, float>")
  start = source.index("/* full vector groups */ {", group) + len("/* full vector groups */ ")
  depth, end = 1, start + 1
  while depth:
    depth += (source[end] == "{") - (source[end] == "}")
    end += 1
  body = source[start:end]
  assert body.count("vecops::vec::widening_dot(") == 4
  # Four independent operands plus one shared operand, rather than eight.
  assert body.count("vecops::vec::load_convert(") == 5


@pytest.mark.parametrize("side", ["A", "B"])
def test_native_shared_k_input_output_overlap_preserves_captured_inputs(compiled, side):
  kernel, dtype, vary_rows = compiled
  values, _, _ = arguments(dtype, vary_rows, 5, 17)
  a, b, _, original_c = values
  original = a if side == "A" else b
  storage = np.full(max(original.nbytes, original_c.nbytes) + 32, 0x7B, dtype=np.uint8)
  source = np.ndarray(original.shape, dtype=original.dtype, buffer=storage, offset=16)
  source[:] = original
  c = np.ndarray(original_c.shape, dtype=original_c.dtype, buffer=storage, offset=16)
  a, b = (source, b) if side == "A" else (a, source)
  prior = c[:1]
  prior[:] = 3
  compute = np.float64 if dtype == "bf16" else np.int64
  oracle = (a.astype(compute) @ b.astype(compute) + prior.copy()) * 2
  oracle += np.arange(c.shape[0])[:, None] + np.arange(c.shape[1])[None, :]
  before = storage.copy()
  kernel.operator.native(a, b, prior, c, **kernel._core.specialization_values)
  np.testing.assert_array_equal(c, oracle.astype(c.dtype))
  np.testing.assert_array_equal(storage[:16], before[:16])
  np.testing.assert_array_equal(storage[16 + c.nbytes :], before[16 + c.nbytes :])


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_non_grouped_inner_dispatch_does_not_recurse(dtype):
  @T.template
  def kernel(
    A: T.Tensor((1, 17), dtype=dtype),
    B: T.Tensor((17, 5), dtype=dtype),
    C: T.Tensor((1, 5), dtype=dtype, access="output"),
  ):
    vector_dot_into((A, B), C, acc_dtype=dtype, prefer_k=True)

  lower(kernel.ir)


def test_native_shared_k_bf16_varied_magnitudes(compiled):
  kernel, dtype, vary_rows = compiled
  if dtype != "bf16":
    pytest.skip("BF16 rounding-order regression")
  values, _, _ = arguments(dtype, vary_rows, 11, 129)
  a, b, prior, c = values
  rng = np.random.default_rng(417)
  a[:] = rng.normal(size=a.shape) * np.exp2(rng.integers(-4, 4, size=a.shape))
  b[:] = rng.normal(size=b.shape) * np.exp2(rng.integers(-4, 4, size=b.shape))
  oracle = (a.astype(np.float64) @ b.astype(np.float64) + prior.copy()) * 2
  oracle += np.arange(c.shape[0])[:, None] + np.arange(c.shape[1])[None, :]
  kernel.operator.native(*values, **kernel._core.specialization_values)
  np.testing.assert_allclose(c, oracle.astype(np.float32), rtol=5e-5, atol=2e-5)
