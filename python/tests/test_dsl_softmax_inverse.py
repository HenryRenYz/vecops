"""Strict softmax division is a library algorithm, not an rcp rewrite."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.frontend.helpers import expand_helper
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._compiler.operations import snapshot
from vecops.language._library.math import softmax_inverse


def template(dtype="float32", rank=2, *, masked=False, accuracy="strict", explicit=False):
  Shape = tuple(T.dynamic(lower=0) for _ in range(rank))
  InputStrides = tuple(T.dynamic() for _ in range(rank))
  OutputStrides = tuple(T.dynamic() for _ in range(rank))

  @T.template
  def kernel(
    A: T.Tensor(Shape, InputStrides, dtype),
    Mask: T.Tensor(Shape, dtype="float32"),
    C: T.Tensor(Shape, OutputStrides, dtype, access="output"),
  ):
    x = T.load(A)
    if explicit:
      T.store(C, x * T.rcp(T.reduce_add(x), accuracy=accuracy))
    elif masked:
      T.store(C, T.library.masked_softmax(x, T.load(Mask), accuracy=accuracy))
    else:
      T.store(C, T.library.softmax(x, accuracy=accuracy))

  return kernel


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("accuracy", ("strict", "fast", "estimate"))
def test_visible_inverse_helper_keeps_dtype_and_accuracy_contract(dtype, accuracy):
  denominator = I.Expr(I.ScalarType(dtype), "local", data="denominator")
  result, prelude = expand_helper(
    softmax_inverse, (denominator, accuracy), catalog=snapshot(), prefix="inverse_test"
  )
  expressions = [node for node in nodes((result, I.Block(prelude))) if isinstance(node, I.Expr)]
  if accuracy == "strict":
    division = next(node for node in expressions if node.op == "div")
    assert division.type.dtype == dtype
    assert division.args[0].type.dtype == dtype
    assert division.args[0].op == "cast"
    assert I.constant(division.args[0].args[0]) == 1
    assert not any(node.op == "rcp" for node in expressions)
  else:
    reciprocal = next(node for node in expressions if node.op == "rcp")
    assert reciprocal.type.dtype == dtype
    assert reciprocal.data == accuracy
    assert not any(node.op == "div" for node in expressions)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("rank", (1, 2))
@pytest.mark.parametrize("streaming", (False, True))
def test_strict_softmax_inverse_is_scalar_division_in_all_schedules(dtype, rank, streaming):
  lowered = CPUBackend().lower(
    template(dtype, rank).ir, (), target="SVE", options={"row_normalization_streaming": streaming}
  )
  expressions = [node for node in nodes(lowered.kernel.body) if isinstance(node, I.Expr)]
  divisions = [node for node in expressions if node.op == "div" and isinstance(node.type, I.ScalarType)]
  assert divisions
  assert any(node.type.dtype == dtype and node.args[0].type.dtype == dtype for node in divisions)
  assert not any(node.op == "rcp" for node in expressions)
  assert "vecops::vec::rcp(" not in lowered.source
  assert " / " in lowered.source
  assert_no_vec_mask_arrays(lowered.source)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("streaming", (False, True))
def test_masked_rank_two_uses_typed_division_of_a_safe_denominator(dtype, streaming):
  lowered = CPUBackend().lower(
    template(dtype, masked=True).ir, (), target="SVE", options={"row_normalization_streaming": streaming}
  )
  expressions = [node for node in nodes(lowered.kernel.body) if isinstance(node, I.Expr)]
  assert any(
    node.op == "div" and isinstance(node.type, I.ScalarType) and node.type.dtype == dtype for node in expressions
  )
  assert any(node.op == "where" and isinstance(node.type, I.ScalarType) for node in expressions)
  assert "vecops::vec::rcp(" not in lowered.source
  assert_no_vec_mask_arrays(lowered.source)


@pytest.mark.parametrize("accuracy", ("fast", "estimate"))
@pytest.mark.parametrize("rank", (1, 2))
def test_non_strict_softmax_retains_native_reciprocal(rank, accuracy):
  source = CPUBackend().lower(template(rank=rank, accuracy=accuracy).ir, (), target="SVE").source
  assert "vecops::vec::rcp(" in source
  assert f"vecops::vec::opt::math::{accuracy}" in source


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_explicit_strict_rcp_still_uses_native_dispatch(dtype):
  source = CPUBackend().lower(template(dtype, rank=1, explicit=True).ir, (), target="SVE").source
  assert "vecops::vec::rcp(" in source
  assert "vecops::vec::opt::math::strict" in source


def inverse_template(dtype):
  N = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((N,), dtype=dtype), C: T.Tensor((N,), dtype=dtype, access="output")):
    for i in T.serial(N, unroll=1):
      T.store(C[i], softmax_inverse(T.load(A[i])))

  return kernel


@pytest.fixture(scope="module")
def native_kernels(tmp_path_factory):
  path = tmp_path_factory.mktemp("softmax-inverse")
  compiler = make_compiler(jobs=4, cache_dir=path / "cache", build_dir=path / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for dtype in ("float32", "float64"):
      for rank in (1, 2):
        for masked in (False, True):
          compiled = T.compile(template(dtype, rank, masked=masked), compiler=compiler)
          shape = (33,) if rank == 1 else (3, 33)
          compiled.operator(
            vecops.TensorMeta(shape, dtype=dtype),
            vecops.TensorMeta(shape, dtype="float32"),
            vecops.TensorMeta(shape, dtype=dtype),
            **compiled._core.specialization_values,
          )
          kernels[dtype, rank, masked] = compiled
      compiled = T.compile(inverse_template(dtype), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((13,), dtype=dtype),
        vecops.TensorMeta((13,), dtype=dtype),
        **compiled._core.specialization_values,
      )
      kernels[dtype, "inverse"] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


def reference(source, mask, masked):
  if not source.shape[-1]:
    return source.copy()
  with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
    scores = np.where(mask, source, -np.inf) if masked else source
    maximum = scores.max(axis=-1, keepdims=True)
    if masked:
      maximum = np.where(maximum == -np.inf, 0, maximum)
    probability = np.exp(scores - maximum)
    total = probability.sum(axis=-1, keepdims=True)
    if masked:
      total = np.where(total > 0, total, 1)
    return probability / total


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("rank", (1, 2))
@pytest.mark.parametrize("masked", (False, True))
def test_native_strict_softmax_zero_tail_masks_and_aliases(native_kernels, dtype, rank, masked):
  kernel = native_kernels[dtype, rank, masked]
  rng = np.random.default_rng(192)
  for width in (0, 1, 31, 33, 64, 129):
    shape = (width,) if rank == 1 else (3, width)
    for layout in ("dense", "negative", "inplace", "shifted"):
      storage = np.empty(np.prod(shape) + 1, dtype=dtype)
      source = storage[:-1].reshape(shape)
      source[:] = rng.normal(size=shape) * 3
      if layout == "negative":
        source = source[..., ::-1]
      output = np.empty_like(source)
      if layout == "inplace":
        output = source
      elif layout == "shifted":
        output = storage[1:].reshape(shape)
      mask = (rng.random(shape) > 0.25).astype(np.float32)
      if masked:
        if rank == 1:
          mask[:] = 0  # The zero-total path must not divide by zero.
        else:
          mask[1] = 0
      expected = reference(source.copy(), mask, masked)
      kernel(source, mask, output)
      tolerance = 3e-6 if dtype == "float32" else 3e-12
      np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
      if masked:
        np.testing.assert_array_equal(output if rank == 1 else output[1], 0)
  # Plain softmax propagates broken (NaN/+inf/all--inf) rows. Masked softmax
  # specifies finite valid logits, but invalid NaN/inf logits must be ignored.
  source = np.asarray([1, np.nan, np.inf, -np.inf, -2], dtype=dtype)
  if rank == 2:
    source = np.stack((source, np.full_like(source, -np.inf), np.zeros_like(source)))
  mask = np.isfinite(source).astype(np.float32) if masked else np.ones_like(source, dtype=np.float32)
  expected = reference(source, mask, masked)
  output = np.empty_like(source)
  kernel(source, mask, output)
  np.testing.assert_allclose(output, expected, rtol=3e-6, atol=3e-6, equal_nan=True)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_native_typed_inverse_ieee_specials_and_signed_zero(native_kernels, dtype):
  limits = np.finfo(dtype)
  source = np.asarray([0, -0.0, np.inf, -np.inf, np.nan, 1, 3, -7, limits.max, limits.tiny], dtype=dtype)
  output = np.empty_like(source)
  with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    expected = np.asarray(1, dtype=dtype) / source
  native_kernels[dtype, "inverse"](source, output)
  np.testing.assert_allclose(output, expected, rtol=0, atol=0, equal_nan=True)
  defined_sign = ~np.isnan(expected)
  np.testing.assert_array_equal(np.signbit(output[defined_sign]), np.signbit(expected[defined_sign]))
