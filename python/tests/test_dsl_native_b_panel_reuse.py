# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Private native B preparation preserves K tails, transforms and aliasing."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import children
from vecops.language._library.matrix_recipe import blocked_matrix

FAMILIES = (("bfloat16", "bfloat16", "float32"), ("uint8", "int8", "int32"))


def template(a_dtype, b_dtype, acc_dtype):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), a_dtype),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), acc_dtype),
    Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), acc_dtype),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), acc_dtype, access="output"),
  ):
    # The physical B memory type deliberately differs from its native type;
    # a layout proof must never bypass this conversion boundary.
    b = T.cast(T.load(B), b_dtype)
    product = T.matmul((T.load(A), b), C, acc=T.load(Prior), acc_dtype=acc_dtype, BM=16, BN=33, BK=17, packing="auto")
    T.store(C, product * 2 + 1)

  return kernel


@pytest.fixture(scope="module", params=FAMILIES)
def native_kernel(request, tmp_path_factory):
  a_dtype, b_dtype, acc_dtype = request.param
  root = tmp_path_factory.mktemp("native-b-panel-" + a_dtype)
  compiler = make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build")
  compiled = T.compile(template(a_dtype, b_dtype, acc_dtype), compiler=compiler)
  return compiled, request.param


def _arguments(family, shape, *, reverse, alias):
  a_dtype, _b_dtype, acc_dtype = family
  if a_dtype == "bfloat16":
    a_type = pytest.importorskip("ml_dtypes").bfloat16
    b_type = a_type
    poison = np.nan
    compute = np.float64
  else:
    a_type, b_type = np.uint8, np.int8
    poison = -11111
    compute = np.int64
  m, n, k = shape
  a_storage = np.full((m + 3, 2 * k + 5), 255 if a_dtype == "uint8" else np.nan, dtype=a_type)
  a = a_storage[1 : m + 1, 2 : 2 + 2 * k : 2]
  if reverse:
    a = a[::-1, ::-1]
  a_values = np.arange(m * k).reshape(m, k) % 9
  a[:] = a_values if a_dtype == "uint8" else a_values / 8

  rows = max(m, k) + 3 if alias else k + 3
  b_storage = np.full((rows, 2 * n + 5), poison, dtype=acc_dtype)
  b = b_storage[1 : k + 1, 2 : 2 + 2 * n : 2]
  if reverse:
    b = b[::-1, ::-1]
  values = np.arange(k * n).reshape(k, n) % 13 - 6
  b[:] = values if acc_dtype == "int32" else values / 8

  output_storage = b_storage if alias else np.full((m + 3, 2 * n + 5), poison, dtype=acc_dtype)
  output = output_storage[1 : m + 1, 2 : 2 + 2 * n : 2]
  if reverse:
    output = output[::-1, ::-1]
  if m:
    prior = output[:1]
    prior[:] = np.arange(n) % 7 - 3
  else:
    prior = np.asarray(np.arange(n) % 7 - 3, dtype=acc_dtype).reshape(1, n)
  expected = ((a.astype(compute) @ b.astype(b_type).astype(compute)) + prior.copy()) * 2 + 1
  expected = expected.astype(acc_dtype)
  untouched = output_storage.copy()
  mask = np.ones(output_storage.shape, dtype=bool)
  mask[1 : m + 1, 2 : 2 + 2 * n : 2] = False
  return a, b, prior, output, expected, output_storage, untouched, mask


def test_native_prepared_b_multiple_phases_conversion_strides_alias_and_empty(native_kernel):
  compiled, family = native_kernel
  cases = (
    ((33, 67, 35), False, False),
    ((33, 35, 65), True, False),
    ((33, 67, 35), True, True),
    ((33, 35, 0), True, True),
    ((0, 35, 17), False, False),
    ((33, 0, 17), False, False),
  )
  for shape, reverse, alias in cases:
    a, b, prior, output, expected, storage, untouched, mask = _arguments(family, shape, reverse=reverse, alias=alias)
    compiled.operator.native(a, b, prior, output, **compiled._core.specialization_values)
    # Small binary fractions make BF16 products and sums exactly representable,
    # while mixed-sign bytes verify signed input interpretation independently.
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(storage[mask], untouched[mask])


@T.func
def _prepare_probe(value, acc_dtype, raw=None):
  prepared = T.alloc((1, 7), dtype="int16")
  T.store(prepared, T.full(prepared.shape, 5, dtype="int16"))
  return prepared


@T.func
def _prepared_probe_leaf(inputs, output, *, acc=None, acc_dtype="float32", raw_a=None, prepared_b=None):
  T.static_assert(prepared_b is not None and prepared_b.dtype == "int16")
  T.static_assert(len(prepared_b.shape) == 2)
  T.store(output, T.full(output.shape, 1, dtype=acc_dtype))


def test_private_b_preparation_is_outside_m_loop_with_exact_cache_steps():
  @T.template
  def kernel(
    A: T.Tensor((33, 35), dtype="float32"),
    B: T.Tensor((35, 67), dtype="float32"),
    C: T.Tensor((33, 67), dtype="float32", access="output"),
  ):
    blocked_matrix(
      (T.load(A), T.load(B)), C, _prepared_probe_leaf, BM=16, BN=33, BK=17, packing="auto", prepare_b=_prepare_probe
    )

  ir = lower(kernel.ir)
  preparations = []

  def visit(block, loops=()):
    for statement in block.statements:
      if isinstance(statement, I.Allocate) and statement.view.dtype == "int16":
        preparations.append(loops)
      for child in children(statement):
        visit(child, loops + ((statement,) if isinstance(statement, I.For) else ()))

  visit(ir.body)
  assert len(preparations) == 1
  loops = preparations[0]
  assert len(loops) == 2
  assert [I.constant(loop.step) for loop in loops] == [33, 17]
  assert all("mi" not in loop.target for loop in loops)
