# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cache windows preserve runtime singleton broadcasts and captured reads.

Native shares the packed-column strict fixture, which first exposed a runtime
DM=1 prior being read out of bounds in the BK3 cache driver. This test does
not require or enable the independent packed-AB dispatch candidate.
"""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler, parameter_snapshot_name
from test_dsl_matmul_sve_optimized import _cross
from test_dsl_sme_family_owners import _assert_inlined_owners, _owners
from test_dsl_sme_packed_ab_col import _lower, _template, _wrapper
from test_dsl_sme_packed_ab_col import (
  test_native_whole_packed_ab_column_strict_boundaries_fusion_alias_and_split_k as _strict_native_fixture,
)
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.frontend.helpers import expand_helper
from vecops.language._compiler.operations import snapshot
from vecops.language._debug.interpreter import Interpreter


@T.func
def _broadcast_window(value, row, column, rows, columns):
  rs = T.where(value.shape[0] == 1, 0, row)
  cs = T.where(value.shape[1] == 1, 0, column)
  re = T.where(value.shape[0] == 1, 1, row + rows)
  ce = T.where(value.shape[1] == 1, 1, column + columns)
  return value[rs:re, cs:ce]


@pytest.mark.parametrize("extent", [1, 9])
@pytest.mark.parametrize("stride", [5, -5])
def test_source_windows_preserve_singletons_and_original_read_identity(extent, stride):
  from vecops.language._compiler.ir.traversal import walk_unique

  shape = (_scalar("dm"), I.literal(1))
  view = I.View("Prior", "float32", shape, (I.literal(stride), I.literal(1)), I.literal(51))
  load = I.Expr(I.TileType("float32", shape), "load", data=view)
  expression = I.binary("add", load, load)
  result, statements = expand_helper(_broadcast_window, (expression, 4, 0, 3, 1), catalog=snapshot(), prefix="window")
  assert {id(v) for v in I.loads_unique(result)} == {id(load)}
  assert load.data is view
  data = np.arange(100, dtype=np.float32)
  reference = Interpreter(I.KernelIR(I.Block(statements), {}, {}), {"Prior": data, "dm": extent})
  reference.block(I.Block(statements))
  actual = reference.value(result)
  start, count = (0, 1) if extent == 1 else (4, 3)
  expected = data[51 + (np.arange(start, start + count) * stride)] * 2
  np.testing.assert_array_equal(actual[:, 0], expected)
  assert sum(v is load for v in walk_unique(result)) == 1


@pytest.mark.parametrize("dm,dn", [(1, 1), (1, 9), (9, 1), (9, 9), (0, 0)])
def test_source_packed_window_does_not_rewrite_physical_abi_origins(dm, dn):
  shape = (_scalar("dm"), _scalar("dn"))
  view = I.View("Packed", "float32", (I.literal(4096),), (I.literal(1),), I.literal(0))
  packed = I.PackedView(view, (I.literal(32), I.literal(32)), "A", "sme", (I.literal(2), I.literal(4)))
  value = I.Expr(I.TileType("float32", shape), "packed_load", data=packed)
  result, statements = expand_helper(
    _broadcast_window,
    (value, 3 if dm else 0, 5 if dn else 0, 2 if dm else 0, 3 if dn else 0),
    catalog=snapshot(),
    prefix="packed_window",
  )
  assert result.args[0] is value and value.data is packed
  reference = Interpreter(I.KernelIR(I.Block(statements), {}, {}), {"dm": dm, "dn": dn})
  reference.block(I.Block(statements))
  assert tuple(reference.value(v) for v in result.type.shape) == (
    1 if dm == 1 else 2 if dm else 0,
    1 if dn == 1 else 3 if dn else 0,
  )


def _scalar(name):
  return I.Expr(I.ScalarType("int64"), "symbol", data=name)


def _evaluate(value, variables):
  if value.op == "literal":
    return value.data
  if value.op == "local":
    return variables[value.data]
  if value.op == "where":
    return _evaluate(value.args[1 if _evaluate(value.args[0], variables) else 2], variables)
  lhs, rhs = (_evaluate(argument, variables) for argument in value.args)
  return {"add": lambda: lhs + rhs, "mul": lambda: lhs * rhs, "eq": lambda: lhs == rhs}[value.op]()


@pytest.mark.parametrize("shape", (None, (64, 1, 257)))
def test_cache_seed_source_retains_runtime_prior_extent_and_origin_guard(shape):
  source = (
    CPUBackend()
    .lower(
      _template(shape=shape, level="blocked").ir,
      (),
      target="AVX512",
      options={"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"},
    )
    .source
  )
  dm = parameter_snapshot_name(source, "D.size(0)")
  # Constant-M induction can fold the broadcast choice for the first block.
  if shape is None:
    assert re.search(rf"\({dm}\) == 1 \? 0 :", source)
  else:
    assert dm in source
  assert "dsl_recipe_dot_" in source
  assert "matmul_blocks(" not in source


@pytest.mark.parametrize("shape,vl,svl", (((64, 1, 257), 64, 64), (None, 16, 64), (None, 64, 16)))
def test_real_cross_dynamic_prior_windows_keep_inline_sme_owners(tmp_path, shape, vl, svl):
  source = _lower(shape=shape, level="blocked", implementation="sme", vl=vl, svl=svl)
  assembly = _cross(tmp_path, source, lambda value: _wrapper(value, shape=shape), vl=vl, svl=svl)
  owners = _owners(assembly, vl=vl, svl=svl)
  assert owners
  # The BK cache driver also emits raw narrow-input ZA packing owners.
  # Validate those roles separately while requiring real matrix owners.
  _assert_inlined_owners(assembly, len(owners), vl=vl, svl=svl, allow_packing=True)
  assert all(not re.search(r"^\s*(?:str|ldr|stp|ldp)\s+(?:z\d|p\d)", owner, re.MULTILINE) for owner in owners)


def test_native_runtime_singleton_prior_cache_driver_strict_oracle(tmp_path):
  _strict_native_fixture(tmp_path)


def _dense_template(level):
  M, N, K, DM, DN = (T.dynamic(lower=0) for _ in range(5))
  D0, D1, C0, C1 = (T.dynamic() for _ in range(4))

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="float32"),
    B: T.Tensor((K, N), dtype="float32"),
    D: T.Tensor((DM, DN), (D0, D1), "float32"),
    C: T.Tensor((M, N), (C0, C1), "float32", access="output"),
  ):
    product = (
      T.library.matmul((T.load(A), T.load(B)), C, acc=T.load(D), packing="none", swap=False, BK=3)
      if level == "blocked"
      else T.library.matmul((T.load(A), T.load(B)), C, acc=T.load(D), packing="none", swap=False)
      if level == "matmul"
      else T.dot((T.load(A), T.load(B)), C, acc=T.load(D))
    )
    T.store(C, T.max(product * 0.5 + 0.125, 0))

  return kernel


def _strided(shape, negative):
  m, n = shape
  value = np.ndarray(
    shape, dtype="float32", buffer=np.empty(max(1, m * (2 * n + 3)), dtype="float32"), strides=((2 * n + 3) * 4, 8)
  )
  return value[::-1, ::-1] if negative else value


def test_native_dense_broadcast_both_axes_nonunit_negative_prior_and_alias(tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernels = {
    level: T.compile(
      _dense_template(level),
      compiler=compiler,
      pass_options={"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"},
    )
    for level in ("dot", "matmul", "blocked")
  }
  with vecops.collect_compile_requests() as collector:
    for kernel in kernels.values():
      kernel.operator(
        vecops.TensorMeta((4, 37), dtype="float32"),
        vecops.TensorMeta((37, 9), dtype="float32"),
        vecops.TensorMeta((4, 9), strides=(21, 2), dtype="float32"),
        vecops.TensorMeta((4, 9), strides=(21, 2), dtype="float32"),
        **kernel._core.specialization_values,
      )
  assert len(collector.requests) == 3
  vecops.compile_batch(collector.requests, parallelism=3, execution_parallelism=1)
  rng = np.random.default_rng(492)
  calls = 0
  for n in (1, 9):
    a = (rng.integers(-2, 3, size=(4, 37)) / 8).astype("float32")
    b = (rng.integers(-2, 3, size=(37, n)) / 8).astype("float32")
    for prior_shape, alias in (((1, 1), False), ((1, n), False), ((4, 1), False), ((4, n), False), ((4, n), True)):
      for negative in (False, True):
        for level, kernel in kernels.items():
          d = _strided(prior_shape, negative)
          d[:] = rng.integers(-2, 3, size=d.shape) / 8
          c = d if alias else _strided((4, n), negative)
          prior = d.copy()
          expected = np.maximum(((a @ b) + prior) * np.float32(0.5) + np.float32(0.125), 0)
          kernel.operator.native(a, b, d, c, **kernel._core.specialization_values)
          np.testing.assert_array_equal(c, expected, err_msg=f"{n}/{prior_shape}/{alias}/{negative}/{level}")
          calls += 1
  assert calls == 60
