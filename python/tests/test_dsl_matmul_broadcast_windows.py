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
from vecops.language._compiler.backends.cpu.cpp.matmul import MatmulEmitter


def _scalar(name):
  return I.Expr(I.ScalarType("int64"), "local", data=name)


def _evaluate(value, variables):
  if value.op == "literal":
    return value.data
  if value.op == "local":
    return variables[value.data]
  if value.op == "where":
    return _evaluate(value.args[1 if _evaluate(value.args[0], variables) else 2], variables)
  lhs, rhs = (_evaluate(argument, variables) for argument in value.args)
  return {"add": lambda: lhs + rhs, "mul": lambda: lhs * rhs, "eq": lambda: lhs == rhs}[value.op]()


class _WindowEmitter(MatmulEmitter):
  def __init__(self):
    self.materialized = {}
    self.snapshots = {}

  def retain_expression(self, value):
    pass


@pytest.mark.parametrize("extent", (1, 9))
@pytest.mark.parametrize("stride", (5, -5))
def test_runtime_and_const_axis_origin_extent_agree_without_out_of_bounds(extent, stride):
  emitter = _WindowEmitter()
  view = I.View("Prior", "float32", (_scalar("dm"), I.literal(1)), (I.literal(stride), I.literal(1)), I.literal(31))
  panel = emitter.slice_matrix_view(view, (I.literal(4), I.literal(0)), (I.literal(3), I.literal(1)))
  assert _evaluate(panel.offset, {"dm": extent}) == 31 + (0 if extent == 1 else 4 * stride)
  assert _evaluate(panel.shape[0], {"dm": extent}) == (1 if extent == 1 else 3)
  for known in (1, 9):
    origin, size = emitter.slice_matrix_axis(I.literal(known), I.literal(4), I.literal(3))
    assert I.constant(origin) == (0 if known == 1 else 4)
    assert I.constant(size) == (1 if known == 1 else 3)


@pytest.mark.parametrize("extent", (1, 9))
def test_sliced_dag_history_and_materialization_keep_identity_and_singletons(extent):
  emitter = _WindowEmitter()
  shape = (_scalar("dm"), I.literal(1))
  original = I.View("Prior", "float32", shape, (I.literal(-5), I.literal(1)), I.literal(17))
  cached = I.View("Snapshot", "float32", shape, (I.literal(1), I.literal(1)), I.literal(0), owned=True)
  load = I.Expr(I.TileType("float32", shape), "load", data=original)
  expression = I.binary("add", load, load)
  emitter.snapshots[id(load)] = cached, "alias_prior"
  emitter.materialized[id(load)] = cached, "cache_enabled"
  sliced = emitter.slice_matrix_value(expression, (I.literal(4), I.literal(0)), (I.literal(3), I.literal(1)))
  left, right = sliced.args
  assert left is right and left is not load
  assert _evaluate(left.type.shape[0], {"dm": extent}) == (1 if extent == 1 else 3)
  assert _evaluate(left.data.offset, {"dm": extent}) == (17 if extent == 1 else -3)
  for mapping, condition in ((emitter.snapshots, "alias_prior"), (emitter.materialized, "cache_enabled")):
    view, enabled = mapping[id(left)]
    assert enabled == condition and view.storage == "Snapshot"
    assert _evaluate(view.offset, {"dm": extent}) == (0 if extent == 1 else 4)
    assert _evaluate(view.shape[0], {"dm": extent}) == (1 if extent == 1 else 3)
    assert mapping[id(load)] == (cached, condition)  # Historical binding is not overwritten.


@pytest.mark.parametrize("extent", (1, 9))
def test_runtime_singleton_arange_uses_zero_panel_origin(extent):
  emitter = _WindowEmitter()
  value = I.Expr(I.TileType("int64", (_scalar("dm"), I.literal(1))), "arange", (I.literal(7),), data=0)
  sliced = emitter.slice_matrix_value(value, (I.literal(4), I.literal(0)), (I.literal(3), I.literal(1)))
  assert _evaluate(sliced.args[0], {"dm": extent}) == (7 if extent == 1 else 11)


@pytest.mark.parametrize("dm,dn", ((1, 1), (1, 9), (9, 1), (9, 9), (0, 0)))
def test_packed_capture_preserves_original_window_origin_and_both_broadcast_axes(dm, dn):
  emitter = _WindowEmitter()
  shape = (_scalar("dm"), _scalar("dn"))
  view = I.View("Packed", "float32", (I.literal(4096),), (I.literal(1),), I.literal(0))
  packed = I.PackedView(view, (I.literal(32), I.literal(32)), "A", "sme", (I.literal(2), I.literal(4)))
  value = I.Expr(I.TileType("float32", shape), "packed_load", data=packed)
  origin = (I.literal(3 if dm else 0), I.literal(5 if dn else 0))
  extent = (I.literal(2 if dm else 0), I.literal(3 if dn else 0))
  sliced = emitter.slice_matrix_value(value, origin, extent)
  variables = {"dm": dm, "dn": dn}
  assert tuple(_evaluate(v, variables) for v in sliced.type.shape) == (
    1 if dm == 1 else 2 if dm else 0,
    1 if dn == 1 else 3 if dn else 0,
  )
  assert tuple(_evaluate(v, variables) for v in sliced.data.origins) == (
    2 + (0 if dm == 1 else 3 if dm else 0),
    4 + (0 if dn == 1 else 5 if dn else 0),
  )
  assert sliced.data.shape == packed.shape and sliced.data.view is packed.view


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
  assert re.search(rf"{dm} == std::int64_t\{{1\}}\) \? std::int64_t\{{0\}} : \w+_block_m_", source)
  assert re.search(rf"{dm} == std::int64_t\{{1\}}\) \? std::int64_t\{{1\}} : \w+_panel_m_", source)
  assert "prologue exactly once" in source and "epilogue exactly once" in source


@pytest.mark.parametrize("shape,vl,svl", (((64, 1, 257), 64, 64), (None, 16, 64), (None, 64, 16)))
def test_real_cross_dynamic_prior_windows_keep_inline_sme_owners(tmp_path, shape, vl, svl):
  source = _lower(shape=shape, level="blocked", implementation="sme", vl=vl, svl=svl)
  assembly = _cross(tmp_path, source, lambda value: _wrapper(value, shape=shape), vl=vl, svl=svl)
  owners = _owners(assembly, vl=vl, svl=svl)
  assert owners
  _assert_inlined_owners(assembly, len(owners), vl=vl, svl=svl)
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
