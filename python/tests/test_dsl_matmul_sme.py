"""Execute SME products on target ARM, sharing dynamic variants per dtype.

Run with CC=clang CXX=clang++ VECOPS_TEST_TARGET=NativeFixedStreamingSVE
and an isolated pytest
base directory. Seven raw/dense/SME packing combinations are batch-compiled
once per element triple; dimensions and all strides remain runtime values.
"""

import os
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler


def _sme_available():
  if os.uname().machine.lower() not in {"aarch64", "arm64"}:
    return False
  try:
    features = Path("/proc/cpuinfo").read_text(encoding="utf-8").split()
  except OSError:
    return False
  return "sme" in features and any(flag in features for flag in ("smefa64", "sme_fa64"))


pytestmark = pytest.mark.skipif(not _sme_available(), reason="SME+FA64 execution requires a target ARM CPU")

ELEMENTS = (
  ("float16", "float16", "float32"),
  ("bfloat16", "bfloat16", "float32"),
  ("float32", "float32", "float32"),
  ("float64", "float64", "float64"),
  ("int8", "int8", "int32"),
  ("uint8", "uint8", "int32"),
  ("int8", "uint8", "int32"),
  ("uint8", "int8", "int32"),
)
PACKING = (("raw", ""), ("dense", "A"), ("dense", "B"), ("dense", "AB"), ("sme", "A"), ("sme", "B"), ("sme", "AB"))
SHAPES = (
  (0, 5, 7),
  (5, 0, 7),
  (5, 7, 0),
  (1, 8, 65),
  (8, 1, 65),
  (2, 2, 7),
  (2, 8, 129),
  (8, 2, 129),
  (4, 4, 257),
  (3, 5, 7),
  (16, 16, 32),
  (17, 19, 33),
  (31, 47, 65),
  (64, 64, 64),
  (3, 129, 33),
  (129, 3, 33),
  (33, 65, 35),
)


def _dtype(name):
  if name == "bfloat16":
    pytest.importorskip("ml_dtypes")
  return np.dtype(name)


def _matrix(rng, shape, dtype, order="C"):
  low = 0 if dtype.startswith("uint") else -3
  values = rng.integers(low, 4, size=shape)
  if dtype.startswith("float") or dtype == "bfloat16":
    values = values / 8
  matrix = _empty_matrix(shape, dtype, order)
  matrix[:] = values
  return matrix


def _empty_matrix(shape, dtype, order="C"):
  # NumPy is free to select either contiguous layout on a singleton axis;
  # empty axes normally receive zero strides. These tests intentionally use
  # schemas with exact (K,1)/(1,K) strides, so construct those strides even
  # for zero/singleton shapes rather than weakening the product's checks.
  rows, columns = shape
  element = _dtype(dtype)
  strides = (
    (columns * element.itemsize, element.itemsize) if order == "C" else (element.itemsize, rows * element.itemsize)
  )
  backing = np.empty(max(1, rows * columns), dtype=element)
  return np.ndarray(shape, dtype=element, buffer=backing, strides=strides)


def _template(ad, bd, acc, format_, sides):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  AS0, AS1, BS0, BS1, DS0, DS1, CS0, CS1 = (T.dynamic() for _ in range(8))
  pack_a, pack_b = "A" in sides, "B" in sides

  @T.template
  def kernel(
    A: T.Tensor((M, K), (AS0, AS1), ad),
    B: T.Tensor((K, N), (BS0, BS1), bd),
    D: T.Tensor((M, N), (DS0, DS1), acc),
    C: T.Tensor((M, N), (CS0, CS1), acc, access="inout"),
    *,
    with_acc: bool = False,
  ):
    a = T.pack(A, side="A", format=format_) if pack_a else T.load(A)
    b = T.pack(B, side="B", format=format_) if pack_b else T.load(B)
    if with_acc:
      T.dot((a, b), C, acc=T.load(D), implementation="sme")
    else:
      T.dot((a, b), C, implementation="sme")

  return kernel


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-matmul-sme")
  return make_compiler(
    target="Native",
    jobs=int(os.environ.get("VECOPS_TEST_JOBS", "8")),
    cache_dir=root / "cache",
    build_dir=root / "build",
  )


@pytest.fixture(scope="module", params=ELEMENTS, ids=lambda x: "-".join(x[:2]))
def products(request, compiler):
  ad, bd, acc = request.param
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for format_, sides in PACKING:
      compiled = T.compile(_template(ad, bd, acc, format_, sides), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((17, 33), dtype=ad),
        vecops.TensorMeta((33, 19), dtype=bd),
        vecops.TensorMeta((17, 19), dtype=acc),
        vecops.TensorMeta((17, 19), dtype=acc),
        with_acc=False,
        **compiled._core.specialization_values,
      )
      kernels[(format_, sides)] = compiled
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  return request.param, kernels


def _run(compiled, a, b, d, c, acc, *, with_acc):
  expected = a.astype(acc) @ b.astype(acc)
  if with_acc:
    expected = expected + d.copy()
  compiled.operator.native(a, b, d, c, with_acc=with_acc, **compiled._core.specialization_values)
  # Binary-fraction inputs keep reference dot products exactly representable
  # for this shape catalog, including all four signedness combinations.
  np.testing.assert_array_equal(c, expected)


def test_sme_formats_shapes_and_all_layout_combinations(products):
  (ad, bd, acc), kernels = products
  for compiled in kernels.values():
    assert_no_vec_mask_arrays(compiled.generated_source)
  rng = np.random.default_rng(73)
  for key, compiled in kernels.items():
    for shape in SHAPES:
      m, n, k = shape
      # Exhaust all eight A/B/C layout combinations for a representative
      # spatial+K tail. Other shapes rotate through direct and gathered axes.
      layouts = (
        tuple(product(("C", "F"), repeat=3))
        if shape == (17, 19, 33)
        else (("C", "C", "C"), ("F", "F", "F"), ("C", "F", "C"))
      )
      for a_order, b_order, c_order in layouts:
        a = _matrix(rng, (m, k), ad, a_order)
        b = _matrix(rng, (k, n), bd, b_order)
        d = _matrix(rng, (m, n), acc, c_order)
        c = np.empty((m, n), dtype=acc, order=c_order)
        for with_acc in (False, True):
          try:
            _run(compiled, a, b, d, c, acc, with_acc=with_acc)
          except AssertionError as error:
            raise AssertionError(
              f"{ad}/{bd}, packing={key}, shape={shape}, layout={(a_order, b_order, c_order)}, acc={with_acc}"
            ) from error
    assert "vecops::vec::sme::mopa" in compiled.generated_source
    assert "vecops::ops::Matmul" not in compiled.generated_source


def test_sme_accumulator_alias_and_broadcast(products):
  (ad, bd, acc), kernels = products
  rng = np.random.default_rng(79)
  for key, compiled in kernels.items():
    for m, n, k in ((1, 19, 13), (19, 1, 13), (17, 19, 33), (33, 65, 35), (5, 7, 0)):
      a, b = _matrix(rng, (m, k), ad), _matrix(rng, (k, n), bd, "F")
      for order in ("C", "F"):
        c = _matrix(rng, (m, n), acc, order)
        try:
          _run(compiled, a, b, c, c, acc, with_acc=True)
          row = _matrix(rng, (1, n), acc)
          d = np.broadcast_to(row, (m, n))
          _run(compiled, a, b, d, c, acc, with_acc=True)
        except AssertionError as error:
          raise AssertionError(f"{ad}/{bd}, packing={key}, shape={(m, n, k)}, C layout={order}") from error


@pytest.mark.parametrize("mode", ("off", "prefetch", "staged"))
def test_sme_fused_broadcast_prologue_epilogue_and_pipeline(compiler, mode):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    D: T.Tensor((1, N), dtype="float32"),
    C: T.Tensor((M, N), (1, M), "float32", access="output"),
  ):
    a = T.cast(T.load(A, dtype="float32") * 2 + 1, "bfloat16")
    value = T.library.matmul((a, B), C, acc=T.load(D), implementation="sme", BM=32, BN=32, BK=16)
    T.store(C, T.max(value * 0.5 + 1, 0))

  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_matmul_pipeline": mode})
  assert_no_vec_mask_arrays(compiled.generated_source)
  rng = np.random.default_rng(83)
  for m, n, k in ((1, 8, 0), (8, 1, 1), (3, 5, 3), (17, 19, 33), (33, 65, 129)):
    a = _matrix(rng, (m, k), "bfloat16")
    b = _matrix(rng, (k, n), "bfloat16", "F")
    d = _matrix(rng, (1, n), "float32")
    c = _empty_matrix((m, n), "float32", "F")
    transformed = (a.astype(np.float32) * 2 + 1).astype(_dtype("bfloat16")).astype(np.float32)
    expected = np.maximum((transformed @ b.astype(np.float32) + d) * 0.5 + 1, 0)
    compiled.operator.native(a, b, d, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, expected)


def test_sme_runtime_cache_bypass_keeps_accumulator_and_epilogue(compiler):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    D: T.Tensor((M, N), dtype="float32"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    value = T.library.matmul((A, B), C, acc=T.load(D), implementation="sme", packing="none")
    T.store(C, T.max(value * 0.5 + 1, 0))

  compiled = T.compile(kernel, compiler=compiler)
  assert_no_vec_mask_arrays(compiled.generated_source)
  assert "bypass cache panels for small runtime SME products" in compiled.generated_source
  rng = np.random.default_rng(89)
  # The same Dynamic artifact visits zero M/N/K, the small whole-problem
  # branch and the large cache-panel branch (including repeated K phases).
  for m, n, k in ((0, 5, 7), (5, 0, 7), (5, 7, 0), (16, 16, 8), (19, 23, 17), (129, 129, 513)):
    a = _matrix(rng, (m, k), "bfloat16")
    b = _matrix(rng, (k, n), "bfloat16", "F")
    d = _matrix(rng, (m, n), "float32")
    expected = np.maximum((a.astype(np.float32) @ b.astype(np.float32) + d) * 0.5 + 1, 0)
    for alias in (False, True):
      c = _empty_matrix((m, n), "float32")
      if alias:
        c[:] = d
      compiled.operator.native(a, b, c if alias else d, c, **compiled._core.specialization_values)
      np.testing.assert_array_equal(c, expected)
