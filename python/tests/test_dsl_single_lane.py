"""One-lane loads retain dtype conversion, masks and snapshot semantics."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-single-lane")
  return make_compiler(
    target="Native", jobs=4, build_dir=root / "build", cache_dir=root / "cache"
  )


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_scalarized_vector_load_and_snapshot(compiler, dtype):
  descriptor = T.Vector("float32", N=1)

  @T.template
  def kernel(
    A: T.Tensor((17,), dtype=dtype, access="inout"),
    C: T.Tensor((17,), dtype="float32", access="output"),
  ):
    for i in T.serial(17, unroll=2):
      x = T.load(A[i : i + 1], dtype="float32", vector=descriptor)
      T.store(A[i : i + 1], x + 2)
      T.store(C[i : i + 1], x * 3)

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(17, dtype=getattr(np, dtype))
  c = np.empty(17, np.float32)
  expected = a.astype(np.float32) * 3
  compiled(a, c)
  np.testing.assert_array_equal(c, expected)
  np.testing.assert_array_equal(a, np.arange(17) + 2)


def test_masked_one_lane_does_not_read_inactive(compiler):
  descriptor = T.Vector("float32", N=1)

  @T.template
  def kernel(A: T.Tensor((1,), dtype="float32"), C: T.Tensor((1,), dtype="float32", access="output")):
    T.store(C, T.load(A, vector=descriptor, mask=False, other=7.0))

  compiled = T.compile(kernel, compiler=compiler)
  out = np.empty(1, np.float32)
  compiled(np.array([np.nan], np.float32), out)
  np.testing.assert_array_equal(out, [7])
