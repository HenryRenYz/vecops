"""Empty dynamic matrices must not enter an invalid small AMX configuration."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import amx_available_for_test, make_compiler


@pytest.mark.parametrize("columns", (0, 1, 7))
def test_dense_empty_prior_matches_declared_layout(columns):
  prior = np.empty((1, columns), np.float32)
  prior[0] = np.arange(columns, dtype=np.float32)
  view = vecops._C.TensorView.from_numpy(prior)
  assert view.sizes == [1, columns]
  assert view.strides == [columns, 1]
  if columns == 0:
    # Explicit empty layouts remain intact; do not relax the binding contract.
    explicit = np.arange(0, dtype=np.float32).reshape(1, 0)
    assert vecops._C.TensorView.from_numpy(explicit).strides == [1, 1]


@pytest.mark.skipif(not amx_available_for_test(), reason="explicit AMX matmul requires an x86 AMX target")
def test_empty_dynamic_amx_and_zero_k_bias(tmp_path):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), "bfloat16"),
    B: T.Tensor((K, N), (N, 1), "bfloat16"),
    D: T.Tensor((1, N), dtype="float32"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    value = T.dot((A, B), C, acc=T.load(D), implementation="amx")
    T.store(C, value + 1)

  compiler = make_compiler(target="Native", jobs=6, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  import ml_dtypes

  for m, n, k in ((0, 5, 7), (5, 0, 7), (0, 0, 0), (1, 5, 0), (5, 1, 0), (17, 19, 0)):
    a = np.empty((m, k), ml_dtypes.bfloat16)
    # Empty NumPy arrays use canonical dense ABI strides, including when a
    # transposed empty ndarray reports both C/F-contiguous. Declare that layout.
    b = np.empty((k, n), ml_dtypes.bfloat16)
    # reshape(1, 0) preserves nonzero NumPy strides; those are intentionally
    # not canonicalized by the ABI. Allocate the declared dense layout instead.
    d = np.empty((1, n), np.float32)
    d[0] = np.arange(n, dtype=np.float32)
    c = np.empty((m, n), np.float32)
    compiled.operator.native(a, b, d, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, np.broadcast_to(d + 1, c.shape))
  assert "empty products need no resources" in compiled.generated_source
