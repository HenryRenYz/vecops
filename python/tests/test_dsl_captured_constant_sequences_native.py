"""Native execution regression for captured tuple-valued DSL parameters."""

from __future__ import annotations

import numpy as np
import vecops.language as T
from dsl_test_utils import make_compiler


def test_native_captured_transpose_axes_and_tile_reshape(tmp_path):
  axes = (1, 0)
  shape = (3, 2)
  compiler = make_compiler(
    target="Native",
    output="quiet",
    jobs=2,
    cache_dir=tmp_path / "cache",
    build_dir=tmp_path / "build",
  )

  @T.template
  def kernel(
    A: T.Tensor((2, 3), dtype="float32"),
    C: T.Tensor((3, 2), dtype="float32", access="output"),
  ):
    transposed = T.load(T.transpose(A, axes=axes))
    reshaped = T.reshape(T.load(A), shape)
    T.store(C, transposed + reshaped)

  compiled = T.compile(kernel, compiler=compiler)
  source = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
  output = np.empty((3, 2), dtype=np.float32)
  compiled(source, output)

  np.testing.assert_array_equal(output, source.T + source.reshape(shape))
