# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Direct MOPA source algorithms retain strided views and checked ownership."""

import os

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.dtypes import ELEMENT_BYTES
from vecops.language._debug.interpreter import Interpreter
from vecops.language._library.sme_matmul import direct_dot


def template(dtype="float32", *, dot=False):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), dtype),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), dtype),
    C: T.Tensor((m, n), (T.dynamic(), 1), dtype, access="output"),
  ):
    if dot:
      T.dot((A, B), C, implementation="sme")
    else:
      direct_dot(A, B, C)

  return kernel


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_direct_mopa_algorithm_uses_source_control_and_strided_vector_loads(dtype):
  artifact = CPUBackend().lower(template(dtype).ir, (), target="SME")
  assert not artifact.backend_ir.matmuls
  assert artifact.backend_ir.hardware
  assert "vecops::vec::sme::mopa<0" in artifact.source
  assert "vecops::vec::opt::strided" in artifact.source
  assert "vecops::vec::sme::store_hor<0>" in artifact.source
  assert "pack_offset" not in artifact.source.partition("void __kernel__")[2]
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_raw_dot_dispatch_stages_direct_mopa_as_dsl_control(dtype):
  artifact = CPUBackend().lower(template(dtype, dot=True).ir, (), target="SME")
  assert not artifact.backend_ir.matmuls
  assert "staged 1 direct SME dot" in "\n".join(artifact.kernel.diagnostics)
  assert "dsl_recipe_sme_dot_" in artifact.source
  assert "sme::mopa<0" in artifact.source


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_source_za_preflight_accepts_clipped_and_empty_tiles(dtype):
  artifact = CPUBackend().lower(template(dtype).ir, (), target="SME")
  plan = next(iter(artifact.backend_ir.hardware.values()))
  for streaming_bytes in (16, 32, 64, 128):

    class MetadataInterpreter(Interpreter):
      native_bytes = streaming_bytes

      def value(self, expression):
        if isinstance(expression, I.Expr) and expression.op == "streaming_lanes":
          return self.native_bytes // ELEMENT_BYTES[expression.data]
        return super().value(expression)

    for m, n, k in ((0, 19, 17), (19, 0, 17), (7, 19, 0), (1, 1, 17), (7, 19, 17), (33, 65, 31)):
      reference = MetadataInterpreter(
        artifact.kernel,
        {"A": np.empty((m, k), dtype=dtype), "B": np.empty((k, n), dtype=dtype), "C": np.empty((m, n), dtype=dtype)},
      )
      for condition, message in plan.checks:
        assert reference.value(condition), (streaming_bytes, m, n, k, message)


@pytest.mark.skipif(os.uname().machine.lower() not in {"aarch64", "arm64"}, reason="native SME target required")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dot", [False, True])
def test_native_source_mopa_strides_empty_tails_and_overlap(dtype, dot, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(template(dtype, dot=dot), compiler=compiler)
  for m, n, k in [(0, 19, 17), (19, 0, 17), (7, 19, 0), (1, 1, 17), (7, 19, 17), (33, 65, 31)]:
    for reverse in [False, True]:
      a = (np.arange(m * k).reshape(m, k) % 7).astype(dtype) / 16
      b = (np.arange(k * n).reshape(k, n) % 11).astype(dtype) / 16
      if reverse:
        a, b = a[::-1, ::-1], b[::-1, ::-1]
      output = np.full((m + 2, n + 3), -17, dtype=dtype)
      expected = a @ b
      kernel.operator.native(a, b, output[1 : m + 1, 1 : n + 1], **kernel._core.specialization_values)
      np.testing.assert_array_equal(output[1 : m + 1, 1 : n + 1], expected)
      np.testing.assert_array_equal(output[[0, -1]], -17)
      np.testing.assert_array_equal(output[:, [0, -1]], -17)
  a = (np.arange(7 * 23).reshape(7, 23) % 7).astype(dtype) / 16
  b = (np.arange(23 * 19).reshape(23, 19) % 11).astype(dtype) / 16
  expected = a.copy() @ b
  kernel.operator.native(a, b, a[:, :19], **kernel._core.specialization_values)
  np.testing.assert_array_equal(a[:, :19], expected)
