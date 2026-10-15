# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Architecture orientation keeps conventional DSL B coordinates and proofs."""

import platform
from types import SimpleNamespace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.environment import freeze
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._compiler.passes.kernel.matmul import orient_matmul


def template(m, n, k, at, bt, ct):
  @T.template
  def kernel(A: T.Tensor((m, k), (1, m) if at else (k, 1), "bfloat16"),
             B: T.Tensor((k, n), (1, k) if bt else (n, 1), "bfloat16"),
             C: T.Tensor((m, n), (1, m) if ct else (n, 1), "float32", access="output")):
    T.library.matmul((A, B), C)
  return kernel


@pytest.mark.parametrize("shape,layout,swap", [
  ((64, 128, 16), (True, True, False), True),
  ((4, 128, 16), (True, True, False), False),
  ((64, 128, 256), (True, True, True), True),
  ((64, 128, 256), (False, True, True), True),
  ((128, 64, 256), (False, True, True), False),
  ((64, 128, 256), (False, False, True), False),
])
def test_amx_layout_shape_policy(shape, layout, swap):
  # Stop after orientation to inspect geometry independently of recipes.
  options = {"cpu_matmul_orientation": "target"}
  environment = freeze(options, target="AVX512")
  kernel = orient_matmul(lower(template(*shape, *layout).ir), SimpleNamespace(options=options, environment=environment))
  products = [s.value for s in nodes(kernel.body) if isinstance(s, I.Store) and s.value.op == "dot"]
  assert products
  expected = shape[:2][::-1] if swap else shape[:2]
  assert tuple(map(I.constant, products[-1].type.shape)) == expected


@pytest.mark.skipif(platform.machine().lower() not in {"x86_64", "amd64"}, reason="AMX orientation")
@pytest.mark.parametrize("ct", [False, True])
def test_native_amx_orientation_rectangular_provenance(ct, tmp_path):
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  a = (np.arange(48 * 32).reshape(32, 48).T % 7 / 8).astype(bf16)
  b = (np.arange(32 * 64).reshape(64, 32).T % 5 / 8).astype(bf16)
  c = np.empty((64, 48), np.float32).T if ct else np.empty((48, 64), np.float32)
  compiled = T.compile(template(48, 64, 32, True, True, ct),
                       compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"),
                       pass_options={"cpu_matmul_orientation": "target"})
  compiled.operator.native(a, b, c, **compiled._core.specialization_values)
  np.testing.assert_array_equal(c, a.astype(np.float32) @ b.astype(np.float32))


def test_column_orientation_promotes_rank_one_bias_provenance(tmp_path):
  @T.template
  def kernel(A: T.Tensor((17, 3), dtype="float32"), B: T.Tensor((3, 19), dtype="float32"),
             Bias: T.Tensor((2, 19), dtype="float32"),
             C: T.Tensor((17, 19), (1, 17), "float32", access="output")):
    product = T.library.matmul((A, B), C, implementation="vector")
    T.store(C, product + T.load(Bias[1, :]))

  compiled = T.compile(kernel, compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  a = np.arange(51, dtype=np.float32).reshape(17, 3) / 8
  b = np.arange(57, dtype=np.float32).reshape(3, 19) / 8
  bias = np.arange(38, dtype=np.float32).reshape(2, 19)
  output = np.empty((19, 17), np.float32).T
  compiled.operator.native(a, b, bias, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, a @ b + bias[1])
