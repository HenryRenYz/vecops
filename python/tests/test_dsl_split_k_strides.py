# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Output-backed split-K preserves its physical column stride on every path.

The original row failure used the vector outer-product continuation, not an
AMX tile. Separate full/fringe AMX gates cover the independently found direct
tile and vector-staging assumptions. No packing/family threshold is changed.
"""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import amx_available_for_test, make_compiler
from test_dsl_sme_packed_col import _pack_template, _template
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def _lower(*, dynamic, implementation):
  return (
    CPUBackend()
    .lower(
      _template(
        shape=None if dynamic else (17, 17, 65),
        format_="auto",
        transformed=True,
        level="blocked",
        implementation=implementation,
      ).ir,
      (),
      target="AVX512",
      options={"cpu_matmul_fusion": "always", "cpu_matmul_pipeline": "off"},
    )
    .source
  )


def _matrix(shape, step):
  m, n = shape
  backing = np.full((m, n * abs(step) + 3), np.float32(37), dtype="float32")
  value = backing[:, : n * abs(step) : abs(step)]
  if step < 0:
    value = value[:, ::-1]
  return value, backing


def test_native_amx_split_k_output_backed_full_and_fringe_column_strides(tmp_path):
  if not amx_available_for_test():
    pytest.skip("explicit AMX BF16 numerical gate requires the matching native x86 target")
  pytest.importorskip("ml_dtypes")
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  options = {"cpu_matmul_fusion": "always", "cpu_matmul_pipeline": "off"}
  count = 48 * 96  # Native AMX B: 16 spatial lanes, K32; N<=33/K<=65.
  pack = T.compile(_pack_template("bfloat16"), compiler=compiler, pass_options=options)
  kernels = {
    dynamic: T.compile(
      _template(
        shape=None if dynamic else (17, 17, 65),
        format_="auto",
        transformed=True,
        level="blocked",
        implementation="amx",
      ),
      compiler=compiler,
      pass_options=options,
    )
    for dynamic in (False, True)
  }
  with vecops.collect_compile_requests() as collector:
    pack.operator(
      vecops.TensorMeta((65, 33), strides=(1, 65), dtype="bfloat16"),
      vecops.TensorMeta((count,), dtype="bfloat16"),
      **pack._core.specialization_values,
    )
    for kernel in kernels.values():
      kernel.operator(
        vecops.TensorMeta((17, 65), strides=(135, 2), dtype="float32"),
        vecops.TensorMeta((65, 17), strides=(1, 65), dtype="bfloat16"),
        vecops.TensorMeta((count,), dtype="bfloat16"),
        vecops.TensorMeta((17, 17), strides=(37, 2), dtype="float32"),
        vecops.TensorMeta((17, 17), strides=(37, 2), dtype="float32"),
        vecops.TensorMeta((1, 17), dtype="float32"),
        **kernel._core.specialization_values,
      )
  assert len(collector.requests) == 3
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  rng = np.random.default_rng(445)
  for dynamic in (False, True):
    shapes = ((17, 17, 65), (16, 16, 32), (33, 33, 35), (1, 17, 33), (17, 17, 0)) if dynamic else ((17, 17, 65),)
    for m, n, k in shapes:
      b = np.ndarray((k, n), dtype="bfloat16", buffer=np.empty(max(1, k * n), dtype="bfloat16"), strides=(2, 2 * k))
      b[:] = rng.integers(-3, 4, size=b.shape) / 8
      bp = np.empty(count, dtype="bfloat16")
      pack.operator.native(b, bp, **pack._core.specialization_values)
      for step in (1, 2, -2):
        a = rng.integers(-4, 5, size=(m, 2 * k + 5)).astype("float32") / np.float32(16)
        a = a[:, : 2 * k : 2]
        if step < 0:
          a = a[:, ::-1]
        transformed = (a.copy() * np.float32(0.5) + np.float32(0.125)).astype("bfloat16").astype("float32")
        product = transformed @ b.astype("float32")
        for alias in (False, True):
          d, _ = _matrix((m, n), step)
          d[:] = rng.integers(-2, 3, size=d.shape) / 8
          c, backing = (d, d.base) if alias else _matrix((m, n), step)
          holes = backing.copy()
          prior = d.copy()
          bias = np.full((1, n), np.float32(0.125), dtype="float32")
          expected = np.maximum(((product + prior) + bias) * np.float32(0.5) + np.float32(0.125), 0)
          kernel = kernels[dynamic]
          kernel.operator.native(a, b, bp, d, c, bias, **kernel._core.specialization_values)
          np.testing.assert_array_equal(c, expected, err_msg=f"{(m, n, k)} step={step} alias={alias}")
          # Verify scatter/negative-stride stores never modify physical holes.
          holes[:, : n * abs(step) : abs(step)] = backing[:, : n * abs(step) : abs(step)]
          np.testing.assert_array_equal(backing, holes)
          if not alias:
            np.testing.assert_array_equal(d, prior)
