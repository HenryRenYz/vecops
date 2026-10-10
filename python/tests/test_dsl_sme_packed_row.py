# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""BF16 single-packed B rows use ZA, without changing other skinny routes.

The matched C++ source registration already consumes native SME-packed B in
its spatial ZA family. This is only a dispatch correction, not implicit A
packing, a threshold change, or a change to the PackedAB tiny/MMLA ladder.
"""

import platform

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from test_dsl_sme_packed_col import _pack_template, _template
from vecops._arm_vector_state import VectorState, get_vector_state

_ROW_GATE = "/* single-packed B row uses ZA spatial lanes */"


def _packed_count(*, machine=None, vector_state=None):
  """Capacity for N<=65/K<=257 with the actual public native pack geometry."""
  machine = (machine or platform.machine()).lower()
  if machine in {"aarch64", "arm64"}:
    svl = (vector_state or get_vector_state()).sme_bytes
    if svl is None:
      pytest.skip("single-packed-B row fixture requires a valid current SME vector length")
    panel, group = 2 * svl // 4, 2
  else:
    panel, group = 16, 32
  return ((65 + panel - 1) // panel) * panel * ((257 + group - 1) // group) * group


@pytest.mark.parametrize("svl", (16, 32, 64, 256))
def test_row_fixture_capacity_uses_runtime_svl_and_multiple_spatial_panels(svl):
  count = _packed_count(machine="aarch64", vector_state=VectorState(16, svl))
  panel = svl // 2
  assert count == ((65 + panel - 1) // panel) * panel * 258
  assert count >= 65 * 257


def test_row_fixture_x86_geometry_and_unavailable_sme_are_explicit():
  assert _packed_count(machine="x86_64", vector_state=VectorState(None, None)) == 80 * 288
  with pytest.raises(pytest.skip.Exception, match="valid current SME vector length"):
    _packed_count(machine="aarch64", vector_state=VectorState(16, None))


def _row(n, *, negative=False):
  backing = np.empty(max(1, 2 * n), dtype="float32")
  value = np.ndarray((1, n), dtype="float32", buffer=backing, strides=(2 * n * 4, 8))
  return value[:, ::-1] if negative else value


def _run_native_row_cases(dtype, tmp_path, *, options):
  if dtype == "bfloat16":
    pytest.importorskip("ml_dtypes")
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  constant_shape = (1, 16 if dtype == "float16" else 64, 257)
  count = _packed_count()
  pack = T.compile(_pack_template(dtype), compiler=compiler, pass_options=options)
  kernels = {
    (dynamic, level): T.compile(
      _template(dtype, shape=None if dynamic else constant_shape, format_="auto", transformed=True, level=level),
      compiler=compiler,
      pass_options=options,
    )
    for dynamic in (False, True)
    for level in ("dot", "matmul", "blocked")
  }
  with vecops.collect_compile_requests() as collector:
    pack.operator(
      vecops.TensorMeta((257, 65), strides=(1, 257), dtype=dtype),
      vecops.TensorMeta((count,), dtype=dtype),
      **pack._core.specialization_values,
    )
    for (dynamic, _), kernel in kernels.items():
      n = 64 if dynamic else constant_shape[1]
      kernel.operator(
        vecops.TensorMeta((1, 257), strides=(519, 2), dtype="float32"),
        vecops.TensorMeta((257, n), strides=(1, 257), dtype=dtype),
        vecops.TensorMeta((count,), dtype=dtype),
        vecops.TensorMeta((1, n), strides=(2 * n, 2), dtype="float32"),
        vecops.TensorMeta((1, n), strides=(2 * n, 2), dtype="float32"),
        vecops.TensorMeta((1, n), dtype="float32"),
        **kernel._core.specialization_values,
      )
  assert len(collector.requests) == 7
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  rng = np.random.default_rng(441)
  for dynamic in (False, True):
    shapes = (
      (constant_shape,) if not dynamic else ((1, 0, 7), (1, 32, 0), (1, 16, 1), (1, 32, 5), (1, 64, 257), (1, 65, 9))
    )
    for m, n, k in shapes:
      b = np.ndarray(
        (k, n),
        dtype=dtype,
        buffer=np.empty(max(1, k * n), dtype=dtype),
        strides=(np.dtype(dtype).itemsize, k * np.dtype(dtype).itemsize),
      )
      b[:] = rng.integers(-3, 4, size=b.shape) / 8
      bp = np.empty(count, dtype=dtype)
      pack.operator.native(b, bp, **pack._core.specialization_values)
      for negative in (False, True):
        a = rng.integers(-4, 5, size=(m, 2 * k + 5)).astype("float32")
        a /= np.float32(16)
        a = a[:, : 2 * k : 2]
        if negative:
          a = a[:, ::-1]
        transformed = (a.copy() * np.float32(0.5) + np.float32(0.125)).astype(dtype).astype("float32")
        product = transformed @ b.astype("float32")
        for alias in (False, True):
          for level in ("dot", "matmul", "blocked"):
            d = _row(n, negative=negative)
            d[:] = rng.integers(-2, 3, size=d.shape) / 8
            c = d if alias else _row(n, negative=negative)
            bias = np.full((1, n), np.float32(0.125), dtype="float32")
            prior = d.copy()
            expected = np.maximum(((product + prior) + bias) * np.float32(0.5) + np.float32(0.125), 0)
            kernel = kernels[dynamic, level]
            kernel.operator.native(a, b, bp, d, c, bias, **kernel._core.specialization_values)
            np.testing.assert_array_equal(
              c, expected, err_msg=f"{dtype}/{level} {(m, n, k)} negative={negative} alias={alias}"
            )


@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_native_single_packed_b_row_strict_fusion_prior_alias_and_tails(dtype, tmp_path):
  _run_native_row_cases(dtype, tmp_path, options={"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"})
