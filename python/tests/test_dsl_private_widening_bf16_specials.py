# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Private explicit vector recipes keep grouped BF16 tails and special values."""

from pathlib import Path

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler


def template():
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.func(inline=False)
  def grouped(a, b, c):
    T.dot((a, b), c, implementation="vector")

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype="bfloat16"),
    B: T.Tensor((k, n), (1, k), "bfloat16"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    grouped(A, B, C)

  return kernel


def test_private_vector_specials_kernel_keeps_grouped_dot():
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  artifact = CPUBackend().lower(template().ir, (), target="AVX512")
  assert "staged AMX matrix instruction recipe" not in "\n".join(artifact.kernel.diagnostics)
  assert "vecops::vec::widening_dot(" in artifact.source
  assert artifact.source.count("vecops_dsl::private_call(") == 1


@pytest.fixture(scope="module", params=(False, True), ids=("default", "preserve-subnormals"))
def compiled(request, tmp_path_factory):
  if "avx512_bf16" not in Path("/proc/cpuinfo").read_text():
    pytest.skip("requires AVX512 BF16 grouped-dot instruction")
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  root = tmp_path_factory.mktemp("private-widening-bf16")
  preserve = request.param
  compiler = make_compiler(
    jobs=1, cache_dir=root / "cache", build_dir=root / "build",
    cflags=["-DVECOPS_PRESERVE_SUBNORMALS=1"] if preserve else [],
  )
  kernel = T.compile(template(), compiler=compiler)
  return kernel, bf16, preserve


def arguments(bf16, k, n):
  # Explicit strides also retain the K-contiguous B contract when K is zero;
  # NumPy's ordinary empty array otherwise reports all-zero strides.
  a_backing = np.empty(max(1, k), dtype=bf16)
  b_backing = np.empty(max(1, k * n), dtype=bf16)
  a = np.ndarray((1, k), dtype=bf16, buffer=a_backing, strides=(k * 2, 2))
  b = np.ndarray((k, n), dtype=bf16, buffer=b_backing, strides=(2, k * 2))
  guarded = np.full(n + 2, -9876, dtype=np.float32)
  return a, b, guarded[1:-1].reshape(1, n), guarded


def invoke(kernel, a, b, c, guarded):
  # Exercise the explicit vector recipe inside the user's private helper.
  kernel.operator.native(a, b, c, **kernel._core.specialization_values)
  np.testing.assert_array_equal(guarded[[0, -1]], -9876)


def test_private_grouped_bf16_empty_and_partial_groups(compiled):
  kernel, bf16, _ = compiled
  for k in (0, 1, 2, 3, 17, 33, 65):
    a, b, c, guarded = arguments(bf16, k, 7)
    a[:] = 1
    b[:] = (np.arange(k * 7).reshape(k, 7) % 9 - 4) / 8
    expected = a.astype(np.float64) @ b.astype(np.float64)
    invoke(kernel, a, b, c, guarded)
    np.testing.assert_array_equal(c, expected.astype(np.float32))


def test_private_grouped_bf16_active_nan_and_infinities(compiled):
  kernel, bf16, _ = compiled
  a, b, c, guarded = arguments(bf16, 17, 7)
  a[:] = 1
  a[0, 2] = 0
  b[:] = 0
  b[:, 0] = 0.5
  b[0, 1] = np.nan
  b[0, 2] = np.inf
  b[0, 3] = -np.inf
  b[0, 4], b[1, 4] = np.inf, -np.inf
  b[2, 5] = np.inf  # Active 0 * Inf must remain NaN, not a masked tail zero.
  invoke(kernel, a, b, c, guarded)
  np.testing.assert_array_equal(np.isnan(c[0]), [False, True, False, False, True, True, False])
  assert np.isposinf(c[0, 2]) and np.isneginf(c[0, 3])
  np.testing.assert_array_equal(c[0, [0, 6]], [8, 0])


def test_private_grouped_bf16_subnormal_policy(compiled):
  kernel, bf16, preserve = compiled
  # The preserve build's generic FP32 path still obeys the caller's FP state.
  # Skip this precision check if that thread already flushes FP32 inputs.
  if np.float32(2.0 ** -133) * np.float32(1) == 0:
    pytest.skip("caller FP environment flushes FP32 subnormal inputs")
  minimum = np.asarray([1], dtype=np.uint16).view(bf16)[0]
  for k in (1, 3, 17):
    a, b, c, guarded = arguments(bf16, k, 2)
    a[:] = minimum
    b[:, 0], b[:, 1] = 2.0 ** 100, -(2.0 ** 100)
    # BF16 dot flushes its subnormal inputs. The preserve build disables that
    # ISA specialization and retains this exactly representable normal sum.
    expected = k * (2.0 ** -33) if preserve else 0
    invoke(kernel, a, b, c, guarded)
    np.testing.assert_array_equal(c, np.asarray([[expected, -expected]], dtype=np.float32))
