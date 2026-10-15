# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Runtime quantization dispatch preserves the exact affine conversion DAG."""

import pytest
import vecops.language as T
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(fused=False, public_helper=False):
  m, n, k, capacity = (T.dynamic(lower=1) for _ in range(4))

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype="float32"),
    BP: T.Tensor((capacity,), dtype="int8"),
    R: T.Tensor((m, 1), dtype="float32"),
    Z: T.Tensor((1,), dtype="int32"),
    D: T.Tensor((m, n), dtype="int32"),
    RS: T.Tensor((m, 1), dtype="float32"),
    CS: T.Tensor((1, n), dtype="float32"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    affine = (
      T.fma(T.load(A), T.load(R), T.load(Z, dtype="float32"))
      if fused
      else T.load(A) * T.load(R) + T.load(Z, dtype="float32")
    )
    a = T.library.quantize_per_row_u8(T.load(A), T.load(R), T.load(Z)) if public_helper else T.cast(affine, "uint8")
    b = T.packed(BP, (k, n), side="B", format="sme")
    value = T.library.matmul((a, b), C, acc=T.load(D), acc_dtype="int32")
    T.store(C, T.cast(value, "float32") * T.load(RS) * T.load(CS))

  return kernel


def test_runtime_quant_exact_affine_route_and_fma_exclusion():
  options = {"_vector_state": VectorState(64, 64), "cpu_runtime_quant_int8": True}
  artifact = CPUBackend().lower(template().ir, (), target="SME", options=options)
  assert "staged exact row-affine U8 quantization with packed I8 grouped GEMV" in artifact.kernel.diagnostics
  assert "runtime quant packed B capacity too small" in artifact.source
  other = CPUBackend().lower(template(True).ir, (), target="SME", options=options)
  assert "staged exact row-affine U8 quantization with packed I8 grouped GEMV" not in other.kernel.diagnostics


def test_runtime_quant_public_helper_uses_the_same_guarded_schedule():
  options = {"_vector_state": VectorState(64, 64), "cpu_runtime_quant_int8": True}
  artifact = CPUBackend().lower(template(public_helper=True).ir, (), target="SME", options=options)
  assert "staged exact row-affine U8 quantization with packed I8 grouped GEMV" in artifact.kernel.diagnostics


@pytest.mark.parametrize("public_helper", [False, True])
def test_native_runtime_quant_recipe_rounding_and_dequantize(tmp_path, public_helper):
  import platform

  import numpy as np
  import pytest
  from dsl_test_utils import make_compiler
  from vecops._arm_vector_state import get_vector_state

  if platform.machine().lower() not in {"aarch64", "arm64"}:
    pytest.skip("automatic runtime quant recipe uses public SME packed ABI")
  span = 2 * get_vector_state().sme_bytes // 4
  compiler = make_compiler(target="Native", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(
    template(public_helper=public_helper), compiler=compiler, pass_options={"cpu_runtime_quant_int8": True}
  )
  for m, n, k in [(1, 35, 7), (2, 17, 65), (3, 3, 1)]:
    a = np.resize(np.asarray([-np.inf, -9.5, -0.1, 0.25, 11.9, 255.9, np.inf, np.nan], np.float32), m * k).reshape(m, k)
    multiplier = np.full((m, 1), 1.25, np.float32)
    zero_point = np.asarray([3], np.int32)
    b = ((np.arange(k * n).reshape(k, n) % 13) - 6).astype(np.int8)
    packed = np.full(((n + span - 1) // span, (k + 3) // 4, span, 4), 127, np.int8)
    for col in range(n):
      for kk in range(k):
        packed[col // span, kk // 4, col % span, kk % 4] = b[kk, col]
    correction = np.full((m, n), 2147483500, np.int32)
    rs = np.full((m, 1), 0.375, np.float32)
    cs = np.linspace(0.1, 1.0, n, dtype=np.float32).reshape(1, n)
    output = np.empty((m, n), np.float32)
    with np.errstate(invalid="ignore"):
      affine = a * multiplier + zero_point.astype(np.float32)
      q = np.clip(np.where(np.isnan(affine), 0, affine), 0, 255).astype(np.uint8)
    total = q.astype(np.int64) @ b.astype(np.int64) + correction.astype(np.int64)
    expected = (total & 0xFFFFFFFF).astype(np.uint32).view(np.int32).astype(np.float32) * rs * cs
    compiled.operator.native(
      a, packed.reshape(-1), multiplier, zero_point, correction, rs, cs, output, **compiled._core.specialization_values
    )
    np.testing.assert_array_equal(output, expected)
