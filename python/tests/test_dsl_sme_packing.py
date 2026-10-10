# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Raw SME ZA packing: public-instruction codegen and bitwise ARM execution."""

import os
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops._arm_vector_state import get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend

DTYPES = ("float16", "bfloat16", "float32", "float64", "int8", "uint8")
WIDTHS = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int8": 1, "uint8": 1}


def _template(dtype, side):
  S, K, PS = (T.dynamic(lower=0) for _ in range(3))
  SS, KS = T.dynamic(), T.dynamic()
  shape, strides = ((S, K), (SS, KS)) if side == "A" else ((K, S), (KS, SS))

  @T.template
  def kernel(
    X: T.Tensor(shape, strides, dtype),
    P: T.Tensor((PS,), dtype=dtype, access="output"),
  ):
    T.pack(X, P, side=side, format="sme")

  return kernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("side", ("A", "B"))
def test_raw_za_packing_keeps_checks_outside_owner_and_only_named_words(dtype, side):
  source = CPUBackend().lower(_template(dtype, side).ir, (), target="SME").source
  assert_no_vec_mask_arrays(source)
  assert "packed output buffer too small" in source
  body = source.partition("void __kernel__")[2]
  assert "dsl_pack_" in body
  assert "pack_offset" in body
  assert ".with_resources(" not in body
  assert "vecops::ops::MatmulPack" not in source


def test_transformed_pack_does_not_claim_raw_za_bitwise_route():
  @T.template
  def kernel(X: T.Tensor((19, 17), dtype="float32"), P: T.Tensor((4096,), dtype="bfloat16", access="output")):
    T.pack(T.load(X) + 1, P, dtype="bfloat16", format="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert "raw K-contiguous SME ZA packing" not in source
  assert "pack_group" in source


def _available():
  if os.uname().machine.lower() not in {"aarch64", "arm64"}:
    return False
  try:
    features = Path("/proc/cpuinfo").read_text(encoding="utf-8").split()
  except OSError:
    return False
  return "sme" in features and any(flag in features for flag in ("smefa64", "sme_fa64"))


def _dtype(dtype):
  if dtype == "bfloat16":
    pytest.importorskip("ml_dtypes")
  return np.dtype(dtype)


def _expected(logical, panel, kp):
  s, k = logical.shape
  output = np.zeros(((s + panel - 1) // panel, (k + kp - 1) // kp, panel, kp), dtype=logical.dtype)
  for row in range(s):
    for column in range(k):
      output[row // panel, column // kp, row % panel, column % kp] = logical[row, column]
  return output.reshape(-1)


@pytest.mark.skipif(not _available(), reason="SME+FA64 bitwise packing requires a target ARM CPU")
@pytest.mark.parametrize("dtype", DTYPES)
def test_raw_za_pack_bitwise_full_tail_negative_space_stride_and_alias(dtype, tmp_path):
  state = get_vector_state()
  assert state.sme_bytes is not None
  width = WIDTHS[dtype]
  panel, kp = 2 * state.sme_bytes // (8 if width == 8 else 4), max(1, 4 // width)
  chunk = state.sme_bytes // width
  shapes = (
    (0, 5),
    (5, 0),
    (1, 1),
    (panel - 1, kp + 1),
    (panel, chunk),
    (2 * panel + 3, 2 * chunk + 1),
    (panel + 1, 5 * chunk + kp - 1),
  )
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for side in ("A", "B"):
      compiled = T.compile(_template(dtype, side), compiler=compiler)
      shape = (19, 17) if side == "A" else (17, 19)
      compiled.operator(
        vecops.TensorMeta(shape, dtype=dtype),
        vecops.TensorMeta((8192,), dtype=dtype),
        **compiled._core.specialization_values,
      )
      kernels[side] = compiled
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  rng = np.random.default_rng(131)
  bit_dtype, memory_dtype = np.dtype(f"uint{width * 8}"), _dtype(dtype)
  for compiled in kernels.values():
    assert_no_vec_mask_arrays(compiled.generated_source)
  for s, k in shapes:
    # Arbitrary storage payloads include NaNs, signed zeros and every integer
    # signedness. Packing compares bits, not approximate numeric values.
    raw = rng.integers(0, 256, size=s * max(1, 2 * k) * width, dtype=np.uint8)
    backing = raw.view(memory_dtype).reshape(s, max(1, 2 * k))
    for variant in ("dense", "negative", "gather"):
      logical = backing[:, :k] if variant != "gather" else backing[:, : 2 * k : 2]
      if variant == "negative":
        logical = logical[::-1]
      expected = _expected(logical.view(bit_dtype), panel, kp)
      for side, compiled in kernels.items():
        data = logical if side == "A" else logical.T
        output = np.full(expected.size + 14, 0x55, dtype=bit_dtype)
        compiled.operator.native(data, output[7:-7].view(memory_dtype), **compiled._core.specialization_values)
        np.testing.assert_array_equal(output[7:-7], expected)
        np.testing.assert_array_equal(output[:7], 0x55)
        np.testing.assert_array_equal(output[-7:], 0x55)
  # The checked alias snapshot must happen before any ZA writes to output.
  s, k = panel + 1, chunk + 1
  count = ((s + panel - 1) // panel) * ((k + kp - 1) // kp) * panel * kp
  for side, compiled in kernels.items():
    storage = rng.integers(0, 256, size=count * width, dtype=np.uint8).view(memory_dtype)
    logical = storage[: s * k].reshape(s, k)[::-1]
    expected = _expected(logical.view(bit_dtype).copy(), panel, kp)
    compiled.operator.native(logical if side == "A" else logical.T, storage, **compiled._core.specialization_values)
    np.testing.assert_array_equal(storage.view(bit_dtype), expected)
