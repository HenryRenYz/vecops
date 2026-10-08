# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Constant GEMV address vectors must never cross an unequal VL/SME boundary.

This is actual target execution, not a source-only no-call assertion. It
reuses the caller-state guard/restoration fixture, captures both lengths in
each cold request, and checks raw/packed-A constant-shape SME dot results.
"""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_arm_vector_state_native import (
  PR_SME_SET_VL,
  PR_SVE_SET_VL,
  arm_vector_control,  # noqa: F401 -- imported pytest fixture supplies guarded state restoration
  native_arm,
)
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def const_decode_template(packed, *, svl_bytes, explicit_strides=False):
  m, n, k = 1, 1024, 1024
  # FP16 SME K-step is two, with the existing public two-tile packing ABI.
  # Geometry is captured from this process, not an assumed 64-byte machine.
  panel = 2 * svl_bytes // 4
  packed_size = ((m + panel - 1) // panel) * ((k + 1) // 2) * panel * 2 if packed else 1

  @T.template
  def kernel(
    A: T.Tensor((m, k), (k, 1) if explicit_strides else None, "float16"),
    B: T.Tensor((k, n), (n, 1) if explicit_strides else None, "float16"),
    P: T.Tensor((packed_size,), dtype="float16", access="output"),
    C: T.Tensor((m, n), (n, 1) if explicit_strides else None, "float32", access="output"),
  ):
    a = T.pack(A, P, side="A", format="sme") if packed else T.load(A)
    T.store(C, T.dot((a, T.load(B)), C, implementation="sme"))

  return kernel, packed_size


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("explicit_strides", (False, True))
def test_const_decode_boundary_uses_streaming_geometry_and_named_vectors(packed, explicit_strides):
  template, _ = const_decode_template(packed, svl_bytes=64, explicit_strides=explicit_strides)
  source = (
    CPUBackend()
    .lower(
      template.ir,
      (),
      target="SME",
      options={"_vector_state": VectorState(16, 64), "cpu_matmul_pipeline": "off"},
    )
    .source
  )
  assert "SME 1x4 outer-product schedule" in source
  assert "sme::mopa" in source
  assert "ScalableTag" in source
  assert_no_vec_mask_arrays(source)


@native_arm
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("explicit_strides", (False, True))
def test_native_const_decode_raw_and_packeda_under_unequal_vector_lengths(packed, explicit_strides, tmp_path, request):
  control = request.getfixturevalue("arm_vector_control")
  sve = control.supported(PR_SVE_SET_VL)
  sme = control.supported(PR_SME_SET_VL)
  smaller = [(vl, svl) for vl in sve for svl in sme if vl < svl]
  larger = [(vl, svl) for vl in sve for svl in sme if vl > svl]
  if not smaller or not larger:
    pytest.skip("CPU/OS must expose both unequal ordinary/streaming length orders")
  pairs = (max(smaller, key=lambda pair: pair[1] / pair[0]), max(larger, key=lambda pair: pair[0] / pair[1]))
  control.restore()
  rng = np.random.default_rng(938)
  a = rng.integers(-3, 4, size=(1, 1024)).astype(np.float16)
  b = rng.integers(-3, 4, size=(1024, 1024)).astype(np.float16)
  # Integer-valued FP16 products/sums are exactly representable in F32 here.
  # Compute the oracle at the original caller VL, before changing either mode.
  expected = np.sum(a.astype(np.float32).reshape(1024, 1) * b.astype(np.float32), axis=0).reshape(1, 1024)
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for pair in pairs:
      control.select(pair)
      template, count = const_decode_template(packed, svl_bytes=pair[1], explicit_strides=explicit_strides)
      compiled = T.compile(template, compiler=compiler, pass_options={"cpu_matmul_pipeline": "off"})
      compiled.operator(
        vecops.TensorMeta((1, 1024), dtype="float16"),
        vecops.TensorMeta((1024, 1024), dtype="float16"),
        vecops.TensorMeta((count,), dtype="float16"),
        vecops.TensorMeta((1, 1024), dtype="float32"),
        **compiled._core.specialization_values,
      )
      kernels[pair] = compiled, count
  assert len(collector.requests) == len(pairs)
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  keys = set()
  for pair, (compiled, count) in kernels.items():
    control.select(pair)
    storage = np.full(1024 + 8, -16384, dtype=np.float32)
    output = storage[4:-4].reshape(1, 1024)
    packed_storage = np.full(count + 8, -123, dtype=np.float16)
    compiled.operator.native(a, b, packed_storage[4:-4], output, **compiled._core.specialization_values)
    control.assert_pair(pair)
    control.restore()
    np.testing.assert_array_equal(output, expected, err_msg=(packed, explicit_strides, pair))
    np.testing.assert_array_equal(storage[:4], -16384)
    np.testing.assert_array_equal(storage[-4:], -16384)
    np.testing.assert_array_equal(packed_storage[:4], -123)
    np.testing.assert_array_equal(packed_storage[-4:], -123)
    assert_no_vec_mask_arrays(compiled.generated_source)
    assert compiled.operator.compiler.vector_state == VectorState(*pair)
    keys.add(compiled.get_cache_info()["key"])
  assert len(keys) == len(pairs)
