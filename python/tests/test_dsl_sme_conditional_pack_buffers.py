# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Execute guarded I.Pack with real typed cache/history tensor inputs.

This is a code-generator primitive diagnostic, not an alternative lowering
or a timing kernel. A probe reports the actually entered implementation.
On ARM, both public C++ TransformPack and an independent scalar layout
oracle verify exact bits. On x86, the real automatic AMX/dense fallback is
checked against the scalar oracle; that is not SME execution evidence.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import amx_tile_available_for_test, make_compiler
from test_dsl_sme_packing import _available, _dtype, _expected
from test_dsl_sme_transformed_packing import _dense, _template, _values
from vecops._arm_vector_state import get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def _declaration(dtype, side):
  S, K, PS = (T.dynamic(lower=0) for _ in range(3))
  SS, KS = T.dynamic(), T.dynamic()
  shape, strides = ((S, K), (SS, KS)) if side == "A" else ((K, S), (KS, SS))

  @T.template
  def kernel(
    X: T.Tensor(shape, strides, "float32"),
    Cached: T.Tensor(shape, strides, dtype),
    History: T.Tensor(shape, strides, "float32"),
    P: T.Tensor((PS,), dtype=dtype, access="output"),
    Probe: T.Tensor((1,), dtype="int32", access="output"),
    *,
    UseCache: bool = False,
    UseHistory: bool = False,
  ):
    if UseCache:
      T.pack(Cached, P, dtype=dtype, side=side, format="auto")
      T.store(Probe, 1)
    else:
      if UseHistory:
        T.pack(T.load(History) * 0.5 + 7.0, P, dtype=dtype, side=side, format="auto")
        T.store(Probe, 2)
      else:
        T.pack(T.load(X) * 0.5 + 7.0, P, dtype=dtype, side=side, format="auto")
        T.store(Probe, 3)

  return kernel


def _source(dtype, side):
  """Cross compilation uses the same source DSL as the native oracle."""
  return '#include "vecops/Kernel.h"\n' + CPUBackend().lower(_declaration(dtype, side).ir, (), target="SME").source


def _geometry(compiled):
  arm = os.uname().machine.lower() in {"aarch64", "arm64"}
  if arm:
    if not _available():
      pytest.skip("this physical pack diagnostic requires SME+FA64 on ARM")
    svl = get_vector_state().sme_bytes
    assert svl is not None and svl > 0
    return "sme", svl // 2, 2
  # The selected generated artifact, not the host label, determines the
  # physical fallback. No assumed 64-byte SVE/SME geometry is used here.
  if amx_tile_available_for_test():
    features = Path("/proc/cpuinfo").read_text(encoding="utf-8").split()
    assert "amx_tile" in features
    return "amx", 16, 32
  return "dense", 1, 1


def _physical(logical, kind, panel, block, side):
  if kind == "dense":
    return np.ascontiguousarray(logical).reshape(-1)
  if kind == "sme":
    return _expected(logical, panel, 2)
  s, k = logical.shape
  groups = (k + block - 1) // block
  out = np.zeros(((s + panel - 1) // panel) * panel * groups * block, logical.dtype)
  for row in range(s):
    for column in range(k):
      prefix = ((row // panel) * groups + column // block) * panel * block
      suffix = (
        (row % panel) * block + column % block
        if side == "A"
        else (column % block // 2) * panel * 2 + (row % panel) * 2 + column % 2
      )
      out[prefix + suffix] = logical[row, column]
  return out


@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
@pytest.mark.parametrize("side", ("A", "B"))
def test_native_guarded_pack_cache_history_false_true_bitwise(tmp_path, dtype, side):
  compiler = make_compiler(
    target="Native", jobs=4, cflags=["-ffp-contract=off"], cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"
  )
  compiled = T.compile(_declaration(dtype, side), compiler=compiler)
  arm = os.uname().machine.lower() in {"aarch64", "arm64"}
  if arm and not _available():
    pytest.skip("this physical pack diagnostic requires SME+FA64 on ARM")
  reference_compiler = (
    make_compiler(
      target="Native", jobs=4, cache_dir=tmp_path / "reference-cache", build_dir=tmp_path / "reference-build"
    )
    if arm
    else None
  )
  pair = {}
  with vecops.collect_compile_requests() as collector:
    shape = (19, 17) if side == "A" else (17, 19)
    strides = (17, 1) if side == "A" else (1, 17)
    compiled.operator(
      vecops.TensorMeta(shape, strides=strides, dtype="float32"),
      vecops.TensorMeta(shape, strides=strides, dtype=dtype),
      vecops.TensorMeta(shape, strides=strides, dtype="float32"),
      vecops.TensorMeta((8192,), dtype=dtype),
      vecops.TensorMeta((1,), dtype="int32"),
    )
    if arm:
      for pipeline in ("affine", "convert"):
        # The public source kernel retains its original compiler policy;
        # strict contraction is required only by the generated DSL source.
        reference = T.compile(_template(dtype, side, pipeline, reference=True), compiler=reference_compiler)
        parameters = ((19, 1), (19, 1), (1, 17), (1, 17))
        if side == "B":
          parameters = tuple(value[::-1] for value in parameters)
        reference.operator(
          vecops.TensorMeta(shape, dtype="float32"),
          *(vecops.TensorMeta(value, dtype="float32") for value in parameters),
          vecops.TensorMeta((8192,), dtype=dtype),
          **reference._core.specialization_values,
        )
        pair[pipeline] = reference
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  kind, panel, block = _geometry(compiled)
  native, bits = _dtype(dtype), np.dtype("uint16")
  shapes = ((0, 5), (5, 0), (1, 1), (panel - 1, 3), (panel, block), (panel + 3, block + 1))
  calls = 0
  for spatial, k in shapes:
    x, *parameters = _values(spatial, k)
    for layout in ("dense", "negative", "gather"):
      # All three inputs honor the same declared strides. Cached already
      # contains native values; history is an earlier FP32 read of X.
      step = 2 if layout == "gather" else 1
      inputs = []
      for values in (x, (x * np.float32(0.25) - np.float32(1)).astype(native), x + np.float32(31)):
        row_step = max(1, step * k)
        backing = np.empty(max(1, spatial * row_step), values.dtype)
        view = np.ndarray(
          (spatial, k),
          values.dtype,
          buffer=backing,
          strides=(row_step * values.dtype.itemsize, step * values.dtype.itemsize),
        )
        view[...] = values
        if layout == "negative":
          view = view[::-1]
        inputs.append(view)
      logical, cached, history = inputs
      for use_cache, use_history in ((False, False), (True, False), (False, True), (True, True)):
        origin = history if use_history else logical
        values = cached if use_cache else (origin * np.float32(0.5) + np.float32(7)).astype(native)
        expected = _physical(values, kind, panel, block, side).view(bits)
        output = np.full(expected.size + 14, 0x55, dtype=bits)
        probe = np.zeros(1, np.int32)
        compiled.operator.native(
          *(value if side == "A" else value.T for value in inputs),
          output[7:-7].view(native),
          probe,
          UseCache=use_cache,
          UseHistory=use_history,
          **compiled._core.specialization_values,
        )
        np.testing.assert_array_equal(output[7:-7], expected)
        np.testing.assert_array_equal(output[:7], 0x55)
        np.testing.assert_array_equal(output[-7:], 0x55)
        assert probe[0] == (1 if use_cache else 2 if use_history else 3)
        if arm:
          # Do not weaken the public reference's alias contract: it gets
          # independent input/output buffers and the same value transform.
          output.fill(0x55)
          reference = pair["convert" if use_cache else "affine"]
          selected = cached.astype(np.float32) if use_cache else origin
          arguments = tuple(_dense(value if side == "A" else value.T) for value in (selected, *parameters))
          reference.operator.native(*arguments, output[7:-7].view(native), **reference._core.specialization_values)
          np.testing.assert_array_equal(output[7:-7], expected)
          np.testing.assert_array_equal(output[:7], 0x55)
          np.testing.assert_array_equal(output[-7:], 0x55)
        calls += 1
  assert calls == 72
