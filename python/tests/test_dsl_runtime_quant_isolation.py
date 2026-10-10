# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Runtime family eligibility must not replace the generic fusion policy."""

import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from test_dsl_matmul_sve_optimized import _strided_output
from test_dsl_sve_runtime_quant import OPTION, _sme_available, _template
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def _lower(*, enabled=True, fusion="auto", vl=64, svl=64, target="SME", **kwargs):
  return CPUBackend().lower(
    _template(**kwargs).ir,
    (),
    target=target,
    options={
      OPTION: enabled,
      "cpu_matmul_fusion": fusion,
      "cpu_matmul_pipeline": "off",
      "_vector_state": VectorState(vl, svl),
    },
  )


AUTO_NATIVE_SHAPES = ((1, 256, 256), (5, 19, 67), (3, 7, 5), (3, 7, 9), (64, 256, 256))


def _collect_auto_native_pair(kernels, panel):
  """Capture through the Python operator; the expert native call bypasses it."""
  m, n, k = AUTO_NATIVE_SHAPES[0]
  packed_size = ((k + 3) // 4) * 4 * ((n + panel - 1) // panel) * panel
  arguments = (
    vecops.TensorMeta((m, k), dtype="float32"),
    vecops.TensorMeta((packed_size,), dtype="int8"),
    vecops.TensorMeta((k, n), dtype="int8"),
    vecops.TensorMeta((m, n), dtype="int32"),
    vecops.TensorMeta((m, n), dtype="int32", access=vecops.TensorAccess.output),
    vecops.TensorMeta((m, 1), dtype="float32"),
    vecops.TensorMeta((m, 1), dtype="float32"),
    vecops.TensorMeta((1, n), dtype="float32"),
    vecops.TensorMeta((1,), dtype="int32"),
  )
  with vecops.collect_compile_requests() as collector:
    for kernel in kernels.values():
      kernel.operator(*arguments, **kernel._core.specialization_values)
  return collector.requests


@pytest.mark.parametrize("level,bk", (("dot", None), ("library", None), ("library", 3)))
def test_quant_auto_native_pair_capture_does_not_require_arm_or_compile(level, bk, tmp_path):
  # Exercise the very same capture entry as the ARM fixture, without its
  # hardware skip or any native compilation/execution. Storage-free metadata
  # cannot be executed. SME source lowering itself needs neither ARM hardware
  # nor an ARM compiler; the off/on sources must remain distinct here too.
  compiler = make_compiler(target="SME", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  template = _template(level=level, bk=bk, output="int32", pipeline="relu")
  kernels = {
    enabled: T.compile(template, compiler=compiler, pass_options={OPTION: enabled}) for enabled in (False, True)
  }
  requests = _collect_auto_native_pair(kernels, 32)
  assert len(requests) == 2
  for kernel, request in zip(kernels.values(), requests, strict=True):
    assert request.operator is kernel.operator.native
    vecops._C.bind_kernel_call(kernel.operator.kernel_def._native(), request.call)
  assert not compiler.build_events
  assert not compiler.build_dir.exists()


def _auto_native_case(shape, panel, *, layout="row", prior="disjoint", reverse_a=False):
  """Independent strict conversion oracle and canonical SME packed-B data."""
  import numpy as np

  m, n, k = shape
  rng = np.random.default_rng(2300 + m + n + k)
  a = rng.integers(-8, 80, size=(m, k)).astype("float32") / 4
  b = rng.integers(-4, 5, size=(k, n), dtype="int8")
  if k in (5, 9):
    # BK=3 rounds to four. An early ReLU or repeated prior injection changes
    # these signed partial sums even though the final epilogue is only ReLU.
    a.fill(5)
    b[:4], b[4:] = -1, 2
  if reverse_a:
    a = a[:, ::-1]
  a[0, 0] = -1.0000001192092896
  rq = np.full((m, 1), 1.9999998807907104, dtype="float32")
  rd, w, z = np.ones((m, 1), dtype="float32"), np.ones((1, n), dtype="float32"), np.array([3], dtype="int32")
  # Separate rounded mul/add: the first uint8 must be one, not the FMA zero.
  q = np.clip(np.trunc(np.float32(np.float32(a * rq) + np.float32(z[0]))), 0, 255).astype("uint8")
  assert q[0, 0] == 1
  product = q.astype("int32") @ b.astype("int32")
  padded = np.zeros((((k + 3) // 4) * 4, ((n + panel - 1) // panel) * panel), dtype="int8")
  padded[:k, :n] = b
  bp = padded.reshape((k + 3) // 4, 4, (n + panel - 1) // panel, panel).transpose(2, 0, 3, 1).copy().reshape(-1)
  c = _strided_output(m, n, "int32", layout)
  c[:] = -(product // 2) - 7
  if prior == "disjoint":
    d = c.copy()
  elif prior == "alias":
    d = c
  elif prior == "broadcast_alias":
    d = np.broadcast_to(c[:1], c.shape)
  else:
    raise AssertionError(f"unknown prior fixture: {prior}")
  initial = c.copy()
  expected = np.maximum(product + d.copy(), 0)
  return (a, bp, b, d, c, rq, rd, w, z), initial, expected


@pytest.mark.parametrize("shape", AUTO_NATIVE_SHAPES)
@pytest.mark.parametrize("prior", ("disjoint", "alias", "broadcast_alias"))
def test_quant_auto_native_fixture_budget_packing_and_alias_contract(shape, prior):
  import numpy as np

  args, initial, expected = _auto_native_case(shape, 32, prior=prior, layout="negative", reverse_a=True)
  a, bp, b, d, c, rq, _, _, z = args
  m, n, k = shape
  footprint = m * k * 4 + k * n + m * n * 4
  assert (footprint <= 128 * 1024) is (m != 64)
  unpacked = bp.reshape((n + 31) // 32, (k + 3) // 4, 32, 4).transpose(1, 3, 0, 2)
  unpacked = unpacked.reshape(((k + 3) // 4) * 4, ((n + 31) // 32) * 32)
  np.testing.assert_array_equal(unpacked[:k, :n], b)
  assert not np.any(unpacked[k:]) and not np.any(unpacked[:k, n:])
  assert (np.shares_memory(d, c)) is (prior != "disjoint")
  assert (d.strides[0] == 0) is (prior == "broadcast_alias")
  assert a.strides[1] < 0 and c.strides[0] < 0 and c.strides[1] < 0
  np.testing.assert_array_equal(c, initial)
  q = np.clip(np.trunc(np.float32(np.float32(a * rq) + np.float32(z[0]))), 0, 255).astype("uint8")
  np.testing.assert_array_equal(expected, np.maximum(q.astype("int32") @ b.astype("int32") + d, 0))
  if k == 9:
    partial = q[:, :4].astype("int32") @ b[:4].astype("int32") + d
    premature_relu = np.maximum(partial, 0) + q[:, 4:].astype("int32") @ b[4:].astype("int32")
    assert not np.array_equal(expected, np.maximum(premature_relu, 0))


@pytest.mark.skipif(not _sme_available(), reason="actual ordinary-SVE and SME packing required")
@pytest.mark.parametrize("level,bk", (("dot", None), ("library", None), ("library", 3)))
def test_quant_arm_native_auto_fallback_and_accepted_split_k(level, bk, tmp_path):
  import numpy as np

  state = get_vector_state()
  assert state.sve_bytes is not None and state.sme_bytes is not None
  panel = state.sme_bytes // 2
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  template = _template(level=level, bk=bk, output="int32", pipeline="relu")
  # One Dynamic kernel pair per level, shared across accepted/rejected shapes
  # and alias layouts. Do not force fusion="always": default "auto" is the
  # original regression (generic fallback requires real A materialization).
  kernels = {
    enabled: T.compile(template, compiler=compiler, pass_options={OPTION: enabled}) for enabled in (False, True)
  }
  requests = _collect_auto_native_pair(kernels, panel)
  assert len(requests) == 2
  vecops.compile_batch(requests, parallelism=compiler.jobs, execution_parallelism=1)
  for shape in AUTO_NATIVE_SHAPES:
    for layout, reverse_a in (("row", False), ("negative", True)):
      for prior in ("disjoint", "alias", "broadcast_alias"):
        args, initial, expected = _auto_native_case(shape, panel, layout=layout, prior=prior, reverse_a=reverse_a)
        outputs = []
        for enabled, kernel in kernels.items():
          args[4][:] = initial  # Restore aliased prior before each kernel.
          kernel.operator.native(*args, **kernel._core.specialization_values)
          label = f"{level}/BK={bk}/enabled={enabled}/{shape}/{layout}/{prior}"
          np.testing.assert_array_equal(args[4], expected, err_msg=label)
          outputs.append(args[4].copy())
        np.testing.assert_array_equal(outputs[0], outputs[1])
