"""Runtime family eligibility must not replace the generic fusion policy."""

import re

import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_matmul_sve_optimized import _cross, _strided_output
from test_dsl_sve_runtime_quant import MARKER, OPTION, _body, _cross_wrapper, _sme_available, _template
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES


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


@pytest.mark.parametrize("shape", (None, (1, 256, 256), (64, 256, 256)))
@pytest.mark.parametrize("level,bk", (("dot", None), ("library", None), ("library", 3)))
@pytest.mark.parametrize("fusion", ("auto", "always", "never"))
def test_quant_option_preserves_normal_fallback_operand_plans(shape, level, bk, fusion):
  off = _lower(enabled=False, shape=shape, level=level, bk=bk, fusion=fusion)
  on = _lower(enabled=True, shape=shape, level=level, bk=bk, fusion=fusion)
  off_plans, on_plans = list(off.backend_ir.matmuls.values()), list(on.backend_ir.matmuls.values())
  assert len(off_plans) == len(on_plans) == 1
  # Frozen expression/view reprs include the complete DAG, materialization
  # destination, condition, cost and reason, but no object-identity addresses.
  assert repr(off_plans[0].operands) == repr(on_plans[0].operands)
  if fusion == "never":
    assert not on_plans[0].sve_runtime_quant
    assert MARKER not in on.source
    assert "dsl_o_dot_input" in on.source


@pytest.mark.parametrize("level,bk", (("dot", None), ("library", None), ("library", 3)))
def test_quant_whole_problem_guard_precedes_fallback_only_preparation(level, bk):
  source = _lower(level=level, bk=bk).source
  assert source.count(MARKER) == 1
  gate = next(line for line in source.splitlines() if MARKER in line)
  assert "matmul_fits(" in gate
  assert "parameter_A_size_0" in gate and "parameter_A_size_1" in gate
  assert "parameter_B_size_1" in gate
  assert ",4,1,4,131072)" in gate
  assert not any(name in gate for name in ("panel_m", "panel_n", "panel_k", "block_k"))
  accepted = _body(source)
  assert "dsl_o_dot_input" not in accepted
  assert "pack A:" not in accepted
  assert "input_cache" not in accepted
  assert "with_resources" not in accepted
  fallback = source[source.index(accepted) + len(accepted) :]
  assert "reload cost exceeds saved tile traffic" in fallback
  assert re.search(r"dsl_o_dot_input_\w+\.resize\(", fallback)
  assert "sme::mopa" in fallback
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("k", (5, 9))
def test_quant_explicit_split_k_has_one_accepted_scope_and_original_phases(k):
  source = _lower(level="library", bk=3, shape=(5, 19, k)).source
  accepted = _body(source)
  assert "block_k" in accepted and "runtime_quant_prior" in accepted
  assert "epilogue exactly once" in accepted
  assert "corrected integer running sum, not correction again" in accepted
  assert f"pack_offset<{CPP_TYPES['int8']}, 2, true>" in accepted
  k_loop = next(line for line in accepted.splitlines() if "for (" in line and "block_k" in line)
  assert f"std::int64_t{{{k}}}" in k_loop and "+=4" in k_loop
  assert "vecops::ops::" not in accepted  # No opaque quantization operator call.
  assert MARKER not in accepted
  assert "pack A:" not in accepted
  assert "dsl_o_dot_input" not in accepted


@pytest.mark.parametrize("kwargs", ({"raw_b": True}, {"intermediate": "float64"}, {"target": "AVX512"}))
def test_quant_static_rejection_keeps_generic_materialization(kwargs):
  source = _lower(**kwargs).source
  assert MARKER not in source
  assert "dsl_o_dot_input" in source
  assert "reload cost exceeds saved tile traffic" in source


@pytest.mark.parametrize(
  "shape,level,bk,vl,svl,quant_expected,generic_expected",
  (
    ((1, 16, 5), "dot", None, 64, 64, True, False),
    ((64, 256, 256), "dot", None, 64, 64, False, True),
    (None, "dot", None, 16, 64, True, True),
    (None, "library", 3, 64, 16, True, True),
  ),
)
def test_quant_isolation_real_arm_assembly(tmp_path, shape, level, bk, vl, svl, quant_expected, generic_expected):
  source = _lower(shape=shape, level=level, bk=bk, vl=vl, svl=svl).source
  assembly = _cross(tmp_path, source, _cross_wrapper, vl=vl, svl=svl)
  assert bool(re.search(r"^\s*(?:usdot|sudot)\b", assembly, re.MULTILINE)) is quant_expected
  assert bool(re.search(r"^\s*usmopa\b", assembly, re.MULTILINE)) is generic_expected
  if not generic_expected:
    assert not re.search(r"^\s*smstart\b", assembly, re.MULTILINE)
  assert not re.search(r"^\s*(?:fmla|fmad|fmadd)\b", assembly, re.MULTILINE)
  assert not re.search(r"^\s*bl[^\n]*(?:WideningDot|widening_dot)", assembly, re.MULTILINE)
  assert_no_vec_mask_arrays(source)


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


@pytest.mark.parametrize("level,bk", (("dot", None), ("library", None), ("library", 3)))
def test_quant_auto_native_template_retains_real_generic_materialization(level, bk):
  off = _lower(enabled=False, level=level, bk=bk, output="int32", pipeline="relu")
  on = _lower(enabled=True, level=level, bk=bk, output="int32", pipeline="relu")
  assert repr(next(iter(off.backend_ir.matmuls.values())).operands) == repr(
    next(iter(on.backend_ir.matmuls.values())).operands
  )
  accepted = _body(on.source)
  fallback = on.source[on.source.index(accepted) + len(accepted) :]
  assert "dsl_o_dot_input" not in accepted
  assert re.search(r"dsl_o_dot_input_\w+\.resize\(", fallback)
  assert "reload cost exceeds saved tile traffic" in fallback


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
