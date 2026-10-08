# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bounded full-input schedule; old tails, captures and output stay explicit."""

import json
import os
import re
import subprocess
from pathlib import Path

import arm_sme_asm_utils as asm_utils
import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import arm_cross_cxx_include_flags, assert_no_vec_mask_arrays, make_compiler
from test_dsl_arm_vector_state_native import PR_SME_SET_VL, PR_SVE_SET_VL
from test_dsl_arm_vector_state_native import arm_vector_control as _arm_vector_control
from test_dsl_transpose_sme_metadata import cross_wrapper
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend

OPTION = "cpu_transpose_sme_const_half_full_input"
OUTPUT_OPTION = "cpu_transpose_sme_const_half_output"
MARKER = "/* full ZA input: named constant-pitch F32 pairs */"
FRINGE = "else /* typed input fringe: retain predicated ZA loads */"


def template(
  *, source_pitch=131, output_pitch=64, capture=False, batch=False, shape=None, types=None, transformed=False
):
  m, n = (T.dynamic(lower=0), T.dynamic(lower=0)) if shape is None else shape
  source_pitch = T.dynamic() if source_pitch is None else source_pitch
  output_pitch = T.dynamic() if output_pitch is None else output_pitch
  memory, destination, compute = ("float32", "float16", "float32") if types is None else types
  if batch:
    a_shape, c_shape = (4, m, n), (4, n, m)
    a_strides, c_strides = (T.dynamic(), source_pitch, 1), (T.dynamic(), output_pitch, 1)
  else:
    a_shape, c_shape = (m, n), (n, m)
    a_strides, c_strides = (source_pitch, 1), (output_pitch, 1)

  @T.template
  def kernel(
    A: T.Tensor(a_shape, a_strides, memory, access="inout" if capture else "input"),
    C: T.Tensor(c_shape, c_strides, destination, access="output"),
  ):
    if batch:
      for item in T.grid(4, parallel="require", schedule="static"):
        T.store(C[item], T.load(T.transpose(A[item]), dtype=compute), implementation="sme")
    else:
      value = T.load(T.transpose(A), dtype=compute)
      if capture:
        T.store(A, 42)
      if transformed:
        T.store(C, value + 1)
      else:
        T.store(C, value, implementation="sme")

  return kernel


def lower(kernel, *, vl=64, svl=64, enabled=True, output=True, target="SME", **options):
  return CPUBackend().lower(
    kernel.ir,
    (),
    target=target,
    options={"_vector_state": VectorState(vl, svl), OPTION: enabled, OUTPUT_OPTION: output, **options},
  )


def full_input(code):
  start = code.index(MARKER)
  return code[start : code.index(FRINGE, start)]


@pytest.mark.parametrize("vl,svl", ((16, 16), (64, 16), (16, 32), (16, 64), (64, 64)))
@pytest.mark.parametrize("source_pitch,output_pitch", ((131, 64), (-131, -64)))
def test_named_pairs_use_captured_streaming_rows_and_constant_signed_addresses(vl, svl, source_pitch, output_pitch):
  code = lower(template(source_pitch=source_pitch, output_pitch=output_pitch), vl=vl, svl=svl).source
  body = full_input(code)
  rows = svl // 4
  assert "for (" not in body and "+=" not in body and "opt::first" not in body
  assert "load_convert(" not in body and "load_hor<0>" not in body
  assert re.findall(r"write_hor<0>\(std::uint32_t\{(\d+)\}", body) == list(map(str, range(rows)))
  assert re.findall(r"input_origin_\d+ \+ std::int64_t\{(-?\d+)\}", body) == [
    str(row * source_pitch) for row in range(rows)
  ]
  operations = [line for line in body.splitlines() if "vecops::vec::load(" in line or "write_hor<0>" in line]
  assert ["load" if "vecops::vec::load(" in line else "write" for line in operations] == [
    kind for _ in range(rows // 2) for kind in ("load", "load", "write", "write")
  ]
  assert re.search(rf"active_m_\d+ == \w+ && \w+ == \w+ && \w+ == {rows}\) {re.escape(MARKER)}", code)
  assert "transpose_source_base_" in body and "dsl_c_parameter_" not in body
  assert "/* overlapping operands */" in code and "/* unsupported strides: preserve gather/scatter */" in code
  assert_no_vec_mask_arrays(code)


@pytest.mark.parametrize("shape", (None, (63, 129)))
@pytest.mark.parametrize("source_pitch,output_pitch", ((131, 64), (-131, -64)))
def test_original_typed_input_and_output_fringe_are_retained(shape, source_pitch, output_pitch):
  kernel = template(source_pitch=source_pitch, output_pitch=output_pitch, shape=shape)
  before, after = (lower(kernel, enabled=flag).source for flag in (False, True))
  old_loads = [line.strip() for line in re.findall(r"[^\n]*load_hor<0>[^\n]*", before)]
  new_loads = [line.strip() for line in re.findall(r"[^\n]*load_hor<0>[^\n]*", after)]
  assert old_loads == new_loads and old_loads
  tail = after[after.index(FRINGE) :]
  assert "load_hor<0>" in tail and "opt::first" in tail
  assert "/* full ZA output: constant F32-to-F16 addresses */" in before
  assert "/* full ZA output: constant F32-to-F16 addresses */" in after
  assert_no_vec_mask_arrays(after)


@pytest.mark.parametrize("svl", (None, 0, True, 8, 15, 17, 128, 256, 272))
def test_unknown_or_over_named_row_budget_retains_exact_old_source(svl):
  kernel = template()
  assert lower(kernel, svl=svl).source == lower(kernel, svl=svl, enabled=False).source


@pytest.mark.parametrize("source_pitch", (None, 0, 2**63 - 1, -(2**63)))
def test_unknown_zero_or_offset_overflow_source_pitch_retains_exact_old_source(source_pitch):
  kernel = template(source_pitch=source_pitch)
  assert lower(kernel).source == lower(kernel, enabled=False).source


@pytest.mark.parametrize(
  "types", (("float16", "float16", "float32"), ("float32", "float32", "float32"), ("float64", "float16", "float64"))
)
def test_other_memory_compute_or_output_dtype_keeps_exact_old_source(types):
  kernel = template(types=types)
  assert lower(kernel).source == lower(kernel, enabled=False).source


@pytest.mark.parametrize("output_pitch", (None, 0, 2**63 - 1))
def test_existing_output_family_legality_is_not_bypassed(output_pitch):
  kernel = template(output_pitch=output_pitch)
  assert lower(kernel).source == lower(kernel, enabled=False).source


def test_default_boolean_and_independent_output_option_contract():
  kernel = template()
  default = CPUBackend().lower(
    kernel.ir, (), target="SME", options={"_vector_state": VectorState(16, 64), OUTPUT_OPTION: True}
  )
  assert default.source == lower(kernel, vl=16, enabled=True).source
  assert MARKER in lower(kernel, vl=16, output=False).source
  for value in (None, 0, 1, "true"):
    with pytest.raises(TypeError, match=OPTION + " must be bool"):
      lower(kernel, enabled=value)


def test_cost_profile_and_elementwise_expression_do_not_establish_input_legality():
  kernel = template()
  assert (
    lower(kernel, svl=None, cpu_cost_target="SME").source
    == lower(kernel, svl=None, enabled=False, cpu_cost_target="SME").source
  )
  assert MARKER not in lower(kernel, target="AVX512", cpu_cost_target="SME").source
  assert MARKER not in lower(template(transformed=True)).source


@pytest.mark.parametrize("source_pitch,output_pitch", ((131, 64), (-131, -64)))
def test_historical_capture_protects_full_input_path_with_original_snapshot_guard(source_pitch, output_pitch):
  code = lower(template(source_pitch=source_pitch, output_pitch=output_pitch, capture=True)).source
  aliases = re.findall(r"const bool (dsl_c_alias_\d+) = ([^\n]+);", code)
  snapshots = re.findall(r"Buffer<float> (dsl_c_snapshot_\d+);", code)
  assert len(aliases) == len(snapshots) == 2
  saved_alias, saved_snapshot = aliases[0][0], snapshots[0]
  output_alias, output_snapshot = aliases[1][0], snapshots[1]
  write = code.index("std::int64_t{42}")
  renewal = code.index("Buffer<float> " + output_snapshot)
  assert code.index(saved_snapshot + ".resize(") < write < renewal
  assert f"({saved_alias}) ||" in aliases[1][1]
  copied = code[renewal : code.index(f"if ({output_alias}) /* overlapping operands */", renewal)]
  loads = [line for line in copied.splitlines() if "vecops::vec::load_convert(" in line]
  assert loads and all(saved_snapshot + ".data()" in line and saved_alias + " ?" in line for line in loads)
  assert code.index(f"if ({output_alias}) /* overlapping operands */") < code.index(MARKER)
  assert output_snapshot + ".data()" in code[code.index("/* overlapping operands */", renewal) :]
  assert_no_vec_mask_arrays(code)


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64)))
@pytest.mark.parametrize("source_pitch,output_pitch", ((131, 64), (-131, -64), (129, 63)))
@pytest.mark.parametrize("shape", (None, (63, 129)))
@pytest.mark.parametrize("enabled", (False, True))
def test_cross_named_full_input_and_typed_fringe_owner(tmp_path, vl, svl, source_pitch, output_pitch, shape, enabled):
  clang, sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG"), os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("real LLVM ARM cross compiler and sysroot required")
  root = Path(__file__).resolve().parents[2]
  generated = lower(
    template(source_pitch=source_pitch, output_pitch=output_pitch, shape=shape), vl=vl, svl=svl, enabled=enabled
  ).source
  cpp = tmp_path / "transpose.cpp"
  cpp.write_text(cross_wrapper(generated, "float32", "float16"))
  object_file = cpp.with_suffix(".o")
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    "-msve-vector-bits=scalable",
    "-ffp-contract=off",
    "-std=c++20",
    "-O3",
    "-c",
    "-DNDEBUG",
    "-DVECOPS_TARGET_SME_FA64=1",
    "-DVECOPS_TARGET_SME_F64F64=1",
    f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={svl * 8}",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
    str(cpp),
    "-o",
    str(object_file),
  ]
  process = subprocess.run(command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".log").write_text(process.stdout + process.stderr)
  cpp.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n")
  assert process.returncode == 0, process.stderr[-6000:]
  dump = subprocess.check_output(["/usr/bin/aarch64-linux-gnu-objdump", "-Cd", str(object_file)], text=True)
  cpp.with_suffix(".dump").write_text(dump)
  # Safe State separates the fixed SM-entry frame and same-SVL ZA body
  # when captured domains differ. Validate both interprocedurally; counting
  # every function containing SMSTART would wrongly conflate those owners.
  assembly_path = cpp.with_suffix(".s")
  assembly_command = ["-S" if token == "-c" else token for token in command[:-1]] + [str(assembly_path)]
  assembly_process = subprocess.run(assembly_command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".assembly.log").write_text(assembly_process.stdout + assembly_process.stderr)
  cpp.with_suffix(".assembly.command.json").write_text(json.dumps(assembly_command, indent=2) + "\n")
  assert assembly_process.returncode == 0, assembly_process.stderr[-6000:]
  checked_bodies = asm_utils.assert_sme_owners(assembly_path.read_text(), 1, equal_domains=vl == svl)
  assert len(checked_bodies) == 1
  owner = next(iter(checked_bodies.values()))
  assert not re.search(r"\sbl(?:r)?\s", owner), "public load/ZA/convert helper must inline"
  assert re.search(r"\bld1w\s+\{za0h", owner), "typed predicated ZA fringe must remain"
  if enabled:
    assert re.search(r"\bld1(?:b|w)\s+\{\s*z\d+", owner), "named complete input must load Z"
    assert re.search(r"\bmov\s+za0h\.s", owner), "named Z inputs must enter ZA horizontally"
  assert not re.search(r"\b(?:str|st1\w*)\s+\{?z\d+[^\n]*\[sp", owner), "no new direct Z spill"
  cpp.with_suffix(".summary.json").write_text(
    json.dumps(
      {
        "VL": vl,
        "SVL": svl,
        "source_pitch": source_pitch,
        "output_pitch": output_pitch,
        "shape": shape,
        "enabled": enabled,
        "compute_body_scalar_SP_accesses": len(re.findall(r"\b(?:ldr|ldp|str|stp)\s+[xw]\d+[^\n]*\[sp", owner)),
        "compute_body_assembly_instructions": len(asm_utils._instructions(owner)[0]),
        "state_entry_and_compute_lifecycles_checked_by_CFG": True,
        "equal_captured_domains": vl == svl,
        "not_target_performance_or_ABI_safety_evidence": True,
      },
      indent=2,
    )
    + "\n"
  )


@pytest.mark.parametrize("svl", (128, 256))
def test_cross_larger_streaming_lengths_keep_original_owner(tmp_path, svl):
  kernel = template(source_pitch=129, output_pitch=63, shape=(63, 129))
  assert lower(kernel, vl=16, svl=svl).source == lower(kernel, vl=16, svl=svl, enabled=False).source
  # The conservative named-row budget is not an architectural rejection.
  # Real scalable code still compiles its original predicated ZA path.
  test_cross_named_full_input_and_typed_fringe_owner(tmp_path, 16, svl, 129, 63, (63, 129), False)


@pytest.fixture(scope="module")
def native_pairs(tmp_path_factory):
  if os.uname().machine.lower() not in {"aarch64", "arm64"}:
    pytest.skip("actual ARM SME target required")
  directory = tmp_path_factory.mktemp("const-half-full-input")
  compiler = make_compiler(jobs=4, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  result = {}
  with vecops.collect_compile_requests() as requests:
    for source_pitch, output_pitch in ((131, 64), (-131, -64)):
      for capture in (False, True):
        pair = tuple(
          T.compile(
            template(source_pitch=source_pitch, output_pitch=output_pitch, capture=capture),
            compiler=compiler,
            pass_options={OUTPUT_OPTION: True, **({OPTION: flag} if flag is not None else {})},
          )
          for flag in (False, True, None)
        )
        for kernel in pair:
          kernel.operator(
            vecops.TensorMeta((31, 33), strides=(source_pitch, 1), dtype="float32"),
            vecops.TensorMeta((33, 31), strides=(output_pitch, 1), dtype="float16"),
          )
        result[source_pitch, output_pitch, capture] = pair
  vecops.compile_batch(requests.requests, parallelism=4)
  return result


def invoke_oracle(kernel, source, source_pitch, output_pitch, shift, *, capture):
  rows, columns = source.shape
  source_span = ((rows - 1) * abs(source_pitch) + columns) * 4 if rows and columns else 0
  target_span = ((columns - 1) * abs(output_pitch) + rows) * 2 if rows and columns else 0
  start = 64 + (0 if shift is None else shift)
  union_end = max(64 + source_span, start + target_span)
  raw = np.full(union_end + 64, 17, dtype=np.uint8)
  source_raw = np.full(64 + source_span + 64, 17, dtype=np.uint8) if shift is None else raw
  source_offset = 64 + ((rows - 1) * abs(source_pitch) * 4 if source_pitch < 0 and rows and columns else 0)
  a = np.ndarray(
    source.shape,
    dtype=np.float32,
    buffer=source_raw,
    offset=source_offset,
    strides=(source_pitch * 4, 4),
  )
  a[...] = source
  output_offset = start + ((columns - 1) * abs(output_pitch) * 2 if output_pitch < 0 and rows and columns else 0)
  output = np.ndarray(
    (columns, rows), dtype=np.float16, buffer=raw, offset=output_offset, strides=(output_pitch * 2, 2)
  )
  with np.errstate(invalid="ignore"):
    expected = a.copy().astype(np.float16).T.copy()
  saved, source_saved = raw.copy(), source_raw.copy()
  kernel(a, output)
  context = rows, columns, source_pitch, output_pitch, shift, capture
  assert np.array_equal(output, expected, equal_nan=True), (*context, "independent conversion oracle")
  special = (expected == 0) | np.isinf(expected)
  assert np.array_equal(np.signbit(output[special]), np.signbit(expected[special])), (*context, "signed zero/infinity")
  assert np.array_equal(raw[:64], saved[:64]) and np.array_equal(raw[union_end:], saved[union_end:]), "union canaries"
  if shift is None:
    assert np.array_equal(source_raw[:64], source_saved[:64]) and np.array_equal(
      source_raw[64 + source_span :], source_saved[64 + source_span :]
    ), "source canaries"
    if capture:
      assert np.all(a == np.float32(42)), "historical intervening write was dropped"
    else:
      assert np.array_equal(source_raw, source_saved), "input-only source was overwritten"
  return output.copy().view(np.uint16)


@pytest.mark.parametrize("shape", ((0, 0), (7, 0), (0, 7)))
@pytest.mark.parametrize("source_pitch,output_pitch", ((131, 64), (-131, -64)))
@pytest.mark.parametrize("shift", (None, 0, 1, 16))
def test_empty_oracle_metadata_and_canaries_do_not_form_out_of_buffer_origins(shape, source_pitch, output_pitch, shift):
  observed = []

  def metadata_only(a, c):
    assert a.shape == shape and c.shape == shape[::-1]
    assert vecops._C.TensorView(a).strides == [source_pitch, 1]
    assert vecops._C.TensorView(c, writable=True).strides == [output_pitch, 1]
    observed.append((a.shape, c.shape))

  source = np.empty(shape, dtype=np.float32)
  result = invoke_oracle(metadata_only, source, source_pitch, output_pitch, shift, capture=False)
  assert len(observed) == 1 and result.shape == shape[::-1] and result.size == 0


@pytest.mark.parametrize("source_pitch,output_pitch", ((131, 64), (-131, -64)))
@pytest.mark.parametrize("capture", (False, True))
def test_arm_native_named_full_input_alias_tails_signed_pitch_and_capture(
  native_pairs, source_pitch, output_pitch, capture
):
  from vecops._arm_vector_state import get_vector_state

  original = get_vector_state()
  for rows, columns in ((0, 0), (7, 0), (0, 7), (1, 1), (7, 9), (16, 32), (31, 33), (63, 129), (64, 127)):
    source = (((np.arange(rows * columns) % 251) - 125) / 8).astype(np.float32).reshape(rows, columns)
    if source.size >= 6:
      source.flat[:6] = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345, 0x7F812345], dtype=np.uint32).view(
        np.float32
      )
    for shift in (None, 0, 1, 16):
      old, new, automatic = (
        invoke_oracle(kernel, source, source_pitch, output_pitch, shift, capture=capture)
        for kernel in native_pairs[source_pitch, output_pitch, capture]
      )
      assert np.array_equal(old, new), (rows, columns, source_pitch, output_pitch, shift, capture, "same-target bits")
      assert np.array_equal(new, automatic), (
        rows,
        columns,
        source_pitch,
        output_pitch,
        shift,
        capture,
        "automatic policy bits",
      )
      assert get_vector_state() == original, "main-thread VL/SVL/SM state leaked"


@pytest.fixture
def arm_control(tmp_path):
  yield from _arm_vector_control.__wrapped__(tmp_path)


@pytest.mark.skipif(os.uname().machine.lower() not in {"aarch64", "arm64"}, reason="actual ARM unequal worker gate")
def test_arm_native_named_full_input_unequal_worker_restore_and_stale_guard(tmp_path, arm_control, monkeypatch):
  from vecops.language._compiler.cost.model import CostProfile

  monkeypatch.setenv("VECOPS_NUM_THREADS", "4")
  assert vecops.cpu_thread_pool_capacity() >= 2, "exclusive worker allocation required"
  control = arm_control
  original = control.original_pair
  sve, sme = control.supported(PR_SVE_SET_VL), control.supported(PR_SME_SET_VL)
  pairs = ((min(sve), max(sme)), (max(sve), min(sme)))
  assert len(set(pairs)) == 2 and all(a != b for a, b in pairs), "two unequal supported pairs required"
  compiler = make_compiler(jobs=4, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  options = {
    OUTPUT_OPTION: True,
    "cost_profiles": (CostProfile("named-half-input-worker", policies=(("grid_worker_cycles", 0),)),),
  }

  def prepare(enabled):
    overrides = {} if enabled is None else {OPTION: enabled}
    return T.compile(template(batch=True), compiler=compiler, pass_options={**options, **overrides})

  def run(kernel, pair, rows, columns):
    control.assert_pair(pair)
    source = (((np.arange(4 * rows * columns) % 251) - 125) / 8).astype(np.float32).reshape(4, rows, columns)
    source_plane = ((rows - 1) * 131 + columns) * 4
    output_plane = ((columns - 1) * 64 + rows) * 2
    source_raw = np.full(64 + 4 * source_plane + 64, 17, dtype=np.uint8)
    raw = np.full(64 + 4 * output_plane + 64, 17, dtype=np.uint8)
    a = np.ndarray(source.shape, dtype=np.float32, buffer=source_raw, offset=64, strides=(source_plane, 524, 4))
    c = np.ndarray((4, columns, rows), dtype=np.float16, buffer=raw, offset=64, strides=(output_plane, 128, 2))
    a[...] = source
    saved = source_raw.copy()
    with np.errstate(invalid="ignore"):
      expected = source.astype(np.float16).transpose(0, 2, 1).copy()
    kernel(a, c)
    assert np.array_equal(c, expected, equal_nan=True), (pair, rows, columns, "worker oracle")
    assert np.all(raw[:64] == 17) and np.all(raw[-64:] == 17), "worker canaries"
    assert np.array_equal(source_raw, saved), "worker input was overwritten"
    control.assert_pair(pair)
    return c.copy().view(np.uint16)

  warm = prepare(False)
  run(warm, original, 31, 33)
  kernels = {}
  with vecops.collect_compile_requests() as requests:
    for pair in pairs:
      control.select(pair)
      for enabled in (False, True, None):
        kernel = prepare(enabled)
        kernel.operator(
          vecops.TensorMeta((4, 31, 33), strides=((31 - 1) * 131 + 33, 131, 1), dtype="float32"),
          vecops.TensorMeta((4, 33, 31), strides=((33 - 1) * 64 + 31, 64, 1), dtype="float16"),
        )
        kernels[pair, enabled] = kernel
  vecops.compile_batch(requests.requests, parallelism=4, execution_parallelism=1)
  for pair in pairs:
    control.select(pair)
    for rows, columns in ((32, 32), (31, 33), (63, 65)):
      old, new, automatic = (run(kernels[pair, enabled], pair, rows, columns) for enabled in (False, True, None))
      assert np.array_equal(old, new), (pair, rows, columns, "same-target worker bits")
      assert np.array_equal(new, automatic), (pair, rows, columns, "automatic worker bits")
  with pytest.raises(RuntimeError, match="SVE VL/SME SVL.*calling thread"):
    run(kernels[pairs[0], True], pairs[1], 31, 33)
  control.select(original)
  run(warm, original, 31, 33)
