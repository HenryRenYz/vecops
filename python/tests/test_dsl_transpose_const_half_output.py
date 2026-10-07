"""Bounded constant-pitch F32 -> F16 epilogue; isolate its original input."""

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

OPTION = "cpu_transpose_sme_const_half_output"
INPUT_OPTION = "cpu_transpose_sme_const_half_full_input"
MARKER = "/* full ZA output: constant F32-to-F16 addresses */"
FRINGE = "else /* typed output fringe: keep cold addresses local */"


def template(*, pitch=64, types=("float32", "float16", "float32"), capture=False, scope=False, batch=False, shape=None):
  m, n = (T.dynamic(lower=0), T.dynamic(lower=0)) if shape is None else shape
  count = 4
  if batch:
    source_shape, output_shape = (count, m, n), (count, n, m)
    strides = (T.dynamic(), pitch if pitch is not None else T.dynamic(), 1)
  else:
    source_shape, output_shape = (m, n), (n, m)
    strides = (pitch if pitch is not None else T.dynamic(), 1)

  @T.template
  def kernel(
    A: T.Tensor(source_shape, dtype=types[0], access="inout" if capture else "input"),
    C: T.Tensor(output_shape, strides, types[1], access="output"),
  ):
    if batch:
      for item in T.grid(count, parallel="require", schedule="static"):
        T.store(C[item], T.load(T.transpose(A[item]), dtype=types[2]), implementation="sme")
    else:
      value = T.load(T.transpose(A), dtype=types[2])
      if capture:
        T.store(A, 42)
      if scope:
        with T.sme.region():
          T.store(C, value, implementation="sme")
      else:
        T.store(C, value, implementation="sme")

  return kernel


def lower(kernel, *, vl=64, svl=64, enabled=True, target="SME", **options):
  return CPUBackend().lower(
    kernel.ir,
    (),
    target=target,
    options={"_vector_state": VectorState(vl, svl), OPTION: enabled, INPUT_OPTION: False, **options},
  )


def full_output(source):
  start = source.index(MARKER)
  return source[start : source.index(FRINGE, start)]


@pytest.mark.parametrize("vl,svl", ((16, 16), (64, 16), (16, 32), (16, 64), (64, 64)))
@pytest.mark.parametrize("pitch", (64, -64, 131))
def test_static_half_output_is_named_bounded_independent_addresses(vl, svl, pitch):
  code = lower(template(pitch=pitch), vl=vl, svl=svl).source
  body = full_output(code)
  columns = svl // 4
  assert "for (" not in body and "+=" not in body and "opt::first" not in body
  assert "vecops::vec::load(" not in code and "experimental Z-load" not in code
  assert re.findall(r"read_ver<0>\([^\n]+std::uint32_t\{(\d+)\}", body) == list(map(str, range(columns)))
  assert re.findall(r"output_origin_\d+ \+ std::int64_t\{(-?\d+)\}", body) == [
    str(column * pitch) for column in range(columns)
  ]
  operations = [line for line in body.splitlines() if "read_ver<0>" in line or "store_convert(" in line]
  assert ["read" if "read_ver" in line else "store" for line in operations] == [
    kind for _ in range(columns // 2) for kind in ("read", "read", "store", "store")
  ]
  fallback = code[code.index(FRINGE) :]
  # The dynamic-width plan has one cold slice loop, not a second paired
  # slice loop. Require its actual loop to be rolled rather than assuming
  # the number of static/dynamic plan alternatives.
  column_loops = re.findall(
    r"#pragma clang loop unroll\(disable\)\n(?:\s*#[^\n]*\n)*\s*for \([^\n]*dsl_c_za_col", fallback
  )
  assert len(column_loops) == 1
  assert "opt::first" in fallback and "load_hor<0>" in code
  assert "/* overlapping operands */" in code and "/* unsupported strides: preserve gather/scatter */" in code
  assert re.search(rf"active_m_\d+ == \w+ && \w+ == \w+ && \w+ == {columns}\)", code)
  assert_no_vec_mask_arrays(code)


@pytest.mark.parametrize("svl", (None, 0, True, 8, 15, 17, 128, 256, 272))
def test_unknown_or_over_budget_svl_keeps_exact_old_source(svl):
  kernel = template()
  assert lower(kernel, svl=svl).source == lower(kernel, svl=svl, enabled=False).source


@pytest.mark.parametrize("svl", (16, 32, 64))
def test_constant_geometry_retains_original_paired_input_and_cold_loops(svl):
  kernel = template(pitch=63, shape=(63, 129))
  old, new = (lower(kernel, svl=svl, enabled=flag).source for flag in (False, True))
  assert re.findall(r"[^\n]*load_hor<0>[^\n]*", old) == re.findall(r"[^\n]*load_hor<0>[^\n]*", new)
  fringe = new[new.index(FRINGE) :]
  loops = re.findall(r"#pragma clang loop unroll\(disable\)\n(?:\s*#[^\n]*\n)*\s*for \([^\n]*dsl_c_za_col", fringe)
  assert len(loops) == 2  # One original paired loop and its original singleton fringe.
  assert "opt::first" in fringe and MARKER not in old
  assert_no_vec_mask_arrays(new)


@pytest.mark.parametrize(
  "types,pitch",
  (
    (("float64", "int8", "float64"), 64),
    (("float64", "float16", "float64"), 64),
    (("float32", "float32", "float32"), 64),
    (("float16", "float16", "float32"), 64),
    (("float32", "float16", "float32"), None),
    (("float32", "float16", "float32"), 0),
    (("float32", "float16", "float32"), 2**63 - 1),
  ),
)
def test_other_physical_dtype_or_unknown_pitch_preserves_old_source(types, pitch):
  kernel = template(types=types, pitch=pitch)
  assert lower(kernel).source == lower(kernel, enabled=False).source


def test_explicit_output_disable_and_boolean_contract_with_original_input_schedule():
  kernel = template()
  default = CPUBackend().lower(
    kernel.ir, (), target="SME", options={"_vector_state": VectorState(16, 64), OPTION: False}
  )
  disabled = lower(kernel, vl=16, enabled=False)
  enabled = lower(kernel, vl=16)
  assert default.source == disabled.source
  assert disabled.kernel.parameters == enabled.kernel.parameters
  assert disabled.lowered.manager.options[OPTION] is False
  assert enabled.lowered.manager.options[OPTION] is True
  before, after = (value.source for value in (disabled, enabled))
  # Exactly the same typed input slices; no input Z-load experiment is mixed.
  assert re.findall(r"[^\n]*load_hor<0>[^\n]*", before) == re.findall(r"[^\n]*load_hor<0>[^\n]*", after)
  for value in (None, 0, 1, "true"):
    with pytest.raises(TypeError, match=OPTION + " must be bool"):
      lower(kernel, enabled=value)


@pytest.mark.parametrize("scope,batch", ((True, False), (False, True)))
def test_existing_owner_and_batch_use_local_origins(scope, batch):
  kernel = template(scope=scope, batch=batch)
  code = lower(kernel, vl=16, svl=64).source
  previous = lower(kernel, vl=16, svl=64, enabled=False).source
  # T.grid emits serial and parallel alternatives. Do not mistake those
  # original two owners for nesting created by this output candidate.
  assert code.count(".with_resources(") == previous.count(".with_resources(") == (2 if batch else 1)
  assert code.count(MARKER) == (2 if batch else 1)
  assert "dsl_c_parameter_" not in full_output(code)
  assert "transpose_target_base_" in full_output(code)
  assert "#define VECOPS_SVE_INLINE_MATH_SPECIALS 1" in code
  assert_no_vec_mask_arrays(code)


def test_cost_profile_cannot_supply_streaming_geometry_or_choose_x86():
  kernel = template()
  assert (
    lower(kernel, svl=None, cpu_cost_target="SME").source
    == lower(kernel, svl=None, enabled=False, cpu_cost_target="SME").source
  )
  code = lower(kernel, target="AVX512", cpu_cost_target="SME").source
  assert MARKER not in code


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("pitch", (64, -64))
def test_historical_capture_declares_write_access_and_keeps_saved_read(pitch, enabled):
  ordinary = lower(template(pitch=pitch), enabled=enabled)
  historical = lower(template(pitch=pitch, capture=True), enabled=enabled)
  assert ordinary.kernel.parameters["A"].access == "input"
  assert historical.kernel.parameters["A"].access == "inout"

  code = historical.source
  aliases = re.findall(r"const bool (dsl_c_alias_\d+) = ([^\n]+);", code)
  snapshots = re.findall(r"Buffer<float> (dsl_c_snapshot_\d+);", code)
  assert len(aliases) == len(snapshots) == 2
  saved_alias, saved_snapshot = aliases[0][0], snapshots[0]
  output_alias, output_snapshot = aliases[1][0], snapshots[1]
  write = code.index("std::int64_t{42}")
  renewal = code.index("Buffer<float> " + output_snapshot)
  assert code.index(saved_snapshot + ".resize(") < write < renewal
  assert f"({saved_alias}) ||" in aliases[1][1], "historical reads must force the protected output branch"

  # Renewing the output's alias snapshot must read the pre-write capture,
  # not the tensor overwritten with 42. The SME fast branch is gated by
  # output_alias, whose expression above includes the historical alias.
  copied = code[renewal : code.index(f"if ({output_alias}) /* overlapping operands */", renewal)]
  loads = [line for line in copied.splitlines() if "vecops::vec::load_convert(" in line]
  assert loads and all(saved_snapshot + ".data()" in line and saved_alias + " ?" in line for line in loads)
  assert output_snapshot + ".data()" in code[code.index("/* overlapping operands */", renewal) :]
  assert_no_vec_mask_arrays(code)


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64)))
@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("shape,pitch", ((None, 64), (None, -64), ((63, 129), 63), ((63, 129), -63)))
def test_cross_complete_owner_preserves_public_inline_and_old_typed_input(tmp_path, vl, svl, enabled, shape, pitch):
  clang, sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG"), os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("real LLVM ARM cross compiler and sysroot required")
  root = Path(__file__).resolve().parents[2]
  cpp = tmp_path / "transpose.cpp"
  generated = lower(template(pitch=pitch, shape=shape), vl=vl, svl=svl, enabled=enabled).source
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
  assembly_path = cpp.with_suffix(".s")
  assembly_command = ["-S" if token == "-c" else token for token in command[:-1]] + [str(assembly_path)]
  assembly_process = subprocess.run(assembly_command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".assembly.log").write_text(assembly_process.stdout + assembly_process.stderr)
  cpp.with_suffix(".assembly.command.json").write_text(json.dumps(assembly_command, indent=2) + "\n")
  assert assembly_process.returncode == 0, assembly_process.stderr[-6000:]
  owners = asm_utils.assert_sme_owners(assembly_path.read_text(), 1, equal_domains=vl == svl)
  owner = next(iter(owners.values()))
  assert not re.search(r"\sbl(?:r)?\s", owner), "no public conversion/ZA helper outline"
  assert re.search(r"\bld1w\s+\{za0h", owner), "original predicated ZA input must remain"
  assert not re.search(r"\b(?:str|st1\w*)\s+\{?z\d+[^\n]*\[sp", owner), "no new Z spill"
  cpp.with_suffix(".summary.json").write_text(
    json.dumps(
      {
        "VL": vl,
        "SVL": svl,
        "pitch": pitch,
        "shape": shape,
        "enabled": enabled,
        "whole_owner_scalar_SP_accesses": len(re.findall(r"\b(?:ldr|ldp|str|stp)\s+[xw]\d+[^\n]*\[sp", owner)),
        "compute_body_assembly_instructions": len(asm_utils._instructions(owner)[0]),
        "state_entry_and_compute_lifecycles_checked_by_CFG": True,
        "no_BL": True,
        "no_direct_Z_SP_store": True,
        "not_target_performance_evidence": True,
      },
      indent=2,
    )
    + "\n"
  )


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64)))
@pytest.mark.parametrize("pitch", (63, -63))
def test_cross_original_cpp_constant_shape_reference_owner(tmp_path, vl, svl, pitch):
  """Compile the unchanged public C++ operator at the same constant geometry."""
  clang, sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG"), os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("real LLVM ARM cross compiler and sysroot required")
  root = Path(__file__).resolve().parents[2]
  reference = root / "benchmarks/dsl_transpose_reference.cpp"
  cpp = tmp_path / "transpose.cpp"
  cpp.write_text(
    "#include <vecops/Kernel.h>\nnamespace vecops::spec { using Compute = float; }\n"
    + reference.read_text()
    + '\nextern "C" void transpose_cpp_fixed(void* ap, void* cp) {\n'
    "auto A=vecops::tensor::make_tensor(static_cast<float*>(ap),"
    "vecops::tensor::make_shape(vecops::meta::cint<63>,vecops::meta::cint<129>),"
    "vecops::tensor::make_strides(vecops::meta::cint<129>,vecops::meta::cint<1>));\n"
    "auto C=vecops::tensor::make_tensor(static_cast<vecops::float16_t*>(cp),"
    "vecops::tensor::make_shape(vecops::meta::cint<129>,vecops::meta::cint<63>),"
    f"vecops::tensor::make_strides(vecops::meta::cint<{pitch}>,vecops::meta::cint<1>));\n"
    "__kernel__(A,C,std::int64_t{1});\n}\n"
  )
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
  assembly_path = cpp.with_suffix(".s")
  assembly_command = ["-S" if token == "-c" else token for token in command[:-1]] + [str(assembly_path)]
  assembly_process = subprocess.run(assembly_command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".assembly.log").write_text(assembly_process.stdout + assembly_process.stderr)
  cpp.with_suffix(".assembly.command.json").write_text(json.dumps(assembly_command, indent=2) + "\n")
  assert assembly_process.returncode == 0, assembly_process.stderr[-6000:]
  asm_utils.assert_sme_owners(assembly_path.read_text(), 1, equal_domains=vl == svl)
  cpp.with_suffix(".summary.json").write_text(
    json.dumps({"VL": vl, "SVL": svl, "pitch": pitch, "shape": [63, 129], "reference": str(reference)}, indent=2) + "\n"
  )


@pytest.fixture(scope="module")
def native_pairs(tmp_path_factory):
  if os.uname().machine.lower() not in {"aarch64", "arm64"}:
    pytest.skip("actual ARM SME target required")
  directory = tmp_path_factory.mktemp("const-half-output")
  compiler = make_compiler(jobs=4, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  result = {}
  with vecops.collect_compile_requests() as requests:
    for pitch in (64, -64):
      for capture in (False, True):
        pair = tuple(
          T.compile(
            template(pitch=pitch, capture=capture),
            compiler=compiler,
            pass_options={OPTION: flag, INPUT_OPTION: False},
          )
          for flag in (False, True)
        )
        for kernel in pair:
          kernel.operator(
            vecops.TensorMeta((31, 33), dtype="float32"),
            vecops.TensorMeta((33, 31), strides=(pitch, 1), dtype="float16"),
          )
        result[pitch, capture] = pair
  vecops.compile_batch(requests.requests, parallelism=4)
  return result


def invoke_oracle(kernel, source, pitch, shift, *, capture):
  rows, columns = source.shape
  span = ((columns - 1) * abs(pitch) + rows) * 2 if rows and columns else 0
  start = 64 + (0 if shift is None else shift)
  raw = np.full(max(start + span, 64 + source.nbytes) + 64, 17, dtype=np.uint8)
  if shift is None:
    a = source.copy()
  else:
    a = np.ndarray(
      source.shape,
      dtype=np.float32,
      buffer=raw,
      offset=64,
      strides=(columns * source.itemsize, source.itemsize),
    )
    a[...] = source
  offset = start + ((columns - 1) * abs(pitch) * 2 if pitch < 0 and columns else 0)
  output = np.ndarray((columns, rows), dtype=np.float16, buffer=raw, offset=offset, strides=(pitch * 2, 2))
  with np.errstate(invalid="ignore"):
    oracle = a.copy().astype(np.float16).T.copy()
  saved = raw.copy()
  kernel(a, output)
  assert np.array_equal(output, oracle, equal_nan=True), (rows, columns, pitch, shift, capture, "independent oracle")
  # array_equal considers +0 and -0 equal. Check zero/infinity signs against
  # the independent conversion too, not merely agreement between both DSLs.
  # NaN payloads remain a same-machine old/new bitwise check below.
  signed_special = (oracle == 0) | np.isinf(oracle)
  assert np.array_equal(np.signbit(output[signed_special]), np.signbit(oracle[signed_special])), (
    rows,
    columns,
    pitch,
    shift,
    capture,
    "independent signed zero/infinity oracle",
  )
  assert np.array_equal(raw[:64], saved[:64]) and np.array_equal(
    raw[max(start + span, 64 + source.nbytes) :], saved[max(start + span, 64 + source.nbytes) :]
  ), "outside union canaries"
  if capture and shift is None:
    assert np.all(a == np.float32(42)), "the intervening source write was not dropped"
  return output.copy().view(np.uint16)


@pytest.mark.parametrize("shape", ((0, 0), (7, 0), (0, 7), (7, 9)))
@pytest.mark.parametrize("pitch", (64, -64))
@pytest.mark.parametrize("shift", (None, 0, 1, 16))
@pytest.mark.parametrize("capture", (False, True))
def test_oracle_input_metadata_obeys_declared_dense_layout(shape, pitch, shift, capture):
  import ast
  import inspect
  import textwrap

  source = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
  raw = np.full(64 + source.nbytes + 64, 17, dtype=np.uint8)
  # Evaluate the fixture's exact A-construction expression. In particular,
  # do not duplicate it with a hand-built TensorMeta or silently normalize
  # its strides. The output oracle is outside this metadata-only test.
  function = ast.parse(textwrap.dedent(inspect.getsource(invoke_oracle))).body[0]
  branch = next(node for node in function.body if isinstance(node, ast.If))
  assignment = branch.body[0] if shift is None else branch.orelse[0]
  assert isinstance(assignment, ast.Assign) and assignment.targets[0].id == "a"
  expression = ast.Expression(body=assignment.value)
  a = eval(
    compile(expression, "fixture_input_metadata", "eval"), {"np": np, "source": source, "raw": raw, "columns": shape[1]}
  )
  if shift is not None:
    assert a.strides == (shape[1] * source.itemsize, source.itemsize)
    assert a.base is raw and a.ctypes.data == raw.ctypes.data + 64
  view = vecops._C.TensorView(a, writable=capture)
  assert tuple(view.sizes) == shape
  assert tuple(view.strides) == (shape[1], 1)
  definition = vecops.KernelDef(
    "fixture::dense_alias_input",
    [
      vecops.TensorDef(
        "A",
        shape,
        (shape[1], 1),
        dtype="float32",
        access=vecops._C.TensorAccess.inout if capture else vecops._C.TensorAccess.input,
      )
    ],
  )
  # The real ndarray adapter is followed by strict schema validation, but
  # no kernel is invoked and none of the numerical oracle is replaced.
  vecops._C.bind_kernel_call(definition._native(), vecops._C.KernelCall([view]))


@pytest.mark.parametrize("pitch", (64, -64))
@pytest.mark.parametrize("capture", (False, True))
def test_arm_native_const_half_full_tails_signed_overlap_and_historical_capture(native_pairs, pitch, capture):
  from vecops._arm_vector_state import get_vector_state

  original = get_vector_state()
  for rows, columns in ((0, 0), (1, 1), (7, 9), (16, 32), (31, 33), (63, 129), (64, 127)):
    source = (((np.arange(rows * columns) % 251) - 125) / 8).astype(np.float32).reshape(rows, columns)
    if source.size >= 6:
      source.flat[:6] = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345, 0x7F812345], dtype=np.uint32).view(
        np.float32
      )
    for shift in (None, 0, 1, 16):
      old, new = (
        invoke_oracle(kernel, source, pitch, shift, capture=capture) for kernel in native_pairs[pitch, capture]
      )
      assert np.array_equal(old, new), (pitch, capture, rows, columns, shift, "same-target strict bits")
      assert get_vector_state() == original, "main-thread VL/SVL/SM state leaked"


@pytest.fixture
def arm_control(tmp_path):
  yield from _arm_vector_control.__wrapped__(tmp_path)


@pytest.mark.skipif(os.uname().machine.lower() not in {"aarch64", "arm64"}, reason="actual ARM unequal state gate")
def test_arm_native_const_half_workers_unequal_state_restore_and_stale_reject(tmp_path, arm_control, monkeypatch):
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
    INPUT_OPTION: False,
    "cost_profiles": (CostProfile("const-half-worker", policies=(("grid_worker_cycles", 0),)),),
  }

  def prepare(enabled):
    return T.compile(template(pitch=64, batch=True), compiler=compiler, pass_options={**options, OPTION: enabled})

  def run(kernel, pair, rows, columns):
    control.assert_pair(pair)
    a = (((np.arange(4 * rows * columns) % 251) - 125) / 8).astype(np.float32).reshape(4, rows, columns)
    with np.errstate(invalid="ignore"):
      expected = a.astype(np.float16).transpose(0, 2, 1).copy()
    plane = ((columns - 1) * 64 + rows) * 2
    raw = np.full(64 + 4 * plane + 64, 17, dtype=np.uint8)
    c = np.ndarray((4, columns, rows), dtype=np.float16, buffer=raw, offset=64, strides=(plane, 128, 2))
    kernel(a, c)
    assert np.array_equal(c, expected, equal_nan=True), (pair, rows, columns, "worker oracle")
    signed_special = (expected == 0) | np.isinf(expected)
    assert np.array_equal(np.signbit(c[signed_special]), np.signbit(expected[signed_special])), (
      pair,
      rows,
      columns,
      "independent worker signed zero/infinity oracle",
    )
    assert np.all(raw[:64] == 17) and np.all(raw[-64:] == 17), "worker canaries"
    control.assert_pair(pair)
    return c.copy().view(np.uint16)

  warm = prepare(False)
  run(warm, original, 31, 33)
  kernels = {}
  with vecops.collect_compile_requests() as requests:
    for pair in pairs:
      control.select(pair)
      for enabled in (False, True):
        kernel = prepare(enabled)
        kernel.operator(
          vecops.TensorMeta((4, 31, 33), dtype="float32"),
          vecops.TensorMeta((4, 33, 31), strides=((33 - 1) * 64 + 31, 64, 1), dtype="float16"),
        )
        kernels[pair, enabled] = kernel
  vecops.compile_batch(requests.requests, parallelism=4, execution_parallelism=1)
  for pair in pairs:
    control.select(pair)
    for rows, columns in ((32, 32), (31, 33), (63, 65)):
      old, new = (run(kernels[pair, enabled], pair, rows, columns) for enabled in (False, True))
      assert np.array_equal(old, new), (pair, rows, columns, "same-target worker bits")
  with pytest.raises(RuntimeError, match="SVE VL/SME SVL.*calling thread"):
    run(kernels[pairs[0], True], pairs[1], 31, 33)
  control.select(original)
  run(warm, original, 31, 33)
  # Imported fixture finally restores the complete original raw prctl flags.
