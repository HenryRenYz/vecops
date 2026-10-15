# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Private SVE pointwise permutations preserve ordered, strict external bits."""

import ctypes
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import (
  arm_cross_cxx_include_flags,
  assert_no_vec_mask_arrays,
  assert_source_algorithms_lowered,
  make_compiler,
)
from test_dsl_store_snapshot_regions import scope
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cost import CPUTarget
from vecops.language._compiler.backends.cpu.cpp.vectors import VectorsEmitter
from vecops.language._compiler.backends.cpu.planning import prepare
from vecops.language._compiler.backends.cpu.register_layout import PointwiseRegisterLayout
from vecops.language._compiler.ir.vector import VectorGroup

MARKER = "private pointwise register layout: paired native 16->32 conversions"


@pytest.mark.parametrize("target", ("SVE", "SME"))
@pytest.mark.parametrize("policy", ("off", "auto", "force"))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_hardware_matmul_plans_are_not_pointwise_regions_when_reprepared(target, policy, dtype):
  @T.template
  def kernel(
    A: T.Tensor((1, 129), dtype=dtype),
    B: T.Tensor((129, 257), (1, 129), dtype),
    C: T.Tensor((1, 257), dtype="float32", access="output"),
  ):
    T.dot((A, B), C)

  options = {"_vector_state": VectorState(32, 64)}
  if policy == "force":
    options["cpu_pointwise_register_layout"] = True
  else:
    options["cpu_pointwise_register_layout_policy"] = policy
  artifact = CPUBackend().lower(kernel.ir, (), target=target, options=options)
  assert_source_algorithms_lowered(artifact.backend_ir)
  # Matrix recipes now produce ordinary states/loops. Re-preparation must
  # retain those and never invent an opaque matrix or pointwise instruction.
  repeated = prepare(artifact.lowered)
  assert_source_algorithms_lowered(repeated)
  assert set(repeated.states) == set(artifact.backend_ir.states)


def pointwise_template(
  source="float16",
  coefficient="float16",
  output="float16",
  *,
  rows=3,
  width=129,
  mode="plain",
  unit=True,
  broadcast=False,
):
  input_inner = 1 if unit else T.dynamic()
  output_inner = 1 if unit else T.dynamic()
  n = 1 if broadcast else width

  @T.template
  def kernel(
    A: T.Tensor((rows, width), (T.dynamic(), input_inner), source, access="inout"),
    G: T.Tensor((n,), (1,), coefficient),
    B: T.Tensor((n,), (1,), coefficient),
    C: T.Tensor((rows, width), (T.dynamic(), output_inner), output, access="output"),
  ):
    value = T.load(A, dtype="float32")
    if mode == "historical":
      T.store(A, 42)
    if mode == "rounding":
      value = T.cast(T.cast(value, dtype="float16"), dtype="float32")
    if mode == "index":
      value = value + T.arange(width)
    if mode == "masked":
      value = T.load(A, dtype="float32", mask=T.arange(width) < width - 1, other=0)
    if mode == "state":
      for iteration in T.serial(2, unroll=1):
        value = value + 1
    if mode == "fma":
      result = T.fma(value - 0.25, T.load(G, dtype="float32"), T.load(B, dtype="float32"))
    else:
      result = ((value - 0.25) * 0.5) * T.load(G, dtype="float32") + T.load(B, dtype="float32")
    T.store(C, result)

  return kernel


def lower(template, *, target="SVE", vl=64, svl=64, enabled=True, **options):
  return CPUBackend().lower(
    template.ir,
    (),
    target=target,
    options={"_vector_state": VectorState(vl, svl), "cpu_pointwise_register_layout": enabled, **options},
  )


@pytest.mark.parametrize(
  "source,coefficient,output",
  (
    ("float16", "float16", "float16"),
    ("bfloat16", "bfloat16", "bfloat16"),
    ("float16", "bfloat16", "bfloat16"),
    ("bfloat16", "float16", "float16"),
  ),
)
def test_closed_full_pointwise_pairs_keep_ordered_tail_and_cold_alias_path(source, coefficient, output):
  artifact = lower(pointwise_template(source, coefficient, output))
  code = artifact.source
  assert MARKER in code
  full = code.split(MARKER, 1)[1].split("/* masked vector tail */", 1)[0]
  accesses = re.findall(r"vecops::vec::(?:load|store)_convert\([^;]+;", full)
  assert len(accesses) == 4
  assert all("cvt::unordered" in access for access in accesses)
  tail, _ = scope(code, "/* masked vector tail */", start=code.index(MARKER))
  assert "cvt::unordered" not in tail
  assert "else /* retain current and historical store snapshots */" in code
  cold, _ = scope(code, "else /* retain current and historical store snapshots */")
  assert "cvt::unordered" not in cold
  assert "dsl_c_snapshot_" in cold
  assert_no_vec_mask_arrays(code)


@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32), (128, 16), (None, None)))
def test_layout_does_not_assume_equal_or_64byte_vector_lengths(vl, svl):
  artifact = lower(pointwise_template(), vl=vl, svl=svl)
  assert MARKER in artifact.source
  assert "vecops::vec::ScalableTag<float, 1>" in artifact.source
  assert "vecops::vec::cvt::unordered" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("mode", ("historical", "rounding", "index", "masked", "state"))
def test_lane_dependent_captured_masked_and_intermediate_rounding_regions_stay_ordered(mode):
  artifact = lower(pointwise_template(mode=mode))
  assert "cvt::unordered" not in artifact.source
  if mode == "historical":
    assert "dsl_c_snapshot_" in artifact.source


@pytest.mark.parametrize(
  "kwargs",
  (
    {"unit": False},
    {"source": "float32"},
    {"coefficient": "float32"},
    {"output": "float32"},
  ),
)
def test_mixed_memory_ratio_and_nonunit_or_negative_axis_uses_existing_path(kwargs):
  assert "cvt::unordered" not in lower(pointwise_template(**kwargs)).source


@pytest.mark.parametrize("target", ("AVX512", "Scalar", "generic"))
def test_cost_override_never_changes_other_isa_layout(target):
  artifact = lower(pointwise_template(), target=target, cpu_cost_target=CPUTarget("arm", "sme", vector_bits=512))
  assert "cvt::unordered" not in artifact.source


def test_switch_scalar_broadcast_and_central_permutation_trait():
  template = pointwise_template(broadcast=True)
  assert MARKER in lower(template).source
  assert "cvt::unordered" not in lower(template, enabled=False).source
  with pytest.raises(TypeError, match="cpu_pointwise_register_layout must be bool"):
    lower(template, enabled=1)
  # arange is centrally ELEMENTWISE but is not permutation-equivariant.
  indexed = lower(pointwise_template(mode="index"))
  assert not indexed.backend_ir.register_layouts


def test_narrow_auto_default_preserves_explicit_disable_and_broad_force():
  template = pointwise_template()
  default = CPUBackend().lower(template.ir, (), target="SVE", options={"_vector_state": VectorState(64, 64)})
  disabled = lower(template, enabled=False)
  enabled = lower(template, enabled=True)
  automatic = CPUBackend().lower(
    template.ir,
    (),
    target="SVE",
    options={"_vector_state": VectorState(64, 64), "cpu_pointwise_register_layout_policy": "auto"},
  )
  assert default.source == automatic.source
  assert not disabled.backend_ir.register_layouts
  assert MARKER not in disabled.source
  assert "cvt::unordered" not in disabled.source
  assert MARKER in enabled.source
  assert enabled.backend_ir.register_layouts
  assert_no_vec_mask_arrays(default.source)


def test_materialized_snapshot_and_hardware_context_block_only_the_affected_region():
  class Probe(VectorsEmitter):
    def line(self, value):
      self.lines.append(value)

  region = SimpleNamespace()
  proof = PointwiseRegisterLayout("native-16-to-32", frozenset({12, 13}), ())
  emitter = Probe()
  emitter.prepared = SimpleNamespace(register_layouts={id(region): proof})
  emitter.lines = []
  emitter.hardware_scope = None
  emitter.snapshots, emitter.materialized, emitter.reduction_materialized = {}, {}, {}
  groups = (VectorGroup(0, 2, 1),)
  for attribute in ("snapshots", "materialized", "reduction_materialized"):
    setattr(emitter, attribute, {12: object()})
    with emitter.pointwise_layout(region, groups, full=True):
      assert emitter.pointwise_register_layout is None
    setattr(emitter, attribute, {})
  emitter.snapshots = {99: object()}  # An unrelated historical capture survives.
  with emitter.pointwise_layout(region, groups, full=True):
    assert emitter.pointwise_register_layout is proof
  assert emitter.snapshots.keys() == {99}
  assert emitter.pointwise_register_layout is None
  emitter.hardware_scope = ("sme", object())
  with emitter.pointwise_layout(region, groups, full=True):
    assert emitter.pointwise_register_layout is None
  emitter.hardware_scope = None
  for full, group in ((False, groups), (True, (VectorGroup(0, 1, 0),))):
    with emitter.pointwise_layout(region, group, full=full):
      assert emitter.pointwise_register_layout is None
  with pytest.raises(RuntimeError, match="test unwind"), emitter.pointwise_layout(region, groups, full=True):
    raise RuntimeError("test unwind")
  assert emitter.pointwise_register_layout is None


def test_actual_library_affine_is_selected_without_changing_statistics_or_fma():
  @T.template
  def kernel(
    A: T.Tensor((8, 4096), dtype="float16"),
    G: T.Tensor((4096,), dtype="float16"),
    B: T.Tensor((4096,), dtype="float16"),
    C: T.Tensor((8, 4096), dtype="float16", access="output"),
  ):
    normalized = T.library.layernorm(T.load(A, dtype="float32"), variance="moments")
    T.store(C, normalized * T.load(G, dtype="float32") + T.load(B, dtype="float32"))

  artifact = lower(kernel)
  original = lower(kernel, enabled=False)
  assert MARKER in artifact.source
  stats = artifact.source.split("/* reduction region:", 1)[1].split("/* private pointwise", 1)[0]
  assert "cvt::unordered" not in stats
  assert "vecops::vec::mul(" in stats
  # Until the first output body's marker, source includes all statistics and
  # their captures/guards, not just a count of unchanged reduction tokens.
  prefix = artifact.source.split(f"/* {MARKER} */", 1)[0]
  assert original.source.startswith(prefix)
  assert "vecops::vec::fma(" not in artifact.source
  assert restore_ordered_options(artifact.source) == original.source
  assert_no_vec_mask_arrays(artifact.source)


def restore_ordered_options(source):
  """The physical plan may change only conversion options and its marker."""
  result = re.sub(rf"^\s*/\* {re.escape(MARKER)} \*/\n", "", source, flags=re.MULTILINE)
  return result.replace(", vecops::vec::cvt::unordered", "")


def test_uniform_loop_scalars_keep_evaluation_inside_the_original_iteration():
  @T.template
  def kernel(
    A: T.Tensor((3, 129), dtype="float16"),
    C: T.Tensor((3, 129), dtype="float16", access="output"),
    count: int = 2,
  ):
    for iteration in T.serial(count, unroll=1):
      scale = T.cast(iteration, "float32") + 0.25
      T.store(C, T.load(A, dtype="float32") * scale)

  original, artifact = lower(kernel, enabled=False), lower(kernel)
  assert MARKER in artifact.source
  assert restore_ordered_options(artifact.source) == original.source
  assert_no_vec_mask_arrays(artifact.source)


def library_template(storage, operation, affine):
  N = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((3, N), (T.dynamic(), 1), storage),
    G: T.Tensor((N,), dtype=storage),
    B: T.Tensor((N,), dtype=storage),
    C: T.Tensor((3, N), (T.dynamic(), 1), storage, access="output"),
  ):
    value = T.load(A, dtype="float32")
    if operation == "softmax":
      result = T.library.softmax(value, accuracy="strict")
    else:
      result = T.library.layernorm(value, variance=operation)
    if affine:
      result = result * T.load(G, dtype="float32") + T.load(B, dtype="float32")
    T.store(C, result)

  return kernel


@pytest.mark.parametrize("storage", ("float16", "bfloat16"))
@pytest.mark.parametrize("operation", ("moments", "centered", "softmax"))
@pytest.mark.parametrize("affine", (False, True))
def test_dynamic_library_layout_changes_only_complete_output_memory_options(storage, operation, affine):
  template = library_template(storage, operation, affine)
  original, candidate = lower(template, enabled=False), lower(template)
  assert restore_ordered_options(candidate.source) == original.source
  if operation != "softmax":
    assert MARKER in candidate.source
  assert_no_vec_mask_arrays(candidate.source)


def cross_wrapper(source, source_dtype, coefficient_dtype, output_dtype):
  types = {"float16": "vecops::float16_t", "bfloat16": "vecops::bfloat16_t"}
  definitions = [
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism=vecops::meta::cint<1>; }",
    source,
    'extern "C" void pointwise_pairs(void* ap,void* gp,void* bp,void* cp,vecops::nint_t n,vecops::nint_t as0,vecops::nint_t cs0) {',
    "auto shape=vecops::tensor::make_shape(vecops::meta::cint<3>,vecops::meta::Any{n});",
    "auto coefficient_shape=vecops::tensor::make_shape(vecops::meta::Any{n});",
  ]
  for name, pointer, dtype, outer in (
    ("A", "ap", source_dtype, "as0"),
    ("G", "gp", coefficient_dtype, None),
    ("B", "bp", coefficient_dtype, None),
    ("C", "cp", output_dtype, "cs0"),
  ):
    shape = "shape" if outer else "coefficient_shape"
    strides = (
      f"vecops::tensor::make_strides(vecops::meta::Any{{{outer}}},vecops::meta::cint<1>)"
      if outer
      else "vecops::tensor::make_strides(vecops::meta::cint<1>)"
    )
    definitions.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<{types[dtype]}*>({pointer}),{shape},{strides});"
    )
  definitions.extend(("__kernel__(A,G,B,C);", "}"))
  return "\n".join(definitions)


def linear_pointwise_loops(assembly):
  """Identify innermost full converting loops, not cold snapshot gathers.

  The retained alias branch may legitimately call native mask-tuple helpers.
  A straight-line complete loop contains direct half-memory inputs/output,
  arithmetic, and just its final conditional back edge.
  """
  lines = assembly.splitlines()
  labels = {line.split(":", 1)[0]: index for index, line in enumerate(lines) if line.startswith(".LBB")}
  loops = []
  for end, line in enumerate(lines):
    match = re.search(r"^\s*b\.[a-z]+\s+(\.LBB\w+)", line)
    if not match or match[1] not in labels or labels[match[1]] >= end:
      continue
    start = labels[match[1]]
    block = "\n".join(lines[start : end + 1])
    if (
      re.search(r"\bld1h\b", block)
      and re.search(r"\bst1h\b", block)
      and re.search(r"\bfmul\b", block)
      and not any(item.startswith(".LBB") for item in lines[start + 1 : end])
    ):
      loops.append(block)
  return loops


@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32)))
@pytest.mark.parametrize(
  "types",
  (
    ("float16", "float16", "float16"),
    ("float16", "bfloat16", "bfloat16"),
  ),
)
@pytest.mark.parametrize("enabled", (False, True))
def test_real_cross_full_pointwise_uses_native_pair_instructions_not_helpers(tmp_path, vl, svl, types, enabled):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for actual ARM cross compilation")
  root = Path(__file__).resolve().parents[2]
  # Dynamic width keeps a real full-vector back edge for every VL; a constant
  # small width may be completely expanded by LLVM into the row loop.
  source = lower(pointwise_template(*types, width=T.dynamic(lower=0)), vl=vl, svl=svl, enabled=enabled).source
  assert_no_vec_mask_arrays(source)
  cpp, assembly, object_file = tmp_path / "pointwise.cpp", tmp_path / "pointwise.s", tmp_path / "pointwise.o"
  cpp.write_text(cross_wrapper(source, *types))
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9-a+sve2+bf16+sme+sme-fa64",
    "-msve-vector-bits=scalable",
    "-std=c++20",
    "-O3",
    "-ffp-contract=off",
    "-DNDEBUG",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
    f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={svl * 8}",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
  ]
  commands = []
  for output, flag in ((assembly, "-S"), (object_file, "-c")):
    call = [*command, flag, str(cpp), "-o", str(output)]
    commands.append(call)
    result = subprocess.run(call, capture_output=True, text=True, check=False)
    output.with_suffix(output.suffix + ".diagnostics").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-8000:]
  cpp.with_suffix(".commands.json").write_text(json.dumps(commands, indent=2))
  elf = object_file.read_bytes()
  assert elf[:4] == b"\x7fELF" and int.from_bytes(elf[18:20], "little") == 183
  code = assembly.read_text()
  assert bool(re.search(r"\bfcvtlt\b", code)) is enabled
  if types[-1] == "float16":
    assert bool(re.search(r"\bfcvtnt\b", code)) is enabled
  full_loops = linear_pointwise_loops(code)
  assert full_loops, "missing complete direct-memory pointwise loop"
  if enabled:
    full_loops = [loop for loop in full_loops if re.search(r"\bfcvtlt\b", loop)]
    assert full_loops, "private register-layout loop was not emitted"
    assert all(not re.search(r"\b(?:uunpklo|uunpkhi|uzp1|whilelo)\b", loop) for loop in full_loops)
  for loop in full_loops:
    assert not re.search(r"\bbl\b", loop), loop
    assert not re.search(r"\b(?:ldr|str)\s+[zp]\d+.*\bsp\b", loop), loop
  cpp.with_suffix(".hot-loops.json").write_text(json.dumps(full_loops, indent=2))
  assert not re.search(r"^\s*(?:smstart|smstop)\b", code, re.MULTILINE)


@pytest.fixture(scope="module")
def native_pointwise_pairs(tmp_path_factory):
  directory = tmp_path_factory.mktemp("pointwise-register-layout")
  compiler = make_compiler(output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  result = {}
  with vecops.collect_compile_requests() as collector:
    for enabled in (False, True):
      for mode, unit, broadcast in (
        ("plain", True, False),
        ("fma", True, False),
        ("historical", True, False),
        ("plain", False, False),
        ("plain", True, True),
        ("rounding", True, False),
      ):
        template = pointwise_template(mode=mode, unit=unit, broadcast=broadcast)
        compiled = T.compile(template, compiler=compiler, pass_options={"cpu_pointwise_register_layout": enabled})
        n = 1 if broadcast else 129
        compiled.operator(
          vecops.TensorMeta((3, 129), dtype="float16"),
          vecops.TensorMeta((n,), dtype="float16"),
          vecops.TensorMeta((n,), dtype="float16"),
          vecops.TensorMeta((3, 129), dtype="float16"),
          **compiled._core.specialization_values,
        )
        result[enabled, mode, unit, broadcast] = compiled
      for types in (
        ("float16", "bfloat16", "bfloat16"),
        ("bfloat16", "float16", "float16"),
        ("bfloat16", "bfloat16", "bfloat16"),
      ):
        template = pointwise_template(*types)
        compiled = T.compile(template, compiler=compiler, pass_options={"cpu_pointwise_register_layout": enabled})
        compiled.operator(
          vecops.TensorMeta((3, 129), dtype=types[0]),
          vecops.TensorMeta((129,), dtype=types[1]),
          vecops.TensorMeta((129,), dtype=types[1]),
          vecops.TensorMeta((3, 129), dtype=types[2]),
          **compiled._core.specialization_values,
        )
        result[(enabled, *types)] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return result


@pytest.mark.parametrize(
  "mode,unit,broadcast",
  (
    ("plain", True, False),
    ("fma", True, False),
    ("historical", True, False),
    ("plain", False, False),
    ("plain", True, True),
    ("rounding", True, False),
  ),
)
def test_native_strict_bits_aliases_negative_stride_rounding_and_uniform_broadcast(
  native_pointwise_pairs, mode, unit, broadcast
):
  width = 129
  original = (
    (((np.arange(3 * width, dtype=np.float32) % 113) - 56) / np.float32(17)).reshape(3, width).astype(np.float16)
  )
  n = 1 if broadcast else width
  gamma = ((np.arange(n, dtype=np.float32) % 13) / np.float32(19) - np.float32(0.5)).astype(np.float16)
  beta = ((np.arange(n, dtype=np.float32) % 17) / np.float32(23) - np.float32(0.25)).astype(np.float16)
  libc = ctypes.CDLL(None)
  fmaf = libc.fmaf
  fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
  fmaf.restype = ctypes.c_float
  for layout in ("dense", "negative-row", "inplace", "shifted", "coefficient-alias", "nonunit"):
    if layout == "nonunit" and unit:
      continue
    outputs = []
    for enabled in (False, True):
      storage = np.empty(3 * width + 1, dtype=np.float16)
      a = storage[:-1].reshape(3, width)
      a[:] = original
      c = np.empty_like(original)
      g = gamma.copy()
      if layout == "inplace":
        c = a
      elif layout == "shifted":
        c = storage[1:].reshape(3, width)
      elif layout == "negative-row":
        a, c = a[::-1], c[::-1]
      elif layout == "nonunit":
        a = a[:, ::-1]
        output_storage = np.full((3, 2 * width), 13, dtype=np.float16)
        c = output_storage[:, ::2]
      elif layout == "coefficient-alias":
        c[0, :n] = gamma
        g = c[0, :n]
      source = a.astype(np.float32)
      coefficient = g.astype(np.float32).copy()
      if mode == "historical":
        # The value is captured before T.store(A,42), including when C aliases.
        assert not np.all(source == 42)
      compiled = native_pointwise_pairs[enabled, mode, unit, broadcast]
      compiled.operator.native(a, g, beta, c, **compiled._core.specialization_values)
      outputs.append(c.copy().view(np.uint16))
      if mode == "fma":
        shifted = source - np.float32(0.25)
        expected = np.empty_like(source)
        for row in range(3):
          for column in range(width):
            index = 0 if broadcast else column
            expected[row, column] = fmaf(shifted[row, column], coefficient[index], beta[index])
        expected = expected.astype(np.float16)
      else:
        if mode == "rounding":
          source = source.astype(np.float16).astype(np.float32)
        expected = (((source - np.float32(0.25)) * np.float32(0.5)) * coefficient + beta.astype(np.float32)).astype(
          np.float16
        )
      np.testing.assert_array_equal(c.view(np.uint16), expected.view(np.uint16), err_msg=(mode, layout, enabled))
      if layout == "nonunit":
        np.testing.assert_array_equal(output_storage[:, 1::2], np.float16(13), err_msg=(mode, layout, enabled))
    np.testing.assert_array_equal(outputs[0], outputs[1], err_msg=(mode, layout))


@pytest.mark.parametrize(
  "types",
  (
    ("float16", "float16", "float16"),
    ("float16", "bfloat16", "bfloat16"),
    ("bfloat16", "float16", "float16"),
    ("bfloat16", "bfloat16", "bfloat16"),
  ),
)
def test_native_mixed_storage_and_special_bits_keep_same_machine_layout_parity(native_pointwise_pairs, types):
  ml_dtypes = pytest.importorskip("ml_dtypes")
  dtypes = {"float16": np.float16, "bfloat16": ml_dtypes.bfloat16}
  # Raw signaling/quiet NaNs, signed zeros, infinities and finite extremes
  # exercise both full pairs and the ordered remainder. The numerical oracle
  # only compares finite bits/NaN classification across architectures; on/off
  # execution on this same machine must preserve every output bit.
  patterns = {
    "float16": np.array(
      [0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E31, 0x7D12, 0x0001, 0x7BFF], dtype=np.uint16
    ),
    "bfloat16": np.array(
      [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC3, 0x7F91, 0x0001, 0x7F7F], dtype=np.uint16
    ),
  }
  original = np.resize(patterns[types[0]], (3, 129)).copy().view(dtypes[types[0]])
  gamma = np.linspace(-0.75, 0.75, 129, dtype=np.float32).astype(dtypes[types[1]])
  beta = np.linspace(0.25, -0.25, 129, dtype=np.float32).astype(dtypes[types[1]])
  for layout in ("dense", "negative-row", "inplace"):
    if layout == "inplace" and types[0] != types[2]:
      continue
    outputs = []
    for enabled in (False, True):
      a = original.copy()
      c = np.empty((3, 129), dtype=dtypes[types[2]])
      if layout == "inplace":
        c = a
      elif layout == "negative-row":
        a, c = a[::-1], c[::-1]
      source = a.astype(np.float32).copy()
      with np.errstate(invalid="ignore", over="ignore"):
        expected = (
          ((source - np.float32(0.25)) * np.float32(0.5)) * gamma.astype(np.float32) + beta.astype(np.float32)
        ).astype(dtypes[types[2]])
      key = (enabled, "plain", True, False) if types == ("float16",) * 3 else (enabled, *types)
      compiled = native_pointwise_pairs[key]
      compiled.operator.native(a, gamma, beta, c, **compiled._core.specialization_values)
      finite = ~np.isnan(expected)
      np.testing.assert_array_equal(np.isnan(c), np.isnan(expected), err_msg=(types, layout, enabled))
      np.testing.assert_array_equal(
        c.view(np.uint16)[finite], expected.view(np.uint16)[finite], err_msg=(types, layout, enabled)
      )
      outputs.append(c.view(np.uint16).copy())
    np.testing.assert_array_equal(outputs[0], outputs[1], err_msg=(types, layout))


@pytest.fixture(scope="module")
def native_library_layout_pairs(tmp_path_factory):
  directory = tmp_path_factory.mktemp("library-register-layout")
  compiler = make_compiler(output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  result = {}
  with vecops.collect_compile_requests() as collector:
    for enabled in (False, True):
      for storage in ("float16", "bfloat16"):
        for operation in ("moments", "centered", "softmax"):
          for affine in (False, True):
            template = library_template(storage, operation, affine)
            compiled = T.compile(template, compiler=compiler, pass_options={"cpu_pointwise_register_layout": enabled})
            compiled.operator(
              vecops.TensorMeta((3, 129), dtype=storage),
              vecops.TensorMeta((129,), dtype=storage),
              vecops.TensorMeta((129,), dtype=storage),
              vecops.TensorMeta((3, 129), dtype=storage),
              **compiled._core.specialization_values,
            )
            result[enabled, storage, operation, affine] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return result


@pytest.mark.parametrize("storage", ("float16", "bfloat16"))
@pytest.mark.parametrize("operation", ("moments", "centered", "softmax"))
@pytest.mark.parametrize("affine", (False, True))
def test_native_library_same_machine_strict_bits_full_tails_and_aliases(
  native_library_layout_pairs, storage, operation, affine
):
  ml_dtypes = pytest.importorskip("ml_dtypes")
  dtype = np.float16 if storage == "float16" else ml_dtypes.bfloat16
  for width in (0, 1, 31, 32, 33, 128, 129, 4096):
    values = (((np.arange(3 * width, dtype=np.float32) % 97) - 48) / np.float32(17)).reshape(3, width)
    gamma = np.linspace(-0.75, 0.75, width, dtype=np.float32).astype(dtype)
    beta = np.linspace(0.25, -0.25, width, dtype=np.float32).astype(dtype)
    for special in (False, True):
      original = values.astype(dtype)
      if special and width:
        original[0, 0] = np.nan
        original[1, 0] = np.inf
        original[2, :] = -0.0
        if width > 1:
          original[1, 1] = -np.inf
          original[2, 1] = 0.0
      for layout in ("dense", "negative-row", "inplace", "shifted"):
        outputs = []
        for enabled in (False, True):
          backing = np.empty(3 * width + 1, dtype=dtype)
          a = backing[:-1].reshape(3, width)
          a[:] = original
          c = np.empty_like(original)
          if layout == "inplace":
            c = a
          elif layout == "shifted":
            c = backing[1:].reshape(3, width)
          elif layout == "negative-row":
            a, c = a[::-1], c[::-1]
          compiled = native_library_layout_pairs[enabled, storage, operation, affine]
          compiled.operator.native(a, gamma, beta, c, **compiled._core.specialization_values)
          outputs.append(c.copy().view(np.uint16))
        np.testing.assert_array_equal(
          outputs[0], outputs[1], err_msg=(storage, operation, affine, width, special, layout)
        )
