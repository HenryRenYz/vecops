"""Narrow auto profitability is separate from the broad forced experiment."""

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import arm_cross_cxx_include_flags, assert_no_vec_mask_arrays, make_compiler
from test_dsl_pointwise_register_layout import MARKER, library_template, linear_pointwise_loops, restore_ordered_options
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cost import CPUTarget
from vecops.language._compiler.backends.cpu.cost.register_layout import conversion_query
from vecops.language._compiler.cost import CostProfile, CostQuery, CostRule, OperationCost
from vecops.language._compiler.environment import freeze


def controlled_template(inputs=3, dtype="float16", *, width=129, mode="plain", unit=True, broadcast=False):
  inner_input = 1 if unit else T.dynamic()
  inner_output = 1 if unit else T.dynamic()
  third_width = 1 if broadcast else width
  third_dtype = "bfloat16" if mode == "mixed" else dtype

  @T.template
  def kernel(
    A: T.Tensor((3, width), (T.dynamic(), inner_input), dtype, access="inout"),
    G: T.Tensor((width,), dtype=dtype),
    B: T.Tensor((third_width,), dtype=third_dtype),
    D: T.Tensor((width,), dtype=dtype),
    C: T.Tensor((3, width), (T.dynamic(), inner_output), dtype, access="output"),
  ):
    x = T.load(A, dtype="float32")
    if mode == "historical":
      T.store(A, 42)
    if mode == "rounding":
      x = T.cast(T.cast(x, dtype="float16"), dtype="float32")
    y = (x - 0.25) * 0.5
    if inputs >= 2:
      g = T.load(G, dtype="float32")
      y = y * g
    if inputs >= 3:
      if mode == "repeated":
        y = y + g
      else:
        y = y + T.load(B, dtype="float32")
    if inputs >= 4:
      y = y * T.load(D, dtype="float32")
    T.store(C, y)

  return kernel


def lower(template, *, policy="auto", target="SVE", vl=64, svl=64, **options):
  return CPUBackend().lower(
    template.ir,
    (),
    target=target,
    options={"_vector_state": VectorState(vl, svl), "cpu_pointwise_register_layout_policy": policy, **options},
  )


def selected_decision(artifact):
  choices = tuple(artifact.backend_ir.register_layout_decisions.values())
  assert choices
  return next((choice for choice in choices if choice.selected), choices[0])


@pytest.mark.parametrize("inputs", (1, 2, 3, 4))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_auto_uses_independent_physical_loads_not_library_names(inputs, dtype):
  artifact = lower(controlled_template(inputs, dtype))
  selected = dtype == "float16" and inputs >= 3
  assert (MARKER in artifact.source) is selected
  decision = selected_decision(artifact)
  assert decision.selected is selected
  if dtype == "float16":
    assert decision.independent_paired_loads == inputs
  if selected:
    assert decision.ordered_cycles - decision.packed_cycles >= 2
    assert 0 < decision.live_registers <= 20
    assert decision.pointwise_compute_cycles <= 16
    assert all("not measured latency" in value for value in decision.cost_provenance)
  else:
    assert artifact.source == lower(controlled_template(inputs, dtype), policy="off").source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("mode,broadcast", (("repeated", False), ("plain", True), ("mixed", False)))
def test_shared_producers_broadcasts_and_mixed_dtype_do_not_reach_three_pairs(mode, broadcast):
  artifact = lower(controlled_template(mode=mode, broadcast=broadcast))
  assert MARKER not in artifact.source
  choice = selected_decision(artifact)
  assert not choice.selected
  if mode != "mixed":
    assert choice.independent_paired_loads == 2


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_auto_keeps_force_off_and_default_contracts_and_validates_policy(dtype):
  template = controlled_template(dtype=dtype)
  forced = lower(template, cpu_pointwise_register_layout=True)
  assert MARKER in forced.source
  assert MARKER not in lower(controlled_template(), cpu_pointwise_register_layout=False).source
  default = CPUBackend().lower(template.ir, (), target="SVE", options={"_vector_state": VectorState(64, 64)})
  assert default.source == lower(template, policy="auto").source
  assert (MARKER in default.source) is (dtype == "float16")
  for invalid in (True, None, [], "force"):
    with pytest.raises(ValueError, match="cpu_pointwise_register_layout_policy"):
      lower(template, policy=invalid)
  with pytest.raises(TypeError, match="cpu_pointwise_register_layout must be bool"):
    lower(template, cpu_pointwise_register_layout=1)


@pytest.mark.parametrize(
  "policy,value",
  (
    ("pointwise_layout_register_budget", 2),
    ("pointwise_layout_compute_budget", 1),
    ("pointwise_layout_min_saved_cycles", 100),
  ),
)
def test_central_policy_can_conservatively_reject_a_legal_region(policy, value):
  override = CostProfile("unit-layout-policy", policies=((policy, value),))
  artifact = lower(controlled_template(), cost_profiles=(override,))
  assert MARKER not in artifact.source
  assert not selected_decision(artifact).selected


def test_cost_rules_are_implementation_scoped_and_pointwise_costs_are_central():
  ordinary = CostQuery("load", "memory", "float32", "float16", access="contiguous")
  before = freeze({"cpu_cost_target": "generic"}).costs.lookup(ordinary)
  sve = freeze({"cpu_cost_target": "SVE"}).costs
  assert sve.lookup(ordinary) == before
  for implementation in ("sve-ordered-pair", "sve-packed-even-odd"):
    cost = sve.lookup(conversion_query("load", "float16", implementation))
    assert cost != before
    assert "heuristic" in cost.provenance and "not measured latency" in cost.provenance
  expensive = CostProfile("unit-pointwise-cost", (CostRule(OperationCost(12, 12), (("operation", "mul"),)),))
  artifact = lower(controlled_template(), cost_profiles=(expensive,))
  assert MARKER not in artifact.source
  assert "compute-heavy" in selected_decision(artifact).reason
  unknown_cost = lower(controlled_template(), cpu_cost_target="AVX512")
  assert MARKER not in unknown_cost.source
  assert "cost rule" in selected_decision(unknown_cost).reason


@pytest.mark.parametrize("target", ("AVX512", "Scalar", "generic"))
def test_cost_override_cannot_enable_other_architectures(target):
  artifact = lower(controlled_template(), target=target, cpu_cost_target=CPUTarget("arm", "sme", vector_bits=512))
  assert MARKER not in artifact.source
  assert not artifact.backend_ir.register_layout_decisions


@pytest.mark.parametrize("mode,unit", (("historical", True), ("rounding", True), ("plain", False)))
def test_auto_keeps_historical_intermediate_rounding_and_nonunit_accesses_ordered(mode, unit):
  source = lower(controlled_template(mode=mode, unit=unit)).source
  assert MARKER not in source
  if mode == "historical":
    assert "dsl_c_snapshot_" in source


@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32), (128, 16), (None, None)))
def test_auto_geometries_use_ordinary_vl_not_streaming_length(vl, svl):
  source = lower(controlled_template(), vl=vl, svl=svl).source
  assert MARKER in source
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
@pytest.mark.parametrize("operation", ("moments", "centered", "softmax"))
@pytest.mark.parametrize("affine", (False, True))
def test_actual_library_uniform_statistics_auto_choice_keeps_prefix_and_tail_exact(dtype, operation, affine):
  template = library_template(dtype, operation, affine)
  candidate, original = lower(template), lower(template, policy="off")
  selected = dtype == "float16" and operation != "softmax" and affine
  assert (MARKER in candidate.source) is selected
  assert restore_ordered_options(candidate.source) == original.source
  if selected:
    prefix = candidate.source.split(f"/* {MARKER} */", 1)[0]
    assert original.source.startswith(prefix)
    assert selected_decision(candidate).independent_paired_loads == 3
  assert_no_vec_mask_arrays(candidate.source)


def cross_wrapper(source):
  definitions = [
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism=vecops::meta::cint<1>; }",
    source,
    'extern "C" void controlled_pairs(vecops::float16_t* ap,vecops::float16_t* gp,vecops::float16_t* bp,vecops::float16_t* dp,vecops::float16_t* cp,vecops::nint_t n) {',
    "auto shape=vecops::tensor::make_shape(vecops::meta::cint<3>,vecops::meta::Any{n});",
    "auto coefficient_shape=vecops::tensor::make_shape(vecops::meta::Any{n});",
    "auto strides=vecops::tensor::make_strides(vecops::meta::Any{n},vecops::meta::cint<1>);",
    "auto coefficient_strides=vecops::tensor::make_strides(vecops::meta::cint<1>);",
    "auto A=vecops::tensor::make_tensor(ap,shape,strides);",
    "auto G=vecops::tensor::make_tensor(gp,coefficient_shape,coefficient_strides);",
    "auto B=vecops::tensor::make_tensor(bp,coefficient_shape,coefficient_strides);",
    "auto D=vecops::tensor::make_tensor(dp,coefficient_shape,coefficient_strides);",
    "auto C=vecops::tensor::make_tensor(cp,shape,strides);",
    "__kernel__(A,G,B,D,C);",
    "}",
  ]
  return "\n".join(definitions)


@pytest.mark.parametrize("inputs", (2, 3, 4))
@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32)))
@pytest.mark.parametrize("policy", ("off", "auto"))
def test_real_cross_controlled_pair_cost_candidate_is_inline_without_spills(tmp_path, inputs, vl, svl, policy):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for actual ARM cross compilation")
  root = Path(__file__).resolve().parents[2]
  source = lower(controlled_template(inputs, width=T.dynamic(lower=0)), policy=policy, vl=vl, svl=svl).source
  assert_no_vec_mask_arrays(source)
  cpp, assembly, obj = tmp_path / "controlled.cpp", tmp_path / "controlled.s", tmp_path / "controlled.o"
  cpp.write_text(cross_wrapper(source))
  flags = [
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
  for destination, flag in ((assembly, "-S"), (obj, "-c")):
    command = [*flags, flag, str(cpp), "-o", str(destination)]
    commands.append(command)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    destination.with_suffix(destination.suffix + ".diagnostics").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-8000:]
  cpp.with_suffix(".commands.json").write_text(json.dumps(commands, indent=2))
  elf = obj.read_bytes()
  assert elf[:4] == b"\x7fELF" and int.from_bytes(elf[18:20], "little") == 183
  loops = linear_pointwise_loops(assembly.read_text())
  assert loops
  selected = policy == "auto" and inputs >= 3
  assert bool(re.search(r"\bfcvtlt\b", assembly.read_text())) is selected
  if selected:
    loops = [loop for loop in loops if re.search(r"\bfcvtlt\b", loop)]
    assert loops
    assert all(not re.search(r"\b(?:uunpklo|uunpkhi|uzp1|whilelo)\b", loop) for loop in loops)
  for loop in loops:
    assert not re.search(r"\bbl\b", loop)
    assert not re.search(r"\b(?:ldr|str)\s+[zp]\d+.*\bsp\b", loop)
  cpp.with_suffix(".hot-loops.json").write_text(json.dumps(loops, indent=2))


@pytest.fixture(scope="module")
def native_auto_pairs(tmp_path_factory):
  root = tmp_path_factory.mktemp("pointwise-auto")
  compiler = make_compiler(output="quiet", cache_dir=root / "cache", build_dir=root / "build")
  result = {}
  with vecops.collect_compile_requests() as collector:
    for dtype in ("float16", "bfloat16"):
      for inputs in (2, 3, 4):
        for policy in ("off", "auto"):
          template = controlled_template(inputs, dtype, width=T.dynamic(lower=0))
          compiled = T.compile(
            template, compiler=compiler, pass_options={"cpu_pointwise_register_layout_policy": policy}
          )
          compiled.operator(
            vecops.TensorMeta((3, 129), dtype=dtype),
            *(vecops.TensorMeta((129,), dtype=dtype) for _ in range(3)),
            vecops.TensorMeta((3, 129), dtype=dtype),
            **compiled._core.specialization_values,
          )
          result[dtype, inputs, policy] = compiled
      for operation in ("moments", "centered"):
        for affine in (False, True):
          for policy in ("off", "auto"):
            template = library_template(dtype, operation, affine)
            compiled = T.compile(
              template, compiler=compiler, pass_options={"cpu_pointwise_register_layout_policy": policy}
            )
            compiled.operator(
              vecops.TensorMeta((3, 129), dtype=dtype),
              vecops.TensorMeta((129,), dtype=dtype),
              vecops.TensorMeta((129,), dtype=dtype),
              vecops.TensorMeta((3, 129), dtype=dtype),
              **compiled._core.specialization_values,
            )
            result[dtype, operation, affine, policy] = compiled
  vecops.compile_batch(collector.requests, parallelism=4)
  return result


@pytest.mark.parametrize("inputs", (2, 3, 4))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_native_controlled_auto_strict_bits_full_tails_specials_and_aliases(native_auto_pairs, inputs, dtype):
  if dtype == "bfloat16":
    memory_dtype = pytest.importorskip("ml_dtypes").bfloat16
  else:
    memory_dtype = np.float16
  random = np.random.default_rng(345 + inputs)
  for width in (0, 1, 31, 32, 33, 127, 128, 129, 4096):
    for special in (False, True):
      data = random.normal(size=(3, width)).astype(memory_dtype)
      g, b, d = (random.normal(size=width).astype(memory_dtype) for _ in range(3))
      if special and width:
        bits = data.view(np.uint16).reshape(-1)
        patterns = (
          (0x7C01, 0x7E00, 0x7C00, 0xFC00, 0x8000, 0)
          if dtype == "float16"
          else (0x7F81, 0x7FC0, 0x7F80, 0xFF80, 0x8000, 0)
        )
        bits[: min(bits.size, len(patterns))] = patterns[: bits.size]
      for layout in ("dense", "negative-row", "inplace", "shifted", "coefficient-alias"):
        outputs = []
        for policy in ("off", "auto"):
          storage = np.zeros((3, width + 1), dtype=memory_dtype)
          a = storage[:, :width]
          a[:] = data
          c = np.empty_like(data)
          own_g = g.copy()
          if layout == "negative-row":
            a = a[::-1]
            c = c[::-1]
          elif layout == "inplace":
            c = a
          elif layout == "shifted":
            c = storage[:, 1:]
          elif layout == "coefficient-alias":
            backing = np.zeros((3, width + 1), dtype=memory_dtype)
            own_g = backing[0, :width]
            own_g[:] = g
            c = backing[:, :width]
          original = a.astype(np.float32)
          with np.errstate(all="ignore"):
            expected = (original - np.float32(0.25)) * np.float32(0.5)
            expected = expected * own_g.astype(np.float32)
            if inputs >= 3:
              expected = expected + b.astype(np.float32)
            if inputs >= 4:
              expected = expected * d.astype(np.float32)
            expected = expected.astype(memory_dtype)
          native_auto_pairs[dtype, inputs, policy](a, own_g, b, d, c)
          finite = np.isfinite(expected)
          np.testing.assert_array_equal(
            c.view(np.uint16)[finite],
            expected.view(np.uint16)[finite],
            err_msg=f"{dtype=} {inputs=} {width=} {layout=} {policy=} {special=}",
          )
          np.testing.assert_array_equal(np.isnan(c), np.isnan(expected))
          outputs.append(c.copy().view(np.uint16))
        # Special payloads are compared on this actual machine, never to an
        # assumed x86/ARM canonical NaN payload.
        np.testing.assert_array_equal(
          outputs[0], outputs[1], err_msg=f"off/auto parity: {dtype=} {inputs=} {width=} {layout=} {special=}"
        )


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
@pytest.mark.parametrize("operation", ("moments", "centered"))
@pytest.mark.parametrize("affine", (False, True))
def test_native_library_auto_uniform_stats_same_machine_strict_bits(native_auto_pairs, dtype, operation, affine):
  memory_dtype = np.float16 if dtype == "float16" else pytest.importorskip("ml_dtypes").bfloat16
  random = np.random.default_rng(387)
  for width in (0, 1, 31, 32, 33, 127, 128, 129, 4096):
    for special in (False, True):
      data = random.normal(size=(3, width)).astype(memory_dtype)
      gamma, beta = (random.normal(size=width).astype(memory_dtype) for _ in range(2))
      if special and width:
        data[0, 0] = float("nan")
        data[1, 0] = float("inf")
        data[2, 0] = -0.0
      for layout in ("dense", "negative-row", "inplace", "shifted"):
        outputs = []
        for policy in ("off", "auto"):
          backing = np.zeros((3, width + 1), dtype=memory_dtype)
          a = backing[:, :width]
          a[:] = data
          c = np.empty_like(data)
          if layout == "negative-row":
            a, c = a[::-1], c[::-1]
          elif layout == "inplace":
            c = a
          elif layout == "shifted":
            c = backing[:, 1:]
          native_auto_pairs[dtype, operation, affine, policy](a, gamma, beta, c)
          outputs.append(c.copy().view(np.uint16))
        np.testing.assert_array_equal(
          outputs[0],
          outputs[1],
          err_msg=f"library off/auto: {dtype=} {operation=} {affine=} {width=} {layout=} {special=}",
        )
