# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bit-preserving streaming fringes and executable DSL packing schedules."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import arm_cross_cxx_include_flags, assert_no_vec_mask_arrays, make_compiler
from test_dsl_sme_family_owners import _assert_packing_owner, _owners
from vecops._arm_vector_state import get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES
from vecops.language._compiler.dtypes import ELEMENT_BYTES


def test_integer_metadata_memo_retains_live_identities_and_leaves_floating_dags_alone():
  from vecops.language._compiler import ir as I
  from vecops.language._compiler.analysis.affine import canonical_integer

  symbol = I.Expr(I.ScalarType("int64"), "raw", data="metadata_pitch")
  extent = I.Expr(I.ScalarType("int64"), "streaming_lanes", data="uint32")
  unchanged = I.Expr(I.ScalarType("int64"), "floordiv", (symbol, extent))
  assert canonical_integer(unchanged) is unchanged
  expression = I.binary("add", I.binary("mul", symbol, I.literal(3)), symbol)
  memo = {}
  result = canonical_integer(expression, _memo=memo)
  assert canonical_integer(expression, _memo=memo) is result
  assert memo[id(expression)][0] is expression
  assert canonical_integer(result, _memo=memo) is result
  floating = I.Expr(I.ScalarType("float32"), "add", (I.literal(1.0), I.literal(2.0)))
  assert canonical_integer(floating) is floating


def packing_template(dtype):
  spatial, reduction, capacity = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((spatial, reduction), dtype=dtype),
    O: T.Tensor((capacity,), dtype=dtype, access="output"),
  ):
    T.pack(A, O, side="A", format="sme")

  return kernel


@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32", "float64"])
def test_sme_packing_fringe_is_library_control_with_small_bit_primitives(dtype):
  source = CPUBackend().lower(packing_template(dtype).ir, (), target="SME").source
  assert "dsl_pack_" in source
  assert "pack_offset" in source
  assert "packed output buffer too small" in source
  assert "vecops::vec::sme::" in source.partition("void __kernel__")[2]
  assert len(source) < 250_000, "instruction schedule source grew beyond its bounded variants"
  assert max(map(len, source.splitlines())) < 4_000, "metadata was printed as expanded expression trees"
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("dtype", ["uint8", "float16"])
def test_sme_fringe_uses_storage_width_without_runtime_alignment_dispatch(dtype):
  from vecops.language._compiler import ir as I
  from vecops.language._compiler.frontend.helpers import expand_helper
  from vecops.language._compiler.ir.traversal import nodes
  from vecops.language._compiler.operations import snapshot
  from vecops.language._library.sme_packing import pack_fringe

  def raw(name):
    return I.Expr(I.ScalarType("int64"), "raw", data=name)

  source = I.View("source", dtype, (raw("M"), raw("K")), (raw("pitch"), I.literal(1)), I.literal(0))
  destination = I.View("output", dtype, (raw("groups"), raw("elements")), (raw("elements"), I.literal(1)), I.literal(0))
  _, statements = expand_helper(
    pack_fringe,
    (source, destination),
    keyword_arguments={"origin": 0, "k_origin": 0, "width": raw("width"), "active_k": raw("K"), "spatial": raw("M")},
    catalog=snapshot(),
    prefix="fringe",
    hardware_kind="sme",
  )
  graph = tuple(nodes(I.Block(statements)))
  assert not any(isinstance(node, I.If) for node in graph)
  loads = [node for node in graph if isinstance(node, I.ArchitectureCall) and node.operation == "sme.load_hor"]
  assert len(loads) == 2
  assert all(node.operands == ("uint8" if dtype == "uint8" else "uint16",) for node in loads)


@pytest.mark.parametrize("operation", ["load", "store"])
def test_sme_bit_memory_rejects_numeric_word_dtype(operation):
  @T.template
  def kernel(A: T.Tensor((4,), dtype="float32"), O: T.Tensor((4,), dtype="float32", access="output")):
    with T.sme.region():
      if operation == "load":
        T.sme.load_bits(A, dtype="float32")
      else:
        value = T.sme.load(A)
        T.sme.store_bits(value, O)

  with pytest.raises(TypeError, match="unsigned"):
    CPUBackend().lower(kernel.ir, (), target="SME")


@pytest.mark.parametrize("dtype", ["uint8", "int8", "float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("svl", [16, 64, 128])
def test_sme_raw_packing_recipe_matches_independent_bits_and_address_model(dtype, svl):
  from dsl_sme_packing_model import PackingModel
  from vecops.language._compiler import ir as I
  from vecops.language._compiler.analysis.affine import bind_instruction_metadata, simplify_loop_metadata
  from vecops.language._compiler.frontend.helpers import expand_helper
  from vecops.language._compiler.operations import snapshot
  from vecops.language._library.sme_packing import pack_raw

  def raw(name):
    return I.Expr(I.ScalarType("int64"), "raw", data="sme_packing." + name)

  source = I.View("sme_packing_input", dtype, (raw("m"), raw("k")), (raw("pitch"), I.literal(1)), I.literal(0))
  size = ELEMENT_BYTES[dtype]
  word_size = 8 if size == 8 else 4
  quantum = word_size // size

  def output_view(spatial, reduction):
    width = I.Expr(I.ScalarType("int64"), "streaming_lanes", data=f"uint{word_size * 8}")
    panel = I.binary("mul", I.literal(2), width)
    rows = I.binary(
      "add", I.binary("floordiv", spatial, panel), I.binary("ne", I.binary("mod", spatial, panel), I.literal(0))
    )
    groups = I.binary(
      "add",
      I.binary("floordiv", reduction, I.literal(quantum)),
      I.binary("ne", I.binary("mod", reduction, I.literal(quantum)), I.literal(0)),
    )
    elements = I.binary("mul", panel, I.literal(quantum))
    return I.View(
      "sme_packing_output",
      dtype,
      (rows, groups, elements),
      (I.binary("mul", groups, elements), elements, I.literal(1)),
      I.literal(0),
    )

  target = output_view(raw("m"), raw("k"))
  _, body = expand_helper(pack_raw, (source, target), catalog=snapshot(), prefix="packing_model", hardware_kind="sme")
  original = I.Block(body)
  prepared = bind_instruction_metadata(simplify_loop_metadata(original))
  specialized = bind_instruction_metadata(simplify_loop_metadata(original, streaming_bytes=svl))
  panel, quantum, chunk = 2 * (svl // word_size), word_size // size, svl // size
  bits_dtype = np.dtype(f"uint{size * 8}")
  for spatial, reduction in ((2**63 - 1, 0), (0, 2**63 - 1)):
    for program in (original, prepared, specialized):
      output = np.full(5, 19, dtype=bits_dtype)
      model = PackingModel(
        program,
        dtype=dtype,
        svl=svl,
        source=np.empty(0, bits_dtype),
        destination=output,
        spatial=spatial,
        reduction=reduction,
        pitch=0,
        capacity=0,
      )
      model.run()
      assert not model.trace, "empty domains must not traverse or access a padded panel"
      np.testing.assert_array_equal(output, np.full(5, 19, bits_dtype))
  for spatial, reduction in ((0, 7), (7, 0), (1, 1), (panel - 1, quantum + 1), (panel + 5, 6 * chunk + 3)):
    pitch = reduction + 3
    capacity = ((spatial + panel - 1) // panel) * ((reduction + quantum - 1) // quantum) * panel * quantum
    storage = np.full(spatial * pitch, 7, dtype=bits_dtype)
    logical = (np.arange(spatial * reduction, dtype=bits_dtype) * 13).reshape(spatial, reduction)
    if spatial:
      storage.reshape(spatial, pitch)[:, :reduction] = logical
    padded = np.zeros(
      ((spatial + panel - 1) // panel * panel, (reduction + quantum - 1) // quantum * quantum), dtype=bits_dtype
    )
    padded[:spatial, :reduction] = logical
    expected = (
      padded.reshape(-1, panel, padded.shape[1] // quantum, quantum).transpose(0, 2, 1, 3).reshape(-1)
      if capacity
      else np.empty(0, dtype=bits_dtype)
    )
    # Bind the same source recipe as the constant-shape compiler adapter.
    # A nonconstant signed pitch remains a runtime value in all versions.
    from dataclasses import replace

    _, constant_body = expand_helper(
      pack_raw,
      (
        replace(source, shape=(I.literal(spatial), I.literal(reduction))),
        output_view(I.literal(spatial), I.literal(reduction)),
      ),
      catalog=snapshot(),
      prefix="packing_constant_model",
      hardware_kind="sme",
      keyword_arguments={"row_unroll": min(16, svl // 4), "group_unroll": min(16, svl // word_size)},
    )
    constant_program = bind_instruction_metadata(simplify_loop_metadata(I.Block(constant_body), streaming_bytes=svl))
    traces = []
    for program in (original, prepared, specialized, constant_program):
      output = np.full(capacity + 5, 19, dtype=bits_dtype)
      model = PackingModel(
        program,
        dtype=dtype,
        svl=svl,
        source=storage,
        destination=output,
        spatial=spatial,
        reduction=reduction,
        pitch=pitch,
        capacity=capacity,
      )
      model.run()
      np.testing.assert_array_equal(output[:capacity], expected)
      np.testing.assert_array_equal(output[capacity:], np.full(5, 19, dtype=bits_dtype))
      traces.append(model.trace)
    assert all(trace == traces[0] for trace in traces), "metadata optimization changed the ordered architecture program"


@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32", "float64"])
def test_native_sme_bit_memory_partial_words_and_zero_padding(dtype, tmp_path):
  state = get_vector_state()
  if state.sme_bytes is None:
    pytest.skip("actual SME target required")
  size = ELEMENT_BYTES[dtype]
  word = "uint64" if size == 8 else "uint32"
  quantum, capacity = ELEMENT_BYTES[word] // size, state.sme_bytes // size
  count, output_count = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((count,), dtype=dtype), O: T.Tensor((output_count,), dtype=dtype, access="output")):
    with T.sme.region():
      value = T.sme.load_bits(A, dtype=word)
      T.sme.store_bits(value, O)

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  bits_dtype = np.dtype(f"uint{size * 8}")
  for length in dict.fromkeys((0, 1, quantum - 1, quantum + 1, capacity - 1, capacity)):
    padded = (length + quantum - 1) // quantum * quantum
    # Float payloads are tested as bits, including NaNs and signed zeros;
    # no conversion/NaN canonicalization is permitted in this memory route.
    bits = np.arange(length, dtype=bits_dtype) * 13 + np.iinfo(bits_dtype).max // 2
    data = bits.view(dtype)
    output = np.empty(padded, dtype=dtype)
    compiled.operator.native(data, output, **compiled._core.specialization_values)
    expected = np.zeros(padded, dtype=bits_dtype)
    expected[:length] = bits
    np.testing.assert_array_equal(output.view(bits_dtype), expected)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_native_sme_storage_view_aliases_packed_word_rows(dtype, tmp_path):
  """Exercise the physical ZA row mapping independently of a pack driver."""
  from vecops.language._library.sme_packing import pack_fringe

  state = get_vector_state()
  if state.sme_bytes is None:
    pytest.skip("actual SME target required")
  width = state.sme_bytes // 4
  ratio = 4 // ELEMENT_BYTES[dtype]

  @T.template
  def kernel(
    A: T.Tensor((2 * width, 1), dtype=dtype), O: T.Tensor((1, 2 * width * ratio), dtype=dtype, access="output")
  ):
    with T.sme.region():
      pack_fringe(A, O, origin=0, k_origin=0, width=T.sme.streaming_lanes("uint32"), active_k=1, spatial=A.shape[0])

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  assert "vecops::vec::sme::write_hor" not in compiled.generated_source
  source = (np.arange(2 * width, dtype=dtype) * 13 + 7).reshape(-1, 1)
  output = np.full((1, 2 * width * ratio), 19, dtype)
  compiled.operator.native(source, output, **compiled._core.specialization_values)
  expected = np.zeros((2 * width, ratio), dtype)
  expected[:, 0] = source[:, 0]
  np.testing.assert_array_equal(output.reshape(2 * width, ratio), expected)


@pytest.mark.parametrize("dtype", ["uint8", "int8", "float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("side", ["A", "B"])
@pytest.mark.parametrize("constant_shape", [False, True])
def test_native_sme_library_pack_bits_pitches_tails_and_empty(dtype, side, constant_shape, tmp_path):
  """Exercise the whole staged driver, not only isolated bit-move primitives."""
  state = get_vector_state()
  if state.sme_bytes is None:
    pytest.skip("actual SME target required")
  storage_dtype = np.dtype(pytest.importorskip("ml_dtypes").bfloat16) if dtype == "bfloat16" else np.dtype(dtype)
  size = storage_dtype.itemsize
  word_size = 8 if size == 8 else 4
  width, quantum = state.sme_bytes // word_size, word_size // size
  panel, chunk = 2 * width, state.sme_bytes // size
  capacity = T.dynamic(lower=0)
  spatial, reduction = (panel + 5, 6 * chunk + 3) if constant_shape else (T.dynamic(lower=0), T.dynamic(lower=0))
  pitch = T.dynamic()  # Raw packing also supports reversed spatial rows.
  shape, strides = ((spatial, reduction), (pitch, 1)) if side == "A" else ((reduction, spatial), (1, pitch))

  @T.template
  def kernel(A: T.Tensor(shape, strides, dtype), O: T.Tensor((capacity,), dtype=dtype, access="output")):
    T.pack(A, O, side=side, format="sme")

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  assert "packed output buffer too small" in compiled.generated_source
  assert "vecops::vec::sme::" in compiled.generated_source
  bits_dtype = np.dtype(f"uint{size * 8}")
  cases = (
    ((spatial, reduction),)
    if constant_shape
    else (
      (0, 7),
      (7, 0),
      (1, 1),
      (1, chunk + 3),
      (panel - 1, quantum + 1),
      (panel, chunk),
      (panel + 5, 6 * chunk + 3),
    )
  )
  for s, k in cases:
    row_pitch = k + 3
    rows = np.full((max(s, 1), row_pitch), 7, dtype=bits_dtype)
    logical = (np.arange(s * k, dtype=bits_dtype) * 13 + np.iinfo(bits_dtype).max // 2).reshape(s, k)
    rows[:s, :k] = logical
    source = np.ndarray((s, k), storage_dtype, buffer=rows, strides=(row_pitch * size, size))
    padded_s = (s + panel - 1) // panel * panel
    groups = (k + quantum - 1) // quantum
    for reverse in (False, True):
      padded = np.zeros((padded_s, groups * quantum), bits_dtype)
      padded[:s, :k] = logical[::-1] if reverse else logical
      expected = padded.reshape(padded_s // panel, panel, groups, quantum).transpose(0, 2, 1, 3).reshape(-1)
      for alias in (False, True):
        # Overlap snapshots must preserve all storage bits and the logical
        # traversal, even when the original spatial stride is negative.
        output_bits = np.full(max(rows.size if alias else 0, expected.size) + 5, 19, dtype=bits_dtype)
        if alias:
          original = np.ndarray((s, k), storage_dtype, buffer=output_bits, strides=(row_pitch * size, size))
          original.view(bits_dtype)[:] = logical
        else:
          original = source
        actual = original[::-1] if reverse else original
        output = output_bits[: expected.size].view(storage_dtype)
        untouched = output_bits[expected.size :].copy()
        compiled.operator.native(actual if side == "A" else actual.T, output, **compiled._core.specialization_values)
        np.testing.assert_array_equal(output_bits[: expected.size], expected)
        np.testing.assert_array_equal(output_bits[expected.size :], untouched)


@pytest.mark.parametrize("dtype", ["uint8", "float16", "float64"])
@pytest.mark.parametrize("svl", [16, 64, 128])
def test_sme_packing_arm_cross_codegen(dtype, svl, tmp_path):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for real FA64 codegen")
  root = Path(__file__).resolve().parents[2]
  source = CPUBackend().lower(packing_template(dtype).ir, (), target="SME").source
  element = CPP_TYPES[dtype]
  wrapper = (
    '#include "vecops/Kernel.h"\n'
    "namespace vecops::spec { inline constexpr auto Parallelism=vecops::meta::cint<1>; }\n"
    + source
    + '\nextern "C" void pack_library(void* a,void* o,vecops::nint_t m,vecops::nint_t k,vecops::nint_t p) {\n'
    + f"auto A=vecops::tensor::make_tensor(static_cast<{element}*>(a),"
    + "vecops::tensor::make_shape(vecops::meta::Any{m},vecops::meta::Any{k}));\n"
    + f"auto O=vecops::tensor::make_tensor(static_cast<{element}*>(o),"
    + "vecops::tensor::make_shape(vecops::meta::Any{p}));\n__kernel__(A,O);\n}\n"
  )
  cpp, assembly = tmp_path / "packing.cpp", tmp_path / "packing.s"
  cpp.write_text(wrapper)
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    "-std=c++20",
    "-O3",
    "-S",
    "-DNDEBUG",
    "-DVECOPS_TARGET_SME_FA64=1",
    "-DVECOPS_TARGET_SME_F64F64=1",
    f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={svl * 8}",
    "-DVECOPS_KERNEL_SVE_BYTES=16",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
    str(cpp),
    "-o",
    str(assembly),
  ]
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".log").write_text(result.stdout + result.stderr)
  assert result.returncode == 0, result.stderr[-6000:]
  code = assembly.read_text()
  owners = _owners(code, vl=16, svl=svl)
  assert owners
  for index, body in enumerate(owners):
    _assert_packing_owner(body, index)
    # Fringes load through the storage-width horizontal view and extract
    # packed words vertically. They need no second ping-pong pair ZA2/3.
    storage = {"uint8": ("b", "b"), "float16": ("h", "h"), "float64": ("d", "d")}[dtype]
    assert re.search(rf"^\s*ld1{storage[0]}\s+.*\bza\d+h\.{storage[1]}", body, re.MULTILINE)
    word = "d" if ELEMENT_BYTES[dtype] == 8 else "s"
    assert re.search(rf"\bza\d+v\.{word}\b", body)
  assert not re.search(r"^\s*\w*mopa\b", code, re.MULTILINE)
  assert_no_vec_mask_arrays(source)
