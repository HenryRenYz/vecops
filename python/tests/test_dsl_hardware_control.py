# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Hardware loops retain one owner and preflight all metadata before entry."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import amx_tile_available_for_test, make_compiler
from vecops._arm_vector_state import get_vector_state
from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.affine import (
  bind_instruction_metadata,
  canonical_integer,
  interval,
  positive_extent,
  simplify_loop_metadata,
)
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.frontend.parser import lower
from vecops.language._debug.interpreter import Interpreter


@T.func
def copy_amx_tiles(source, destination):
  for tile in T.serial(source.shape[0], unroll=2):
    value = T.amx.load(0, source[tile, :, :])
    T.amx.store(value, destination[tile, :, :])


def amx_template():
  count = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((count, 2, 2), dtype="int32"),
    C: T.Tensor((count, 2, 2), dtype="int32", access="inout"),
    D: T.Tensor((count, 2, 2), dtype="int32", access="output"),
    enabled: bool = True,
  ):
    original = T.load(C)
    with T.amx.region(((2, 8),)):
      if enabled:
        copy_amx_tiles(A, C)
    T.store(D, original)

  return kernel


def test_hardware_loop_preflight_source():
  artifact = CPUBackend().lower(amx_template().ir, (), target="AVX512")
  source = artifact.source
  assert source.count(".with_resources(") == 1
  assert source.count(".with_configuration(") == 1
  owner = source.split(".with_configuration(", 1)[1].split("#else", 1)[0]
  assert "throw" not in owner and "check_writable" not in owner
  assert "for (vecops::nint_t" in owner
  assert "#pragma clang loop unroll_count(2)" in owner
  assert "vecops::vec::amx::load<0>" in owner
  assert "vecops::vec::amx::store<0>" in owner
  assert source.index("hardware view index is outside") < source.index(".with_resources(")
  assert "VECOPS_INLINE_LAMBDA_NOEXCEPT" in owner


def test_owned_hardware_view_and_immutable_metadata():
  @T.template
  def kernel(A: T.Tensor((7, 2, 2), dtype="int32"), C: T.Tensor((7, 2, 2), dtype="int32", access="output")):
    scratch = T.alloc((7, 2, 2), dtype="int32")
    T.copy(A, scratch)
    with T.amx.region(((2, 8),)):
      stop = scratch.shape[0]
      for i in T.serial(0, stop, 3, unroll=1):
        T.amx.load(0, scratch[i, :, :])
        T.amx.store(0, C[i, :, :])

  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  assert ".with_configuration(" in source
  assert "#pragma GCC unroll 1" in source
  assert "hardware view index is outside" in source


def test_hardware_square_transpose_loops_are_not_fused():
  @T.template
  def kernel(A: T.Tensor((4, 4), dtype="float32", access="inout")):
    with T.sme.region():
      T.sme.transpose_tile(A, A)

  artifact = CPUBackend().lower(kernel.ir, (), target="SME")
  regions = [s for s in artifact.lowered.kernel.body.statements if isinstance(s, I.HardwareRegion)]
  assert len(regions) == 1
  assert len([s for s in regions[0].body.statements if isinstance(s, I.For)]) == 2


@pytest.mark.parametrize("dtype", ("int8", "float16", "bfloat16", "float32", "float64"))
def test_sme_transpose_tile_is_source_parsed(dtype):
  @T.template
  def kernel(A: T.Tensor((3, 5), dtype=dtype), C: T.Tensor((5, 3), dtype=dtype, access="output")):
    with T.sme.region():
      T.sme.transpose_tile(A, C)

  ir = lower(kernel.ir)
  region = ir.body.statements[0]
  assert isinstance(region, I.HardwareRegion)
  assert [s.body.statements[0].operation for s in region.body.statements if isinstance(s, I.For)] == [
    "sme.load_hor",
    "sme.store_ver",
  ]
  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert source.count(".with_resources(") == 1
  owner = source.split("VECOPS_INLINE_LAMBDA_NOEXCEPT", 1)[1].split("#else", 1)[0]
  assert "throw" not in owner and "std::vector" not in owner
  assert "vecops::vec::sme::load_hor<0>" in source
  assert "vecops::vec::sme::store_ver<0>" in source
  assert "std::array<vecops::vec::" not in source


@pytest.mark.parametrize("extent", [1, 7, 16, 17, 33])
def test_correlated_nested_interval_proves_clipped_root(extent):
  outer = I.Expr(I.ScalarType("int64"), "local", data="outer")
  inner = I.Expr(I.ScalarType("int64"), "local", data="inner")
  total = I.literal(extent)
  bound = I.binary("min", I.literal(16), I.binary("sub", total, outer))
  lo, hi = interval(
    I.binary("add", outer, inner),
    {"outer": (I.literal(0), I.literal(extent - 1)), "inner": (I.literal(0), I.binary("sub", bound, I.literal(1)))},
  )
  assert I.constant(lo) == 0
  assert I.constant(hi) == extent - 1


def test_hardware_affine_interval_preserves_cancellation():
  i = I.Expr(I.ScalarType("int64"), "local", data="i")
  width = I.literal(16)
  value = I.binary("sub", I.binary("add", i, width), i)
  lo, hi = interval(value, {"i": (I.literal(0), I.literal(1023))})
  assert I.constant(lo) == I.constant(hi) == 16


def test_integer_metadata_simplification_is_domain_scoped_and_keeps_ssa():
  extent = I.Expr(I.ScalarType("int64"), "raw", data="M")
  index = I.Expr(I.ScalarType("int64"), "local", data="i")
  clipped = I.binary("max", I.literal(0), I.binary("min", index, extent))
  width = I.Expr(I.ScalarType("int64"), "local", data="width")
  operation = I.Assign("address", I.binary("add", clipped, width))
  first = I.For("i", I.literal(0), extent, I.literal(1), I.Block([operation]))
  second = I.For("i", I.literal(-2), extent, I.literal(1), I.Block([operation]))
  rewritten = simplify_loop_metadata(I.Block([first, second]))
  a, b = (loop.body.statements[0].value for loop in rewritten.statements)
  assert I.same_expr(a, I.binary("add", index, width))
  assert not I.same_expr(a, b)
  assert any(v.op == "local" and v.data == "width" for v in I.traversal.walk(a))
  assert first.body.statements[0] is operation


def test_metadata_clamps_do_not_expand_into_cartesian_arms():
  from vecops.language._compiler.ir.traversal import nodes

  i = I.Expr(I.ScalarType("int64"), "local", data="i")
  value = I.literal(0)
  for offset in range(12):
    value = I.binary("add", value, I.binary("min", I.binary("add", i, I.literal(offset)), I.literal(20)))
  normalized = canonical_integer(value)
  assert len(tuple(nodes(normalized))) < 150
  reference = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
  for index in (-30, -1, 0, 12, 25):
    reference.environment["i"] = index
    assert reference.value(normalized) == sum(min(index + offset, 20) for offset in range(12))


def test_native_extent_products_are_positive_but_unknown_parameters_are_not():
  from vecops.language._compiler.operations import snapshot

  width = I.Expr(I.ScalarType("int64"), "streaming_lanes", data="uint32")
  assert positive_extent(I.binary("mul", I.literal(2), width), snapshot())
  assert not positive_extent(I.binary("mul", I.literal(-1), width), snapshot())
  assert not positive_extent(I.Expr(I.ScalarType("int64"), "raw", data="unknown"), snapshot())


def test_instruction_metadata_ssa_stays_in_its_loop_and_branch_scope():
  i = I.Expr(I.ScalarType("int64"), "local", data="i")
  repeated = I.binary("mul", I.binary("add", i, I.literal(1)), I.literal(4))
  view = I.View("A", "uint8", (repeated,), (I.literal(1),), repeated)
  move = I.ArchitectureCall("sme.load_hor", view=view, operands=("uint32",), slice_index=i)
  condition = I.Expr(I.ScalarType("bool"), "raw", data="enabled")
  branch = I.If(condition, I.Block([move, move]), I.Block([move]))
  loop = I.For("i", I.literal(0), I.literal(3), I.literal(1), I.Block([branch]))
  result = bind_instruction_metadata(I.Block([loop]))
  assert len(result.statements) == 1 and isinstance(result.statements[0], I.For)
  changed = result.statements[0].body.statements[0]
  assert isinstance(changed, I.If)
  first, second = changed.body.statements[-2:]
  other = changed.otherwise.statements[-1]
  assert first.view.offset.op == "local"
  assert first.view.offset is first.view.shape[0] is second.view.offset
  assert other.view.offset.data != first.view.offset.data
  assert all(isinstance(s, I.Assign) for s in changed.body.statements[:-2])
  assert move.view.offset is repeated


def test_sme_full_word_packing_is_source_parsed_and_operation_ordered():
  from vecops.language._compiler.frontend.helpers import expand_helper
  from vecops.language._compiler.ir.traversal import nodes
  from vecops.language._compiler.operations import snapshot
  from vecops.language._library.sme_packing import pack_full_words

  def raw(name):
    return I.Expr(I.ScalarType("int64"), "raw", data=name)

  source = I.View("source", "float16", (raw("M"), raw("K")), (raw("pitch"), I.literal(1)), I.literal(0))
  elements = I.binary("mul", raw("width"), I.literal(4))
  destination = I.View("output", "float16", (raw("groups"), elements), (elements, I.literal(1)), I.literal(0))
  result, body = expand_helper(
    pack_full_words,
    (source, destination),
    catalog=snapshot(),
    prefix="packing_library",
    keyword_arguments={name: raw(name) for name in ("origin", "width", "chunk", "chunks")},
    hardware_kind="sme",
  )
  assert result is None
  graph = tuple(nodes(I.Block(body)))
  assert any(isinstance(node, I.Prefetch) for node in graph)
  moves = [node for node in graph if isinstance(node, I.ArchitectureCall)]
  assert {node.tile for node in moves} == {0, 1, 2, 3}
  assert all(node.operands == ("uint32",) for node in moves)
  assert not any(isinstance(node, I.Control) for node in graph)
  transfer = next(
    node
    for node in graph
    if isinstance(node, I.For)
    and any(
      isinstance(child, I.ArchitectureCall) and child.operation == "sme.store_ver" for child in node.body.statements
    )
  )
  instructions = [node.operation for node in transfer.body.statements if isinstance(node, I.ArchitectureCall)]
  assert instructions == ["sme.load_hor", "sme.load_hor", "sme.store_ver", "sme.store_ver"]


def test_sme_packing_stages_executable_library_before_emission(monkeypatch):
  from vecops.language._compiler.backends.cpu.cpp.emitter import generate
  from vecops.language._compiler.frontend import helpers

  capacity = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((37, 129), dtype="float16"), O: T.Tensor((capacity,), dtype="float16", access="output")):
    T.pack(A, O, side="A", format="sme")

  artifact = CPUBackend().lower(kernel.ir, (), target="SME", options={"cpu_sme_packing_dsl": True})
  assert isinstance(artifact.backend_ir.sme_packing_programs["float16", (37, 129)], I.Block)
  assert "float16" not in artifact.backend_ir.sme_packing_programs, "do not stage an unused generic recipe"

  def forbidden(*args, **kwargs):
    raise AssertionError("packing algorithms must be staged before terminal printing")

  monkeypatch.setattr(helpers, "expand_helper", forbidden)
  source, _ = generate(artifact.backend_ir)
  assert "sme_packing_float16_" in source
  assert "vecops::vec::sme::load_hor<2>" in source
  assert "vecops::vec::sme::store_ver<3>" in source
  assert "std::array<vecops::vec::" not in source


@pytest.mark.parametrize("pitch", [None, 256, 259, -259])
def test_sme_packing_unroll_keeps_runtime_address_products_bounded(pitch):
  from vecops._arm_vector_state import VectorState
  from vecops.language._compiler.ir.traversal import nodes

  stride = T.dynamic() if pitch is None else pitch
  capacity = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((64, 256), (stride, 1), "bfloat16"), O: T.Tensor((capacity,), dtype="bfloat16", access="output")
  ):
    T.pack(A, O, side="A", format="sme")

  artifact = CPUBackend().lower(
    kernel.ir, (), target="SME", options={"cpu_sme_packing_dsl": True, "_vector_state": VectorState(64, 64)}
  )
  program = artifact.backend_ir.sme_packing_programs["bfloat16", (64, 256)]
  transfers = [
    node
    for node in nodes(program)
    if isinstance(node, I.For)
    and sum(isinstance(s, I.ArchitectureCall) and s.operation == "sme.load_hor" for s in node.body.statements) == 2
  ]
  assert transfers and all(node.unroll == (16 if pitch == 256 else 2) for node in transfers)


@pytest.mark.parametrize("pitch", [None, 129, -133])
def test_sme_owner_restores_constants_but_not_alias_dependent_strides(pitch):
  import re

  stride = T.dynamic() if pitch is None else pitch
  capacity = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((37, 129), (stride, 1), "float16"),
    O: T.Tensor((capacity,), dtype="float16", access="output"),
  ):
    T.pack(A, O, side="A", format="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME", options={"cpu_sme_packing_dsl": True}).source
  match = re.search(
    r"const vecops::nint_t \w+_sme_pack_stride_value_\d+=([^,]+), "
    r"\w+_sme_pack_m_value_\d+=std::int64_t\{37\}, \w+_sme_pack_k_value_\d+=std::int64_t\{129\};",
    source,
  )
  assert match, "known dimensions were reloaded through reference captures"
  if pitch == 129:
    assert match[1] == "std::int64_t{129}"
  else:
    # A reversed constant input gets a positive dense snapshot when aliasing;
    # a dynamic pitch is unknown. Neither may borrow the original constant.
    assert "std::int64_t{" not in match[1]


def test_sme_packing_uses_all_external_packed_dimensions():
  @T.template
  def kernel(A: T.Tensor((37, 129), dtype="uint8"), O: T.Tensor((2, 33, 32, 4), dtype="uint8", access="output")):
    T.pack(A, O, side="A", format="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME", options={"cpu_sme_packing_dsl": True}).source
  actual = "(std::int64_t{2}) * (std::int64_t{33}) * (std::int64_t{32}) * (std::int64_t{4})"
  assert f"if (({actual}) < (" in source and "packed output buffer too small" in source
  assert "sme_pack_capacity_value" not in source, "packed logical axes replace per-store capacity clamps"


def test_two_sme_library_programs_keep_separate_lexical_names():
  @T.template
  def kernel(
    A: T.Tensor((16, 16), dtype="float32"),
    B: T.Tensor((16, 16), dtype="float32"),
    C: T.Tensor((16, 16), dtype="float32", access="output"),
    D: T.Tensor((16, 16), dtype="float32", access="output"),
  ):
    with T.sme.region():
      T.copy(T.transpose(A), C, implementation="sme")
      T.copy(T.transpose(B), D, implementation="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME", options={"cpu_transpose_sme_dsl": True}).source
  assert source.count(".with_resources(") == 1
  assert source.count("/* library instruction program */ {") == 2


def test_hardware_prefetch_is_checked_before_ownership():
  extent = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((extent,), dtype="float32")):
    with T.sme.region():
      for i in T.serial(extent, unroll=1):
        T.prefetch(A[i])

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert source.index("hardware view index is outside") < source.index(".with_resources(")
  owner = source.split("VECOPS_INLINE_LAMBDA_NOEXCEPT", 1)[1].split("#else", 1)[0]
  assert "vecops::vec::prefetch" in owner
  assert "throw" not in owner


def test_hardware_prefetch_retains_scalar_index_bounds():
  @T.template
  def kernel(A: T.Tensor((8,), dtype="float32")):
    with T.sme.region():
      for i in T.serial(8, unroll=1):
        T.prefetch(A[i + 1])

  artifact = CPUBackend().lower(kernel.ir, (), target="SME")
  plans = tuple(artifact.backend_ir.hardware.values())
  assert len(plans) == 1
  reference = Interpreter(artifact.lowered.kernel, {"A": np.empty(8, dtype=np.float32)})
  assert any(not reference.value(condition) for condition, _ in plans[0].checks)


@pytest.mark.parametrize("dtype", ["uint8", "int8", "float16", "float32"])
@pytest.mark.parametrize("side", ["A", "B"])
def test_native_sme_dsl_full_word_packing(dtype, side, tmp_path):
  from vecops.language._compiler.dtypes import ELEMENT_BYTES

  state = get_vector_state()
  if state.sme_bytes is None:
    pytest.skip("actual SME target required")
  panel = 2 * (state.sme_bytes // 4)
  quantum = max(1, 4 // ELEMENT_BYTES[dtype])
  spatial, k = panel + 5, 3 * (state.sme_bytes // ELEMENT_BYTES[dtype]) + 1
  capacity = ((spatial + panel - 1) // panel) * ((k + quantum - 1) // quantum) * panel * quantum
  shape, strides = ((spatial, k), (k, 1)) if side == "A" else ((k, spatial), (1, k))

  @T.template
  def kernel(A: T.Tensor(shape, strides, dtype), O: T.Tensor((capacity,), dtype=dtype, access="output")):
    T.pack(A, O, side=side, format="sme")

  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_sme_packing_dsl": True})
  data = ((np.arange(spatial * k) % 13) / (4 if dtype.startswith("float") else 1)).astype(dtype).reshape(spatial, k)
  output = np.empty(capacity, dtype=dtype)
  compiled.operator.native(data if side == "A" else data.T, output, **compiled._core.specialization_values)
  expected = np.zeros(((spatial + panel - 1) // panel, (k + quantum - 1) // quantum, panel, quantum), dtype=dtype)
  for row in range(spatial):
    for column in range(k):
      expected[row // panel, column // quantum, row % panel, column % quantum] = data[row, column]
  np.testing.assert_array_equal(output, expected.reshape(-1))
  assert "sme_packing_" in compiled.generated_source


@pytest.mark.parametrize(
  "source_dtype,output_dtype,compute",
  [
    ("float16", "float32", "float32"),
    ("float32", "float16", "float32"),
    ("float16", "float64", "float64"),
    ("float64", "float32", "float64"),
  ],
)
def test_sme_dsl_transpose_conversion_source(source_dtype, output_dtype, compute):
  @T.template
  def kernel(A: T.Tensor((3, 5), dtype=source_dtype), C: T.Tensor((5, 3), dtype=output_dtype, access="output")):
    with T.sme.region():
      T.sme.transpose_tile(A, C, compute_dtype=compute)

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  owner = source.split("VECOPS_INLINE_LAMBDA_NOEXCEPT", 1)[1].split("#else", 1)[0]
  assert "throw" not in owner and "check_writable" not in owner
  if source_dtype != compute:
    assert "vecops::vec::load_convert" in owner and "vecops::vec::sme::write_hor" in owner
  if output_dtype != compute:
    assert "vecops::vec::sme::read_ver" in owner and "vecops::vec::store_convert" in owner
  assert "std::array<vecops::vec::" not in source


def test_sme_matrix_transpose_has_one_owner_and_correlated_tail_bounds():
  m, n = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((m, n), dtype="float32"), C: T.Tensor((n, m), dtype="float32", access="output")):
    with T.sme.region():
      T.sme.transpose_matrix(A, C)

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert source.count(".with_resources(") == 1
  assert "SME matrix transpose requires disjoint operands" in source
  assert "vecops::vec::sme::load_hor<0>" in source
  assert "vecops::vec::sme::store_ver<0>" in source
  assert "std::array<vecops::vec::" not in source


@pytest.mark.parametrize("dtype,compute", [("uint8", "uint8"), ("float16", "float32"), ("float32", "float64")])
@pytest.mark.parametrize("streaming_bytes", [16, 32, 64, 128])
def test_sme_matrix_preflight_accepts_empty_and_clipped_tiles(dtype, compute, streaming_bytes):
  from vecops.language._compiler.dtypes import ELEMENT_BYTES

  m, n = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((m, n), dtype=dtype), C: T.Tensor((n, m), dtype=compute, access="output")):
    with T.sme.region():
      T.sme.transpose_matrix(A, C, compute_dtype=compute)

  artifact = CPUBackend().lower(kernel.ir, (), target="SME")
  plan = next(iter(artifact.backend_ir.hardware.values()))

  class MetadataInterpreter(Interpreter):
    def value(self, expression):
      if isinstance(expression, I.Expr) and expression.op == "streaming_lanes":
        return streaming_bytes // ELEMENT_BYTES[expression.data]
      return super().value(expression)

  for rows, columns in ((0, 31), (31, 0), (3, 5), (31, 65), (129, 63)):
    reference = MetadataInterpreter(
      artifact.lowered.kernel,
      {"A": np.empty((rows, columns), dtype=dtype), "C": np.empty((columns, rows), dtype=compute)},
    )
    for condition, message in plan.checks:

      def describe(value):
        return f"{value.op}({', '.join(describe(a) for a in value.args)})" if value.args else str(value.data)

      assert reference.value(condition), (rows, columns, message, describe(condition))


def test_automatic_transpose_stages_dsl_before_cpp_emission(monkeypatch):
  from vecops.language._compiler.backends.cpu.cpp.emitter import generate
  from vecops.language._compiler.frontend import helpers

  m, n = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((m, n), dtype="float16"), C: T.Tensor((n, m), dtype="float32", access="output")):
    T.transpose_copy(A, C, compute_dtype="float32", implementation="sme")

  artifact = CPUBackend().lower(kernel.ir, (), target="SME", options={"cpu_transpose_sme_dsl": True})
  plans = tuple(artifact.backend_ir.transposes.values())
  assert len(plans) == 1 and isinstance(plans[0].sme_program, I.Block)

  def forbidden(*args, **kwargs):
    raise AssertionError("DSL staging leaked into terminal printing")

  monkeypatch.setattr(helpers, "expand_helper", forbidden)
  source, _ = generate(artifact.backend_ir)
  assert "sme_library_" in source
  assert "vecops::vec::sme::write_hor<0>" in source
  assert "vecops::vec::sme::store_ver<0>" in source
  assert "std::array<vecops::vec::" not in source


@pytest.mark.parametrize(
  "size,input_dtype,output_dtype,compute,pairs,static",
  [
    (1024, "uint8", "uint8", "uint8", True, False),
    (1024, "float32", "float32", "float32", True, False),
    (128, "float32", "float16", "float32", True, True),
    (16, "float16", "float64", "float64", False, False),
  ],
)
def test_automatic_sme_dsl_preserves_pair_and_static_schedules(size, input_dtype, output_dtype, compute, pairs, static):
  from vecops._arm_vector_state import VectorState
  from vecops.language._compiler.ir.traversal import nodes

  @T.template
  def kernel(
    A: T.Tensor((size, size), dtype=input_dtype), C: T.Tensor((size, size), dtype=output_dtype, access="output")
  ):
    T.transpose_copy(A, C, compute_dtype=compute, implementation="sme")

  artifact = CPUBackend().lower(
    kernel.ir,
    (),
    target="SME",
    options={
      "_vector_state": VectorState(64, 64),
      "cpu_transpose_sme_dsl": True,
      "cpu_transpose_sme_const_half_output": static,
      "cpu_transpose_sme_const_half_full_input": static,
    },
  )
  plan = next(iter(artifact.backend_ir.transposes.values()))
  assert plan.sme_pairs == pairs and plan.sme_program is not None
  loops = [node for node in nodes(plan.sme_program) if isinstance(node, I.For)]
  assert any(I.constant(loop.step) == 2 for loop in loops) == pairs
  assert any(isinstance(node, I.If) for node in nodes(plan.sme_program)) == static
  assert "std::array<vecops::vec::" not in artifact.source


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_streaming_mopa_loop_codegen(dtype):
  @T.template
  def kernel(
    A: T.Tensor((2, 7), (1, 2), dtype),
    B: T.Tensor((7, 2), dtype=dtype),
    C: T.Tensor((2, 2), dtype=dtype, access="output"),
  ):
    with T.sme.region():
      T.sme.zero_za()
      for k in T.serial(7, unroll=2):
        a = T.sme.load(A[:, k])
        b = T.sme.load(B[k, :])
        T.sme.mopa(0, a, b)
      T.sme.store_tile(0, C)

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert "vecops::vec::sme::mopa<0" in source
  assert "#pragma clang loop unroll_count(2)" in source
  assert "std::array<vecops::vec::" not in source


def test_sme_compute_dtype_uses_shared_normalizer():
  @T.template
  def kernel(A: T.Tensor((3, 5), dtype="float16"), C: T.Tensor((5, 3), dtype="float32", access="output")):
    with T.sme.region():
      T.sme.transpose_tile(A, C, compute_dtype=T.float32)

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert "ScalableTag<float, 0>" in source
  assert "vecops::vec::sme::write_hor" in source


def test_sme_rank_one_store_retains_preflight_and_write_effect():
  @T.template
  def kernel(C: T.Tensor((3, 5), dtype="float32", access="inout")):
    with T.sme.region():
      T.sme.store_hor(0, 7, C[2, :])

  artifact = CPUBackend().lower(kernel.ir, (), target="SME")
  plan = next(iter(artifact.backend_ir.hardware.values()))
  assert len(plan.writes) == 1
  source = artifact.source
  owner = source.index(".with_resources(")
  for message in (
    "hardware view index is outside its tensor",
    "SME operand must have unit column stride",
    "SME operand must fit one streaming word/tile",
    "SME ZA slice index is outside its streaming tile",
  ):
    assert source.index(message) < owner


def word_reinterpret_template():
  extent = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((extent,), dtype="uint8"), C: T.Tensor((extent,), dtype="uint8", access="output")):
    with T.sme.region():
      T.sme.load_hor(2, 0, A, word_dtype=T.uint32)
      T.sme.store_hor(2, 0, C, word_dtype=T.uint32)

  return kernel


def test_raw_za_memory_word_width_is_independent_of_storage_elements():
  source = CPUBackend().lower(word_reinterpret_template().ir, (), target="SME").source
  assert "ScalableTag<std::uint32_t, 0>" in source
  assert "vecops::vec::sme::load_hor<2>" in source
  assert "vecops::vec::sme::store_hor<2>" in source
  assert " / 4)" in source
  assert "load_convert" not in source.split("void __kernel__", 1)[1]


def test_native_raw_za_word_prefix_preserves_fractional_suffix(tmp_path):
  state = get_vector_state()
  if state.sme_bytes is None:
    pytest.skip("actual SME target required")
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(word_reinterpret_template(), compiler=compiler)
  for count in (0, 1, 3, 7, state.sme_bytes - 1, state.sme_bytes, state.sme_bytes + 3):
    a = np.arange(count, dtype=np.uint8)
    c = np.full(count, 0xA5, dtype=np.uint8)
    compiled.operator.native(a, c, **compiled._core.specialization_values)
    expected = np.full(count, 0xA5, dtype=np.uint8)
    expected[: count // 4 * 4] = a[: count // 4 * 4]
    np.testing.assert_array_equal(c, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_sme_mopa_loop(dtype, tmp_path):
  if get_vector_state().sme_bytes is None:
    pytest.skip("actual SME target required")

  @T.template
  def kernel(
    A: T.Tensor((2, 7), (1, 2), dtype),
    B: T.Tensor((7, 2), dtype=dtype),
    C: T.Tensor((2, 2), dtype=dtype, access="output"),
  ):
    with T.sme.region():
      T.sme.zero_za()
      for k in T.serial(7, unroll=2):
        a = T.sme.load(A[:, k])
        b = T.sme.load(B[k, :])
        T.sme.mopa(0, a, b)
      T.sme.store_tile(0, C)

  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  a = (np.arange(14).reshape(7, 2) / 4).astype(dtype).T.copy(order="F")
  b = (np.arange(14).reshape(7, 2) / 8).astype(dtype)
  c = np.empty((2, 2), dtype=dtype)
  compiled = T.compile(kernel, compiler=compiler)
  compiled.operator.native(a, b, c, **compiled._core.specialization_values)
  np.testing.assert_array_equal(c, a @ b)


@pytest.mark.parametrize("mode", ("ordinary_load", "allocation", "unbounded", "divide", "mutable"))
def test_unsafe_hardware_control_rejected(mode):
  @T.template
  def kernel(A: T.Tensor((4, 2, 2), dtype="int32"), C: T.Tensor((4, 2, 2), dtype="int32", access="output")):
    with T.amx.region(((2, 8),)):
      if mode == "ordinary_load":
        for i in T.serial(4):
          T.store(C[i, :, :], T.load(A[i, :, :]))
      elif mode == "allocation":
        T.alloc((2, 2), dtype="int32")
      elif mode == "unbounded":
        while True:
          T.amx.zero(0)
      elif mode == "divide":
        for i in T.serial(T.load(A[0, 0, 0])):
          T.amx.zero(0)
      else:
        i = 0
        for j in T.serial(4):
          i += 1  # noqa: SIM113 -- intentionally rejected mutable instruction-region state
          T.amx.zero(0)

  with pytest.raises(TypeError, match="hardware"):
    CPUBackend().lower(kernel.ir, (), target="AVX512")


def test_native_amx_loop_preserves_captures(tmp_path):
  if not amx_tile_available_for_test():
    pytest.skip("native AMX tile support required")
  compiler = make_compiler(
    target="Native", cc="gcc", cxx="g++", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"
  )
  compiled = T.compile(amx_template(), compiler=compiler)
  for count in (0, 1, 7):
    a = np.arange(count * 4, dtype=np.int32).reshape(count, 2, 2)
    for enabled in (False, True):
      c = np.full_like(a, 73)
      d = np.empty_like(a)
      compiled.operator.native(a, c, d, enabled, **compiled._core.specialization_values)
      np.testing.assert_array_equal(c, a if enabled else np.full_like(a, 73))
      np.testing.assert_array_equal(d, np.full_like(a, 73))


@pytest.mark.parametrize(
  "dtype,out_dtype,compute,width",
  [
    ("int8", "int8", "int8", 1),
    ("float16", "float16", "float16", 2),
    ("bfloat16", "bfloat16", "bfloat16", 2),
    ("float32", "float32", "float32", 4),
    ("float64", "float64", "float64", 8),
    ("float16", "float32", "float32", 4),
    ("float32", "float16", "float32", 4),
    ("float16", "float64", "float64", 8),
    ("float64", "float32", "float64", 8),
  ],
)
def test_native_sme_dsl_transpose_tile(dtype, out_dtype, compute, width, tmp_path):
  state = get_vector_state()
  if state.sme_bytes is None:
    pytest.skip("actual SME target required")
  lanes = state.sme_bytes // width
  m, n = T.dynamic(lower=0), T.dynamic(lower=0)
  source_pitch, target_pitch = T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((m, n), (source_pitch, 1), dtype),
    C: T.Tensor((n, m), (target_pitch, 1), out_dtype, access="output"),
  ):
    with T.sme.region():
      T.sme.transpose_tile(A, C, compute_dtype=compute)

  numpy_dtype = np.dtype(pytest.importorskip("ml_dtypes").bfloat16) if dtype == "bfloat16" else np.dtype(dtype)
  input_width = numpy_dtype.itemsize
  output_dtype = np.dtype(pytest.importorskip("ml_dtypes").bfloat16) if out_dtype == "bfloat16" else np.dtype(out_dtype)
  bits = np.dtype(f"uint{output_dtype.itemsize * 8}")
  raw = dtype == out_dtype == compute
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  rng = np.random.default_rng(531)
  for rows, columns in ((0, lanes), (lanes, 0), (1, 1), (min(3, lanes), min(5, lanes)), (lanes, lanes)):
    payload = rng.integers(0, 256, size=rows * columns * input_width, dtype=np.uint8)
    a = (
      payload.view(numpy_dtype).reshape(rows, columns)
      if raw
      else ((np.arange(rows * columns) % 17 - 8) / 4).astype(numpy_dtype).reshape(rows, columns)
    )
    for negative in (False, True):
      source = a[::-1] if negative else a
      c = np.empty((columns, rows), dtype=output_dtype)
      target = c[::-1] if negative else c
      compiled.operator.native(source, target, **compiled._core.specialization_values)
      expected = source.T.copy() if raw else source.astype(np.dtype(compute)).astype(output_dtype).T.copy()
      np.testing.assert_array_equal(target.view(bits), expected.view(bits))
    if rows == columns and raw:
      expected = a.T.copy().view(bits)
      compiled.operator.native(a, a, **compiled._core.specialization_values)
      np.testing.assert_array_equal(a.view(bits), expected)
  # A rejected capacity must fail before the owner, not corrupt ZA state.
  a = np.ones((lanes + 1, 1), dtype=numpy_dtype)
  c = np.full((1, lanes + 1), 7, dtype=output_dtype)
  with pytest.raises(Exception, match="streaming"):
    compiled.operator.native(a, c, **compiled._core.specialization_values)
  np.testing.assert_array_equal(c, np.full_like(c, 7))


@pytest.mark.parametrize(
  "dtype,output_dtype,compute",
  [
    ("uint8", "uint8", "uint8"),
    ("float16", "float32", "float32"),
    ("float32", "float16", "float32"),
    ("float64", "float32", "float32"),
  ],
)
def test_native_sme_matrix_transpose(dtype, output_dtype, compute, tmp_path):
  if get_vector_state().sme_bytes is None:
    pytest.skip("actual SME target required")
  m, n = T.dynamic(lower=0), T.dynamic(lower=0)
  source_pitch, target_pitch = T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((m, n), (source_pitch, 1), dtype), C: T.Tensor((n, m), (target_pitch, 1), output_dtype, access="output")
  ):
    with T.sme.region():
      T.sme.transpose_matrix(A, C, compute_dtype=compute)

  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  for rows, columns in ((0, 31), (31, 0), (3, 5), (31, 65), (129, 63)):
    a = np.zeros((rows, columns + 3), dtype=dtype)[:, :columns]
    a[:] = ((np.arange(rows * columns) % 17) / 4).reshape(rows, columns)
    for negative in (False, True):
      source = a[::-1] if negative else a
      output = np.empty((columns, rows + 5), dtype=output_dtype)[:, :rows]
      target = output[::-1] if negative else output
      compiled.operator.native(source, target, **compiled._core.specialization_values)
      np.testing.assert_array_equal(target, source.astype(compute).astype(output_dtype).T)
