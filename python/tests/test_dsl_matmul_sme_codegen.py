"""SME instruction routes, layout contracts and peeled K schedules."""

import re

import pytest
import vecops.language as T
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.environment import freeze


@pytest.mark.parametrize("target,cost_target", (("SME", "AVX512"), ("AVX512", "SME"), ("generic", "SME")))
def test_matmul_codegen_architecture_is_independent_of_cost_overrides(target, cost_target):
  M, N, K = (T.dynamic(lower=1) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    T.dot((A, B), C)

  options = {"cpu_cost_target": cost_target}
  environment = freeze(options, target=target)
  assert environment.codegen_target.architecture == {"SME": "arm", "AVX512": "x86", "generic": "generic"}[target]
  source = CPUBackend().lower(kernel.ir, (), target=target, options=options).source
  assert ("AMX 2x2 register-tile schedule" in source) == (target != "SME")
  assert ("SME 2x2 outer-product schedule" in source) == (target != "AVX512")


def test_frozen_codegen_target_participates_in_environment_identity():
  arm = freeze({"cpu_cost_target": "generic"}, target="SME")
  x86 = freeze({"cpu_cost_target": "generic"}, target="AVX512")
  assert arm.target == x86.target
  assert arm.fingerprint != x86.fingerprint


def test_explicit_mismatching_architecture_keeps_diagnostic():
  @T.template
  def kernel(
    A: T.Tensor((25, 32), dtype="bfloat16"),
    B: T.Tensor((32, 41), dtype="bfloat16"),
    C: T.Tensor((25, 41), dtype="float32", access="output"),
  ):
    T.dot((A, B), C, implementation="sme")

  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  assert '#error "explicit sme dot requires matching target' in source
  assert "sme::mopa" not in source


@pytest.mark.parametrize("column_major", (False, True))
def test_sme_raw_accumulator_uses_direct_za_memory(column_major):
  M, N, K = 25, 41, 33
  stride = (1, M) if column_major else (N, 1)

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    D: T.Tensor((M, N), stride, "float32"),
    C: T.Tensor((M, N), stride, "float32", access="output"),
  ):
    T.dot((A, B), C, acc=T.load(D), implementation="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  axis = "ver" if column_major else "hor"
  assert f"vecops::vec::sme::load_{axis}<" in source
  assert f"vecops::vec::sme::store_{axis}<" in source
  assert "vecops::vec::sme::read_hor<" not in source
  assert "vecops::vec::sme::read_ver<" not in source
  assert "whole K-group u32 gather" in source
  assert "full K groups: no per-element tail tests" in source
  assert "one partial K group" in source


@pytest.mark.parametrize("column_major", (False, True))
@pytest.mark.parametrize("step", (-53, 53))
@pytest.mark.parametrize("metadata_snapshot", (False, True))
def test_sme_direct_slices_hoist_base_and_signed_stride(column_major, step, metadata_snapshot):
  M, N, K = 25, 41, 33
  stride = (1, step) if column_major else (step, 1)

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    D: T.Tensor((M, N), stride, "float32"),
    C: T.Tensor((M, N), stride, "float32", access="output"),
  ):
    T.dot((A, B), C, acc=T.load(D), implementation="sme")

  source = (
    CPUBackend()
    .lower(kernel.ir, (), target="SME", options={"cpu_parameter_metadata_snapshot": metadata_snapshot})
    .source
  )
  assert "sme_slice_base_" in source
  assert re.search(rf"sme_slice_stride_\d+=std::int64_t\{{{step}\}};", source)
  axis = "ver" if column_major else "hor"
  stores = [line for line in source.splitlines() if f"sme::store_{axis}<" in line]
  assert stores
  assert all(re.search(r"sme_slice_base_\d+\+\([^)]+\)\*dsl_c_sme_slice_stride_\d+", line) for line in stores)
  # ZA asm clobbers memory; C.raw_data()/stride() must be absent from the
  # slice instruction, or LLVM would reload its metadata after every store.
  assert all("C.data()" not in line and "C.strides()" not in line for line in stores)
  pointer = "C.data()"
  if metadata_snapshot:
    entry_pointer = re.search(r"const auto (\w+)\s*=\s*C\.data\(\);", source)
    assert entry_pointer is not None
    pointer = entry_pointer[1]
  # Trace the hoisted base back to C's descriptor/snapshot binding, not just
  # an absence-of-descriptor substring after the global snapshot rewrite.
  assert any("sme_slice_base_" in line and pointer in line for line in source.splitlines())
  # Empty meta blocks do not even form a tile-origin pointer.
  assert re.search(r"sme_slice_base_\d+=\(\(.+\)>0 && \(.+\)>0 \?", source)


@pytest.mark.parametrize("metadata_snapshot", (False, True))
def test_sme_direct_slices_hoist_runtime_row_stride(metadata_snapshot):
  M, N, K = (T.dynamic(lower=1) for _ in range(3))
  CS = T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), (CS, 1), "float32", access="output"),
  ):
    T.dot((A, B), C, implementation="sme")

  source = (
    CPUBackend()
    .lower(kernel.ir, (), target="SME", options={"cpu_parameter_metadata_snapshot": metadata_snapshot})
    .source
  )
  stride = "C.stride(0)"
  if metadata_snapshot:
    entry_stride = re.search(r"const auto (\w+)\s*=\s*C\.stride\(0\);", source)
    assert entry_stride is not None
    stride = entry_stride[1]
  assert re.search(rf"const vecops::nint_t dsl_c_sme_slice_stride_\d+={re.escape(stride)};", source)
  stores = [line for line in source.splitlines() if "sme::store_hor<" in line]
  assert stores
  assert all("C.stride(0)" not in line for line in stores)


@pytest.mark.parametrize("dynamic_spatial", (False, True))
def test_sme_runtime_small_product_dispatch_precedes_cache_query(dynamic_spatial):
  M, N = (T.dynamic(lower=0), T.dynamic(lower=0)) if dynamic_spatial else (19, 23)
  K = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    D: T.Tensor((M, N), dtype="float32"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    result = T.library.matmul((A, B), C, acc=T.load(D), packing="none", implementation="sme")
    T.store(C, T.max(result + 1, 0))

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert "vecops_dsl::matmul_fits(" in source
  marker = "/* bypass cache panels for small runtime SME products */"
  assert marker in source
  assert source.index(marker) < source.index("=vecops_dsl::matmul_blocks")
  # The direct branch still fuses C-in and epilogue. Only the cache branch
  # owns intermediate C phases; no extra kernel call or opaque native op.
  direct, cache = source[source.index(marker) :].split("=vecops_dsl::matmul_blocks", 1)
  assert "sme::load_hor<" in direct and "sme::read_hor<" in direct
  assert "prologue exactly once" not in direct
  assert "prologue exactly once" in cache and "epilogue exactly once" in cache
  assert "vecops::ops::Matmul" not in source
  assert "/* pack A:" not in source and "/* pack B:" not in source


@pytest.mark.parametrize("blocks", ({"BM": 32}, {"BN": 32}, {"BK": 32}))
def test_sme_explicit_cache_tiles_never_take_runtime_bypass(blocks):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  BM, BN, BK = (blocks.get(key) for key in ("BM", "BN", "BK"))

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    T.library.matmul((A, B), C, implementation="sme", BM=BM, BN=BN, BK=BK)

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert "bypass cache panels" not in source
  assert "=vecops_dsl::matmul_blocks" in source


def test_sme_dynamic_extents_are_captured_before_the_noexcept_owner():
  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    T.dot((A, B), C, implementation="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  head, body = source.split(".with_resources(vecops::execution::sme::Region{}", 1)
  for axis in ("m", "n", "k"):
    assert re.search(rf"const vecops::nint_t dsl_\w+_sme_extent_{axis}_\d+=", head)
  # Family dispatch, K bounds and spatial masks use captured scalars; public
  # MOPA's memory clobber must not force repeated shape-metadata reads.
  loops = re.findall(r"for \(vecops::nint_t dsl_\w+_sme_[mnk]_\d+[^\n]+", body)
  assert loops
  for loop in loops:
    assert "sme_extent_" in loop
    assert ".size(" not in loop


@pytest.mark.parametrize("dtype", ("float32", "int32"))
def test_sme_broadcast_seed_uses_one_za_instruction(dtype):
  input_dtype = "bfloat16" if dtype == "float32" else "int8"

  @T.template
  def kernel(
    A: T.Tensor((25, 32), dtype=input_dtype),
    B: T.Tensor((32, 41), dtype=input_dtype),
    D: T.Tensor((1, 41), dtype=dtype),
    C: T.Tensor((25, 41), dtype=dtype, access="output"),
  ):
    T.dot((A, B), C, acc=T.load(D), implementation="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert "broadcast C seed: one ZA instruction" in source
  assert "vecops::vec::sme::load_hor<" not in source
  assert "vecops::vec::sme::write_hor<" not in source
  if dtype == "int32":
    assert "vecops::vec::sme::addha<" in source
  else:
    assert "vecops::vec::fill(" in source


@pytest.mark.parametrize("column_major", (False, True))
def test_sme_fused_epilogue_keeps_vector_math_but_uses_contiguous_slices(column_major):
  M, N, K = 25, 41, 32
  stride = (1, M) if column_major else (N, 1)

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), stride, "bfloat16", access="output"),
  ):
    result = T.dot((A, B), C, implementation="sme")
    T.store(C, T.max(result, 0))

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  axis = "ver" if column_major else "hor"
  assert f"vecops::vec::sme::read_{axis}<" in source
  assert "vecops::vec::sme::store_hor<" not in source
  assert "vecops::vec::sme::store_ver<" not in source
  assert "vecops::vec::store_convert(" in source


def test_sme_rectangular_families_and_fp64_expanded_catalog():
  @T.template
  def kernel(
    A: T.Tensor((33, 35), dtype="float64"),
    B: T.Tensor((35, 65), dtype="float64"),
    C: T.Tensor((33, 65), dtype="float64", access="output"),
  ):
    T.dot((A, B), C, implementation="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  for family in ("1x4", "4x1", "2x2", "4x2", "2x4"):
    assert f"SME {family} outer-product schedule" in source
  assert "vecops::vec::sme::mopa<7," in source
  assert "sme_operand[" not in source


def test_sme_shared_epilogue_helpers_use_streaming_domain():
  @T.template
  def kernel(
    A: T.Tensor((25, 33), dtype="bfloat16"),
    B: T.Tensor((33, 41), dtype="bfloat16"),
    C: T.Tensor((25, 41), dtype="float32", access="output"),
  ):
    result = T.dot((A, B), C, implementation="sme")
    T.store(C, T.cast(T.cast(result, "float64") + 1, "float32"))

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  # Wide expression fragmentation shares the ordinary vector helper with
  # packing/GEMV. Inside manual SM it must bind to SVL, never normal SVE VL.
  assert re.search(r"vector_size<\w*epilogue_half_tag_\d+, true>", source)
  assert "vecops::vec::sme::streaming_lanes<float>()" in source


def test_sme_staged_schedule_separates_partial_k_group():
  @T.template
  def kernel(
    A: T.Tensor((25, 33), dtype="bfloat16"),
    B: T.Tensor((33, 41), (1, 33), "bfloat16"),
    C: T.Tensor((25, 41), dtype="float32", access="output"),
  ):
    T.dot((A, B), C, implementation="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME", options={"cpu_matmul_pipeline": "staged"}).source
  assert "operand pipeline prime" in source
  assert "one partial K group" in source
  assert "true &&" in source
  assert "whole K-group u32 gather" in source


def test_sme_small_automatic_families_stay_outside_streaming():
  @T.template
  def kernel(
    A: T.Tensor((2, 129), dtype="bfloat16"),
    B: T.Tensor((129, 8), dtype="bfloat16"),
    C: T.Tensor((2, 8), dtype="float32", access="output"),
  ):
    a = T.pack(A, side="A", format="sme")
    b = T.pack(B, side="B", format="sme")
    T.dot((a, b), C)

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert "SME ordinary-SVE skinny/packed-dot family" in source
  assert "widening vector dot" in source
  assert source.index("SME ordinary-SVE skinny/packed-dot family") < source.index("sme_dot_session")
  assert "packed_k_stream" in source


@pytest.mark.parametrize(
  "dtype,acc_dtype", (("bfloat16", "float32"), ("float16", "float32"), ("int8", "int32"), ("uint8", "int32"))
)
@pytest.mark.parametrize("k", (17, 19))
def test_sme_packed_vector_fixed_k_tail_does_not_form_fractional_u32_tag(dtype, acc_dtype, k):
  @T.template
  def kernel(
    AP: T.Tensor((1024,), dtype=dtype),
    BP: T.Tensor((1024,), dtype=dtype),
    C: T.Tensor((2, 3), dtype=acc_dtype, access="output"),
  ):
    a = T.packed(AP, (2, k), side="A", format="sme")
    b = T.packed(BP, (k, 3), side="B", format="sme")
    T.dot((a, b), C, acc_dtype=acc_dtype, implementation="vector")

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  fixed_tails = re.findall(r"using (\w+)\s*=\s*vecops::vec::FixedTag<[^;]+,\s*1>;", source)
  assert fixed_tails
  assert "packed_vector_" in source  # Scalar packed-offset tail; no Vec array.
  for line in source.splitlines():
    if "packed_k_stream" in line and "ViewAs" in line:
      assert not any(re.search(rf"\b{tag}\b", line) for tag in fixed_tails)
