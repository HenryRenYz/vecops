# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Non-SME constant conversions use native gathers only with a captured SVE VL."""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops._arm_vector_state import VectorState
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cost import CPUTarget
from vecops.language._compiler.backends.cpu.transpose import plan_transpose, small_vla_conversion
from vecops.language._compiler.environment import freeze
from vecops.language._compiler.ir.traversal import nodes

CONVERSIONS = tuple(
  (source, output, "float64" if "float64" in (source, output) else "float32")
  for source in ("int8", "float16", "float32", "float64")
  for output in ("int8", "float16", "float32", "float64")
  if source != output
)


def transpose_template(
  source="float16",
  output="float32",
  compute="float32",
  *,
  shape=(16, 16),
  implementation="auto",
  arbitrary_strides=False,
  expression="load",
):
  m, n = shape
  source_strides = tuple(T.dynamic() for _ in range(2)) if arbitrary_strides else (n, 1)
  output_strides = tuple(T.dynamic() for _ in range(2)) if arbitrary_strides else (m, 1)

  @T.template
  def kernel(
    A: T.Tensor((m, n), source_strides, source),
    C: T.Tensor((n, m), output_strides, output, access="output"),
  ):
    if expression == "load":
      T.transpose_copy(A, C, compute_dtype=compute, implementation=implementation)
    else:
      value = T.load(T.transpose(A), dtype=compute)
      if expression == "affine":
        value = value * 0.5 + 1
      elif expression == "cast":
        value = T.cast(value, dtype="float16")
      T.store(C, value, implementation=implementation)

  return kernel


def lower(template, *, vl=64, svl=64, target="SME", cost=None):
  options = {"_vector_state": VectorState(vl, svl)}
  if cost is not None:
    options["cpu_cost_target"] = cost
  return CPUBackend().lower(template.ir, (), target=target, options=options)


def output_store(artifact):
  return next(
    node
    for node in nodes(artifact.kernel.body)
    if isinstance(node, I.Store) and node.destination.storage == "C" and len(node.destination.shape) == 2
  )


@pytest.mark.parametrize("source,output,compute", CONVERSIONS)
def test_small_constant_conversion_uses_single_word_existing_memory_lowering(source, output, compute):
  artifact = lower(transpose_template(source, output, compute))
  statement = output_store(artifact)
  assert id(statement) not in artifact.backend_ir.transposes
  region = artifact.lowered.plans[id(statement)]
  assert region.unroll == 1
  assert region.base_power == 0
  assert tuple(region.groupings) == (0,)
  assert region.groupings[0][0].power == 0
  assert statement.vector_extent is None  # No fabricated local-SIMD proof.
  assert artifact.kernel.vector_regions is artifact.lowered.regions
  assert sum(item is region for item in artifact.lowered.regions) == 1
  assert "unroll-vector-operations" in region.pass_trace
  assert "vecops::vec::local_interleave" not in artifact.source
  assert "vecops::vec::load_convert" in artifact.source
  assert "vecops::vec::store_convert" in artifact.source
  # The existing alias snapshots remain available on overlapping calls.
  assert "dsl_c_alias_" in artifact.source
  assert "dsl_c_snapshot_" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize(
  "vl,svl,selected",
  ((16, 64, False), (32, 16, True), (64, 32, True), (128, 16, True), (None, 64, False), (None, None, False)),
)
def test_selection_uses_ordinary_vl_not_streaming_vl_or_cost_model(vl, svl, selected):
  artifact = lower(transpose_template(), vl=vl, svl=svl, cost=CPUTarget("x86", "avx512", vector_bits=512))
  statement = output_store(artifact)
  assert (id(statement) not in artifact.backend_ir.transposes) is selected
  assert artifact.lowered.plans[id(statement)].unroll == (1 if selected else 2)


@pytest.mark.parametrize("target", ("AVX512", "Scalar", "generic"))
def test_arm_cost_override_never_changes_other_codegen_targets(target):
  artifact = lower(transpose_template(), target=target, cost=CPUTarget("arm", "sme", vector_bits=512))
  statement = output_store(artifact)
  assert id(statement) in artifact.backend_ir.transposes
  assert artifact.lowered.plans[id(statement)].unroll == 2


@pytest.mark.parametrize("implementation", ("vector", "sme", "gather"))
def test_explicit_backend_overrides_keep_their_existing_policy(implementation):
  artifact = lower(transpose_template(implementation=implementation))
  statement = output_store(artifact)
  assert (id(statement) in artifact.backend_ir.transposes) is (implementation != "gather")
  assert artifact.lowered.plans[id(statement)].unroll == 2


@pytest.mark.parametrize(
  "kwargs",
  (
    {"source": "float32", "output": "float32", "compute": "float32"},
    {"shape": (17, 17)},
    {"shape": (1024, 1024)},
    {"shape": (T.dynamic(lower=0), T.dynamic(lower=0))},
    {"expression": "affine"},
    {"source": "float64", "expression": "cast"},
  ),
)
def test_raw_large_dynamic_and_nontrivial_producers_keep_existing_transpose(kwargs):
  artifact = lower(transpose_template(**kwargs))
  statement = output_store(artifact)
  assert id(statement) in artifact.backend_ir.transposes
  assert artifact.lowered.plans[id(statement)].unroll == 2


def test_intermediate_load_rounding_and_unknown_strides_use_general_memory_semantics():
  artifact = lower(transpose_template("float64", "float32", "float16", arbitrary_strides=True))
  statement = output_store(artifact)
  assert id(statement) not in artifact.backend_ir.transposes
  region = artifact.lowered.plans[id(statement)]
  assert region.dtype == "float16"
  assert region.roots[0].type.dtype == "float16"
  assert next(I.loads_unique(region.roots[0])).data.dtype == "float64"
  assert region.effects[0].store.destination.dtype == "float32"
  assert "vecops::vec::ScalableTag<vecops::float16_t, 0>" in artifact.source
  assert "dsl_c_alias_" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize(
  "source,shape,selected",
  (
    ("float16", (16, 16), True),
    ("float16", (1, 257), False),
    ("float64", (1, 257), True),
    ("float64", (63, 129), True),
    ("float64", (64, 128), True),
    ("float64", (3, 2731), False),
  ),
)
def test_vla_gather_reuses_shared_sme_conversion_rejection_thresholds(source, shape, selected):
  compute = "float64" if source == "float64" else "float32"
  template = transpose_template(source, "float32", compute, shape=shape)
  environment = freeze({"_vector_state": VectorState(64, 64)}, target="SME")
  options = {"_compiler_environment": environment}
  # 257 is prime, hence that exact threshold case has a unit axis. Construct
  # the logical rank-two store to audit policy before the frontend folds it.
  m, n = shape
  extents = (I.literal(n), I.literal(m))
  zero, one = I.literal(0), I.literal(1)
  input_view = I.View("A", source, extents, (one, I.literal(n)), zero, writable=False)
  original = I.Store(
    I.View("C", "float32", extents, (I.literal(m), one), zero),
    I.Expr(I.TileType(compute, extents), "load", data=input_view),
  )
  plan = plan_transpose(original, options)
  assert plan is not None
  assert (not plan.sme_preferred) is selected
  assert small_vla_conversion(plan, options) is selected
  artifact = lower(template)
  if 1 in shape:
    # The prime 257-element threshold case necessarily has a unit axis: the generic
    # canonicalizer removes that axis before the CPU transpose planning stage.
    # Audit the shared policy before canonicalization, and separately preserve
    # the resulting ordinary rank-one streaming path.
    assert not artifact.backend_ir.transposes
    assert_no_vec_mask_arrays(artifact.source)
    return
  statement = output_store(artifact)
  assert (id(statement) not in artifact.backend_ir.transposes) is selected
  assert artifact.lowered.plans[id(statement)].unroll == (1 if selected else 2)
  if selected:
    assert "vecops::vec::local_interleave" not in artifact.source
    assert "vecops::vec::load_convert" in artifact.source
    assert "vecops::vec::store_convert" in artifact.source
  else:
    assert artifact.backend_ir.transposes[id(statement)].sme_preferred
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("vl,svl,selected", ((16, 64, False), (32, 16, True), (64, 32, True), (None, 64, False)))
@pytest.mark.parametrize("output", ("int8", "float16", "float32"))
def test_medium_fp64_conversion_requires_captured_ordinary_vl_even_with_cost_override(vl, svl, selected, output):
  artifact = lower(
    transpose_template("float64", output, "float64", shape=(63, 129)),
    vl=vl,
    svl=svl,
    cost=CPUTarget("x86", "avx512", vector_bits=512),
  )
  statement = output_store(artifact)
  assert (id(statement) not in artifact.backend_ir.transposes) is selected
  assert artifact.lowered.plans[id(statement)].unroll == (1 if selected else 2)


@pytest.mark.parametrize("implementation", ("vector", "sme", "gather"))
def test_medium_conversion_keeps_explicit_policy(implementation):
  artifact = lower(transpose_template("float64", "float16", "float64", shape=(63, 129), implementation=implementation))
  statement = output_store(artifact)
  assert (id(statement) in artifact.backend_ir.transposes) is (implementation != "gather")
  assert artifact.lowered.plans[id(statement)].unroll == 2


@pytest.mark.parametrize("expression", ("affine", "cast"))
def test_medium_nontrivial_producer_keeps_register_plan(expression):
  artifact = lower(transpose_template("float64", "float32", "float32", shape=(63, 129), expression=expression))
  statement = output_store(artifact)
  assert id(statement) in artifact.backend_ir.transposes
  assert artifact.lowered.plans[id(statement)].unroll == 2


def test_medium_intermediate_rounding_and_alias_snapshots_remain_in_vector_ir():
  artifact = lower(transpose_template("float64", "float32", "float16", shape=(63, 129), arbitrary_strides=True))
  statement = output_store(artifact)
  assert id(statement) not in artifact.backend_ir.transposes
  region = artifact.lowered.plans[id(statement)]
  assert region.dtype == "float16"
  assert region.roots[0].type.dtype == "float16"
  assert next(I.loads_unique(region.roots[0])).data.dtype == "float64"
  assert region.effects[0].store.destination.dtype == "float32"
  assert "dsl_c_alias_" in artifact.source
  assert "dsl_c_snapshot_" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


MEDIUM_CONVERSIONS = (
  ("float64", "int8", "float64"),
  ("float64", "float16", "float64"),
  ("float64", "float32", "float64"),
  ("float64", "float32", "float16"),
)


@pytest.fixture(scope="module")
def native_medium_conversions(tmp_path_factory):
  directory = tmp_path_factory.mktemp("medium-vla-transpose")
  compiler = make_compiler(jobs=4, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for types in MEDIUM_CONVERSIONS:
      compiled = T.compile(transpose_template(*types, shape=(63, 129), arbitrary_strides=True), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((63, 129), strides=(129, 1), dtype=types[0]),
        vecops.TensorMeta((129, 63), strides=(63, 1), dtype=types[1]),
        **compiled._core.specialization_values,
      )
      kernels[types] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("source,output,compute", MEDIUM_CONVERSIONS)
def test_native_medium_conversions_preserve_tails_alias_negative_strides_and_rounding(
  native_medium_conversions, source, output, compute
):
  compiled = native_medium_conversions[source, output, compute]
  assert_no_vec_mask_arrays(compiled.generated_source)
  values = ((np.arange(63 * 129, dtype=np.float64).reshape(63, 129) % 127) - 63) / 8 + 0.00017
  original = values.astype(source)
  for layout in ("dense", "negative-input", "negative-output", "strided-output", "shared-storage", "shifted-alias"):
    a = original.copy()
    if layout == "negative-input":
      a = a[::-1, ::-1]
    if layout == "strided-output":
      c = np.empty((129, 126), dtype=output)[:, ::2]
    elif layout in ("shared-storage", "shifted-alias"):
      offset = np.dtype(output).itemsize if layout == "shifted-alias" else 0
      storage = np.empty(original.nbytes + offset, dtype=np.uint8)
      a = np.ndarray((63, 129), dtype=source, buffer=storage)
      a[...] = original
      c = np.ndarray((129, 63), dtype=output, buffer=storage, offset=offset)
    else:
      c = np.empty((129, 63), dtype=output)
      if layout == "negative-output":
        c = c[::-1, ::-1]
    expected = a.astype(compute).T.astype(output)
    compiled.operator.native(a, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, expected)


@pytest.fixture(scope="module")
def native_small_conversions(tmp_path_factory):
  directory = tmp_path_factory.mktemp("small-vla-transpose")
  compiler = make_compiler(jobs=4, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for types in (*CONVERSIONS, ("float64", "float32", "float16")):
      compiled = T.compile(transpose_template(*types, arbitrary_strides=True), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((16, 16), strides=(16, 1), dtype=types[0]),
        vecops.TensorMeta((16, 16), strides=(16, 1), dtype=types[1]),
        **compiled._core.specialization_values,
      )
      kernels[types] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("source,output,compute", (*CONVERSIONS, ("float64", "float32", "float16")))
def test_native_small_conversions_preserve_alias_negative_strides_and_rounding(
  native_small_conversions, source, output, compute
):
  compiled = native_small_conversions[source, output, compute]
  assert_no_vec_mask_arrays(compiled.generated_source)
  values = ((np.arange(256, dtype=np.float64).reshape(16, 16) % 127) - 63) / 8
  values += 0.00017  # Expose the explicit float16 compute-rounding boundary.
  original = values.astype(source)
  for layout in ("dense", "negative-input", "negative-output", "strided-output", "shared-storage"):
    a = original.copy()
    if layout == "negative-input":
      a = a[:, ::-1]
    if layout == "strided-output":
      c = np.empty((16, 32), dtype=output)[:, ::2]
    elif layout == "shared-storage":
      storage = np.empty(256 * max(np.dtype(source).itemsize, np.dtype(output).itemsize), dtype=np.uint8)
      a = np.ndarray((16, 16), dtype=source, buffer=storage)
      a[...] = original
      c = np.ndarray((16, 16), dtype=output, buffer=storage)
    else:
      c = np.empty((16, 16), dtype=output)
      if layout == "negative-output":
        c = c[:, ::-1]
    expected = a.astype(compute).T.astype(output)
    compiled.operator.native(a, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, expected)
  # The only arrays in generated code are scalar bookkeeping; Vec/Mask
  # storage continues to use the normal named-vector lowering.
  assert not re.search(r"std::(?:array|vector)\s*<\s*vecops::vec::(?:Vec|Mask)", compiled.generated_source)
