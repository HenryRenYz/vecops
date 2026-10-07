"""A current store's disjoint region may release only its fresh captures."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops._arm_vector_state import VectorState
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cost import CPUTarget
from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter
from vecops.language._compiler.environment import freeze
from vecops.language._compiler.ir.traversal import nodes


def region_template(boundary="fresh", *, masked=False, scratch=False):
  M, N = T.dynamic(lower=0), T.dynamic(lower=0)
  AS0, AS1, BS0, BS1, CS0, CS1 = (T.dynamic() for _ in range(6))

  @T.template
  def kernel(
    A: T.Tensor((M, N), (AS0, AS1), T.float32, access="inout"),
    B: T.Tensor((M, N), (BS0, BS1), T.float32),
    C: T.Tensor((M, N), (CS0, CS1), T.float32, access="output"),
    D: T.Tensor((M, N), dtype=T.float32, access="output"),
    repeat: int = 1,
  ):
    original = T.load(A)
    if boundary == "before_write":
      T.store(A, 42)
    for iteration in T.serial(repeat, unroll=1):
      value = original if boundary != "per_iteration" else T.load(A)
      if scratch:
        temporary = T.alloc((M, N + iteration), dtype=T.float32)
        T.store(temporary[:, :N], value)
        value = T.load(temporary[:, :N])
      result = T.fma(value, T.cast(0.5, T.float32), T.load(B))
      if masked:
        T.store(C, result, mask=value > 0)
      else:
        T.store(C, result)
      T.compiler_fence()
    if boundary == "future":
      T.store(A, 42)
      for _ in T.serial(1, unroll=1):
        T.store(D, original + 5)

  return kernel


def plain_template(*, masked=False):
  N, AS, BS, CS = T.dynamic(lower=0), T.dynamic(), T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((N,), (AS,), T.float32),
    B: T.Tensor((N,), (BS,), T.float32),
    C: T.Tensor((N,), (CS,), T.float32, access="output"),
  ):
    a = T.load(A)
    result = T.fma(a, T.cast(0.5, T.float32), T.load(B))
    if masked:
      T.store(C, result, mask=a > 0)
    else:
      T.store(C, result)

  return kernel


def transpose_region_template(source="int8", output="float32", compute="float32"):
  """Same repeated whole-operator scope as the maintained transpose bench."""

  @T.template
  def kernel(
    A: T.Tensor((16, 16), (16, 1), source),
    C: T.Tensor((16, 16), (16, 1), output, access="output"),
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      T.transpose_copy(A, C, compute_dtype=compute)
      T.compiler_fence()

  return kernel


def scope(source, marker, *, start=0):
  position = source.index(marker, start)
  opening = source.index("{", position + len(marker))
  depth = 1
  for closing in range(opening + 1, len(source)):
    depth += (source[closing] == "{") - (source[closing] == "}")
    if depth == 0:
      return source[opening + 1 : closing], closing
  raise AssertionError(f"unclosed generated region {marker!r}")


def versioned_regions(source):
  regions, stop = [], 0
  while "/* fresh store snapshots absent */" in source[stop:]:
    fast, fast_end = scope(source, "/* fresh store snapshots absent */", start=stop)
    fallback, stop = scope(source, "/* retain current and historical store snapshots */", start=fast_end)
    regions.append((fast, fallback))
  return regions


@pytest.mark.parametrize("target", ("SVE", "AVX512", "Scalar"))
@pytest.mark.parametrize("masked", (False, True))
def test_fresh_alias_guard_is_outside_vector_dag_and_cold_path_retains_both_captures(target, masked):
  artifact = CPUBackend().lower(plain_template(masked=masked).ir, (), target=target)
  regions = versioned_regions(artifact.source)
  assert len(regions) == 1
  fast, fallback = regions[0]
  assert "dsl_c_alias_" not in fast
  assert "dsl_c_snapshot_" not in fast
  assert "dsl_c_alias_" in fallback
  assert "dsl_c_snapshot_" in fallback
  assert "vecops::vec::load_convert" in fast
  assert "vecops::vec::store_convert" in fast
  assert "vecops::vec::fma" in fast
  assert all(region.unroll == 2 for region in artifact.lowered.plans.values())
  # Cloning emitted scopes must not duplicate or mutate the prepared IR.
  stores = [n for n in nodes(artifact.kernel.body) if isinstance(n, I.Store)]
  assert len(stores) == 1
  assert len(artifact.lowered.plans) == 1
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("boundary", ("before_write", "future"))
def test_capture_before_effect_or_outer_loop_remains_in_versioned_fast_region(boundary):
  artifact = CPUBackend().lower(region_template(boundary).ir, (), target="SVE")
  fast, fallback = next(pair for pair in versioned_regions(artifact.source) if "vecops::vec::fma" in pair[0])
  # The enclosing serial-region guard protects original A. Only B's fresh
  # capture is released here; A must still address its historical buffer.
  assert "dsl_c_snapshot_" in fast
  assert "dsl_c_alias_" in fast
  assert fast.count("dsl_c_snapshot_") < fallback.count("dsl_c_snapshot_")
  assert_no_vec_mask_arrays(artifact.source)


def test_resizable_owned_scratch_is_live_and_never_a_global_pointer_snapshot():
  source = CPUBackend().lower(region_template(scratch=True).ir, (), target="SVE").source
  fast, _ = next(pair for pair in versioned_regions(source) if "vecops::vec::fma" in pair[0])
  assert re.search(r"dsl_f_\w+\.data\(\)", fast)
  assert ".resize(" in source
  assert not re.search(r"const auto dsl_c_parameter_\w+ = dsl_f_\w+\.data\(\)", source)
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64), (None, None)))
def test_region_versioning_preserves_codegen_domain_and_requested_unroll(vl, svl):
  options = {"_vector_state": VectorState(vl, svl), "cpu_cost_target": CPUTarget("x86", "avx512", vector_bits=512)}
  artifact = CPUBackend().lower(plain_template().ir, (), target="SVE", options=options)
  fast, _ = versioned_regions(artifact.source)[0]
  assert all(region.unroll == 2 for region in artifact.lowered.plans.values())
  assert "ScalableTag<float" in fast
  assert "streaming_lanes" not in fast
  assert "_mm" not in fast
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize(
  "source,output,compute",
  (
    ("int8", "float32", "float32"),
    ("int8", "float16", "float32"),
    ("float16", "float32", "float32"),
    ("float16", "float64", "float64"),
  ),
)
def test_repeated_vla_conversion_guards_each_current_store_not_each_vector(source, output, compute):
  artifact = CPUBackend().lower(
    transpose_region_template(source, output, compute).ir,
    (),
    target="SME",
    options={"_vector_state": VectorState(64, 16)},
  )
  assert not artifact.backend_ir.transposes
  fast, fallback = versioned_regions(artifact.source)[0]
  assert "dsl_c_alias_" not in fast and "dsl_c_snapshot_" not in fast
  assert "dsl_c_alias_" in fallback and "dsl_c_snapshot_" in fallback
  assert "vecops::vec::load_convert" in fast and "vecops::vec::store_convert" in fast
  assert all(plan.unroll == 1 for plan in artifact.lowered.plans.values())
  assert_no_vec_mask_arrays(artifact.source)


def test_policy_switch_keeps_exact_old_scalar_alias_selection_and_checks_type():
  artifact = CPUBackend().lower(plain_template().ir, (), target="SVE", options={"store_snapshot_regions": False})
  assert not versioned_regions(artifact.source)
  assert "dsl_c_alias_" in artifact.source
  assert "dsl_c_snapshot_" in artifact.source
  for invalid in (0, 1, None, "true"):
    with pytest.raises(TypeError, match="store_snapshot_regions must be bool"):
      CPUBackend().lower(plain_template().ir, (), target="SVE", options={"store_snapshot_regions": invalid})


def eligibility_emitter(value):
  environment = freeze(target="SVE")
  view = I.View("C", "float32", value.type.shape, (I.literal(1),), I.literal(0))
  statement = I.Store(view, value)
  load = next(I.loads_unique(value))
  entry = (view, "current_alias")
  emitter = MemoryEmitter()
  emitter.environment = environment
  emitter.vector_ir = SimpleNamespace(manager=SimpleNamespace(options={}))
  emitter.snapshots = {id(load): entry}
  emitter.store_snapshot_provenance = {id(statement): (statement, {id(load): entry})}
  return emitter, statement, SimpleNamespace(roots=(value,)), load, entry


@pytest.mark.parametrize("opcode", ("state_ref", "dot", "atomic_add", "scan", "unknown_plugin"))
def test_effect_state_and_opaque_boundaries_are_never_versioned(opcode):
  shape = (I.literal(17),)
  view = I.View("A", "float32", shape, (I.literal(1),), I.literal(0))
  load = I.Expr(I.TileType("float32", shape), "load", data=view)
  value = I.Expr(load.type, opcode, (load,))
  emitter, statement, plan, _, _ = eligibility_emitter(value)
  assert not emitter.fresh_store_snapshots(statement, plan)


def test_exact_provenance_identity_and_historical_maps_cannot_be_released():
  shape = (I.literal(17),)
  view = I.View("A", "float32", shape, (I.literal(1),), I.literal(0))
  load = I.Expr(I.TileType("float32", shape), "load", data=view)
  emitter, statement, plan, load, entry = eligibility_emitter(load)
  assert emitter.fresh_store_snapshots(statement, plan) == {id(load): entry}
  # A recapture or dictionary restoration to a different region's entry is
  # not proved false by the recorded current-store guard.
  emitter.snapshots[id(load)] = (view, "historical_alias")
  assert not emitter.fresh_store_snapshots(statement, plan)
  emitter.snapshots[id(load)] = entry
  other = I.Store(statement.destination, statement.value)
  emitter.store_snapshot_provenance[id(statement)] = (other, {id(load): entry})
  assert not emitter.fresh_store_snapshots(statement, plan)


def test_branch_locals_do_not_escape_or_replace_historical_materialization_maps():
  shape = (I.literal(17),)
  view = I.View("A", "float32", shape, (I.literal(1),), I.literal(0))
  load = I.Expr(I.TileType("float32", shape), "load", data=view)
  emitter, statement, plan, _, fresh_entry = eligibility_emitter(load)
  historical = (view, "historical_alias")
  emitter.snapshots[904] = historical
  emitter.snapshots[905] = historical  # Existing proxy of a historical load.
  emitter.materialized = {906: (view, "producer_enabled")}
  emitter.reduction_materialized = {907: (view, "older_reduction_enabled")}
  emitter.chunk_cache = {"outside": "outer_cpp_local"}
  emitter.vector_ir.plans = {id(statement): plan}
  emitter.emitted_regions = []
  before = tuple(
    dict(mapping)
    for mapping in (emitter.snapshots, emitter.materialized, emitter.reduction_materialized, emitter.chunk_cache)
  )
  seen = []
  emitter.opening = lambda _: None
  emitter.close = lambda: None

  def traversal(region, callback):
    assert region is plan
    seen.append(
      tuple(
        dict(mapping)
        for mapping in (emitter.snapshots, emitter.materialized, emitter.reduction_materialized, emitter.chunk_cache)
      )
    )
    # Emission creates proxy mappings and branch-local cache entries. None
    # may leak into the other branch or into a later nested hardware region.
    emitter.snapshots[999] = (view, "branch_alias")
    emitter.materialized[998] = (view, "branch_producer")
    emitter.reduction_materialized[997] = (view, "branch_reduction")
    emitter.chunk_cache["branch"] = "branch_cpp_local"

  emitter.traverse_region = traversal
  emitter.streaming_store(statement)
  assert len(seen) == 2
  assert id(load) not in seen[0][0]
  assert seen[1][0][id(load)] is fresh_entry
  for snapshot_map, producer_map, reduction_map, cache in seen:
    assert snapshot_map[904] is historical and snapshot_map[905] is historical
    assert producer_map == before[1]
    assert reduction_map == before[2]
    assert not cache
    assert 999 not in snapshot_map
  assert (
    tuple(
      dict(mapping)
      for mapping in (emitter.snapshots, emitter.materialized, emitter.reduction_materialized, emitter.chunk_cache)
    )
    == before
  )
  assert not emitter.emitted_regions


@pytest.fixture(scope="module")
def native_store_region_kernels(tmp_path_factory):
  path = tmp_path_factory.mktemp("store-snapshot-regions")
  compiler = make_compiler(jobs=4, output="quiet", cache_dir=path / "cache", build_dir=path / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for boundary in ("fresh", "before_write", "future", "per_iteration", "scratch", "masked"):
      template = region_template(
        "fresh" if boundary in {"scratch", "masked"} else boundary,
        masked=boundary == "masked",
        scratch=boundary == "scratch",
      )
      for policy in (False, True):
        compiled = T.compile(template, compiler=compiler, pass_options={"store_snapshot_regions": policy})
        metadata = (vecops.TensorMeta((3, 17), dtype="float32"),) * 4
        compiled.operator(*metadata, 3, **compiled._core.specialization_values)
        kernels[boundary, policy] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("boundary", ("fresh", "before_write", "future", "per_iteration", "scratch", "masked"))
@pytest.mark.parametrize("policy", (False, True))
def test_native_regions_match_independent_sequential_oracle_for_aliases_captures_and_resizes(
  native_store_region_kernels, boundary, policy
):
  compiled = native_store_region_kernels[boundary, policy]
  rng = np.random.default_rng(2844)
  for width in (0, 1, 7, 17, 65):
    for layout in ("dense", "negative", "strided-output", "inplace", "shifted", "broadcast-read", "broadcast-alias"):
      count = 3 * width
      actual = rng.normal(size=count + 1).astype(np.float32)
      expected = actual.copy()
      a, expected_a = actual[:count].reshape(3, width), expected[:count].reshape(3, width)
      if layout == "negative":
        a, expected_a = a[:, ::-1], expected_a[:, ::-1]
      b = rng.normal(size=(3, width)).astype(np.float32)
      if layout == "broadcast-read" and width:
        b = np.broadcast_to(b[:, :1], (3, width))
      output_storage = np.full((3, 2 * width), -19, dtype=np.float32)
      expected_output_storage = output_storage.copy()
      c, expected_c = output_storage[:, :width], expected_output_storage[:, :width]
      if layout == "negative":
        c, expected_c = c[:, ::-1], expected_c[:, ::-1]
      elif layout == "strided-output":
        c, expected_c = output_storage[:, ::2], expected_output_storage[:, ::2]
      elif layout == "inplace":
        c, expected_c = a, expected_a
      elif layout == "shifted":
        c = actual[1 : count + 1].reshape(3, width)
        expected_c = expected[1 : count + 1].reshape(3, width)
      if layout == "broadcast-alias" and width:
        b = np.broadcast_to(c[:, :1], (3, width))
      expected_b = b.copy()
      original = expected_a.copy()
      if boundary == "before_write":
        expected_a[:] = 42
      for _ in range(3):
        value = expected_a.copy() if boundary == "per_iteration" else original
        if layout == "broadcast-alias" and width:
          expected_b = np.broadcast_to(expected_c[:, :1], (3, width)).copy()
        # FMA is explicit in the DSL; one float64 computation followed by
        # F32 rounding is exact for these F32 inputs and the power-of-two 0.5.
        result = (value.astype(np.float64) * 0.5 + expected_b.astype(np.float64)).astype(np.float32)
        if boundary == "masked":
          np.copyto(expected_c, result, where=value > 0)
        else:
          expected_c[:] = result
      other = np.full((3, width), -73, dtype=np.float32)
      expected_other = other.copy()
      if boundary == "future":
        expected_a[:] = 42
        expected_other[:] = original + np.float32(5)
      compiled.operator.native(a, b, c, other, 3, **compiled._core.specialization_values)
      label = f"boundary={boundary}, policy={policy}, width={width}, layout={layout}"
      np.testing.assert_array_equal(c, expected_c, err_msg=label)
      np.testing.assert_array_equal(a, expected_a, err_msg=label)
      np.testing.assert_array_equal(other, expected_other, err_msg=label)
      np.testing.assert_array_equal(actual, expected, err_msg=label)
      np.testing.assert_array_equal(output_storage, expected_output_storage, err_msg=label)
