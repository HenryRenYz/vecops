# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Semantic catalogs, target overlays, DAG costs and compilation isolation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.cost import CPUTarget, resolve_target
from vecops.language._compiler.backends.cpu.implementations import (
  CPUImplementation,
  register_implementation,
  unregister_implementation,
)
from vecops.language._compiler.backends.cpu.policies import recompute_limit
from vecops.language._compiler.cost import (
  CostProfile,
  CostQuery,
  CostRule,
  OperationCost,
  estimate,
  register_profile,
  unregister_profile,
)
from vecops.language._compiler.environment import freeze
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.operations import (
  OpDef,
  Trait,
  register_operation,
  snapshot,
  unregister_operation,
  use_catalog,
)
from vecops.language._compiler.pipeline import kernel_pipeline


def input_value(dtype="float32", memory_dtype=None):
  shape = (I.literal(257),)
  view = I.View("A", memory_dtype or dtype, shape, (I.literal(1),), I.literal(0))
  return I.Expr(I.TileType(dtype, shape), "load", data=view)


def chain(count):
  value = input_value()
  for index in range(count):
    value = I.binary("add", value, I.literal(index + 1))
  return value


def test_cost_accumulates_whole_dag_instead_of_expensive_membership():
  model = freeze().costs
  ten = estimate(chain(10), model)
  four = estimate(I.Expr(input_value().type, "div", (input_value(), I.literal(2))), model)
  assert ten.compute_cycles == 10 and four.compute_cycles == 4
  assert ten.compute_cycles >= 6 > four.compute_cycles
  assert ten.compute_cycles >= 3 and four.compute_cycles >= 3
  assert recompute_limit(10, model) < recompute_limit(4, model)


def test_shared_dag_roots_conversion_and_boundaries():
  model = freeze().costs
  x = input_value()
  shared = I.binary("mul", x, x)
  y = I.binary("add", shared, shared)
  cost = estimate((shared, y), model)
  assert cost.compute_cycles == 2 and cost.memory_cycles == 1
  assert estimate(y, model, boundaries=(id(shared),)).compute_cycles == 1
  assert estimate(input_value(memory_dtype="bfloat16"), model).compute_cycles == 1
  reduction = I.Expr(I.ScalarType("float32"), "reduce_add", (chain(10),), 2)
  assert estimate(I.binary("add", y, reduction), model).compute_cycles == 3


def test_cycles_latency_and_physical_word_count_are_distinct():
  model = freeze(target="native").costs
  x = input_value()
  fma = I.Expr(x.type, "fma", (x, x, x))
  cost = estimate(fma, model, words=4)
  assert cost.compute_cycles == 4
  assert cost.memory_cycles == 4
  assert cost.critical_path_cycles > 0
  independent = estimate((chain(3), chain(3)), model)
  dependent = estimate(chain(6), model)
  assert independent.compute_cycles == dependent.compute_cycles == 6
  assert independent.critical_path_cycles < dependent.critical_path_cycles
  with pytest.raises(ValueError, match="word count"):
    estimate(fma, model, words=0)


def test_overlay_precedence_selectors_unknown_and_ambiguity():
  low = CostProfile("unit-low", (CostRule(OperationCost(9, 12), (("operation", "add"),)),))
  high = CostProfile("unit-high", (CostRule(OperationCost(2, 3), (("operation", "add"), ("dtype", "float32"))),))
  model = freeze({"cost_profiles": (low, high)}).costs
  assert estimate(chain(1), model).compute_cycles == 2
  unknown = I.Expr(input_value().type, "unit-unknown", (input_value(),))
  assert estimate(unknown, model).compute_cycles > 0
  assert estimate(unknown, model).unknown == ("unit-unknown",)
  ambiguous = CostProfile(
    "unit-ambiguous",
    (
      CostRule(OperationCost(2, 2), (("operation", "add"),)),
      CostRule(OperationCost(3, 3), (("dtype", "float32"),)),
    ),
  )
  with pytest.raises(ValueError, match="ambiguous"):
    estimate(chain(1), freeze({"cost_profiles": (ambiguous,)}).costs)
  with pytest.raises(ValueError, match="finite"):
    OperationCost(float("nan"), 1)
  with pytest.raises(ValueError, match="selector"):
    CostProfile("invalid", (CostRule(OperationCost(1, 1), (("unknown_field", 1),)),))


def test_explicit_cross_target_never_probes_local_machine(monkeypatch):
  resolve_target.cache_clear()
  import vecops.language._compiler.backends.cpu.cost as cpu_cost

  monkeypatch.setattr(cpu_cost.platform, "machine", lambda: pytest.fail("host probe during cross compile"))
  sve = freeze({"cpu_cost_target": "SVE"})
  assert sve.target.architecture == "arm" and sve.target.isa == "sve"
  assert [p.name for p in sve.costs.profiles] == ["default", "cpu", "sve"]
  query = CostQuery("rcp", "math", "float32", accuracy="estimate")
  assert sve.costs.lookup(query).throughput_cycles == 1
  assert freeze({"cpu_cost_target": "AVX512"}).costs.lookup(query).throughput_cycles == 16
  unknown_cpu = freeze({"cpu_cost_target": CPUTarget("x86", "avx512", "unknown-processor", 512)})
  assert unknown_cpu.costs.profiles[-1].name == "avx512"


@pytest.mark.parametrize("target", ("generic", "AVX512", "SVE", "SME", CPUTarget("arm", "sve")))
def test_explicit_lowering_geometry_never_probes_build_host(monkeypatch, target):
  import vecops._arm_vector_state as vector_state

  monkeypatch.setattr(vector_state, "get_vector_state", lambda: pytest.fail("vector-state probe during cross lowering"))
  environment = freeze(target=target)
  assert environment.vector_state == vector_state.VectorState(None, None)
  captured = vector_state.VectorState(16, 64)
  assert freeze({"_vector_state": captured}, target=target).vector_state == captured


def test_standalone_native_lowering_captures_current_vector_geometry(monkeypatch):
  import vecops._arm_vector_state as vector_state

  captured = vector_state.VectorState(32, 16)
  monkeypatch.setattr(vector_state, "get_vector_state", lambda: captured)
  assert freeze(target="Native").vector_state == captured


def test_registered_semantics_capability_and_cost_snapshots():
  old = freeze()
  name = "unit-new-operation"
  register_operation(OpDef(name, "arithmetic", Trait.ELEMENTWISE | Trait.CSE | Trait.TRANSPOSE_STREAMABLE))
  register_implementation(CPUImplementation(name, "abs"))
  profile = CostProfile("unit-registered-profile", (CostRule(OperationCost(7, 9), (("operation", name),)),))
  register_profile(profile)
  try:
    current = freeze({"cost_profiles": (profile.name,)})
    assert current.operations.has(name, Trait.ELEMENTWISE)
    assert name in current.operations.names(Trait.CSE)
    assert current.implementations.get(name).vector_name == "abs"
    assert not old.operations.has(name, Trait.ELEMENTWISE)
    assert current.fingerprint != old.fingerprint
    with use_catalog(old.operations):
      from vecops.language._compiler.operations import has

      assert not has(name, Trait.CSE)
    expression = I.Expr(input_value().type, name, (input_value(),))
    assert estimate(expression, current.costs).compute_cycles == 7
    # No profile entry needed to give a new arithmetic operation a useful cost.
    assert estimate(expression, freeze().costs).compute_cycles == 1
    with pytest.raises(ValueError, match="already registered"):
      register_operation(OpDef(name, "arithmetic"))
  finally:
    unregister_profile(profile.name)
    unregister_implementation(name)
    unregister_operation(name)
  assert estimate(expression, current.costs).compute_cycles == 7


def test_profile_changes_invalidate_decorator_cache():
  calls = []

  @T.jit
  def factory(n):
    calls.append(n)
    return lambda: n

  first = factory(3)
  assert factory(3) is first
  register_operation(OpDef("unit-cache-operation", "arithmetic", Trait.ELEMENTWISE))
  try:
    assert factory(3) is not first
    assert len(calls) == 2
  finally:
    unregister_operation("unit-cache-operation")


def test_long_affine_reduction_source_can_be_materialized():
  @T.template
  def kernel(
    A: T.Tensor((257,), dtype="float32"),
    B: T.Tensor((257,), dtype="float32", access="output"),
    C: T.Tensor((1,), dtype="float32", access="output"),
  ):
    x = T.load(A)
    x = ((((x * 2 + 1) * 3 + 2) * 4 + 3) * 5 + 4) * 6 + 5
    total = T.reduce_add(x)
    T.store(B, x / total)
    C[0] = total

  ir = kernel_pipeline().run(lower(kernel.ir))
  assert ir.materializations


def test_scalar_conditional_does_not_add_both_arms():
  model = freeze().costs
  condition = I.Expr(I.ScalarType("bool"), "raw", data="condition")
  value = I.literal(2.0)
  a = I.Expr(value.type, "sqrt", (value,))
  b = I.Expr(value.type, "exp", (value,))
  select = I.Expr(value.type, "where", (condition, a, b))
  assert (
    estimate(select, model).compute_cycles
    == max(estimate(a, model).compute_cycles, estimate(b, model).compute_cycles) + 1
  )
  assert (
    estimate(replace(select, args=(I.literal(True), a, b)), model).compute_cycles
    == estimate(a, model).compute_cycles + 1
  )


def test_environment_is_shared_by_all_pipeline_stages_and_threads():
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  seen = []
  name = "unit-catalog-isolation"

  @T.passes.register_pass(name, before="canonicalize-memory")
  def observe(ir, context):
    seen.append(context.environment)
    return ir

  @T.template
  def kernel(A: T.Tensor((257,), dtype="float32"), C: T.Tensor((257,), dtype="float32", access="output")):
    T.store(C, T.exp(T.load(A)))

  try:
    with ThreadPoolExecutor(max_workers=2) as pool:
      artifacts = list(pool.map(lambda _: CPUBackend().lower(kernel.ir, ()), range(2)))
    assert len(seen) == 2 and seen[0] is seen[1]
    assert artifacts[0].source == artifacts[1].source
    assert artifacts[0].lowered.manager.options["_compiler_environment"] is seen[0]
    changed = CPUBackend().lower(
      kernel.ir,
      (),
      options={"cost_profiles": (CostProfile("unit-change", policies=(("math_chunk_compute_budget", 999),)),)},
    )
    assert changed.source != artifacts[0].source
  finally:
    T.passes.unregister_pass(name)


def test_chunk_policy_uses_whole_region_and_live_value_budget():
  from vecops.language._compiler.lowering.tile_to_vector import lower_region
  from vecops.language._compiler.pipeline import cpu_vector_pipeline

  value = chain(10)
  region = lower_region(value.type.shape, "float32", (value,), manager=cpu_vector_pipeline())
  assert region.base_power == 1
  assert region.cost.compute_cycles == 20  # two physical words
  x = input_value()
  roots = tuple(I.binary("add", x, I.literal(index)) for index in range(10))
  wide = lower_region(x.type.shape, "float32", roots, manager=cpu_vector_pipeline())
  assert wide.base_power == 0  # independent outputs exceed the wider live budget
  assert wide.cost.live_vectors >= 10


def test_registry_changes_during_a_pass_do_not_change_its_ir_contract():
  import operator

  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  name, pass_name = "unit-late-fold", "unit-register-late-fold"
  observations = []

  @T.passes.register_pass(pass_name, before="canonicalize-memory")
  def late(ir, context):
    if name not in snapshot().definitions:
      register_operation(OpDef(name, "arithmetic", Trait.CSE, arity=2, fold=operator.add))
    observations.append(I.binary(name, I.literal(1), I.literal(2)).op)
    return ir

  @T.template
  def kernel(A: T.Tensor((17,), dtype="float32"), C: T.Tensor((17,), dtype="float32", access="output")):
    T.store(C, T.load(A) + 1)

  try:
    CPUBackend().lower(kernel.ir, ())
    CPUBackend().lower(kernel.ir, ())
    assert observations == [name, "literal"]
  finally:
    T.passes.unregister_pass(pass_name)
    unregister_operation(name)


def test_registered_profiles_invalidate_outer_cache_even_for_other_targets():
  calls = []

  @T.jit
  def factory():
    calls.append(1)
    return lambda: 1

  first = factory()
  profile = CostProfile("unit-other-target-profile", version="2")
  register_profile(profile)
  try:
    assert factory() is not first and len(calls) == 2
    with pytest.raises(ValueError, match="unconditional CSE"):
      register_operation(OpDef("unit-unsafe-cse", "memory", Trait.CSE, effects="read"))
  finally:
    unregister_profile(profile.name)


@pytest.mark.parametrize("input_dtype,output_dtype", [("int8", "int8"), ("float16", "float64")])
def test_immutable_dynamic_extents_preserve_automatic_transpose(input_dtype, output_dtype):
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  M, N = T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((M, N), (N, 1), dtype=input_dtype), C: T.Tensor((N, M), (M, 1), dtype=output_dtype, access="output")
  ):
    T.transpose_copy(A, C, compute_dtype=output_dtype)

  artifact = CPUBackend().lower(kernel.ir, (), target="SME")
  plans = tuple(artifact.backend_ir.transposes.values())
  assert len(plans) == 1 and plans[0].sme
  # Same-representation input now loads directly into ZA; converted input
  # still needs load_convert followed by a vector-to-ZA transfer.
  if input_dtype == output_dtype:
    assert "sme::load_hor" in artifact.source
  else:
    assert "sme::write_hor" in artifact.source
  assert not snapshot().has("raw", Trait.CAPTURE_SCALAR)
  assert snapshot().has("local", Trait.CAPTURE_SCALAR)
