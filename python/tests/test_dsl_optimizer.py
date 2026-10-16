# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compiler programs, semantic cleanup and motion preserve observable behavior."""

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.effects import ValueEffects
from vecops.language._compiler.ir.verification import IRVerificationError, verify_kernel, verify_vector
from vecops.language._compiler.operations import OpDef, register_operation, snapshot, unregister_operation
from vecops.language._compiler.passes.manager import PassContext
from vecops.language._compiler.pipeline import kernel_pipeline


def context(stage="kernel-cleanup"):
  return PassContext(stage)


def kernel(statements):
  return I.KernelIR(I.Block(statements), {}, {})


def view(name="C"):
  return I.View(name, "int64", (), (), I.literal(0))


def test_explicit_repeated_invocations_and_option_analysis_isolation():
  P = T.passes

  @P.register_pass("optimizer-test-repeat", stage="optimizer-test")
  def callback(value, ctx):
    observed = ctx.analysis("option", lambda _: ctx.options.get("amount", 1), value)
    value.append(observed)
    return P.PassResult(value, changed=False)

  try:
    manager = P.PassManager(
      [
        P.PassInvocation("optimizer-test-repeat", {"amount": 2}, label="first"),
        P.PassInvocation("optimizer-test-repeat", {"amount": 3}, label="second"),
        "optimizer-test-repeat",
      ],
      stage="optimizer-test",
    )
    assert manager.run([]) == [2, 3, 1]
    assert len(manager.specs) == 3
  finally:
    P.unregister_pass("optimizer-test-repeat")


@pytest.mark.parametrize("limit,expected", [(1, 4), (3, 2), (10, 0)])
def test_fixed_point_groups_are_bounded_and_stop_on_actual_convergence(limit, expected):
  P = T.passes
  calls = []

  @P.register_pass("optimizer-test-countdown", stage="optimizer-test")
  def callback(value, ctx):
    calls.append(value[0])
    changed = value[0] > 0
    value[0] = max(0, value[0] - 1)
    return P.PassResult(value, changed=changed)

  try:
    manager = P.PassManager(
      [P.PassGroup("countdown", ("optimizer-test-countdown",), max_iterations=limit)], stage="optimizer-test"
    )
    assert manager.run([5]) == [expected]
    assert len(calls) == min(limit, 6)
  finally:
    P.unregister_pass("optimizer-test-countdown")


def test_groups_freeze_callbacks_and_require_matching_stage():
  P = T.passes

  @P.register_pass("optimizer-test-snapshot", stage="optimizer-test")
  def callback(value, ctx):
    value.append(ctx.stage)
    return P.PassResult(value, changed=False)

  try:
    manager = P.PassManager([P.PassGroup("scope", ("optimizer-test-snapshot",), stage="optimizer-test")])
    with pytest.raises(ValueError, match="unknown kernel"):
      P.PassManager(["optimizer-test-snapshot"])
    P.unregister_pass("optimizer-test-snapshot")
    assert manager.run([]) == ["optimizer-test"]
  finally:
    if any(p.name == "optimizer-test-snapshot" for p in P.registered_passes()):
      P.unregister_pass("optimizer-test-snapshot")


def test_cleanup_prunes_dead_arms_without_evaluating_invalid_arithmetic():
  from vecops.language._compiler.passes.kernel.cleanup import simplify_ir

  invalid = I.Expr(I.ScalarType("int64"), "floordiv", (I.literal(1), I.literal(0)))
  ir = kernel(
    [
      I.If(
        I.literal(False),
        I.Block([I.Store(view(), invalid)]),
        I.Block([I.Assert(I.literal(True), "true"), I.Store(view(), I.literal(7))]),
      )
    ]
  )
  result = simplify_ir(ir, context())
  assert result.changed
  assert len(ir.body.statements) == 1
  assert I.constant(ir.body.statements[0].value) == 7
  assert not simplify_ir(ir, context()).changed


def test_cleanup_preserves_float_rounding_and_signed_zero_operations():
  from vecops.language._compiler.passes.kernel.cleanup import simplify_ir

  value = I.Expr(I.ScalarType("float32"), "raw", data="x")
  expression = I.Expr(value.type, "add", (value, I.literal(0.0, "float32")))
  ir = kernel([I.Store(I.View("C", "float32", (), (), I.literal(0)), expression)])
  simplify_ir(ir, context())
  assert ir.body.statements[0].value is expression


def test_dce_keeps_history_and_traps_but_removes_unused_provenance():
  from vecops.language._compiler.passes.kernel.cleanup import dead_code_elimination

  old = I.Expr(I.ScalarType("int64"), "load", data=view("A"))
  unused = I.Expr(I.ScalarType("int64"), "load", data=view("B"))
  trap = I.Expr(I.ScalarType("int64"), "checked_shape", (I.literal(-1),))
  ir = kernel(
    [
      I.BindTile(old),
      I.BindTile(unused),
      I.Assign("dead", I.literal(3)),
      I.Assign("check", trap),
      I.Store(view("A"), I.literal(2)),
      I.Store(view(), old),
    ]
  )
  result = dead_code_elimination(ir, context())
  assert result.changed
  assert [s.value for s in ir.body.statements if isinstance(s, I.BindTile)] == [old]
  assert [s.target for s in ir.body.statements if isinstance(s, I.Assign)] == ["check"]


def test_unknown_operation_is_not_discardable_or_speculatable():
  effects = ValueEffects(snapshot())
  value = I.Expr(I.ScalarType("int64"), "optimizer-unknown")
  assert not effects.discardable(value)
  assert not effects.speculatable(value)
  with pytest.raises(ValueError, match="nontrapping"):
    register_operation(OpDef("optimizer-invalid", "test", speculatable=True))
  definition = OpDef("optimizer-total", "test", may_trap=False, speculatable=True)
  register_operation(definition)
  try:
    assert ValueEffects(snapshot()).speculatable(I.Expr(value.type, definition.name))
  finally:
    unregister_operation(definition.name)


def test_licm_moves_total_metadata_but_keeps_reads_checks_and_carried_values():
  from vecops.language._compiler.passes.kernel.motion import loop_invariant_code_motion

  n = I.Expr(I.ScalarType("int32"), "raw", data="n")
  invariant = I.Expr(I.ScalarType("int64"), "mul", (n, I.literal(2)))
  i = I.Expr(n.type, "local", data="i")
  loop = I.For(
    "i",
    I.literal(0),
    n,
    I.literal(1),
    I.Block(
      [
        I.Assign("width", invariant),
        I.Assign("read", I.Expr(n.type, "load", data=view("A"))),
        I.Assign("check", I.Expr(n.type, "checked_shape", (n,))),
        I.Assign("varying", I.Expr(n.type, "add", (i, I.literal(1)))),
      ]
    ),
  )
  ir = kernel([loop])
  assert loop_invariant_code_motion(ir, context("kernel-loop")).changed
  assert ir.body.statements[0].target == "width"
  assert [s.target for s in ir.body.statements[1].body.statements] == ["read", "check", "varying"]


def test_unswitch_does_not_drop_trapping_loop_bounds():
  from vecops.language._compiler.passes.kernel.motion import loop_unswitching

  flag = I.Expr(I.ScalarType("bool"), "raw", data="flag")
  branch = I.If(flag, I.Block([I.Store(view(), I.literal(1))]), I.Block())
  stop = I.Expr(I.ScalarType("int64"), "checked_shape", (I.literal(-1),))
  ir = kernel([I.For("i", I.literal(0), stop, I.literal(1), I.Block([branch]))])
  assert not loop_unswitching(ir, context("kernel-loop")).changed
  ir.body.statements[0] = I.For("i", I.literal(0), I.literal(3), I.literal(1), I.Block([branch]))
  assert loop_unswitching(ir, context("kernel-loop")).changed
  assert isinstance(ir.body.statements[0], I.If)


def test_verifier_rejects_bad_state_and_vector_dependencies():
  from vecops.language._compiler.ir import vector as V

  state = I.TileState("state", I.TileType("float32", (I.literal(4),)))
  invalid = I.Expr(I.TileType("float64", state.type.shape), "fill", (I.literal(0),))
  with pytest.raises(IRVerificationError, match="dtype"):
    verify_kernel(kernel([I.DeclareState(state, invalid)]))
  typ = I.VectorType("float32", state.type.shape)
  value = I.Expr(typ, "fill", (I.literal(0),))
  region = V.VectorRegion(state.type.shape, "float32", 1, (value,), [V.VectorOp(0, value, (0,), typ)], (0,))
  with pytest.raises(IRVerificationError, match="precede"):
    verify_vector(region)
  invalid_index = I.Expr(I.TileType("int64", (I.literal(5),)), "arange", (I.literal(0),), 1)
  with pytest.raises(IRVerificationError, match="axis"):
    verify_kernel(kernel([I.Store(view(), invalid_index)]))


def test_terminal_verifier_separates_lazy_capture_history_from_execution():
  product = I.Expr(I.TileType("int64", (I.literal(1),)), "dot")
  bound = I.BindTile(product)
  original = kernel([bound])
  assert verify_kernel(original, terminal=True) is original
  assert original.body.statements == [bound]
  # Sharing the same descriptor with an executable statement must not make it
  # legal merely because the ownership walk first encounters its lazy binding.
  with pytest.raises(IRVerificationError, match="unlowered dot"):
    verify_kernel(kernel([bound, I.Store(view(), product)]), terminal=True)
  invalid = I.Expr(I.TileType("int64", (I.literal(1),)), "arange", (I.literal(0),), 1)
  with pytest.raises(IRVerificationError, match="axis"):
    verify_kernel(kernel([I.BindTile(invalid)]), terminal=True)


def test_default_pipeline_exposes_repeated_cleanup_and_explicit_empty_is_empty():
  names = [s.name for s in kernel_pipeline().specs]
  assert names.count("expression-cse") > 1
  assert names.index("cpu-expand-layout-recipes") < names.index(
    "expression-cse", names.index("cpu-expand-layout-recipes")
  )
  assert kernel_pipeline([]).specs == ()


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "int64"])
def test_integer_range_analysis_does_not_assume_overflow_is_impossible(dtype):
  from vecops.language._compiler.analysis.facts import IntegerFacts, ValueFacts

  x = I.Expr(I.ScalarType(dtype), "raw", data="x")
  one = I.Expr(x.type, "literal", data=1)
  result = ValueFacts().infer(I.Expr(x.type, "add", (x, one)))
  assert result is None
  bounded = ValueFacts({("raw", "x"): IntegerFacts(0, 30, 2)})
  fact = bounded.infer(I.Expr(x.type, "add", (x, one)))
  if dtype.startswith("uint"):
    assert fact is None, "unsigned scalar promotion is deliberately unmodeled"
  else:
    assert (fact.lower, fact.upper, fact.divisor, fact.residue) == (1, 31, 2, 1)


@pytest.mark.parametrize("payload", ["load", "trap", "unknown", "state"])
def test_effects_include_immediate_owned_metadata_expressions(payload):
  from vecops.language._compiler.operations.registry import OperationCatalog

  definitions = dict(snapshot().definitions)
  definitions["optimizer-owned"] = OpDef("optimizer-owned", "metadata", may_trap=False, speculatable=True)
  catalog = OperationCatalog(definitions, "optimizer-owned-effects")
  if payload == "load":
    value = I.Expr(I.ScalarType("int64"), "load", data=view("A"))
  elif payload == "trap":
    value = I.Expr(I.ScalarType("int64"), "broadcast_dim", (I.literal(2), I.literal(3)))
  elif payload == "state":
    value = I.Expr(I.ScalarType("int64"), "state_ref", data="state")
  else:
    value = I.Expr(I.ScalarType("int64"), "optimizer-unknown")
  wrapped = I.Expr(I.ScalarType("int64"), "optimizer-owned", data={"metadata": (value,)})
  effects = ValueEffects(catalog)
  assert not effects.discardable(wrapped)
  assert not effects.speculatable(wrapped)
  shaped = I.Expr(I.TileType("int64", (value,)), "fill", (I.literal(0),))
  assert not effects.discardable(shaped), "type-owned evaluation must retain its effects too"


@pytest.mark.parametrize("payload", ["store", "block", "nested-block", "control", "row-map", "unknown-node", "descriptor-subclass", "state-subclass"])
def test_owned_statement_metadata_is_an_effect_barrier_for_dce(payload):
  from dataclasses import dataclass

  from vecops.language._compiler.passes.kernel.cleanup import dead_code_elimination

  @dataclass(frozen=True)
  class UnknownMetadata(I.Node):
    value: I.Expr

  @dataclass(frozen=True)
  class ExtendedView(I.View):
    body: object = None

  @dataclass(frozen=True)
  class ExtendedState(I.TileState):
    body: object = None

  store = I.Store(view(), I.literal(7))
  data = {
    "store": store,
    "block": I.Block([store]),
    "nested-block": {"program": (I.Block([store]),)},
    "control": I.Control("break"),
    "row-map": I.RowMap(I.literal(0), I.Block([store]), I.literal(7)),
    "unknown-node": UnknownMetadata(I.literal(7)),
    "descriptor-subclass": ExtendedView("C", "int64", (), (), I.literal(0), body=I.Block([store])),
    "state-subclass": ExtendedState("carry", I.TileType("int64", (I.literal(4),)), body=I.Block([store])),
  }[payload]
  definition = OpDef("optimizer-owned-statement", "metadata", may_trap=False, speculatable=True)
  register_operation(definition)
  try:
    # Internal extension metadata is legal IR; no existing builtin miscompile
    # is asserted. The pure wrapper cannot hide an owned statement's effects.
    wrapped = I.Expr(I.ScalarType("int64"), definition.name, data=data)
    ir = verify_kernel(kernel([I.Assign("unused", wrapped)]))
    ctx = context()
    effects = ValueEffects(ctx.environment.operations)
    result = dead_code_elimination(ir, ctx)
    verify_kernel(result.ir)
    assert ir.body.statements == [I.Assign("unused", wrapped)]
    assert not result.changed
    assert effects.owned_expressions(wrapped) is None
    assert not effects.discardable(wrapped)
    assert not effects.speculatable(wrapped)
  finally:
    unregister_operation(definition.name)


@pytest.mark.parametrize("descriptor", [
  "scalar", "tile", "vector", "tag", "view", "region", "packed", "dot", "tile-state",
  "domain", "schedule", "reducer", "axis-fold", "axis-result",
])
def test_owned_value_descriptors_keep_precise_effect_dependencies(descriptor):
  from vecops.language._compiler.ir.iteration import AxisFold, AxisResult, IterationDomain, LoopSchedule, ReductionSpec
  from vecops.language._compiler.operations.registry import OperationCatalog

  extent = I.literal(4)
  fold = AxisFold((extent,), 0, ("sum",), 2)
  data = {
    "scalar": I.ScalarType("int64"),
    "tile": I.TileType("int64", (extent,)),
    "vector": I.VectorType("int64", (extent,)),
    "tag": I.VectorTag(N=4),
    "view": I.View("A", "int64", (extent,), (I.literal(1),), I.literal(0)),
    "region": I.ViewRegion((extent,), (I.literal(1),), (I.literal(0),), (extent,), (1,), (0,), (False,)),
    "packed": I.PackedView(view("A"), (extent,), "a", "dense"),
    "dot": I.DotOptions(),
    "tile-state": I.TileState("carry", I.TileType("int64", (extent,))),
    "domain": IterationDomain((extent,)),
    "schedule": LoopSchedule(),
    "reducer": ReductionSpec("sum", "int64"),
    "axis-fold": fold,
    "axis-result": AxisResult(fold, 0),
  }[descriptor]
  definitions = dict(snapshot().definitions)
  definitions["optimizer-value-descriptor"] = OpDef("optimizer-value-descriptor", "metadata", may_trap=False, speculatable=True)
  catalog = OperationCatalog(definitions, "optimizer-value-descriptors")
  wrapped = I.Expr(I.ScalarType("int64"), "optimizer-value-descriptor", data=data)
  effects = ValueEffects(catalog)
  assert effects.owned_expressions(wrapped) is not None
  assert effects.discardable(wrapped)
  assert effects.speculatable(wrapped)
  # A dependency hidden by one of the allowed descriptor classes is still
  # evaluated, and cannot disappear behind its otherwise pure container.
  if descriptor in {"tile", "vector", "view", "region", "packed", "tile-state", "domain", "axis-fold", "axis-result"}:
    object.__setattr__(extent, "op", "optimizer-unknown")
    assert not ValueEffects(catalog).discardable(wrapped)
    assert not ValueEffects(catalog).speculatable(wrapped)


@pytest.mark.parametrize("shape_trap", [False, True])
def test_tile_state_descriptor_preserves_state_read_and_shape_effects(shape_trap):
  extent = I.Expr(I.ScalarType("int64"), "broadcast_dim", (I.literal(2), I.literal(3))) if shape_trap else I.literal(4)
  typ = I.TileType("int64", (extent,))
  state = I.TileState("carry", typ)
  read = I.Expr(typ, "state_ref", data=state)
  effects = ValueEffects(snapshot())
  assert effects.owned_expressions(read) is not None, "a named descriptor is not an opaque executable body"
  assert not effects.discardable(read)
  assert not effects.speculatable(read)
  if shape_trap:
    # Even a pure scalar shell must retain a shape check owned by the state
    # descriptor. Merely allowing its class does not mark all contents pure.
    shell = I.Expr(I.ScalarType("int64"), "identity", (I.literal(7),), data=state)
    assert not effects.discardable(shell)
    assert not effects.speculatable(shell)


@pytest.mark.parametrize("kind", ["data", "args", "container"])
def test_owned_metadata_and_expression_cycles_are_conservatively_unsafe(kind):
  value = I.Expr(I.ScalarType("int64"), "identity", (I.literal(1),))
  if kind == "container":
    data = []
    data.append(data)
    object.__setattr__(value, "data", data)
  else:
    object.__setattr__(value, kind, (value,) if kind == "args" else value)
  effects = ValueEffects(snapshot())
  assert not effects.discardable(value)
  assert not effects.speculatable(value)


def test_owned_metadata_diamonds_are_memoized_without_transitive_expression_scans(monkeypatch):
  from vecops.language._compiler.operations.registry import OperationCatalog

  definitions = dict(snapshot().definitions)
  definitions["optimizer-owned"] = OpDef("optimizer-owned", "metadata", may_trap=False, speculatable=True)
  catalog = OperationCatalog(definitions, "optimizer-owned-diamond")
  value = I.literal(7)
  metadata = (value,)
  for _ in range(24):
    metadata = (metadata, metadata)
  for _ in range(24):
    value = I.Expr(value.type, "optimizer-owned", data=(value, value, metadata))
  visits = 0
  original = ValueEffects._metadata_edges

  def counted(self, subject):
    nonlocal visits
    visits += 1
    assert visits < 1000, "shared ownership DAG was expanded as a tree"
    return original(self, subject)

  monkeypatch.setattr(ValueEffects, "_metadata_edges", counted)
  effects = ValueEffects(catalog)
  assert effects.discardable(value) and effects.speculatable(value)
  assert visits < 300


@pytest.mark.parametrize("divisor", [128, 256, 511])
def test_narrow_typed_division_cannot_claim_positive_divisor_after_narrowing(divisor):
  expression = I.Expr(I.ScalarType("int8"), "floordiv", (I.literal(-128, "int8"), I.literal(divisor)))
  assert not ValueEffects(snapshot()).speculatable(expression)


def test_validated_dynamic_extents_share_range_and_alignment_facts():
  from vecops.language._compiler.analysis.facts import ValueFacts
  from vecops.language._compiler.frontend.parser import lower
  from vecops.language._compiler.passes.kernel.facts import propagate_value_facts

  N = T.dynamic(lower=0, upper=4096, alignment=16)

  @T.template
  def example(A: T.Tensor((N,), dtype="float32"), C: T.Tensor((1,), dtype="int64", access="output")):
    T.store(C[0], N % 16)

  ir = lower(example.ir)
  facts = ValueFacts.from_kernel(ir)
  value = next(s.value for s in ir.body.statements if isinstance(s, I.Store))
  assert facts.infer(value).constant == 0
  assert propagate_value_facts(ir, context()).changed
  assert I.constant(ir.body.statements[-1].value) == 0
  # The same algebraic conclusion cannot discard an actual memory read.
  read = I.Expr(I.ScalarType("int64"), "load", data=view("A"))
  expression = I.Expr(read.type, "mod", (read, I.literal(1)))
  ir = kernel([I.Store(view(), expression)])
  assert not propagate_value_facts(ir, context()).changed
  assert ir.body.statements[0].value is expression


def test_expression_keys_preserve_float_payloads_and_shared_shape_dags():
  import struct

  from vecops.language._compiler.analysis.keys import ExpressionKeys

  keys = ExpressionKeys()
  nan_a, nan_b = (struct.unpack("!d", bytes.fromhex(bits))[0] for bits in ("7ff8000000000001", "7ff8000000000002"))
  assert keys.data(nan_a) != keys.data(nan_b)
  assert keys.data(0.0) != keys.data(-0.0)
  assert keys.data(True) != keys.data(1)
  # Sixty shared diamonds represent 2**60 paths but only 61 owned nodes.
  size = I.literal(1)
  for _ in range(60):
    size = I.Expr(size.type, "add", (size, size))
  keys.type(I.TileType("float32", (size,)))
  assert len(keys.metadata) == 61


def test_integer_motion_does_not_introduce_overflow_on_an_empty_path():
  from vecops.language._compiler.passes.kernel.motion import loop_invariant_code_motion

  n = I.Expr(I.ScalarType("int64"), "raw", data="n")
  overflowing = I.Expr(n.type, "mul", (n, I.literal(2)))
  loop = I.For("i", I.literal(0), I.literal(0), I.literal(1), I.Block([I.Assign("value", overflowing)]))
  ir = kernel([loop])
  assert not loop_invariant_code_motion(ir, context("kernel-loop")).changed
  assert ir.body.statements == [loop]
  assert not ValueEffects(context().environment.operations).speculatable(I.Expr(n.type, "neg", (n,)))


def test_licm_keeps_per_iteration_scalar_reinitialization():
  from vecops.language._compiler.passes.kernel.motion import loop_invariant_code_motion

  x = I.Expr(I.ScalarType("int64"), "local", data="x")
  loop = I.For(
    "i",
    I.literal(0),
    I.literal(3),
    I.literal(1),
    I.Block(
      [
        I.Assign("x", I.literal(2)),
        I.Assign("x", I.binary("add", x, I.literal(1)), declare=False),
        I.Store(view(), x),
      ]
    ),
  )
  ir = kernel([loop])
  assert not loop_invariant_code_motion(ir, context("kernel-loop")).changed
  assert ir.body.statements == [loop]
  assert len(loop.body.statements) == 3


def test_native_licm_reinitializes_mutated_loop_local_on_each_iteration(tmp_path):
  import numpy as np
  from dsl_test_utils import make_compiler

  @T.template
  def example(C: T.Tensor((3,), dtype="int64", access="output"), seed: int):
    for row in T.serial(3, unroll=1):
      value = seed
      value = value + 1
      C[row] = value

  compiled = T.compile(example, compiler=make_compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  output = np.empty(3, dtype="int64")
  for seed in (2, -3):
    compiled(output, seed)
    np.testing.assert_array_equal(output, seed + 1)


def test_default_cleanup_accepts_extension_passes_and_freezes_their_callbacks():
  P = T.passes

  @P.register_pass("optimizer-test-cleanup-plugin", stage="kernel-cleanup", after="simplify-ir")
  def callback(value, ctx):
    value.diagnostics.append("cleanup plugin")
    return P.PassResult(value, changed=False)

  try:
    manager = kernel_pipeline()
    assert "optimizer-test-cleanup-plugin" in [spec.name for spec in manager.specs]
    P.unregister_pass("optimizer-test-cleanup-plugin")
    ir = manager.run(kernel([]))
    assert "cleanup plugin" in ir.diagnostics
  finally:
    if any(p.name == "optimizer-test-cleanup-plugin" for p in P.registered_passes()):
      P.unregister_pass("optimizer-test-cleanup-plugin")


@pytest.mark.parametrize("historical", [False, True])
def test_native_scalar_broadcast_projection_keeps_dynamic_singleton_and_capture(tmp_path, historical):
  import numpy as np
  from dsl_test_utils import make_compiler

  rows, stride0, stride1 = (T.dynamic(lower=0), T.dynamic(), T.dynamic())

  @T.template
  def example(
    A: T.Tensor((rows, 1), (stride0, stride1), "float64", access="inout"),
    C: T.Tensor((17,), dtype="float32", access="output"),
  ):
    initial = T.broadcast_to(T.cast(T.load(A), "float32"), (17, 1))
    if historical:
      T.store(A, T.full(A.shape, 99, dtype="float64"))
    for row in T.serial(17, unroll=1):
      T.store(C[row], T.element_at(initial, (row, 0)))

  compiled = T.compile(example, compiler=make_compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  for count in (1, 17):
    a = np.linspace(-2, 2, count, dtype="float64").reshape(count, 1)[::-1]
    expected = np.broadcast_to(a.astype("float32"), (17, 1))[:, 0].copy()
    output = np.empty(17, dtype="float32")
    compiled(a, output)
    np.testing.assert_array_equal(output, expected)
    if historical:
      np.testing.assert_array_equal(a, 99)


def test_cse_does_not_equate_reads_across_mutable_scalar_versions():
  from vecops.language._compiler.passes.kernel.canonicalize import expression_cse

  typ = I.ScalarType("int64")
  first = I.Expr(typ, "add", (I.Expr(typ, "local", data="x"), I.literal(1)))
  second = I.Expr(typ, "add", (I.Expr(typ, "local", data="x"), I.literal(1)))
  ir = kernel(
    [
      I.Assign("x", I.literal(1)),
      I.Store(view("A"), first),
      I.Assign("x", I.literal(2), declare=False),
      I.Store(view("B"), second),
    ]
  )
  expression_cse(ir, context())
  assert ir.body.statements[1].value is not ir.body.statements[3].value


def test_effect_proofs_survive_rewriting_and_object_collection():
  import gc
  import weakref

  from vecops.language._compiler.analysis.effects import ValueEffects

  effects = ValueEffects(context().environment.operations)
  harmless = I.literal(1)
  reference = weakref.ref(harmless)
  assert effects.discardable(harmless)
  assert effects.speculatable(harmless)
  del harmless
  gc.collect()
  # This must be deterministic even on runtimes that delay ID reuse. An
  # identity-keyed proof owns its subject for the whole analysis lifetime.
  assert reference() is not None
  for ordinal in range(64):
    unknown = I.Expr(I.ScalarType("int64"), "unknown-effect-probe", data=ordinal)
    assert not effects.discardable(unknown)
    assert not effects.speculatable(unknown)
  del effects
  gc.collect()
  assert reference() is None


@pytest.mark.parametrize("same_capture", [False, True])
def test_cleanup_identity_requires_distinct_read_effects_to_be_discardable(same_capture):
  from vecops.language._compiler.passes.kernel.cleanup import simplify_ir

  source = view("A")
  original = I.Expr(I.ScalarType("int64"), "load", data=source)
  current = original if same_capture else I.Expr(original.type, "load", data=source)
  value = I.Expr(original.type, "min", (original, current))
  ir = kernel([I.BindTile(original), I.Store(source, I.literal(0)), I.Store(view("C"), value)])
  simplify_ir(ir, context())
  result = ir.body.statements[-1].value
  if same_capture:
    assert result is original
  else:
    assert result.op == "min" and result.args == (original, current)


@pytest.mark.parametrize("barrier", ["none", "read", "capture", "partial", "public", "trap"])
def test_dead_private_store_requires_exact_overwrite_and_no_observer(barrier):
  from dataclasses import replace

  from vecops.language._compiler.passes.kernel.cleanup import dead_store_elimination

  owned = I.View("temporary", "int64", (I.literal(4),), (I.literal(1),), I.literal(0), owned=True)
  zero = I.Expr(I.TileType("int64", owned.shape), "fill", (I.literal(0),))
  one = I.Expr(zero.type, "fill", (I.literal(1),))
  initial = I.Store(owned, zero)
  final = I.Store(owned, one)
  middle = []
  if barrier == "read":
    final = replace(final, value=I.Expr(zero.type, "load", data=owned))
  elif barrier == "capture":
    middle = [I.BindTile(I.Expr(zero.type, "load", data=owned))]
  elif barrier == "partial":
    final = replace(final, destination=replace(owned, shape=(I.literal(3),)))
  elif barrier == "public":
    owned = replace(owned, owned=False)
    initial, final = replace(initial, destination=owned), replace(final, destination=owned)
  elif barrier == "trap":
    initial = replace(initial, value=I.Expr(zero.type, "unknown-op"))
  ir = kernel([I.Allocate(owned), initial, *middle, final])
  assert dead_store_elimination(ir, context()).changed == (barrier == "none")
  assert (initial in ir.body.statements) == (barrier != "none")


@pytest.mark.parametrize("mode,shape", [("row", (5,)), ("column", (3,)), ("transpose", (5, 3))])
def test_native_cleanup_preserves_index_coordinate_maps(tmp_path, mode, shape):
  import numpy as np
  from dsl_test_utils import make_compiler

  @T.template
  def coordinates(C: T.Tensor(shape, dtype="int64", access="output")):
    indices = T.indices((3, 5), axis=1)
    if mode == "row":
      T.store(C, indices[1, :])
    elif mode == "column":
      T.store(C, indices[:, 2])
    else:
      T.store(C, T.transpose(indices))

  compiled = T.compile(coordinates, compiler=make_compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  output = np.empty(shape, dtype=np.int64)
  compiled(output)
  full = np.broadcast_to(np.arange(5, dtype=np.int64), (3, 5))
  expected = full[1, :] if mode == "row" else full[:, 2] if mode == "column" else full.T
  np.testing.assert_array_equal(output, expected)
