# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Lexical integer range proofs, independent of any operator recipe."""

import pytest
from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.affine import branch_bounds, canonical_integer, interval, simplify_loop_metadata
from vecops.language._debug.interpreter import Interpreter


def local(name):
  return I.Expr(I.ScalarType("int64"), "local", data=name)


@pytest.mark.parametrize("operation", ("lt", "le", "gt", "ge", "eq", "ne"))
@pytest.mark.parametrize("coefficient", (-1, 1))
@pytest.mark.parametrize("taken", (False, True))
def test_branch_interval_contains_every_reachable_integer(operation, coefficient, taken):
  interpreter = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
  index, extent = local("i"), local("extent")
  condition = I.binary(operation, I.binary("mul", I.literal(coefficient), index), extent)
  refined = branch_bounds(condition, {"i": (I.literal(-10), I.literal(10))}, taken=taken)
  for size in range(-12, 13):
    interpreter.environment["extent"] = size
    lower, upper = (interpreter.value(v) for v in refined["i"])
    for value in range(-10, 11):
      interpreter.environment["i"] = value
      if bool(interpreter.value(condition)) == taken:
        assert lower <= value <= upper


def test_complete_tile_branch_removes_only_its_proven_clamps():
  origin, row, extent = local("origin"), local("row"), local("extent")
  start = I.binary("add", I.binary("add", origin, I.literal(16)), row)
  clamped = I.binary("max", I.literal(0), I.binary("min", start, extent))
  width = I.binary("max", I.literal(0), I.binary("min", I.literal(16), I.binary("sub", extent, origin)))
  condition = I.binary("ge", I.binary("sub", extent, origin), I.literal(32))
  body = I.Block([I.For("row", I.literal(0), I.literal(16), I.literal(1), I.Block([I.Assign("address", clamped)]))])
  program = I.Block(
    [
      I.For(
        "origin", I.literal(0), extent, I.literal(32), I.Block([I.If(condition, body, body), I.Assign("tail", width)])
      )
    ]
  )
  optimized = simplify_loop_metadata(program)
  branch, tail = optimized.statements[0].body.statements
  assert I.same_expr(branch.body.statements[0].body.statements[0].value, canonical_integer(start))
  assert not I.same_expr(branch.otherwise.statements[0].body.statements[0].value, canonical_integer(start))
  assert not I.same_expr(tail.value, I.literal(16)), "full-panel facts must not leak after the branch"


def test_inner_index_cannot_become_an_outer_index_bound():
  outer, inner = local("outer"), local("inner")
  bounds = {"outer": (I.literal(0), I.literal(20)), "inner": (outer, I.literal(30))}
  refined = branch_bounds(I.binary("lt", outer, inner), bounds)
  assert refined["outer"] == bounds["outer"]
  assert I.same_expr(refined["inner"][0], canonical_integer(I.binary("add", outer, I.literal(1))))


def test_disjunction_and_floating_comparison_do_not_invent_integer_facts():
  index = local("i")
  bounds = {"i": (I.literal(0), I.literal(20))}
  left, right = I.binary("lt", index, I.literal(4)), I.binary("gt", index, I.literal(16))
  assert branch_bounds(I.binary("or", left, right), bounds) is bounds
  floating = I.Expr(I.ScalarType("float32"), "local", data="x")
  assert branch_bounds(I.binary("lt", floating, I.literal(4.0)), bounds) is bounds


@pytest.mark.parametrize("operation", ("eq", "ne"))
def test_exact_clipped_extent_facts_are_branch_local(operation):
  extent = I.binary("max", I.literal(0), I.binary("min", I.literal(16), local("length")))
  condition = I.binary(operation, extent, I.literal(16))
  payload = I.Block([I.Assign("active", extent)])
  optimized = simplify_loop_metadata(I.Block([I.If(condition, payload, payload), I.Assign("after", extent)]))
  branch, after = optimized.statements
  selected, other = (branch.body, branch.otherwise) if operation == "eq" else (branch.otherwise, branch.body)
  assert I.constant(selected.statements[0].value) == 16
  assert I.constant(other.statements[0].value) is None
  assert I.constant(after.value) is None


def test_exact_equality_does_not_fold_floating_or_disjunctive_payloads():
  value = I.Expr(I.ScalarType("float32"), "local", data="value")
  condition = I.binary("eq", value, I.literal(1.0))
  result = simplify_loop_metadata(I.Block([I.If(condition, I.Block([I.Assign("copy", value)]), I.Block([]))]))
  assert result.statements[0].body.statements[0].value.op == "local"
  width = I.binary("min", local("n"), I.literal(16))
  condition = I.binary("or", I.binary("eq", width, I.literal(16)), I.binary("eq", local("flag"), I.literal(1)))
  result = simplify_loop_metadata(I.Block([I.If(condition, I.Block([I.Assign("copy", width)]), I.Block([]))]))
  assert I.constant(result.statements[0].body.statements[0].value) is None


def test_exact_equality_does_not_substitute_an_opaque_integer_read():
  read = I.Expr(I.ScalarType("int64"), "unknown_read")
  condition = I.binary("eq", read, I.literal(16))
  result = simplify_loop_metadata(I.Block([I.If(condition, I.Block([I.Assign("copy", read)]), I.Block([]))]))
  assert result.statements[0].body.statements[0].value.op == "unknown_read"


def test_branch_and_loop_scopes_do_not_capture_outer_literal_bindings():
  program = I.Block(
    [
      I.Assign("i", I.literal(7)),
      I.For("i", I.literal(0), I.literal(3), I.literal(1), I.Block([I.Assign("inside", local("i"))])),
      I.If(I.literal(True), I.Block([I.Assign("only_here", I.literal(5))]), I.Block([])),
      I.Assign("after", local("only_here")),
      I.Assign("outer", local("i")),
    ]
  )
  optimized = simplify_loop_metadata(program)
  assert optimized.statements[1].body.statements[0].value.op == "local"
  assert optimized.statements[3].value.op == "local"
  assert I.constant(optimized.statements[4].value) == 7


@pytest.mark.parametrize("svl", (16, 32, 128, 256))
def test_only_captured_streaming_geometry_propagates_through_literal_ssa(svl):
  lanes = I.Expr(I.ScalarType("int64"), "streaming_lanes", data="float32")
  runtime = local("runtime")
  program = I.Block([I.Assign("width", lanes), I.Assign("query", runtime), I.Assign("end", local("width"))])
  original = simplify_loop_metadata(program)
  assert original.statements[0].value.op == "streaming_lanes"
  assert original.statements[2].value.op == "local"
  optimized = simplify_loop_metadata(program, streaming_bytes=svl)
  assert I.constant(optimized.statements[0].value) == svl // 4
  assert I.constant(optimized.statements[2].value) == svl // 4
  assert optimized.statements[1].value is runtime or I.same_expr(optimized.statements[1].value, runtime)


def test_geometry_specialization_folds_trip_counts_and_branch_comparisons():
  width = I.Expr(I.ScalarType("int64"), "streaming_lanes", data="float32")
  trips = I.binary("floordiv", I.literal(256), local("width"))
  condition = I.binary("ge", local("trips"), I.literal(4))
  program = I.Block([I.Assign("width", width), I.Assign("trips", trips), I.If(condition, I.Block([]), I.Block([]))])
  optimized = simplify_loop_metadata(program, streaming_bytes=128)
  assert I.constant(optimized.statements[1].value) == 8
  assert I.constant(optimized.statements[2].condition) is True
  # Runtime, floating and potentially undefined arithmetic is not evaluated.
  zero_division = I.Expr(I.ScalarType("int64"), "floordiv", (I.literal(1), I.literal(0)))
  assert canonical_integer(zero_division).op == "floordiv"


def test_pure_stride_products_cancel_without_merging_opaque_reads():
  first = I.binary("mul", local("row"), local("pitch"))
  second = I.binary("mul", local("row"), local("pitch"))
  value = I.binary("sub", I.binary("add", first, I.literal(32)), second)
  assert I.constant(canonical_integer(value)) == 32
  read = lambda: I.Expr(I.ScalarType("int64"), "unknown_read", (local("row"),))
  opaque = canonical_integer(I.binary("sub", read(), read()))
  assert I.constant(opaque) is None


def test_runtime_shape_guards_and_full_chunk_floor_relation():
  size = I.Expr(I.ScalarType("int64"), "raw", data="shape.k")
  index = local("chunk")
  guard = I.binary("ge", size, I.literal(32))
  bounds = branch_bounds(guard, {})
  assert I.constant(bounds[("raw", "shape.k")][0]) == 32
  upper = I.binary("sub", I.binary("floordiv", size, I.literal(32)), I.literal(1))
  bounds["chunk"] = I.literal(1), upper
  # The final *complete* chunk ends at floor(K/32)*32 <= K, even with
  # runtime K. No rounding up or assertion from the user is needed.
  end = I.binary("sub", I.binary("mul", I.binary("add", index, I.literal(1)), I.literal(32)), size)
  assert I.constant(interval(end, bounds)[1]) == 0
  from vecops.language._compiler.analysis.affine import simplify_ordered

  clipped = I.binary("min", I.binary("mul", I.binary("add", index, I.literal(1)), I.literal(32)), size)
  assert I.same_expr(simplify_ordered(clipped, bounds), clipped.args[0])
  interpreter = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
  remainder = canonical_integer(
    I.binary("sub", I.binary("mul", I.binary("floordiv", size, I.literal(32)), I.literal(32)), size)
  )
  for value in (-97, -32, -1, 0, 1, 32, 65, 1000):
    interpreter.environment["shape.k"] = value
    assert interpreter.value(remainder) == -(value % 32)


def test_new_abi_metadata_remains_outer_to_existing_induction_bounds():
  size = I.Expr(I.ScalarType("int64"), "raw", data="shape.m")
  index = local("i")
  bounds = {"i": (I.literal(0), I.binary("sub", size, I.literal(1)))}
  refined = branch_bounds(I.binary("ge", I.binary("sub", size, index), I.literal(32)), bounds)
  assert tuple(refined) == (("raw", "shape.m"), "i")
  assert I.same_expr(refined["i"][1], canonical_integer(I.binary("sub", size, I.literal(32))))
  assert I.constant(refined[("raw", "shape.m")][0]) == -(1 << 63)


@pytest.mark.parametrize("svl", (0, 8, 24, 512, True))
def test_invalid_captured_streaming_length_is_rejected(svl):
  with pytest.raises(ValueError, match="streaming length"):
    simplify_loop_metadata(I.Block([]), streaming_bytes=svl)


def test_overflow_guards_do_not_create_unrepresentable_metadata_literals():
  from vecops.language._compiler.ir.traversal import walk_unique

  maximum = (1 << 63) - 1
  interpreter = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
  width = I.Expr(I.ScalarType("int64"), "raw", data="streaming.width")
  for factor in (1, 2):
    padding = I.binary("sub", I.binary("mul", I.literal(factor), width), I.literal(1))
    guarded_limit = canonical_integer(I.binary("sub", I.literal(maximum), padding))
    assert all(
      -(1 << 63) <= value.data <= maximum
      for value in walk_unique(guarded_limit)
      if value.op == "literal" and type(value.data) is int
    )
    for lanes in (8, 16, 32, 64, 256):
      interpreter.environment["streaming.width"] = lanes
      assert interpreter.value(guarded_limit) == maximum - (factor * lanes - 1)


def test_dynamic_interior_nested_slice_and_dense_reshape_keep_exact_coordinates():
  """Dynamic full panels simplify, while residual and empty domains stay exact.

  A dense reshape changes the logical axes, not the physical base offset.
  Model the checked reshape's fresh region instead of claiming original-root
  grid ownership; captured reads must keep their identity across both paths.
  """
  size = I.Expr(I.ScalarType("int64"), "raw", data="shape.m")
  origin, row = local("origin"), local("row")
  tile_start = I.binary("max", I.literal(0), I.binary("min", origin, size))
  tile_end = I.binary("max", tile_start, I.binary("min", I.binary("add", origin, I.literal(32)), size))
  tile_width = I.binary("sub", tile_end, tile_start)
  # Slice the second half of an already clipped panel and reshape its dense
  # extent into (extent, 1); the absolute origin remains the outer offset.
  nested_start = I.binary("max", I.literal(0), I.binary("min", I.literal(16), tile_width))
  nested_width = I.binary("sub", tile_width, nested_start)
  absolute = I.binary("add", tile_start, nested_start)
  reshaped = I.View(
    "A",
    "float32",
    (nested_width, I.literal(1)),
    (I.literal(1), I.literal(1)),
    absolute,
    region=I.ViewRegion(
      (nested_width, I.literal(1)),
      (I.literal(1), I.literal(1)),
      (I.literal(0), I.literal(0)),
      (nested_width, I.literal(1)),
      (1, 1),
      (0, 1),
      (False, False),
      affine_partition=False,
    ),
  )
  row_address = I.binary("add", absolute, row)
  read = I.Expr(I.TileType("float32", reshaped.shape), "load", data=reshaped)
  payload = I.Block(
    [
      I.BindTile(read),
      I.Store(reshaped, read),
      I.For("row", I.literal(0), nested_width, I.literal(1), I.Block([I.Assign("address", row_address)])),
    ]
  )
  condition = I.binary("ge", I.binary("sub", size, origin), I.literal(32))
  program = I.Block([I.For("origin", I.literal(0), size, I.literal(32), I.Block([I.If(condition, payload, payload)]))])
  optimized = simplify_loop_metadata(program)
  branch = optimized.statements[0].body.statements[0]
  full_store, tail_store = branch.body.statements[1], branch.otherwise.statements[1]
  assert I.constant(full_store.destination.shape[0]) == 16
  assert I.same_expr(full_store.destination.offset, canonical_integer(I.binary("add", origin, I.literal(16))))
  assert I.constant(tail_store.destination.shape[0]) is None
  assert full_store.destination.region.affine_partition is False
  assert tail_store.destination.region.affine_partition is False
  assert branch.body.statements[0].value is read
  assert full_store.value is tail_store.value is read, "metadata proofs must not clone or retime captured reads"
  interpreter = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
  # Empty/negative loop domains must not acquire new row iterations.
  for extent in (-5, 0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 64, 65, 97):
    interpreter.environment["shape.m"] = extent
    for base in range(0, extent, 32):
      interpreter.environment["origin"] = base
      selected = full_store if interpreter.value(condition) else tail_store
      assert interpreter.value(selected.destination.offset) == interpreter.value(reshaped.offset)
      assert interpreter.value(selected.destination.shape[0]) == interpreter.value(nested_width)
      for index in range(interpreter.value(nested_width)):
        interpreter.environment["row"] = index
        selected_body = branch.body if interpreter.value(condition) else branch.otherwise
        rewritten = selected_body.statements[2].body.statements[0].value
        assert interpreter.value(rewritten) == interpreter.value(row_address)


def test_affine_membership_visits_shared_dag_once_per_definition_scope(monkeypatch):
  from vecops.language._compiler.analysis import affine as module

  leaf = local("unrelated")
  value = leaf
  depth = 24
  for _ in range(depth):
    # Non-affine metadata commonly shares both arms after helper expansion.
    value = I.Expr(I.ScalarType("int64"), "min", (value, value))
  visits = []
  original = module.metadata_reference

  def counted(expression):
    visits.append(id(expression))
    return original(expression)

  monkeypatch.setattr(module, "metadata_reference", counted)
  assert not module.affine_contains(value, {"index"}, {})
  assert len(visits) == depth + 1
  # A later independent query must not reuse the first query's false result.
  assert module.affine_contains(value, {"unrelated"}, {})


def test_affine_membership_preserves_definition_cycles_and_owned_metadata():
  from vecops.language._compiler.analysis.affine import affine_contains

  a, b, target = local("a"), local("b"), local("index")
  definitions = {"a": b, "b": a}
  assert not affine_contains(a, {"index"}, definitions)
  definitions["b"] = I.Expr(I.ScalarType("int64"), "min", (a, target))
  assert affine_contains(a, {"index"}, definitions)
  assert not affine_contains(a, {"index"}, definitions, seen=frozenset({"a"}))
  view = I.View("A", "float32", (target,), (I.literal(1),), I.literal(0))
  metadata = I.Expr(I.ScalarType("int64"), "metadata_wrapper", data=view)
  assert affine_contains(metadata, {"index"}, {})


def test_affine_membership_visits_ssa_definition_diamond_once(monkeypatch):
  from vecops.language._compiler.analysis import affine as module

  value, definitions, depth = local("unrelated"), {}, 24
  for level in range(depth):
    definitions[f"a{level}"] = definitions[f"b{level}"] = value
    value = I.Expr(I.ScalarType("int64"), "min", (local(f"a{level}"), local(f"b{level}")))
  visits = []
  original = module.metadata_reference

  def counted(expression):
    visits.append(id(expression))
    return original(expression)

  monkeypatch.setattr(module, "metadata_reference", counted)
  assert not module.affine_contains(value, {"index"}, definitions)
  assert len(visits) == 3 * depth + 1
  assert len(set(visits)) == len(visits)


def test_affine_membership_handles_owned_data_cycles_and_seeded_exclusions():
  from vecops.language._compiler.analysis.affine import affine_contains

  cycle = I.Expr(I.ScalarType("int64"), "metadata_wrapper")
  object.__setattr__(cycle, "data", cycle)
  assert not affine_contains(cycle, {"index"}, {})
  object.__setattr__(cycle, "args", (local("index"),))
  assert affine_contains(cycle, {"index"}, {})
  a, b = local("a"), local("b")
  definitions = {"a": I.Expr(I.ScalarType("int64"), "min", (b, local("index"))), "b": a}
  assert affine_contains(b, {"index"}, definitions)
  assert not affine_contains(b, {"index"}, definitions, seen=frozenset({"a"}))
  assert affine_contains(a, {"a"}, definitions, seen=frozenset({"a"}))


def test_affine_membership_matches_recursive_oracle_on_small_definition_graphs():
  import random

  from vecops.language._compiler.analysis.affine import affine_contains, metadata_reference
  from vecops.language._compiler.ir.traversal import nodes

  def oracle(value, variables, definitions, seen):
    reference = metadata_reference(value)
    if reference is not None and reference in variables or value.op in {"local", "frozen"} and value.data in variables:
      return True
    if value.op in {"local", "frozen"} and value.data in definitions and value.data not in seen:
      return oracle(definitions[value.data], variables, definitions, seen | {value.data})
    return any(oracle(a, variables, definitions, seen) for a in value.args) or any(
      isinstance(a, I.Expr) and oracle(a, variables, definitions, seen) for a in nodes(value.data)
    )

  rng, names = random.Random(317), ("a", "b", "c", "index", "other")
  for _ in range(300):
    pool = [local(name) for name in names] + [I.literal(1)]
    pool.append(I.Expr(I.ScalarType("int64"), "local", (local("index"),), data="a"))
    for index in range(5):
      if rng.randrange(2):
        pool.append(I.Expr(I.ScalarType("int64"), "min", (rng.choice(pool), rng.choice(pool))))
      else:
        view = I.View(f"A{index}", "float32", (rng.choice(pool),), (I.literal(1),), I.literal(0))
        pool.append(I.Expr(I.ScalarType("int64"), "metadata_wrapper", data=view))
    definitions = {name: rng.choice(pool) for name in names[:3]}
    variables = set(rng.sample(names, rng.randrange(3)))
    seen = frozenset(rng.sample(names[:3], rng.randrange(4)))
    value = rng.choice(pool)
    assert affine_contains(value, variables, definitions, seen) == oracle(value, variables, definitions, seen)


def test_canonical_memo_retains_temporary_inputs_and_is_query_catalog_local():
  import gc
  import weakref

  from vecops.language._compiler.operations import OpDef, snapshot, use_catalog
  from vecops.language._compiler.operations.registry import OperationCatalog

  value = I.Expr(I.ScalarType("int64"), "add", (local("i"), I.literal(5)))
  reference, memo = weakref.ref(value), {}
  canonical_integer(value, _memo=memo)
  del value
  gc.collect()
  assert reference() is not None  # id-based keys cannot outlive their inputs
  memo.clear()
  gc.collect()
  assert reference() is None

  source = I.Expr(I.ScalarType("int64"), "memo_probe", (I.literal(1), I.literal(2)))
  base = snapshot()
  for result in (41, -7):
    definitions = dict(base.definitions)
    definitions["memo_probe"] = OpDef("memo_probe", "metadata", fold=lambda a, b, result=result: result)
    catalog = OperationCatalog(definitions, f"probe-{result}")
    with use_catalog(catalog):
      assert tuple(I.constant(v) for v in interval(source, {})) == (result, result)


def test_query_cache_distinguishes_bounds_order_endpoints_and_ordering():
  from vecops.language._compiler.analysis import affine as analysis

  value = local("i")
  lo, hi, other = I.literal(0), I.literal(7), I.literal(19)
  calls = []

  @analysis._query_cached("interval")
  def probe(value, bounds, *, _ordered=True):
    calls.append((tuple(bounds), tuple(bounds.values()), _ordered))
    return object()

  @analysis._query_cached("ordered")
  def query(value, bounds):
    first = probe(value, bounds)
    assert probe(value, dict(bounds)) is first
    assert probe(value, dict(reversed(tuple(bounds.items())))) is not first
    assert probe(value, bounds, _ordered=False) is not first
    bounds["i"] = (lo, other)
    assert probe(value, bounds) is not first
    return first

  query(value, {"i": (lo, hi), "j": (lo, other)})
  assert len(calls) == 4
  assert analysis._metadata_query.get() is None
  probe(value, {"i": (lo, hi), "j": (lo, other)})
  assert len(calls) == 5, "no metadata facts may survive a top-level query"


def test_query_cache_releases_state_after_exception():
  from vecops.language._compiler.analysis import affine as analysis

  @analysis._query_cached("ordered")
  def fail(value, bounds):
    analysis.interval(value, bounds)
    raise RuntimeError("proof interrupted")

  with pytest.raises(RuntimeError, match="proof interrupted"):
    fail(local("i"), {"i": (I.literal(0), I.literal(3))})
  assert analysis._metadata_query.get() is None
  assert tuple(I.constant(x) for x in interval(local("i"), {"i": (I.literal(11), I.literal(14))})) == (11, 14)


def test_query_cache_isolated_between_concurrent_lowerings():
  from concurrent.futures import ThreadPoolExecutor
  from threading import Barrier

  from vecops.language._compiler.analysis import affine as analysis

  barrier = Barrier(2)
  value = local("i")

  @analysis._query_cached("ordered")
  def query(value, bounds):
    state = analysis._metadata_query.get()
    result = analysis.interval(value, bounds)
    barrier.wait(timeout=10)
    assert state is analysis._metadata_query.get()
    return state, tuple(I.constant(x) for x in result)

  def run(start):
    result = query(value, {"i": (I.literal(start), I.literal(start + 3))})
    assert analysis._metadata_query.get() is None
    return result

  with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(run, (0, 100)))
  assert results[0][0] is not results[1][0]
  assert [r[1] for r in results] == [(0, 3), (100, 103)]


def test_query_cache_restores_outer_catalog_after_nested_catalog():
  from vecops.language._compiler.analysis import affine as analysis
  from vecops.language._compiler.operations import OpDef, snapshot, use_catalog
  from vecops.language._compiler.operations.registry import OperationCatalog

  base = snapshot()
  catalogs = []
  for result in (41, -7):
    definitions = dict(base.definitions)
    definitions["nested_memo_probe"] = OpDef("nested_memo_probe", "metadata", fold=lambda a, b, result=result: result)
    catalogs.append(OperationCatalog(definitions, f"nested-probe-{result}"))
  source = I.Expr(I.ScalarType("int64"), "nested_memo_probe", (I.literal(1), I.literal(2)))

  @analysis._query_cached("ordered")
  def query(value, bounds):
    outer_state = analysis._metadata_query.get()
    first = analysis.canonical_integer(source)
    assert I.constant(first) == 41
    assert tuple(I.constant(v) for v in analysis.interval(source, bounds)) == (41, 41)
    with use_catalog(catalogs[1]):
      # Direct canonical calls must not use the outer catalog's shared memo.
      assert I.constant(analysis.canonical_integer(source)) == -7
      assert tuple(I.constant(v) for v in analysis.interval(source, bounds)) == (-7, -7)
      assert I.constant(analysis.canonical_integer(source)) == -7
    assert analysis._metadata_query.get() is outer_state
    assert analysis.canonical_integer(source) is first
    assert tuple(I.constant(v) for v in analysis.interval(source, bounds)) == (41, 41)
    return first

  with use_catalog(catalogs[0]):
    assert I.constant(query(I.literal(0), {})) == 41
  assert analysis._metadata_query.get() is None


def test_query_canonical_cache_preserves_nonidempotent_traversals():
  from vecops.language._compiler.analysis import affine as analysis

  i, k = local("i"), local("k")

  def sub(a, b):
    return I.binary("sub", a, b)

  def maximum(a, b):
    return I.binary("max", a, b)

  shifted = sub(i, I.literal(-1))
  clamp = maximum(maximum(k, shifted), maximum(I.literal(-2), k))
  source = sub(sub(I.literal(2), clamp), maximum(i, sub(shifted, maximum(k, shifted))))
  # This real clamp expression is not a fixed point after the first traversal.
  first = analysis.canonical_integer(source)
  second = analysis.canonical_integer(first)
  assert not I.same_expr(first, second)
  interpreter = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
  for index in range(-5, 6):
    for extent in range(-5, 6):
      interpreter.environment.update(i=index, k=extent)
      assert interpreter.value(source) == interpreter.value(first) == interpreter.value(second)

  @analysis._query_cached("ordered")
  def query(value, bounds):
    actual_first = analysis.canonical_integer(source)
    assert I.same_expr(actual_first, first)
    assert analysis.canonical_integer(source) is actual_first
    actual_second = analysis.canonical_integer(actual_first)
    assert I.same_expr(actual_second, second)
    assert not I.same_expr(actual_second, actual_first)
    assert analysis.canonical_integer(actual_first) is actual_second
    # Cross-traversal cache records only true top-level inputs, not all
    # synthesized recursive results from those canonical traversals.
    shared = analysis._metadata_query.get()["canonical"]
    assert set(shared) == {id(source), id(actual_first)}
    return actual_second

  assert I.same_expr(query(I.literal(0), {}), second)
  assert analysis._metadata_query.get() is None



def test_integer_term_keys_bound_shared_nonlinear_dag_cost(monkeypatch):
  from vecops.language._compiler.operations.registry import OperationCatalog

  original, lookups = OperationCatalog.get, []

  def observe(self, operation):
    lookups.append(operation)
    return original(self, operation)

  monkeypatch.setattr(OperationCatalog, "get", observe)
  shared = local("i")
  for _ in range(64):
    shared = I.Expr(shared.type, "floordiv", (shared, shared))
  result = canonical_integer(I.Expr(shared.type, "sub", (shared, shared)))
  assert I.constant(result) == 0
  assert len(lookups) < 1000  # expanded structural keys require 2**64 visits


def test_integer_term_numbering_preserves_symbol_namespaces_and_effects():
  from vecops.language._compiler.operations import OpDef, snapshot, use_catalog
  from vecops.language._compiler.operations.registry import OperationCatalog

  kind = I.ScalarType("int64")
  first = I.Expr(kind, "term_probe", (local("i"),))
  second = I.Expr(kind, "term_probe", (local("i"),))
  difference = I.Expr(kind, "sub", (first, second))
  base = snapshot()
  for effect in ("none", "read", "unknown", "none"):
    definitions = dict(base.definitions)
    definitions["term_probe"] = OpDef("term_probe", "metadata", effects=effect)
    with use_catalog(OperationCatalog(definitions, f"term-{effect}")):
      result = canonical_integer(difference)
      assert (I.constant(result) == 0) == (effect == "none")
  raw = I.Expr(kind, "raw", data="i")
  assert I.constant(canonical_integer(I.Expr(kind, "sub", (raw, local("i"))))) is None
