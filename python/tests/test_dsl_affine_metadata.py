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
    assert all(-(1 << 63) <= value.data <= maximum for value in walk_unique(guarded_limit)
               if value.op == "literal" and type(value.data) is int)
    for lanes in (8, 16, 32, 64, 256):
      interpreter.environment["streaming.width"] = lanes
      assert interpreter.value(guarded_limit) == maximum - (factor * lanes - 1)
