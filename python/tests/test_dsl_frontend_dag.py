# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Capture/state membership checks are linear in shared expression DAG size."""

from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import Frontend
from vecops.language._compiler.ir.traversal import walk, walk_unique
from vecops.language._compiler.operations import Trait


class CountingCatalog:
  def __init__(self, budget=300):
    self.calls = 0
    self.budget = budget

  def has(self, expression, traits):
    self.calls += 1
    assert self.calls <= self.budget, "a shared DAG was expanded as an exponential tree"
    if traits == Trait.CAPTURE_SCALAR:
      return expression.op == "local"
    if traits == Trait.STATE_REFERENCE:
      return expression.op == "state_ref"
    return False


def frontend():
  result = Frontend.__new__(Frontend)
  result.operations = CountingCatalog()
  result.block = I.Block()
  result.counter = 0
  return result


def shared_dag(leaf, depth=28):
  result = leaf
  for _ in range(depth):
    result = I.Expr(leaf.type, "mul", (result, result))
  return result


def test_freeze_preserves_shared_edges_and_captures_once_per_assignment():
  parser = frontend()
  scalar = I.Expr(I.ScalarType("float32"), "local", data="scale")
  typ = I.TileType("float32", (I.literal(37),))
  graph = shared_dag(I.Expr(typ, "fill", (scalar,)))
  first = parser.freeze_tile(graph)
  assert len(parser.block.statements) == 1
  for expression in walk_unique(first):
    if expression.op == "mul":
      assert expression.args[0] is expression.args[1]
  # The same original scalar after a later update needs a new snapshot, not
  # an assignment-spanning memo entry retaining its previous runtime value.
  parser.block.statements.append(I.Assign("scale", I.literal(2), False))
  second = parser.freeze_tile(graph)
  captures = [statement for statement in parser.block.statements if statement.value is scalar]
  assert len(captures) == 2
  assert captures[0].target != captures[1].target
  assert first is not second


def test_state_membership_without_state_does_not_expand_the_shared_dag():
  parser = frontend()
  typ = I.TileType("float32", (I.literal(37),))
  graph = shared_dag(I.Expr(typ, "fill", (I.literal(1),)))
  assert parser.snapshot_state(graph) is graph
  assert not parser.block.statements
  assert parser.operations.calls == 30  # 28 shared multiplies, fill, scalar leaf.


def test_state_membership_still_creates_an_independent_snapshot():
  parser = frontend()
  typ = I.TileType("float32", (I.literal(37),))
  state = I.TileState("carried", typ)
  graph = shared_dag(I.Expr(typ, "state_ref", data=state))
  snapshot = parser.snapshot_state(graph)
  assert snapshot.op == "state_ref"
  assert snapshot.data is not state
  assert parser.block.statements[0].value is graph


def test_unique_expression_walk_does_not_change_occurrence_walk_or_metadata_scope():
  leaf = I.literal(3)
  shape = I.literal(7)
  typ = I.TileType("float32", (shape,))
  fill = I.Expr(typ, "fill", (leaf,))
  graph = I.Expr(typ, "add", (fill, fill))
  assert list(walk(graph)) == [graph, fill, leaf, fill, leaf]
  assert list(walk_unique(graph)) == [graph, fill, leaf]
  assert all(node is not shape for node in walk_unique(graph))


def test_memory_dependency_identity_walk_preserves_distinct_equal_loads(monkeypatch):
  from vecops.language._compiler.ir import kernel

  class MemoryCatalog(CountingCatalog):
    def has(self, expression, traits):
      super().has(expression, traits)
      return expression.op == "load"

  catalog = MemoryCatalog()
  monkeypatch.setattr(kernel, "operation_snapshot", lambda: catalog)
  shape = (I.literal(37),)
  view = I.View("A", "float32", shape, (I.literal(1),), I.literal(0))
  first = I.Expr(I.TileType("float32", shape), "load", data=view)
  second = I.Expr(first.type, "load", data=view)
  small = I.Expr(first.type, "add", (first, first))
  assert list(I.loads(small)) == [first, first]
  assert list(I.loads_unique(small)) == [first]
  # Structural equality does not mean a captured read has the same lifetime.
  # Separate reads may be on opposite sides of a write or helper boundary.
  distinct = I.Expr(first.type, "add", (first, second))
  actual = list(I.loads_unique(distinct))
  assert len(actual) == 2 and actual[0] is first and actual[1] is second
  graph = shared_dag(first)
  assert list(I.loads_unique(graph)) == [first]
