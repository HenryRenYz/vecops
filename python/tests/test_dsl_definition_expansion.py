# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Definition expansion preserves ownership, query scope, and DAG sharing."""

import pytest
from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.affine import expand_definitions
from vecops.language._debug.interpreter import Interpreter


def local(name):
  return I.Expr(I.ScalarType("int64"), "local", data=name)


def test_empty_and_unrelated_definitions_preserve_reader_identity():
  value = local("n")
  root = I.Expr(value.type, "add", (value, I.literal(1)))
  assert expand_definitions(root, {}) is root
  assert expand_definitions(root, {"other": I.literal(7)}) is root
  view = I.View("A", "int64", (), (), I.literal(0))
  read = I.Expr(value.type, "load", data=view)
  assert expand_definitions(read, {"other": I.literal(7)}) is read


def test_substitution_changes_only_affected_argument_edges():
  value = local("n")
  other = I.literal(1)
  root = I.Expr(value.type, "add", (value, other))
  replacement = I.literal(7)
  changed = expand_definitions(root, {"n": replacement})
  assert changed is not root and changed.args == (replacement, other) and changed.type is root.type
  interpreter = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
  assert interpreter.value(changed) == 8


@pytest.mark.parametrize("depth", (1, 24))
def test_shared_diamond_is_rewritten_once_without_expanding_incoming_edges(depth):
  value = local("n")
  root = value
  for _ in range(depth):
    root = I.Expr(value.type, "add", (root, root))
  replacement = I.literal(7)
  changed = expand_definitions(root, {"n": replacement})
  for _ in range(depth):
    assert changed.args[0] is changed.args[1]
    changed = changed.args[0]
  assert changed is replacement


def test_cycle_and_exclusions_are_query_local():
  a, b = local("a"), local("b")
  defs = {"a": b, "b": a}
  assert expand_definitions(a, defs) is a
  assert expand_definitions(a, defs, frozenset({"a"})) is a
  assert expand_definitions(a, defs, {"a"}) is a
  assert expand_definitions(a, defs, frozenset({"b"})) is b
  assert expand_definitions(a, {"a": I.literal(9)}).data == 9
  assert expand_definitions(a, {}) is a


def test_same_node_under_different_exclusions_has_distinct_results():
  a, b = local("a"), local("b")
  pair = I.Expr(a.type, "add", (a, b))
  defs = {"a": pair, "b": I.literal(7)}
  changed = expand_definitions(pair, defs)
  assert changed.args[0].args[0] is a
  assert changed.args[0].args[1] is changed.args[1] is defs["b"]
  assert expand_definitions(pair, defs, frozenset({"a"})).args[0] is a


def test_data_owned_geometry_is_not_an_argument_substitution_edge():
  extent = local("n")
  view = I.View("A", "int64", (extent,), (I.literal(1),), I.literal(0))
  value = I.Expr(I.TileType("int64", (extent,)), "load", data=view)
  assert expand_definitions(value, {"n": I.literal(8)}) is value
  assert value.data.shape[0] is extent
