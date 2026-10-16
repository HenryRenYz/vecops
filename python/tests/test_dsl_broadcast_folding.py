# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Literal shape folds retain dynamic checks and their original control path."""

import pytest
from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.affine import canonical_integer
from vecops.language._compiler.analysis.effects import ValueEffects
from vecops.language._compiler.operations import Trait, snapshot, use_catalog
from vecops.language._compiler.passes.kernel.cleanup import simplify_ir
from vecops.language._compiler.passes.manager import PassContext


def broadcast(a, b):
  return I.Expr(I.ScalarType("int64"), "broadcast_dim", (I.literal(a, "int64"), I.literal(b, "int64")))


def cleanup(statements):
  kernel = I.KernelIR(I.Block(statements), {}, {})
  context = PassContext("kernel-cleanup")
  with use_catalog(context.environment.operations):
    simplify_ir(kernel, context)
  return kernel.body.statements


@pytest.mark.parametrize("a,b,expected", [(32, 32, 32), (1, 32, 32), (32, 1, 32), (0, 1, 0),
                                          (1, 0, 0), (0, 0, 0), (-3, -3, -3), (1, -3, -3)])
def test_literal_broadcast_follows_the_runtime_compatibility_rules(a, b, expected):
  value = broadcast(a, b)
  result = cleanup([I.Assign("result", value)])[0].value
  assert I.constant(result) == expected
  assert result.type == value.type
  canonical = canonical_integer(value)
  assert I.constant(canonical) == expected
  assert canonical.type == value.type


def test_nested_compatible_shape_checks_fold_before_scalar_predicates():
  value = I.Expr(I.ScalarType("int64"), "broadcast_dim", (broadcast(32, 32), broadcast(1, 32)))
  condition = I.Expr(I.ScalarType("bool"), "ge", (value, I.literal(16)))
  result = cleanup([I.If(condition, I.Block([I.Assign("result", I.literal(7))]), I.Block([]))])
  assert len(result) == 1 and I.constant(result[0].value) == 7


@pytest.mark.parametrize("left_strong,right_strong", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("result_weak", [False, True])
def test_shape_fold_preserves_the_explicit_result_type_with_weak_or_mixed_arguments(left_strong, right_strong, result_weak):
  a, b = (I.literal(32, "int64" if strong else None) for strong in (left_strong, right_strong))
  value = I.Expr(I.ScalarType("int64", weak=result_weak), "broadcast_dim", (a, b))
  for result in (cleanup([I.Assign("result", value)])[0].value, canonical_integer(value)):
    assert I.constant(result) == 32
    assert result.type == value.type


@pytest.mark.parametrize("dtype,extent", [("int8", 128), ("int32", 32), ("int64", 1 << 63), ("bool", 1)])
def test_shape_fold_never_narrows_or_introduces_an_unrepresentable_constant(dtype, extent):
  value = I.Expr(I.ScalarType(dtype), "broadcast_dim", (I.literal(extent), I.literal(extent)))
  assert cleanup([I.Assign("result", value)])[0].value is value
  assert canonical_integer(value) is value


@pytest.mark.parametrize("a,b", [(2, 3), (0, 2), (-3, 2)])
def test_incompatible_executed_literal_check_remains_a_trapping_operation(a, b):
  value = broadcast(a, b)
  result = cleanup([I.Assign("result", value)])[0].value
  assert result is value
  assert canonical_integer(value) is value
  definition = snapshot().get(result)
  assert definition.may_trap and not definition.speculatable
  assert not ValueEffects(snapshot()).discardable(result)
  with pytest.raises(ValueError, match="incompatible DSL broadcast extents"):
    definition.fold(a, b)


@pytest.mark.parametrize("kind", ["branch", "where", "short-circuit"])
def test_dead_path_does_not_raise_its_incompatible_shape_check_during_cleanup(kind):
  invalid = broadcast(2, 3)
  if kind == "branch":
    statement = I.If(I.literal(False), I.Block([I.Assign("bad", invalid)]), I.Block([I.Assign("result", I.literal(9))]))
  elif kind == "where":
    statement = I.Assign("result", I.Expr(invalid.type, "where", (I.literal(False), invalid, I.literal(9, "int64"))))
  else:
    statement = I.Assign("result", I.Expr(I.ScalarType("bool"), "and", (I.literal(False), invalid)))
  result = cleanup([statement])[0].value
  assert I.constant(result) == (False if kind == "short-circuit" else 9)
  # General affine queries may inspect both arms before control cleanup.
  assert canonical_integer(invalid) is invalid


@pytest.mark.parametrize("a,b", [(2.0, 2.0), (True, 1), (1, False)])
def test_registry_fold_does_not_treat_floating_or_boolean_values_as_integer_shapes(a, b):
  with pytest.raises(ValueError, match="must be integers"):
    snapshot().get("broadcast_dim").fold(a, b)


def test_literal_fold_does_not_change_dynamic_shape_operation_traits():
  definition = snapshot().get("broadcast_dim")
  assert definition.family == "shape" and definition.arity == 2
  assert definition.traits == Trait.CSE
  assert definition.effects == "none" and definition.may_trap and not definition.speculatable
  dynamic = I.Expr(I.ScalarType("int64"), "local", data="n")
  value = I.Expr(dynamic.type, "broadcast_dim", (dynamic, I.literal(32, "int64")))
  assert cleanup([I.Assign("result", value)])[0].value is value
  assert not ValueEffects(snapshot()).speculatable(value)
