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


@pytest.mark.parametrize("left_dtype,right_dtype", [(None, "int64"), ("int64", None), ("int32", "int64"),
                                                    ("uint8", "int64")])
def test_equal_integer_extents_preserve_distinct_scalar_types(left_dtype, right_dtype):
  left, right = I.literal(128, left_dtype), I.literal(128, right_dtype)
  assert not I.same_expr(left, right)
  assert I.same_extent(left, right) and I.same_extent(right, left)
  assert I.broadcast_shape((left,), (right,)) == (left,)
  assert I.broadcast_shape((right,), (left,)) == (right,)
  assert not I.same_expr(left, right)


@pytest.mark.parametrize("dtype,data", [("int8", 128), ("uint8", -1), ("int64", 1 << 63),
                                       ("float32", 128), ("bool", True)])
def test_extent_equality_does_not_validate_malformed_or_noninteger_literals(dtype, data):
  value = I.Expr(I.ScalarType(dtype), "literal", data=data)
  assert not I.same_extent(value, I.literal(data))


def test_extent_literal_proof_rejects_nonleaf_and_extended_descriptors():
  from dataclasses import dataclass

  @dataclass(frozen=True)
  class ExtendedScalar(I.ScalarType):
    metadata: object = None

  @dataclass(frozen=True, eq=False)
  class ExtendedExpression(I.Expr):
    metadata: object = None

  load = I.Expr(I.ScalarType("int64"), "load", data=I.View("A", "int64", (), (), I.literal(0)))
  normal = I.literal(128)
  values = (
    I.Expr(I.ScalarType("int64"), "literal", (load,), data=128),
    I.Expr(ExtendedScalar("int64", metadata=load), "literal", data=128),
    ExtendedExpression(I.ScalarType("int64"), "literal", data=128, metadata=load),
  )
  for value in values:
    assert not I.same_extent(value, normal)
    assert not I.same_extent(normal, value)


@pytest.mark.parametrize("source_strong,destination_strong", [(False, True), (True, False)])
@pytest.mark.parametrize("masked", [False, True])
def test_store_broadcast_accepts_equal_weak_and_strong_dimensions(source_strong, destination_strong, masked):
  from types import SimpleNamespace

  from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter

  class Emitter(MemoryEmitter):
    prepared = SimpleNamespace(transposes={})

    def __init__(self):
      self.lines = []

    def scalar(self, value):
      return str(I.constant(value))

    def checked_shape(self, shape):
      return f"checked_shape({shape})"

    def line(self, value):
      self.lines.append(value)

  source = I.literal(128, "int64" if source_strong else None)
  destination = I.literal(128, "int64" if destination_strong else None)
  view = I.View("C", "float32", (destination,), (I.literal(1),), I.literal(0))
  value = I.Expr(I.TileType("float32", (source,)), "fill", (I.literal(0, "float32"),))
  mask = I.Expr(I.TileType("bool", (source,), mask=True), "fill", (I.literal(True),)) if masked else None
  emitter = Emitter()
  emitter.check_store(I.Store(view, value, mask))
  assert not any("check_broadcast" in line for line in emitter.lines)
  mismatch = I.Expr(I.TileType("float32", (I.literal(127),)), "fill", (I.literal(0, "float32"),))
  with pytest.raises(TypeError, match="does not broadcast"):
    emitter.check_store(I.Store(view, mismatch))
  with pytest.raises(TypeError, match="not broadcast-compatible"):
    I.broadcast_shape((source,), (I.literal(127),))


@pytest.mark.parametrize("dtype,m,n,k", [("uint8", 2, 4, 257), ("float16", 19, 21, 17),
                                        ("bfloat16", 16, 1, 65)])
def test_sme_small_packed_shapes_lower_after_typed_broadcast_folding(dtype, m, n, k):
  import vecops.language as T
  from vecops._arm_vector_state import VectorState

  packed_size = T.dynamic(lower=1)
  output_dtype = "int32" if dtype == "uint8" else "float32"

  @T.template
  def kernel(
    AP: T.Tensor((packed_size,), dtype=dtype),
    BP: T.Tensor((packed_size,), dtype=dtype),
    C: T.Tensor((m, n), dtype=output_dtype, access="output"),
  ):
    a = T.packed(AP, (m, k), side="A", format="sme")
    b = T.packed(BP, (k, n), side="B", format="sme")
    T.matmul((a, b), C)

  lowered = T.lower(kernel, target="SME", pass_options={"_vector_state": VectorState(64, 64)})
  assert lowered.artifact.source
