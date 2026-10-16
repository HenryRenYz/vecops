# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Literal cached choices print only the arm whose storage can be live."""

from dataclasses import replace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.effects import read_accesses
from vecops.language._compiler.analysis.loops import effects, fusion_guard
from vecops.language._compiler.backends.cpu import backend
from vecops.language._compiler.backends.cpu.cpp.vectors import VectorsEmitter
from vecops.language._compiler.operations import snapshot


def template():
  @T.template
  def kernel(A: T.Tensor((8,), dtype="float32"), C: T.Tensor((8,), dtype="float32", access="output"), enabled: bool):
    T.store(C, T.load(A))

  return kernel


def install_cached_ir(monkeypatch, guard_kind, *, opaque=True):
  original = backend.lower

  def lower(*args, **kwargs):
    kernel = original(*args, **kwargs)
    store = kernel.body.statements[-1]
    producer = store.value
    cache = replace(store.destination, storage="conditional_cache", owned=True)
    condition = (I.Expr(I.ScalarType("bool"), "raw", data="enabled") if guard_kind == "dynamic"
                 else I.literal(guard_kind == "true"))
    wrapper = I.Expr(producer.type, "cached_value", (producer,), (cache, condition))
    # Valid C++ whose raw arithmetic is deliberately opaque to scalar facts.
    # The parent must survive cleanup, including its cached descriptor.
    sibling = I.Expr(I.ScalarType("float32"), "raw", data="(1.0f + 2.0f)") if opaque else I.literal(3, "float32")
    value = I.Expr(producer.type, "add", (wrapper, sibling))
    kernel.body.statements = [
      I.If(condition, I.Block([I.Allocate(cache), I.Store(cache, producer)]), I.Block([])),
      I.Store(store.destination, value),
    ]
    return kernel

  monkeypatch.setattr(backend, "lower", lower)


@pytest.mark.parametrize("guard_kind", ["false", "true", "dynamic"])
def test_opaque_parent_emits_only_reachable_literal_cache_arm(monkeypatch, guard_kind):
  install_cached_ir(monkeypatch, guard_kind)
  artifact = backend.CPUBackend().lower(template().ir, (), target="Scalar")
  if guard_kind == "false":
    assert "conditional_cache" not in artifact.source
  else:
    assert "conditional_cache" in artifact.source
  assert "if (false)" not in artifact.source
  assert "if (true)" not in artifact.source
  if guard_kind == "dynamic":
    assert "if (enabled)" in artifact.source


class Probe(VectorsEmitter):
  def __init__(self):
    self.chunk_cache = {}
    self.materialized = {}
    self.selected = []
    self.conditions = []

  def retain_expression(self, value):
    return value

  def vector(self, value, *args):
    if value.op == "cached_value":
      return super().vector(value, *args)
    self.selected.append(value)
    return "selected"

  def scalar(self, value):
    return "guard"

  def conditional_vector(self, typ, tag, condition, true_value, false_value):
    self.conditions.append(condition)
    return "conditional"


@pytest.mark.parametrize("guard_kind", ["false", "true", "dynamic", "read_operand", "integer", "expr_subclass", "type_subclass"])
def test_literal_choice_requires_exact_effect_free_boolean_leaf(guard_kind):
  shape = (I.literal(8),)
  view = I.View("cache", "float32", shape, (I.literal(1),), I.literal(0))
  producer = I.Expr(I.TileType("float32", shape), "load", data=view)
  guard = I.literal(guard_kind == "true")
  if guard_kind == "dynamic":
    guard = I.Expr(I.ScalarType("bool"), "raw", data="enabled")
  elif guard_kind == "read_operand":
    guard = I.Expr(I.ScalarType("bool"), "literal", (producer,), False)
  elif guard_kind == "integer":
    guard = I.Expr(I.ScalarType("bool"), "literal", data=0)
  elif guard_kind == "expr_subclass":
    class UnknownExpr(I.Expr):
      pass
    guard = UnknownExpr(I.ScalarType("bool"), "literal", data=False)
  elif guard_kind == "type_subclass":
    class UnknownType(I.ScalarType):
      pass
    guard = I.Expr(UnknownType("bool"), "literal", data=False)
  wrapper = I.Expr(producer.type, "cached_value", (producer,), (view, guard))
  emitter = Probe()
  emitter.vector(wrapper, "tag", shape, ("index",), "active")
  if guard_kind in {"false", "true"}:
    assert not emitter.conditions and len(emitter.selected) == 1
    if guard_kind == "false":
      assert emitter.selected[0] is producer
    else:
      assert emitter.selected[0].op == "load" and emitter.selected[0].data is view
  else:
    assert emitter.conditions == ["guard"] and not emitter.selected


@pytest.mark.parametrize("choice", [False, True, None])
def test_dependency_views_follow_cache_choice_without_synthetic_read_identities(choice):
  shape = (I.literal(8),)
  source = I.View("A", "float32", shape, (I.literal(1),), I.literal(0))
  cache = replace(source, storage="cache", owned=True)
  producer = I.Expr(I.TileType("float32", shape), "load", data=source)
  condition = (I.Expr(I.ScalarType("bool"), "raw", data="enabled") if choice is None else I.literal(choice))
  wrapper = I.Expr(producer.type, "cached_value", (producer,), (cache, condition))
  reads = read_accesses((wrapper, wrapper), snapshot())
  # Historical capture identity queries keep their original argument-edge
  # contract, even when semantic dependency queries select the cache arm.
  assert tuple(I.loads_unique(wrapper)) == (producer,)
  if choice is None:
    assert reads is None  # Do not evaluate an inactive cache address for aliasing.
  else:
    assert len(reads) == 1
    assert reads[0].expression is (wrapper if choice else producer)
    assert reads[0].view is (cache if choice else source)


def test_cache_read_dependency_prevents_cross_loop_fusion():
  shape = (I.literal(8),)
  source = I.View("A", "float32", shape, (I.literal(1),), I.literal(0))
  cache = replace(source, storage="cache", writable=True, owned=True)
  output = replace(source, storage="C", writable=True)
  producer = I.Expr(I.TileType("float32", shape), "load", data=source)
  wrapper = I.Expr(producer.type, "cached_value", (producer,), (cache, I.literal(True)))
  left = I.For("i", I.literal(0), I.literal(3), I.literal(1), I.Block([I.Store(cache, producer)]))
  right = replace(left, target="j", body=I.Block([I.Store(output, wrapper)]))
  catalog = snapshot()
  assert [read.view.storage for read in effects(right, catalog).reads] == ["cache"]
  assert fusion_guard(left, right, catalog) is None


def test_private_state_reference_is_not_an_ordinary_view_alias_barrier():
  state = I.TileState("state", I.TileType("float32", (I.literal(8),)))
  reference = I.Expr(state.type, "state_ref", data=state)
  assert read_accesses(reference, snapshot()) == ()


@pytest.mark.parametrize("guard_kind", ["false", "true", "dynamic"])
@pytest.mark.parametrize("opaque", [False, True])
def test_native_opaque_parent_literal_cache_choice(monkeypatch, tmp_path, guard_kind, opaque):
  install_cached_ir(monkeypatch, guard_kind, opaque=opaque)
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(template(), compiler=compiler)
  a = np.arange(8, dtype=np.float32)
  for enabled in (False, True):
    storage = np.full(12, np.nan, dtype=np.float32)
    c = storage[2:10]
    compiled.operator.native(a, c, enabled, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, a + 3)
    assert np.isnan(storage[:2]).all() and np.isnan(storage[10:]).all()
