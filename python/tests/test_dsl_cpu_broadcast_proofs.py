# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Scoped certificates preserve geometry identities and reject stale domains."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.broadcasts import (
  broadcast_key,
  collect_broadcast_proofs,
)
from vecops.language._compiler.backends.cpu.verification import verify_broadcast_proofs


def local(name, *, dtype="int64", frozen=False):
  return I.Expr(I.ScalarType(dtype), "frozen" if frozen else "local", data=name)


def clamp_example():
  i = local("i")
  zero, limit = I.literal(0), I.literal(17)
  start = I.binary("max", zero, I.binary("min", i, limit))
  end = I.binary("max", start, I.binary("min", I.binary("add", i, I.literal(16)), limit))
  destination = I.binary("sub", end, start)
  simplified = I.binary("min", I.literal(16), I.binary("sub", limit, i))
  source = local("captured", frozen=True)
  view = I.View("C", "float32", (destination,), (I.literal(1),), zero)
  value = I.Expr(I.TileType("float32", (source,)), "literal", data=0.0)
  store = I.Store(view, value)
  body = I.Block([I.Assign("captured", simplified), store])
  loop = I.For("i", zero, limit, I.literal(16), body)
  return I.KernelIR(I.Block([loop]), {}, {}), store, source, destination


def prepared(kernel, proofs):
  return SimpleNamespace(vectors=SimpleNamespace(kernel=kernel), broadcast_proofs=proofs)


def test_simplified_capture_and_historical_clamp_keep_their_original_identities():
  kernel, store, source, destination = clamp_example()
  proof = collect_broadcast_proofs(kernel)[broadcast_key(store, source, destination)]
  assert proof.store is store and proof.source is source and proof.destination is destination
  assert store.value.type.shape[0] is source and store.destination.shape[0] is destination
  assert proof.occurrences == 1
  verify_broadcast_proofs(prepared(kernel, {broadcast_key(store, source, destination): proof}))


def test_scalar_capture_is_not_visible_before_its_declaration():
  kernel, _store, _source, _destination = clamp_example()
  body = kernel.body.statements[0].body
  body.statements.reverse()
  assert not collect_broadcast_proofs(kernel)


def test_shared_store_must_be_safe_at_every_lexical_occurrence():
  kernel, store, source, destination = clamp_example()
  loop = kernel.body.statements[0]
  second = replace(loop, start=I.literal(1), stop=I.literal(16), step=I.literal(1))
  kernel.body.statements.append(second)
  proof = collect_broadcast_proofs(kernel)[broadcast_key(store, source, destination)]
  assert proof.occurrences == 2
  kernel.body.statements[-1] = replace(second, start=I.literal(18), stop=I.literal(19))
  assert not collect_broadcast_proofs(kernel)


def test_loop_counter_write_and_opaque_effect_kill_the_domain():
  for mutation in (
    I.Assign("i", I.literal(18), False),
    I.Assign("opaque", I.Expr(I.ScalarType("int64"), "raw", data="mutate(i)")),
  ):
    kernel, _store, _source, _destination = clamp_example()
    kernel.body.statements[0].body.statements.insert(0, mutation)
    assert not collect_broadcast_proofs(kernel)


def test_branch_write_does_not_leak_an_extent_equality_after_the_join():
  kernel, _store, _source, _destination = clamp_example()
  body = kernel.body.statements[0].body
  condition = I.Expr(I.ScalarType("bool"), "raw", data="unknown_flag")
  body.statements.insert(1, I.If(condition, I.Block([I.Assign("captured", I.literal(3), False)]), I.Block()))
  assert not collect_broadcast_proofs(kernel)


@pytest.mark.parametrize("dtype", ["int8", "uint64", "bool"])
def test_unmodelled_scalar_extent_types_never_receive_a_certificate(dtype):
  kernel, _store, _source, _destination = clamp_example()
  body = kernel.body.statements[0].body
  assignment = body.statements[0]
  body.statements[0] = replace(assignment, value=I.Expr(I.ScalarType(dtype), "raw", data="unknown_extent"))
  assert not collect_broadcast_proofs(kernel)


def test_unknown_and_overflowing_extents_are_not_elided():
  for extent in (
    local("unbound"),
    I.binary("add", I.literal((1 << 63) - 1), I.literal(1)),
    I.Expr(I.ScalarType("int64"), "raw", data="may_throw()"),
  ):
    view = I.View("C", "float32", (I.literal(4),), (I.literal(1),), I.literal(0))
    value = I.Expr(I.TileType("float32", (extent,)), "literal", data=0.0)
    store = I.Store(view, value)
    assert not collect_broadcast_proofs(I.KernelIR(I.Block([store]), {}, {}))


def test_verifier_rejects_detached_subjects_and_changed_final_domains():
  kernel, store, source, destination = clamp_example()
  key = broadcast_key(store, source, destination)
  proof = collect_broadcast_proofs(kernel)[key]
  with pytest.raises(ValueError, match="identity"):
    verify_broadcast_proofs(prepared(kernel, {key: replace(proof, store=replace(store))}))
  with pytest.raises(ValueError, match="occurrence"):
    verify_broadcast_proofs(prepared(kernel, {key: replace(proof, occurrences=2)}))
  loop = kernel.body.statements[0]
  kernel.body.statements[0] = replace(loop, start=I.literal(18), stop=I.literal(19))
  with pytest.raises(ValueError, match="not proved"):
    verify_broadcast_proofs(prepared(kernel, {key: proof}))


def test_repeated_operand_pairs_are_counted_once_per_actual_store_occurrence():
  kernel, store, source, destination = clamp_example()
  store = replace(store, mask=I.Expr(I.TileType("bool", (source,)), "literal", data=True))
  body = kernel.body.statements[0].body
  body.statements[-1] = I.ElementwiseGroup((store, store))
  proof = collect_broadcast_proofs(kernel)[broadcast_key(store, source, destination)]
  assert proof.occurrences == 2


def test_verifier_uses_the_artifact_operation_snapshot_after_registry_changes():
  from vecops.language._compiler.operations import snapshot, use_catalog
  from vecops.language._compiler.operations.registry import OperationCatalog

  kernel, store, source, destination = clamp_example()
  catalog = snapshot()
  proofs = collect_broadcast_proofs(kernel)
  artifact = prepared(kernel, proofs)
  artifact.vectors.manager = SimpleNamespace(options={"_compiler_environment": SimpleNamespace(operations=catalog)})
  definitions = dict(catalog.definitions)
  definitions["min"] = replace(definitions["min"], effects="unknown", speculatable=False)
  with use_catalog(OperationCatalog(definitions, "broadcast-later-registry")):
    assert not collect_broadcast_proofs(kernel)
    verify_broadcast_proofs(artifact)
    with pytest.raises(ValueError, match="not proved"):
      verify_broadcast_proofs(prepared(kernel, proofs))
  assert broadcast_key(store, source, destination) in proofs


def test_emission_consumes_only_the_exact_prepared_subject_without_rewriting_geometry():
  from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter

  kernel, store, _source, _destination = clamp_example()
  proof = collect_broadcast_proofs(kernel)
  emitter = MemoryEmitter()
  emitter.prepared = SimpleNamespace(transposes={}, broadcast_proofs=proof)
  emitted = []
  emitter.line = emitted.append
  emitter.scalar = lambda _value: "extent"
  emitter.checked_shape = lambda _shape: "checked_shape()"
  emitter.check_store(store)
  assert not any("check_broadcast" in line for line in emitted)
  emitted.clear()
  emitter.check_store(replace(store))
  assert any("check_broadcast" in line for line in emitted)


def test_singleton_compatibility_never_becomes_coordinate_equality():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis

  source = local("one", frozen=True)
  destination = I.literal(4)
  view = I.View("C", "float32", (destination,), (I.literal(1),), I.literal(0))
  store = I.Store(view, I.Expr(I.TileType("float32", (source,)), "literal", data=0.0))
  kernel = I.KernelIR(I.Block([I.Assign("one", I.literal(1)), store]), {}, {})
  analysis = collect_broadcast_analysis(kernel)
  proof = analysis.proofs[broadcast_key(store, source, destination)]
  assert not proof.equal and not analysis.geometry


def test_shared_subject_intersects_equality_separately_from_compatibility():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis

  source, destination = local("extent", frozen=True), I.literal(4)
  view = I.View("C", "float32", (destination,), (I.literal(1),), I.literal(0))
  store = I.Store(view, I.Expr(I.TileType("float32", (source,)), "literal", data=0.0))
  group = I.ElementwiseGroup((store,))
  kernel = I.KernelIR(
    I.Block([I.Assign("extent", I.literal(4)), group, I.Assign("extent", I.literal(1), False), group]), {}, {}
  )
  analysis = collect_broadcast_analysis(kernel)
  proof = analysis.proofs[broadcast_key(store, source, destination)]
  assert proof.occurrences == 2 and not proof.equal
  assert not analysis.geometry


def test_syntactically_known_broadcasts_do_not_need_certificates():
  extent = local("unbound")
  for source in (I.literal(1), extent):
    destination = extent
    store = I.Store(
      I.View("C", "float32", (destination,), (I.literal(1),), I.literal(0)),
      I.Expr(I.TileType("float32", (source,)), "literal", data=0.0),
    )
    assert not collect_broadcast_proofs(I.KernelIR(I.Block([store]), {}, {}))


def test_geometry_is_exact_subject_scoped_and_restored_on_exception():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter

  kernel, store, source, destination = clamp_example()
  analysis = collect_broadcast_analysis(kernel)
  emitter = MemoryEmitter()
  emitter.prepared = SimpleNamespace(extent_geometry=analysis.geometry)
  emitter.scalar = lambda value: "source" if value is source else "destination"
  original_cache = {("outer",): "outer_vector"}
  emitter.chunk_cache = original_cache
  assert "?" in emitter.coordinates((source,), (destination,), ("i",))[0]
  with pytest.raises(RuntimeError, match="abort"), emitter.extent_geometry(store):
    assert emitter.chunk_cache == {} and emitter.chunk_cache is not original_cache
    assert emitter.coordinates((source,), (destination,), ("i",)) == ("i",)
    assert emitter.active_extent_aliases[id(destination)] == (destination, source)
    emitter.chunk_cache[("inside",)] = "inside_vector"
    with emitter.extent_geometry(replace(store)):
      assert not emitter.same_geometry(source, destination)
      assert not emitter.active_extent_aliases
      assert not emitter.chunk_cache
    assert emitter.same_geometry(source, destination)
    assert emitter.chunk_cache == {("inside",): "inside_vector"}
    raise RuntimeError("abort")
  assert emitter.chunk_cache is original_cache
  assert not emitter.active_extent_aliases
  assert not emitter.same_geometry(source, destination)


def test_geometry_equality_preserves_direct_state_register_consumption():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter
  from vecops.language._compiler.backends.cpu.cpp.state import StateEmitter

  class Emitter(MemoryEmitter, StateEmitter):
    pass

  kernel, store, source, destination = clamp_example()
  analysis = collect_broadcast_analysis(kernel)
  emitter = Emitter()
  state = I.TileState("saved", I.TileType("float32", (source,)))
  value = I.Expr(state.type, "state_ref", data=state)
  plan = SimpleNamespace(pieces=(object(),))
  emitter.prepared = SimpleNamespace(extent_geometry=analysis.geometry, states={"saved": plan})
  emitter.chunk_cache = {}
  emitter.state_tags = lambda *_: ("tag",)
  emitter.state_coordinates = lambda _: (("0",),)
  emitter.same_register_tag = lambda a, b: a == b
  with emitter.extent_geometry(store):
    assert emitter.state_vector(value, "tag", (destination,), ("0",), "active", 0) == "saved_r0"
  assert state.type.shape[0] is source and store.destination.shape[0] is destination


def test_memory_proofs_share_one_lexical_walk(monkeypatch):
  from vecops.language._compiler.backends.cpu import memory_proofs

  kernel, store, source, destination = clamp_example()
  walk = memory_proofs.walk_lexical_scopes
  calls = []

  def observe(*args):
    calls.append(args[0])
    return walk(*args)

  monkeypatch.setattr(memory_proofs, "walk_lexical_scopes", observe)
  result = memory_proofs.collect_memory_proofs(kernel)
  assert calls == [kernel]
  assert result.broadcasts[broadcast_key(store, source, destination)].equal
  assert result.geometry[id(store)].subject is store


def test_geometry_verifier_rejects_forged_classes_and_subjects():
  from vecops.language._compiler.backends.cpu.broadcasts import ExtentGeometry, collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.verification import verify_extent_geometry

  kernel, store, source, destination = clamp_example()
  analysis = collect_broadcast_analysis(kernel)
  artifact = prepared(kernel, analysis.proofs)
  artifact.extent_geometry = analysis.geometry
  verify_extent_geometry(artifact)
  artifact.extent_geometry = {id(store): ExtentGeometry(replace(store), ((source, destination),))}
  with pytest.raises(ValueError, match="identity"):
    verify_extent_geometry(artifact)
  artifact.extent_geometry = {id(store): ExtentGeometry(store, ((source, I.literal(1)),))}
  with pytest.raises(ValueError, match="not proved"):
    verify_extent_geometry(artifact)


def test_group_geometry_unites_equal_pairs_without_leaking_to_a_copied_group():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter

  kernel, first, source, destination = clamp_example()
  other = local("second", frozen=True)
  second = I.Store(first.destination, I.Expr(I.TileType("float32", (other,)), "literal", data=1.0))
  group = I.ElementwiseGroup((first, second))
  body = kernel.body.statements[0].body
  body.statements.insert(1, I.Assign("second", source))
  body.statements[-1] = group
  analysis = collect_broadcast_analysis(kernel)
  emitter = MemoryEmitter()
  emitter.prepared = SimpleNamespace(extent_geometry=analysis.geometry)
  emitter.chunk_cache = {}
  with emitter.extent_geometry(group):
    assert emitter.same_geometry(source, destination)
    assert emitter.same_geometry(other, destination)
    assert emitter.same_geometry(source, other)
  with emitter.extent_geometry(replace(group)):
    assert not emitter.same_geometry(source, other)
  assert group.stores == (first, second)


def test_pipeline_registers_one_memory_proof_pass():
  from vecops.language._compiler.passes.manager import registered_passes
  from vecops.language._compiler.pipeline import load_builtin_passes

  load_builtin_passes()
  names = [entry.name for entry in registered_passes()]
  assert names.count("cpu-plan-memory-proofs") == 1
  assert "cpu-plan-partition-proofs" not in names
  assert "cpu-plan-broadcast-proofs" not in names


def test_prepared_representative_prefers_an_available_immutable_scalar():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.cpp.expressions import ExpressionsEmitter
  from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter

  class Emitter(MemoryEmitter, ExpressionsEmitter):
    pass

  kernel, store, source, destination = clamp_example()
  analysis = collect_broadcast_analysis(kernel)
  plan = analysis.geometry[id(store)]
  assert plan.representatives == (source,)
  emitter = Emitter()
  emitter.prepared = SimpleNamespace(extent_geometry=analysis.geometry)
  from vecops.language._compiler.operations import snapshot

  emitter.environment = SimpleNamespace(operations=snapshot())
  emitter.hardware_scalar_bindings = {}
  emitter.hardware_scope = None
  emitter.chunk_cache = {}
  with emitter.extent_geometry(store):
    assert emitter.scalar(destination) == emitter.scalar(source) == "captured"
    assert store.destination.shape[0] is destination
  assert not emitter.active_extent_aliases
  assert emitter.scalar(source) == "captured"


def test_future_mutation_disqualifies_a_scalar_as_preferred_representative():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis

  kernel, store, source, _destination = clamp_example()
  kernel.body.statements[0].body.statements.append(I.Assign("captured", I.literal(2), False))
  analysis = collect_broadcast_analysis(kernel)
  assert analysis.geometry[id(store)].representatives[0] is not source


def test_geometry_representatives_reject_cross_class_alias_cycles():
  from vecops.language._compiler.backends.cpu.broadcasts import _representatives

  a, b = local("a"), local("b")
  x, y = I.binary("add", b, I.literal(0)), I.binary("add", a, I.literal(0))
  assert _representatives(((a, x), (b, y)), set(), set()) == (None, None)


def test_geometry_verifier_rejects_a_different_scalar_representative():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.verification import verify_extent_geometry

  kernel, store, _source, destination = clamp_example()
  analysis = collect_broadcast_analysis(kernel)
  artifact = prepared(kernel, analysis.proofs)
  artifact.extent_geometry = {id(store): replace(analysis.geometry[id(store)], representatives=(destination,))}
  with pytest.raises(ValueError, match="representatives"):
    verify_extent_geometry(artifact)


def separate_group_captures():
  from copy import deepcopy

  kernel, first, a, destination = clamp_example()
  b = local("other", frozen=True)
  second = I.Store(
    replace(first.destination, shape=(deepcopy(destination),)), I.Expr(I.TileType("float32", (b,)), "literal", data=1.0)
  )
  group = I.ElementwiseGroup((first, second))
  body = kernel.body.statements[0].body
  body.statements.insert(1, I.Assign("other", deepcopy(body.statements[0].value)))
  body.statements[-1] = group
  return kernel, group, a, b


def test_group_iteration_root_connects_independent_historical_captures_only_in_group():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter

  kernel, group, a, b = separate_group_captures()
  analysis = collect_broadcast_analysis(kernel)
  emitter = MemoryEmitter()
  emitter.prepared = SimpleNamespace(extent_geometry=analysis.geometry)
  emitter.chunk_cache = {}
  with emitter.extent_geometry(group):
    assert emitter.same_geometry(a, b)
  for store in group.stores:
    with emitter.extent_geometry(store):
      assert not emitter.same_geometry(a, b)


def test_shared_group_cross_store_equality_intersects_all_execution_domains():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter

  kernel, group, a, b = separate_group_captures()
  loop = kernel.body.statements[0]
  # Both local stores remain valid at the first site. In the second site the
  # second capture has an independent value, so no group equality may escape.
  copied = I.Block([I.Assign("captured", I.literal(4)), I.Assign("other", I.literal(1)), group])
  kernel.body.statements.append(replace(loop, start=I.literal(0), stop=I.literal(1), body=copied))
  analysis = collect_broadcast_analysis(kernel)
  emitter = MemoryEmitter()
  emitter.prepared = SimpleNamespace(extent_geometry=analysis.geometry)
  emitter.chunk_cache = {}
  with emitter.extent_geometry(group):
    assert not emitter.same_geometry(a, b)


def test_group_singleton_source_is_not_unified_with_the_iteration_root():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis

  one = local("one", frozen=True)
  shape = (I.literal(4),)
  view = I.View("C", "float32", shape, (I.literal(1),), I.literal(0))
  first = I.Store(view, I.Expr(I.TileType("float32", shape), "literal", data=2.0))
  second = I.Store(replace(view, storage="D"), I.Expr(I.TileType("float32", (one,)), "literal", data=3.0))
  group = I.ElementwiseGroup((first, second))
  kernel = I.KernelIR(I.Block([I.Assign("one", I.literal(1)), group]), {}, {})
  analysis = collect_broadcast_analysis(kernel)
  assert not analysis.geometry
  assert not analysis.proofs[broadcast_key(second, one, shape[0])].equal


def test_group_cross_store_geometry_covers_every_aligned_dimension():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter

  a, b = local("rows", frozen=True), local("cols", frozen=True)
  shape = (I.literal(2), I.literal(3))
  first = I.Store(
    I.View("C", "float32", shape, (I.literal(3), I.literal(1)), I.literal(0)),
    I.Expr(I.TileType("float32", shape), "literal", data=2.0),
  )
  second = I.Store(
    replace(first.destination, storage="D", shape=(a, b)), I.Expr(I.TileType("float32", (a, b)), "literal", data=3.0)
  )
  group = I.ElementwiseGroup((first, second))
  kernel = I.KernelIR(I.Block([I.Assign("rows", I.literal(2)), I.Assign("cols", I.literal(3)), group]), {}, {})
  analysis = collect_broadcast_analysis(kernel)
  emitter = MemoryEmitter()
  emitter.prepared = SimpleNamespace(extent_geometry=analysis.geometry)
  emitter.chunk_cache = {}
  with emitter.extent_geometry(group):
    assert emitter.same_geometry(a, shape[0]) and emitter.same_geometry(b, shape[1])
    assert not emitter.same_geometry(a, shape[1])


def test_verifier_uses_actual_final_group_vector_root_shape():
  from vecops.language._compiler.backends.cpu.broadcasts import collect_broadcast_analysis
  from vecops.language._compiler.backends.cpu.verification import verify_extent_geometry

  kernel, group, _a, _b = separate_group_captures()
  analysis = collect_broadcast_analysis(kernel)
  artifact = prepared(kernel, analysis.proofs)
  artifact.extent_geometry = analysis.geometry
  artifact.vectors.plans = {id(group): SimpleNamespace(shape=(I.literal(99),))}
  with pytest.raises(ValueError, match="not proved"):
    verify_extent_geometry(artifact)
