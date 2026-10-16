# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Late owned-region proofs preserve lexical domains and guarded fallbacks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter
from vecops.language._compiler.backends.cpu.partitions import collect_partition_proofs, partition_key
from vecops.language._compiler.backends.cpu.verification import verify_partition_proofs


def geometry(*, delta=1, stride=1, step=1, width=2, dtype="int64", row=None):
  root = I.View(
    "scratch", "float32", (I.literal(16), I.literal(8)), (I.literal(8), I.literal(stride)), I.literal(0), owned=True
  )
  row = row if row is not None else I.Expr(I.ScalarType(dtype), "local", data="i")

  def view(start):
    region = I.ViewRegion(
      root.shape,
      root.strides,
      (start, I.literal(0)),
      (I.literal(1), I.literal(width)),
      (step, 1),
      (1,),
      (False, True),
      exact_starts=(start, I.literal(0)),
      affine_partition=False,
    )
    return I.View(
      root.storage,
      root.dtype,
      (I.literal(width),),
      (I.literal(stride),),
      I.binary("mul", start, I.literal(8)),
      owned=True,
      region=region,
    )

  first, second = view(row), view(I.binary("add", row, I.literal(delta)))
  value = I.Expr(I.TileType("float32", (I.literal(width),)), "literal", data=0.0)
  group = I.ElementwiseGroup((I.Store(first, value), I.Store(second, value)))
  return root, first, second, group


def kernel(root, *statements):
  return I.KernelIR(I.Block([I.Allocate(root), *statements]), {}, {})


def loop(body, start=0, stop=8):
  return I.For("i", I.literal(start), I.literal(stop), I.literal(1), body)


def test_clipped_other_axis_does_not_hide_proven_row_separation():
  root, first, second, group = geometry()
  ir = kernel(root, loop(I.Block([group])))
  proofs = collect_partition_proofs(ir)
  proof = proofs[partition_key(group, first, second)]
  assert proof.occurrences == 1
  emitter = MemoryEmitter()
  emitter.prepared = SimpleNamespace(partition_proofs=proofs)
  emitter.storage_views = {root.storage: root}
  assert emitter.register_disjoint_proven(first, second, group=group)
  assert not emitter.register_disjoint_proven(first, second)
  assert not emitter.register_disjoint_proven(first, second, group=I.ElementwiseGroup(group.stores))


def test_shared_block_requires_proof_at_every_lexical_occurrence():
  root, first, second, group = geometry()
  shared = I.Block([group])
  ir = kernel(root, loop(shared), loop(shared, 1, 7))
  proof = collect_partition_proofs(ir)[partition_key(group, first, second)]
  assert proof.occurrences == 2
  unsafe = kernel(root, loop(shared), loop(shared, 15, 16))
  assert not collect_partition_proofs(unsafe)


@pytest.mark.parametrize(
  "options",
  [
    {"delta": 0},
    {"stride": -1},
    {"stride": 2},
    {"step": -1},
    {"width": 9},
    {"dtype": "uint64"},
  ],
)
def test_unknown_or_overlapping_geometry_keeps_runtime_guard(options):
  root, _first, _second, group = geometry(**options)
  assert not collect_partition_proofs(kernel(root, loop(I.Block([group]))))


def test_signed_overflow_and_unknown_induction_are_not_affine_proofs():
  root, _first, _second, group = geometry()
  limit = (1 << 63) - 1
  assert not collect_partition_proofs(kernel(root, loop(I.Block([group]), limit - 1, limit)))
  unknown = I.For(
    "i", I.literal(0), I.Expr(I.ScalarType("int64"), "raw", data="unknown_bound"), I.literal(1), I.Block([group])
  )
  assert not collect_partition_proofs(kernel(root, unknown))


@pytest.mark.parametrize("step", [I.literal(0), I.Expr(I.ScalarType("int64"), "raw", data="unknown_step")])
def test_zero_or_unknown_loop_steps_do_not_establish_a_domain(step):
  root, _first, _second, group = geometry()
  statement = replace(loop(I.Block([group])), step=step)
  assert not collect_partition_proofs(kernel(root, statement))


def test_reverse_loop_uses_its_proven_signed_domain():
  root, first, second, group = geometry()
  statement = replace(loop(I.Block([group]), 7, -1), step=I.literal(-1))
  assert partition_key(group, first, second) in collect_partition_proofs(kernel(root, statement))


def test_loop_writes_invalidate_captured_scalar_equalities():
  root, first, second, group = geometry()
  # The group reads i; writing its induction slot defeats the stable domain.
  update = I.Assign("i", I.binary("add", I.Expr(I.ScalarType("int64"), "local", data="i"), I.literal(1)), False)
  assert not collect_partition_proofs(kernel(root, loop(I.Block([update, group]))))
  safe = collect_partition_proofs(kernel(root, loop(I.Block([group]))))
  assert partition_key(group, first, second) in safe


def test_opaque_scalar_operation_invalidates_the_domain():
  root, _first, _second, group = geometry()
  opaque = I.Assign("opaque", I.Expr(I.ScalarType("int64"), "raw", data="mutate(i)"))
  assert not collect_partition_proofs(kernel(root, loop(I.Block([opaque, group]))))


def test_only_required_validated_abi_dimensions_supply_entry_facts():
  from vecops.language.expr import Dynamic
  from vecops.language.types import TensorType

  raw = I.Expr(I.ScalarType("int64"), "raw", data="B.size(0)")
  root, first, second, group = geometry(row=raw)
  ir = kernel(root, group)
  annotation = TensorType((Dynamic(lower=0, upper=7, name="K"),), (1,), "float32")
  ir.parameters = {"B": annotation}
  ir.bindings = {"K": raw}
  assert partition_key(group, first, second) in collect_partition_proofs(ir)
  ir.parameters = {"B": replace(annotation, optional=True)}
  assert not collect_partition_proofs(ir)


def test_provenance_alone_does_not_override_actual_address_mapping():
  root, first, _second, group = geometry()
  wrong = replace(group.stores[1], destination=replace(group.stores[1].destination, offset=first.offset))
  assert not collect_partition_proofs(kernel(root, loop(I.Block([I.ElementwiseGroup((group.stores[0], wrong))]))))


def test_partition_verifier_rejects_detached_or_incomplete_certificates():
  root, first, second, group = geometry()
  ir = kernel(root, loop(I.Block([group])))
  proofs = collect_partition_proofs(ir)
  prepared = SimpleNamespace(vectors=SimpleNamespace(kernel=ir), partition_proofs=proofs)
  verify_partition_proofs(prepared)
  key = partition_key(group, first, second)
  prepared.partition_proofs = {key: replace(proofs[key], occurrences=2)}
  with pytest.raises(Exception, match="lexical occurrence"):
    verify_partition_proofs(prepared)
  prepared.partition_proofs = proofs
  ir.body.statements[1] = loop(I.Block([group]), 15, 16)
  with pytest.raises(Exception, match="final lexical domains"):
    verify_partition_proofs(prepared)


def test_allocation_metadata_is_not_reinterpreted_after_a_dependency_write():
  root, first, second, group = geometry()
  count = I.Expr(I.ScalarType("int64"), "local", data="count")
  changed_root = replace(root, shape=(count, root.shape[1]))
  stores = tuple(
    replace(
      store,
      destination=replace(store.destination, region=replace(store.destination.region, root_shape=changed_root.shape)),
    )
    for store in group.stores
  )
  changed = I.ElementwiseGroup(stores)
  ir = I.KernelIR(
    I.Block([I.Assign("count", I.literal(16)), I.Allocate(changed_root), loop(I.Block([changed]))]), {}, {}
  )
  assert collect_partition_proofs(ir)
  ir.body.statements.insert(2, I.Assign("count", I.literal(32), False))
  assert not collect_partition_proofs(ir)


def test_bind_tile_retains_history_without_rebinding_the_allocation():
  root, first, second, group = geometry()
  ir = kernel(root, I.BindTile(group.stores[0].value), loop(I.Block([group])))
  assert partition_key(group, first, second) in collect_partition_proofs(ir)


def test_verifier_rejects_cyclic_lexical_scope():
  body = I.Block([])
  body.statements.append(loop(body))
  prepared = SimpleNamespace(vectors=SimpleNamespace(kernel=I.KernelIR(body, {}, {})), partition_proofs={})
  with pytest.raises(Exception, match=r"cyclic.*CPU partition"):
    verify_partition_proofs(prepared)


def test_custom_catalog_cannot_promote_trapping_metadata_into_a_proof():
  from vecops.language._compiler.operations import snapshot, use_catalog
  from vecops.language._compiler.operations.registry import OperationCatalog

  root, _first, _second, group = geometry()
  definitions = dict(snapshot().definitions)
  definitions["add"] = replace(definitions["add"], may_trap=True, speculatable=False)
  with use_catalog(OperationCatalog(definitions, "partition-trapping-add")):
    assert not collect_partition_proofs(kernel(root, loop(I.Block([group]))))


def external_kernel(group, *, optional=False, dtype="float32", statements=()):
  from vecops.language.expr import Dynamic
  from vecops.language.types import TensorType

  stores = tuple(replace(store, destination=replace(store.destination, owned=False)) for store in group.stores)
  external = I.ElementwiseGroup(stores)
  annotation = TensorType((Dynamic(lower=0, name="P"),), (1,), dtype, optional, "output")
  ir = I.KernelIR(I.Block([*statements, loop(I.Block([external]))]), {"scratch": annotation}, {})
  return ir, external


def test_required_abi_same_pointer_spans_do_not_infer_reshape_capacity():
  _root, _first, _second, group = geometry()
  # A rank-one ABI buffer can back rank-two logical packing regions. The
  # certificate proves their actual addresses differ, not that P >= 128.
  ir, external = external_kernel(group)
  original = list(ir.body.statements)
  proofs = collect_partition_proofs(ir)
  assert partition_key(external, *(store.destination for store in external.stores)) in proofs
  assert ir.body.statements == original
  assert not any(store.destination.owned for store in external.stores)


@pytest.mark.parametrize("optional,dtype", [(True, "float32"), (False, "int32")])
def test_optional_or_different_dtype_abi_pointer_has_no_certificate(optional, dtype):
  _root, _first, _second, group = geometry()
  ir, _group = external_kernel(group, optional=optional, dtype=dtype)
  assert not collect_partition_proofs(ir)


@pytest.mark.parametrize(
  "offset,stride,width,expected",
  [
    (1, 1, 4, False),  # overlapping addresses
    (1, 2, 4, False),  # distinct interleaved elements, overlapping bounding spans
    (8, 2, 4, True),  # separated noncontiguous spans are still sufficient
    (8, -1, 4, False),
    ((1 << 63) - 1, 1, 4, False),
  ],
)
def test_actual_abi_address_spans_are_conservative(offset, stride, width, expected):
  _root, first, second, group = geometry(width=width)
  first = replace(first, offset=I.literal(0), strides=(I.literal(stride),))
  second = replace(second, offset=I.literal(offset), strides=(I.literal(stride),))
  group = I.ElementwiseGroup(
    (replace(group.stores[0], destination=first), replace(group.stores[1], destination=second))
  )
  ir, _external = external_kernel(group)
  assert bool(collect_partition_proofs(ir)) is expected


@pytest.mark.parametrize("mutation", ["allocate", "scalar"])
def test_abi_storage_shadowing_invalidates_entry_pointer_eligibility(mutation):
  root, _first, _second, group = geometry()
  statement = I.Allocate(root) if mutation == "allocate" else I.Assign("scratch", I.literal(0))
  ir, _external = external_kernel(group, statements=(statement,))
  assert not collect_partition_proofs(ir)


def test_shared_abi_group_intersects_pointer_eligibility_at_each_occurrence():
  _root, _first, _second, group = geometry()
  ir, _external = external_kernel(group)
  shared = ir.body.statements[0]
  ir.body.statements.extend((I.Assign("scratch", I.literal(0)), shared))
  assert not collect_partition_proofs(ir)


def test_different_abi_arguments_may_alias_and_unknown_extent_keeps_guard():
  _root, first, second, group = geometry()
  changed = replace(group.stores[1], destination=replace(second, storage="other"))
  ir, _external = external_kernel(I.ElementwiseGroup((group.stores[0], changed)))
  ir.parameters["other"] = ir.parameters["scratch"]
  assert not collect_partition_proofs(ir)
  unknown = I.Expr(I.ScalarType("int64"), "raw", data="scratch.size(0)")
  changed = replace(group.stores[0], destination=replace(first, shape=(unknown,)))
  ir, _external = external_kernel(I.ElementwiseGroup((changed, group.stores[1])))
  assert not collect_partition_proofs(ir)


@pytest.mark.parametrize(
  "metadata", [I.Block([]), object(), {"captured": I.Expr(I.ScalarType("int64"), "raw", data="mutate()")}]
)
def test_opaque_or_effectful_owned_offset_metadata_cannot_prove_abi_separation(metadata):
  _root, first, _second, group = geometry()
  offset = I.Expr(I.ScalarType("int64"), "add", (I.literal(0), I.literal(0)), metadata)
  changed = replace(group.stores[0], destination=replace(first, offset=offset))
  ir, _external = external_kernel(I.ElementwiseGroup((changed, group.stores[1])))
  assert not collect_partition_proofs(ir)


def test_expression_subclasses_and_argument_cycles_are_not_geometry_proofs():
  from vecops.language._compiler.backends.cpu.partitions import _Scope

  class ExtendedExpr(I.Expr):
    pass

  value = ExtendedExpr(I.ScalarType("int64"), "literal", data=1)
  assert not _Scope({}).safe(value)
  value = I.Expr(I.ScalarType("int64"), "add", (I.literal(0), I.literal(1)))
  object.__setattr__(value, "args", (I.literal(0), value))
  assert not _Scope({}).safe(value)


def test_shared_walker_observes_prewrite_scope_after_opaque_invalidation():
  from vecops.language._compiler.backends.cpu.partitions import walk_lexical_scopes

  x = I.Expr(I.ScalarType("int64"), "local", data="x")
  ir = I.KernelIR(
    I.Block(
      [
        I.Assign("x", I.literal(3)),
        I.Assign("copy", x),
        I.Assign("opaque", I.Expr(I.ScalarType("int64"), "raw", data="mutate()")),
        I.Assign("after", x),
      ]
    ),
    {},
    {},
  )
  seen = []
  walk_lexical_scopes(ir, lambda statement, scope: seen.append((statement.target, scope.range(x))))
  assert seen == [("x", None), ("copy", (3, 3)), ("opaque", None), ("after", None)]
