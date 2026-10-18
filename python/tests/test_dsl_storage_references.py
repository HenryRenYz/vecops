# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""One final-DAG storage query replaces repeated allocation-time walks."""

import re
from dataclasses import replace

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu import storage
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter
from vecops.language._compiler.backends.cpu.verification import verify_prepared
from vecops.language._compiler.environment import freeze


def view(name, dtype="float32"):
  return I.View(name, dtype, (I.literal(17),), (I.literal(1),), I.literal(0), owned=True)


def load(name, dtype="float32"):
  value = view(name, dtype)
  return I.Expr(I.TileType(dtype, value.shape), "load", data=value)


def query(block):
  return storage.referenced_storage(block, freeze(target="Scalar").operations)


def test_declarations_are_dead_but_historical_reads_keep_their_provenance():
  block = I.Block([I.Allocate(view("unused")), I.BindTile(load("historical"))])
  assert query(block) == frozenset({"historical"})
  block.statements.clear()
  assert query(block) == frozenset()


def test_view_payloads_shapes_and_packed_operands_are_uses():
  packed = I.PackedView(view("packed"), (I.literal(17),), "A", "test-format")
  metadata = I.Expr(I.ScalarType("int64"), "shape_proof", data=view("metadata"))
  expression = I.Expr(I.TileType("float32", (metadata,)), "payload", data=(packed,))
  block = I.Block([I.Store(view("output"), expression)])
  assert query(block) == frozenset({"packed", "metadata", "output"})


def test_structured_regions_masks_indices_and_architecture_views_are_retained():
  grouped = I.ElementwiseGroup((I.Store(view("one"), load("A")), I.Store(view("two"), load("B"))))
  fold = I.ReductionGroup((I.Assign("sum", I.Expr(I.ScalarType("float32"), "reduce_add", (load("C"),), 2)),))
  indexed = I.IndexedStore(view("indexed"), load("indices", "int64"), load("values"), load("mask", "bool"))
  loop = I.For("i", I.literal(0), I.literal(2), I.literal(1), I.Block([fold]))
  unrolled = I.UnrolledFor(loop, I.Block([grouped]))
  condition = I.Expr(I.ScalarType("int64"), "load", data=view("condition", "int64"))
  block = I.Block(
    [I.If(condition, I.Block([unrolled, indexed]), I.Block([I.ArchitectureCall("read", view=view("architecture"))]))]
  )
  assert query(block) == frozenset(
    {"one", "two", "A", "B", "C", "indexed", "indices", "values", "mask", "condition", "architecture"}
  )


def test_pack_compatibility_query_preserves_source_and_destination():
  packed = I.PackedView(view("packed"), (I.literal(17),), "A", "test-format")
  assert query(I.Block([I.Pack(load("source"), packed)])) == frozenset({"source", "packed"})


def test_shared_value_metadata_is_walked_once_across_all_statements(monkeypatch):
  shared = load("shared")
  for _ in range(20):
    shared = I.Expr(shared.type, "add", (shared, shared))
  block = I.Block([I.Store(view(f"out_{index}"), shared) for index in range(32)])
  original, visits = storage.nodes, []

  def observe(value, seen=None):
    for node in original(value, seen):
      visits.append(node)
      yield node

  monkeypatch.setattr(storage, "nodes", observe)
  result = query(block)
  assert result == frozenset({"shared", *(f"out_{index}" for index in range(32))})
  assert sum(isinstance(node, I.View) and node.storage == "shared" for node in visits) == 1
  assert len(visits) < 64


def test_planned_storage_is_verified_and_the_printer_never_rewalks_allocations(monkeypatch):
  from test_dsl_branch_local_buffers import template

  def forbidden(*args):
    raise AssertionError("source emission rewalked allocation uses")

  monkeypatch.setattr(MemoryEmitter, "used_storage", forbidden)
  artifact = CPUBackend().lower(template().ir, (), target="AVX512")
  prepared = artifact.backend_ir
  assert type(prepared.storage.referenced) is frozenset
  assert any("scratch" in name for name in prepared.storage.referenced)
  with pytest.raises(ValueError, match="stale referenced storage plan"):
    verify_prepared(replace(prepared, storage=replace(prepared.storage, referenced=frozenset())))
  with pytest.raises(ValueError, match="invalid referenced storage plan"):
    verify_prepared(replace(prepared, storage=replace(prepared.storage, referenced=set(prepared.storage.referenced))))


def test_unused_promoted_scratch_still_has_no_buffer():
  @T.template
  def kernel(A: T.Tensor((17,), dtype="float32"), O: T.Tensor((17,), dtype="float32", access="output")):
    _unused = T.alloc((17,), dtype="float32")
    T.store(O, T.load(A) + 1)

  artifact = CPUBackend().lower(kernel.ir, (), target="Scalar")
  assert artifact.backend_ir.storage.referenced == frozenset({"A", "O"})
  assert re.search(r"Buffer<float> \w*unused", artifact.source) is None
