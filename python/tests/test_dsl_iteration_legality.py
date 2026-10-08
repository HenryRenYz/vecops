# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Small legality regressions; equal domains alone never authorize fusion."""

from dataclasses import replace

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.loops import fusion_guard, nonoverlap_proven
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.iteration import FoldLoop, contract
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._compiler.operations import current


@pytest.mark.parametrize("stride", [1, -2, 0, None])
@pytest.mark.parametrize("permuted", [False, True])
def test_simd_scatter_proves_unique_induction_without_unchecked_hint(stride, permuted):
  step = T.dynamic() if stride is None else stride

  @T.template
  def kernel(A: T.Tensor((37,), dtype="float32"), C: T.Tensor((37,), (step,), "float32", access="output")):
    for i in T.simd(37, unroll=1):
      indices = i % 3 if permuted else i
      T.scatter(C, indices, T.gather(A, i))

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  indexed = any(isinstance(node, I.IndexedStore) for node in nodes(artifact.kernel.body))
  assert indexed == (permuted or stride in {0, None})
  if not indexed:
    stores = [node for node in nodes(artifact.kernel.body) if isinstance(node, I.Store)]
    assert stores and all(store.vector_extent is not None for store in stores)


def point(storage, target="i", stride=1):
  one, size = I.literal(1), I.literal(17)
  index = I.Expr(I.ScalarType("int64"), "local", data=target)
  region = I.ViewRegion((size,), (I.literal(stride),), (index,), (one,), (1,), (0,), (False,))
  return I.View(storage, "float32", (), (), I.binary("mul", index, I.literal(stride)), region=region)


def loop(source, destination, target="i"):
  read = I.Expr(I.ScalarType("float32"), "load", data=point(source, target))
  return I.For(target, I.literal(0), I.literal(17), I.literal(1), I.Block([I.Store(point(destination, target), read)]))


def test_layout_proof_and_alias_versioning():
  view = I.View("A", "float32", (I.literal(17),), (I.literal(-2),), I.literal(0))
  assert nonoverlap_proven(view)
  assert not nonoverlap_proven(replace(view, strides=(I.literal(0),)))
  overlapping = replace(view, shape=(I.literal(3), I.literal(3)), strides=(I.literal(2), I.literal(1)))
  assert not nonoverlap_proven(overlapping)
  result = fusion_guard(loop("A", "B"), loop("B", "C", "j"), current())
  assert result is not None
  assert len(result[1]) == 2 and all(e.op == "view_disjoint" for e in result[1])


def test_mutable_bound_and_barriers_refuse_fusion():
  left, right = loop("A", "B"), loop("B", "C", "j")
  left.body.statements.append(I.Assign("bound", I.literal(1), declare=False))
  right = replace(right, stop=I.Expr(I.ScalarType("int64"), "local", data="bound"))
  assert fusion_guard(left, right, current()) is None
  blocked = loop("A", "B")
  blocked.body.statements.append(I.Control("break"))
  assert fusion_guard(blocked, loop("B", "C", "j"), current()) is None


def test_ordered_recurrence_contract_and_live_memory_reductions():
  @T.reducer(identity=0.0, associative=False)
  def sum_(a, b):
    return a + b

  @T.template
  def kernel(A: T.Tensor((17,), dtype="float32", access="inout"), C: T.Tensor((3,), dtype="float32", access="output")):
    C[0] = T.reduce(T.load(A), reducer=sum_)
    C[1] = T.reduce_add(T.load(A))
    T.store(A, T.load(A) * 2)
    C[2] = T.reduce_add(T.load(A))

  artifact = CPUBackend().lower(kernel.ir, ())
  loops = [n for n in nodes(artifact.kernel.body) if isinstance(n, I.For)]
  assert any(contract(n).kind == "reduce" and contract(n).ordered for n in loops)
  folds = [n for n in nodes(artifact.kernel.body) if isinstance(n, FoldLoop)]
  assert len(folds) == 2


def test_associative_scan_is_still_ordered():
  @T.reducer(identity=0.0, associative=True)
  def sum_(a, b):
    return a + b

  @T.template
  def kernel(A: T.Tensor((17,), dtype="float32"), C: T.Tensor((17,), dtype="float32", access="output")):
    T.store(C, T.scan(T.load(A), reducer=sum_) * 2)

  artifact = CPUBackend().lower(kernel.ir, ())
  loops = [n for n in nodes(artifact.kernel.body) if isinstance(n, I.For) and n.kind == "scan"]
  assert loops and all(contract(n).ordered and contract(n).reducers for n in loops)
