# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Dense B cache panels are constructed outside their M reuse loop."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import children
from vecops.language._library.matmul import vector_dot_into
from vecops.language._library.matrix_recipe import blocked_matrix


@T.func
def _vector_leaf(
  inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, raw_a=None, raw_b=None, packed_a=None, packed_b=None
):
  vector_dot_into(inputs, output, acc=acc, acc_dtype=acc_dtype, epilogue=epilogue)


def template(packing, *, dynamic=False):
  m, n, k = (T.dynamic(lower=0) for _ in range(3)) if dynamic else (11, 7, 9)

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype="float32"),
    B: T.Tensor((k, n), dtype="float32"),
    D: T.Tensor((1, n), dtype="float32"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    a = T.load(A) * 0.5 - 1
    b = T.load(B) * 0.25 + 1
    blocked_matrix((a, b), C, _vector_leaf, acc=T.load(D), BM=4, BN=3, BK=5, packing=packing)

  return kernel


@pytest.mark.parametrize("packing", ("none", "A", "B", "AB", "auto"))
def test_dense_b_pack_is_outside_m_loop_and_a_remains_inside(packing):
  kernel = lower(template(packing).ir)
  packs = []

  def visit(block, loops=()):
    for statement in block.statements:
      if isinstance(statement, I.Pack):
        packs.append((statement.destination.side, loops, statement.destination.format))
      for child in children(statement):
        visit(child, loops + ((statement.target,) if isinstance(statement, I.For) else ()))

  visit(kernel.body)
  assert all(format_ == "dense" for _, _, format_ in packs)
  expected = ({"A"} if packing in {"A", "AB"} else set()) | ({"B"} if packing in {"B", "AB"} else set())
  assert {side for side, _, _ in packs} == expected
  for side, loops, _ in packs:
    assert len(loops) == (2 if side == "B" else 3)
    assert all("mi" not in name for name in loops) if side == "B" else "mi" in loops[-1]


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("matrix-b-panel-reuse")
  return make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build")


@pytest.mark.parametrize("packing", ("B", "AB"))
def test_dense_b_panel_reuse_preserves_transforms_alias_and_k_zero(compiler, packing):
  compiled = T.compile(template(packing, dynamic=True), compiler=compiler)
  rng = np.random.default_rng(207)
  for m, n, k in ((11, 7, 9), (11, 7, 0), (0, 7, 9), (11, 0, 9)):
    a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
    # Nonempty B and the broadcast initializer alias output. All input and
    # initializer values must survive until the completed K phases finish.
    backing = rng.uniform(-1, 1, (max(m, k, 1), n)).astype(np.float32)
    b, c, d = backing[:k], backing[:m], backing[:1]
    expected = (a * 0.5 - 1) @ (b.copy() * 0.25 + 1) + d.copy()
    compiled.operator.native(a, b, d, c, **compiled._core.specialization_values)
    np.testing.assert_allclose(c, expected, rtol=2e-5, atol=2e-5)


@T.func
def _probe_leaf(
  inputs, output, *, acc=None, acc_dtype="float32", epilogue=None, raw_a=None, raw_b=None, packed_a=None, packed_b=None
):
  T.store(output, T.full(output.shape, 1, dtype=acc_dtype))


@T.func
def _sme_probe_leaf(
  inputs,
  output,
  *,
  acc=None,
  acc_dtype="float32",
  epilogue=None,
  raw_a=None,
  raw_b=None,
  packed_a=None,
  packed_b=None,
  broadcast_acc=False,
  pack_raw_inputs=False,
):
  T.static_assert(pack_raw_inputs)
  _probe_leaf(
    inputs,
    output,
    acc=acc,
    acc_dtype=acc_dtype,
    epilogue=epilogue,
    raw_a=raw_a,
    raw_b=raw_b,
    packed_a=packed_a,
    packed_b=packed_b,
  )


def packed_template(*, configured=False, packing="auto", sme=False):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))
  leaf = _sme_probe_leaf if sme else _probe_leaf

  @T.template
  def kernel(
    AP: T.Tensor((4096,), dtype="float32"),
    BP: T.Tensor((4096,), dtype="float32"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    a = T.packed(AP, (m, k), side="A", format="sme")
    b = T.packed(BP, (k, n), side="B", format="sme")
    blocked_matrix(
      (a, b), C, leaf, packed_a=AP, packed_b=BP, BM=4 if configured else None, packing=packing, pack_raw_inputs=sme
    )

  return kernel


@pytest.mark.parametrize("sme", (False, True))
def test_unconfigured_external_packed_pair_is_statically_whole_problem(sme):
  from vecops.language._compiler.ir.traversal import nodes

  kernel = lower(packed_template(sme=sme).ir)
  assert not any(isinstance(node, (I.Allocate, I.For)) for node in nodes(kernel.body))


@pytest.mark.parametrize("configured,packing", ((True, "auto"), (False, "none"), (False, "B")))
def test_external_packed_pair_keeps_explicit_cache_policy(configured, packing):
  from vecops.language._compiler.ir.traversal import nodes

  kernel = lower(packed_template(configured=configured, packing=packing, sme=True).ir)
  statements = tuple(nodes(kernel.body))
  assert any(isinstance(node, I.Allocate) for node in statements)
  assert any(isinstance(node, I.For) for node in statements)
