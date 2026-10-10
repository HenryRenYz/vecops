# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fusion is a DAG/cost decision, including inlined helper and packing paths."""

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.cost import estimate
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import nodes, walk


@T.func
def affine(x):
  return x * 2 + 1


@T.func
def expensive_tail(x):
  return T.exp(affine(x))


def plans(template, policy="auto", **options):
  options = {"cpu_matmul_fusion": policy, **options}
  return CPUBackend().lower(template.ir, (), target="AVX512", options=options)


def caches(artifact):
  return [
    statement
    for statement in nodes(artifact.kernel.body)
    if isinstance(statement, I.Store) and statement.destination.storage.endswith("_input_cache")
  ]


@pytest.mark.parametrize(
  "columns,expensive,materialized",
  [(1, False, False), (1, True, False), (64, False, False), (64, True, True), (4096, False, True)],
)
def test_reload_cost_across_helper_calls(columns, expensive, materialized):
  @T.template
  def kernel(
    A: T.Tensor((17, 33), dtype="float32"),
    B: T.Tensor((33, columns), dtype="float32"),
    C: T.Tensor((17, columns), dtype="float32", access="output"),
  ):
    a = expensive_tail(T.load(A)) if expensive else affine(T.load(A))
    T.dot((a, B), C, implementation="vector")

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  assert not artifact.backend_ir.matmuls
  cached = [
    statement
    for statement in nodes(artifact.kernel.body)
    if isinstance(statement, I.Store) and statement.destination.storage.endswith("_0_input_cache")
  ]
  assert bool(cached) == materialized
  original = lower(kernel.ir)
  product = next(
    value
    for statement in nodes(original.body)
    if isinstance(statement, I.Store)
    for value in walk(statement.value)
    if value.op == "dot"
  )
  cost = estimate(product.args[0], artifact.lowered.manager.options["_compiler_environment"].costs).compute_cycles
  assert cost >= (16 if expensive else 2)


def test_expensive_packed_b_tail_is_materialized_before_cache_traversal():
  @T.template
  def kernel(
    A: T.Tensor((4096, 33), dtype="float32"),
    B: T.Tensor((33, 65), dtype="float32"),
    C: T.Tensor((4096, 65), dtype="float32", access="output"),
  ):
    T.library.matmul((A, expensive_tail(T.load(B))), C, packing="B", BM=16, BN=32)

  artifact = plans(kernel)
  assert any(statement.destination.storage.endswith("_1_input_cache") for statement in caches(artifact))
  assert not artifact.backend_ir.matmuls
  assert "packed-layout recipe" in "\n".join(artifact.kernel.diagnostics)
  assert not caches(plans(kernel, "always"))
  assert caches(plans(kernel, "never"))


def test_dynamic_reload_cost_remains_runtime_predicate():
  N = T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((17, 33), dtype="float32"),
    B: T.Tensor((33, N), dtype="float32"),
    C: T.Tensor((17, N), dtype="float32", access="output"),
  ):
    T.dot((expensive_tail(T.load(A)), B), C)

  artifact = plans(kernel)
  assert caches(artifact)
  conditions = [
    statement.value
    for statement in nodes(artifact.kernel.body)
    if isinstance(statement, I.Assign) and statement.target.endswith("_cache_enabled")
  ]
  assert conditions and all(I.constant(value) is None for value in conditions)
  assert not caches(plans(kernel, "always"))
  for invalid in (0, 9, True):
    with pytest.raises(ValueError, match="cpu_gemv_k_unroll"):
      plans(kernel, cpu_gemv_k_unroll=invalid)


def test_load_conversion_participates_in_recompute_cost():
  @T.template
  def kernel(
    A: T.Tensor((17, 33), dtype="float32"),
    B: T.Tensor((33, 4096), dtype="bfloat16"),
    C: T.Tensor((17, 4096), dtype="float32", access="output"),
  ):
    T.dot((T.load(A, dtype="bfloat16"), B), C)

  artifact = plans(kernel)
  assert [statement.destination.storage.rsplit("_", 3)[-3] for statement in caches(artifact)] == ["0"]
  assert "vecops::vec::load_convert(" in artifact.source


def test_layernorm_statistics_not_recomputed_in_dot_operand():
  @T.template
  def kernel(
    A: T.Tensor((17, 33), dtype="float32"),
    B: T.Tensor((33, 65), dtype="float32"),
    C: T.Tensor((17, 65), dtype="float32", access="output"),
  ):
    a = affine(T.library.layernorm(T.load(A)))
    T.library.matmul((a, B), C, packing="A")

  artifact = plans(kernel, "never")
  assert not artifact.backend_ir.matmuls
  cached = caches(artifact)
  assert cached
  assert any(e.op in {"mul", "add", "sub"} for e in walk(cached[0].value))
  assert not any(e.op == "normalize" for e in walk(cached[0].value))


def test_dynamic_packing_bypass_uses_gemv_recompute_policy():
  M, N, K = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="float32"),
    B: T.Tensor((K, N), dtype="float32"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    T.library.matmul((expensive_tail(T.load(A)), expensive_tail(T.load(B))), C)

  artifact = plans(kernel)
  original = lower(kernel.ir)
  product = next(
    value
    for statement in nodes(original.body)
    if isinstance(statement, I.Store)
    for value in walk(statement.value)
    if value.op == "dot"
  )

  def evaluate(e, m, n):
    for dimension, value in zip(product.type.shape, (m, n)):
      if I.same_expr(e, dimension):
        return value
    if e.op == "literal":
      return e.data
    if e.op == "where":
      return evaluate(e.args[1 if evaluate(e.args[0], m, n) else 2], m, n)
    a, b = (evaluate(arg, m, n) for arg in e.args)
    return {"and": lambda: a and b, "gt": lambda: a > b, "add": lambda: a + b, "floordiv": lambda: a // b}[e.op]()

  conditions = [
    statement.value
    for statement in nodes(artifact.kernel.body)
    if isinstance(statement, I.Assign) and statement.target.endswith("_cache_enabled")
  ]
  assert len(conditions) == 2
  a, b = conditions
  assert evaluate(a, 1, 256) is True
  assert evaluate(b, 1, 256) is False
  assert evaluate(a, 256, 1) is False
  assert evaluate(b, 256, 1) is True
