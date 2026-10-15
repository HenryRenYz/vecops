# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Complete private input capture permits whole-K output without partial."""

import numpy as np
import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import lower
from vecops.language._debug.interpreter import Interpreter
from vecops.language._library.matrix_recipe import _whole_k_matrix


@T.func
def capture(value, acc_dtype, raw=None):
  saved = T.alloc(value.shape, dtype=value.dtype)
  for row in T.serial(value.shape[0]):
    T.store(saved[row, :], value[row, :])
  return saved


@T.func
def scalar_leaf(inputs, output, *, acc=None, acc_dtype="float32", prepared_a=None, prepared_b=None):
  T.store(output, T.zeros(output.shape, dtype=acc_dtype) if acc is None else acc)
  for mi in T.serial(output.shape[0]):
    for ki in T.serial(inputs[0].shape[1]):
      T.store(
        output[mi, :], T.load(output[mi, :]) + T.element_at(T.load(prepared_a), (mi, ki)) * T.load(prepared_b)[ki, :]
      )


@T.func
def finish(value, row, column):
  return value * 2 + 1


def template(m, n, k, panel, seeded, epilogue):
  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), "float32"),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), "float32"),
    Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), "float32"),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), "float32", access="output"),
  ):
    _whole_k_matrix(
      (T.load(A) * 2 + 1, T.load(B) - 1),
      C,
      scalar_leaf,
      capture,
      capture,
      panel,
      acc=T.load(Prior) if seeded else None,
      acc_dtype="float32",
      epilogue=finish if epilogue else None,
      raw_a=None,
      raw_b=None,
    )

  return lower(kernel.ir)


class AllocationInterpreter(Interpreter):
  def __init__(self, kernel, arguments):
    self.allocations = []
    super().__init__(kernel, arguments)

  def value(self, expression):
    if isinstance(expression, I.Expr) and expression.op == "value_transform" and expression.data[0] == "broadcast_to":
      return np.broadcast_to(self.value(expression.args[0]), tuple(int(self.value(v)) for v in expression.type.shape))
    if isinstance(expression, I.Expr) and expression.op == "broadcast_dim":
      a, b = (self.value(v) for v in expression.args)
      if a != b and a != 1 and b != 1:
        raise ValueError("incompatible broadcast dimensions")
      return b if a == 1 else a
    return super().value(expression)

  def block(self, block):
    for statement in block.statements:
      if isinstance(statement, I.Allocate):
        self.allocations.append(tuple(int(self.value(v)) for v in statement.view.shape))
      super().block(I.Block([statement]))


@pytest.mark.parametrize("alias", [None, "A", "B"])
@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("panel,epilogue", [(5, False), (2, False), (5, True)])
def test_captured_output_matches_independent_oracle_with_alias_and_tail(alias, seeded, reverse, panel, epilogue):
  m, n, k = 3, 5, 7
  backing = np.full((9, 17), -111, np.float32)
  c = backing[1 : 1 + m, 2 : 2 + 2 * n : 2]
  a = backing[1 : 1 + m, 2 : 2 + 2 * k : 2] if alias == "A" else np.arange(m * k, dtype=np.float32).reshape(m, k) % 5
  b = backing[1 : 1 + k, 2 : 2 + 2 * n : 2] if alias == "B" else np.arange(k * n, dtype=np.float32).reshape(k, n) % 3
  a[:] = np.arange(m * k).reshape(m, k) % 5 - 2
  b[:] = np.arange(k * n).reshape(k, n) % 3 - 1
  if reverse:
    a, b, c = a[::-1, ::-1], b[::-1, ::-1], c[::-1, ::-1]
  prior = c[:1]
  prior[:] = np.arange(n) % 4
  expected = (a.copy().astype(np.int64) * 2 + 1) @ (b.copy().astype(np.int64) - 1)
  if seeded:
    expected += prior.copy().astype(np.int64)
  if epilogue:
    expected = expected * 2 + 1
  before = backing.copy()
  mask = np.ones(backing.shape, bool)
  mask[1 : 1 + m, 2 : 2 + 2 * n : 2] = False
  program = template(m, n, k, panel, seeded, epilogue)
  runner = AllocationInterpreter(program, {"A": a, "B": b, "Prior": prior, "C": c})
  runner.block(program.body)
  np.testing.assert_array_equal(c, expected.astype(np.float32))
  np.testing.assert_array_equal(backing[mask], before[mask])
  # Only direct unseeded output eliminates the whole result allocation;
  # seeded direct output has one initializer snapshot, never a second partial.
  assert runner.allocations.count((m, n)) == int(seeded or panel < n or epilogue)


@pytest.mark.parametrize("m,n,k", [(3, 5, 0), (0, 5, 7), (3, 0, 7)])
@pytest.mark.parametrize("seeded", [False, True])
def test_captured_output_empty_dimensions_and_zero_k(m, n, k, seeded):
  a, b = np.zeros((m, k), np.float32), np.zeros((k, n), np.float32)
  prior = np.arange(n, dtype=np.float32).reshape(1, n)
  output = np.full((m, n), -111, np.float32)
  program = template(m, n, k, 5, seeded, False)
  runner = AllocationInterpreter(program, {"A": a, "B": b, "Prior": prior, "C": output})
  runner.block(program.body)
  expected = np.broadcast_to(prior if seeded else np.zeros_like(prior), output.shape)
  np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize(
  "panel,seeded,epilogue,expected_partial",
  [(35, False, False, 0), (35, True, False, 1), (17, False, False, 1), (35, False, True, 1)],
)
def test_sme_whole_capture_removes_result_partial_only_on_safe_branch(panel, seeded, epilogue, expected_partial):
  from vecops._arm_vector_state import VectorState
  from vecops.language._compiler.backends.cpu.backend import CPUBackend
  from vecops.language._compiler.ir.traversal import nodes
  from vecops.language._library.sme_matmul import dot_into, prepare_a, prepare_b

  @T.template
  def kernel(
    A: T.Tensor((17, 19), dtype="bfloat16"),
    B: T.Tensor((19, 35), dtype="bfloat16"),
    Prior: T.Tensor((1, 35), dtype="float32"),
    C: T.Tensor((17, 35), dtype="float32", access="output"),
  ):
    _whole_k_matrix(
      (T.load(A), T.load(B)),
      C,
      dot_into,
      prepare_a,
      prepare_b,
      panel,
      acc=T.load(Prior) if seeded else None,
      acc_dtype="float32",
      epilogue=finish if epilogue else None,
      raw_a=A,
      raw_b=B,
    )

  program = lower(kernel.ir)
  result_scratch = [
    s
    for s in nodes(program.body)
    if isinstance(s, I.Allocate)
    and s.view.dtype == "float32"
    and tuple(I.constant(v) for v in s.view.shape) == (17, 35)
  ]
  assert len(result_scratch) == expected_partial
  artifact = CPUBackend().lower(kernel.ir, (), target="SME", options={"_vector_state": VectorState(64, 64)})
  assert "vecops::vec::sme::mopa<" in artifact.source


def test_dynamic_whole_capture_keeps_branch_private_views_out_of_scalar_joins():
  from test_dsl_matrix_captured_output_native import template as native_template
  from vecops.language._compiler.ir.traversal import nodes

  program = lower(native_template("bfloat16", "bfloat16", "float32").ir)
  assert any(isinstance(statement, I.If) for statement in nodes(program.body))
