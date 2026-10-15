# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Private SME A panels retain full-K geometry across N consumers."""

import pytest
import vecops.language as T
from vecops._arm_vector_state import VectorState
from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.affine import expand_definitions
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.dtypes import ELEMENT_BYTES
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import children, nodes
from vecops.language._debug.interpreter import Interpreter
from vecops.language._library.sme_matmul import dot_into, prepare_a, prepare_b


@pytest.mark.parametrize("dtype,acc_dtype,k", [("bfloat16", "float32", 19), ("int8", "int32", 65),
                                               ("float32", "float32", 17), ("bfloat16", "float32", 0)])
def test_prepared_a_has_chunk_padded_private_shape(dtype, acc_dtype, k):
  @T.template
  def kernel(A: T.Tensor((17, k), dtype=dtype)):
    prepare_a(T.load(A), acc_dtype, raw=A)

  program = lower(kernel.ir)
  definitions = {s.target: s.value for s in nodes(program.body) if isinstance(s, I.Assign) and s.declare}
  allocations = [s.view for s in nodes(program.body) if isinstance(s, I.Allocate)]
  packed = next(v for v in allocations if len(v.shape) == 3)
  interpreter = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
  original = interpreter.value

  def value(expression):
    if expression.op == "streaming_lanes":
      return 64 // ELEMENT_BYTES[expression.data]
    return original(expression)

  interpreter.value = value
  shape = tuple(value(expand_definitions(extent, definitions)) for extent in packed.shape)
  width, chunk = 64 // ELEMENT_BYTES[acc_dtype], 64 // ELEMENT_BYTES[dtype]
  group = ELEMENT_BYTES[acc_dtype] // ELEMENT_BYTES[dtype]
  assert shape == ((17 + 2 * width - 1) // (2 * width), (k + chunk - 1) // chunk * (chunk // group), 2 * width * group)


@pytest.mark.parametrize("both", [False, True])
@pytest.mark.parametrize("transformed", [False, True])
def test_prepared_a_is_created_before_n_loop_and_consumed_without_repair(both, transformed):
  @T.template
  def kernel(A: T.Tensor((17, 19), (-41, -2), "bfloat16"),
             B: T.Tensor((19, 35), dtype="bfloat16"),
             C: T.Tensor((17, 35), dtype="float32", access="output")):
    a = T.cast(T.load(A, dtype="float32") * 0.5, "bfloat16") if transformed else T.load(A)
    left = prepare_a(a, "float32", raw=None if transformed else A)
    for ni in T.serial(0, 35, 17):
      right = B[:, ni:ni + 17]
      if both:
        prepared_right = prepare_b(T.load(right), "float32", raw=right)
        dot_into((a, T.load(right)), C[:, ni:ni + 17], prepared_a=left, prepared_b=prepared_right)
      else:
        dot_into((a, T.load(right)), C[:, ni:ni + 17], prepared_a=left, raw_b=right)

  program = lower(kernel.ir)
  allocations = []

  def visit(block, loops=()):
    for statement in block.statements:
      if isinstance(statement, I.Allocate) and len(statement.view.shape) == 3:
        allocations.append((statement.view.storage, loops))
      for child in children(statement):
        visit(child, loops + ((statement.target,) if isinstance(statement, I.For) else ()))

  visit(program.body)
  assert sum(not scope for _, scope in allocations) == 1
  assert any(scope for _, scope in allocations), "B must remain local to its N panel"
  artifact = CPUBackend().lower(kernel.ir, (), target="SME", options={"_vector_state": VectorState(64, 64)})
  assert "vecops::vec::sme::mopa<" in artifact.source
  assert "packed input buffer too small" not in artifact.source


def test_prepared_a_rejects_incompatible_accumulator_layout():
  @T.template
  def kernel(A: T.Tensor((17, 19), dtype="int16")):
    prepare_a(T.load(A), "int64")

  with pytest.raises(Exception, match="static_assert"):
    lower(kernel.ir)
