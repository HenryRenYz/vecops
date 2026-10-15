# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Whole-K scheduling probes without emulating matrix instructions or packing."""

import numpy as np
import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import children
from vecops.language._debug.interpreter import Interpreter
from vecops.language._library.matrix_recipe import _cache_matrix, _whole_k_matrix


@T.func
def prepare_a(value, acc_dtype, raw=None):
  result = T.alloc((1, 1), dtype="int16")
  T.store(result, 71)
  return result


@T.func
def prepare_b(value, acc_dtype, raw=None):
  result = T.alloc((1, 1), dtype="int8")
  T.store(result, 23)
  return result


@T.func
def leaf(
  inputs,
  output,
  *,
  acc=None,
  acc_dtype="float32",
  prepared_a=None,
  prepared_b=None,
  raw_a=None,
  raw_b=None,
  packed_a=None,
  packed_b=None,
  epilogue=None,
  broadcast_acc=False,
  pack_raw_inputs=False,
):
  value = T.full(output.shape, inputs[0].shape[1], dtype=acc_dtype)
  if acc is not None:
    value = value + acc
  T.store(output, value)


@T.func
def finish(value, row, column):
  return T.max(value * 2 - 1, 0)


def template(
  *,
  m=5,
  n=7,
  k=17,
  direct=False,
  configured=False,
  seeded=True,
  packed=False,
  dtype="float32",
  raw=False,
  strided=False,
):
  a_strides = (2 * k, 2) if strided else None
  b_strides = (2, 2 * k) if strided else (1, k) if raw else None

  @T.template
  def kernel(
    A: T.Tensor((m, k), a_strides, dtype),
    B: T.Tensor((k, n), b_strides, dtype),
    D: T.Tensor((1, n), dtype="float32"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    if direct:
      _whole_k_matrix(
        (T.load(A), T.load(B)),
        C,
        leaf,
        prepare_a,
        prepare_b,
        3,
        acc=T.load(D) if seeded else None,
        acc_dtype="float32",
        epilogue=finish,
        raw_a=A if raw else None,
        raw_b=B if raw else None,
      )
    else:
      _cache_matrix(
        (T.load(A), T.load(B)),
        C,
        leaf,
        acc=T.load(D) if seeded else None,
        acc_dtype="float32",
        epilogue=finish,
        BM=2 if configured else None,
        BN=3 if configured else None,
        BK=5 if configured else None,
        span=32,
        quantum=1,
        packing="auto",
        raw_a=A if raw else None,
        raw_b=B if raw else None,
        pack_raw_inputs=False,
        prepare_b=prepare_b,
        cache_raw_b=None,
        whole_b_panel=False,
        prepare_a=prepare_a,
        packed_pair=packed,
      )

  return kernel


class ProbeInterpreter(Interpreter):
  def __init__(self, kernel, arguments):
    self.a_preparations = self.b_preparations = 0
    super().__init__(kernel, arguments)

  def value(self, expression):
    if isinstance(expression, I.Expr):
      if expression.op == "value_transform" and expression.data[0] == "broadcast_to":
        return np.broadcast_to(self.value(expression.args[0]), tuple(int(self.value(v)) for v in expression.type.shape))
      if expression.op == "streaming_lanes":
        return 64 // {"float32": 4, "bfloat16": 2, "int16": 2, "int8": 1}[expression.data]
      if expression.op == "ceildiv":
        a, b = (self.value(v) for v in expression.args)
        return -(-a // b)
      if expression.op == "broadcast_dim":
        a, b = (self.value(v) for v in expression.args)
        if a != b and a != 1 and b != 1:
          raise ValueError("incompatible probe broadcast extents")
        return b if a == 1 else a
      if expression.op == "cache_bytes":
        return 32768 if expression.data == "L1" else 2097152
    return super().value(expression)

  def block(self, block):
    for statement in block.statements:
      if isinstance(statement, I.Allocate):
        self.a_preparations += statement.view.dtype == "int16"
        self.b_preparations += statement.view.dtype == "int8"
      super().block(I.Block([statement]))


def test_whole_k_structure_has_one_a_preparation_and_only_n_loop():
  kernel = lower(template(direct=True).ir)
  preparations, loops, partial_stores = [], [], []

  def visit(block, stack=()):
    for statement in block.statements:
      if isinstance(statement, I.Allocate) and statement.view.dtype in {"int16", "int8"}:
        preparations.append((statement.view.dtype, stack))
      if isinstance(statement, I.For):
        loops.append(statement)
      if isinstance(statement, I.Store) and statement.value.op == "fill":
        partial_stores.append(statement)
      for child in children(statement):
        visit(child, stack + ((statement.target,) if isinstance(statement, I.For) else ()))

  visit(kernel.body)
  assert [dtype for dtype, _ in preparations] == ["int16", "int8"]
  assert preparations[0][1] == ()
  assert len(preparations[1][1]) == 1 and "ni" in preparations[1][1][0]
  assert not any("ki" in loop.target or "mi" in loop.target for loop in loops)
  # The probe markers are written, but no zero-seeding Store is emitted.
  assert not any(statement.value.args and I.constant(statement.value.args[0]) == 0 for statement in partial_stores)


@pytest.mark.parametrize("seeded", (False, True))
def test_whole_k_debug_probe_broadcast_seed_epilogue_and_aliased_seed(seeded):
  kernel = lower(template(direct=True, seeded=seeded).ir)
  backing = np.arange(17 * 7, dtype=np.float32).reshape(17, 7)
  output, bias = backing[:5], backing[:1]
  expected = np.maximum((17 + (bias.copy() if seeded else 0)) * 2 - 1, 0)
  runner = ProbeInterpreter(kernel, {"A": np.zeros((5, 17), np.float32), "B": backing, "D": bias, "C": output})
  runner.block(kernel.body)
  np.testing.assert_array_equal(output, np.broadcast_to(expected, output.shape))
  assert runner.a_preparations == 1 and runner.b_preparations == 3


@pytest.mark.parametrize(
  "k,configured,packed,expected_a",
  ((17, False, False, 1), (0, False, False, 0), (17, True, False, 0), (17, False, True, 0), (8200, False, False, 0)),
)
def test_cache_selection_full_k_or_safe_phased_fallback(k, configured, packed, expected_a):
  # K=8200 rounds to 8208 float32 lanes: even one 32-column B panel
  # exceeds its 1 MiB budget; this exercises physical rather than logical K.
  kernel = lower(template(k=k, configured=configured, packed=packed).ir)
  bias = np.arange(7, dtype=np.float32).reshape(1, 7)
  output = np.empty((5, 7), np.float32)
  runner = ProbeInterpreter(
    kernel, {"A": np.zeros((5, k), np.float32), "B": np.zeros((k, 7), np.float32), "D": bias, "C": output}
  )
  runner.block(kernel.body)
  np.testing.assert_array_equal(output, np.broadcast_to(np.maximum((k + bias) * 2 - 1, 0), output.shape))
  assert runner.a_preparations == expected_a
  if k == 0:
    assert runner.b_preparations == 0


def test_physical_a_padding_budget_rejects_logically_fitting_input():
  m, n, k = 3841, 7, 257
  assert m * k * 4 < 4194304
  assert ((m + 31) // 32 * 32) * ((k + 15) // 16 * 16) * 4 > 4194304
  kernel = lower(template(m=m, n=n, k=k).ir)
  bias = np.arange(n, dtype=np.float32).reshape(1, n)
  output = np.empty((m, n), np.float32)
  runner = ProbeInterpreter(
    kernel, {"A": np.zeros((m, k), np.float32), "B": np.zeros((k, n), np.float32), "D": bias, "C": output}
  )
  runner.block(kernel.body)
  assert runner.a_preparations == 0
  np.testing.assert_array_equal(output, np.broadcast_to((k + bias) * 2 - 1, output.shape))


@pytest.mark.parametrize(
  "m,n,k,raw,strided,expected_b",
  (
    (128, 4096, 4096, True, False, 1),
    (128, 4096, 4096, False, False, 32),
    (127, 4096, 4096, True, False, 32),
    (128, 4096, 4096, True, True, 32),
    (128, 4097, 4095, True, False, 33),
  ),
)
def test_bf16_whole_b_requires_native_raw_stride_reuse_and_physical_budget(m, n, k, raw, strided, expected_b):
  if n == 4097:
    assert n * k * 2 <= 33554432
    assert ((n + 31) // 32 * 32) * ((k + 31) // 32 * 32) * 2 > 33554432
  kernel = lower(template(m=m, n=n, k=k, dtype="bfloat16", raw=raw, strided=strided, seeded=False).ir)
  # Inputs are shape/layout metadata probes only: the dummy leaf consumes
  # K, not BF16 payloads. Real SME BF16 arithmetic is tested separately.
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  a = np.empty((m, 2 * k), bf16)[:, ::2] if strided else np.empty((m, k), bf16)
  b = np.empty((n, 2 * k), bf16)[:, ::2].T if strided else np.empty((k, n), bf16, order="F")
  output = np.empty((m, n), np.float32)
  runner = ProbeInterpreter(kernel, {"A": a, "B": b, "D": np.zeros((1, n), np.float32), "C": output})
  runner.block(kernel.body)
  assert runner.a_preparations == 1
  assert runner.b_preparations == expected_b
  np.testing.assert_array_equal(output, np.full_like(output, 2 * k - 1))
