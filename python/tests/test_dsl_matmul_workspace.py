# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Rank-two matmul exposes checked, explicit result-workspace reuse."""

import numpy as np
import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.frontend.parser import lower
from vecops.language._debug.interpreter import Interpreter


@pytest.mark.parametrize(
  "dtype,acc_dtype", [("float32", "float32"), ("float64", "float64"), ("int8", "int32"), ("int16", "int64")]
)
def test_matmul_workspace_infers_accumulation_dtype_and_preserves_epilogue(dtype, acc_dtype):
  @T.template
  def kernel(
    A: T.Tensor((3, 3), dtype=dtype),
    B: T.Tensor((3, 3), dtype=dtype),
    Prior: T.Tensor((1, 3), dtype=acc_dtype),
    Scratch: T.Tensor((3, 3), dtype=acc_dtype, access="output"),
    C: T.Tensor((3, 3), dtype=acc_dtype, access="output"),
  ):
    value = T.library.matmul((A, B), C, acc=T.load(Prior), workspace=Scratch, implementation="vector")
    T.store(C, value * 2 + 1)

  a = (np.arange(9).reshape(3, 3) % 5 - 2).astype(dtype)
  b = (np.arange(9).reshape(3, 3) % 3 - 1).astype(dtype)
  prior = np.arange(3).reshape(1, 3).astype(acc_dtype)
  scratch, c = np.empty((3, 3), acc_dtype), np.empty((3, 3), acc_dtype)
  program = lower(kernel.ir)
  Interpreter(program, {"A": a, "B": b, "Prior": prior, "Scratch": scratch, "C": c}).block(program.body)
  expected = (a.astype(np.int64) @ b.astype(np.int64) + prior) * 2 + 1
  np.testing.assert_array_equal(c, expected)


@pytest.mark.parametrize("overlap", ["A", "B", "C", "Prior", "producer"])
def test_matmul_workspace_rejects_any_overlapping_read_or_output_before_write(overlap):
  read_workspace = overlap == "producer"

  @T.template
  def kernel(
    A: T.Tensor((3, 3), dtype="float32"),
    B: T.Tensor((3, 3), dtype="float32"),
    Prior: T.Tensor((1, 3), dtype="float32"),
    Scratch: T.Tensor((3, 3), dtype="float32", access="output"),
    C: T.Tensor((3, 3), dtype="float32", access="output"),
  ):
    T.library.matmul(
      (T.load(A) + T.load(Scratch) if read_workspace else T.load(A), B),
      C,
      acc=T.load(Prior),
      workspace=Scratch,
      implementation="vector",
    )

  scratch = np.full((3, 3), -111, np.float32)
  a, b, c = (np.full((3, 3), i, np.float32) for i in (1, 2, 3))
  prior = np.ones((1, 3), np.float32)
  if overlap == "A":
    a = scratch
  elif overlap == "B":
    b = scratch
  elif overlap == "C":
    c = scratch
  elif overlap == "Prior":
    prior = scratch[:1]
  before_scratch, before_c = scratch.copy(), c.copy()
  program = lower(kernel.ir)
  runner = Interpreter(program, {"A": a, "B": b, "Prior": prior, "Scratch": scratch, "C": c})
  with pytest.raises(AssertionError, match="workspace overlaps"):
    runner.block(program.body)
  np.testing.assert_array_equal(scratch, before_scratch)
  np.testing.assert_array_equal(c, before_c)


def test_matmul_workspace_uses_existing_workspace_context():
  @T.template
  def kernel(
    A: T.Tensor((3, 7), dtype="float32"),
    B: T.Tensor((7, 5), dtype="float32"),
    C: T.Tensor((3, 5), dtype="float32", access="output"),
  ):
    scratch = T.workspace(C.shape, dtype="float32")
    for repeat in T.serial(2):
      T.library.matmul((A, B), C, workspace=scratch, implementation="vector")

  program = lower(kernel.ir)
  assert any(isinstance(s, I.WorkspaceAllocate) for s in program.body.statements)
  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  assert "vecops::execution::WorkspaceContext& dsl_workspace" in source
  assert "dsl_workspace.tensor<float>" in source


def test_native_rank_two_workspace_reuses_calls_and_preserves_aliases(tmp_path):
  from dsl_test_utils import make_compiler
  from vecops.workspace_observer import drain, tracking

  @T.template
  def kernel(
    A: T.Tensor((3, 7), (T.dynamic(), T.dynamic()), dtype="float32"),
    B: T.Tensor((7, 5), (T.dynamic(), T.dynamic()), dtype="float32"),
    C: T.Tensor((3, 5), (T.dynamic(), T.dynamic()), dtype="float32", access="output"),
  ):
    scratch = T.workspace(C.shape, dtype="float32")
    value = T.library.matmul((T.load(A) + 1, B), C, acc=T.load(C[:1, :]), workspace=scratch, implementation="vector")
    T.store(C, value * 2 + 1)

  compiled = T.compile(
    kernel, compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  )
  for alias in (None, "A", "B"):
    backing = np.arange(49, dtype=np.float32).reshape(7, 7) % 5 - 2
    c = backing[:3, :5]
    a = backing[:3, :] if alias == "A" else np.arange(21, dtype=np.float32).reshape(3, 7) % 3 - 1
    b = backing[:, :5] if alias == "B" else np.arange(35, dtype=np.float32).reshape(7, 5) % 5 - 2
    a, b, c = a[:, ::-1], b[::-1, ::-1], c[:, ::-1]
    for invocation in range(2):
      expected = ((a.copy() + 1) @ b.copy() + c[:1, :].copy()) * 2 + 1
      with tracking():
        drain()
        compiled(a, b, c)
        events = drain()
      np.testing.assert_array_equal(c, expected)
      if invocation:
        assert events and all(e["tier"] == 3 for e in events)
