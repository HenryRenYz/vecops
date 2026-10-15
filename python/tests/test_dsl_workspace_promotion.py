# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Replay promotion is opt-in, metadata-only and limited to top-level sites."""

from types import SimpleNamespace

import numpy as np
import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.workspace import promote_workspace_scratch
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._compiler.passes.manager import PassContext


def promote(program, enabled=True):
  prepared = SimpleNamespace(vectors=SimpleNamespace(kernel=program))
  result = promote_workspace_scratch(prepared, PassContext("cpu-memory", {"cpu_workspace_scratch": enabled}))
  return result


@pytest.mark.parametrize("enabled", [False, True])
def test_top_level_metadata_site_promotes_without_changing_view_or_accesses(enabled):
  n = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((n, 7), dtype="float32"), C: T.Tensor((n, 7), dtype="float32", access="output")):
    scratch = T.alloc(A.shape, dtype="float32")
    for row in T.serial(A.shape[0]):
      T.store(scratch[row, :], T.load(A[row, :]) + 1)
    for row in T.serial(C.shape[0]):
      T.store(C[row, :], T.load(scratch[row, :]) * 2)

  program = lower(kernel.ir)
  old = next(s for s in program.body.statements if isinstance(s, I.Allocate))
  result = promote(program, enabled)
  updated = next(s for s in program.body.statements if isinstance(s, (I.Allocate, I.WorkspaceAllocate)))
  assert updated.view is old.view
  assert isinstance(updated, I.WorkspaceAllocate) == enabled
  assert result.changed == enabled
  if not enabled:
    assert updated is old
  source = CPUBackend().lower(kernel.ir, (), target="AVX512", options={"cpu_workspace_scratch": enabled}).source
  assert ("WorkspaceContext& dsl_workspace" in source) == enabled
  assert ("dsl_workspace.tensor<float>" in source) == enabled


@pytest.mark.parametrize("scope", ["loop", "branch"])
def test_nested_allocation_retains_invocation_lifetime(scope):
  use_loop = scope == "loop"

  @T.template
  def kernel(A: T.Tensor((3, 7), dtype="float32"), C: T.Tensor((3, 7), dtype="float32", access="output")):
    if use_loop:
      for row in T.serial(3):
        scratch = T.alloc((7,), dtype="float32")
        T.store(scratch, T.load(A[row, :]) + 1)
        T.store(C[row, :], T.load(scratch))
    else:
      if T.element_at(T.load(A), (0, 0)) > 0:
        scratch = T.alloc((3, 7), dtype="float32")
        T.store(scratch, T.load(A) + 1)
        T.store(C, T.load(scratch))

  program = lower(kernel.ir)
  promote(program)
  assert not any(isinstance(s, I.WorkspaceAllocate) for s in nodes(program.body))


def test_payload_extent_is_never_promoted():
  @T.template
  def kernel(A: T.Tensor((1,), dtype="int64"), C: T.Tensor((7,), dtype="float32", access="output")):
    count = T.element_at(T.load(A), (0,))
    scratch = T.alloc((count,), dtype="float32")
    for i in T.serial(count):
      T.store(scratch[i : i + 1], 1)
    T.store(C, T.load(scratch[:7]))

  program = lower(kernel.ir)
  promote(program)
  assert not any(isinstance(s, I.WorkspaceAllocate) for s in nodes(program.body))


@pytest.mark.parametrize("conditional", [False, True])
def test_mutated_metadata_local_is_not_a_replay_extent(conditional):
  @T.template
  def kernel(A: T.Tensor((7,), dtype="int64"), C: T.Tensor((7,), dtype="float32", access="output")):
    count = A.shape[0]
    if conditional:
      if T.element_at(T.load(A), (0,)) > 0:
        count = T.element_at(T.load(A), (1,))
    else:
      count = T.element_at(T.load(A), (1,))
    scratch = T.alloc((count,), dtype="float32")
    for i in T.serial(count):
      T.store(scratch[i : i + 1], 1)
    T.store(C, T.load(scratch[:7]))

  program = lower(kernel.ir)
  promote(program)
  assert not any(isinstance(s, I.WorkspaceAllocate) for s in nodes(program.body))


@pytest.mark.parametrize("enabled", [1, "true", None])
def test_workspace_scratch_option_requires_a_boolean(enabled):
  program = I.KernelIR(I.Block([]), {}, {})
  with pytest.raises(TypeError, match="cpu_workspace_scratch must be bool"):
    promote(program, enabled)


def test_default_workspace_scratch_option_preserves_generated_body():
  @T.template
  def kernel(A: T.Tensor((3, 7), dtype="float32"), C: T.Tensor((3, 7), dtype="float32", access="output")):
    scratch = T.alloc(A.shape, dtype="float32")
    for row in T.serial(3):
      T.store(scratch[row, :], T.load(A[row, :]) + 1)
    for row in T.serial(3):
      T.store(C[row, :], T.load(scratch[row, :]))

  default = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  disabled = CPUBackend().lower(kernel.ir, (), target="AVX512", options={"cpu_workspace_scratch": False}).source
  assert default == disabled


def test_native_promoted_scratch_replays_capacity_without_new_heap_site(tmp_path):
  from dsl_test_utils import make_compiler
  from vecops.workspace_observer import drain, tracking

  rows = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((rows, 7), dtype="float32"), C: T.Tensor((rows, 7), dtype="float32", access="output")):
    scratch = T.alloc(A.shape, dtype="float32")
    for row in T.serial(rows):
      T.store(scratch[row, :], T.load(A[row, :]) + 1)
    for row in T.serial(rows):
      T.store(C[row, :], T.load(scratch[row, :]) * 2)

  compiled = T.compile(
    kernel,
    compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"),
    pass_options={"cpu_workspace_scratch": True},
  )
  a, c = np.arange(21, dtype=np.float32).reshape(3, 7), np.empty((3, 7), np.float32)
  with tracking():
    drain()
    compiled(a, c)
    first = drain()
    np.testing.assert_array_equal(c, (a + 1) * 2)
    a[:] = a + 3
    compiled(a, c)
    second = drain()
  np.testing.assert_array_equal(c, (a + 1) * 2)
  assert first and second
  assert all(event["tier"] == 3 for event in second), second
  assert not any(event["tier"] == 2 for event in second), second


def test_metadata_proof_visits_shared_dag_once_and_rejects_definition_cycles():
  class Catalog:
    calls = 0

    def get(self, value):
      self.calls += 1
      return SimpleNamespace(effects="none")

    def has(self, value, traits):
      return False

  catalog = Catalog()
  context = SimpleNamespace(options={"cpu_workspace_scratch": True}, environment=SimpleNamespace(operations=catalog))
  extent = I.literal(1)
  for _ in range(18):
    extent = I.Expr(I.ScalarType("int64"), "add", (extent, extent))
  scratch = I.View("saved", "float32", (extent,), (I.literal(1),), I.literal(0), owned=True)
  output = I.View("C", "float32", (extent,), (I.literal(1),), I.literal(0), writable=True)
  program = I.KernelIR(
    I.Block([I.Allocate(scratch), I.Store(output, I.Expr(I.TileType("float32", (extent,)), "load", data=scratch))]),
    {},
    {},
  )
  result = promote_workspace_scratch(SimpleNamespace(vectors=SimpleNamespace(kernel=program)), context)
  assert result.changed
  assert catalog.calls == 18

  cycle = I.Expr(I.ScalarType("int64"), "local", data="cycle")
  scratch = I.View("saved", "float32", (cycle,), (I.literal(1),), I.literal(0), owned=True)
  output = I.View("C", "float32", (cycle,), (I.literal(1),), I.literal(0), writable=True)
  program = I.KernelIR(
    I.Block(
      [
        I.Assign("cycle", cycle, declare=True),
        I.Allocate(scratch),
        I.Store(output, I.Expr(I.TileType("float32", (cycle,)), "load", data=scratch)),
      ]
    ),
    {},
    {},
  )
  result = promote_workspace_scratch(SimpleNamespace(vectors=SimpleNamespace(kernel=program)), context)
  assert not result.changed
