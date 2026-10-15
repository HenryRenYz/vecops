# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Only lexical hardware owners with exact effects may enter a checked grid."""

import pytest
import vecops.language as T
from vecops._arm_vector_state import VectorState
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.grid import plan_grid
from vecops.language._compiler.environment import freeze
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import nodes


def amx_copy_template():
  @T.template
  def kernel(A: T.Tensor((8, 16, 16), dtype="float32"), C: T.Tensor((8, 16, 16), dtype="float32", access="output")):
    for batch in T.grid(8, parallel="require"):
      with T.amx.region(((16, 64),)):
        tile = T.amx.load(0, A[batch, :, :])
        T.amx.store(tile, C[batch, :, :])

  return kernel


def sme_copy_template():
  @T.template
  def kernel(A: T.Tensor((8, 16, 16), dtype="float32"), C: T.Tensor((8, 16, 16), dtype="float32", access="output")):
    for batch in T.grid(8, parallel="require"):
      with T.sme.region():
        for row in T.serial(16):
          T.sme.load_hor(0, row, A[batch, row, :])
        for row in T.serial(16):
          T.sme.store_hor(0, row, C[batch, row, :])

  return kernel


@pytest.mark.parametrize("kind", ["amx", "sme"])
def test_lexical_hardware_memory_effects_produce_runtime_disjoint_guards(kind):
  kernel = amx_copy_template() if kind == "amx" else sme_copy_template()
  options = {"_vector_state": VectorState(64, 64)} if kind == "sme" else {}
  artifact = CPUBackend().lower(kernel.ir, (), target="SME" if kind == "sme" else "AVX512", options=options)
  plans = tuple(artifact.backend_ir.grids.values())
  assert len(plans) == 1 and plans[0].safe
  assert plans[0].reads == {"A"} and plans[0].writes == {"C"}
  assert "verified grid ownership" in artifact.source
  assert "required parallel grid failed its alias/layout guard" in artifact.source
  assert "overlap(" in artifact.source


@pytest.mark.parametrize("kind", ["amx", "sme"])
def test_hardware_grid_rejects_shared_output_across_programs(kind):
  use_amx = kind == "amx"

  @T.template
  def kernel(A: T.Tensor((8, 16, 16), dtype="float32"), C: T.Tensor((16, 16), dtype="float32", access="output")):
    for batch in T.grid(8, parallel="require"):
      if use_amx:
        with T.amx.region(((16, 64),)):
          tile = T.amx.load(0, A[batch, :, :])
          T.amx.store(tile, C)
      else:
        with T.sme.region():
          T.sme.zero_za()
          T.sme.store_tile(0, C)

  with pytest.raises(TypeError, match="same region"):
    CPUBackend().lower(
      kernel.ir, (), target="AVX512" if use_amx else "SME", options={"_vector_state": VectorState(64, 64)}
    )


def test_unknown_or_unowned_hardware_call_keeps_grid_serial():
  program = lower(amx_copy_template().ir)
  grid = next(s for s in nodes(program.body) if isinstance(s, I.Grid))
  environment = freeze(target="AVX512")
  grid.body.statements = [I.ArchitectureCall("amx.zero")]
  assert not plan_grid(grid, environment).safe
  grid.body.statements = [I.HardwareRegion("amx", I.Block([I.ArchitectureCall("amx.unknown")]), ((16, 64),))]
  assert not plan_grid(grid, environment).safe


def test_streaming_vector_from_another_owner_is_not_an_effect_summary():
  program = lower(sme_copy_template().ir)
  grid = next(s for s in nodes(program.body) if isinstance(s, I.Grid))
  destination = next(
    s.view for s in nodes(grid.body) if isinstance(s, I.ArchitectureCall) and s.operation == "sme.store_hor"
  )
  foreign = I.SMEVector("escaped", "float32", I.literal(16), 999)
  grid.body.statements = [
    I.HardwareRegion("sme", I.Block([I.ArchitectureCall("sme.store", view=destination, vector=foreign)]))
  ]
  result = plan_grid(grid, freeze(target="SME", options={"_vector_state": VectorState(64, 64)}))
  assert not result.safe and "escapes" in result.reason


def test_hardware_slice_index_payload_read_participates_in_alias_proof():
  from dataclasses import replace

  program = lower(sme_copy_template().ir)
  grid = next(s for s in nodes(program.body) if isinstance(s, I.Grid))
  destination = next(
    s.view for s in nodes(grid.body) if isinstance(s, I.ArchitectureCall) and s.operation == "sme.store_hor"
  )
  # A scalar read outside this program's batch partition is still an effect,
  # even when its only use is a ZA slice index rather than a vector operand.
  region = replace(destination.region, starts=(I.literal(0),) * 3, exact_starts=(I.literal(0),) * 3)
  foreign = replace(destination, offset=I.literal(0), region=region)
  load = I.Expr(I.TileType("float32", foreign.shape), "load", data=foreign)
  index = I.Expr(I.ScalarType("int64"), "element_at", (load, I.literal(0)))
  grid.body.statements = [
    I.HardwareRegion(
      "sme", I.Block([I.ArchitectureCall("sme.store_hor", view=destination, operands=("float32",), slice_index=index)])
    )
  ]
  result = plan_grid(grid, freeze(target="SME", options={"_vector_state": VectorState(64, 64)}))
  assert not result.safe and "cross-program" in result.reason


def test_grid_effect_scan_visits_shared_expression_dag_once():
  from dataclasses import replace

  environment = freeze(target="AVX512")

  class CountingOperations:
    calls = 0

    def get(self, expression):
      self.calls += 1
      return environment.operations.get(expression)

    def has(self, expression, trait):
      return environment.operations.has(expression, trait)

  operations = CountingOperations()
  expression = I.literal(1)
  for _ in range(18):
    expression = I.Expr(I.ScalarType("int64"), "add", (expression, expression))
  program = lower(amx_copy_template().ir)
  grid = next(s for s in nodes(program.body) if isinstance(s, I.Grid))
  grid.body.statements = [I.Assign("shared", expression)]
  assert plan_grid(grid, replace(environment, operations=operations)).safe
  assert operations.calls == 19


@pytest.mark.parametrize("extra_target", [False, True])
def test_nested_column_clipping_keeps_only_exact_batch_ownership(extra_target):
  if extra_target:

    @T.template
    def kernel(C: T.Tensor((8, 16), dtype="float32", access="output")):
      for batch, column in T.grid(8, 4, parallel="require"):
        panel = C[batch, 0:8]
        # Several requested columns clamp to the same nonempty destination.
        T.store(panel[T.max(0, column - 2) : 8][0:1], 1)
  else:

    @T.template
    def kernel(C: T.Tensor((8, 16), dtype="float32", access="output")):
      for batch in T.grid(8, parallel="require"):
        panel = C[batch, 0:8]
        T.store(panel[1:4], 1)

  program = lower(kernel.ir)
  grid = next(s for s in nodes(program.body) if isinstance(s, I.Grid))
  destination = next(s.destination for s in nodes(grid.body) if isinstance(s, I.Store))
  assert not destination.region.affine_partition
  result = plan_grid(grid, freeze(target="AVX512"))
  assert result.safe is (not extra_target)


def native_batch_template(a_dtype="bfloat16", b_dtype="bfloat16", acc_dtype="float32"):
  batch, m, n, k = 8, 32, 128, 65

  @T.template
  def kernel(
    A: T.Tensor((batch, m, k), (T.dynamic(), T.dynamic(), T.dynamic()), acc_dtype),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), acc_dtype),
    Prior: T.Tensor((1, 1, n), (T.dynamic(), T.dynamic(), T.dynamic()), acc_dtype),
    C: T.Tensor((batch, m, n), (T.dynamic(), T.dynamic(), T.dynamic()), acc_dtype, access="output"),
  ):
    a = T.cast(T.load(A) * 2 + 1, a_dtype)
    b = T.cast(T.load(B) - 1, b_dtype)
    value = T.library.batched_matmul(
      (a, b), C, acc=T.load(Prior), acc_dtype=acc_dtype, parallel="require", shared_packing=True
    )
    T.store(C, value * 2 + 1)

  return kernel


@pytest.mark.parametrize("target", ["AVX512", "SME"])
def test_native_batch_owner_whitelist_and_cost_are_program_invariant(target):
  artifact = CPUBackend().lower(
    native_batch_template().ir, (), target=target, options={"_vector_state": VectorState(64, 64)}
  )
  plan = next(iter(artifact.backend_ir.grids.values()))
  assert plan.safe
  grid = next(s for s in nodes(artifact.kernel.body) if isinstance(s, I.Grid))
  locals_ = {s.target for s in nodes(grid.body) if isinstance(s, (I.For, I.Assign))}
  assert not any(
    isinstance(s, I.Expr) and s.op in {"local", "frozen"} and s.data in locals_ for s in nodes(plan.cycles)
  )
  assert "grid_lanes" in artifact.source


@pytest.mark.parametrize("family", [("bfloat16", "bfloat16", "float32"), ("uint8", "int8", "int32")])
def test_native_matrix_batch_grid_owns_private_state_on_two_and_four_workers(family, tmp_path):
  import numpy as np
  import vecops
  from dsl_test_utils import make_compiler
  from vecops.workspace_observer import drain_memory, tracking

  compiled = T.compile(
    native_batch_template(*family),
    compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"),
    debug=T.DebugOptions(memory=True, timeline=True),
  )
  batch, m, n, k = 8, 32, 128, 65
  memory_dtype = np.dtype(family[2])
  for workers in (2, 4):
    for reverse, alias in [(False, None), (True, None), (False, "A"), (True, "B")]:
      backing = np.full((batch, m, max(k, n) + 7), -111, memory_dtype)
      c = backing[:, :, :n]
      a = backing[:, :, :k] if alias == "A" else np.empty((batch, m, k), memory_dtype)
      b = backing.reshape(-1, backing.shape[-1])[:k, :n] if alias == "B" else np.empty((k, n), memory_dtype)
      a[:] = (
        np.arange(a.size).reshape(a.shape) % 4 if family[0] == "uint8" else np.arange(a.size).reshape(a.shape) % 5 - 2
      )
      b[:] = np.arange(b.size).reshape(b.shape) % 5 - 2
      if reverse:
        a, b, c = a[..., ::-1], b[::-1, ::-1], c[..., ::-1]
      prior = c[:1, :1, :]
      prior[:] = np.arange(n) % 5
      # Small integer conversion, products and sums are exact in BF16/FP32.
      expected = ((a.copy().astype(np.int64) * 2 + 1) @ (b.copy().astype(np.int64) - 1) + prior.copy()) * 2 + 1
      before = backing.copy()
      outside = np.ones(backing.shape, bool)
      outside[:, :, :n] = False
      with vecops.cpu_thread_limit(workers), tracking():
        drain_memory()
        compiled(a, b, prior, c)
        events = drain_memory()
      np.testing.assert_array_equal(c, expected.astype(memory_dtype))
      np.testing.assert_array_equal(backing[outside], before[outside])
      active = {e["worker"] for e in events if e["site"] == "grid-program" and e["action"] == 2}
      assert len(active) == workers, events
