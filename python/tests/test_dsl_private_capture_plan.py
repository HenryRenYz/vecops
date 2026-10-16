# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Private-call captures preserve writeback, lexical scope and live owners."""

import re
from dataclasses import replace

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.captures import plan_private_captures
from vecops.language._compiler.backends.cpu.planning import PreparedKernel
from vecops.language._compiler.ir.vector import VectorKernel
from vecops.language._compiler.ir.verification import IRVerificationError, verify_private_captures
from vecops.language._compiler.passes.manager import PassContext


def scalar(name, operation="local", dtype="int64"):
  return I.Expr(I.ScalarType(dtype), operation, data=name)


def prepared(body, *, parameters=None, bindings=None):
  kernel = I.KernelIR(I.Block(body), parameters or {}, bindings or {})
  return PreparedKernel(VectorKernel(kernel), {}, {}, {}, {}, {})


def plan(body, **kwargs):
  return plan_private_captures(prepared(body, **kwargs), PassContext("cpu-memory"))


def test_only_existing_immutable_scalars_cross_the_call_by_value():
  call = I.ExitRegion("exit", I.Block([
    I.Assign("inside", scalar("immutable")),
    I.Assign("returned", I.literal(True), False),
    I.Assign("result", scalar("mutable"), False),
    I.Assign("mutable", I.literal(9), False),
    I.Assert(scalar("row"), "induction stays a reference"),
  ]), outlined=True)
  result = plan([
    I.Assign("immutable", I.literal(7)), I.Assign("mutable", I.literal(2)),
    I.Assign("returned", I.literal(False)), I.Assign("result", I.literal(0)),
    I.For("row", I.literal(0), I.literal(4), I.literal(1), I.Block([call])),
  ])
  capture = result.private_captures[id(call)]
  assert capture.region is call
  assert capture.scalars == ("immutable",)
  assert capture.metadata == capture.pointers == ()
  with pytest.raises(TypeError):
    result.private_captures[id(call)] = capture


def test_dominance_branch_scope_and_nested_call_are_lexical():
  inner = I.ExitRegion("inner", I.Block([I.Assert(scalar("within"), "nested")]), outlined=True)
  outer = I.ExitRegion("outer", I.Block([
    I.Assign("within", scalar("before")), inner,
    I.Assert(scalar("later"), "not declared at call site"),
    I.Assert(scalar("branch_only"), "not in caller scope"),
  ]), outlined=True)
  result = plan([
    I.Assign("before", I.literal(3)),
    I.If(I.literal(True), I.Block([I.Assign("branch_only", I.literal(4))]), I.Block()),
    outer, I.Assign("later", I.literal(5)),
  ])
  assert result.private_captures[id(outer)].scalars == ("before",)
  assert result.private_captures[id(inner)].scalars == ("within",)


def test_entry_metadata_symbols_and_pointer_values_do_not_copy_owners_or_state():
  n = scalar("N", "symbol")
  a = I.View("A", "float32", (n,), (scalar("A.stride(0)", "raw"),), I.literal(0))
  scratch = I.View("scratch", "float32", (n,), (I.literal(1),), I.literal(0), owned=True)
  state = I.TileState("state", I.TileType("float32", (n,)))
  load = I.Expr(I.TileType("float32", (n,)), "load", data=a)
  call = I.ExitRegion("exit", I.Block([
    I.Allocate(scratch), I.Store(scratch, load), I.UpdateState(state, load),
    I.Assert(scalar("mode", "raw"), "parameter"),
  ]), outlined=True)
  result = plan([I.DeclareState(state, load), call],
                parameters={"A": T.Tensor((T.dynamic(),), dtype="float32"), "mode": int},
                bindings={"N": scalar("A.size(0)", "raw")})
  capture = result.private_captures[id(call)]
  assert capture.scalars == ("mode",)
  assert capture.metadata == ("A.size(0)", "A.stride(0)")
  assert capture.pointers == ("A",)
  assert capture.snapshot_reads == (load,)
  assert "scratch" not in capture.scalars and "state" not in capture.scalars


def test_logical_subviews_capture_complete_physical_abi_root_metadata():
  # The logical vector has no raw size/stride expressions. An alias guard
  # still needs every dimension of its actual rank-two ABI root.
  output = I.View("C", "float32", (I.literal(17),), (I.literal(1),), I.literal(0))
  call = I.ExitRegion("exit", I.Block([I.Store(output, I.literal(1, "float32"))]), outlined=True)
  result = plan([call], parameters={"C": T.Tensor((T.dynamic(), T.dynamic()), dtype="float32")})
  capture = result.private_captures[id(call)]
  assert capture.pointers == ("C",)
  assert capture.metadata == ("C.size(0)", "C.size(1)", "C.stride(0)", "C.stride(1)")
  verify_private_captures(result)


def test_geometry_reconstruction_keeps_its_previous_definition_rule():
  call = I.ExitRegion("exit", I.Block([I.Assert(scalar("x"), "use")]), outlined=True)
  result = plan([I.Assign("x", I.literal(1)), I.Assign("y", I.literal(2)),
                 I.Assign("y", I.literal(3), False), call])
  assert set(result.immutable_geometry) == {"x"}
  assert result.private_captures[id(call)].scalars == ("x",)


@pytest.mark.parametrize("invalid", ["subject", "mutable", "scope", "owner", "metadata", "snapshot"])
def test_verifier_rejects_invalid_capture_proofs(invalid):
  call = I.ExitRegion("exit", I.Block([I.Assert(scalar("x"), "use")]), outlined=True)
  result = plan([I.Assign("x", I.literal(1)), I.Assign("mutable", I.literal(2)),
                 call, I.Assign("mutable", I.literal(3), False), I.Assign("later", I.literal(4))])
  capture = result.private_captures[id(call)]
  verify_private_captures(result)
  if invalid == "subject":
    capture = replace(capture, region=replace(call))
  elif invalid == "mutable":
    capture = replace(capture, scalars=("mutable",))
  elif invalid == "scope":
    capture = replace(capture, scalars=("later",))
  elif invalid == "owner":
    capture = replace(capture, pointers=("scratch",))
  elif invalid == "metadata":
    capture = replace(capture, metadata=("scratch.size(0)",))
  else:
    capture = replace(capture, snapshot_reads=(scalar("not_a_historical_read"),))
  with pytest.raises(IRVerificationError, match="private"):
    verify_private_captures(replace(result, private_captures={id(call): capture}))


def test_active_hardware_scope_retains_reference_capture():
  call = I.ExitRegion("exit", I.Block([I.Assert(scalar("x"), "use")]), outlined=True)
  result = plan([I.Assign("x", I.literal(1)), I.HardwareRegion("sme", I.Block([call]))])
  capture = result.private_captures[id(call)]
  assert capture.scalars == capture.metadata == capture.pointers == ()
  assert capture.snapshot_reads == ()
  verify_private_captures(result)


def test_completed_plan_verification_uses_its_frozen_operation_semantics():
  from types import SimpleNamespace

  from vecops.language._compiler.ir.verification import verify_prepared
  from vecops.language._compiler.operations import (
    OpDef,
    Trait,
    current,
    register_operation,
    snapshot,
    unregister_operation,
    use_catalog,
  )

  name = "unit-private-capture-read"
  register_operation(OpDef(name, "memory", Trait.MEMORY_READ, effects="read"))
  try:
    environment = SimpleNamespace(operations=snapshot())
    view = I.View("A", "float32", (I.literal(17),), (I.literal(1),), I.literal(0))
    read = I.Expr(I.TileType("float32", view.shape), name, data=view)
    call = I.ExitRegion("exit", I.Block([I.Store(view, read)]), outlined=True)
    ir = prepared([call], parameters={"A": T.Tensor((17,), dtype="float32")})
    ir.vectors.manager = SimpleNamespace(options={"_compiler_environment": environment})
    result = plan_private_captures(ir, PassContext("cpu-memory", {"_compiler_environment": environment}))
    assert result.private_captures[id(call)].snapshot_reads == (read,)

    # A later plugin registry no longer classifies this operation as a read.
    unregister_operation(name)
    register_operation(OpDef(name, "changed-semantics"))
    with use_catalog(snapshot()):
      assert not current().has(read, Trait.MEMORY_READ)
      verify_prepared(result)
      # Manual plans without a captured environment deliberately use the
      # current catalog; this negative control would also fail before the fix.
      result.vectors.manager = None
      with pytest.raises(IRVerificationError, match="detached/non-read"):
        verify_prepared(result)
  finally:
    unregister_operation(name)


@pytest.mark.parametrize("snapshots", [False, True])
def test_emitter_consumes_the_plan_and_keeps_writeback_slots_by_reference(snapshots):
  n = T.dynamic(lower=0)

  @T.func(inline=False)
  def helper(old, destination, bias):
    T.store(destination, old + bias)
    return bias + 1

  @T.template
  def kernel(A: T.Tensor((n,), dtype="float32", access="inout"),
             C: T.Tensor((n,), dtype="float32", access="output"), *, bias: int):
    before = T.load(A)
    result = helper(before, C, bias)
    T.store(A, before + result)
    assert result > 0

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512",
                                options={"cpu_parameter_metadata_snapshot": snapshots})
  captures = artifact.backend_ir.private_captures
  assert len(captures) == 1
  selected = next(iter(captures.values()))
  assert "bias" in selected.scalars
  assert not any("helper_result" in name or "helper_returned" in name for name in selected.scalars)
  call_line = next(line for line in artifact.source.splitlines() if "vecops_dsl::private_call([" in line)
  assert "[&, bias" in call_line
  assert ("parameter_A_data" in call_line) is snapshots
  assert ("parameter_C_data" in call_line) is snapshots
  # A historical read is protected before the call. Its already evaluated
  # condition is copied; the snapshot Buffer itself must remain a reference.
  assert re.search(r"\bdsl_c_alias_\d+\b", call_line)
  assert "_snapshot_" not in call_line
  assert bool(selected.snapshot_reads)
  for field in ("size", "stride"):
    assert (f"parameter_C_{field}_0" in call_line) is snapshots
  trace = artifact.backend_ir.pass_trace
  assert trace.index("cpu-plan-private-captures") > trace.index("cpu-plan-storage")
  assert trace.index("cpu-plan-private-captures") > trace.index("cpu-plan-pointwise-register-layout")
  assert trace.index("verify-cpu-plan") > trace.index("cpu-plan-private-captures")
