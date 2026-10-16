# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Closed programs, not partial word shape alone, justify fixed state storage."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.planning import PreparedKernel
from vecops.language._compiler.backends.cpu.registers import plan_register_programs, register_transpose
from vecops.language._compiler.backends.cpu.state import StatePiece, closed_fixed_lanes, plan_state
from vecops.language._compiler.backends.cpu.verification import verify_closed_fixed_states
from vecops.language._compiler.ir.vector import RegisterProgram
from vecops.language._compiler.operations import snapshot


def fixture(*, budget=12, target="avx512", bits=512, update=True, outlined=False):
  value = register_transpose(2, 2, 8, dtype="float64").outputs[0]
  state = I.TileState("closed", value.type)
  declaration = I.DeclareState(state, value)
  statements = [declaration]
  if update:
    write = I.UpdateState(state, value)
    statements.append(I.ExitRegion("helper", I.Block([write]), outlined=True) if outlined else write)
  architecture = "aarch64" if target in {"sve", "sme"} else "x86"
  environment = SimpleNamespace(
    codegen_target=SimpleNamespace(architecture=architecture, isa=target, vector_bits=bits),
    operations=snapshot(),
    vector_state=None,
  )
  options = {"cpu_state_register_budget": budget, "_compiler_environment": environment}
  kernel = I.KernelIR(I.Block(statements), {}, {})
  vectors = SimpleNamespace(kernel=kernel, manager=SimpleNamespace(options=options))
  prepared = PreparedKernel(vectors, {}, {}, {}, {}, {})
  context = SimpleNamespace(options=options, environment=environment)
  selected = environment.codegen_target
  # A forged/reused x86 program must not authorize ARM state placement.
  if target in {"sve", "sme"}:
    environment.codegen_target = SimpleNamespace(architecture="x86", isa="avx512", vector_bits=512)
  prepared = plan_register_programs(prepared, context)
  environment.codegen_target = selected
  return prepared, context, state, value


def test_closed_fixed_state_consumes_original_program_and_all_updates():
  prepared, context, state, value = fixture()
  result = plan_state(prepared, context)
  plan = result.states[state.name]
  assert plan.closed_fixed and plan.pieces == (StatePiece((0,), 2),)
  assert result.register_programs[id(value)].program.outputs[0] is value
  verify_closed_fixed_states(result)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"budget": 0},
    {"target": "scalar"},
    {"target": "neon"},
    {"bits": None},
    {"target": "sve", "bits": 512},
    {"target": "sme", "bits": 512},
    {"outlined": True},
  ],
)
def test_unproved_target_budget_or_private_call_keeps_original_placement(kwargs):
  prepared, context, state, _ = fixture(**kwargs)
  result = plan_state(prepared, context)
  assert not result.states[state.name].closed_fixed
  verify_closed_fixed_states(result)


def test_missing_update_program_retains_scalable_word():
  prepared, context, state, value = fixture(update=False)
  other = replace(value)
  prepared.vectors.kernel.body.statements.append(I.UpdateState(state, other))
  result = plan_state(prepared, context)
  assert not result.states[state.name].closed_fixed
  assert result.states[state.name].pieces == (StatePiece((0,), 0, True),)


def test_matrix_accumulator_does_not_acquire_fixed_state_marker():
  prepared, context, state, _ = fixture()
  reference = I.Expr(state.type, "state_ref", data=state)
  product = I.Expr(state.type, "dot", (reference, reference, reference))
  prepared.vectors.kernel.body.statements.append(
    I.Store(I.View("output", "float64", state.type.shape, (I.literal(1),), I.literal(0)), product)
  )
  assert not plan_state(prepared, context).states[state.name].closed_fixed


def test_reinterpret_bytes_and_cyclic_dag_are_rechecked():
  prepared, context, state, value = fixture(update=False)
  source_type = replace(state.type, dtype="uint32")
  source = I.Expr(source_type, "register_input", data=0)
  changed = replace(value, args=(source,))
  plan = prepared.register_programs[id(value)]
  invalid = replace(
    plan,
    program=RegisterProgram((source,), (source, changed), (changed,)),
    tags={id(source): source_type.tag, id(changed): changed.type.tag},
  )
  assert (
    closed_fixed_lanes(state, [I.DeclareState(state, changed)], {id(changed): invalid}, context.environment) is None
  )
  # Cycle rejection must not recurse indefinitely even for a forged plan.
  object.__setattr__(changed, "args", (changed,))
  assert (
    closed_fixed_lanes(state, [I.DeclareState(state, changed)], {id(changed): invalid}, context.environment) is None
  )


def test_duplicate_or_missing_declaration_does_not_prove_closed_state():
  prepared, context, state, value = fixture(update=False)
  for statements in ([I.UpdateState(state, value)], [I.DeclareState(state, value)] * 2):
    assert closed_fixed_lanes(state, statements, prepared.register_programs, context.environment) is None


def transpose_template(shape):
  import sys
  from pathlib import Path

  sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
  from dsl_transpose import templates

  return templates(*shape, "Const", "float64", "float64", "float64")[0]


def test_public_transpose_planning_proves_fixed_states_before_emission():
  artifact = CPUBackend().lower(transpose_template((2, 8)).ir, (), target="AVX512")
  plans = [plan for plan in artifact.backend_ir.states.values() if plan.closed_fixed]
  assert plans and all(plan.pieces == (StatePiece((0,), 2),) for plan in plans)
  verify_closed_fixed_states(artifact.backend_ir)


@pytest.mark.parametrize("change", ["mask", "scalar", "pow", "matrix", "unknown_n", "partial", "large"])
def test_partial_or_changed_descriptor_cannot_borrow_closed_program(change):
  prepared, context, state, value = fixture()
  typ = value.type
  if change == "mask":
    typ = replace(typ, mask=True)
  elif change == "scalar":
    typ = I.TileType(typ.dtype, typ.shape)
  elif change == "pow":
    typ = replace(typ, tag=replace(typ.tag, pow=1))
  elif change == "matrix":
    typ = replace(typ, tag=replace(typ.tag, domain="matrix"))
  elif change == "unknown_n":
    typ = replace(typ, tag=replace(typ.tag, N=None))
  elif change == "partial":
    typ = replace(typ, shape=(I.literal(1),))
  else:
    typ = replace(typ, tag=replace(typ.tag, N=128), shape=(I.literal(128),))
  changed = replace(state, type=typ)
  statements = [I.DeclareState(changed, value)]
  assert closed_fixed_lanes(changed, statements, prepared.register_programs, context.environment) is None


@pytest.mark.parametrize(
  "change", ["missing_update", "wrong_output", "missing_operation", "changed_tag", "piece", "budget", "call"]
)
def test_verifier_rejects_stale_or_forged_closed_state(change):
  prepared, context, state, value = fixture()
  result = plan_state(prepared, context)
  assert result.states[state.name].closed_fixed
  if change == "missing_update":
    result.vectors.kernel.body.statements.append(I.UpdateState(state, replace(value)))
  elif change in {"wrong_output", "missing_operation", "changed_tag"}:
    plan = result.register_programs[id(value)]
    program, tags = plan.program, dict(plan.tags)
    if change == "wrong_output":
      program = replace(program, outputs=(replace(value),))
    elif change == "missing_operation":
      program = replace(program, operations=program.operations[:-1])
    else:
      tags[id(value)] = replace(value.type.tag, N=1)
    result = replace(
      result, register_programs={**result.register_programs, id(value): replace(plan, program=program, tags=tags)}
    )
  elif change == "piece":
    result = replace(result, states={state.name: replace(result.states[state.name], pieces=(StatePiece((0,), 4),))})
  elif change == "budget":
    context.options["cpu_state_register_budget"] = 0
  else:
    result.vectors.kernel.body.statements.append(
      I.ExitRegion("helper", I.Block([I.UpdateState(state, value)]), outlined=True)
    )
  with pytest.raises(ValueError, match="closed fixed-state"):
    verify_closed_fixed_states(result)


@pytest.mark.parametrize("shape", [(2, 8), (8, 8), (128, 128)])
def test_native_closed_fixed_transpose_full_owner_and_input(tmp_path, shape):
  # Use the original public transpose recipe, including simultaneous state updates.
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(transpose_template(shape), compiler=compiler)
  source = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 8
  saved = source.copy()
  owner = np.empty(np.prod(shape) + 32, dtype=np.float64)
  output = owner[16:-16].reshape(shape[::-1])
  for poison in (90, 195):
    for repeat in (1, 7):
      owner.view(np.uint8).fill(poison)
      before = owner.copy().view(np.uint8).copy()
      kernel(source, output, repeat=repeat)
      np.testing.assert_array_equal(output, source.T)
      np.testing.assert_array_equal(source, saved)
      np.testing.assert_array_equal(owner[:16].view(np.uint8), before[:128])
      np.testing.assert_array_equal(owner[-16:].view(np.uint8), before[-128:])
