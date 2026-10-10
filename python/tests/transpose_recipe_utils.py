# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Assertions on executable transpose recipes, independent of C++ names."""

from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.affine import affine, canonical_integer, expand_definitions
from vecops.language._compiler.ir.traversal import nodes


def transpose_plan(artifact):
  return next(iter(artifact.backend_ir.transposes.values()))


def static_schedule(artifact, operation):
  program = transpose_plan(artifact).sme_program
  for statement in nodes(program):
    if isinstance(statement, I.If):
      calls = [s for s in statement.body.statements if isinstance(s, I.ArchitectureCall)]
      if any(call.operation == operation for call in calls):
        return statement
  raise AssertionError(f"missing static {operation} schedule")


def normalized_calls(artifact, operation):
  """Compare SSA expressions, not the numbering of emitted metadata names."""
  program = transpose_plan(artifact).sme_program
  definitions = {s.target: s.value for s in nodes(program) if isinstance(s, I.Assign) and s.declare}

  def scalar(value):
    def key(value):
      return value.type, value.op, tuple(map(key, value.args)), repr(value.data)

    return key(canonical_integer(expand_definitions(value, definitions)))

  return [
    (s.tile, scalar(s.slice_index), scalar(s.view.offset), tuple(map(scalar, s.view.shape)))
    for s in nodes(program)
    if isinstance(s, I.ArchitectureCall) and s.operation == operation
  ]


def assert_static_pairs(artifact, operation, width, pitch):
  branch = static_schedule(artifact, operation)
  calls = [s for s in branch.body.statements if isinstance(s, I.ArchitectureCall)]
  first, second = ("sme.load", "sme.write_hor") if operation == "sme.load" else ("sme.read_ver", "sme.store")
  assert [s.operation for s in calls] == [op for _ in range(width // 2) for op in (first, first, second, second)]
  sliced = [s for s in calls if s.slice_index is not None]
  assert [I.constant(s.slice_index) for s in sliced] == list(range(width))
  assert not any(isinstance(s, I.For) for s in nodes(branch.body))
  program = transpose_plan(artifact).sme_program
  definitions = {s.target: s.value for s in nodes(program) if isinstance(s, I.Assign) and s.declare}
  induction = {s.target for s in nodes(program) if isinstance(s, I.For)}
  offsets = [s.view.offset for s in calls if s.view is not None]
  constants = [I.constant(affine(expand_definitions(offset, definitions), induction, {})[1]) for offset in offsets]
  assert constants == [index * pitch for index in range(width)]
  assert any(isinstance(s, I.For) for s in nodes(branch.otherwise)), "cold fringes retain bounded loops"
  return branch
