# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Conservative scalar LICM and bounded loop unswitching.

Only total integer/boolean computations move. Reads, shape checks, floating
math, hardware state, allocation and synchronization retain lexical order.
Unswitching is limited to a single branch body and a code-size budget; it never
changes recurrence/reduction order or crosses an outlined/hardware owner.
"""

from dataclasses import replace

from ... import ir as I
from ...analysis.effects import ValueEffects, assigned_scalars, scalar_references
from ...analysis.facts import ValueFacts
from ...ir.traversal import children, nodes
from ..manager import PassResult, register_pass


@register_pass("loop-invariant-code-motion", stage="kernel-loop")
def loop_invariant_code_motion(ir, context):
  effects = ValueEffects(context.environment.operations, ValueFacts.from_kernel(ir))
  changed = False

  def visit(body):
    nonlocal changed
    output = []
    for statement in body.statements:
      if isinstance(statement, (I.HardwareRegion, I.ExitRegion)):
        # Optimize inside ownership, never move through its entry boundary.
        visit(statement.body)
      else:
        for child in children(statement):
          visit(child)
      if isinstance(statement, I.For):
        modified = assigned_scalars(statement.body) | {statement.target}
        definitions = {}
        for node in nodes(statement.body):
          if isinstance(node, (I.Assign, I.For)):
            definitions[node.target] = definitions.get(node.target, 0) + 1
        retained = []
        for operation in statement.body.statements:
          if (
            isinstance(operation, I.Assign)
            and operation.declare
            # A loop-local initialization must repeat if a later operation
            # updates the same slot, even when the initializer is constant.
            and definitions.get(operation.target) == 1
            and effects.speculatable(operation.value)
            and not scalar_references(operation.value) & modified
          ):
            output.append(operation)
            modified.discard(operation.target)
            changed = True
          else:
            retained.append(operation)
        if len(retained) != len(statement.body.statements):
          statement = replace(statement, body=I.Block(retained))
      output.append(statement)
    body.statements = output

  visit(ir.body)
  return PassResult(ir, changed=changed)


@register_pass("loop-unswitching", stage="kernel-loop", after="loop-invariant-code-motion")
def loop_unswitching(ir, context):
  budget = context.options.get("loop_unswitch_budget", 32)
  if type(budget) is not int or budget < 0:
    raise ValueError("loop_unswitch_budget must be a nonnegative integer")
  effects = ValueEffects(context.environment.operations, ValueFacts.from_kernel(ir))
  changed = False

  def visit(body):
    nonlocal changed
    output = []
    for statement in body.statements:
      for child in children(statement):
        visit(child)
      if isinstance(statement, I.For) and len(statement.body.statements) == 1:
        branch = statement.body.statements[0]
        if (
          isinstance(branch, I.If)
          and effects.speculatable(branch.condition)
          and all(effects.speculatable(value) for value in (statement.start, statement.stop, statement.step))
          and not scalar_references(branch.condition) & (assigned_scalars(statement.body) | {statement.target})
          and sum(1 for _ in nodes(branch)) <= budget
          and not any(isinstance(node, (I.HardwareRegion, I.ExitRegion, I.Exit, I.Control)) for node in nodes(branch))
        ):
          statement = I.If(
            branch.condition,
            I.Block([replace(statement, body=branch.body)]),
            I.Block([replace(statement, body=branch.otherwise)]) if branch.otherwise.statements else I.Block(),
          )
          changed = True
      output.append(statement)
    body.statements = output

  visit(ir.body)
  return PassResult(ir, changed=changed)
