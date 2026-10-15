# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Kernel passes: fusion."""

from dataclasses import replace

from ... import ir as I
from ...analysis.loops import fusion_guard
from ...ir.iteration import FoldLoop
from ...ir.traversal import children, walk
from ...operations import Trait
from ..manager import register_pass


def matching_shape(a, b):
  return len(a) == len(b) and all(I.same_expr(x, y) for x, y in zip(a, b))


@register_pass("fuse-reductions", after=("plan-computation-reuse",))
def fuse_reductions(ir, context):
  limit = context.options.get("max_fused_reductions", 2)
  if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
    raise ValueError("max_fused_reductions must be a positive integer")

  def fuse(block):
    result, group, epilogues = [], [], []

    def flush():
      if group:
        result.append(FoldLoop(tuple(group)))
        group.clear()
      result.extend(epilogues)
      epilogues.clear()

    for statement in block.statements:
      if isinstance(statement, FoldLoop) and statement.explicit_tuple:
        # A user-written associative tuple fold is one logical traversal.
        # The opportunistic fusion limit applies to independent folds only.
        flush()
        result.append(statement)
        continue
      assignments = (
        statement.assignments
        if isinstance(statement, FoldLoop) and not statement.stores
        else (
          (statement,)
          if isinstance(statement, I.Assign)
          and context.environment.operations.has(statement.value, Trait.SCALAR_REDUCTION)
          else ()
        )
      )
      if assignments:
        for assignment in assignments:
          reduction = assignment.value
          source = reduction.args[0]
          if group:
            previous = group[0].value.args[0]
            compatible = (
              source.type.dtype == previous.type.dtype
              and FoldLoop((assignment,)).domain.matches(FoldLoop((group[0],)).domain)
              and reduction.data == group[0].value.data
            )
            if epilogues:
              captured = {id(e) for a in group for e in I.loads(a.value.args[0])}
              compatible &= all(id(e) in captured for e in I.loads(source))
            dependent = any(
              context.environment.operations.has(e, Trait.REFERENCE)
              and e.data in {a.target for a in (*group, *epilogues) if isinstance(a, I.Assign)}
              for e in walk(source)
            )
            if not compatible or dependent or len(group) >= limit:
              flush()
          group.append(assignment)
      elif (
        group
        and (
          isinstance(statement, I.Store)
          and not statement.destination.shape
          or isinstance(statement, I.Assign)
          and statement.declare
          and isinstance(statement.value.type, I.ScalarType)
        )
        and all(
          context.environment.operations.get(e).effects == "none"
          and not context.environment.operations.has(e, Trait.REDUCTION)
          for e in walk(statement.value)
        )
        and getattr(statement, "mask", None) is None
      ):
        # A scalar epilogue can follow the complete fused fold. Later folds
        # may cross it only when their memory inputs are the same captured
        # values, never a fresh read of potentially overwritten storage.
        epilogues.append(statement)
      else:
        flush()
        for child in children(statement):
          fuse(child)
        result.append(statement)
    flush()
    block.statements = result

  fuse(ir.body)
  return ir


@register_pass("fuse-reduction-elementwise", after=("fuse-reductions",))
def fuse_reduction_elementwise(ir, context):
  def fuse(block):
    result = []
    for statement in block.statements:
      for child in children(statement):
        fuse(child)
      previous = result[-1] if result else None
      assignments = (
        (previous,)
        if isinstance(previous, I.Assign) and context.environment.operations.has(previous.value, Trait.SCALAR_REDUCTION)
        else previous.assignments
        if isinstance(previous, I.ReductionGroup)
        else ()
      )
      if assignments and isinstance(statement, I.Store):
        targets = {a.target for a in assignments}
        operands = (statement.value,) + ((statement.mask,) if statement.mask is not None else ())
        dependent = any(
          context.environment.operations.has(e, Trait.REFERENCE) and e.data in targets
          for v in operands
          for e in walk(v)
        )
        if not dependent and matching_shape(assignments[0].value.args[0].type.shape, statement.destination.shape):
          stores = previous.stores if isinstance(previous, I.ReductionGroup) else ()
          if not stores:
            result[-1] = FoldLoop(assignments, (statement,))
            continue
      result.append(statement)
    block.statements = result

  fuse(ir.body)
  return ir


@register_pass("fuse-loop-regions", after=("fuse-elementwise",), before=("unroll-serial",))
def fuse_loop_regions(ir, context):
  if not context.options.get("loop_fusion", True):
    return ir

  def visit(block):
    result = []
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      left = result[-1] if result else None
      proof = fusion_guard(left, statement, context.environment.operations)
      if proof is not None:
        body, guards = proof
        combined = replace(
          left, body=I.Block([*left.body.statements, *body.statements]), unroll=max(left.unroll, statement.unroll)
        )
        grouped = []
        for operation in combined.body.statements:
          previous = grouped[-1] if grouped else None
          stores = (
            (previous,)
            if isinstance(previous, I.Store)
            else previous.stores
            if isinstance(previous, I.ElementwiseGroup)
            else ()
          )
          if (
            isinstance(operation, I.Store)
            and stores
            and matching_shape(stores[0].destination.shape, operation.destination.shape)
          ):
            grouped[-1] = I.ElementwiseGroup((*stores, operation))
          else:
            grouped.append(operation)
        combined = replace(combined, body=I.Block(grouped))
        if guards:
          condition = I.literal(True)
          for guard in guards:
            condition = I.binary("and", condition, guard)
          result[-1] = I.If(condition, I.Block([combined]), I.Block([left, statement]))
          ir.diagnostics.append("loop fusion: pointwise iteration domain with runtime root alias versioning")
        else:
          result[-1] = combined
          ir.diagnostics.append("loop fusion: shared iteration domain and independent carried state")
        continue
      result.append(statement)
    block.statements = result

  visit(ir.body)
  return ir


@register_pass("fuse-elementwise", after=("fuse-reduction-elementwise",))
def fuse_elementwise(ir, context):
  def fuse(block):
    result = []
    for statement in block.statements:
      for child in children(statement):
        fuse(child)
      previous = result[-1] if result else None
      stores = (
        (previous,)
        if isinstance(previous, I.Store)
        else previous.stores
        if isinstance(previous, I.ElementwiseGroup)
        else ()
      )
      if (
        isinstance(statement, I.Store)
        and stores
        and matching_shape(stores[0].destination.shape, statement.destination.shape)
      ):
        result[-1] = I.ElementwiseGroup((*stores, statement))
      else:
        result.append(statement)
    block.statements = result

  fuse(ir.body)
  return ir
