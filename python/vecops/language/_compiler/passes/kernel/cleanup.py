# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Reusable semantic cleanup after IR-producing transforms.

These passes report actual changes for bounded fixed-point groups. They do
not reassociate floats, merge memory reads, speculate checks, or discard lazy
read provenance still needed by a consumer. Allocation/snapshot placement is
deliberately outside this layer.
"""

from dataclasses import fields, replace

from ... import ir as I
from ...analysis.effects import ValueEffects
from ...analysis.facts import ValueFacts, type_range
from ...ir.traversal import children, nodes
from ...operations import Trait
from ..manager import PassResult, register_pass


@register_pass("simplify-ir", stage="kernel-cleanup")
def simplify_ir(ir, context):
  changed = False
  memo = {}
  effects = ValueEffects(context.environment.operations)

  def expression(value):
    nonlocal changed
    if id(value) in memo:
      return memo[id(value)]
    arguments = value.args
    # Do not even visit an unexecuted scalar arm: compile-time arithmetic or
    # shape checks in that arm need not be valid on the selected path.
    if value.op == "where" and isinstance(value.type, I.ScalarType):
      condition = expression(arguments[0])
      if type(I.constant(condition)) is bool:
        result = expression(arguments[1 if I.constant(condition) else 2])
        if result.type == value.type:
          memo[id(value)] = result
          changed = True
          return result
    if value.op in {"and", "or"} and isinstance(value.type, I.ScalarType):
      left = expression(arguments[0])
      if type(I.constant(left)) is bool and I.constant(left) == (value.op == "or"):
        memo[id(value)] = left
        changed = True
        return left
    args = tuple(expression(a) for a in arguments)
    result = value if all(a is b for a, b in zip(args, arguments)) else replace(value, args=args)
    if (
      len(args) == 2
      and all(isinstance(a.type, I.ScalarType) for a in args)
      and context.environment.operations.get(value).fold is not None
    ):
      # Reuse the language's typed integer identities and constant folding.
      # Never apply x*0, floating reassociation or unchecked narrowing here.
      try:
        candidate = I.binary(value.op, *args)
      except (ArithmeticError, ValueError, OverflowError):
        candidate = result
      if value.op == "broadcast_dim" and candidate.op == "literal":
        # A shape check has an explicit integer result type even if its
        # dimensions originated as weak literals. Preserve that provenance;
        # never use this rule to narrow a fold or to alter floating promotion.
        limits = type_range(result.type.dtype) if isinstance(result.type, I.ScalarType) else None
        if (limits is not None and result.type.dtype != "bool" and candidate.type.dtype == result.type.dtype
            and type(candidate.data) is int and limits.lower <= candidate.data <= limits.upper):
          candidate = I.Expr(result.type, "literal", data=candidate.data)
        else:
          candidate = result
      if candidate.type == result.type and (candidate.op != result.op or candidate.op == "literal"):
        # Constructor identities can equate two structurally matching reads.
        # Captures have value/version identity: dropping a distinct read or
        # check is legal only with its own effect proof, even for min(x,x).
        selected_argument = any(candidate is argument for argument in args)
        if not (selected_argument or candidate.op == "literal") or all(
          argument is candidate or effects.discardable(argument) for argument in args
        ):
          result = candidate
    if result.op == "identity" and len(result.args) == 1 and result.args[0].type == result.type:
      result = result.args[0]
    changed |= result is not value
    memo[id(value)] = result
    return result

  def visit(value):
    if isinstance(value, I.Expr):
      return expression(value)
    if isinstance(value, I.Block):
      block(value)
      return value
    if isinstance(value, tuple):
      result = tuple(visit(item) for item in value)
      return value if all(a is b for a, b in zip(value, result)) else result
    if isinstance(value, I.Node):
      updates = {field.name: visit(getattr(value, field.name)) for field in fields(value)}
      return value if all(item is getattr(value, key) for key, item in updates.items()) else replace(value, **updates)
    return value

  def block(body):
    nonlocal changed
    output = []
    for statement in body.statements:
      if isinstance(statement, I.If):
        condition = expression(statement.condition)
        if type(I.constant(condition)) is bool:
          selected = statement.body if I.constant(condition) else statement.otherwise
          block(selected)
          output.extend(selected.statements)
          changed = True
          continue
      statement = visit(statement)
      if isinstance(statement, I.Assert) and I.constant(statement.condition) is True:
        changed = True
        continue
      output.append(statement)
    body.statements = output

  block(ir.body)
  return PassResult(ir, changed=changed)


@register_pass("dead-store-elimination", stage="kernel-cleanup", after="simplify-ir", before="dead-code-elimination")
def dead_store_elimination(ir, context):
  """Eliminate overwritten private initialization without crossing effects.

  Only adjacent, complete, identical unmasked destinations qualify. The first
  value and address metadata must be harmless; the replacement cannot observe
  that private storage. Public/possibly aliased stores, historical captures,
  fences, partial writes and state owners keep their original order.
  """
  catalog = context.environment.operations
  effects = ValueEffects(catalog, ValueFacts.from_kernel(ir))
  private = {value.view.storage for value in nodes(ir.body) if isinstance(value, I.Allocate) and value.view.owned}
  changed = False

  def overwritten(previous, current):
    if not isinstance(previous, I.Store) or not isinstance(current, I.Store):
      return False
    a, b = previous.destination, current.destination
    if previous.mask is not None or current.mask is not None or a.storage not in private or a.storage != b.storage:
      return False
    if a.dtype != b.dtype or len(a.shape) != len(b.shape):
      return False
    if not all(I.same_expr(x, y) for x, y in zip((*a.shape, *a.strides, a.offset), (*b.shape, *b.strides, b.offset))):
      return False
    if not effects.discardable(previous.value) or not all(
      effects.speculatable(value) for value in (*a.shape, *a.strides, a.offset)
    ):
      return False
    for value in nodes(current.value):
      if isinstance(value, I.View) and value.storage == a.storage:
        return False
      if isinstance(value, I.Expr):
        effect = catalog.get(value).effects
        if effect != "none" and not (effect == "read" and value.op == "load" and isinstance(value.data, I.View)):
          return False
    return True

  def visit(body):
    nonlocal changed
    output = []
    for statement in body.statements:
      for child in children(statement):
        visit(child)
      if output and overwritten(output[-1], statement):
        output.pop()
        changed = True
      output.append(statement)
    body.statements = output

  visit(ir.body)
  return PassResult(ir, changed=changed)


@register_pass("dead-code-elimination", stage="kernel-cleanup")
def dead_code_elimination(ir, context):
  effects = ValueEffects(context.environment.operations)
  live_nodes, referenced_names, storage_uses = set(), set(), set()

  for node in nodes(ir.body, skip=(I.BindTile, I.Allocate)):
    if isinstance(node, I.Expr):
      live_nodes.add(id(node))
      if node.op in {"local", "frozen", "symbol"} and isinstance(node.data, str):
        referenced_names.add(node.data)
    elif isinstance(node, I.View):
      storage_uses.add(node.storage)
  capture_memo = {}

  def live_capture(value):
    if isinstance(value, tuple):
      return any(live_capture(item) for item in value)
    if not isinstance(value, I.Node):
      return False
    if id(value) not in capture_memo:
      own = (
        isinstance(value, I.Expr)
        and id(value) in live_nodes
        and context.environment.operations.has(value, Trait.MEMORY_READ | Trait.STATE_REFERENCE)
      )
      capture_memo[id(value)] = own or any(live_capture(getattr(value, field.name)) for field in fields(value))
    return capture_memo[id(value)]

  changed = False

  def visit(body):
    nonlocal changed
    output = []
    for statement in body.statements:
      remove = False
      if isinstance(statement, I.Assign) and statement.declare:
        remove = statement.target not in referenced_names and effects.discardable(statement.value)
      elif isinstance(statement, I.BindTile):
        remove = id(statement.value) not in live_nodes and not live_capture(statement.value)
      elif isinstance(statement, I.Allocate):
        # Allocate is a lazy storage declaration; its metadata is only
        # evaluated at a real use. Workspace sites have distinct authority
        # and are deliberately not handled by this rule.
        remove = statement.view.storage not in storage_uses
      if remove:
        changed = True
        continue
      for child in children(statement):
        visit(child)
      output.append(statement)
    body.statements = output

  visit(ir.body)
  return PassResult(ir, changed=changed)
