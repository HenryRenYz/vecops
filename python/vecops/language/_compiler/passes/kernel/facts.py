# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fold total integer predicates with sequential, lexical value facts.

Facts never move evaluation, change memory/state identities, or escape a loop
or branch. Each fact environment owns its expression memo: one DAG node may
denote different local values at different use sites.
"""

from dataclasses import replace

from ... import ir as I
from ...analysis.affine import canonical_integer
from ...analysis.effects import ValueEffects
from ...analysis.facts import ScalarAliases, ValueFacts, induction_facts, type_range
from ...ir.traversal import children, nodes, walk_unique
from ...operations import Trait
from ..manager import PassResult, register_pass


def _writes(value):
  result = set()
  for node in nodes(value, skip=(I.Expr, I.View)):
    if isinstance(node, (I.Assign, I.For, I.TeamReduce)):
      result.add(node.target)
    elif isinstance(node, I.Grid):
      result.update(node.targets)
  return result


def _single_definitions(body):
  """Count syntactic writes, including shared blocks at each execution site."""
  counts, declarations, counters = {}, set(), set()

  def visit(block):
    for statement in block.statements:
      if isinstance(statement, I.Assign):
        counts[statement.target] = counts.get(statement.target, 0) + 1
        if statement.declare:
          declarations.add(statement.target)
      elif isinstance(statement, I.ReductionGroup):
        visit(I.Block(list(statement.assignments)))
      if isinstance(statement, (I.For, I.TeamReduce)):
        counters.add(statement.target)
      elif isinstance(statement, I.Grid):
        counters.update(statement.targets)
      elif isinstance(statement, (I.UnrolledFor, I.PipelineFor)):
        counters.add(statement.loop.target)
      for child in children(statement):
        visit(child)

  visit(body)
  return {name for name in declarations - counters if counts[name] == 1}


@register_pass("propagate-value-facts", stage="kernel-cleanup", before="simplify-ir")
def propagate_value_facts(ir, context):
  catalog = context.environment.operations
  base = ValueFacts.from_kernel(ir).bindings
  changed = False
  single_definitions = _single_definitions(ir.body)
  opaque_effects = ValueEffects(catalog)
  opaque_memo = {}

  known = (I.Assign, I.Assert, I.If, I.For, I.While, I.Control, I.Exit, I.TeamReduce,
           I.ReductionGroup, I.ElementwiseGroup, I.BindTile, I.Store, I.IndexedStore,
           I.Allocate, I.Print, I.DeclareState, I.UpdateState, I.Prefetch, I.Barrier,
           I.Grid, I.Team, I.HardwareRegion, I.ExitRegion, I.UnrolledFor, I.PipelineFor,
           I.Pack, I.WorkspaceAllocate, I.ArchitectureCall)

  def opaque_expression(value):
    if value not in opaque_memo:
      # Owned metadata is not rewritten, but it can still evaluate an opaque
      # operation (for example a View address). Retain subjects in this memo.
      opaque_memo[value] = True
      owned = opaque_effects.owned_expressions(value)
      opaque_memo[value] = (
        catalog.get(value).effects == "unknown"
        or value.op == "raw" and not opaque_effects.discardable(value)
        or owned is None
        or any(opaque_expression(argument) for argument in value.args)
        or any(opaque_expression(node) for node in owned)
      )
    return opaque_memo[value]

  def opaque(value):
    if isinstance(value, I.Expr):
      return opaque_expression(value)
    if isinstance(value, I.Block):
      return any(not isinstance(statement, known) or opaque(statement) for statement in value.statements)
    return any(isinstance(node, I.Expr) and opaque_expression(node) for node in nodes(value, skip=(I.Block,))) or any(
      opaque(child) for child in children(value)
    )

  class Scope:
    def __init__(self, bindings, aliases=None):
      self.facts = ValueFacts(bindings)
      self.effects = ValueEffects(catalog, self.facts)
      self.memo, self.total = {}, {}
      self.aliases = aliases or ScalarAliases()

    def copy(self):
      return Scope(self.facts.bindings, self.aliases.copy())

    def invalidate(self, keys=(), alias_keys=()):
      # All range-dependent expression/effect proofs query ValueFacts.infer.
      # A fresh, unrelated SSA declaration cannot alter any cached answer.
      if self.facts.queried.isdisjoint(keys) and self.aliases.queried.isdisjoint(alias_keys):
        return
      self.facts.memo.clear()
      self.facts.owned.clear()
      self.facts.queried.clear()
      self.effects = ValueEffects(catalog, self.facts)
      self.memo.clear()
      self.total.clear()
      self.aliases.queried.clear()

    def forget(self, names):
      alias_keys = self.aliases.write(names)
      keys = [key for key in self.facts.bindings if key[1] in names]
      if keys:
        for key in keys:
          del self.facts.bindings[key]
      self.invalidate(keys, alias_keys)
      return self

    def aliasable(self, value):
      # Restrict equalities to exact modeled native integers/booleans. This
      # deliberately excludes promoted narrow/unsigned locals and all owned
      # metadata, reads, hardware queries and plugin operations.
      references = {"local", "frozen", "raw", "symbol"}
      arithmetic = {"add", "sub", "mul", "min", "max", "floordiv", "mod", "bit_and", "neg"}
      comparisons = {"lt", "le", "gt", "ge", "eq", "ne"}
      logical = {"and", "or", "not"}
      if not ScalarAliases.bounded(value):
        return False
      for node in walk_unique(value):
        if type(node.type) is not I.ScalarType or node.type.dtype not in {"int64", "bool"}:
          return False
        if node.op in references:
          if node.args or type(node.data) is not str or self.facts.infer(node) is None:
            return False
          # Only validated entry metadata is a stable raw/symbol leaf. A
          # bare hardware-geometry identifier has no lexical write to track.
          if node.op in {"raw", "symbol"} and (node.op, node.data) not in base:
            return False
        elif node.op == "literal":
          if node.args or type(node.data) not in (int, bool) or self.facts.infer(node) is None:
            return False
        elif node.data is not None:
          return False
        elif node.op in arithmetic:
          if (len(node.args) != (1 if node.op == "neg" else 2) or node.type.dtype != "int64"
              or any(a.type.dtype != "int64" for a in node.args)):
            return False
        elif node.op in comparisons:
          if len(node.args) != 2 or node.type.dtype != "bool" or any(a.type.dtype != "int64" for a in node.args):
            return False
        elif node.op in logical:
          if (len(node.args) != (1 if node.op == "not" else 2) or node.type.dtype != "bool"
              or any(a.type.dtype != "bool" for a in node.args)):
            return False
        else:
          return False
      return self.safe(value)

    def assign_alias(self, statement, original, value, fact):
      eligible = (
        statement.target in single_definitions and statement.declare
        and value.op != "literal" and (fact is None or fact.constant is None)
        and self.aliasable(original) and self.aliasable(value)
      )
      dependencies = self.aliases.dependencies(original) if eligible else frozenset()
      eligible = eligible and dependencies is not None and not any(
        key[1] == statement.target for key, _version in dependencies
      )
      changed_keys = self.aliases.write({statement.target})
      if eligible:
        changed_keys |= self.aliases.define(statement.target, value, dependencies)
      self.invalidate(alias_keys=changed_keys)

    def bind(self, name, value):
      keys = (("local", name), ("frozen", name))
      if all(self.facts.bindings.get(key) == value for key in keys):
        return self
      for key in keys:
        self.facts.bindings.pop(key, None)
        if value is not None:
          self.facts.bindings[key] = value
      self.invalidate(keys)
      return self

    def safe(self, value):
      if value not in self.total:
        self.total[value] = (
          self.facts.integer_semantics_modeled(value)
          and self.effects.speculatable(value)
          and self.facts.arithmetic_fits(value)
          and all(self.safe(argument) for argument in value.args)
        )
      return self.total[value]

    def expression(self, value):
      nonlocal changed
      if value.op == "literal" or value in self.memo:
        return self.memo.get(value, value)
      limits = type_range(value.type.dtype) if isinstance(value.type, I.ScalarType) else None
      # Reads/state/captured tiles carry value identity. Never clone their
      # indices, View payload or type metadata independently per use site.
      if limits is None or catalog.get(value).effects != "none" or any(
        catalog.has(value, trait) for trait in (Trait.MEMORY_READ, Trait.STATE_REFERENCE)
      ):
        return value
      if opaque(value):
        return value
      if not self.facts.integer_semantics_modeled(value):
        # Keep the whole expression, including binding operand types. A
        # later canonical/literal fold must not bypass conversion legality.
        return value
      args = tuple(self.expression(argument) for argument in value.args)
      result = value if all(a is b for a, b in zip(args, value.args)) else replace(value, args=args)
      if value.op in {"local", "frozen"}:
        alias = self.aliases.lookup(value)
        if alias is not None and alias.type == value.type and self.aliasable(alias) and self.safe(value):
          result = alias
      if self.safe(value):
        fact = self.facts.infer(result)
        if fact is not None and fact.constant is not None:
          data = bool(fact.constant) if value.type.dtype == "bool" else fact.constant
          result = I.Expr(value.type, "literal", data=data)
        else:
          candidate = result
          if result.op in {"min", "max"}:
            left, right = (self.facts.infer(argument) for argument in result.args)
            if left is not None and right is not None:
              if left.upper <= right.lower:
                candidate = result.args[result.op == "max"]
              elif right.upper <= left.lower:
                candidate = result.args[result.op == "min"]
          # Only cancel integers if the original AND candidate have total,
          # representable intermediates. Never reassociate floating values.
          try:
            candidate = canonical_integer(candidate)
          except (ArithmeticError, ValueError):
            candidate = result
          constant = I.constant(candidate)
          if type(constant) in (int, bool) and limits.lower <= constant <= limits.upper:
            data = bool(constant) if value.type.dtype == "bool" else constant
            candidate = I.Expr(value.type, "literal", data=data)
          if candidate.type == value.type and self.safe(candidate):
            result = result if I.same_expr(candidate, result) else candidate
      changed |= result is not value
      self.memo[value] = result
      return result

  def bounds_references(loop):
    return {
      node.data for value in (loop.start, loop.stop, loop.step) for node in nodes(value)
      if isinstance(node, I.Expr) and node.op in {"local", "frozen", "raw", "symbol"}
      and isinstance(node.data, str)
    }

  def block(body, scope):
    output = []
    for statement in body.statements:
      if isinstance(statement, I.Assign):
        original = statement.value
        value = scope.expression(statement.value)
        statement = statement if value is statement.value else replace(statement, value=value)
        fact = scope.facts.infer(value) if scope.safe(value) else None
        if isinstance(value.type, I.ScalarType) and fact == type_range(value.type.dtype):
          fact = None
        if opaque(value):
          scope = Scope(base)
        scope.assign_alias(statement, original, value, fact)
        scope = scope.bind(statement.target, fact)
      elif isinstance(statement, I.Assert):
        condition = scope.expression(statement.condition)
        statement = statement if condition is statement.condition else replace(statement, condition=condition)
        if opaque(condition):
          scope = Scope(base)
      elif isinstance(statement, I.If):
        condition = scope.expression(statement.condition)
        statement = statement if condition is statement.condition else replace(statement, condition=condition)
        if opaque(condition):
          scope = Scope(base)
        block(statement.body, scope.copy())
        block(statement.otherwise, scope.copy())
        scope = Scope(base) if opaque(statement) else scope.forget(_writes(statement))
      elif isinstance(statement, I.For):
        written = _writes(statement.body)
        unknown = opaque(statement)
        incoming = (Scope(base) if unknown else scope.copy()).forget(written | {statement.target})
        header = {name: incoming.expression(getattr(statement, name)) for name in ("start", "stop", "step")}
        if any(value is not getattr(statement, name) for name, value in header.items()):
          statement = replace(statement, **header)
        fact = None
        if (
          not unknown and statement.target not in written
          and not bounds_references(statement) & written
          and all(incoming.safe(value) for value in (statement.start, statement.stop, statement.step))
        ):
          fact = induction_facts(statement, incoming.facts)
        block(statement.body, incoming.bind(statement.target, fact))
        scope = Scope(base) if unknown else scope.forget(written | {statement.target})
      elif isinstance(statement, I.While):
        written = _writes(statement.body)
        unknown = opaque(statement)
        incoming = (Scope(base) if unknown else scope.copy()).forget(written)
        condition = incoming.expression(statement.condition)
        statement = statement if condition is statement.condition else replace(statement, condition=condition)
        block(statement.body, incoming)
        scope = Scope(base) if unknown else scope.forget(written)
      elif tuple(children(statement)):
        # Owners, teams, grids and unrolled/pipelined wrappers do not export
        # facts. Treat their execution multiplicity conservatively.
        written = _writes(statement)
        for child in children(statement):
          block(child, (Scope(base) if opaque(statement) else scope.copy()).forget(written))
        scope = Scope(base) if opaque(statement) else scope.forget(written)
      elif isinstance(statement, I.Store):
        if opaque(statement):
          scope = Scope(base)
        else:
          value = scope.expression(statement.value)
          statement = statement if value is statement.value else replace(statement, value=value)
      elif isinstance(statement, (I.Control, I.Exit, I.TeamReduce, I.ReductionGroup)):
        scope = scope.forget(_writes(statement)) if isinstance(statement, (I.TeamReduce, I.ReductionGroup)) else Scope(base)
      elif not isinstance(statement, known) or opaque(statement):
        scope = Scope(base)
      output.append(statement)
    body.statements = output

  block(ir.body, Scope(base))
  return PassResult(ir, changed=changed)
