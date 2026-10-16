# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Plan bounded, same-allocation register-store separation before emission.

Certificates are intersections over lexical occurrences, not node-identity
caches. A shared group can occur under different induction domains. Unknown
arithmetic, ownership or execution scopes retain runtime alias versioning.
"""

from dataclasses import dataclass, replace
import re
from types import MappingProxyType

from ... import ir as I
from ...analysis.affine import canonical_integer, interval, simplify_ordered
from ...analysis.effects import ValueEffects, assigned_scalars, scalar_references
from ...analysis.facts import ScalarAliases, ValueFacts, induction_facts
from ...ir.traversal import children, nodes, walk_unique
from ...passes.manager import register_pass
from ...operations import current


@dataclass(frozen=True)
class PartitionProof:
  group: I.ElementwiseGroup
  first: I.View
  second: I.View
  occurrences: int


def partition_key(group, first, second):
  return id(group), *sorted((id(first), id(second)))


class _Scope:
  def __init__(self, base, bindings=None, aliases=None, roots=None):
    self.base = base
    self.facts = ValueFacts(base if bindings is None else bindings)
    self.aliases = aliases or ScalarAliases()
    self.roots = dict(roots or {})
    self.normalized = {}

  def copy(self):
    return _Scope(self.base, self.facts.bindings, self.aliases.copy(), self.roots)

  def forget(self, names):
    self.aliases.write(names)
    # Allocate evaluates metadata once. A later write to any of that view's
    # symbolic dependencies cannot reinterpret the existing allocation as a
    # newly sized root, even when current scalar facts are exact again.
    self.roots = {storage: root for storage, root in self.roots.items()
                  if not scalar_references(root) & names}
    bindings = {key: value for key, value in self.facts.bindings.items() if key[1] not in names}
    self.facts = ValueFacts(bindings)
    self.normalized.clear()
    return self

  def bind(self, name, fact):
    self.forget({name})
    if fact is not None:
      self.facts.bindings.update({(kind, name): fact for kind in ("local", "frozen")})

  def safe(self, value):
    """Bound absolute intermediate magnitudes, including normalization sums.

    This deliberately supports only native signed integer metadata. A
    triangle bound also protects cancellation/reassociation intermediates;
    merely proving the final subtraction small would not establish that.
    """
    magnitudes = {}
    ordered = list(walk_unique(value))
    if len(ordered) > ScalarAliases.max_nodes:
      return False
    def visit(node):
      if id(node) in magnitudes:
        return True
      if not all(visit(argument) for argument in node.args):
        return False
      if type(node.type) is not I.ScalarType or node.type.dtype != "int64":
        return False
      definition = current().get(node)
      if definition.effects != "none" or definition.may_trap and node.op not in {"floordiv", "mod"}:
        return False
      if node.op in {"local", "frozen", "raw", "symbol", "literal"}:
        if node.args or node.op in {"raw", "symbol"} and (node.op, node.data) not in self.base:
          return False
        fact = self.facts.infer(node)
        if fact is None:
          return False
        size = max(abs(fact.lower), abs(fact.upper))
      elif node.op in {"add", "sub", "mul", "min", "max", "floordiv", "mod"} and len(node.args) == 2:
        sizes = [magnitudes.get(id(argument)) for argument in node.args]
        if any(size is None for size in sizes) or not self.facts.integer_semantics_modeled(node):
          return False
        if node.op == "mul":
          size = sizes[0] * sizes[1]
        elif node.op in {"floordiv", "mod"}:
          divisor = self.facts.infer(node.args[1])
          if divisor is None or divisor.constant is None or divisor.constant <= 0:
            return False
          size = sizes[0] + divisor.constant
        else:
          # min/max ordering queries also form the operand difference.
          size = sum(sizes)
      else:
        return False
      if size > (1 << 63) - 1 or not self.facts.arithmetic_fits(node):
        return False
      magnitudes[id(node)] = size
      return True
    return visit(value)

  def bounds(self, value):
    references = {(node.op, node.data) for node in walk_unique(value)
                  if node.op in {"local", "raw", "symbol"}}
    result = {}
    for (kind, name), fact in self.facts.bindings.items():
      if (kind, name) not in references:
        continue
      key = name if kind == "local" else (kind, name)
      result[key] = I.literal(fact.lower), I.literal(fact.upper)
    return result

  def normalize(self, value):
    if value in self.normalized:
      return self.normalized[value]
    if not self.safe(value):
      self.normalized[value] = None
      return None
    original = value
    if value.op in {"local", "frozen"}:
      alias = self.aliases.lookup(value)
      if alias is not None and alias.type == value.type:
        value = alias
      fact = self.facts.infer(value)
      if fact is not None and fact.constant is not None:
        value = I.Expr(value.type, "literal", data=fact.constant)
      elif value.op == "frozen":
        value = replace(value, op="local")
    args = tuple(self.normalize(argument) for argument in value.args)
    if any(argument is None for argument in args):
      self.normalized[original] = None
      return None
    value = replace(value, args=args) if any(a is not b for a, b in zip(args, value.args)) else value
    value = canonical_integer(simplify_ordered(value, self.bounds(value)))
    self.normalized[original] = value if self.safe(value) else None
    return self.normalized[original]

  def range(self, value):
    value = self.normalize(value)
    if value is None:
      return None
    endpoints = interval(value, self.bounds(value))
    if endpoints is None or not all(self.safe(endpoint) for endpoint in endpoints):
      return None
    limits = tuple(I.constant(endpoint) for endpoint in endpoints)
    return limits if all(type(limit) is int for limit in limits) else None

  def assign(self, statement):
    value = self.normalize(statement.value)
    fact = None if value is None else self.facts.infer(value)
    dependencies = None if value is None else self.aliases.dependencies(statement.value)
    self.bind(statement.target, fact)
    if dependencies is not None and all(key[1] != statement.target for key, _version in dependencies):
      self.aliases.define(statement.target, value, dependencies)


def _separated(scope, first, second):
  if first.storage != second.storage or not first.owned or not second.owned or first.dtype != second.dtype:
    return False
  root = scope.roots.get(first.storage)
  if root is None or not root.owned or root.dtype != first.dtype or not root.shape or len(root.shape) != len(root.strides):
    return False
  expected = I.literal(1)
  for extent, stride in zip(reversed(root.shape), reversed(root.strides)):
    difference = scope.range(I.binary("sub", stride, expected))
    if difference != (0, 0):
      return False
    expected = I.binary("mul", expected, extent)
  if not scope.safe(expected):
    return False
  intervals = []
  for view in (first, second):
    region = view.region
    rank = len(root.shape)
    if region is None or any(len(values) != rank for values in (
      region.root_shape, region.root_strides, region.starts, region.widths, region.steps, region.clipped,
      region.exact_starts or region.starts,
    )):
      return False
    if len(region.axes) != len(view.shape) or len(view.shape) != len(view.strides) or (
      len(set(region.axes)) != len(region.axes) or any(type(axis) is not int or not 0 <= axis < rank for axis in region.axes)
    ):
      return False
    if any(step != 1 for step in region.steps):
      return False
    if not all(I.same_extent(a, b) for a, b in zip(
      (*region.root_shape, *region.root_strides), (*root.shape, *root.strides)
    )):
      return False
    starts = region.exact_starts or region.starts
    offset = root.offset
    for start, stride in zip(starts, root.strides):
      offset = I.binary("add", offset, I.binary("mul", start, stride))
    if scope.range(I.binary("sub", offset, view.offset)) != (0, 0):
      return False
    widths = [I.literal(1) for _axis in root.shape]
    for axis, extent, stride in zip(region.axes, view.shape, view.strides):
      if scope.range(I.binary("sub", stride, root.strides[axis])) != (0, 0):
        return False
      widths[axis] = extent
    axes = []
    for start, width, requested, extent in zip(starts, widths, region.widths, root.shape):
      lower, count = scope.range(start), scope.range(width)
      coverage = scope.range(I.binary("sub", width, requested))
      remainder = scope.range(I.binary("sub", I.binary("add", start, width), extent))
      if (lower is None or count is None or coverage is None or remainder is None
          or lower[0] < 0 or count[0] < 0 or coverage[1] > 0 or remainder[1] > 0):
        return False
      axes.append((start, width))
    intervals.append(axes)
  for (start_a, width_a), (start_b, width_b) in zip(*intervals):
    gap = scope.range(I.binary("sub", I.binary("sub", start_b, start_a), width_a))
    reverse = scope.range(I.binary("sub", I.binary("sub", start_a, start_b), width_b))
    if gap is not None and gap[0] >= 0 or reverse is not None and reverse[0] >= 0:
      return True
  return False


def collect_partition_proofs(kernel):
  """Intersect each group/pair proof across every lexical occurrence."""
  required = {name: value for name, value in kernel.parameters.items() if not getattr(value, "optional", False)}
  bindings = {}
  for name, value in kernel.bindings.items():
    match = re.fullmatch(r"([A-Za-z_]\w*)\.(?:size|stride)\(\d+\)", value.data) if (
      value.op == "raw" and isinstance(value.data, str)
    ) else None
    if match is not None and match[1] in required:
      bindings[name] = value
  base = ValueFacts.from_kernel(replace(kernel, parameters=required, bindings=bindings)).bindings
  proofs, counts, failed, active = {}, {}, set(), set()
  catalog = current()
  effects = ValueEffects(catalog)

  def opaque(statement):
    return any(
      catalog.get(value).effects == "unknown"
      or value.op == "raw" and not effects.discardable(value)
      or effects.owned_expressions(value) is None
      for value in nodes(statement, skip=(I.Block,)) if isinstance(value, I.Expr)
    )

  def allocations(statement):
    return {node.view.storage for node in nodes(statement, skip=(I.Expr, I.View)) if isinstance(node, I.Allocate)}

  def writes(statement):
    result = assigned_scalars(statement)
    for node in nodes(statement, skip=(I.Expr, I.View)):
      if isinstance(node, I.Grid):
        result.update(node.targets)
      elif isinstance(node, I.TeamReduce):
        result.add(node.target)
    return result

  def block(body, scope):
    if id(body) in active:
      raise ValueError("cyclic block in CPU partition planning")
    active.add(id(body))
    uncertain = False
    try:
      for statement in body.statements:
        if opaque(statement):
          scope = _Scope(base)
          uncertain = True
        if isinstance(statement, I.Assign):
          scope.assign(statement)
        elif isinstance(statement, I.Allocate):
          scope.roots[statement.view.storage] = statement.view
        elif isinstance(statement, I.BindTile):
          # This retains historical readers; it neither evaluates its value
          # nor rebinds storage (cpp.control handles the same distinction).
          pass
        elif isinstance(statement, I.ElementwiseGroup):
          occurrence_pairs = set()
          for index, first in enumerate(statement.stores):
            for second in statement.stores[index + 1:]:
              key = partition_key(statement, first.destination, second.destination)
              if key in occurrence_pairs:
                continue
              occurrence_pairs.add(key)
              counts[key] = counts.get(key, 0) + 1
              if not _separated(scope, first.destination, second.destination):
                failed.add(key)
              proofs[key] = PartitionProof(statement, first.destination, second.destination, counts[key])
        elif isinstance(statement, I.For):
          written = writes(statement.body)
          incoming = scope.copy().forget(written | {statement.target})
          references = set().union(*(scalar_references(v) for v in (statement.start, statement.stop, statement.step)))
          header = [incoming.normalize(v) for v in (statement.start, statement.stop, statement.step)]
          fact = None
          if statement.target not in written and not references & written and all(v is not None for v in header):
            fact = induction_facts(replace(statement, start=header[0], stop=header[1], step=header[2]), incoming.facts)
          incoming.bind(statement.target, fact)
          if block(statement.body, incoming):
            scope = _Scope(base)
            uncertain = True
          else:
            scope.forget(written | {statement.target})
          for storage in allocations(statement):
            scope.roots.pop(storage, None)
        elif tuple(children(statement)):
          written = writes(statement)
          child_unknown = False
          for child in children(statement):
            child_unknown |= block(child, scope.copy().forget(written))
          if child_unknown:
            scope = _Scope(base)
            uncertain = True
          else:
            scope.forget(written)
          for storage in allocations(statement):
            scope.roots.pop(storage, None)
        elif isinstance(statement, (I.Control, I.Exit, I.TeamReduce)):
          scope = _Scope(base)
        elif isinstance(statement, I.ReductionGroup):
          scope.forget(writes(statement))
        elif not isinstance(statement, (
          I.Store, I.IndexedStore, I.Assert, I.BindTile, I.DeclareState, I.UpdateState,
          I.ArchitectureCall, I.AMXTile, I.SMEVector, I.SMEMask,
        )):
          scope = _Scope(base)
          uncertain = True
    finally:
      active.remove(id(body))
    return uncertain

  block(kernel.body, _Scope(base))
  return MappingProxyType({key: proof for key, proof in proofs.items() if key not in failed})


@register_pass("cpu-plan-partition-proofs", stage="cpu-memory", after=("cpu-plan-storage", "cpu-plan-register-programs"))
def plan_partition_proofs(prepared, context):
  return replace(prepared, partition_proofs=collect_partition_proofs(prepared.vectors.kernel))
