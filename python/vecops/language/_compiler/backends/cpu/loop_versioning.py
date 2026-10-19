# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Plan conservative root-alias checks for short ordinary pointwise loops.

The guarded loop keeps every iteration and effect in its original order.
An unsuccessful, empty or unrepresentable root check retains ordinary
emission. A certificate is intersected across lexical occurrences and never
turns a descriptor into a capacity or nonalias promise.
"""

from dataclasses import dataclass, replace
from itertools import islice
from types import MappingProxyType

from ....types import TensorType
from ... import ir as I
from ...analysis.effects import assigned_scalars, scalar_references
from ...analysis.facts import induction_facts
from ...dtypes import ELEMENT_BYTES, dtype_name
from ...ir.pipeline import PipelineFor
from ...ir.traversal import children, nodes, walk_unique
from ...operations import Trait
from .partitions import walk_lexical_scopes


@dataclass(frozen=True)
class LoopVersionPlan:
  loop: I.For
  stores: tuple
  roots: tuple
  pairs: tuple
  occurrences: int
  minimum_iterations: int = 2


def same_roots(first, second):
  """Compare immutable logical origins, including normalized lexical captures."""
  if type(first) is not tuple or type(second) is not tuple or len(first) != len(second):
    return False
  for a, b in zip(first, second):
    if (
      type(a) is not I.View
      or type(b) is not I.View
      or a.storage != b.storage
      or a.dtype != b.dtype
      or a.writable is not b.writable
      or a.owned is not False
      or b.owned is not False
      or a.region is not None
      or b.region is not None
      or len(a.shape) != len(b.shape)
      or len(a.strides) != len(b.strides)
      or any(not I.same_expr(x, y) for x, y in zip((*a.shape, *a.strides, a.offset), (*b.shape, *b.strides, b.offset)))
    ):
      return False
  return True


def _ordinary_loop(loop):
  step = I.constant(loop.step)
  return (
    type(loop) is I.For
    and loop.kind == "serial"
    and loop.unroll == 1
    and not loop.associative
    and not loop.reducers
    and type(step) is int
    and step != 0
  )


def _loop_scope(loop, scope):
  if not _ordinary_loop(loop):
    return None
  written = assigned_scalars(loop.body)
  references = set().union(*(scalar_references(x) for x in (loop.start, loop.stop, loop.step)))
  if loop.target in written or references & written:
    return None
  incoming = scope.copy().forget(written | {loop.target})
  header = tuple(incoming.normalize(x) for x in (loop.start, loop.stop, loop.step))
  if any(x is None or not incoming.safe(x) for x in header):
    return None
  fact = induction_facts(replace(loop, start=header[0], stop=header[1], step=header[2]), incoming.facts)
  if fact is None:
    return None
  entry = incoming.copy()
  incoming.bind(loop.target, fact)
  return entry, incoming, written


def _eligible(loop, scope, catalog, native_bytes):
  scopes = _loop_scope(loop, scope)
  if scopes is None:
    return None
  entry, incoming, _ = scopes
  stores, reads, outputs, roots = [], set(), set(), {}
  allowed = (I.Block, I.Assign, I.Store, I.ElementwiseGroup, I.If, I.For)
  structure = tuple(islice(nodes(loop.body, skip=(I.Expr, I.View)), 65))
  if len(structure) > 64 or any(type(n) not in allowed for n in structure):
    return None

  def view(v, state, *, output=False):
    if (
      type(v) is not I.View
      or v.owned
      or state.abi.get(v.storage) != v.dtype
      or v.dtype not in ELEMENT_BYTES
      or not v.shape
      or len(v.shape) != len(v.strides)
    ):
      return False
    # Do not introduce a disjoint fact for geometry whose original span could
    # throw. Bounds include every arithmetic intermediate in the original view.
    if any(not state.safe(x) for x in (*v.shape, *v.strides, v.offset)):
      return False
    end = v.offset
    for n, stride in zip(v.shape, v.strides):
      bounds = state.range(n)
      if bounds is None or bounds[0] < 0:
        return False
      # A magnitude proof covers either stride sign without reassociation.
      end = I.binary(
        "add", end, I.binary("mul", I.binary("max", I.literal(0), I.binary("sub", n, I.literal(1))), stride)
      )
    if not state.safe(I.binary("mul", I.binary("add", end, I.literal(1)), I.literal(ELEMENT_BYTES[v.dtype]))):
      return False
    # Use the exact enclosing logical root, rather than checking one ABI
    # descriptor span and asserting a different sliced/reshaped root disjoint.
    region = v.region
    shape, strides, offset = (
      (v.shape, v.strides, v.offset) if region is None else (region.root_shape, region.root_strides, region.root_offset)
    )
    if offset is None or len(shape) != len(strides):
      return False
    geometry = tuple(entry.normalize(x) for x in (*shape, *strides, offset))
    if any(x is None or not entry.safe(x) for x in geometry):
      return False
    count = len(shape)
    root = I.View(v.storage, v.dtype, geometry[:count], geometry[count : 2 * count], geometry[-1], writable=v.writable)
    previous = roots.get(v.storage)
    if previous is not None and (
      previous.dtype != root.dtype
      or len(previous.shape) != len(root.shape)
      or any(
        not I.same_expr(a, b)
        for a, b in zip(
          (*previous.shape, *previous.strides, previous.offset), (*root.shape, *root.strides, root.offset)
        )
      )
    ):
      return False
    roots[v.storage] = root
    if output:
      extents = tuple(state.range(x) for x in v.shape)
      if any(x is None or x[0] < 0 or x[1] <= 0 for x in extents):
        return False
      count = 1
      for n in extents:
        count *= n[1]
      # Root checks pay for themselves on repeated short tiles; wide stores
      # already use the target's prepared streaming/alignment schedules.
      if count * ELEMENT_BYTES[v.dtype] >= native_bytes * 8:
        return False
    return True

  def expression(value, state):
    terms = tuple(islice(walk_unique(value), 129))
    if len(terms) > 128:
      return False
    for term in terms:
      definition = catalog.get(term)
      if (
        definition.effects not in {"none", "read"}
        or catalog.has(term, Trait.REDUCTION | Trait.REGISTER_TRANSFORM | Trait.VECTOR_BOUNDARY | Trait.STATE_REFERENCE)
        or isinstance(term.type, I.VectorType)
        or getattr(term.type, "mask", False)
      ):
        return False
      if catalog.has(term, Trait.MEMORY_READ):
        if term.op != "load" or not view(term.data, state):
          return False
        reads.add(term.data.storage)
      elif isinstance(term.type, I.TileType) and not catalog.has(term, Trait.ELEMENTWISE):
        return False
    return True

  def block(body, state):
    for statement in body.statements:
      if type(statement) is I.Assign:
        value = statement.value
        # A captured numeric cast is still evaluated inside the original loop.
        # Its bounded integer operand is total metadata; it is not view geometry
        # and need not acquire an integer range for the floating destination.
        numeric_cast = (
          type(value.type) is I.ScalarType
          and value.type.dtype in {"float32", "float64"}
          and value.op == "cast"
          and len(value.args) == 1
          and state.safe(value.args[0])
        )
        if type(value.type) is not I.ScalarType or not (state.safe(value) or numeric_cast):
          return False
        state.assign(statement)
      elif type(statement) in (I.Store, I.ElementwiseGroup):
        members = statement.stores if type(statement) is I.ElementwiseGroup else (statement,)
        for store in members:
          if (
            store.mask is not None
            or store.vector_extent is not None
            or not view(store.destination, state, output=True)
            or not expression(store.value, state)
          ):
            return False
          stores.append(store)
          outputs.add(store.destination.storage)
      elif type(statement) is I.For:
        nested = _loop_scope(statement, state)
        if nested is None:
          return False
        _, child, written = nested
        if not block(statement.body, child):
          return False
        state.forget(written | {statement.target})
      elif type(statement) is I.If:
        if not expression(statement.condition, state):
          return False
        if not block(statement.body, state.copy()) or not block(statement.otherwise, state.copy()):
          return False
        state.forget(assigned_scalars(statement))
      else:
        return False
    return True

  if not block(loop.body, incoming) or not 1 <= len(stores) <= 8 or not reads:
    return None
  # A shared storage cannot become disjoint from itself; retain its ordinary
  # snapshot and same-mapping behavior instead of adding an always-false test.
  if reads & outputs:
    return None
  pairs = tuple(sorted((a, b) for a in reads for b in outputs))
  if len(pairs) > 4:
    return None
  return tuple(stores), tuple(roots[name] for name in sorted(reads | outputs)), pairs


def _candidate(loop, abi, catalog, native_bytes):
  """Reject ordinary large/matrix kernels before a second lexical walk."""
  if not _ordinary_loop(loop):
    return False
  start, stop, step = I.constant(loop.start), I.constant(loop.stop), I.constant(loop.step)
  if type(start) is int and type(stop) is int:
    distance = (stop - start) * (1 if step > 0 else -1)
    if distance <= 0 or (distance + abs(step) - 1) // abs(step) < 2:
      return False
  structure = tuple(islice(nodes(loop.body, skip=(I.Expr, I.View)), 65))
  if len(structure) > 64 or any(
    type(n) not in (I.Block, I.Assign, I.Store, I.ElementwiseGroup, I.If, I.For) for n in structure
  ):
    return False
  stores = tuple(n for n in structure if type(n) is I.Store)
  if not 1 <= len(stores) <= 8:
    return False
  for store in stores:
    v = store.destination
    if (
      store.mask is not None
      or store.vector_extent is not None
      or type(v) is not I.View
      or v.owned
      or abi.get(v.storage) != v.dtype
      or not v.shape
    ):
      return False
    count = 1
    for dimension in v.shape:
      extent = I.constant(dimension)
      if extent is not None:
        if type(extent) is not int or extent <= 0:
          return False
        count *= extent
      # Dynamic extents receive their lexical upper bound in the proof walk.
      # Known wide axes still reject large/matrix regions before that walk.
    if count * ELEMENT_BYTES.get(v.dtype, native_bytes * 8) >= native_bytes * 8:
      return False
    terms = tuple(islice(walk_unique(store.value), 129))
    if len(terms) > 128:
      return False
    for term in terms:
      if (
        catalog.get(term).effects not in {"none", "read"}
        or catalog.has(term, Trait.REDUCTION | Trait.REGISTER_TRANSFORM | Trait.VECTOR_BOUNDARY | Trait.STATE_REFERENCE)
        or isinstance(term.type, I.VectorType)
        or getattr(term.type, "mask", False)
      ):
        return False
      if catalog.has(term, Trait.MEMORY_READ):
        view = term.data
        if term.op != "load" or type(view) is not I.View or view.owned or abi.get(view.storage) != view.dtype:
          return False
      elif isinstance(term.type, I.TileType) and not catalog.has(term, Trait.ELEMENTWISE):
        return False
  return True


def make_loop_version_collector(kernel, catalog, target):
  if target is None or target.architecture != "x86" or target.isa != "avx512" or not target.vector_bits:
    return None, lambda: MappingProxyType({})
  blocked, active, candidates = set(), set(), set()
  abi = {
    name: dtype_name(value.dtype)
    for name, value in kernel.parameters.items()
    if type(value) is TensorType and not value.optional
  }

  def contexts(body, forbidden=False):
    if id(body) in active:
      raise ValueError("cyclic block in loop version planning")
    active.add(id(body))
    for statement in body.statements:
      if type(statement) is I.For:
        if forbidden:
          blocked.add(id(statement))
        elif _candidate(statement, abi, catalog, target.vector_bits // 8):
          candidates.add(id(statement))
      for child in children(statement):
        contexts(
          child,
          forbidden
          or isinstance(statement, (I.Grid, I.Team, I.HardwareRegion, I.ReductionGroup, I.UnrolledFor, PipelineFor)),
        )
    active.remove(id(body))

  contexts(kernel.body)
  candidates.difference_update(blocked)
  if not candidates:
    return None, lambda: MappingProxyType({})
  plans, failed, counts = {}, set(), {}

  def observe(statement, scope):
    if type(statement) is not I.For or id(statement) not in candidates:
      return
    identity = id(statement)
    counts[identity] = counts.get(identity, 0) + 1
    eligible = None if identity in blocked else _eligible(statement, scope, catalog, target.vector_bits // 8)
    if eligible is None:
      failed.add(identity)
      return
    stores, roots, pairs = eligible
    previous = plans.get(identity)
    if previous is not None and (
      not same_roots(previous.roots, roots)
      or previous.pairs != pairs
      or len(previous.stores) != len(stores)
      or any(a is not b for a, b in zip(previous.stores, stores))
    ):
      failed.add(identity)
    plans[identity] = LoopVersionPlan(statement, stores, roots, pairs, counts[identity])

  return observe, lambda: MappingProxyType({i: p for i, p in plans.items() if i not in failed})


def collect_loop_versions(kernel, catalog, target):
  """Standalone verifier/testing entry; normal compilation shares one walk."""
  observe, finish = make_loop_version_collector(kernel, catalog, target)
  if observe is not None:
    walk_lexical_scopes(kernel, observe)
  return finish()
