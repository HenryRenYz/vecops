"""Shared loop legality analysis, including runtime alias-versioning guards.

Unknown effects, synchronization and non-pointwise cross-iteration accesses
remain barriers. Dynamic address equality is never assumed from trip count.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from .. import ir as I
from ..ir.iteration import domain
from ..ir.traversal import nodes, walk
from ..operations import Trait


@dataclass(frozen=True)
class Access:
  view: I.View
  indices: I.Expr | None = None
  node: object | None = None


@dataclass
class LoopEffects:
  reads: list
  writes: list
  references: set
  updates: set
  opaque: bool = False


def trivial_true(e):
  if I.constant(e) is True:
    return True
  if e.op == "eq" and I.same_expr(*e.args):
    return True
  if e.op == "or":
    return any(trivial_true(a) for a in e.args)
  if e.op == "and":
    return all(trivial_true(a) for a in e.args)
  return False


def effects(loop, catalog):
  result = LoopEffects([], [], set(), set())
  for value in (loop.start, loop.stop, loop.step):
    for e in walk(value):
      if catalog.has(e, Trait.REFERENCE):
        result.references.add(e.data)
      if catalog.get(e).effects != "none":
        result.opaque = True
  body_nodes = tuple(nodes(loop.body))
  changed = {s.target for s in body_nodes if isinstance(s, I.Assign) and not s.declare}
  captures = {
    s.target: s.value
    for s in body_nodes
    if isinstance(s, I.Assign)
    and s.declare
    and isinstance(s.value.type, I.ScalarType)
    and not any(catalog.has(e, Trait.REFERENCE) and e.data in changed for e in walk(s.value))
  }
  memo = {}

  def resolve(value):
    if isinstance(value, I.Expr):
      if id(value) not in memo:
        if value.op == "frozen" and value.data in captures:
          memo[id(value)] = resolve(captures[value.data])
        else:
          memo[id(value)] = replace(
            value, args=tuple(resolve(a) for a in value.args), type=resolve(value.type), data=resolve(value.data)
          )
      return memo[id(value)]
    if isinstance(value, tuple):
      return tuple(resolve(v) for v in value)
    if isinstance(value, I.Node):
      return replace(value, **{f.name: resolve(getattr(value, f.name)) for f in fields(value)})
    return value

  for node in body_nodes:
    if isinstance(node, I.Expr):
      if catalog.has(node, Trait.REFERENCE):
        result.references.add(node.data)
      if node.op == "load":
        result.reads.append(Access(resolve(node.data), node=node))
      elif node.op == "gather":
        view = node.data[0] if isinstance(node.data, tuple) else node.data
        result.reads.append(Access(resolve(view), resolve(node.args[0]), node))
      elif catalog.get(node).effects not in {"none", "read"}:
        result.opaque = True
    elif isinstance(node, I.Assign) and not node.declare:
      result.updates.add(node.target)
    elif isinstance(node, I.Store):
      result.writes.append(Access(resolve(node.destination), node=node))
    elif isinstance(node, I.IndexedStore):
      result.writes.append(Access(resolve(node.destination), resolve(node.indices), node))
      result.opaque |= not node.unique
    elif isinstance(node, I.Assert):
      result.opaque |= not trivial_true(resolve(node.condition))
    elif not isinstance(
      node,
      (
        I.Block,
        I.Expr,
        I.View,
        I.ViewRegion,
        I.Assign,
        I.Store,
        I.IndexedStore,
        I.BindTile,
        I.ScalarType,
        I.TileType,
        I.VectorTag,
      ),
    ):
      # Unknown executable nodes (including helper exits) are barriers. Pure
      # type/layout metadata is explicitly whitelisted, never an escape hatch.
      result.opaque = True
  return result


def rename(value, old, new):
  memo = {}

  def visit(v):
    if isinstance(v, I.Expr):
      if id(v) not in memo:
        data = new if v.op == "local" and v.data == old else visit(v.data)
        memo[id(v)] = replace(v, args=tuple(visit(a) for a in v.args), data=data, type=visit(v.type))
      return memo[id(v)]
    if isinstance(v, tuple):
      return tuple(visit(x) for x in v)
    if isinstance(v, I.Block):
      return I.Block([visit(s) for s in v.statements])
    if isinstance(v, I.Node):
      return replace(v, **{f.name: visit(getattr(v, f.name)) for f in fields(v)})
    return v

  return visit(value)


def _partition(access, loop):
  if access.indices is not None:
    index = access.indices
    if index.op != "arange" or index.args[0].op != "local" or index.args[0].data != loop.target:
      return False
    width = index.type.shape[0]
    return (
      I.same_expr(width, loop.step)
      or width.op == "min"
      and any(I.same_expr(v, loop.step) for v in width.args)
      or I.constant(width) is not None
      and I.constant(loop.step) is not None
      and 0 <= I.constant(width) <= I.constant(loop.step)
    )
  region = access.view.region
  if region is None:
    return False
  for start, width in zip(region.starts, region.widths):
    bounded = (
      I.constant(width) == 1
      or I.same_expr(width, loop.step)
      or width.op == "min"
      and any(I.same_expr(v, loop.step) for v in width.args)
      or I.constant(width) is not None
      and I.constant(loop.step) is not None
      and 0 <= I.constant(width) <= abs(I.constant(loop.step))
    )
    if start.op == "local" and start.data == loop.target and bounded:
      return True
  return False


def _same_access(a, b):
  return (
    a.view.storage == b.view.storage
    and I.same_expr(a.view.offset, b.view.offset)
    and len(a.view.shape) == len(b.view.shape)
    and all(I.same_expr(x, y) for x, y in zip(a.view.shape, b.view.shape))
    and all(I.same_expr(x, y) for x, y in zip(a.view.strides, b.view.strides))
    and (
      a.indices is None
      and b.indices is None
      or a.indices is not None
      and b.indices is not None
      and I.same_expr(a.indices, b.indices)
    )
  )


def _root(access):
  view, region = access.view, access.view.region
  if region is None:
    return None
  return I.View(
    view.storage, view.dtype, region.root_shape, region.root_strides, I.literal(0), view.writable, view.owned
  )


def nonoverlap_proven(view):
  """Compile-time sufficient proof; dynamic/interleaved layouts stay guarded."""
  shape = tuple(I.constant(v) for v in view.shape)
  strides = tuple(I.constant(v) for v in view.strides)
  if any(v is None for v in (*shape, *strides)) or any(v < 0 for v in shape):
    return False
  if any(v == 0 for v in shape):
    return True
  covered = 1
  for stride, extent in sorted((abs(s), n) for n, s in zip(shape, strides) if n > 1):
    if stride < covered:
      return False
    covered += (extent - 1) * stride
    if covered > (1 << 63) - 1:
      return False
  return True


def fusion_guard(left, right, catalog):
  """Return (renamed right body, alias guards) or None when fusion is illegal."""
  if not isinstance(left, I.For) or not isinstance(right, I.For) or not domain(left).matches(domain(right)):
    return None
  right = replace(right, target=left.target, body=rename(right.body, right.target, left.target))
  a, b = effects(left, catalog), effects(right, catalog)
  if a.opaque or b.opaque or a.updates & b.references or b.updates & a.references:
    return None
  guards, seen = [], set()
  pairs = [(w, r) for w in a.writes for r in b.reads + b.writes]
  pairs += [(r, w) for r in a.reads for w in b.writes]
  for x, y in pairs:
    if x.view.storage == y.view.storage:
      if not _same_access(x, y) or not _partition(x, left) or not _partition(y, right):
        return None
      root = _root(x)
      if root is None:
        return None
      key = (root.storage, root.storage)
      if key not in seen:
        if not nonoverlap_proven(root):
          guards.append(I.Expr(I.ScalarType("bool"), "view_nonoverlap", data=root))
        seen.add(key)
    elif not (x.view.owned or y.view.owned):
      rx, ry = _root(x), _root(y)
      if rx is None or ry is None:
        return None
      key = tuple(sorted((rx.storage, ry.storage)))
      if key not in seen:
        guards.append(I.Expr(I.ScalarType("bool"), "view_disjoint", data=(rx, ry)))
        seen.add(key)
  replacements = {}
  for read in b.reads:
    stores = [w for w in a.writes if w.view.storage == read.view.storage]
    if (
      len(stores) == 1
      and isinstance(stores[0].node, I.Store)
      and read.node.op == "load"
      and not stores[0].node.mask
      and not read.node.args
      and _same_access(stores[0], read)
      and _partition(read, right)
    ):
      # Round to the producer's storage dtype before conversion to the
      # consumer's compute dtype. A low-precision store is not an identity.
      stored = I.Expr(
        I.ScalarType(stores[0].view.dtype)
        if isinstance(read.node.type, I.ScalarType)
        else I.shaped_type(stores[0].view.dtype, read.node.type.shape, read.node.type),
        "cast",
        (stores[0].node.value,),
      )
      replacements[id(read.node)] = replace(read.node, op="cast", args=(stored,), data=None)
  if replacements:
    from ..ir.traversal import rewrite_expressions

    wrapper = type("Region", (), {"body": right.body})()
    rewrite_expressions(wrapper, lambda e: replacements.get(id(e), e))
  return right.body, tuple(guards)
