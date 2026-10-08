# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Opt-in two-bank scheduling with an invocation-level no-alias guard.

Only straight-line, fixed logical tile extents are eligible. Unsupported loops
retain ordered prefetch scheduling; data reads are never moved without proof.
"""

from dataclasses import fields, replace

from ... import ir as I
from ...ir.traversal import values, walk
from ...operations import Trait


def buffer_loop(ir, loop, context, identity):
  allowed = (I.Assign, I.BindTile, I.Store, I.ElementwiseGroup)
  if any(not isinstance(s, allowed) or isinstance(s, I.Assign) and not s.declare for s in loop.body.statements):
    return None
  definitions = {s.target: s.value for s in loop.body.statements if isinstance(s, I.Assign)}

  def expand(value, index, seen=frozenset()):
    if isinstance(value, I.Expr):
      if value.op == "local" and value.data == loop.target:
        return index
      if value.op in {"local", "frozen"} and value.data in definitions and value.data not in seen:
        return expand(definitions[value.data], index, seen | {value.data})
      return replace(
        value, args=tuple(expand(a, index, seen) for a in value.args), data=expand(value.data, index, seen)
      )
    if isinstance(value, tuple):
      return tuple(expand(v, index, seen) for v in value)
    if isinstance(value, I.Node):
      return replace(value, **{f.name: expand(getattr(value, f.name), index, seen) for f in fields(value)})
    return value

  loads, stores = {}, []
  for s in loop.body.statements:
    stores.extend(s.stores if isinstance(s, I.ElementwiseGroup) else (s,) if isinstance(s, I.Store) else ())
  catalog = context.environment.operations
  for expression in values(loop.body):
    for e in walk(expression):
      if catalog.get(e).effects not in {"none", "read"} or catalog.has(
        e, Trait.REDUCTION | Trait.STATE_REFERENCE | Trait.REGISTER_TRANSFORM | Trait.VECTOR_BOUNDARY
      ):
        return None
      if e.op == "load":
        if not isinstance(e.type, I.TileType) or e.data.owned or e.data.region is None:
          return None
        loads[id(e)] = e
  if not loads or not stores or any(s.destination.region is None or s.destination.owned for s in stores):
    return None
  index = I.Expr(I.ScalarType("int64"), "local", data=loop.target)
  next_index = I.binary("add", index, loop.step)
  allocations, prime, lookahead, substitutes, guards = [], [], [], {}, []

  def root(view):
    r = view.region
    return I.View(view.storage, view.dtype, r.root_shape, r.root_strides, I.literal(0), view.writable)

  for number, load in enumerate(loads.values()):
    logical_shape = tuple(expand(n, index) for n in load.type.shape)
    if any(
      e.op == "local" and e.data == loop.target or catalog.get(e).effects != "none"
      for n in logical_shape
      for e in walk(n)
    ):
      return None
    if any(catalog.get(e).effects != "none" for a in load.args for e in walk(a)):
      return None
    strides, count = [None] * len(logical_shape), I.literal(1)
    for a in reversed(range(len(logical_shape))):
      strides[a], count = count, I.binary("mul", count, logical_shape[a])
    name = f"dsl_pipeline_{identity}_{number}"
    full = I.View(name, load.type.dtype, (I.literal(2), *logical_shape), (count, *strides), I.literal(0), owned=True)
    allocations.append(I.Allocate(full))
    phase = I.binary("mod", I.binary("floordiv", I.binary("sub", index, loop.start), loop.step), I.literal(2))
    current = I.View(name, load.type.dtype, logical_shape, tuple(strides), I.binary("mul", phase, count), owned=True)
    following = replace(current, offset=I.binary("mul", I.binary("sub", I.literal(1), phase), count))
    prime.append(I.Store(replace(current, offset=I.literal(0)), expand(load, loop.start)))
    lookahead.append(I.Store(following, expand(load, next_index)))
    substitutes[id(load)] = I.Expr(I.TileType(load.type.dtype, logical_shape), "load", data=current)
    guards.extend(
      I.Expr(I.ScalarType("bool"), "view_disjoint", data=(root(load.data), root(s.destination))) for s in stores
    )
  memo = {}

  def rewrite(value):
    if isinstance(value, I.Expr):
      if id(value) in substitutes:
        return substitutes[id(value)]
      if id(value) not in memo:
        memo[id(value)] = replace(value, args=tuple(rewrite(a) for a in value.args))
      return memo[id(value)]
    if isinstance(value, tuple):
      return tuple(rewrite(v) for v in value)
    if isinstance(value, I.Block):
      return I.Block([rewrite(s) for s in value.statements])
    if isinstance(value, I.Node):
      return replace(value, **{f.name: rewrite(getattr(value, f.name)) for f in fields(value)})
    return value

  comparison = "lt" if I.constant(loop.step) > 0 else "gt"
  body = I.Block(
    [I.If(I.binary(comparison, next_index, loop.stop), I.Block(lookahead), I.Block()), *rewrite(loop.body).statements]
  )
  buffered = replace(loop, body=body, unroll=1)
  first = I.If(I.binary(comparison, loop.start, loop.stop), I.Block(prime), I.Block())
  condition = I.literal(True)
  for guard in guards:
    condition = I.binary("and", condition, guard)
  ir.diagnostics.append(f"CPU software pipeline: {len(loads)} two-bank loads guarded by root non-alias proof")
  return I.If(condition, I.Block([*allocations, first, buffered]), I.Block([loop]))
