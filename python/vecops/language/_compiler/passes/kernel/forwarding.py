# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Forward exact private tile round trips using complete ownership/use proofs."""

from dataclasses import fields

from ... import ir as I
from ...analysis.effects import ValueEffects
from ...ir.traversal import children
from ...operations import Trait
from ..manager import PassResult, register_pass


@register_pass(
  "forward-private-stores", stage="kernel-cleanup", after="dead-store-elimination", before="dead-code-elimination"
)
def forward_private_stores(ir, context):
  """Forward an adjacent, single-consumer private tile through its store.

  Exact dense storage and equal element types make the store/load round trip
  an identity. All other views/uses escape; masks, conversion, control flow,
  metadata reads and historical captures before the write remain boundaries.
  The original producer read objects survive substitution, so ordinary alias
  protection still snapshots them before a possibly overlapping output write.
  """
  enabled = context.options.get("private_store_forwarding", True)
  if type(enabled) is not bool:
    raise TypeError("private_store_forwarding must be bool")
  if not enabled:
    return PassResult(ir, changed=False)
  catalog = context.environment.operations
  effects = ValueEffects(catalog)
  parents, views, subjects = {}, {}, {}
  containers, active_containers = set(), set()
  active_nodes = set()
  builtin_nodes = frozenset(
    value for value in vars(I).values() if isinstance(value, type) and issubclass(value, I.Node)
  )
  opaque = False

  def collect(value, owner=None, field=None):
    nonlocal opaque
    if value is None or type(value) in (bool, int, float, str, bytes):
      return
    if type(value) in (tuple, list, dict):
      key = id(value), id(owner), field
      if id(value) in active_containers:
        opaque = True
        return
      if key not in containers:
        containers.add(key)
        active_containers.add(id(value))
        if type(value) is dict:
          for name, item in value.items():
            collect(name, owner, field)
            collect(item, owner, field)
        else:
          for item in value:
            collect(item, owner, field)
        active_containers.remove(id(value))
      return
    if type(value) not in builtin_nodes:
      opaque = True
      return
    identity = id(value)
    if owner is not None:
      parents.setdefault(identity, []).append((owner, field))
    if identity in subjects:
      opaque |= identity in active_nodes
      return
    subjects[identity] = value
    active_nodes.add(identity)
    if type(value) is I.View:
      views.setdefault(value.storage, []).append(value)
    for descriptor in fields(value):
      collect(getattr(value, descriptor.name), value, descriptor.name)
    active_nodes.remove(identity)

  collect(ir.body)
  for name in ("bindings", "materializations", "vector_regions"):
    collect(getattr(ir, name), ir, name)
  if opaque:
    return PassResult(ir, changed=False)
  for value in subjects.values():
    if type(value) is I.Expr and value.op == "raw" and not effects.discardable(value):
      return PassResult(ir, changed=False)
    if type(value) is I.Control and value.kind not in ('asm volatile("" ::: "memory")', "break", "continue", "return"):
      return PassResult(ir, changed=False)

  def dense(view):
    stride = 1
    if I.constant(view.offset) != 0 or not view.shape:
      return False
    for extent, actual in reversed(tuple(zip(view.shape, view.strides))):
      n = I.constant(extent)
      if type(n) is not int or n < 0 or I.constant(actual) != stride:
        return False
      stride *= n
      if stride > (1 << 63) - 1:
        return False
    return True

  def same_view(a, b):
    return (
      a.dtype == b.dtype
      and len(a.shape) == len(b.shape)
      and len(a.strides) == len(b.strides)
      and all(
        I.same_expr(x, y)
        for first, second in ((a.shape, b.shape), (a.strides, b.strides), ((a.offset,), (b.offset,)))
        for x, y in zip(first, second, strict=True)
      )
    )

  pointwise_values = {}

  def pointwise(value):
    if value in pointwise_values:
      return pointwise_values[value]
    pointwise_values[value] = False
    definition = catalog.get(value)
    owned = effects.owned_expressions(value)
    if type(value.type) not in (I.TileType, I.ScalarType) or owned is None:
      return False
    if type(value.type) is I.ScalarType and not effects.discardable(value):
      return False
    if value.op == "load":
      if (
        value.args
        or type(value.data) is not I.View
        or type(value.type) is not I.TileType
        or definition.effects != "read"
        or not catalog.has(value, Trait.MEMORY_READ)
      ):
        return False
    elif (
      definition.effects != "none"
      or definition.may_trap
      or (type(value.type) is I.TileType and not catalog.has(value, Trait.ELEMENTWISE))
    ):
      return False
    if catalog.has(value, Trait.REDUCTION | Trait.REGISTER_TRANSFORM | Trait.STATE_REFERENCE | Trait.VECTOR_BOUNDARY):
      return False
    pointwise_values[value] = all(effects.discardable(v) for v in owned) and all(pointwise(a) for a in value.args)
    return pointwise_values[value]

  def private_load(producer):
    destination = producer.destination
    if not destination.owned or not dense(destination):
      return None
    allocations, loads = [], []
    for view in views.get(destination.storage, ()):
      if not view.owned or len(view.shape) != len(destination.shape) or not same_view(view, destination):
        return None
      for owner, field in parents.get(id(view), ()):
        if type(owner) is I.Allocate and field == "view":
          allocations.append(owner)
        elif owner is producer and field == "destination":
          continue
        elif type(owner) is I.Expr and owner.op == "load" and field == "data" and not owner.args:
          loads.append(owner)
        else:
          return None
    if len(allocations) != 1 or len(loads) != 1 or len(parents.get(id(allocations[0]), ())) != 1:
      return None
    load = loads[0]
    if (
      type(load.type) is not I.TileType
      or type(producer.value.type) is not I.TileType
      or load.type.dtype != destination.dtype
      or producer.value.type.dtype != destination.dtype
      or load.type.mask
      or producer.value.type.mask
      or len(load.type.shape) != len(destination.shape)
      or len(producer.value.type.shape) != len(destination.shape)
      or not all(I.same_expr(a, b) for a, b in zip(load.type.shape, destination.shape, strict=True))
      or not all(I.same_expr(a, b) for a, b in zip(producer.value.type.shape, destination.shape, strict=True))
    ):
      return None
    return load

  def consumed_here(load, consumer, bindings):
    pending, seen, found = [load], set(), False
    while pending:
      value = pending.pop()
      if id(value) in seen:
        continue
      seen.add(id(value))
      for owner, field in parents.get(id(value), ()):
        if owner is consumer and field == "value":
          found = True
        elif type(owner) is I.BindTile and id(owner) in bindings and field == "value":
          continue
        elif type(owner) is I.Expr and field == "args":
          pending.append(owner)
        else:
          return False
    return found

  changed = False

  def visit(body):
    nonlocal changed
    output = []
    index = 0
    while index < len(body.statements):
      producer = body.statements[index]
      for child in children(producer):
        visit(child)
      end = index + 1
      while end < len(body.statements) and type(body.statements[end]) is I.BindTile:
        end += 1
      consumer = body.statements[end] if end < len(body.statements) else None
      if (
        type(producer) is I.Store
        and type(consumer) is I.Store
        and len(parents.get(id(producer), ())) == len(parents.get(id(consumer), ())) == 1
        and producer.mask is None
        and consumer.mask is None
        and producer.implementation == consumer.implementation == "auto"
        and producer.vector_extent is None
        and consumer.vector_extent is None
        and pointwise(producer.value)
        and pointwise(consumer.value)
      ):
        load = private_load(producer)
        bindings = {id(s) for s in body.statements[index + 1 : end]}
        if (
          load is not None
          and consumer.destination.dtype == producer.destination.dtype
          and type(consumer.value.type) is I.TileType
          and consumer.value.type.dtype == producer.destination.dtype
          and len(consumer.destination.shape) == len(producer.destination.shape)
          and all(I.same_expr(a, b) for a, b in zip(consumer.destination.shape, producer.destination.shape))
          and consumed_here(load, consumer, bindings)
        ):
          # Rewrite only the proved consumer and its intervening captures.
          # Global replacement could change a different dynamic occurrence.
          from ...ir.traversal import rewrite_expressions

          fragment = I.KernelIR(I.Block(body.statements[index + 1 : end + 1]), {}, {})
          rewrite_expressions(
            fragment, lambda value, producer=producer, load=load: producer.value if value is load else value
          )
          output.extend(fragment.body.statements)
          changed = True
          index = end + 1
          continue
      output.append(producer)
      index += 1
    body.statements = output

  visit(ir.body)
  return PassResult(ir, changed=changed)
