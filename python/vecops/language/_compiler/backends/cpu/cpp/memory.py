"""C++ emission: memory."""

from __future__ import annotations
from .... import ir as I
from ..capabilities import CPP_TYPES


class MemoryEmitter:
  def pointer(self, view, coordinates=()):
    base = f"{view.storage}.data()"
    offset = self.scalar(view.offset)
    terms = [offset]
    for coordinate, stride in zip(coordinates, view.strides):
      terms.append(f"({coordinate}) * ({self.scalar(stride)})")
    return f"({base} + ({' + '.join(terms)}))"

  def coordinates(self, shape, root_shape, coords):
    offset = len(root_shape) - len(shape)
    if offset < 0:
      raise TypeError("a higher-rank tile cannot broadcast to a lower-rank consumer")
    result = []
    for axis, extent in enumerate(shape):
      coordinate = coords[offset + axis]
      if I.constant(extent) == 1:
        coordinate = "0"
      elif not I.same_expr(extent, root_shape[offset + axis]) and I.constant(extent) is None:
        coordinate = f"(({self.scalar(extent)}) == 1 ? 0 : ({coordinate}))"
      result.append(coordinate)
    return tuple(result)

  def full_fixed_access(self, tag, active):
    """A fixed piece is full only when its exact logical count is known.

    Do not propagate a blanket full-vector flag into scalar reductions or
    scalable/tail consumers nested in the same expression.
    """
    return tag in self.fixed_tag_lanes and str(self.fixed_tag_lanes[tag]) == str(active)

  def memory_options(self, mask, tag, root_shape, coords, active, axis):
    if mask is None:
      return (
        "vecops::vec::opt::unmasked"
        if getattr(self, "full_vector", False) or self.full_fixed_access(tag, active)
        else f"vecops::vec::opt::first({active})"
      ), None
    predicate = self.vector(mask, tag, root_shape, coords, active, axis)
    name = self.unique("memory_mask")
    self.line(
      f"auto {name} = vecops::vec::mask_and({tag}{{}}, vecops::vec::mwhilelt({tag}{{}}, 0, {active}), {predicate});"
    )
    return f"vecops::vec::opt::masked({name})", name

  def store(self, statement, *, plan=None):
    destination, value = statement.destination, statement.value
    if not destination.shape:
      if statement.mask is not None:
        self.opening(f"if ({self.scalar(statement.mask)})")
      self.line(f"*{self.pointer(destination)} = static_cast<{CPP_TYPES[destination.dtype]}>({self.scalar(value)});")
      if statement.mask is not None:
        self.close()
      return
    self.check_store(statement)
    if self.matmul_store(statement):
      return
    if self.transpose_store(statement, self.prepared.transposes.get(id(statement))):
      return
    self.streaming_store(statement, plan=plan)

  def streaming_store(self, statement, *, plan=None):
    plan = plan or self.vector_ir.plans.get(id(statement))
    if plan is None:
      plan = self.prepared.store(statement)
      self.emitted_regions.append(plan)
    elif id(statement) not in self.vector_ir.plans:
      self.emitted_regions.append(plan)
    self.traverse_region(
      plan, lambda tags, coords, active, roots, state: self.write_effects(plan, tags, coords, active, roots)
    )

  def check_store(self, statement):
    destination, value = statement.destination, statement.value
    shape = ", ".join(self.scalar(v) for v in destination.shape)
    strides = ", ".join(self.scalar(v) for v in destination.strides)
    transpose = self.prepared.transposes.get(id(statement))
    expected, dense = I.literal(1), True
    for extent, stride in zip(reversed(destination.shape), reversed(destination.strides)):
      dense = dense and I.same_expr(stride, expected)
      expected = I.binary("mul", expected, extent)
    if dense or transpose is not None and transpose.dense_destination:
      self.line(f"vecops_dsl::checked_shape({{{shape}}}); /* proven dense writable layout */")
    else:
      self.line(f"vecops_dsl::check_writable<{len(destination.shape)}>({{{shape}}}, {{{strides}}});")
    for operand in (value, statement.mask):
      if operand is None or not isinstance(operand.type, I.TileType):
        continue
      if len(operand.type.shape) > len(destination.shape):
        raise TypeError("store source cannot have a higher rank than destination")
      for source_extent, destination_extent in zip(reversed(operand.type.shape), reversed(destination.shape)):
        if I.constant(source_extent) == 1 or I.same_expr(source_extent, destination_extent):
          continue
        if source_extent.op == destination_extent.op == "literal":
          raise TypeError("store source does not broadcast to the destination shape")
        self.line(f"vecops_dsl::check_broadcast({self.scalar(source_extent)}, {self.scalar(destination_extent)});")

  def write_effects(self, region, tags, coordinates, active, roots):
    effects = [effect for effect in region.effects if effect.kind == "store_convert"]
    stores = tuple(effect.store for effect in effects)
    values = [
      value
      for effect in effects
      for value in (
        (roots[effect.value_root], roots[effect.mask_root]) if effect.mask_root >= 0 else (roots[effect.value_root],)
      )
    ]
    self.write_roots(stores, tags, coordinates, active, values)

  def write_roots(self, stores, tags, coordinates, active, roots):
    index = 0
    for statement in stores:
      values = roots[index]
      index += 1
      masks = roots[index] if statement.mask is not None else None
      if masks is not None:
        index += 1
      destination = statement.destination
      value = statement.value
      dtype = (
        value.args[0] if value.op == "cast" and value.type.dtype == statement.destination.dtype else value
      ).type.dtype
      for slot, (base, coords, count) in enumerate(zip(tags, coordinates, active)):
        tag = self.rebound_tag(dtype, base)
        self.opening(f"if ({count} > 0)")
        options = (
          "vecops::vec::opt::unmasked" if self.full_fixed_access(tag, count) else f"vecops::vec::opt::first({count})"
        )
        if masks is not None:
          mask = (
            I.Expr(statement.mask.type, "vector_ref", data=masks[slot])
            if isinstance(statement.mask.type, I.TileType)
            else statement.mask
          )
          options, _ = self.memory_options(mask, tag, destination.shape, coords, count, len(destination.shape) - 1)
        pointer = self.pointer(destination, coords)
        if I.constant(destination.strides[-1]) == 1:
          self.line(f"vecops::vec::store_convert({tag}{{}}, {pointer}, {values[slot]}, {options});")
        else:
          stride = self.scalar(destination.strides[-1])
          self.opening(f"if (({stride}) == 1)")
          self.line(f"vecops::vec::store_convert({tag}{{}}, {pointer}, {values[slot]}, {options});")
          self.close()
          self.opening("else")
          self.line(
            f"vecops::vec::store_convert({tag}{{}}, {pointer}, {values[slot]}, {options}, vecops::vec::opt::strided({stride}));"
          )
          self.close()
        self.close()

  def expression_loads(self, block):
    result = {}
    for statement in block.statements:
      if isinstance(statement, I.ReductionGroup):
        result.update(self.expression_loads(I.Block([*statement.assignments, *statement.stores])))
      if isinstance(statement, I.ElementwiseGroup):
        result.update(self.expression_loads(I.Block(list(statement.stores))))
      if isinstance(statement, I.UnrolledFor):
        result.update(self.expression_loads(statement.copies))
        result.update(self.expression_loads(statement.loop.body))
      for value in (
        getattr(statement, "value", None),
        getattr(statement, "condition", None),
        getattr(statement, "mask", None),
        getattr(statement, "indices", None),
      ):
        if isinstance(value, I.Expr):
          for load in I.loads(value):
            result[id(load)] = load
      for name in ("body", "otherwise"):
        child = getattr(statement, name, None)
        if isinstance(child, I.Block):
          result.update(self.expression_loads(child))
    return result

  def overlap(self, source, destination):
    def span(view):
      shape = ", ".join(self.scalar(v) for v in view.shape)
      strides = ", ".join(self.scalar(v) for v in view.strides)
      return f"vecops_dsl::span({self.pointer(view)}, {{{shape}}}, {{{strides}}}, sizeof({CPP_TYPES[view.dtype]}))"

    return f"vecops_dsl::overlap({span(source)}, {span(destination)})"

  def same_mapping(self, a, b):
    if len(a.shape) != len(b.shape) or a.dtype != b.dtype:
      return "false"
    terms = [f"static_cast<const void*>({self.pointer(a)}) == static_cast<const void*>({self.pointer(b)})"]
    terms += [f"({self.scalar(x)}) == ({self.scalar(y)})" for x, y in zip(a.shape, b.shape)]
    terms += [f"({self.scalar(x)}) == ({self.scalar(y)})" for x, y in zip(a.strides, b.strides)]
    return "(" + " && ".join(terms) + ")"

  def snapshot(self, load, destination, must_preserve):
    view = load.data
    if self.verified_grid is not None and view.storage != destination.storage:
      # The enclosing runtime guard proved disjoint parameter root spans.
      # Worker-private allocations cannot alias a parameter or another slot.
      known = self.verified_grid.reads | self.verified_grid.writes | self.verified_grid.private
      if view.storage in known and destination.storage in known:
        return
    if not view.shape or (view.owned or destination.owned) and view.storage != destination.storage:
      return
    condition = self.unique("alias")
    overlap = self.overlap(view, destination)
    if not must_preserve:
      overlap += f" && !{self.same_mapping(view, destination)}"
    old = self.snapshots.get(id(load))
    if old is not None:
      overlap = f"({old[1]}) || ({overlap})"
    nonempty = " && ".join(f"({self.scalar(n)}) > 0" for n in view.shape + destination.shape)
    self.line(f"const bool {condition} = ({nonempty}) && ({overlap});")
    name = self.unique("snapshot")
    dtype = load.type.dtype
    self.line(f"vecops_dsl::Buffer<{CPP_TYPES[dtype]}> {name};")
    shape = load.type.shape
    statement, plan = self.prepared.snapshot(load, name)
    saved = statement.destination
    self.opening(f"if ({condition})")
    count = ", ".join(self.scalar(v) for v in shape)
    self.line(f"{name}.resize(vecops_dsl::checked_shape({{{count}}}));")
    self.store(statement, plan=plan)
    self.close()
    self.snapshots[id(load)] = (saved, condition)

  def protect_store(self, statement, future):
    current = {id(load): load for load in I.loads(statement.value)}
    if statement.mask is not None:
      current.update({id(load): load for load in I.loads(statement.mask)})
    preserved = set(self.active_loads) & set(future)
    # Matrix inputs are re-read across output tiles. Even identical mappings
    # are not an elementwise-safe in-place update (C=A/B needs a snapshot).
    from ....ir.traversal import walk

    matrix_inputs = {
      id(load)
      for value in walk(statement.value)
      if value.op == "dot"
      for operand in value.args[:2]
      for load in I.loads(operand)
    }
    transformed_inputs = {
      id(load) for value in walk(statement.value) if value.op == "value_transform" for load in I.loads(value)
    }
    for identity in preserved:
      current[identity] = self.active_loads[identity]
    for identity, load in current.items():
      self.snapshot(
        load,
        statement.destination,
        identity in preserved or identity in matrix_inputs or identity in transformed_inputs or load.op == "gather",
      )

  def stores(self, block):
    for statement in block.statements:
      if isinstance(statement, (I.Store, I.IndexedStore)):
        yield statement
      if isinstance(statement, (I.ReductionGroup, I.ElementwiseGroup)):
        yield from statement.stores
      if isinstance(statement, I.UnrolledFor):
        yield from self.stores(statement.loop.body)
      for name in ("body", "otherwise"):
        child = getattr(statement, name, None)
        if isinstance(child, I.Block):
          yield from self.stores(child)

  def protect_region(self, region, future):
    dependencies = self.expression_loads(region)
    dependencies.update(future)
    destinations = {s.destination.storage for s in self.stores(region)}
    for identity in set(self.active_loads) & set(dependencies):
      for destination in destinations:
        root = self.storage_views.get(destination)
        if root is not None:
          self.snapshot(self.active_loads[identity], root, True)

  def used_storage(self, block):
    result = set()
    for statement in block.statements:
      if isinstance(statement, I.Pack):
        result.add(statement.destination.view.storage)
        result.update(load.data.storage for load in I.loads(statement.source))
      if isinstance(statement, I.ReductionGroup):
        result.update(self.used_storage(I.Block([*statement.assignments, *statement.stores])))
      if isinstance(statement, I.ElementwiseGroup):
        result.update(self.used_storage(I.Block(list(statement.stores))))
      if isinstance(statement, I.UnrolledFor):
        result.update(self.used_storage(statement.copies))
        result.update(self.used_storage(statement.loop.body))
      for value in (
        getattr(statement, "value", None),
        getattr(statement, "condition", None),
        getattr(statement, "mask", None),
        getattr(statement, "indices", None),
      ):
        if isinstance(value, I.Expr):
          result.update(load.data.storage for load in I.loads(value))
      if isinstance(statement, (I.Store, I.IndexedStore)):
        result.add(statement.destination.storage)
      if isinstance(statement, I.ArchitectureCall) and statement.view is not None:
        result.add(statement.view.storage)
      for name in ("body", "otherwise"):
        child = getattr(statement, name, None)
        if isinstance(child, I.Block):
          result.update(self.used_storage(child))
    return result
