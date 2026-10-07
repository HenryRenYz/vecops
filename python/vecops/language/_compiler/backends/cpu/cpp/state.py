"""State C++ emission. Register pieces are locals, including sizeless SVE.

All placement and vector regions were prepared before emission. Memory is a
deliberate placement fallback, never an array of Vec or predicate objects.
"""

from dataclasses import replace

from .... import ir as I
from ....ir.traversal import walk
from ....operations import Trait
from ..capabilities import CPP_TYPES


class StateEmitter:
  def state_tags(self, plan, dtype):
    tags = []
    for piece in plan.pieces:
      if piece.scalable:
        origin = f"vecops::vec::ScalableTag<{CPP_TYPES[plan.state.type.dtype]}, {piece.power}>"
        tag = f"vecops::vec::Rebind<{CPP_TYPES[dtype]}, {origin}>" if dtype != plan.state.type.dtype else origin
      else:
        origin = f"vecops::vec::FixedTag<{CPP_TYPES[plan.state.type.dtype]}, {piece.lanes}>"
        tag = f"vecops::vec::FixedTag<{CPP_TYPES[dtype]}, {piece.lanes}>"
        self.fixed_tag_lanes[tag] = piece.lanes
        self.fixed_tag_lanes[origin] = piece.lanes
      self.tag_dtypes[origin] = plan.state.type.dtype
      self.state_tag_origins[tag] = origin
      self.state_tag_origins[origin] = origin
      tags.append(tag)
    return tuple(tags)

  def state_coordinates(self, plan):
    tags = self.state_tags(plan, plan.state.type.dtype)
    return tuple(
      tuple(str(c) for c in p.coordinates[:-1])
      + ("0" if p.coordinates[-1] == 0 else f"({p.coordinates[-1]} * {self.vector_size(tag)})",)
      if p.scalable
      else tuple(str(c) for c in p.coordinates)
      for p, tag in zip(plan.pieces, tags)
    )

  def state_counts(self, plan):
    return tuple(
      str(piece.lanes)
      if piece.lanes
      else f"std::clamp<vecops::nint_t>(({self.scalar(plan.state.type.shape[-1])}) - ({c[-1]}), 0, {self.vector_size(tag)})"
      for piece, tag, c in zip(plan.pieces, self.state_tags(plan, plan.state.type.dtype), self.state_coordinates(plan))
    )

  def emit_state(self, statement):
    state = statement.state
    plan = self.prepared.states[state.name]
    self.storage_views[state.name] = plan.view
    for a, b in zip(state.type.shape, statement.value.type.shape):
      if not I.same_expr(a, b):
        self.line(
          f'if (({self.scalar(a)}) != ({self.scalar(b)})) throw std::invalid_argument("tile state shape changes");'
        )
    region = self.prepared.state_regions[id(statement)]
    self.emitted_regions.append(region)
    if plan.pieces:
      tags = self.state_tags(plan, state.type.dtype)
      for tag in tags:
        self.tag_dtypes[tag] = state.type.dtype
      coords = self.state_coordinates(plan)
      counts = self.state_counts(plan)
      # Evaluating every piece before assignment preserves simultaneous reads
      # when an update consumes another lane/row of the old state.
      self.chunk_cache = {}
      register_region = self.prepared.state_regions[(id(statement), "register")]
      roots = self.emit_operations(
        register_region, tags, coords, counts, full=all(not p.scalable or p.full for p in plan.pieces)
      )[0]
      for index, root in enumerate(roots):
        name = f"{state.name}_r{index}"
        self.line(f"{'auto ' if isinstance(statement, I.DeclareState) else ''}{name} = {root};")
    else:
      if isinstance(statement, I.DeclareState):
        count = ", ".join(self.scalar(s) for s in state.type.shape)
        self.allocate_buffer(plan.view.dtype, state.name, f"vecops_dsl::checked_shape({{{count}}})")
      # A private staging buffer keeps all old-state reads alive until an
      # update finishes, including non-elementwise recurrences/alias copies.
      destination = plan.view
      if isinstance(statement, I.UpdateState):
        name = self.unique("state_next")
        destination = I.View(name, plan.view.dtype, plan.view.shape, plan.view.strides, plan.view.offset, owned=True)
        count = ", ".join(self.scalar(s) for s in state.type.shape)
        self.allocate_buffer(destination.dtype, name, f"vecops_dsl::checked_shape({{{count}}})")
      effects = tuple(replace(e, store=I.Store(destination, e.store.value)) for e in region.effects)
      proxy = replace(region, effects=effects)
      if not self.matmul_store(I.Store(destination, statement.value)):
        self.traverse_region(
          proxy, lambda tags, coords, active, roots, _: self.write_effects(proxy, tags, coords, active, roots)
        )
      if isinstance(statement, I.UpdateState):
        self.line(f"std::swap({state.name}, {destination.storage}); /* simultaneous tile state update */")
    self.chunk_cache = {}
    self.materialized.clear()
    self.reduction_materialized.clear()

  def state_region(self, region):
    """Use a static register schedule when a consumer has the same extent."""
    plans = [
      self.prepared.states[e.data.name]
      for root in region.roots
      for e in walk(root)
      if self.environment.operations.has(e, Trait.STATE_REFERENCE)
    ]
    if not plans or not all(p.pieces for p in plans) or region.iteration_axis is not None:
      return None
    if any(
      len(region.shape) != len(p.state.type.shape)
      or not all(I.same_expr(a, b) for a, b in zip(region.shape, p.state.type.shape))
      for p in plans
    ):
      return None
    if any(p.pieces != plans[0].pieces for p in plans):
      return None
    return plans[0]

  def traverse_state_region(self, plan, region, callback, setup, finish):
    tags = self.state_tags(plan, region.dtype)
    for tag in tags:
      self.tag_dtypes[tag] = region.dtype
    coords, counts = self.state_coordinates(plan), self.state_counts(plan)
    state = setup(tags) if setup else None
    roots = self.emit_operations(region, tags, coords, counts, full=all(not p.scalable or p.full for p in plan.pieces))
    callback(tags, coords, counts, roots, state)
    if finish:
      finish(tags, state)

  def state_vector(self, value, tag, root_shape, coords, active, axis):
    plan = self.prepared.states[value.data.name]
    if not plan.pieces:
      loaded = I.Expr(I.TileType(value.type.dtype, value.type.shape), "load", data=plan.view)
      result = self.vector(loaded, tag, root_shape, coords, active, axis)
      if value.type.mask:
        name = self.unique("state_predicate")
        self.line(f"auto {name} = vecops::vec::cmpne({tag}{{}}, {result}, vecops::vec::fill({tag}{{}}, 0));")
        return name
      return result
    own_coords = self.coordinates(value.type.shape, root_shape, coords)
    own_tags = self.state_tags(plan, value.type.dtype)
    for index, (piece, own_tag, coordinate) in enumerate(zip(plan.pieces, own_tags, self.state_coordinates(plan))):
      if (
        own_coords == coordinate
        and axis == len(root_shape) - 1
        and (tag == own_tag or tag == f"vecops::vec::Rebind<{CPP_TYPES[value.type.dtype]}, {own_tag}>")
      ):
        return f"{value.data.name}_r{index}"
    # Other traversal axes and mixed piece widths use scalar lane extraction.
    # This is a correctness fallback, not an implicit Vec array/spill.
    name, lane = self.unique("state_read"), self.unique("state_lane")
    representation, fill = ("Mask", "mfill") if value.type.mask else ("Vec", "fill")
    self.line(f"vecops::vec::{representation}<{tag}> {name} = vecops::vec::{fill}({tag}{{}}, 0);")
    self.opening(f"for (vecops::nint_t {lane}=0; {lane}<({active}); ++{lane})")
    own_axis = axis - (len(root_shape) - len(value.type.shape))
    indices = tuple(f"(({c}) + {lane})" if a == own_axis else c for a, c in enumerate(own_coords))
    for index, (piece, own_tag, coordinate) in enumerate(zip(plan.pieces, own_tags, self.state_coordinates(plan))):
      conditions = [f"({c})==({p})" for c, p in zip(indices[:-1], coordinate[:-1])]
      offset = coordinate[-1]
      width = self.vector_size(own_tag) if piece.scalable else str(piece.lanes)
      conditions.append(f"({indices[-1]})>=({offset}) && ({indices[-1]})<({offset})+({width})")
      self.opening("if (" + " && ".join(conditions) + ")")
      get, set_ = "get", "set"
      element = f"vecops::vec::{get}({own_tag}{{}}, {value.data.name}_r{index}, ({indices[-1]})-{offset})"
      self.line(f"{name}=vecops::vec::{set_}({tag}{{}}, {name}, {lane}, {element});")
      self.close()
    self.close()
    return name
