# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""State C++ emission. Register pieces are locals, including sizeless SVE.

All placement and vector regions were prepared before emission. Memory is a
deliberate placement fallback, never an array of Vec or predicate objects.
"""

from contextlib import contextmanager
from dataclasses import replace

from .... import ir as I
from ....dtypes import ELEMENT_BYTES
from ....ir.traversal import walk
from ....operations import Trait
from ..capabilities import CPP_TYPES


class StateEmitter:
  @contextmanager
  def register_state_access(self, plan):
    """Apply a complete-piece proof only inside this register traversal.

    This includes its stores: a complete scalable piece must not reintroduce
    a runtime first-lanes predicate while draining already complete state.
    """
    reuse = self.vector_ir.manager.options.get("cpu_state_vector_reuse", False)
    if type(reuse) is not bool:
      raise TypeError("cpu_state_vector_reuse must be bool")
    full = all(not piece.scalable or piece.full for piece in plan.pieces)
    previous = getattr(self, "full_vector", False)
    if reuse:
      self.full_vector = full
    try:
      yield reuse, full
    finally:
      self.full_vector = previous

  def state_tag(self, dtype, scalable, extent):
    kind = "ScalableTag" if scalable else "FixedTag"
    tag = f"vecops::vec::{kind}<{CPP_TYPES[dtype]}, {extent}>"
    self.tag_dtypes[tag] = dtype
    self.tag_geometry[tag] = scalable, extent
    if not scalable:
      self.fixed_tag_lanes[tag] = extent
    return tag

  def state_tags(self, plan, dtype):
    return tuple(
      self.state_tag(
        dtype,
        piece.scalable,
        piece.power + ELEMENT_BYTES[dtype].bit_length() - ELEMENT_BYTES[plan.state.type.dtype].bit_length()
        if piece.scalable
        else piece.lanes,
      )
      for piece in plan.pieces
    )

  def same_register_tag(self, first, second):
    """Compare physical geometry, including aliases made by vector traversal."""
    return first == second or (
      first in self.tag_geometry
      and second in self.tag_geometry
      and self.tag_geometry[first] == self.tag_geometry[second]
      and self.tag_dtypes[first] == self.tag_dtypes[second]
    )

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
      else f"vecops_dsl::clamp_value<vecops::nint_t>(({self.scalar(plan.state.type.shape[-1])}) - ({c[-1]}), 0, {self.vector_size(tag)})"
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
      with self.register_state_access(plan) as (reuse, full):
        roots = self.emit_operations(register_region, tags, coords, counts, full=full, reuse_projected=reuse)[0]
      for index, root in enumerate(roots):
        name = f"{state.name}_r{index}"
        self.line(f"{'auto ' if isinstance(statement, I.DeclareState) else ''}{name} = {root};")
    else:
      if isinstance(statement, I.DeclareState):
        count = ", ".join(self.scalar(s) for s in state.type.shape)
        self.allocate_buffer(plan.view.dtype, state.name, self.checked_shape(count))
      # A private staging buffer keeps all old-state reads alive until an
      # update finishes, including non-elementwise recurrences/alias copies.
      destination = plan.view
      if isinstance(statement, I.UpdateState):
        name = self.unique("state_next")
        destination = I.View(name, plan.view.dtype, plan.view.shape, plan.view.strides, plan.view.offset, owned=True)
        count = ", ".join(self.scalar(s) for s in state.type.shape)
        self.allocate_buffer(destination.dtype, name, self.checked_shape(count))
      effects = tuple(replace(e, store=I.Store(destination, e.store.value)) for e in region.effects)
      proxy = replace(region, effects=effects)
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
      or not all(self.same_geometry(a, b) for a, b in zip(region.shape, p.state.type.shape))
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
    with self.register_state_access(plan) as (reuse, full):
      state = setup(tags) if setup else None
      roots = self.emit_operations(region, tags, coords, counts, full=full, reuse_projected=reuse)
      callback(tags, coords, counts, roots, state)
      if finish:
        finish(tags, state)

  def fixed_register_prefix(self, plan, value, tag, root_shape, own_coords, active, axis):
    """Read a complete fixed prefix without extracting individual lanes.

    X86 fixed and scalable tags at the native width share a representation.
    Use that bridge only for a zero-origin, fully active fixed vector; the
    public fixed-to-fixed bitcast truncates its low bytes. ARM sizeless
    vectors and shifted, masked or partially active reads retain their
    existing projection paths.
    """
    target = self.environment.codegen_target
    wanted = self.fixed_tag_lanes.get(tag)
    if not (
      target.architecture == "x86"
      and target.vector_bits
      and not value.type.mask
      and len(plan.pieces) == 1
      and plan.pieces[0].scalable
      and plan.pieces[0].coordinates == (0,)
      and plan.pieces[0].power == 0
      and len(value.type.shape) == len(root_shape) == 1
      and axis in (-1, 0)
      and own_coords == ("0",)
      and type(wanted) is int
      and I.constant(value.type.shape[0]) == wanted
      and str(active) == str(wanted)
      and value.type.dtype == plan.state.type.dtype
      and self.tag_dtypes.get(tag) == value.type.dtype
    ):
      return None
    native = target.vector_bits // (8 * ELEMENT_BYTES[value.type.dtype])
    if not 0 < wanted <= native:
      return None
    source_tag = f"vecops::vec::FixedTag<{CPP_TYPES[value.type.dtype]}, {native}>"
    register = f"{value.data.name}_r0"
    self.line(
      f"static_assert(std::is_same_v<vecops::vec::Vec<{source_tag}>, std::remove_cvref_t<decltype({register})>>);"
    )
    return f"vecops::vec::bitcast({tag}{{}}, {source_tag}{{}}, {register}) /* fixed register prefix */"

  def state_vector(self, value, tag, root_shape, coords, active, axis):
    plan = self.prepared.states[value.data.name]
    if not plan.pieces:
      loaded = I.Expr(I.TileType(value.type.dtype, value.type.shape), "load", data=plan.view)
      result = self.vector(loaded, tag, root_shape, coords, active, axis)
      if value.type.mask:
        name = self.unique("state_predicate")
        self.line(f"auto {name} = vecops::vec::cmpne({tag}{{}}, {result}, vecops::vec::zeros({tag}{{}}));")
        return name
      return result
    own_coords = self.coordinates(value.type.shape, root_shape, coords)
    own_tags = self.state_tags(plan, value.type.dtype)
    for index, (piece, own_tag, coordinate) in enumerate(zip(plan.pieces, own_tags, self.state_coordinates(plan))):
      if (
        own_coords == coordinate
        and axis == len(root_shape) - 1
        and (I.constant(value.type.shape[-1]) != 1 or self.same_geometry(value.type.shape[-1], root_shape[-1]))
        and (
          self.same_register_tag(tag, own_tag)
          or tag == f"vecops::vec::Rebind<{CPP_TYPES[value.type.dtype]}, {own_tag}>"
        )
      ):
        return f"{value.data.name}_r{index}"
    prefix = self.fixed_register_prefix(plan, value, tag, root_shape, own_coords, active, axis)
    if prefix is not None:
      return prefix
    # A dynamic logical prefix is still one physical register. Generic store
    # traversal can hide its zero origin behind a loop variable or broadcast
    # projection. Version that geometry rather than reconstruct every lane;
    # genuinely shifted traversals retain the scalar correctness fallback.
    word = (
      len(plan.pieces) == 1
      and len(value.type.shape) == 1
      and plan.pieces[0].scalable
      and plan.pieces[0].coordinates == (0,)
      and axis == len(root_shape) - 1
      and (
        self.same_register_tag(tag, own_tags[0])
        or tag == f"vecops::vec::Rebind<{CPP_TYPES[value.type.dtype]}, {own_tags[0]}>"
      )
    )
    # Other traversal axes and mixed piece widths use scalar lane extraction.
    # This is a correctness fallback, not an implicit Vec array/spill.
    name, lane = self.unique("state_read"), self.unique("state_lane")
    representation, fill = ("Mask", "mfill") if value.type.mask else ("Vec", "fill")
    zero = f"vecops::vec::mfill({tag}{{}}, false)" if value.type.mask else f"vecops::vec::zeros({tag}{{}})"
    self.line(f"vecops::vec::{representation}<{tag}> {name} = {zero};")
    if word:
      extent = self.scalar(value.type.shape[-1])
      register = f"{value.data.name}_r0"
      singleton = f"vecops::vec::{fill}({tag}{{}}, vecops::vec::get({own_tags[0]}{{}}, {register}, 0))"
      self.opening(f"if (({extent}) == 1 || ({coords[-1]}) == 0) /* physical register prefix */")
      self.line(f"{name} = ({extent}) == 1 ? {singleton} : {register};")
      self.close()
      self.opening("else /* shifted register projection */")
    self.opening(f"for (vecops::nint_t {lane}=0; {lane}<({active}); ++{lane})")
    own_axis = axis - (len(root_shape) - len(value.type.shape))
    indices = []
    offset = len(root_shape) - len(value.type.shape)
    for a, c in enumerate(own_coords):
      extent = I.constant(value.type.shape[a])
      if a != own_axis or extent == 1:
        indices.append(c)
      elif extent is None and not self.same_geometry(value.type.shape[a], root_shape[offset + a]):
        # Broadcasting a singleton register does not advance its source lane.
        # Projection of the vector origin alone cannot express that stride.
        indices.append(f"(({self.scalar(value.type.shape[a])}) == 1 ? 0 : (({c}) + {lane}))")
      else:
        indices.append(f"(({c}) + {lane})")
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
    if word:
      self.close()
    return name
