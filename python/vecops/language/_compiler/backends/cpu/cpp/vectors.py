"""C++ emission: vectors."""

from __future__ import annotations

from contextlib import contextmanager

from .... import ir as I
from ....dtypes import ELEMENT_BYTES
from ....ir.vector import VectorGroup
from ..capabilities import CPP_TYPES


class VectorsEmitter:
  @contextmanager
  def pointwise_layout(self, region, groups, *, full):
    """Keep a private memory permutation within exactly one complete body."""
    previous = getattr(self, "pointwise_register_layout", None)
    proof = self.prepared.register_layouts.get(id(region))
    self.pointwise_register_layout = (
      proof
      if full
      and proof is not None
      and self.hardware_scope is None
      and all(group.power >= 1 for group in groups)
      and not any(
        identity in mapping
        for identity in proof.expressions
        for mapping in (self.snapshots, self.materialized, self.reduction_materialized)
      )
      else None
    )
    try:
      if self.pointwise_register_layout is not None:
        self.line("/* private pointwise register layout: paired native 16->32 conversions */")
      yield
    finally:
      self.pointwise_register_layout = previous

  def vector(self, value, tag, root_shape, coords, active, axis=-1):
    self.retain_expression(value)
    axis %= len(root_shape)
    key = (id(value), tag, tuple(coords), axis)
    if key in self.chunk_cache:
      return self.chunk_cache[key]
    cached = self.materialized.pop(id(value), None)
    if cached is not None:
      view, condition = cached
      # Keep recomputation inside the false arm, not eagerly before selection.
      name = self.unique("reused")
      previous_cache = self.chunk_cache
      self.chunk_cache = dict(previous_cache)
      representation = "Mask" if value.type.mask else "Vec"
      self.line(f"vecops::vec::{representation}<{tag}> {name};")
      self.opening(f"if ({condition})")
      load = I.Expr(I.TileType(view.dtype, view.shape), "load", data=view)
      result = self.vector(load, tag, root_shape, coords, active, axis)
      self.line(f"{name} = {result};")
      self.close()
      self.chunk_cache = dict(previous_cache)
      self.opening("else")
      result = self.vector(value, tag, root_shape, coords, active, axis)
      self.line(f"{name} = {result};")
      self.close()
      # Branch-local intermediates cannot be referenced by later consumers.
      self.chunk_cache = previous_cache
      self.materialized[id(value)] = cached
      self.chunk_cache[key] = name
      return name
    dtype = value.type.dtype
    if value.op == "state_ref":
      result = self.state_vector(value, tag, root_shape, coords, active, axis)
      self.chunk_cache[key] = result
      return result
    if isinstance(value.type, I.ScalarType):
      if dtype == "bool":
        return f"vecops::vec::mfill({tag}{{}}, {self.scalar(value)})"
      return f"vecops::vec::fill({tag}{{}}, static_cast<vecops::vec::ElementOf<{tag}>>({self.scalar(value)}))"
    if dtype != self.tag_dtypes[tag]:
      own_tag = self.rebound_tag(dtype, tag)
      original = self.vector(value, own_tag, root_shape, coords, active, axis)
      name = self.unique("convert")
      self.line(f"auto {name} = vecops::vec::convert({tag}{{}}, {own_tag}{{}}, {original});")
      self.chunk_cache[key] = name
      return name
    if value.op == "vector_ref":
      return value.data
    vcoords = self.coordinates(value.type.shape, root_shape, coords)
    primitive = self.primitive_vector(value, tag, root_shape, coords, active, axis)
    if primitive is not NotImplemented:
      name = self.unique("primitive")
      self.line(f"auto {name} = {primitive};")
      self.chunk_cache[key] = name
      return name
    if value.op == "packed_load":
      result = self.packed_vector(value, tag, root_shape, coords, active, axis)
      self.chunk_cache[key] = result
      return result
    if value.op == "dot":
      register = getattr(self, "dot_registers", {}).get(id(value))
      if register is None:
        raise TypeError("matrix result requires a fused rank-two store")
      self.chunk_cache[key] = register[1]
      return register[1]
    if value.op == "load":
      view = value.data
      own_coords = self.coordinates(view.shape, root_shape, coords)
      pointer = self.pointer(view, own_coords)
      view_axis = axis - (len(root_shape) - len(view.shape))
      for operand in value.args:
        if isinstance(operand.type, I.TileType):
          if len(operand.type.shape) > len(value.type.shape):
            raise TypeError("load mask/inactive value cannot exceed the loaded region's rank")
          for a, b in zip(reversed(operand.type.shape), reversed(value.type.shape)):
            if not getattr(self, "matrix_validated", False) and I.constant(a) != 1 and not I.same_expr(a, b):
              self.line(f"vecops_dsl::check_broadcast({self.scalar(a)}, {self.scalar(b)});")
      broadcast = (
        view_axis < 0
        or self.fixed_tag_lanes.get(tag) == 1
        and str(active) == "1"
        and self.environment.target.architecture != "x86"
        or I.constant(view.shape[view_axis]) == 1
        and not I.same_expr(view.shape[view_axis], root_shape[axis])
      )
      options, mask = self.memory_options(value.args[0] if value.args else None, tag, root_shape, coords, active, axis)
      options += self.conversion_layout_option()
      other = None
      if value.args:
        other = self.unique("inactive")
        inactive = self.vector(value.args[1], tag, root_shape, coords, active, axis)
        self.line(f"auto {other} = {inactive};")
        options += f", vecops::vec::opt::merge({other})"
      if broadcast:
        expression = f"vecops::vec::fill({tag}{{}}, static_cast<vecops::vec::ElementOf<{tag}>>(*{pointer}))"
        if mask is not None:
          expression = f"vecops::vec::blend({tag}{{}}, {other}, {mask}, (vecops::vec::mask_any({tag}{{}}, {mask}) ? {expression} : {other}))"
      else:
        stride = self.scalar(view.strides[view_axis])
        contiguous = f"vecops::vec::load_convert({tag}{{}}, {pointer}, {options})"
        if I.constant(view.strides[view_axis]) == 1:
          expression = contiguous
        else:
          strided = f"vecops::vec::load_convert({tag}{{}}, {pointer}, {options}, vecops::vec::opt::strided({stride}))"
          expression = f"(({stride}) == 1 ? {contiguous} : {strided})"
        if I.constant(view.shape[view_axis]) is None and not I.same_expr(view.shape[view_axis], root_shape[axis]):
          filled = f"vecops::vec::fill({tag}{{}}, static_cast<vecops::vec::ElementOf<{tag}>>(*{pointer}))"
          if mask is not None:
            filled = f"vecops::vec::blend({tag}{{}}, {other}, {mask}, (vecops::vec::mask_any({tag}{{}}, {mask}) ? {filled} : {other}))"
          expression = f"(({self.scalar(view.shape[view_axis])}) == 1 ? {filled} : {expression})"
      snapshot = self.snapshots.get(id(value))
      if snapshot is not None:
        saved, condition = snapshot
        saved_pointer = self.pointer(saved, own_coords)
        if not value.args and saved.dtype == view.dtype:
          # Select the address/layout, not between two complete load DAGs.
          # Identical snapshot/source strides keep contiguous loads contiguous
          # and let the C++ optimizer hoist invariant alias selection.
          selected_pointer = f"({condition} ? {saved_pointer} : {pointer})"
          if broadcast:
            expression = (
              f"vecops::vec::fill({tag}{{}}, static_cast<vecops::vec::ElementOf<{tag}>>(*{selected_pointer}))"
            )
          else:
            source_stride, saved_stride = view.strides[view_axis], saved.strides[view_axis]
            same_stride = I.same_expr(source_stride, saved_stride)
            selected_stride = (
              self.scalar(source_stride)
              if same_stride
              else f"({condition} ? {self.scalar(saved_stride)} : {self.scalar(source_stride)})"
            )
            extra = (
              ""
              if same_stride and I.constant(source_stride) == 1
              else f", vecops::vec::opt::strided({selected_stride})"
            )
            expression = f"vecops::vec::load_convert({tag}{{}}, {selected_pointer}, {options}{extra})"
            if I.constant(view.shape[view_axis]) is None and not I.same_expr(view.shape[view_axis], root_shape[axis]):
              filled = f"vecops::vec::fill({tag}{{}}, static_cast<vecops::vec::ElementOf<{tag}>>(*{selected_pointer}))"
              expression = f"(({self.scalar(view.shape[view_axis])}) == 1 ? {filled} : {expression})"
          name = self.unique("v")
          self.line(f"auto {name} = {expression};")
          self.chunk_cache[key] = name
          return name
        snapshot_stride = self.scalar(saved.strides[view_axis]) if view_axis >= 0 else "0"
        # Snapshot already includes the original mask and inactive population.
        saved_options = (
          "vecops::vec::opt::unmasked" if getattr(self, "full_vector", False) else f"vecops::vec::opt::first({active})"
        )
        saved_value = (
          f"vecops::vec::load_convert({tag}{{}}, {saved_pointer}, {saved_options}, vecops::vec::opt::strided({snapshot_stride}))"
          if not broadcast
          else f"vecops::vec::fill({tag}{{}}, *{saved_pointer})"
        )
        expression = f"({condition} ? {saved_value} : {expression})"
    elif value.op == "fill":
      expression = (
        f"vecops::vec::mfill({tag}{{}}, {self.scalar(value.args[0])})"
        if value.type.mask
        else (f"vecops::vec::fill({tag}{{}}, static_cast<vecops::vec::ElementOf<{tag}>>({self.scalar(value.args[0])}))")
      )
    elif value.op == "arange":
      start = self.scalar(value.args[0])
      own_axis = value.data if value.data is not None else len(value.type.shape) - 1
      vector_axis = axis - (len(root_shape) - len(value.type.shape))
      operation = "iota" if own_axis == vector_axis else "fill"
      expression = f"vecops::vec::{operation}({tag}{{}}, static_cast<vecops::vec::ElementOf<{tag}>>(({start}) + ({vcoords[own_axis]})))"
    elif value.op == "cast":
      source = value.args[0]
      if source.op == "load" and source.type.dtype == source.data.dtype:
        # load_convert preserves lane count and avoids a memory-side temporary.
        proxy = I.Expr(I.TileType(dtype, value.type.shape), "load", source.args, source.data)
        if id(source) in self.snapshots:
          self.snapshots[id(proxy)] = self.snapshots[id(source)]
        expression = self.vector(proxy, tag, root_shape, coords, active, axis)
      elif source.type.dtype == dtype:
        return self.vector(source, tag, root_shape, coords, active, axis)
      else:
        source_tag = self.rebound_tag(source.type.dtype, tag)
        original = self.vector(source, source_tag, root_shape, coords, active, axis)
        expression = f"vecops::vec::convert({tag}{{}}, {source_tag}{{}}, {original})"
    elif value.op == "where":
      condition, a, b = value.args
      mask = self.vector(condition, tag, root_shape, coords, active, axis)
      x = self.vector(a, tag, root_shape, coords, active, axis)
      y = self.vector(b, tag, root_shape, coords, active, axis)
      expression = f"vecops::vec::blend({tag}{{}}, {y}, {mask}, {x})"
    elif value.op == "axis_reduce":
      from ....ir.iteration import AxisResult

      if isinstance(value.data, AxisResult):
        return self.axis_fold(value, tag, vcoords, active, axis, root_shape)
      reduction_axis, operation, unroll = value.data
      source = value.args[0]
      own_axis = axis - (len(root_shape) - len(value.type.shape))
      if own_axis < 0:
        raise TypeError("broadcast axis reductions require an explicit row loop")
      source_axis = own_axis + (reduction_axis <= own_axis)
      reduction_index = self.unique("reduction_axis")
      identity = self.identity(dtype, operation)
      banks = [self.unique("axis_acc") for _ in range(unroll)]
      for accumulator in banks:
        self.line(f"auto {accumulator} = vecops::vec::fill({tag}{{}}, {identity});")
      plan = self.prepared.axis(value, unroll)
      self.emitted_regions.append(plan)
      extent = self.scalar(source.type.shape[reduction_axis])
      self.line(f"vecops::nint_t {reduction_index} = 0;")
      self.opening(
        f"for (; vecops_dsl::full_group({reduction_index}, {extent}, 1, {unroll}); {reduction_index} += {unroll})"
      )
      coords = [
        vcoords[:reduction_axis] + (f"({reduction_index} + {slot})",) + vcoords[reduction_axis:]
        for slot in range(unroll)
      ]
      roots = self.emit_operations(plan, [tag] * unroll, coords, [active] * unroll, axis=source_axis)
      for slot, accumulator in enumerate(banks):
        self.line(f"{accumulator} = vecops::vec::{operation}({tag}{{}}, {accumulator}, {roots[0][slot]});")
      self.close()
      self.opening(f"for (; {reduction_index} < ({extent}); ++{reduction_index})")
      single = self.prepared.axis(value, 1)
      source_coords = vcoords[:reduction_axis] + (reduction_index,) + vcoords[reduction_axis:]
      roots = self.emit_operations(single, [tag], [source_coords], [active], axis=source_axis)
      self.line(f"{banks[0]} = vecops::vec::{operation}({tag}{{}}, {banks[0]}, {roots[0][0]});")
      self.close()
      accumulator = banks[0]
      for other in banks[1:]:
        self.line(f"{accumulator} = vecops::vec::{operation}({tag}{{}}, {accumulator}, {other});")
      self.chunk_cache[key] = accumulator
      return accumulator
    elif self.environment.implementations.get(value.op).lane_fallback:
      args = [self.vector(a, tag, root_shape, coords, active, axis) for a in value.args]
      name, lane = self.unique("lane_fallback"), self.unique("scalar_lane")
      self.line(f"auto {name} = vecops::vec::fill({tag}{{}}, vecops::vec::ElementOf<{tag}>{{0}});")
      self.opening(f"for (vecops::nint_t {lane} = 0; {lane} < {active}; ++{lane})")
      arguments = ", ".join(f"vecops::vec::get({tag}{{}}, {a}, {lane})" for a in args)
      self.line(
        f"{name} = vecops::vec::set({tag}{{}}, {name}, {lane}, static_cast<vecops::vec::ElementOf<{tag}>>(vecops_dsl::{value.op}({arguments})));"
      )
      self.close()
      self.chunk_cache[key] = name
      return name
    else:
      args = [self.vector(a, tag, root_shape, coords, active, axis) for a in value.args]
      implementation = self.environment.implementations.get(value.op)
      op = implementation.vector_name
      if op is None:
        raise TypeError(f"no CPU vector implementation registered for {value.op}")
      if implementation.wordwise_math:
        accuracy = {"strict": "strict", "fast": "fast", "estimate": "estimate"}[value.data or "strict"]
        args.append(f"vecops::vec::opt::math::{accuracy}")
      if op == "wordwise_math":
        operation, accuracy = value.data
        expression = f"vecops_dsl::math_words({tag}{{}}, {args[0]}, vecops::vec::{operation}, vecops::vec::opt::math::{accuracy or 'strict'})"
        name = self.unique()
        self.line(f"auto {name} = {expression};")
        self.chunk_cache[key] = name
        return name
      if op == "identity":
        return args[0]
      if op in {"shl", "shr"} and isinstance(value.args[1].type, I.ScalarType):
        count = I.constant(value.args[1])
        args[1] = (
          f"vecops::meta::Const<{count}>{{}}"
          if isinstance(count, int) and count >= 0
          else f"static_cast<int>({self.scalar(value.args[1])})"
        )
      expression = f"vecops::vec::{op}({tag}{{}}, {', '.join(args)})"
    name = self.unique()
    self.line(f"auto {name} = {expression};")
    self.chunk_cache[key] = name
    return name

  def rebound_tag(self, dtype, base):
    if dtype == "bool" or dtype == self.tag_dtypes[base]:
      return base
    # State consumers reuse the same lane geometry across casts. Canonicalize
    # repeated rebinds so reading the original dtype resolves directly to its
    # named register, rather than a scalar get/set reconstruction loop.
    origin = self.state_tag_origins.get(base)
    if origin is not None:
      tag = origin if dtype == self.tag_dtypes[origin] else f"vecops::vec::Rebind<{CPP_TYPES[dtype]}, {origin}>"
      self.tag_dtypes[tag] = dtype
      self.state_tag_origins[tag] = origin
      if origin in self.fixed_tag_lanes:
        self.fixed_tag_lanes[tag] = self.fixed_tag_lanes[origin]
      return tag
    tag = f"vecops::vec::Rebind<{CPP_TYPES[dtype]}, {base}>"
    self.tag_dtypes[tag] = dtype
    if base in self.fixed_tag_lanes:
      self.fixed_tag_lanes[tag] = self.fixed_tag_lanes[base]
    return tag

  def emit_operations(self, region, tags, coords, active, *, axis=-1, full=False):
    previous_cache, references = self.chunk_cache, {}
    uses = {operation.index: 0 for operation in region.operations}
    for operation in region.operations:
      for index in operation.operands:
        if index is not None:
          uses[index] += 1
    for index in region.root_indices:
      if index is not None:
        uses[index] += 1
    for operation in region.operations:
      for slot, (base, c, count) in enumerate(zip(tags, coords, active)):
        tag = self.rebound_tag(operation.expression.type.dtype, base)
        name = self.unique("vector_value")
        if not full:
          representation = "Mask" if operation.type.mask else "Vec"
          self.line(f"vecops::vec::{representation}<{tag}> {name};")
          self.opening(f"if ({count} == 0)")
          empty = "mfill" if operation.type.mask else "fill"
          zero = "false" if operation.type.mask else f"vecops::vec::ElementOf<{tag}>{{0}}"
          self.line(f"{name} = vecops::vec::{empty}({tag}{{}}, {zero});")
          self.close()
          self.opening("else")
        self.chunk_cache = {}
        if operation.opaque:
          expression = operation.expression
        else:
          args = tuple(
            I.Expr(region.operations[index].type, "vector_ref", data=references[(index, slot)])
            if index is not None
            else arg
            for arg, index in zip(operation.expression.args, operation.operands)
          )
          if operation.expression.op == "cast":
            source = operation.expression.args[0]
            source_index = operation.operands[0]
            if (
              source.op == "load"
              and source.type.dtype == source.data.dtype
              and source_index is not None
              and uses[source_index] == 1
              and id(source) not in self.materialized
              and id(source) not in self.reduction_materialized
            ):
              # Keep the lexical read's snapshot selection at the converting
              # load boundary. A vector_ref would hide that edge and prevent
              # load_convert fusion. Reuse already scheduled mask/inactive
              # registers rather than evaluating their DAG a second time.
              source_operation = region.operations[source_index]
              source_args = tuple(
                I.Expr(region.operations[index].type, "vector_ref", data=references[(index, slot)])
                if index is not None
                else arg
                for arg, index in zip(source.args, source_operation.operands)
              )
              proxy = I.Expr(source.type, "load", source_args, source.data)
              if id(source) in self.snapshots:
                self.snapshots[id(proxy)] = self.snapshots[id(source)]
              args = (proxy,)
          expression = I.Expr(operation.type, operation.expression.op, args, operation.expression.data)
          if operation.native_math_words:
            expression = I.Expr(
              operation.type, "wordwise_math", args, (operation.expression.op, operation.expression.data)
            )
          if id(operation.expression) in self.snapshots:
            self.snapshots[id(expression)] = self.snapshots[id(operation.expression)]
        value = self.vector(expression, tag, region.shape, c, count, axis)
        if full:
          self.line(f"auto {name} = {value};")
        else:
          self.line(f"{name} = {value};")
          self.close()
        references[(operation.index, slot)] = name
    roots = []
    for expression, index in zip(region.roots, region.root_indices):
      roots.append(
        [
          references[(index, slot)]
          if index is not None
          else self.vector(expression, self.rebound_tag(expression.type.dtype, tag), region.shape, c, a, axis)
          for slot, (tag, c, a) in enumerate(zip(tags, coords, active))
        ]
      )
    self.chunk_cache = previous_cache
    return roots

  def traverse_region(self, region, callback, *, setup=None, finish=None, project_tail=None, restore_tail=None):
    """Print explicit Vector IR schedules, not recursively lowered tile bodies."""
    state_plan = self.state_region(region)
    if state_plan is not None:
      self.traverse_state_region(state_plan, region, callback, setup, finish)
      return
    variants = [(power, region.base_power, region.groupings[power]) for power in sorted(region.groupings, reverse=True)]
    if region.base_power > 0:
      from ....ir.vector import group_vectors

      variants.append((None, 0, group_vectors(region.unroll, 0, 0)))
    for variant, (power, base_power, groups) in enumerate(variants):
      conditional = power is not None and (power > 0 or region.base_power > 0)
      if conditional:
        self.line(f"#{'if' if variant == 0 else 'elif'} VEC_MAX_POW >= {base_power + power + region.rebind_headroom}")
      elif variant:
        self.line("#else")
      base = self.unique("base_tag")
      self.tag_dtypes[base] = region.dtype
      self.line(f"using {base} = vecops::vec::ScalableTag<{CPP_TYPES[region.dtype]}, {base_power}>;")
      tags = []
      for group in groups:
        tag = self.unique("group_tag")
        self.tag_dtypes[tag] = region.dtype
        self.line(f"using {tag} = vecops::vec::ScalableTag<{CPP_TYPES[region.dtype]}, {group.power}>;")
        tags.append(tag)
      state = setup(tags) if setup else None
      coordinates = []
      for extent in region.shape[:-1]:
        coordinate = self.unique("axis")
        if region.outer_unroll == 1:
          self.line("#if defined(__clang__)")
          self.line("#pragma clang loop unroll(disable)")
          self.line("#elif defined(__GNUC__)")
          self.line("#pragma GCC unroll 1")
          self.line("#endif")
        self.opening(f"for (vecops::nint_t {coordinate} = 0; {coordinate} < ({self.scalar(extent)}); ++{coordinate})")
        coordinates.append(coordinate)
      coordinate = self.unique("lane")
      stop = self.scalar(region.shape[-1])

      def body(
        full,
        groups=groups,
        tags=tags,
        coordinate=coordinate,
        base=base,
        coordinates=coordinates,
        stop=stop,
        state=state,
      ):
        old_full = getattr(self, "full_vector", False)
        self.full_vector = full
        coords, active = [], []
        for group, tag in zip(groups, tags):
          chunk = coordinate if group.first == 0 else f"({coordinate} + {group.first} * {self.vector_size(base)})"
          coords.append(tuple(coordinates) + (chunk,))
          count = self.unique("active")
          value = (
            self.vector_size(tag)
            if full
            else f"std::clamp<vecops::nint_t>(({stop}) - {chunk}, 0, {self.vector_size(tag)})"
          )
          self.line(f"const auto {count} = {value};")
          active.append(count)
        previous_cache = self.chunk_cache
        try:
          with self.pointwise_layout(region, groups, full=full):
            roots = self.emit_operations(region, tags, coords, active, full=full)
            self.chunk_cache = {}
            callback(tags, coords, active, roots, state)
        finally:
          self.chunk_cache = previous_cache
          self.full_vector = old_full

      if region.single_chunk and base_power == region.base_power and len(groups) == 1:
        # The surrounding SIMD loop already strip-mined this exact vector.
        # Legal oversize fallbacks still use the ordinary streaming traversal.
        self.line(f"const vecops::nint_t {coordinate}=0; /* proven local SIMD chunk */")
        extent = region.shape[-1]
        full = (
          extent.op == "lanes"
          and extent.data[1] + ELEMENT_BYTES[region.dtype].bit_length() - ELEMENT_BYTES[extent.data[0]].bit_length()
          == base_power
        )
        body(full)
      elif region.full_chunks:
        self.line(f"vecops::nint_t {coordinate} = 0;")
        self.opening(
          f"for (; vecops_dsl::full_vectors({coordinate}, {stop}, {self.vector_size(base)}, {region.unroll}); {coordinate} += {region.unroll} * {self.vector_size(base)}) /* full vector groups */"
        )
        body(True)
        self.close()
        self.opening(f"if ({coordinate} < ({stop})) /* masked vector tail */")
        fragments = region.native_tails.get(power)
        if fragments and (setup is None or project_tail is not None and restore_tail is not None):
          tail_tag = self.unique("tail_tag")
          self.tag_dtypes[tail_tag] = region.dtype
          self.line(
            f"using {tail_tag} = vecops::vec::ScalableTag<{CPP_TYPES[region.dtype]}, {fragments[0].power}>; /* native-word masked remainder */"
          )
          tail_tags = tuple(tail_tag for _ in fragments)
          tail_groups = tuple(VectorGroup(fragment.first, 1, fragment.power) for fragment in fragments)
          tail_state = project_tail(tags, state, tail_tags, fragments) if project_tail else None
          body(False, groups=tail_groups, tags=tail_tags, base=tail_tag, state=tail_state)
          if restore_tail:
            restore_tail(tags, state, tail_tags, fragments, tail_state)
        else:
          body(False)
        self.close()
      else:
        self.opening(
          f"for (vecops::nint_t {coordinate} = 0; {coordinate} < ({stop}); {coordinate} += {region.unroll} * {self.vector_size(base)})"
        )
        body(False)
        self.close()
      for _ in region.shape[:-1]:
        self.close()
      if finish:
        finish(tags, state)
    if len(variants) > 1:
      self.line("#endif")

  def reduction(self, expression):
    return self.reductions((expression,), plan=self.vector_ir.plans.get(id(expression)))

  def axis_fold(self, value, tag, vcoords, active, axis, root_shape):
    fold = value.data.fold
    key = ("axis_fold", id(fold), tag, tuple(vcoords), active, axis)
    if key in self.chunk_cache:
      return self.chunk_cache[key][value.data.index]
    own_axis = axis - (len(root_shape) - len(value.type.shape))
    if own_axis < 0:
      raise TypeError("broadcast axis folds require an explicit row loop")
    source_axis = own_axis + (fold.axis <= own_axis)
    index = self.unique("fold_axis")
    banks = [[self.unique("axis_acc") for _ in range(fold.unroll)] for _ in fold.sources]
    for op, accumulators in zip(fold.operations, banks):
      for accumulator in accumulators:
        self.line(f"auto {accumulator}=vecops::vec::fill({tag}{{}}, {self.identity(value.type.dtype, op)});")
    extent = self.scalar(fold.sources[0].type.shape[fold.axis])
    self.line(f"vecops::nint_t {index}=0;")
    self.opening(
      f"for (;vecops_dsl::full_group({index},{extent},1,{fold.unroll});{index}+={fold.unroll}) /* fused axis fold */"
    )
    coords = [vcoords[: fold.axis] + (f"({index}+{slot})",) + vcoords[fold.axis :] for slot in range(fold.unroll)]
    plan = self.prepared.axis(value, fold.unroll)
    self.emitted_regions.append(plan)
    roots = self.emit_operations(plan, [tag] * fold.unroll, coords, [active] * fold.unroll, axis=source_axis)
    for r, op in enumerate(fold.operations):
      for slot, accumulator in enumerate(banks[r]):
        self.line(f"{accumulator}=vecops::vec::{op}({tag}{{}},{accumulator},{roots[r][slot]});")
    self.close()
    self.opening(f"for (;{index}<({extent});++{index})")
    coords = vcoords[: fold.axis] + (index,) + vcoords[fold.axis :]
    single = self.prepared.axis(value, 1)
    roots = self.emit_operations(single, [tag], [coords], [active], axis=source_axis)
    for r, op in enumerate(fold.operations):
      self.line(f"{banks[r][0]}=vecops::vec::{op}({tag}{{}},{banks[r][0]},{roots[r][0]});")
    self.close()
    for op, accumulators in zip(fold.operations, banks):
      for other in accumulators[1:]:
        self.line(f"{accumulators[0]}=vecops::vec::{op}({tag}{{}},{accumulators[0]},{other});")
    results = tuple(a[0] for a in banks)
    self.chunk_cache[key] = results
    return results[value.data.index]

  def reductions(self, expressions, *, plan=None, stores=()):
    """Emit reductions in ordinary lexical scopes, preserving local invariants.

    A by-reference IIFE can be outlined by the C++ compiler and hide immutable
    tensor metadata/scalars behind aliasable capture memory. Explicit result
    locals avoid that boundary and permit loop-invariant code motion.
    """
    sources = [expression.args[0] for expression in expressions]
    dtype = sources[0].type.dtype
    operations = [self.environment.operations.get(expression).reduction for expression in expressions]
    result = self.unique("reduction_result")
    types = [CPP_TYPES[expression.type.dtype] for expression in expressions]
    result_type = f"std::tuple<{', '.join(types)}>" if len(types) > 1 else types[0]
    self.line(f"{result_type} {result}{{}};")
    self.opening("/* reduction region: no capture boundary */")
    for source in sources:
      cached = self.ir.materializations.get(id(source))
      if cached is not None and id(source) not in self.materialized:
        count = ", ".join(self.scalar(v) for v in cached.shape)
        self.opening(f"if ({cached.storage}_enabled && !{cached.storage}.data())")
        self.line(f"{cached.storage}.resize(vecops_dsl::checked_shape({{{count}}}));")
        self.close()
    plan = plan or self.prepared.reduction(expressions, stores)
    operations = [effect.operation for effect in plan.initializations]
    for store in stores:
      self.check_store(store)

    def setup(tags):
      banks = [[self.unique("acc") for _ in tags] for _ in expressions]
      for accumulators, op in zip(banks, operations):
        for accumulator, tag in zip(accumulators, tags):
          self.line(f"auto {accumulator} = vecops::vec::fill({tag}{{}}, {self.identity(dtype, op)});")
      return banks

    def update(tags, coordinates, active, roots, banks):
      for effect in plan.effects:
        if effect.kind != "accumulate":
          continue
        source_index = effect.value_root
        source, accumulators, op = sources[source_index], banks[effect.index], effect.operation
        for slot, (tag, coords, count) in enumerate(zip(tags, coordinates, active)):
          value = roots[source_index][slot]
          cached = self.ir.materializations.get(id(source))
          if cached is not None and id(source) not in self.materialized:
            condition = f"{cached.storage}_enabled"
            full = getattr(self, "full_vector", False)
            # Full traversal groups already prove a positive, in-bounds
            # physical extent. A first(count) option unnecessarily constructs
            # a multiword SVE predicate, whose tuple accessors may require
            # out-of-line calls across an ISA target boundary.
            self.opening(f"if ({condition})" if full else f"if ({condition} && {count} > 0)")
            memory_option = "vecops::vec::opt::unmasked" if full else f"vecops::vec::opt::first({count})"
            self.line(
              f"vecops::vec::store_convert({tag}{{}}, {self.pointer(cached, coords)}, {value}, {memory_option});"
            )
            self.close()
            self.reduction_materialized[id(source)] = (cached, condition)
          # A full vector group has no inactive lane. Besides an unnecessary
          # select, constructing a multiword SVE predicate can introduce ABI
          # helper calls and spills on otherwise call-free reduction sweeps.
          # Tail vectors still need the operation's identity, even when a
          # load happens to zero-fill its own inactive lanes: e.g. max/min
          # identities are not zero, and later elementwise ops may change it.
          masked = value
          if not getattr(self, "full_vector", False):
            masked = self.unique("masked")
            self.line(
              f"auto {masked} = vecops::vec::blend({tag}{{}}, vecops::vec::fill({tag}{{}}, {self.identity(dtype, op)}), vecops::vec::mwhilelt({tag}{{}}, 0, {count}), {value});"
            )
          self.line(f"{accumulators[slot]} = vecops::vec::{op}({tag}{{}}, {accumulators[slot]}, {masked});")
      self.write_effects(plan, tags, coordinates, active, roots)

    def finish(tags, banks):
      results = []
      for effect in plan.finalizations:
        accumulators, op = banks[effect.index], effect.operation
        parts = [f"vecops::vec::reduce_{op}({tag}{{}}, {acc})" for tag, acc in zip(tags, accumulators)]
        combined = parts[0]
        for part in parts[1:]:
          combined = f"({combined} + {part})" if op == "add" else f"std::{op}({combined}, {part})"
        results.append(combined)
      self.line(f"{result} = {'std::tuple{' + ', '.join(results) + '}' if len(results) > 1 else results[0]};")

    def project_tail(tags, banks, tail_tags, fragments):
      projected = []
      for accumulators in banks:
        parts = []
        for fragment in fragments:
          tag, value = tags[fragment.parent], accumulators[fragment.parent]
          for upper in fragment.path:
            value = f"vecops::vec::{'upper' if upper else 'lower'}({tag}{{}}, {value})"
            tag = f"vecops::vec::Half<{tag}>"
          name = self.unique("tail_acc")
          self.line(f"auto {name} = {value};")
          parts.append(name)
        projected.append(parts)
      return projected

    def restore_tail(tags, banks, tail_tags, fragments, projected):
      for accumulators, parts in zip(banks, projected):
        for parent, (tag, accumulator) in enumerate(zip(tags, accumulators)):
          leaves = {fragment.path: part for fragment, part in zip(fragments, parts) if fragment.parent == parent}

          def join(tag, path=(), leaves=leaves):
            if path in leaves:
              return leaves[path]
            half = f"vecops::vec::Half<{tag}>"
            low, high = join(half, (*path, False)), join(half, (*path, True))
            name = self.unique("tail_join")
            self.line(f"auto {name} = vecops::vec::concat({tag}{{}}, {low}, {high});")
            return name

          self.line(f"{accumulator} = {join(tag)};")

    self.traverse_region(plan, update, setup=setup, finish=finish, project_tail=project_tail, restore_tail=restore_tail)
    self.close()
    return result

  def identity(self, dtype, op):
    cpp = CPP_TYPES[dtype]
    if op == "add":
      return f"{cpp}{{0}}"
    if dtype.startswith("float"):
      return ("-" if op == "max" else "") + f"std::numeric_limits<{cpp}>::infinity()"
    return f"std::numeric_limits<{cpp}>::{'lowest' if op == 'max' else 'max'}()"
