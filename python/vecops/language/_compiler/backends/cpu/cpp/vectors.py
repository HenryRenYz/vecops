"""C++ emission: vectors."""

from __future__ import annotations
from .... import ir as I
from ..capabilities import CPP_TYPES


class VectorsEmitter:
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
      own_tag = f"vecops::vec::Rebind<{CPP_TYPES[dtype]}, {tag}>"
      self.tag_dtypes[own_tag] = dtype
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
        or I.constant(view.shape[view_axis]) == 1
        and not I.same_expr(view.shape[view_axis], root_shape[axis])
      )
      options, mask = self.memory_options(value.args[0] if value.args else None, tag, root_shape, coords, active, axis)
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
        source_tag = f"vecops::vec::Rebind<{CPP_TYPES[source.type.dtype]}, {tag}>"
        self.tag_dtypes[source_tag] = source.type.dtype
        original = self.vector(source, source_tag, root_shape, coords, active, axis)
        expression = f"vecops::vec::convert({tag}{{}}, {source_tag}{{}}, {original})"
    elif value.op == "where":
      condition, a, b = value.args
      mask = self.vector(condition, tag, root_shape, coords, active, axis)
      x = self.vector(a, tag, root_shape, coords, active, axis)
      y = self.vector(b, tag, root_shape, coords, active, axis)
      expression = f"vecops::vec::blend({tag}{{}}, {y}, {mask}, {x})"
    elif value.op == "axis_reduce":
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
    tag = f"vecops::vec::Rebind<{CPP_TYPES[dtype]}, {base}>"
    self.tag_dtypes[tag] = dtype
    if base in self.fixed_tag_lanes:
      self.fixed_tag_lanes[tag] = self.fixed_tag_lanes[base]
    return tag

  def emit_operations(self, region, tags, coords, active, *, axis=-1, full=False):
    previous_cache, references = self.chunk_cache, {}
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

  def traverse_region(self, region, callback, *, setup=None, finish=None):
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
        self.opening(f"for (vecops::nint_t {coordinate} = 0; {coordinate} < ({self.scalar(extent)}); ++{coordinate})")
        coordinates.append(coordinate)
      coordinate = self.unique("lane")
      stop = self.scalar(region.shape[-1])

      def body(full):
        coords, active = [], []
        for group, tag in zip(groups, tags):
          chunk = coordinate if group.first == 0 else f"({coordinate} + {group.first} * vecops::vec::size({base}{{}}))"
          coords.append(tuple(coordinates) + (chunk,))
          count = self.unique("active")
          value = (
            f"vecops::vec::size({tag}{{}})"
            if full
            else f"std::clamp<vecops::nint_t>(({stop}) - {chunk}, 0, vecops::vec::size({tag}{{}}))"
          )
          self.line(f"const auto {count} = {value};")
          active.append(count)
        previous_cache = self.chunk_cache
        roots = self.emit_operations(region, tags, coords, active, full=full)
        self.chunk_cache = {}
        callback(tags, coords, active, roots, state)
        self.chunk_cache = previous_cache

      if region.full_chunks:
        self.line(f"vecops::nint_t {coordinate} = 0;")
        self.opening(
          f"for (; vecops_dsl::full_vectors({coordinate}, {stop}, vecops::vec::size({base}{{}}), {region.unroll}); {coordinate} += {region.unroll} * vecops::vec::size({base}{{}})) /* full vector groups */"
        )
        body(True)
        self.close()
        self.opening(f"if ({coordinate} < ({stop})) /* masked vector tail */")
        body(False)
        self.close()
      else:
        self.opening(
          f"for (vecops::nint_t {coordinate} = 0; {coordinate} < ({stop}); {coordinate} += {region.unroll} * vecops::vec::size({base}{{}}))"
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
            self.opening(f"if ({condition} && {count} > 0)")
            self.line(
              f"vecops::vec::store_convert({tag}{{}}, {self.pointer(cached, coords)}, {value}, vecops::vec::opt::first({count}));"
            )
            self.close()
            self.reduction_materialized[id(source)] = (cached, condition)
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

    self.traverse_region(plan, update, setup=setup, finish=finish)
    self.close()
    return result

  def identity(self, dtype, op):
    cpp = CPP_TYPES[dtype]
    if op == "add":
      return f"{cpp}{{0}}"
    if dtype.startswith("float"):
      return ("-" if op == "max" else "") + f"std::numeric_limits<{cpp}>::infinity()"
    return f"std::numeric_limits<{cpp}>::{'lowest' if op == 'max' else 'max'}()"
