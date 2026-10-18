# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Register and indexed-memory terminal lowering; never arrays of Vec/Mask."""

from .... import ir as I
from ....analysis.projection import complete_grouped_prefix, scalar_memory_projection
from ....dtypes import ELEMENT_BYTES
from ....ir.traversal import walk_unique
from ..capabilities import CPP_TYPES


class PrimitivesEmitter:
  def prefetch(self, statement):
    view = statement.view
    condition = " && ".join(f"({self.scalar(n)})>0" for n in view.shape) or "true"
    self.opening(f"if ({condition})")
    self.line(
      f"vecops::vec::prefetch(vecops::vec::FixedTag<{CPP_TYPES[view.dtype]}, 1>{{}}, {self.pointer(view)}, vecops::vec::mem::prefetch_{statement.level.lower()}, vecops::vec::mem::prefetch_{'read' if statement.intent == 'read' else 'write'}, vecops::vec::mem::prefetch_{'keep' if statement.temporal else 'stream'});"
    )
    self.close()

  def element_at(self, value):
    source, *indices = value.args
    projected = scalar_memory_projection(
      source, indices, blocked=self.materialized.keys() | self.reduction_materialized.keys()
    )
    if projected is not None:
      load, coordinates, casts = projected
      coords = tuple(self.scalar(v) for v in coordinates)
      pointer = self.pointer(load.data, coords)
      snapshot = self.snapshots.get(id(load))
      if snapshot is not None and snapshot[0].dtype == load.data.dtype:
        saved, condition = snapshot
        pointer = f"({condition} ? {self.pointer(saved, coords)} : {pointer})"
      scalar = self.scalar_convert(load.type.dtype, f"*{pointer}", load.data.dtype)
      if snapshot is not None and snapshot[0].dtype != load.data.dtype:
        # A captured converting load stores compute elements. Its source
        # still has the memory dtype: select converted values, since these
        # pointers have different types (and different element widths).
        saved, condition = snapshot
        captured = self.scalar_convert(load.type.dtype, f"*{self.pointer(saved, coords)}", saved.dtype)
        scalar = f"({condition} ? {captured} : {scalar})"
      source_dtype = load.type.dtype
      for dtype in casts:
        scalar = self.scalar_convert(dtype, scalar, source_dtype)
        source_dtype = dtype
      return scalar
    coords = tuple(self.scalar(v) for v in indices)
    tag = self.unique("scalar_tag")
    self.line(f"using {tag} = vecops::vec::FixedTag<{CPP_TYPES[source.type.dtype]}, 1>;")
    self.tag_dtypes[tag] = source.type.dtype
    self.fixed_tag_lanes[tag] = 1
    result = self.vector(source, tag, source.type.shape, coords, "1")
    return f"vecops::vec::get({tag}{{}}, {result}, 0)"

  def widening_dot_vector(self, value, tag, own_coords, active, traversal_extent=None):
    # The operation changes lane geometry: each output consumes a complete
    # byte-span group, with absent source lanes contributing zero. Retain
    # a full-vector proof only when each output has a complete source group.
    group = value.data
    source_coords = (f"(({own_coords[0]}) * {group})",)
    previous_full = getattr(self, "full_vector", False)
    # Scalable SIMD widths are multiples of 16 bytes, including SVE's
    # minimum VL. Do not derive this fact from a cost-only target override.
    lane_multiple = self.fixed_tag_lanes.get(tag, 1)
    geometry = self.tag_geometry.get(tag)
    target = self.environment.codegen_target
    if (
      geometry
      and geometry[0]
      and (
        target.architecture == "x86"
        and target.isa in {"avx", "avx2", "avx512"}
        or target.architecture == "arm"
        and target.isa in {"neon", "sve", "sme"}
      )
    ):
      minimum_bytes = 16
      power = geometry[1]
      lane_multiple = max(1, (minimum_bytes << max(0, power)) // (ELEMENT_BYTES[value.type.dtype] << max(0, -power)))
    operands, tags = [], []
    try:
      self.full_vector = False
      for source in value.args:
        source_extent = I.constant(source.type.shape[0])
        # The typed operation has ceildiv(K, group) output elements.
        # With K divisible by group, a complete output traversal cannot
        # reach a partial source group. Do not require ceildiv to have
        # been folded to a literal in the scalar metadata IR.
        exact_domain = traversal_extent is not None and (
          I.same_expr(traversal_extent, value.type.shape[0])
          or type(source_extent) is int
          and I.constant(traversal_extent) == (source_extent + group - 1) // group
        )
        full_source = (
          previous_full
          and type(source_extent) is int
          and (
            source_extent % group == 0 or exact_domain and complete_grouped_prefix(source_extent, group, lane_multiple)
          )
        )
        self.full_vector = full_source
        source_tag = f"vecops::vec::ViewAs<{CPP_TYPES[source.type.dtype]}, {tag}>"
        self.tag_dtypes[source_tag] = source.type.dtype
        if tag in self.tag_geometry:
          # ViewAs preserves physical bytes, unlike a lane-count Rebind.
          self.tag_geometry[source_tag] = self.tag_geometry[tag]
        if tag in self.fixed_tag_lanes:
          self.fixed_tag_lanes[source_tag] = self.fixed_tag_lanes[tag] * group
        count = f"vecops_dsl::clamp_value<vecops::nint_t>(({self.scalar(source.type.shape[0])}) - ({source_coords[0]}), 0, ({active}) * {group})"
        # VLA SVE FixedTag supports at most 16 bytes. A one-result byte
        # contraction can still evaluate four FP64 producer lanes (32 B).
        # Materialize only those exceptional fixed-tag inputs lane by lane;
        # scalable full groups retain their legalized vector traversal.
        fixed_lanes = self.fixed_tag_lanes.get(source_tag)
        wide_fixed_producer = (
          self.environment.target.architecture == "arm"
          and fixed_lanes is not None
          and any(
            isinstance(node.type, I.TileType) and fixed_lanes * ELEMENT_BYTES[node.type.dtype] > 16
            for node in walk_unique(source)
          )
        )
        if wide_fixed_producer:
          operand, lane = self.unique("dot_fixed_input"), self.unique("dot_fixed_lane")
          self.line(f"auto {operand} = vecops::vec::zeros({source_tag}{{}});")
          self.opening(f"for (vecops::nint_t {lane}=0; {lane}<({count}); ++{lane})")
          index = I.Expr(I.ScalarType("int64"), "raw", data=f"(({source_coords[0]}) + {lane})")
          scalar = self.element_at(I.Expr(I.ScalarType(source.type.dtype), "element_at", (source, index)))
          self.line(f"{operand} = vecops::vec::set({source_tag}{{}}, {operand}, {lane}, {scalar});")
          self.close()
        else:
          operand = self.vector(source, source_tag, source.type.shape, source_coords, count, 0)
        # A transformed producer can turn masked-load zeros into nonzeros.
        # Sanitize its final value, rather than just the underlying load.
        if full_source:
          operands.append(operand)
          tags.append(source_tag)
          continue
        sanitized = self.unique("dot_input")
        self.line(
          f"auto {sanitized} = vecops::vec::blend({source_tag}{{}}, vecops::vec::zeros({source_tag}{{}}), vecops::vec::mwhilelt({source_tag}{{}}, 0, {count}), {operand});"
        )
        operands.append(sanitized)
        tags.append(source_tag)
    finally:
      self.full_vector = previous_full
    return f"vecops::vec::widening_dot({tag}{{}}, {tags[0]}{{}}, {tags[1]}{{}}, {', '.join(operands)})"

  def primitive_vector(self, value, tag, shape, coords, active, axis):
    if self.environment.operations.get(value).family == "sve_register":
      if self.environment.codegen_target.isa not in {"sve", "sme"}:
        raise TypeError("ordinary SVE register operations require an SVE compilation target")
      if value.op == "sve.iota":
        own_coords = self.coordinates(value.type.shape, shape, coords)
        own_axis = axis - (len(shape) - len(value.type.shape))
        start = f"static_cast<{CPP_TYPES[value.type.dtype]}>({own_coords[0]})"
        if own_axis < 0:
          return f"vecops::vec::fill({tag}{{}}, {start})"
        return f"vecops::vec::iota({tag}{{}}, {start})"
      operands, tags = [], []
      for source in value.args:
        own_tag = f"vecops::vec::ScalableTag<{CPP_TYPES[source.type.dtype]}, 0>"
        self.tag_dtypes[own_tag] = source.type.dtype
        tags.append(own_tag)
        operands.append(self.vector(source, own_tag, source.type.shape, ("0",), f"vecops::vec::size({own_tag}{{}})", 0))
      operation = value.op.removeprefix("sve.")
      own_tag = f"vecops::vec::ScalableTag<{CPP_TYPES[value.type.dtype]}, 0>"
      self.tag_dtypes[own_tag] = value.type.dtype
      self.tag_geometry[own_tag] = True, 0
      if operation == "mmla":
        call = f"vecops::vec::sve::mmla({own_tag}{{}}, {tags[0]}{{}}, {tags[1]}{{}}, {', '.join(operands)})"
      else:
        template = f"<{value.data}>" if value.data is not None else ""
        call = f"vecops::vec::sve::{operation}{template}({tags[0]}{{}}, {', '.join(operands)})"
      own_coords = self.coordinates(value.type.shape, shape, coords)
      if self.same_register_tag(tag, own_tag) and own_coords == ("0",):
        return call
      # Scratch placement or a differently shaped consumer may request a
      # prefix/multiword/broadcast traversal. Compute the ISA word once, then
      # project its real lanes; never disguise one native word as many words.
      word, result, lane = self.unique("sve_word"), self.unique("sve_projected"), self.unique("sve_lane")
      self.line(f"auto {word} = {call};")
      self.line(f"auto {result} = vecops::vec::zeros({tag}{{}});")
      self.opening(f"for (vecops::nint_t {lane}=0; {lane}<({active}); ++{lane})")
      own_axis = axis - (len(shape) - len(value.type.shape))
      index = own_coords[0] if own_axis < 0 else f"(({own_coords[0]}) + {lane})"
      self.opening(f"if (({index}) >= 0 && ({index}) < vecops::vec::size({own_tag}{{}}))")
      self.line(
        f"{result} = vecops::vec::set({tag}{{}}, {result}, {lane}, vecops::vec::get({own_tag}{{}}, {word}, {index}));"
      )
      self.close()
      self.close()
      return result
    if value.op == "widening_dot":
      own_coords = self.coordinates(value.type.shape, shape, coords)
      own_axis = axis - (len(shape) - len(value.type.shape))

      def broadcast():
        # Grouping happens before broadcasting. Repeat the selected grouped
        # scalar when the consumer vectorizes another axis or a singleton.
        index = I.Expr(I.ScalarType("int64"), "raw", data=own_coords[0])
        scalar = self.element_at(I.Expr(I.ScalarType(value.type.dtype), "element_at", (value, index)))
        return f"vecops::vec::fill({tag}{{}}, {scalar})"

      if own_axis < 0 or (I.constant(value.type.shape[0]) == 1 and not I.same_expr(value.type.shape[0], shape[axis])):
        return broadcast()
      if I.constant(value.type.shape[0]) is None and not I.same_expr(value.type.shape[0], shape[axis]):
        return self.conditional_vector(
          value.type,
          tag,
          f"({self.scalar(value.type.shape[0])}) == 1",
          broadcast,
          lambda: self.widening_dot_vector(value, tag, own_coords, active, shape[axis]),
        )
      return self.widening_dot_vector(value, tag, own_coords, active, shape[axis])
    if value.op == "bitreinterpret":
      if id(value) in self.prepared.register_programs:
        return self.geometry_vector(value, tag, shape, coords, active, axis)
      return NotImplemented  # Same-width elementwise spelling is in vector().
    if self.environment.operations.get(value).family == "register_geometry":
      return self.geometry_vector(value, tag, shape, coords, active, axis)
    if value.op == "gather":
      saved = self.snapshots.get(id(value))
      if saved is not None:
        name = self.unique("indexed_saved")
        self.line(f"vecops::vec::Vec<{tag}> {name};")
        cache = dict(self.chunk_cache)
        self.opening(f"if ({saved[1]})")
        self.line(
          f"{name} = {self.vector(I.Expr(value.type, 'load', data=saved[0]), tag, shape, coords, active, axis)};"
        )
        self.close()
        self.chunk_cache = dict(cache)
        self.opening("else")
        del self.snapshots[id(value)]
        self.line(f"{name} = {self.primitive_vector(value, tag, shape, coords, active, axis)};")
        self.snapshots[id(value)] = saved
        self.close()
        self.chunk_cache = cache
        return name
      *coordinates, predicate, other = value.args
      itag = self.rebound_tag("int64", tag)
      indices_name = self.unique("indices")
      for memory_axis, index in enumerate(coordinates):
        indices = self.vector(index, itag, shape, coords, active, axis)
        scaled = f"vecops::vec::mul({itag}{{}}, {indices}, vecops::vec::fill({itag}{{}}, {self.scalar(value.data.strides[memory_axis])}))"
        self.line(
          f"auto {indices_name} = {scaled};"
          if memory_axis == 0
          else f"{indices_name} = vecops::vec::add({itag}{{}}, {indices_name}, {scaled});"
        )
        # Per-axis logical checks precede the physical address calculation.
        self.check_indices(index, predicate, itag, shape, coords, active, axis, value.data, memory_axis)
      options, _ = self.memory_options(predicate, tag, shape, coords, active, axis)
      # Inactive indices may be out of bounds: do not form their addresses.
      inactive = self.unique("indexed_inactive")
      self.line(f"auto {inactive} = {self.vector(other, tag, shape, coords, active, axis)};")
      return f"vecops::vec::load_convert({tag}{{}}, {self.pointer(value.data)}, {options}, vecops::vec::opt::merge({inactive}), vecops::vec::opt::indexed({indices_name}))"
    if value.op == "local_shuf":
      source = self.vector(value.args[0], tag, shape, coords, active, axis)
      order = ", ".join(map(str, value.data))
      return f"vecops::vec::local_shuf({tag}{{}}, {source}, vecops::vec::opt::LaneOrder<{order}>{{}})"
    if value.op == "shuf":
      source = self.vector(value.args[0], tag, shape, coords, active, axis)
      dtype = f"int{ELEMENT_BYTES[value.type.dtype] * 8}"
      itag = self.rebound_tag(dtype, tag)
      indices = self.vector(value.args[1], itag, shape, coords, active, axis)
      return f"vecops::vec::shuf({tag}{{}}, {source}, {indices})"
    if value.op == "value_transform":
      source = value.args[0]
      kind, axes = value.data
      own = self.coordinates(value.type.shape, shape, coords)
      own_axis = axis - (len(shape) - len(value.type.shape))
      # Consumer axes are not value axes when a transform is broadcast into
      # a higher-rank expression. Never vectorize an absent/singleton axis.
      if own_axis < 0 or (
        I.constant(value.type.shape[own_axis]) == 1 and not (self.fixed_tag_lanes.get(tag) == 1 and str(active) == "1")
      ):
        indices = tuple(I.Expr(I.ScalarType("int64"), "raw", data=c) for c in own)
        scalar = self.element_at(I.Expr(I.ScalarType(value.type.dtype), "element_at", (value, *indices)))
        fill = "mfill" if value.type.mask else "fill"
        return f"vecops::vec::{fill}({tag}{{}}, {scalar})"

      def transformed(source_coords, source_axis):
        # Retain the consumer extent along the mapped axis so dynamic size-one
        # sources still take the normal memory broadcast path.
        source_shape = list(source.type.shape)
        source_shape[source_axis] = shape[axis]
        return self.vector(source, tag, tuple(source_shape), source_coords, active, source_axis)

      if kind == "row":
        return transformed((self.scalar(axes), *own), own_axis + 1)
      if kind == "slice":
        mapped = [self.scalar(start) for start in value.args[1:]]
        for output_axis, source_axis in enumerate(axes):
          mapped[source_axis] = f"(({mapped[source_axis]}) + ({own[output_axis]}))"
        # A window maps into the producer's original coordinates; it is not
        # broadcasting that producer to a smaller physical/logical shape.
        return self.vector(source, tag, source.type.shape, tuple(mapped), active, axes[own_axis])
      if kind == "expand_dims":
        mapped = own[:axes] + own[axes + 1 :]
        if own_axis == axes:
          indices = tuple(I.Expr(I.ScalarType("int64"), "raw", data=c) for c in mapped)
          scalar = self.element_at(I.Expr(I.ScalarType(source.type.dtype), "element_at", (source, *indices)))
          fill = "mfill" if value.type.mask else "fill"
          return f"vecops::vec::{fill}({tag}{{}}, {scalar})"
        return transformed(mapped, own_axis - (own_axis > axes))
      if kind == "squeeze":
        mapped = own[:axes] + ("0",) + own[axes:]
        return transformed(mapped, own_axis + (own_axis >= axes))
      if kind == "broadcast_to":
        return self.vector(source, tag, value.type.shape, own, active, own_axis)
      if kind == "transpose":
        mapped = [None] * len(axes)
        for i, a in enumerate(axes):
          mapped[a] = own[i]
        return transformed(tuple(mapped), axes[own_axis])
      if kind == "reshape" and len(source.type.shape) == len(value.type.shape) == 1:
        # A validated rank-one reshape preserves every coordinate. The
        # frontend's element-count assertion remains before consumption;
        # changing the extent's spelling does not require lane-wise gather
        # and set, even when a full word was obtained from a clipped slice.
        # Keep the original producer/capture and its conversion boundaries.
        return transformed(own, own_axis)
      projection = getattr(self.prepared, "reshape_suffixes", {}).get(id(value))
      if (
        kind == "reshape"
        and self.environment.codegen_target.architecture == "x86"
        and projection is not None
        and projection.subject is value
      ):
        common = projection.dimensions
        output_prefix = len(value.type.shape) - common
        source_prefix = len(source.type.shape) - common
        if own_axis >= output_prefix:
          prefix = "0"
          for coordinate, extent in zip(own[:output_prefix], value.type.shape[:output_prefix]):
            prefix = f"(({prefix}) * ({self.scalar(extent)}) + ({coordinate}))"
          mapped = []
          for index, extent in enumerate(source.type.shape[:source_prefix]):
            divisor = " * ".join(f"({self.scalar(n)})" for n in source.type.shape[index + 1 : source_prefix]) or "1"
            mapped.append(f"(({prefix}) / ({divisor})) % ({self.scalar(extent)})")
          mapped.extend(own[output_prefix:])
          result = self.unique("reshape_suffix")
          zero = f"vecops::vec::mfill({tag}{{}}, false)" if value.type.mask else f"vecops::vec::zeros({tag}{{}})"
          self.line(f"auto {result} = {zero}; /* preserved reshape suffix */")
          condition = " && ".join(
            [f"({active}) > 0", *(f"({self.scalar(n)}) > 0" for n in source.type.shape[:source_prefix])]
          )
          previous = self.chunk_cache
          self.chunk_cache = dict(previous)
          self.opening(f"if ({condition})")
          try:
            produced = transformed(tuple(mapped), own_axis + len(source.type.shape) - len(value.type.shape))
            if not self.full_vector:
              mask = f"vecops::vec::mwhilelt({tag}{{}}, 0, {active})"
              produced = (
                f"vecops::vec::mask_and({tag}{{}}, {mask}, {produced})"
                if value.type.mask
                else f"vecops::vec::blend({tag}{{}}, {zero}, {mask}, {produced})"
              )
            self.line(f"{result} = {produced};")
          finally:
            self.close()
            self.chunk_cache = previous
          return result
      # General reshape can cross row boundaries within one physical vector.
      # Lane-wise fallback remains register-only; contiguous reads are folded
      # into a view by a separate pass rather than a hidden emitter optimizer.
      result, lane = self.unique("reshape"), self.unique("reshape_lane")
      zero = f"vecops::vec::mfill({tag}{{}}, false)" if value.type.mask else f"vecops::vec::zeros({tag}{{}})"
      self.line(f"auto {result} = {zero};")
      self.opening(f"for (vecops::nint_t {lane}=0; {lane}<({active}); ++{lane})")
      offset = "0"
      for i, (c, extent) in enumerate(zip(own, value.type.shape)):
        c = f"({c}+{lane})" if i == own_axis else c
        offset = f"(({offset}) * ({self.scalar(extent)}) + ({c}))"
      mapped = []
      for i in range(len(source.type.shape)):
        divisor = " * ".join(f"({self.scalar(n)})" for n in source.type.shape[i + 1 :]) or "1"
        mapped.append(
          I.Expr(
            I.ScalarType("int64"), "raw", data=f"(({offset}) / ({divisor})) % ({self.scalar(source.type.shape[i])})"
          )
        )
      scalar = self.element_at(I.Expr(I.ScalarType(source.type.dtype), "element_at", (source, *mapped)))
      self.line(f"{result} = vecops::vec::set({tag}{{}}, {result}, {lane}, {scalar});")
      self.close()
      return result
    return NotImplemented

  def geometry_scalar(self, value, index):
    """Legal fallback for cross-word geometry; no array of native vectors.

    The small closed-register fast path is planned before emission. This
    fallback handles large virtual values and partial/scalar consumers with
    exact logical indexing, including endian-correct bit reinterpretation.
    """
    operation, source = value.op, value.args[0] if value.args else None
    if self.environment.operations.get(value).family != "register_geometry":
      coordinate = I.Expr(I.ScalarType("int64"), "raw", data=index)
      return self.element_at(I.Expr(I.ScalarType(value.type.dtype), "element_at", (value, coordinate)))
    if operation == "bitreinterpret":
      scalar = self.geometry_scalar(source, index)
      return f"std::bit_cast<{CPP_TYPES[value.type.dtype]}>(static_cast<{CPP_TYPES[source.type.dtype]}>({scalar}))"
    if operation in {"lower", "upper"}:
      offset = 0 if operation == "lower" else self.scalar(value.type.shape[0])
      return self.geometry_scalar(source, f"(({index})+{offset})")
    if operation == "bitcast":
      before, after = ELEMENT_BYTES[source.type.dtype], ELEMENT_BYTES[value.type.dtype]
      uint_source, uint_target = f"std::uint{before * 8}_t", f"std::uint{after * 8}_t"
      if after >= before:
        words = []
        for piece in range(after // before):
          scalar = self.geometry_scalar(source, f"(({index})*{after // before}+{piece})")
          bits = f"std::bit_cast<{uint_source}>(static_cast<{CPP_TYPES[source.type.dtype]}>({scalar}))"
          shift = f"(std::endian::native==std::endian::little ? {piece * before * 8} : {(after - before - piece * before) * 8})"
          words.append(f"(static_cast<{uint_target}>({bits}) << {shift})")
        bits = " | ".join(words)
      else:
        ratio = before // after
        scalar = self.geometry_scalar(source, f"(({index})/{ratio})")
        bits = f"std::bit_cast<{uint_source}>(static_cast<{CPP_TYPES[source.type.dtype]}>({scalar}))"
        shift = (
          f"(std::endian::native==std::endian::little ? ({index})%{ratio} : {ratio - 1}-({index})%{ratio})*{after * 8}"
        )
        bits = f"({bits} >> ({shift}))"
      return f"std::bit_cast<{CPP_TYPES[value.type.dtype]}>(static_cast<{uint_target}>({bits}))"
    if operation == "concat":
      half = self.scalar(source.type.shape[0])
      predicate = I.Expr(I.ScalarType("bool"), "raw", data=f"({index})<{half}")
      return self.scalar_select(
        predicate,
        lambda: self.geometry_scalar(source, index),
        lambda: self.geometry_scalar(value.args[1], f"(({index})-{half})"),
        value.type.dtype,
      )
    if operation in {"interleave", "local_interleave_lower", "local_interleave_upper"}:
      if operation == "interleave":
        mapped = f"(({index})/2)"
      else:
        block = 16 // ELEMENT_BYTES[value.type.dtype]
        offset = 0 if operation.endswith("lower") else block // 2
        mapped = f"(({index})/{block}*{block}+({index})%{block}/2+{offset})"
      predicate = I.Expr(I.ScalarType("bool"), "raw", data=f"({index})%2==0")
      return self.scalar_select(
        predicate,
        lambda: self.geometry_scalar(source, mapped),
        lambda: self.geometry_scalar(value.args[1], mapped),
        value.type.dtype,
      )
    block = 16 // ELEMENT_BYTES[value.type.dtype]
    blocks = I.constant(value.type.shape[0]) // block

    def select(ordinal):
      chosen = value.data[ordinal]
      argument = value.args[chosen // blocks]
      mapped = f"({chosen % blocks * block}+({index})%{block})"
      if ordinal == len(value.data) - 1:
        return self.geometry_scalar(argument, mapped)
      predicate = I.Expr(I.ScalarType("bool"), "raw", data=f"({index})/{block}=={ordinal}")
      return self.scalar_select(
        predicate,
        lambda: self.geometry_scalar(argument, mapped),
        lambda: select(ordinal + 1),
        value.type.dtype,
      )

    return select(0)

  def geometry_vector(self, value, tag, shape, coords, active, axis):
    plan = self.prepared.register_programs.get(id(value))
    extent = I.constant(value.type.shape[0])
    geometry = self.tag_geometry.get(tag)
    target = self.environment.codegen_target
    native_bytes = (
      self.environment.vector_state.sve_bytes or 16
      if target.architecture == "arm"
      else (target.vector_bits or 128) // 8
    )
    if value.op == "bitcast" and value.args[0].op == "fill" and len(shape) == 1:
      source = value.args[0]
      before, after = ELEMENT_BYTES[source.type.dtype], ELEMENT_BYTES[value.type.dtype]
      fixed = self.fixed_tag_lanes.get(tag)
      minimum_bytes = (
        fixed * after
        if fixed is not None
        else (16 if target.architecture == "arm" else native_bytes) * 2 ** geometry[1]
        if geometry is not None and geometry[0]
        else 0
      )
      if before > after and minimum_bytes >= before:
        # A uniform word is periodic in its smaller lanes. Rotate for a
        # projected, possibly unaligned prefix, then broadcast and reinterpret
        # the entire register instead of extracting/inserting every byte.
        own = self.coordinates(value.type.shape, shape, coords)
        shift = f"static_cast<int>((({own[-1]}) % {before // after}) * {after * 8})"
        bits = f"std::bit_cast<std::uint{before * 8}_t>(static_cast<{CPP_TYPES[source.type.dtype]}>({self.scalar(source.args[0])}))"
        rotated = f"std::rotr({bits}, (std::endian::native == std::endian::little ? {shift} : -({shift})))"
        scalar = f"std::bit_cast<{CPP_TYPES[source.type.dtype]}>({rotated})"
        source_tag = f"vecops::vec::ViewAs<{CPP_TYPES[source.type.dtype]}, {tag}>"
        return f"vecops::vec::bitcast({tag}{{}}, {source_tag}{{}}, vecops::vec::fill({source_tag}{{}}, {scalar}))"
    capacity = (
      (native_bytes // ELEMENT_BYTES[value.type.dtype]) * (1 << geometry[1])
      if geometry is not None and geometry[0] and geometry[1] >= 0
      else None
    )
    complete = len(shape) == 1 and I.same_expr(shape[0], value.type.shape[0]) and tuple(coords) == ("0",)
    if (
      plan is not None
      and complete
      and (self.fixed_tag_lanes.get(tag) == extent or capacity is not None and capacity >= extent)
    ):
      # Closed fixed state updates use the same instruction program as stores.
      # Partial/multi-piece consumers retain exact indexed legalization.
      output_tag, result = self.register_program(
        plan.program,
        lambda index, own: self.vector(
          plan.program.inputs[index],
          own,
          plan.program.inputs[index].type.shape,
          ("0",),
          str(I.constant(plan.program.inputs[index].type.shape[0])),
        ),
        physical_tags=plan.tags,
      )
      result = result[0]
      if target.architecture != "arm" and capacity is not None:
        # State may use a larger physical word than its logical prefix. Pad
        # with named zero registers, not scalar insert loops or Vec arrays.
        while extent < capacity:
          next_tag, padded = self.unique("geometry_pad_tag"), self.unique("geometry_pad")
          self.line(f"using {next_tag}=vecops::vec::FixedTag<{CPP_TYPES[value.type.dtype]}, {extent * 2}>;")
          self.line(
            f"auto {padded}=vecops::vec::concat({next_tag}{{}}, {result}, vecops::vec::zeros({output_tag}{{}}));"
          )
          output_tag, result, extent = next_tag, padded, extent * 2
      return result
    result, lane = self.unique("geometry"), self.unique("geometry_lane")
    self.line(f"auto {result}=vecops::vec::zeros({tag}{{}});")
    own = self.coordinates(value.type.shape, shape, coords)
    self.opening(f"for (vecops::nint_t {lane}=0; {lane}<({active}); ++{lane})")
    scalar = self.geometry_scalar(value, f"(({own[-1]})+{lane})")
    self.line(f"{result}=vecops::vec::set({tag}{{}}, {result}, {lane}, {scalar});")
    self.close()
    return result

  def check_indices(self, index, predicate, tag, shape, coords, active, axis, view, memory_axis=0):
    indices = self.vector(index, tag, shape, coords, active, axis)
    valid = self.unique("valid_indices")
    self.line(
      f"auto {valid} = vecops::vec::mask_and({tag}{{}}, vecops::vec::cmpge({tag}{{}}, {indices}, vecops::vec::fill({tag}{{}}, 0)), vecops::vec::cmplt({tag}{{}}, {indices}, vecops::vec::fill({tag}{{}}, {self.scalar(view.shape[memory_axis])})));"
    )
    _, mask = self.memory_options(predicate, tag, shape, coords, active, axis)
    self.line(
      f'if (vecops::vec::mask_any({tag}{{}}, vecops::vec::mask_and({tag}{{}}, {mask}, vecops::vec::mask_not({tag}{{}}, {valid})))) throw std::out_of_range("indexed memory index out of bounds");'
    )

  def mask_query(self, expression):
    source = expression.args[0]
    result = self.unique("mask_result")
    op = expression.op
    init = (
      "true"
      if op in {"mask_all", "mask_none"}
      else "false"
      if op == "mask_any"
      else "-1"
      if op in {"mask_first", "mask_last"}
      else "0"
    )
    self.line(f"{CPP_TYPES[expression.type.dtype]} {result} = {init};")
    plan = self.prepared.primitives[id(expression)]
    self.emitted_regions.append(plan)

    def query(tags, coords, counts, roots, state):
      for tag, c, count, mask in zip(tags, coords, counts, roots[0]):
        active = self.unique("mask_active")
        operation = "mask_or" if op == "mask_all" else "mask_and"
        tail = f"vecops::vec::mwhilelt({tag}{{}}, 0, {count})"
        if op == "mask_all":
          tail = f"vecops::vec::mask_not({tag}{{}}, {tail})"
        self.line(f"auto {active} = vecops::vec::{operation}({tag}{{}}, {mask}, {tail});")
        value = f"vecops::vec::{op}({tag}{{}}, {active})"
        if op in {"mask_first", "mask_last"}:
          found = self.unique("mask_found")
          self.line(f"auto {found} = {value};")
          offset = "0"
          for coordinate, extent in zip(c, source.type.shape):
            offset = f"(({offset})*({self.scalar(extent)})+({coordinate}))"
          condition = f"{found} >= 0" + (f" && {result} < 0" if op == "mask_first" else "")
          self.line(f"if ({condition}) {result} = ({offset}) + {found};")
        else:
          operator = "+=" if op == "mask_count" else "|=" if op == "mask_any" else "&="
          self.line(f"{result} {operator} {value};")

    self.traverse_region(plan, query)
    return result

  def indexed_store(self, statement, future):
    # A scatter is never pointwise-in-place safe, even for identical roots.
    dependencies = {}
    for value in (statement.indices, statement.value, statement.mask):
      if value is not None:
        dependencies.update((id(load), load) for load in I.loads(value))
    for identity in self.live_loads(future):
      dependencies[identity] = self.active_loads[identity]
    for load in dependencies.values():
      self.snapshot(load, statement.destination, True)
    plan = self.prepared.primitives[id(statement)]
    self.emitted_regions.append(plan)
    shape = statement.indices.type.shape

    def write(tags, coords, counts, roots, state):
      for slot, (base, c, count) in enumerate(zip(tags, coords, counts)):
        tag = self.rebound_tag(statement.value.type.dtype, base)
        itag = self.rebound_tag("int64", base)
        index = I.Expr(I.VectorType("int64", shape), "vector_ref", data=roots[0][slot])
        predicate = (
          I.Expr(statement.mask.type, "vector_ref", data=roots[2][slot])
          if statement.mask is not None
          else I.literal(True)
        )
        self.check_indices(index, predicate, itag, shape, c, count, -1, statement.destination)
        indices = self.unique("scatter_indices")
        self.line(
          f"auto {indices} = vecops::vec::mul({itag}{{}}, {roots[0][slot]}, vecops::vec::fill({itag}{{}}, {self.scalar(statement.destination.strides[0])}));"
        )
        options, mask = self.memory_options(predicate, tag, shape, c, count, -1)
        if statement.unique:
          self.line(
            f"vecops::vec::store_convert({tag}{{}}, {self.pointer(statement.destination)}, {roots[1][slot]}, {options}, vecops::vec::opt::indexed({indices}));"
          )
        else:
          # Duplicate indices require ordered scalar writes, but the numeric
          # conversion still follows store_convert's public default contract.
          lane = self.unique("scatter_lane")
          self.opening(f"for (vecops::nint_t {lane}=0; {lane}<({count}); ++{lane})")
          self.opening(f"if (vecops::vec::get({tag}{{}}, {mask}, {lane}))")
          converted = self.scalar_convert(
            statement.destination.dtype,
            f"vecops::vec::get({tag}{{}}, {roots[1][slot]}, {lane})",
            statement.value.type.dtype,
          )
          self.line(
            f"{self.pointer(statement.destination)}[vecops::vec::get({itag}{{}}, {indices}, {lane})] = {converted};"
          )
          self.close()
          self.close()

    self.traverse_region(plan, write)
