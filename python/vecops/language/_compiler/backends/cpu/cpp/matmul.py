"""Print planned vector/FMA and matrix-instruction schedules.

The emitter owns no algorithm-selection passes and never calls ops::Matmul
or its private kernel. Registers are always independent C++ identifiers.
"""

from contextlib import contextmanager
from dataclasses import replace

from .... import ir as I
from ....dtypes import ELEMENT_BYTES
from ....ir.traversal import walk, walk_unique
from ....operations import Trait
from ..capabilities import CPP_TYPES
from ..matmul import runtime_quant_candidate
from .sme_packing import SMEPackingEmitter

FORMATS = {"auto": -1, "dense": 0, "amx": 1, "sme": 2}


class MatmulEmitter(SMEPackingEmitter):
  def matrix_loop(self, role, index, bindings, *, initialize=True, bound=None):
    """Print a previously planned iteration domain with target bindings."""
    loop = self.active_matmul_plan.schedule.loop(role)

    def bind(expr):
      return str(bindings[expr.data]) if expr.op == "schedule_parameter" else self.scalar(expr)

    start, stop, step = (bind(v[0]) for v in (loop.domain.starts, loop.domain.extents, loop.domain.steps))
    kind = loop.bound if bound is None else bound
    stop = f"std::max<vecops::nint_t>(1,{stop})" if kind == "nonempty" else stop
    condition = (
      f"({stop})-{index}>({step})"
      if kind == "lookahead"
      else f"({stop})-{index}>=({step})"
      if kind == "full-tiles"
      else f"{index}<({stop})"
    )
    init = f"vecops::nint_t {index}={start}" if initialize else ""
    self.opening(f"for ({init};{condition};{index}+={step})")

  def matrix_prefetch(self, operand, coords, guard):
    """Guard before forming a future address, including zero-sized operands."""
    if self.active_matmul_plan.schedule.pipeline == "off":
      return
    self.opening(f"if ({guard}) /* bounded future panel hint */")
    if operand.op == "packed_load":
      spatial, ki = coords if operand.data.side == "A" else coords[::-1]
      offset = self.packed_offset(operand.data, spatial, ki)
      self.line(f"__builtin_prefetch({self.pointer(operand.data.view)}+({offset}),0,3);")
    else:
      for load in I.loads(operand):
        own = self.coordinates(load.data.shape, operand.type.shape, coords)
        self.line(f"__builtin_prefetch({self.pointer(load.data, own)},0,3);")
    self.close()

  def matrix_lookahead(self, role, index, bindings, load, compute):
    """Peel prime/drain and rotate two named operand banks, no Vec arrays.

    Loading the next K slice before issuing current MACs exposes load latency
    to the compiler without changing MAC order or loading beyond the tail.
    """
    loop = self.active_matmul_plan.schedule.loop(role)
    stop = str(bindings[loop.domain.extents[0].data])
    step = str(bindings[loop.domain.steps[0].data])
    self.opening(f"if (({stop})>0) /* operand pipeline prime */")
    current = load("0")
    self.line(f"vecops::nint_t {index}=0;")
    self.matrix_loop(role, index, bindings, initialize=False, bound="lookahead")
    future = load(f"({index})+({step})")
    compute(current)
    for (_, destination), (_, source) in zip(current, future):
      self.line(f"{destination}={source};")
    self.close()
    compute(current)
    self.close()

  def matrix_permutation(self, network, inputs):
    return self.register_program(network, lambda row, tag: inputs[row], prefix="pack_permute")

  def amx_b_dense_condition(self, operand):
    loads = tuple(I.loads(operand))
    if not loads or any(len(load.data.shape) != 2 for load in loads):
      return None
    return " && ".join(
      f"(({self.scalar(load.data.strides[0])})==1 || ({self.scalar(load.data.shape[0])})==1)" for load in loads
    )

  def amx_b_register_pack(self, operand, origin, ki, spatial, k, store, *, spatial_full=False):
    """K-contiguous B: load/convert rows, transpose dwords (VNNI groups)."""
    kr = 64 // ELEMENT_BYTES[operand.type.dtype]
    tag = self.matrix_tag(operand.type.dtype, lanes=kr)
    bits = self.matrix_tag("uint32", lanes=16)
    known = I.constant(operand.type.shape[0])
    full = getattr(self, "matrix_k_full", False) or isinstance(known, int) and known % kr == 0
    previous_full = getattr(self, "full_vector", False)
    self.full_vector = full
    active = str(kr) if full else f"std::min<vecops::nint_t>({kr},({k})-({ki}))"
    values = []
    for row in range(16):
      value = self.unique("pack_b_dense")
      self.line(f"auto {value}=vecops::vec::zeros({bits}{{}});")
      if not spatial_full:
        self.opening(f"if (({origin})+{row}<({spatial}))")
      loaded = self.matrix_value(operand, tag, (ki, f"({origin})+{row}"), active, 0)
      if not full:
        loaded = self.zero_matrix_tail(tag, loaded, active)
      self.line(f"{value}=vecops::vec::bitcast({bits}{{}},{tag}{{}},{loaded});")
      if not spatial_full:
        self.close()
      values.append(value)
    self.line("#if defined(CPU_CAPABILITY_AVX512)")
    outtag, outputs = self.matrix_permutation(self.prepared.packing_permutation_x86, values)
    for group, value in enumerate(outputs):
      store(group, outtag, value)
    self.line("#else")
    outtag, outputs = self.matrix_permutation(self.prepared.packing_permutation, values)
    for group, value in enumerate(outputs):
      store(group, outtag, value)
    self.line("#endif")
    self.full_vector = previous_full

  def matrix_operands(self, product):
    return getattr(self, "dot_inputs", {}).get(id(product), product.args[:2])

  def matrix_tag(self, dtype, *, lanes=None, power=0):
    tag = self.unique("matrix_tag")
    declaration = f"FixedTag<{CPP_TYPES[dtype]}, {lanes}>" if lanes else f"ScalableTag<{CPP_TYPES[dtype]}, {power}>"
    self.line(f"using {tag} = vecops::vec::{declaration};")
    self.tag_dtypes[tag] = dtype
    if not hasattr(self, "matrix_tag_lanes"):
      self.matrix_tag_lanes = {}
    self.matrix_tag_lanes[tag] = lanes
    if lanes is not None:
      # Fixed one-lane A loads are full accesses. Keeping a synthetic tail
      # predicate here prevents folding scalar load+broadcast into memory FMA.
      self.fixed_tag_lanes[tag] = lanes
    if not hasattr(self, "matrix_tag_powers"):
      self.matrix_tag_powers = {}
    self.matrix_tag_powers[tag] = power
    return tag

  def packing_size(self, shape, dtype, format_, side):
    spatial, k = shape if side == "A" else shape[::-1]
    return f"vecops_dsl::pack_size<{CPP_TYPES[dtype]}, {FORMATS[format_]}>({self.scalar(spatial)}, {self.scalar(k)})"

  def packed_offset(self, packed, s, k):
    shape = packed.shape
    if packed.origins:
      os, ok = packed.origins if packed.side == "A" else packed.origins[::-1]
      s, k = f"(({s})+({self.scalar(os)}))", f"(({k})+({self.scalar(ok)}))"
    spatial, extent_k = shape if packed.side == "A" else shape[::-1]
    return f"vecops_dsl::pack_offset<{CPP_TYPES[packed.view.dtype]}, {FORMATS[packed.format]}, {'true' if packed.side == 'B' else 'false'}>({s}, {k}, {self.scalar(spatial)}, {self.scalar(extent_k)})"

  def packed_vector(self, value, tag, root_shape, coords, active, axis):
    packed = value.data
    if packed.format == "auto":
      result = self.unique("automatic_packed_vector")
      self.line(f"vecops::vec::Vec<{tag}> {result};")

      def assign(format_):
        own = replace(value, data=replace(packed, format=format_))
        self.retain_expression(own)
        vector = self.packed_vector(own, tag, root_shape, coords, active, axis)
        self.line(f"{result}={vector};")

      self.line("#if defined(HAS_AMX_TILE)")
      self.opening(f"if constexpr (vecops_dsl::packing_format<{CPP_TYPES[packed.view.dtype]},-1> == 1)")
      assign("amx")
      self.close()
      self.opening("else")
      assign("dense")
      self.close()
      self.line("#elif defined(HAS_SME)")
      assign("sme")
      self.line("#else")
      assign("dense")
      self.line("#endif")
      return result
    own = self.coordinates(value.type.shape, root_shape, coords)
    own_axis = axis - (len(root_shape) - len(value.type.shape))
    along_k = own_axis == (1 if packed.side == "A" else 0)
    s, k = own if packed.side == "A" else own[::-1]
    pointer = self.pointer(packed.view)
    offset = self.packed_offset(packed, s, k)
    native = self.tag_dtypes[tag] == packed.view.dtype
    if (
      packed.format == "dense"
      or along_k
      and packed.side == "A"
      and packed.format == "amx"
      and ELEMENT_BYTES[self.tag_dtypes[tag]] >= ELEMENT_BYTES[packed.view.dtype]
    ):
      name = self.unique("packed_stream")
      if along_k:
        stride = ""
      else:
        extent_k = packed.shape[1 if packed.side == "A" else 0]
        stride = f",vecops::vec::opt::strided({self.scalar(extent_k)})"
      self.line(
        f"auto {name}=vecops::vec::load_convert({tag}{{}},{pointer}+({offset}),vecops::vec::opt::first({active}){stride});"
      )
      return name
    kp = max(1, 4 // ELEMENT_BYTES[packed.view.dtype])
    # A fixed K tail may contain less than one packed instruction group.
    # Reinterpreting one bf16 lane as uint32 (or 1-3 i8 lanes as uint32)
    # is not a valid ViewAs. Preserve the grouped stream for whole groups;
    # use the scalar packed-offset path below for incomplete fixed vectors.
    complete_groups = self.fixed_tag_lanes.get(tag, kp) % kp == 0
    if along_k and complete_groups and (packed.format == "sme" or packed.side == "B" and packed.format == "amx"):
      input_tag = f"vecops::vec::Rebind<{CPP_TYPES[packed.view.dtype]},{tag}>"
      word_type = "std::uint64_t" if ELEMENT_BYTES[packed.view.dtype] == 8 else "std::uint32_t"
      bits = f"vecops::vec::ViewAs<{word_type},{input_tag}>"
      panel = f"vecops_dsl::pack_panel<{CPP_TYPES[packed.view.dtype]},{FORMATS[packed.format]}>()"
      name = self.unique("packed_k_stream")
      self.line(
        f"auto {name}=vecops::vec::bitcast({input_tag}{{}},{bits}{{}},vecops::vec::load({bits}{{}},reinterpret_cast<const {word_type}*>({pointer}+({offset})),vecops::vec::opt::first((({active})+{kp}-1)/{kp}),vecops::vec::opt::strided({panel})));"
      )
      if not native:
        converted = self.unique("packed_stream_conversion")
        self.line(f"auto {converted}=vecops::vec::convert({tag}{{}},{input_tag}{{}},{name});")
        name = converted
      return self.zero_matrix_tail(tag, name, active)
    name, lane = self.unique("packed_vector"), self.unique("packed_lane")
    self.line(f"auto {name} = vecops::vec::zeros({tag}{{}});")
    self.opening(f"for (vecops::nint_t {lane}=0; {lane} < {active}; ++{lane})")
    indices = [f"({c}) + {lane}" if i == own_axis else c for i, c in enumerate(own)]
    s, k = indices if packed.side == "A" else indices[::-1]
    pointer = self.pointer(packed.view)
    offset = self.packed_offset(packed, s, k)
    self.line(
      f"{name} = vecops::vec::set({tag}{{}}, {name}, {lane}, static_cast<vecops::vec::ElementOf<{tag}>>({pointer}[{offset}]));"
    )
    self.close()
    return name

  def matrix_expression_width(self, value):
    if value.op == "dot":
      return ELEMENT_BYTES[value.type.dtype]
    memory = ELEMENT_BYTES[value.data.dtype] if value.op == "load" else 0
    return max(
      (
        memory,
        ELEMENT_BYTES[value.type.dtype],
        *(self.matrix_expression_width(a) for a in value.args if isinstance(a.type, I.TileType)),
      )
    )

  def matrix_value(self, value, tag, coords, active, axis, *, fragmented=False):
    width = self.matrix_expression_width(value)
    n = getattr(self, "matrix_tag_lanes", {}).get(tag)
    power = getattr(self, "matrix_tag_powers", {}).get(tag, 0)
    dtype = self.tag_dtypes[tag]
    split = (
      n is not None
      and n > 1
      and n * width > 64
      or n is None
      and (1 << max(0, power)) * width > ELEMENT_BYTES[dtype] * (1 << max(0, -power))
    )
    if split:
      half = self.unique("matrix_half_tag")
      self.line(f"using {half}=vecops::vec::Half<{tag}>;")
      self.tag_dtypes[half] = dtype
      self.matrix_tag_lanes[half] = n // 2 if n is not None else None
      self.matrix_tag_powers[half] = power - 1
      halfwidth = self.vector_size(half)
      values = []
      for part in range(2):
        own = list(coords)
        own[axis] = f"({coords[axis]})+{part}*({halfwidth})"
        count = f"std::max<vecops::nint_t>(0,std::min<vecops::nint_t>({halfwidth},({active})-{part}*({halfwidth})))"
        name = self.unique("matrix_load_fragment")
        self.line(f"auto {name}=vecops::vec::zeros({half}{{}});")
        self.opening(f"if (({count})>0)")
        loaded = self.matrix_value(value, half, tuple(own), count, axis, fragmented=True)
        loaded = self.zero_matrix_tail(half, loaded, count)
        self.line(f"{name}={loaded};")
        self.close()
        values.append(name)
      result = self.unique("matrix_input")
      self.line(f"auto {result}=vecops::vec::concat({tag}{{}},{values[0]},{values[1]});")
      return result
    self.chunk_cache = {}
    return self.vector(value, tag, value.type.shape, coords, active, axis)

  def zero_matrix_tail(self, tag, value, active):
    name = self.unique("matrix_tail_zero")
    self.line(
      f"auto {name}=vecops::vec::blend({tag}{{}},vecops::vec::zeros({tag}{{}}),vecops::vec::mwhilelt({tag}{{}},0,{active}),{value});"
    )
    return name

  def pack_matrix(self, statement, *, dense_plan=None):
    source, packed = statement.source, statement.destination
    destination = packed.view
    shape, dtype, side, format_ = packed.shape, destination.dtype, packed.side, packed.format
    spatial, logical_k = shape if side == "A" else shape[::-1]
    s_extent, k_extent = self.scalar(spatial), self.scalar(logical_k)
    count = self.packing_size(shape, dtype, format_, side)
    # External output may have rank 1 or the exact C++ packed rank. Require
    # dense physical storage and enough capacity, never trust a raw pointer.
    actual = " * ".join(f"({self.scalar(v)})" for v in destination.shape) or "1"
    expected = I.literal(1)
    for extent, stride in zip(reversed(destination.shape), reversed(destination.strides)):
      self.line(
        f'if (({self.scalar(stride)}) != ({self.scalar(expected)})) throw std::invalid_argument("packed output must be dense");'
      )
      expected = I.binary("mul", expected, extent)
    self.line(f'if (({actual}) < ({count})) throw std::invalid_argument("packed output buffer too small");')
    for load in I.loads(source):
      self.snapshot(load, destination, True)
    self.line(f"/* pack {side}: input transform + storage conversion, zero padded tails */")
    code = f"vecops_dsl::packing_format<{CPP_TYPES[dtype]}, {FORMATS[format_]}>"
    self.opening(f"if constexpr ({code} == 0)")
    prepared = self.prepared.dense_packs.get(id(statement))
    if prepared is not None:
      copied, aliases = prepared
      for original, transposed in aliases:
        if id(original) in self.snapshots:
          saved, enabled = self.snapshots[id(original)]
          saved = (
            replace(saved, shape=(saved.shape[0], I.literal(1)), strides=(saved.strides[0], I.literal(0)))
            if len(saved.shape) == 1
            else replace(saved, shape=saved.shape[::-1], strides=saved.strides[::-1])
          )
          self.snapshots[id(transposed)] = (saved, enabled)
      self.store(copied)
    else:
      # Bind the frozen memory plan to this panel's logical origins and
      # physical destination. Printing does not rerun lowering or planning.
      logical = (spatial, logical_k)
      dense = replace(destination, shape=logical, strides=(logical_k, I.literal(1)))
      copied = source if side == "A" else self.transpose_matrix_expression(source)
      sink = I.Store(dense, copied, implementation="vector")
      plan = replace(dense_plan, statement=sink) if dense_plan is not None else None
      if not self.transpose_store(sink, plan, fallback=lambda: self.pack_dense(dense, copied)):
        self.pack_dense(dense, copied)
    self.close()
    self.opening("else")
    self.pack_interleaved(source, packed, s_extent, k_extent)
    self.close()

  def transpose_matrix_expression(self, expression):
    memo = {}
    result = I.transpose_matrix(expression, memo)
    for identity, value in memo.items():
      self.retain_expression(value)
      for mapping in (self.snapshots, self.materialized):
        if identity in mapping:
          view, enabled = mapping[identity]
          view = (
            replace(view, shape=(view.shape[0], I.literal(1)), strides=(view.strides[0], I.literal(0)))
            if len(view.shape) == 1
            else replace(view, shape=view.shape[::-1], strides=view.strides[::-1])
          )
          mapping[id(value)] = (view, enabled)
    return result

  def pack_dense(self, destination, source):
    # Dense conversion is also used for unusual ISA/dtype combinations. No
    # opaque packer call: memory ops stay visible to vector lowering.
    tag = self.matrix_tag(source.type.dtype)
    m, n = (self.scalar(v) for v in destination.shape)
    row, col = self.unique("pack_row"), self.unique("pack_col")
    width = self.vector_size(tag)
    self.opening(f"for (vecops::nint_t {row}=0; {row} < {m}; ++{row})")
    self.opening(f"for (vecops::nint_t {col}=0; {col} < {n}; {col} += {width})")
    active = f"std::min<vecops::nint_t>({width}, ({n})-{col})"
    value = self.matrix_value(source, tag, (row, col), active, 1)
    self.line(
      f"vecops::vec::store_convert({tag}{{}}, {self.pointer(destination, (row, col))}, {value}, vecops::vec::opt::first({active}));"
    )
    self.close()
    self.close()

  def pack_interleaved(self, source, packed, spatial, logical_k):
    dtype, format_ = packed.view.dtype, packed.format
    cpp, code = CPP_TYPES[dtype], FORMATS[format_]
    panel = self.unique("panel")
    block = self.unique("pack_kblock")
    self.line(f"const vecops::nint_t {panel} = vecops_dsl::pack_panel<{cpp}, {code}>();")
    self.line(f"const vecops::nint_t {block} = vecops_dsl::pack_kblock<{cpp}, {code}>();")
    amx = ELEMENT_BYTES[dtype] <= 2
    if amx:
      self.line("#if defined(HAS_AMX_TILE)")
      self.opening(f"if constexpr (vecops_dsl::packing_format<{cpp}, {code}> == 1)")
      self.pack_amx_vectors(source, packed, spatial, logical_k)
      self.close()
      self.opening("else")
      self.line("#endif")
    sme_raw = self.sme_raw_pack(source, packed, spatial, logical_k, code)
    self.pack_group_vectors(source, packed, spatial, logical_k, panel, block)
    if sme_raw:
      self.close_sme_raw_pack()
    if amx:
      self.line("#if defined(HAS_AMX_TILE)")
      self.close()
      self.line("#endif")

  def interleave_group(self, tag, values, dtype):
    if len(values) == 1:
      return tag, values[0]
    key = dtype, self.matrix_tag_lanes[tag], self.matrix_tag_powers[tag]
    program = self.prepared.packing_groups[key]
    output_tag, outputs = self.register_program(program, lambda index, own: values[index], prefix="pack_group")
    return output_tag, outputs[0]

  def pack_amx_vectors(self, source, packed, spatial, logical_k):
    dtype, side = packed.view.dtype, packed.side
    width = ELEMENT_BYTES[dtype]
    kr = 64 // width
    s, k, _row = (self.unique(v) for v in ("pack_panel_origin", "pack_k_origin", "pack_row"))
    self.line(f"vecops::nint_t {s}=0;")

    def panel(full_spatial):
      self.line(f"vecops::nint_t {k}=0;")
      self.opening(f"for (;({logical_k})-{k}>={kr};{k}+={kr})")
      self.pack_amx_tile(source, packed, s, k, spatial, logical_k, full_spatial, True)
      self.close()
      known = I.constant(packed.shape[1 if side == "A" else 0])
      if not (isinstance(known, int) and known % kr == 0):
        self.opening(f"if ({k}<({logical_k}))")
        self.pack_amx_tile(source, packed, s, k, spatial, logical_k, full_spatial, False)
        self.close()

    self.opening(f"for (;({spatial})-{s}>=16;{s}+=16) /* full packing panels */")
    panel(True)
    self.close()
    known = I.constant(packed.shape[0 if side == "A" else 1])
    if not (isinstance(known, int) and known % 16 == 0):
      self.opening(f"if ({s}<({spatial})) /* one spatial packing tail */")
      panel(False)
      self.close()

  def pack_amx_tile(self, source, packed, s, k, spatial, logical_k, full_spatial, full_k):
    """Full panels keep load/convert unmasked; transform padding only at tails."""
    dtype, side = packed.view.dtype, packed.side
    kr, kp = 64 // ELEMENT_BYTES[dtype], 4 // ELEMENT_BYTES[dtype]
    row = self.unique("pack_row")
    previous_k = getattr(self, "matrix_k_full", False)
    previous_full = getattr(self, "full_vector", False)
    self.matrix_k_full = full_k
    if side == "A":
      tag = self.matrix_tag(dtype, lanes=kr)
      active = str(kr) if full_k else f"std::min<vecops::nint_t>({kr}, ({logical_k})-{k})"
      self.full_vector = full_k
      self.opening(f"for (vecops::nint_t {row}=0; {row}<16; ++{row})")
      value = self.unique("packed_a_row")
      self.line(f"auto {value} = vecops::vec::zeros({tag}{{}});")
      if not full_spatial:
        self.opening(f"if ({s}+{row} < ({spatial}))")
      loaded = self.matrix_value(source, tag, (f"({s}+{row})", k), active, 1)
      if not full_k:
        loaded = self.zero_matrix_tail(tag, loaded, active)
      self.line(f"{value} = {loaded};")
      if not full_spatial:
        self.close()
      offset = self.packed_offset(packed, f"({s}+{row})", k)
      self.line(f"vecops::vec::store({tag}{{}}, {self.pointer(packed.view)}+({offset}), {value});")
      self.close()
    else:
      dense = self.amx_b_dense_condition(source)
      if dense:
        self.opening(f"if ({dense}) /* dword transpose avoids narrow gathers */")

        def store_group(group, tag, value):
          offset = self.packed_offset(packed, s, f"({k})+{group * kp}")
          self.line(
            f"vecops::vec::store({tag}{{}}, reinterpret_cast<std::uint32_t*>({self.pointer(packed.view)}+({offset})),{value});"
          )

        self.amx_b_register_pack(source, s, k, spatial, logical_k, store_group, spatial_full=full_spatial)
        self.close()
        self.opening("else")
      tag = self.matrix_tag(dtype, lanes=16)
      active = "16" if full_spatial else f"std::min<vecops::nint_t>(16, ({spatial})-{s})"
      self.full_vector = full_spatial
      self.opening(f"for (vecops::nint_t {row}=0; {row} < {kr}; {row} += {kp})")
      values = []
      for group in range(kp):
        value = self.unique("packed_b_column")
        self.line(f"auto {value} = vecops::vec::zeros({tag}{{}});")
        if not full_k:
          self.opening(f"if ({k}+{row}+{group} < ({logical_k}))")
        loaded = self.matrix_value(source, tag, (f"({k}+{row}+{group})", s), active, 1)
        if not full_spatial:
          loaded = self.zero_matrix_tail(tag, loaded, active)
        self.line(f"{value} = {loaded};")
        if not full_k:
          self.close()
        values.append(value)
      outtag, value = self.interleave_group(tag, values, dtype)
      offset = self.packed_offset(packed, s, f"({k}+{row})")
      self.line(f"vecops::vec::store({outtag}{{}}, {self.pointer(packed.view)}+({offset}), {value});")
      self.close()
      if dense:
        self.close()
    self.matrix_k_full, self.full_vector = previous_k, previous_full

  def pack_group_vectors(self, source, packed, spatial, logical_k, panel, block):
    dtype, side = packed.view.dtype, packed.side
    kp = max(1, 4 // ELEMENT_BYTES[dtype])
    power = -(kp.bit_length() - 1)
    tag = self.matrix_tag(dtype, power=power)
    lanes = self.vector_size(tag)
    sp, kg, s = (self.unique(v) for v in ("pack_panel_origin", "pack_group", "pack_spatial"))
    self.opening(f"for (vecops::nint_t {sp}=0; {sp} < ({spatial}); {sp} += {panel})")
    self.opening(f"for (vecops::nint_t {kg}=0; {kg} < ({logical_k}); {kg} += {kp})")
    self.opening(f"for (vecops::nint_t {s}={sp}; {s} < {sp}+{panel}; {s} += {lanes})")
    chunk = f"std::min<vecops::nint_t>({lanes}, {sp}+{panel}-{s})"
    active = f"std::max<vecops::nint_t>(0, std::min<vecops::nint_t>({chunk}, ({spatial})-{s}))"
    values = []
    for group in range(kp):
      value = self.unique("packed_group")
      self.line(f"auto {value} = vecops::vec::zeros({tag}{{}});")
      self.opening(f"if ({kg}+{group} < ({logical_k}) && ({active}) > 0)")
      coords = (s, f"({kg}+{group})") if side == "A" else (f"({kg}+{group})", s)
      loaded = self.matrix_value(source, tag, coords, active, 0 if side == "A" else 1)
      loaded = self.zero_matrix_tail(tag, loaded, active)
      self.line(f"{value} = {loaded};")
      self.close()
      values.append(value)
    outtag, value = self.interleave_group(tag, values, dtype)
    offset = self.packed_offset(packed, s, kg)
    self.line(
      f"vecops::vec::store({outtag}{{}}, {self.pointer(packed.view)}+({offset}), {value}, vecops::vec::opt::first(({chunk})*{kp}));"
    )
    self.close()
    self.close()
    self.close()

  def matmul_store(self, statement):
    previous_grouped = getattr(self, "matrix_grouped_epilogue", False)
    self.matrix_grouped_epilogue = id(statement) in self.prepared.grouped_epilogues
    try:
      return self._matmul_store(statement)
    finally:
      self.matrix_grouped_epilogue = previous_grouped

  def _matmul_store(self, statement):
    products = {id(e): e for e in walk(statement.value) if e.op == "dot"}
    if not products:
      return False
    if len(products) != 1:
      raise TypeError("one fused store currently supports one matrix product; use an owned tile between dependent dots")
    product = next(iter(products.values()))
    if len(statement.destination.shape) != 2:
      raise TypeError("matrix product must be stored into a rank-two region")
    old_materialized = dict(self.materialized)
    old_validated = getattr(self, "matrix_validated", False)

    def operands(value):
      yield value, None
      if value.op != "dot":
        for argument in value.args:
          # Shape-changing operations are boundaries, not elementwise DAG
          # edges. Validate children in their own coordinates (e.g. a row
          # reduction expanded to [M,1] is not a [M] column broadcast).
          if self.environment.operations.has(value, Trait.ELEMENTWISE | Trait.MEMORY_READ):
            yield argument, value
          yield from operands(argument)

    for root in (*product.args, statement.value):
      if not isinstance(root.type, I.TileType):
        continue
      for value, parent in operands(root):
        if not isinstance(value.type, I.TileType):
          continue
        if parent is not None:
          for source, destination in zip(reversed(value.type.shape), reversed(parent.type.shape)):
            if I.constant(source) == 1 or I.same_expr(source, destination):
              continue
            self.line(f"vecops_dsl::check_broadcast({self.scalar(source)},{self.scalar(destination)});")
        if value.op == "packed_load":
          packed = value.data
          view = packed.view
          expected = I.literal(1)
          for extent, stride in zip(reversed(view.shape), reversed(view.strides)):
            self.line(
              f'if (({self.scalar(stride)})!=({self.scalar(expected)})) throw std::invalid_argument("packed input must be dense");'
            )
            expected = I.binary("mul", expected, extent)
          count = self.packing_size(packed.shape, view.dtype, packed.format, packed.side)
          size = ", ".join(self.scalar(v) for v in view.shape)
          self.line(
            f'if (vecops_dsl::checked_shape({{{size}}})<({count})) throw std::invalid_argument("packed input buffer too small");'
          )
          if not view.owned:
            self.line(
              f'if ({self.overlap(view, statement.destination)}) throw std::invalid_argument("packed input and matrix output may not overlap");'
            )
    self.matrix_validated = True
    m, n = (self.scalar(v) for v in product.type.shape)
    # Dynamic small families use exact TILECFG row/column counts. Empty
    # matrices must skip resource entry (rows=0 with nonzero columns would
    # otherwise be an invalid AMX configuration), not just skip their loops.
    self.opening(f"if (({m})>0 && ({n})>0) /* empty products need no resources */")
    plan = self.prepared.matmuls[id(product)]
    self.active_matmul_plan = plan
    self.emitted_regions.append(plan)
    self.dot_inputs = {id(product): tuple(policy.expression for policy in plan.operands)}
    a, b = self.matrix_operands(product)
    dtype, implementation = product.type.dtype, product.data.implementation
    native_half = a.type.dtype == b.type.dtype and a.type.dtype in {"bfloat16", "float16"} and dtype == "float32"
    native_int = a.type.dtype in {"int8", "uint8"} and b.type.dtype in {"int8", "uint8"} and dtype == "int32"
    amx_types = native_half or native_int
    sme_types = amx_types or a.type.dtype == b.type.dtype == dtype and dtype in {"float32", "float64"}
    if implementation == "amx" and not amx_types or implementation == "sme" and not sme_types:
      raise TypeError(f"unsupported {implementation} dot dtype combination")
    # Elide impossible *architecture* branches before printing families and
    # transformed packing producers. Generic artifacts retain both branches;
    # per-ISA feature/dtype availability remains guarded by C++ macros. This
    # consults the actual frozen lowering target, never cpu_cost_target.
    target = self.environment.codegen_target
    amx = amx_types and target.architecture != "arm" and target.isa != "scalar"
    sme = sme_types and target.architecture != "x86" and target.isa != "scalar"
    # This family is ordinary SVE, despite consuming the SME packing ABI.
    # Select it before FA64/streaming guards. One shared fallback retains all
    # existing hardware dispatch and cache policies on an ineligible shape.
    runtime_quant = sme and implementation == "auto" and self.sve_runtime_quant(statement, product)
    if runtime_quant:
      self.opening("")
    # A family candidate is not a selected algorithm. Prepare the normal
    # fusion/materialization/packing policy only in the generic branch;
    # neither a rejected guard nor an ISA fallback inherits the experiment's
    # raw-quantization policy. The accepted branch consumes the original DAG.
    for policy in plan.operands:
      expression = policy.expression
      self.line(f"/* input fusion: {policy.reason}; estimated op cost {policy.operation_cost} */")
      if policy.materialization is not None:
        view = policy.materialization.destination
        count = ", ".join(self.scalar(v) for v in view.shape)
        # Hoist the scratch owner like packing/partial buffers: a serial loop
        # reuses capacity, while each invocation still owns independent data.
        self.buffer_declarations[view.storage] = f"  vecops_dsl::Buffer<{CPP_TYPES[view.dtype]}> {view.storage};"
        enabled = self.unique("input_cache")
        self.line(f"const bool {enabled}=({self.scalar(policy.condition)});")
        self.opening(f"if ({enabled})")
        self.line(f"{view.storage}.resize(vecops_dsl::checked_shape({{{count}}}));")
        self.store(policy.materialization)
        self.close()
        self.materialized[id(expression)] = (view, enabled)
    # Hardware selection is target-conditional. No host ISA assumptions leak
    # into cross-compiled artifacts; explicit requests never silently fallback.
    guard = (
      "HAS_AMX_BF16" if a.type.dtype == "bfloat16" else "HAS_AMX_FP16" if a.type.dtype == "float16" else "HAS_AMX_INT8"
    )
    if amx and implementation in {"auto", "amx"}:
      self.line(f"#if defined({guard})")
      if implementation == "auto":
        m, n = (self.scalar(v) for v in product.type.shape)
        k = self.scalar(a.type.shape[1])
        # Skinny vector leaves avoid tile configuration. Wider raw BF16
        # row-GEMV selection is refined from paired benchmark measurements.
        skinny = f"(({m}) == 1) || (({n}) == 1) || (({m}) > 0 && ({n}) <= 16 / ({m}))"
        self.opening(f"if ({skinny})")
        if product.data.tiled and any(
          v is not None for v in (product.data.block_m, product.data.block_n, product.data.block_k)
        ):
          cache = self.matrix_cache_blocks(product, 4, 4, 32)
          self.blocked_matrix_matmul(statement, product, cache, "dense")
        else:
          self.vector_gemv_matmul(statement, product)
        self.close()
        self.opening("else")
      self.amx_matmul(statement, product)
      if implementation == "auto":
        self.close()
      self.line("#else")
    if sme and implementation in {"auto", "sme"}:
      smeguard = "HAS_SME_F64F64" if a.type.dtype == "float64" else "HAS_SME"
      self.line(
        f"#if defined({smeguard}) && defined(HAS_SME_FA64) && (!defined(HAS_FIXED_SVE_BITS) || (defined(FIXED_STREAMING_SVE_BITS) && FIXED_SVE_BITS == FIXED_STREAMING_SVE_BITS))"
      )
      if implementation == "auto":
        # Match the C++ SME SmallVector family before entering SM: these
        # kernels use ordinary SVE and need no ZA initialization/drain.
        limit = 16 if a.type.dtype == "float16" else 64
        skinny = f"(({m})==1 && ({n})<={limit}) || (({n})==1 && ({m})<={limit})"
        packed = all(e.op == "packed_load" and e.data.format in {"sme", "auto"} for e in (a, b))
        if packed:
          # The packed-dot ladder mirrors the C++ SME primary/tiny leaves.
          # K-contiguous gathers now serve both packed A and B, so no lane
          # extraction loop or streaming-state overhead is necessary.
          k = self.scalar(a.type.shape[1])
          max_k = 1025 if a.type.dtype == "bfloat16" else 513
          primary = f"((({m})==2 && ({n})==8) || (({m})==8 && ({n})==2)) && ({k})<={max_k}"
          tiny = f"({m})==2 && (({n})==2 || ({n})==4) && ({k})<={max_k}"
          square = f"((({m})==4 && ({n})==4) || (({m})==4 && ({n})==2)) && ({k})<=257"
          skinny = f"({skinny}) || ({primary}) || ({tiny}) || ({square})"
          if (
            a.type.dtype == b.type.dtype == "bfloat16"
            and dtype == "float32"
            and a.data.side == "A"
            and b.data.side == "B"
            and not a.data.origins
            and not b.data.origins
            and self.hardware_scope is None
          ):
            # Four complete physical M blocks are already available in the
            # native packing ABI. The ordinary K-vectorized GEMV leaf must
            # gather these spatial panels, whereas the existing 4x1 ZA owner
            # reads four contiguous A vectors and one shared B per K group.
            # Limit this correction to whole, explicitly identified panels;
            # cache windows and the primary/tiny PackedDot ladder stay intact.
            lanes = self.sme_scalar_lanes("float32")
            spatial = f"({n})==1 && ({m})>3*({lanes}) && ({m})<=4*({lanes})"
            skinny = f"({skinny}) && !({spatial}) /* packed AB four-M-block column uses ZA */"
        elif (
          b.op == "packed_load"
          and b.data.format in {"sme", "auto"}
          and not any(value.op == "packed_load" for value in walk_unique(a))
        ):
          # The generic N-vectorized leaf has only one useful lane for a
          # column GEMV and reads this packed B through scalar lane inserts.
          # Unlike a raw K-contiguous B, a single SME-packed B is not an
          # ordinary-SVE skinny candidate in the native scheduler. Preserve
          # the existing shape limits but route N==1 through the shared ZA
          # spatial family (four M tiles at M==4*ZA lanes), including its
          # fused raw A, accumulator and output paths. No implicit packing.
          skinny = f"({skinny}) && ({n})!=1 /* single-packed B column uses ZA spatial lanes */"
          if a.type.dtype == "bfloat16":
            # A single SME-packed B is likewise not K-contiguous for the
            # ordinary row leaf: rebuilding every N vector uses scalar lane
            # inserts. The shared 1x2/1x4 ZA family loads its native K groups
            # directly. Leave FP16, raw and PackedAB dispatch unchanged.
            skinny = f"({skinny}) && ({m})!=1 /* single-packed B row uses ZA spatial lanes */"
        self.opening(f"if ({skinny}) /* SME ordinary-SVE skinny/packed-dot family */")
        if packed:
          if not self.sve_packed_mmla(statement, product):
            self.vector_gemv_matmul(statement, product)
        else:
          self.vector_matrix_leaf(statement, product)
        self.close()
        self.opening("else")
      self.sme_matmul(statement, product)
      if implementation == "auto":
        self.close()
      self.line("#else")
    if implementation in {"amx", "sme"}:
      self.line(f'#error "explicit {implementation} dot requires matching target ISA/dtype and vector lengths"')
    else:
      # With K-contiguous columns, an inner-product leaf avoids a gather per
      # N lane. General dense [K,N] B uses the outer-product family.
      cache = self.matrix_cache_blocks(product, 4, 16, 32)
      if cache is not None:
        self.blocked_matrix_matmul(statement, product, cache, "dense")
      else:
        self.vector_matrix_leaf(statement, product)
    if sme and implementation in {"auto", "sme"}:
      self.line("#endif")
    if amx and implementation in {"auto", "amx"}:
      self.line("#endif")
    if runtime_quant:
      self.close()
    self.close()
    self.chunk_cache = {}
    self.dot_inputs = {}
    self.materialized = old_materialized
    self.matrix_validated = old_validated
    return True

  def vector_matrix_leaf(self, statement, product):
    b = self.matrix_operands(product)[1]
    if b.op == "load" and I.constant(b.data.strides[0]) == 1 or b.op == "packed_load" and b.data.format == "dense":
      self.vector_gemv_matmul(statement, product)
    else:
      self.vector_matmul(statement, product)

  def sve_runtime_quant(self, statement, product):
    """Print an opt-in row-vector dot with one shared general fallback.

    Four quantized K bytes form each repeated A word. Conversion happens
    before tail zeroing, so a nonzero zero-point cannot poison padded lanes.
    VL controls arithmetic lanes; the independent SVL-sized packed panel
    bounds every B load. Neither ZA nor a streaming state owner is needed.
    """
    if not self.active_matmul_plan.sve_runtime_quant or self.hardware_scope is not None:
      return False
    a, b = self.matrix_operands(product)
    if not runtime_quant_candidate(a, b, product.type.dtype, self.environment.operations):
      return False
    if product.data.implementation != "auto":
      return False
    explicit_blocks = product.data.tiled and any(
      v is not None for v in (product.data.block_m, product.data.block_n, product.data.block_k)
    )
    m, n = (self.scalar(v) for v in product.type.shape)
    k = self.scalar(a.type.shape[1])
    packed = b.data
    os, ok = packed.origins or (I.literal(0), I.literal(0))
    # Logical B origins are (K,N), unlike the physical spatial/K layout.
    origin_k, origin_n = self.scalar(os), self.scalar(ok)
    format_code = FORMATS[packed.format]
    gate = f"vecops_dsl::packing_format<vecops::int8_t,{format_code}> == 2 && ({origin_k})%4==0"
    # One whole-problem three-matrix storage budget, for dot and library
    # alike. Include the producer's FP32 storage width, not post-cast U8 or
    # MC/NC/KC leaf extents. A cache leaf must never retry a rejected family.
    gate += (
      f" && vecops_dsl::matmul_fits({m},{n},{k},{self.matrix_expression_width(a)},1,4,"
      f"{self.active_matmul_plan.runtime_unblocked_bytes})"
    )
    self.line("#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE_MATMUL_INT8) && defined(HAS_SME)")
    self.opening(f"if ({gate}) /* ordinary-SVE runtime-quantized packed-B dot */")
    if explicit_blocks:
      span = "2*vecops::vec::sme::streaming_lanes<vecops::int32_t>()"
      cache = self.matrix_cache_blocks(product, span, span, 4, format_="sme")
      self.blocked_matrix_matmul(statement, product, cache, "sme", k_alignment=4, runtime_quant_family=True)
    else:
      self.sve_runtime_quant_rows(statement, product, m, n, k, origin_n)
    self.close()
    self.line("else")
    self.line("#endif")
    return True

  def sve_runtime_quant_rows(self, statement, product, m, n, k, origin_n):
    """FP32 pointwise DAG -> four U8 bytes -> ordinary-SVE widening dot."""
    a, b = self.matrix_operands(product)
    tag = self.matrix_tag("int32")
    btag, atag, bits = (self.matrix_tag(dtype) for dtype in ("int8", "uint8", "uint32"))
    qtag, wordtag = self.matrix_tag("uint8", lanes=4), self.matrix_tag("uint32", lanes=1)
    lanes, panel, mm, nn, kk, active, bulk, weights = (
      self.unique(v)
      for v in (
        "runtime_quant_lanes",
        "runtime_quant_panel",
        "runtime_quant_m",
        "runtime_quant_n",
        "runtime_quant_k",
        "runtime_quant_active",
        "runtime_quant_bulk",
        "runtime_quant_weights",
      )
    )
    self.line(f"const vecops::nint_t {lanes}={self.vector_size(tag)};")
    self.line(f"const vecops::nint_t {panel}=vecops_dsl::pack_panel<vecops::int8_t,{FORMATS[b.data.format]}>();")
    self.line(f"const vecops::nint_t {bulk}=({k})-({k})%4;")
    self.opening(f"for (vecops::nint_t {mm}=0;{mm}<({m});++{mm})")
    self.opening(f"for (vecops::nint_t {nn}=0;{nn}<({n});)")
    self.line(
      f"const vecops::nint_t {active}=std::min<vecops::nint_t>({lanes},std::min<vecops::nint_t>(({n})-{nn},{panel}-(({origin_n})+{nn})%{panel}));"
    )
    self.line(
      f"const auto* const {weights}=({k})>0 ? {self.pointer(b.data.view)}+({self.packed_offset(b.data, nn, '0')}) : {self.pointer(b.data.view)};"
    )
    result = self.unique("runtime_quant_prior")
    self.line(f"auto {result}=vecops::vec::zeros({tag}{{}});")
    phase = getattr(self, "matrix_phase", None)
    coords = (mm, nn)
    if phase is not None:
      self.opening(f"if ({phase[0]})")
    if product.args[2:]:
      self.chunk_cache = {}
      initial = self.vector(product.args[2], tag, product.type.shape, coords, active, 1)
      self.line(f"{result}={initial};")
    if phase is not None:
      self.close()
      self.opening("else /* corrected integer running sum, not correction again */")
      stride = phase[2].strides[1]
      extra = "" if I.constant(stride) == 1 else f",vecops::vec::opt::strided({self.scalar(stride)})"
      self.line(
        f"{result}=vecops::vec::load({tag}{{}},{self.pointer(phase[2], coords)},vecops::vec::opt::first({active}){extra});"
      )
      self.close()
    self.line(f"vecops::nint_t {kk}=0;")

    def step(count, full):
      previous_full = getattr(self, "full_vector", False)
      self.full_vector = full
      quantized = self.matrix_value(a, qtag, (mm, kk), count, 1)
      if not full:
        zero = self.unique("runtime_quant_tail_zero")
        self.line(
          f"auto {zero}=vecops::vec::blend({qtag}{{}},vecops::vec::zeros({qtag}{{}}),vecops::vec::mwhilelt({qtag}{{}},0,{count}),{quantized});"
        )
        quantized = zero
      group, repeated, values = (
        self.unique(v) for v in ("runtime_quant_word", "runtime_quant_repeated", "runtime_quant_b")
      )
      self.line(
        f"const auto {group}=vecops::vec::get({wordtag}{{}},vecops::vec::bitcast({wordtag}{{}},{qtag}{{}},{quantized}),0);"
      )
      self.line(f"auto {repeated}=vecops::vec::bitcast({atag}{{}},{bits}{{}},vecops::vec::fill({bits}{{}},{group}));")
      self.line(
        f"auto {values}=vecops::vec::load({btag}{{}},{weights}+{kk}*{panel},vecops::vec::opt::first({active}*4),vecops::vec::opt::zero);"
      )
      self.line(f"{result}=vecops::vec::widening_dot({tag}{{}},{atag}{{}},{btag}{{}},{repeated},{values},{result});")
      self.full_vector = previous_full

    self.opening(f"for (;{kk}<{bulk};{kk}+=4) /* four complete quantized K lanes */")
    step("4", True)
    self.close()
    self.opening(f"if ({kk}<({k})) /* quantized K tail: mask after conversion */")
    step(f"({k})-{kk}", False)
    self.close()
    self.vector_matrix_epilogue(statement, product, tag, coords, active, result)
    self.line(f"{nn}+={active};")
    self.close()
    self.close()
    self.chunk_cache = {}

  def sve_packed_mmla(self, statement, product, *, fallback=None):
    """Ordinary-SVE segment MACs over the existing SME PackedAB ABI.

    Keep a general widening-dot fallback: a same-VL/SVL pair alone does not
    prove that one register can contain a particular small matrix. Sliced
    packed windows must also stay in a panel and start on a K-group boundary.
    The final partial K group is legal only at the original matrix's end.
    BF16 clears its nonlogical K half explicitly, including external packed
    storage whose padding is not zero. Full-group register loads remain
    bounded by the current physical panel, even for shifted cache windows.
    """
    a, b = self.matrix_operands(product)
    dtype = a.type.dtype
    if (
      self.hardware_scope is not None
      or dtype != b.type.dtype
      or (dtype, product.type.dtype) not in {("bfloat16", "float32"), ("int8", "int32"), ("uint8", "int32")}
      or any(operand.op != "packed_load" or operand.data.format not in {"sme", "auto"} for operand in (a, b))
      # An optional auto-packed A has no storage on its raw branch. The SME
      # family dispatches that branch before K; ordinary MMLA must not mistake
      # the conditional descriptor for an unconditionally available buffer.
      or any(id(operand) in getattr(self, "sme_optional_packs", {}) for operand in (a, b))
    ):
      return False
    state = self.environment.vector_state
    vl, svl = getattr(state, "sve_bytes", None), getattr(state, "sme_bytes", None)
    if vl is not None and svl is not None and vl != svl:
      return False
    m, n = (self.scalar(v) for v in product.type.shape)
    k = self.scalar(a.type.shape[1])
    known_m, known_n = (I.constant(v) for v in product.type.shape)
    max_k, kp = (1025, 2) if dtype == "bfloat16" else (513, 4)
    cases = (
      (2, 2, 16, max_k),
      (2, 4, 32, max_k),
      (2, 8, 64, max_k),
      (4, 2, 32, 257),
      (4, 4, 32, 257),
      (8, 2, 64, max_k),
    )
    cases = tuple(
      case
      for case in cases
      if (not isinstance(known_m, int) or known_m == case[0])
      and (not isinstance(known_n, int) or known_n == case[1])
      and (vl is None or vl >= case[2])
    )
    if not cases:
      return False
    feature = "__ARM_FEATURE_SVE_BF16" if dtype == "bfloat16" else "__ARM_FEATURE_SVE_MATMUL_INT8"
    self.line(f"#if defined({feature})")
    bytes_ = self.vector_size("vecops::vec::ScalableTag<vecops::uint8_t, 0>")
    panels = tuple(
      f"vecops_dsl::pack_panel<{CPP_TYPES[dtype]}, {FORMATS[operand.data.format]}>()" for operand in (a, b)
    )
    shape_gate = " || ".join(
      f"(({m})=={rows} && ({n})=={columns} && ({k})<={limit} && ({bytes_})>={minimum})"
      for rows, columns, minimum, limit in cases
    )
    geometry = [f"({panels[0]})==({panels[1]})", f"({bytes_})==2*({panels[0]})"]
    full_loads = []
    for operand, spatial, panel in zip((a, b), (m, n), panels):
      data = operand.data
      origins = data.origins or (I.literal(0), I.literal(0))
      os, ok = origins if data.side == "A" else origins[::-1]
      os, ok = self.scalar(os), self.scalar(ok)
      original_k = self.scalar(data.shape[1 if data.side == "A" else 0])
      geometry.extend(
        (
          f"({os})>=0 && ({spatial})<=({panel})-({os})%({panel})",
          f"({ok})>=0 && ({ok})%{kp}==0",
          f"(({k})%{kp}==0 || ({k})==({original_k})-({ok}))",
        )
      )
      # A full ordinary-SVE register may read spatial padding, but never
      # cross the current physical K group's panel boundary. MMLA segments
      # do not reduce across rows/columns: padding contributes only to
      # discarded outputs, irrespective of its stored value.
      full_loads.append(f"({os})%({panel})<=({panel})-({bytes_})/{ELEMENT_BYTES[dtype] * kp}")
    self.opening(f"if (({shape_gate}) && {' && '.join(geometry)}) /* ordinary-SVE PackedDot MMLA */")
    atag, tag = self.matrix_tag(dtype), self.matrix_tag(product.type.dtype)
    groups, group = self.unique("mmla_groups"), self.unique("mmla_group")
    self.line(f"const vecops::nint_t {groups}=({k})/{kp}+(({k})%{kp}!=0);")
    bases, strides = [], []
    for operand, panel in zip((a, b), panels):
      base, stride = self.unique("mmla_packed_base"), self.unique("mmla_group_stride")
      pointer = self.pointer(operand.data.view)
      # An empty K has no allocation. Do not form even a zero-offset pointer
      # expression through that null storage while initializing accumulators.
      offset = self.packed_offset(operand.data, "0", "0")
      self.line(f"const auto* {base}=({k})>0 ? {pointer}+({offset}) : {pointer};")
      self.line(f"const vecops::nint_t {stride}=({panel})*{kp};")
      bases.append(base)
      strides.append(stride)
    row_families = tuple(sorted({case[0] for case in cases}, reverse=True))
    for index, rows in enumerate(row_families):
      self.opening(f"{'if' if index == 0 else 'else if'} (({m})=={rows}) /* MMLA {rows}-row family */")
      self.sve_packed_mmla_rows(
        statement, product, atag, tag, rows, n, k, groups, group, bases, strides, kp, full_loads
      )
      self.close()
    self.close()
    self.line("else")
    # The fallback belongs to the same lexical if on MMLA targets and is an
    # unconditional block elsewhere. Emit the large GEMV family only once.
    self.line("#endif")
    self.opening("")
    (fallback or self.vector_gemv_matmul)(statement, product)
    self.close()
    return True

  def sve_packed_mmla_rows(
    self, statement, product, atag, tag, rows, n, k, groups, group, bases, strides, kp, full_loads
  ):
    """Paired K groups, named accumulators, then logical-coordinate fusion."""
    bf16 = self.tag_dtypes[atag] == "bfloat16"
    full_groups = self.unique("mmla_full_groups") if bf16 else groups
    if bf16:
      self.line(f"const vecops::nint_t {full_groups}=({k})/{kp};")
    accumulators = tuple(self.unique("mmla_accumulator") for _ in range(2 if rows == 4 else 1))
    for accumulator in accumulators:
      self.line(f"auto {accumulator}=vecops::vec::zeros({tag}{{}});")

    def clear_partial_k(value):
      # The final physical dword has only its low BF16 half in logical K.
      # Clear bits, not FP values: arbitrary NaN/nonzero external padding
      # must not participate in MMLA or alter valid lanes.
      bits = "vecops::vec::ScalableTag<vecops::uint32_t, 0>"
      self.line(
        f"{value}=vecops::vec::bitcast({atag}{{}},{bits}{{}},"
        f"vecops::vec::bit_and({bits}{{}},vecops::vec::bitcast({bits}{{}},{atag}{{}},{value}),"
        f"vecops::vec::fill({bits}{{}},vecops::uint32_t{{0xffff}})));"
      )

    def step(paired, *, full=False, partial=False):
      inputs = []
      for base, stride, active in zip(bases, strides, (str(rows * kp), f"({n})*{kp}")):
        first, second = self.unique("mmla_group_first"), self.unique("mmla_group_second")
        options = "" if full else f",vecops::vec::opt::first({active}),vecops::vec::opt::zero"
        self.line(f"auto {first}=vecops::vec::load({atag}{{}},{base}+{group}*{stride}{options});")
        if partial and not paired:
          self.opening(f"if ({group}=={full_groups}) /* partial logical K, not padding */")
          clear_partial_k(first)
          self.close()
        if paired:
          self.line(f"auto {second}=vecops::vec::load({atag}{{}},{base}+({group}+1)*{stride}{options});")
          if partial:
            clear_partial_k(second)
        else:
          self.line(f"auto {second}=vecops::vec::zeros({atag}{{}});")
        joined = self.unique("mmla_joined")
        self.line(f"auto {joined}=vecops::vec::sve::join_k_groups({atag}{{}},{first},{second});")
        inputs.append(joined)
      for index, accumulator in enumerate(accumulators):
        left, right = inputs
        broadcast = self.unique("mmla_segment")
        source = right if rows == 8 else left
        self.line(f"auto {broadcast}=vecops::vec::sve::broadcast_segment<{index}>({atag}{{}},{source});")
        left, right = (left, broadcast) if rows == 8 else (broadcast, right)
        self.line(
          f"{accumulator}=vecops::vec::sve::mmla({tag}{{}},{atag}{{}},{atag}{{}},{left},{right},{accumulator});"
        )

    self.line(f"vecops::nint_t {group}=0;")
    if bf16:
      # Unswitch once, outside the K loop. Full-group unmasked loads avoid
      # BF16 inactive-lane blending without weakening masked-load semantics
      # for shifted packed windows near a physical panel's end.
      self.opening(f"if ({' && '.join(full_loads)}) /* MMLA full physical register fits */")
      self.opening(f"for (;({full_groups})-{group}>=2;{group}+=2) /* MMLA two complete K groups */")
      step(True, full=True)
      self.close()
      self.close()
      self.opening("else")
      self.opening(f"for (;({full_groups})-{group}>=2;{group}+=2) /* MMLA masked two complete K groups */")
      step(True)
      self.close()
      self.close()
      self.opening(f"if (({groups})-{group}==2) /* MMLA one complete and one partial K group */")
      step(True, partial=True)
      self.close()
      self.opening(f"else if ({group}<({groups})) /* MMLA odd group: no second-group read */")
      step(False, partial=True)
    else:
      self.opening(f"for (;({groups})-{group}>=2;{group}+=2) /* MMLA two complete K groups */")
      step(True)
      self.close()
      self.opening(f"if ({group}<({groups})) /* MMLA odd group: no second-group read */")
      step(False)
    self.close()
    phase = getattr(self, "matrix_phase", None)
    initial = product.args[2:]
    direct = (
      rows == 8
      and not initial
      and phase is None
      and statement.value is product
      and statement.destination.dtype == product.type.dtype
      and statement.mask is None
    )
    if direct:
      stride0, stride1 = (self.scalar(s) for s in statement.destination.strides)
      self.opening(f"if (({stride0})==2 && ({stride1})==1) /* contiguous 8x2 MMLA output */")
      self.line(
        f"vecops::vec::store({tag}{{}},{self.pointer(statement.destination, ('0', '0'))},{accumulators[0]},vecops::vec::opt::first(16));"
      )
      self.close()
      self.opening("else")
    for row in range(rows):
      value = self.unique("mmla_output_row")
      accumulator = accumulators[row // 2] if rows == 4 else accumulators[0]
      if rows == 8:
        segment = self.unique("mmla_output_segment")
        self.line(f"auto {segment}=vecops::vec::sve::broadcast_segment<{row // 2}>({tag}{{}},{accumulator});")
        accumulator = segment
      self.line(f"auto {value}=vecops::vec::sve::row_pair<{row % 2}>({tag}{{}},{accumulator});")
      coords = (str(row), "0")
      self.chunk_cache = {}
      if phase is not None:
        self.opening(f"if ({phase[0]})")
      if initial:
        seed = self.vector(initial[0], tag, product.type.shape, coords, n, 1)
        self.line(f"{value}=vecops::vec::add({tag}{{}},{value},{seed});")
      if phase is not None:
        self.close()
        self.opening("else")
        stride = phase[2].strides[1]
        extra = "" if I.constant(stride) == 1 else f",vecops::vec::opt::strided({self.scalar(stride)})"
        self.line(
          f"{value}=vecops::vec::add({tag}{{}},{value},vecops::vec::load({tag}{{}},{self.pointer(phase[2], coords)},vecops::vec::opt::first({n}){extra}));"
        )
        self.close()
      self.vector_matrix_epilogue(statement, product, tag, coords, n, value)
    if direct:
      self.close()
    self.chunk_cache = {}

  def vector_matrix_epilogue(self, statement, product, tag, coords, active, value, axis=1, *, legalized=False):
    phase = getattr(self, "matrix_phase", None)
    if phase is not None:
      self.opening(f"if ({phase[1]}) /* epilogue exactly once */")
    self.matrix_epilogue(statement, product, tag, coords, active, value, axis, fragmented=legalized)
    if phase is not None:
      self.close()
      self.opening("else")
      stride = phase[2].strides[axis]
      extra = "" if I.constant(stride) == 1 else f",vecops::vec::opt::strided({self.scalar(stride)})"
      self.line(
        f"vecops::vec::store({tag}{{}},{self.pointer(phase[2], coords)},{value},vecops::vec::opt::first({active}){extra});"
      )
      self.close()

  def matrix_epilogue(self, statement, product, tag, coords, active, accumulator, axis=1, *, fragmented=False):
    value = statement.value
    if value.op == "cast" and value.type.dtype == statement.destination.dtype:
      value = value.args[0]

    def width(e):
      return (
        max((ELEMENT_BYTES[e.type.dtype], *(width(a) for a in e.args if isinstance(a.type, I.TileType))))
        if e.op != "dot"
        else ELEMENT_BYTES[e.type.dtype]
      )

    if (
      not fragmented
      and width(value) > ELEMENT_BYTES[product.type.dtype]
      and getattr(self, "matrix_tag_lanes", {}).get(tag) != 1
    ):
      half = self.unique("epilogue_half_tag")
      self.line(f"using {half}=vecops::vec::Half<{tag}>;")
      self.tag_dtypes[half] = product.type.dtype
      halfwidth = self.vector_size(half)
      for part, op in enumerate(("lower", "upper")):
        fragment = self.unique("epilogue_fragment")
        self.line(f"auto {fragment}=vecops::vec::{op}({tag}{{}},{accumulator});")
        own = list(coords)
        own[axis] = f"({coords[axis]})+{part}*({halfwidth})"
        count = f"std::max<vecops::nint_t>(0,std::min<vecops::nint_t>({halfwidth},({active})-{part}*({halfwidth})))"
        self.opening(f"if (({count})>0)")
        self.matrix_epilogue(statement, product, half, tuple(own), count, fragment, axis, fragmented=True)
        self.close()
      return
    self.dot_registers = {id(product): (tag, accumulator)}
    self.chunk_cache = {(id(product), tag, tuple(coords), axis): accumulator}
    output_tag = self.rebound_tag(value.type.dtype, tag)
    value = self.vector(value, output_tag, statement.destination.shape, coords, active, axis)
    options, _ = self.memory_options(statement.mask, output_tag, statement.destination.shape, coords, active, axis)
    destination = statement.destination
    stride = self.scalar(destination.strides[axis])
    extra = "" if I.constant(destination.strides[axis]) == 1 else f", vecops::vec::opt::strided({stride})"
    self.line(
      f"vecops::vec::store_convert({output_tag}{{}}, {self.pointer(destination, coords)}, {value}, {options}{extra});"
    )
    self.dot_registers = {}

  def vector_gemv_matmul(self, statement, product):
    n = self.scalar(product.type.shape[1])
    self.opening(f"if (({n})==1) /* column GEMV: share B across four M rows */")
    self.vector_k_matmul(statement, product, transpose=True)
    self.close()
    self.opening("else")
    self.vector_k_matmul(statement, product)
    self.close()

  def vector_k_matmul(self, statement, product, *, transpose=False, remainder_columns=0):
    a, b = self.matrix_operands(product)
    if transpose:
      a, b = b, a
    initial = product.args[2:]
    dtype = product.type.dtype
    # Half inputs retain their rounding boundary before grouped FP32 MACs.
    # The public primitive selects FMLALB/T on ordinary SVE2, or a portable
    # widening fallback elsewhere; no ISA instruction leaks into this driver.
    narrow = ELEMENT_BYTES[a.type.dtype] == ELEMENT_BYTES[b.type.dtype] < ELEMENT_BYTES[dtype] and (
      a.type.dtype == b.type.dtype
      and a.type.dtype in {"bfloat16", "float16"}
      and dtype == "float32"
      or a.type.dtype in {"int8", "uint8"}
      and b.type.dtype in {"int8", "uint8"}
      and dtype == "int32"
    )
    atag = self.matrix_tag(a.type.dtype if narrow else dtype)
    btag = self.matrix_tag(b.type.dtype if narrow else dtype)
    tag = self.unique("dot_acc_tag")
    self.line(f"using {tag} = vecops::vec::ViewAs<{CPP_TYPES[dtype]}, {atag}>;" if narrow else f"using {tag} = {atag};")
    self.tag_dtypes[tag] = dtype
    scalar = self.matrix_tag(dtype, lanes=1)
    m, n = (self.scalar(v) for v in product.type.shape)
    if transpose:
      m, n = n, m
    k = self.scalar(a.type.shape[0 if transpose else 1])
    mi, ni, ki = (self.unique(v) for v in ("gemv_m", "gemv_n", "gemv_k"))
    width = self.vector_size(atag)
    tail = remainder_columns > 0
    column_count = remainder_columns or self.active_matmul_plan.vector_columns
    unroll = (
      max((4 + column_count - 1) // column_count, self.active_matmul_plan.k_unroll)
      if tail
      else self.active_matmul_plan.k_unroll
    )
    known_n = I.constant(b.type.shape[0 if transpose else 1])
    if not self.matrix_grouped_epilogue and not tail and isinstance(known_n, int) and known_n > 0:
      column_count = min(column_count, known_n)
      # A two-column product needs two K banks, not two independent
      # single-column remainder loops that redundantly load A.
      unroll = max(unroll, (4 + column_count - 1) // column_count)
    separate_tail = not self.matrix_grouped_epilogue
    end = f"({n})-({n})%{column_count}" if separate_tail and not tail else n
    start = "0"
    if tail:
      remainder = f"({n})%{self.active_matmul_plan.vector_columns}"
      start = f"({n})-({remainder})" if column_count == 2 else f"({n})-({remainder})%2"
      if column_count == 2:
        end = f"({n})-({remainder})%2"
    self.line("/* row/column GEMV and small-M/N family, widening vector dot */")
    self.matrix_loop("gemv.m", mi, {"M": m, "one": 1})
    self.matrix_loop("gemv.n", ni, {"N": end, "columns": column_count, "n_start": start})
    accumulators = []
    for bank in range(unroll):
      columns = []
      for column in range(column_count):
        name = self.unique("gemv_acc")
        self.line(f"auto {name} = vecops::vec::zeros({tag}{{}});")
        columns.append(name)
      accumulators.append(columns)

    def step(index, active, bank, full=False, full_columns=False):
      previous_full = getattr(self, "full_vector", False)
      self.full_vector = full
      av = self.matrix_value(a, atag, (index, mi) if transpose else (mi, index), active, 0 if transpose else 1)
      if not full:
        av = self.zero_matrix_tail(atag, av, active)
      for column, accumulator in enumerate(accumulators[bank]):
        if not full_columns:
          self.opening(f"if ({ni}+{column}<({n}))")
        bv = self.matrix_value(
          b,
          btag,
          (f"({ni}+{column})", index) if transpose else (index, f"({ni}+{column})"),
          active,
          1 if transpose else 0,
        )
        if not full:
          bv = self.zero_matrix_tail(btag, bv, active)
        if narrow:
          self.line(
            f"{accumulator} = vecops::vec::widening_dot({tag}{{}}, {atag}{{}}, {btag}{{}}, {av}, {bv}, {accumulator});"
          )
        elif dtype.startswith("float"):
          self.line(f"{accumulator} = vecops::vec::fmadd({tag}{{}}, {av}, {bv}, {accumulator});")
        else:
          self.line(
            f"{accumulator} = vecops::vec::add({tag}{{}}, {accumulator}, vecops::vec::mul({tag}{{}}, {av}, {bv}));"
          )
        if not full_columns:
          self.close()
      self.full_vector = previous_full

    self.line(f"vecops::nint_t {ki}=0;")
    bulk = self.unique("gemv_bulk_end")
    # K was validated as nonnegative before dispatch. Compute the rounded
    # limit once, rather than doing signed-128 overflow checks every K step.
    # Both full loops advance only as far as K; one masked tail cannot wrap.
    self.line(f"const vecops::nint_t {bulk}=({k})-({k})%({unroll}*({width}));")

    def k_loop(full_columns):
      self.matrix_loop("gemv.k", ki, {"K": bulk, "k_group": f"{unroll}*({width})"}, initialize=False)
      distance = self.active_matmul_plan.schedule.loop("gemv.k").schedule.distance
      advance = f"{distance * unroll}*({width})"
      future = f"({ki})+({advance})"
      guard = f"({k})-({ki})>({advance})"
      self.matrix_prefetch(a, (mi, future), guard)
      self.matrix_prefetch(b, (future, ni), guard)
      for bank in range(unroll):
        step(f"({ki})+{bank}*({width})", width, bank, True, full_columns)
      self.close()
      self.matrix_loop("gemv.k", ki, {"K": k, "k_group": width}, initialize=False, bound="full-tiles")
      step(ki, width, 0, True, full_columns)
      self.close()
      self.opening(f"if ({ki}<({k})) /* one masked GEMV tail */")
      step(ki, f"({k})-{ki}", 0, False, full_columns)
      self.close()

    if separate_tail:
      # Complete output groups and one-column remainders have independent
      # schedules: no per-column K predicates or unused accumulator banks.
      self.line(
        "/* full output group */"
        if not tail
        else "/* two-column output remainder */"
        if column_count == 2
        else "/* single-column output remainder */"
      )
      k_loop(True)
    else:
      # Group mathematical epilogues even for a partial group; computing one
      # SIMD exp beats scalar libm for each of up to three remaining results.
      self.opening(f"if (({n})-{ni}>={column_count}) /* full output group */")
      k_loop(True)
      self.close()
      self.opening("else /* one partial output group */")
      k_loop(False)
      self.close()
    if self.matrix_grouped_epilogue:
      footprint = max(
        ELEMENT_BYTES[statement.destination.dtype],
        self.matrix_expression_width(statement.value),
        *(self.matrix_expression_width(seed) for seed in initial),
      )
      self.line("#if defined(CPU_CAPABILITY_SVE)")
      self.gemv_grouped_output(
        statement, product, tag, accumulators, mi, ni, n, transpose, min(column_count, 16 // footprint)
      )
      self.line("#else")
      self.gemv_grouped_output(statement, product, tag, accumulators, mi, ni, n, transpose, column_count)
      self.line("#endif")
      self.close()
      self.close()
      return
    for column, accumulator in enumerate(accumulators[0]):
      self.opening(f"if ({ni}+{column}<({n}))")
      for bank in range(1, unroll):
        self.line(f"{accumulator}=vecops::vec::add({tag}{{}},{accumulator},{accumulators[bank][column]});")
      value = self.unique("gemv_scalar")
      self.line(f"auto {value} = vecops::vec::fill({scalar}{{}}, vecops::vec::reduce_add({tag}{{}}, {accumulator}));")
      coords = (f"({ni}+{column})", mi) if transpose else (mi, f"({ni}+{column})")
      phase = getattr(self, "matrix_phase", None)
      if phase is not None:
        self.opening(f"if ({phase[0]})")
      if initial:
        self.chunk_cache = {}
        seed = self.vector(initial[0], scalar, product.type.shape, coords, "1", 1)
        self.line(f"{value} = vecops::vec::add({scalar}{{}}, {value}, {seed});")
      if phase is not None:
        self.close()
        self.opening("else")
        self.line(
          f"{value}=vecops::vec::add({scalar}{{}},{value},vecops::vec::load({scalar}{{}},{self.pointer(phase[2], coords)}));"
        )
        self.close()
      self.vector_matrix_epilogue(statement, product, scalar, coords, "1", value)
      self.close()
    self.close()
    self.close()
    # Known narrowed groups divide N exactly and need no remainder.
    if separate_tail and not tail and column_count > 1 and column_count == self.active_matmul_plan.vector_columns:
      if column_count > 2:
        self.vector_k_matmul(statement, product, transpose=transpose, remainder_columns=2)
      self.vector_k_matmul(statement, product, transpose=transpose, remainder_columns=1)

  def gemv_grouped_output(self, statement, product, tag, accumulators, mi, ni, n, transpose, lanes):
    """Group completed independent dots before math, never scalar exp per C."""
    dtype = product.type.dtype
    result_tag = self.matrix_tag(dtype, lanes=lanes)
    axis = 0 if transpose else 1
    for start in range(0, len(accumulators[0]), lanes):
      result = self.unique("gemv_group")
      self.line(f"auto {result}=vecops::vec::zeros({result_tag}{{}});")
      for lane in range(min(lanes, len(accumulators[0]) - start)):
        column = start + lane
        self.opening(f"if ({ni}+{column}<({n}))")
        accumulator = accumulators[0][column]
        for bank in range(1, len(accumulators)):
          self.line(f"{accumulator}=vecops::vec::add({tag}{{}},{accumulator},{accumulators[bank][column]});")
        self.line(
          f"{result}=vecops::vec::set({result_tag}{{}},{result},{lane},vecops::vec::reduce_add({tag}{{}},{accumulator}));"
        )
        self.close()
      active = f"std::min<vecops::nint_t>({lanes},({n})-({ni}+{start}))"
      self.opening(f"if (({active})>0)")
      coords = (f"({ni}+{start})", mi) if transpose else (mi, f"({ni}+{start})")
      phase = getattr(self, "matrix_phase", None)
      if phase is not None:
        self.opening(f"if ({phase[0]})")
      if product.args[2:]:
        self.chunk_cache = {}
        seed = self.vector(product.args[2], result_tag, product.type.shape, coords, active, axis)
        self.line(f"{result}=vecops::vec::add({result_tag}{{}},{result},{seed});")
      if phase is not None:
        self.close()
        self.opening("else")
        stride = phase[2].strides[axis]
        extra = "" if I.constant(stride) == 1 else f",vecops::vec::opt::strided({self.scalar(stride)})"
        self.line(
          f"{result}=vecops::vec::add({result_tag}{{}},{result},vecops::vec::load({result_tag}{{}},{self.pointer(phase[2], coords)},vecops::vec::opt::first({active}){extra}));"
        )
        self.close()
      self.vector_matrix_epilogue(statement, product, result_tag, coords, active, result, axis, legalized=True)
      self.close()

  def amx_operand_tile(self, operand, side, origin, ki, spatial, k, scratch, tile, *, tile_rows=16):
    dtype = operand.type.dtype
    width = ELEMENT_BYTES[dtype]
    kr, kp = 64 // width, 4 // width
    cursor = getattr(self, "matrix_amx_cursors", {}).get(tile)
    if cursor is not None and (operand.op == "packed_load" or getattr(self, "matrix_k_full", False)):
      pointer, stride, guaranteed = cursor
      if not guaranteed:
        self.opening(f"if (({origin}) < ({spatial}))")
      self.line(f"vecops::vec::amx::load<{tile}>({pointer}, {stride});")
      if not guaranteed:
        self.close()
        self.opening("else")
        self.line(f"vecops::vec::amx::zero<{tile}>();")
        self.close()
      return
    if operand.op == "packed_load" and operand.data.format in {"amx", "auto"}:
      offset = self.packed_offset(operand.data, origin, ki)
      self.opening(f"if (({origin}) < ({spatial}))")
      self.line(f"vecops::vec::amx::load<{tile}>({self.pointer(operand.data.view)}+({offset}), 64);")
      self.close()
      self.opening("else")
      self.line(f"vecops::vec::amx::zero<{tile}>();")
      self.close()
      return
    # A raw native K-contiguous full block can go straight into a tile.
    direct = side == "A" and operand.op == "load" and not operand.args and operand.data.dtype == dtype
    if direct:
      view = operand.data
      known = I.constant(operand.type.shape[0])
      mr = tile_rows
      k_guard = "" if getattr(self, "matrix_k_full", False) else f" && ({ki})+{kr}<=({k})"
      condition = f"({origin})+{mr}<=({spatial}){k_guard} && ({self.scalar(view.strides[1])})==1"
      self.opening(f"if ({condition})")
      pointer = self.pointer(view, (origin, ki))
      if id(operand) in self.snapshots:
        saved, alias = self.snapshots[id(operand)]
        pointer = f"({alias} ? {self.pointer(saved, (origin, ki))} : {pointer})"
        stride = f"({alias} ? {self.scalar(saved.strides[0])} : {self.scalar(view.strides[0])})"
      else:
        stride = self.scalar(view.strides[0])
      self.line(f"vecops::vec::amx::load<{tile}>({pointer}, ({stride})*{width});")
      self.close()
      self.opening("else")
    if side == "A":
      tag = self.matrix_tag(dtype, lanes=kr)
      row = self.unique("amx_a_row")
      known = I.constant(operand.type.shape[1])
      full = getattr(self, "matrix_k_full", False) or isinstance(known, int) and known % kr == 0
      previous_full = getattr(self, "full_vector", False)
      self.full_vector = full
      active = str(kr) if full else f"std::min<vecops::nint_t>({kr},({k})-({ki}))"
      self.opening(f"for (vecops::nint_t {row}=0; {row}<16; ++{row})")
      value = self.unique("amx_a")
      self.line(f"auto {value} = vecops::vec::zeros({tag}{{}});")
      self.opening(f"if (({origin})+{row}<({spatial}))")
      loaded = self.matrix_value(operand, tag, (f"({origin})+{row}", ki), active, 1)
      if not full:
        loaded = self.zero_matrix_tail(tag, loaded, active)
      self.line(f"{value} = {loaded};")
      self.close()
      self.full_vector = previous_full
      self.line(f"vecops::vec::store({tag}{{}}, {scratch}+{row}*{kr}, {value});")
      self.close()
    else:
      dense = self.amx_b_dense_condition(operand)
      if dense:
        self.opening(f"if ({dense}) /* AMX B dword register packing */")

        def store_group(group, tag, value):
          self.line(f"vecops::vec::store({tag}{{}},reinterpret_cast<std::uint32_t*>({scratch})+{group * 16},{value});")

        self.amx_b_register_pack(
          operand, origin, ki, spatial, k, store_group, spatial_full=getattr(self, "matrix_n_full", False)
        )
        self.close()
        self.opening("else")
      tag = self.matrix_tag(dtype, lanes=16)
      group = self.unique("amx_b_group")
      active = f"std::max<vecops::nint_t>(0,std::min<vecops::nint_t>(16,({spatial})-({origin})))"
      self.opening(f"for (vecops::nint_t {group}=0; {group}<{kr}; {group}+={kp})")
      values = []
      for element in range(kp):
        value = self.unique("amx_b")
        self.line(f"auto {value} = vecops::vec::zeros({tag}{{}});")
        self.opening(f"if (({ki})+{group}+{element}<({k}) && ({active})>0)")
        loaded = self.matrix_value(operand, tag, (f"({ki})+{group}+{element}", origin), active, 1)
        loaded = self.zero_matrix_tail(tag, loaded, active)
        self.line(f"{value} = {loaded};")
        self.close()
        values.append(value)
      outtag, value = self.interleave_group(tag, values, dtype)
      self.line(f"vecops::vec::store({outtag}{{}}, {scratch}+{group}*16, {value});")
      self.close()
      if dense:
        self.close()
    self.line(f"vecops::vec::amx::load<{tile}>({scratch},64);")
    if direct:
      self.close()

  def amx_matmul(
    self,
    statement,
    product,
    *,
    family=None,
    skip_cache=False,
    single_m_tile=False,
    full_extents=(False, False, False),
  ):
    a, b = self.matrix_operands(product)
    initial = product.args[2:]
    dtype = product.type.dtype
    m, n = (self.scalar(v) for v in product.type.shape)
    k = self.scalar(a.type.shape[1])
    kr = 64 // ELEMENT_BYTES[a.type.dtype]
    cm, cn = (I.constant(v) for v in product.type.shape)
    if family is None and (cm is None or cn is None):
      explicit_cache = product.data.tiled and any(
        v is not None for v in (product.data.block_m, product.data.block_n, product.data.block_k)
      )
      shared_cache = (
        self.active_matmul_plan.amx_shared_cache_dispatch
        and product.data.tiled
        and cm is None
        and cn is None
        and not skip_cache
      )
      if explicit_cache or shared_cache:
        if not explicit_cache:
          # Every family used this identical whole-problem bypass decision.
          # Unswitch once, then share NC/KC/MC packing outside family dispatch
          # instead of cloning its producer and alias DAG five times.
          limit = self.active_matmul_plan.runtime_unblocked_bytes
          self.opening(
            f"if (vecops_dsl::matmul_fits({m},{n},{k},{ELEMENT_BYTES[a.type.dtype]},{ELEMENT_BYTES[b.type.dtype]},{ELEMENT_BYTES[dtype]},{limit})) /* one runtime cache decision */"
          )
          self.amx_matmul(statement, product, skip_cache=True)
          self.close()
          self.opening("else /* shared cache packing before family dispatch */")
        # Explicit cache phases share one packing wrapper. Duplicating a
        # transformed pack for every family greatly inflates C++ optimization
        # time; runtime quanta retain the requested family alignment without
        # cloning the panel producer. Shortened leaves select their own family.
        mr = f"(({m})<=16 ? 16 : ({n})<=16 ? 48 : 32)"
        nr = f"(({m})<=16 ? (({n})<=16 ? 16 : 48) : ({n})<=16 ? 16 : 32)"
        if isinstance(cm, int) and cm > 0:
          mr = f"std::min<vecops::nint_t>({mr},{((cm + 15) // 16) * 16})"
        if isinstance(cn, int) and cn > 0:
          nr = f"std::min<vecops::nint_t>({nr},{((cn + 15) // 16) * 16})"
        cache = self.matrix_cache_blocks(product, mr, nr, kr, format_="amx")
        minimum_m = 32 if isinstance(cm, int) and cm > 16 else 16
        minimum_n = 32 if isinstance(cn, int) and cn > 16 else 16
        self.blocked_matrix_matmul(
          statement, product, cache, "amx", spatial_alignment=(minimum_m, minimum_n), k_alignment=kr
        )
        if not explicit_cache:
          self.close()
        return
      # Select the register family before its cache alignment. A universal
      # 32x32 panel needlessly gives a 1x3 leaf partial 48-column groups.
      self.opening(f"if (({m})<=16) /* dynamic small-M family */")
      self.opening(f"if (({n})<=16)")
      self.amx_matmul(statement, product, family=(1, 1), skip_cache=skip_cache)
      self.close()
      self.opening("else")
      self.amx_matmul(statement, product, family=(1, 3), skip_cache=skip_cache)
      self.close()
      self.close()
      self.opening(f"else if (({n})<=16) /* dynamic small-N family */")
      self.amx_matmul(statement, product, family=(3, 1), skip_cache=skip_cache)
      self.close()
      self.opening(f"else if (({m})<=32) /* one primary M group */")
      self.amx_matmul(statement, product, family=(2, 2), skip_cache=skip_cache, single_m_tile=True)
      self.close()
      self.opening("else")
      self.amx_matmul(statement, product, family=(2, 2), skip_cache=skip_cache)
      self.close()
      return
    rows, columns = 2, 2
    if family is not None:
      rows, columns = family
    elif isinstance(cm, int) and cm <= 16:
      rows, columns = 1, min(3, max(1, (cn + 15) // 16)) if isinstance(cn, int) else 3
    elif isinstance(cn, int) and cn <= 16:
      rows, columns = min(3, max(1, (cm + 15) // 16)) if isinstance(cm, int) else 3, 1
    if isinstance(cm, int) and cm > 0:
      rows = min(rows, (cm + 15) // 16)
    if isinstance(cn, int) and cn > 0:
      columns = min(columns, (cn + 15) // 16)
    known_k = I.constant(a.type.shape[1])
    full_m, full_n, full_k = (
      proven or isinstance(known, int) and known >= 0 and known % alignment == 0
      for proven, known, alignment in zip(full_extents, (cm, cn, known_k), (rows * 16, columns * 16, kr))
    )
    acc_count = rows * columns
    single_m_tile = single_m_tile or rows == 1 or isinstance(cm, int) and cm <= rows * 16
    automatic = product.data.tiled and all(
      v is None for v in (product.data.block_m, product.data.block_n, product.data.block_k)
    )
    bypass = not skip_cache and automatic and (cm is None or cn is None)
    if bypass:
      # Dynamic broad matrices still benefit from panel packing/runtime layout
      # specialization where the same-size Const kernel can be simplified by
      # the C++ compiler. Do not use the constant cutoff indiscriminately.
      limit = self.active_matmul_plan.runtime_unblocked_bytes
      self.opening(
        f"if (vecops_dsl::matmul_fits({m},{n},{k},{ELEMENT_BYTES[a.type.dtype]},{ELEMENT_BYTES[b.type.dtype]},{ELEMENT_BYTES[dtype]},{limit})) /* bypass cache panels for small runtime products */"
      )
      self.amx_matmul(statement, product, family=(rows, columns), skip_cache=True, single_m_tile=single_m_tile)
      self.close()
      self.opening("else")
    cache = None if skip_cache else self.matrix_cache_blocks(product, rows * 16, columns * 16, kr, format_="amx")
    if cache is not None:
      self.blocked_matrix_matmul(
        statement,
        product,
        cache,
        "amx",
        spatial_alignment=(rows * 16, columns * 16),
        k_alignment=kr,
        amx_family=(rows, columns),
        single_m_tile=single_m_tile,
        amx_full_extents=(full_m, full_n, full_k),
      )
      if bypass:
        self.close()
      return
    a, b = self.automatic_matrix_packing(product, "amx", single_m_tile=single_m_tile)
    cfg, session, scope = (self.unique(v) for v in ("dot_tilecfg", "dot_session", "dot_scope"))
    self.line(f"vecops::execution::amx::TileConfiguration {cfg};")
    row_counts = (
      tuple(min(16, cm - r * 16) for r in range(rows))
      if isinstance(cm, int) and 0 < cm <= rows * 16
      else ((m,) if rows == 1 and cm is None else (16,) * rows)
    )
    col_counts = (
      tuple(min(16, cn - c * 16) for c in range(columns))
      if isinstance(cn, int) and 0 < cn <= columns * 16
      else ((n,) if columns == 1 and cn is None else (16,) * columns)
    )
    for tile in range(acc_count):
      self.line(
        f"{cfg}.rows[{tile}]={row_counts[tile // columns]}; {cfg}.column_bytes[{tile}]=({col_counts[tile % columns]})*4;"
      )
    for r in range(rows):
      self.line(f"{cfg}.rows[{acc_count + r}]={row_counts[r]}; {cfg}.column_bytes[{acc_count + r}]=64;")
    for c in range(columns):
      self.line(
        f"{cfg}.rows[{acc_count + rows + c}]=16; {cfg}.column_bytes[{acc_count + rows + c}]=({col_counts[c]})*4;"
      )
    atmp = [self.unique("amx_a_staging") for _ in range(rows)]
    btmp = [self.unique("amx_b_staging") for _ in range(columns)]
    ctmp = [self.unique("amx_c_staging") for _ in range(acc_count)]
    for name in atmp:
      self.line(f"alignas(64) {CPP_TYPES[a.type.dtype]} {name}[{16 * kr}];")
    for name in btmp:
      self.line(f"alignas(64) {CPP_TYPES[b.type.dtype]} {name}[{16 * kr}];")
    for name in ctmp:
      self.line(f"alignas(64) {CPP_TYPES[dtype]} {name}[256];")
    self.line(f"vecops::execution::ExecutionSession {session};")
    self.opening(
      f"{session}.with_resources(vecops::execution::amx::Region{{}}, [&](auto& {scope}) VECOPS_INLINE_LAMBDA_NOEXCEPT"
    )
    self.opening(f"{scope}.with_configuration({cfg}, [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT")
    mi, ni, ki = (self.unique(v) for v in ("amx_m", "amx_n", "amx_k"))
    self.line(f"/* AMX {rows}x{columns} register-tile schedule; no cache packing in dot */")
    m, n, starts, cache_loops = self.matrix_cache_region(cache, m, n)
    order = self.active_matmul_plan.amx_m_major or I.literal(False)
    if I.constant(order) is not None:
      loops = (
        ("amx.m", mi, {"M": m, "m_start": starts[0], "AMR": rows * 16}),
        ("amx.n", ni, {"N": n, "n_start": starts[1], "ANR": columns * 16}),
      )
      for role, index, bindings in loops if I.constant(order) else loops[::-1]:
        self.matrix_loop(role, index, bindings)
    else:
      # Select coordinate order once per product/panel, not a second copy of
      # the complete full/tail/fused instruction body. Only the outer spatial
      # loops contain these selects; the K hot loop remains unchanged.
      selected, outer, inner = (self.unique(v) for v in ("amx_m_major", "amx_outer", "amx_inner"))
      self.line(f"const bool {selected}={self.scalar(order)}; /* tall strip traversal */")
      self.matrix_loop(
        "amx.outer",
        outer,
        {"outer_extent": f"{selected}?({m}):({n})", "outer_step": f"{selected}?{rows * 16}:{columns * 16}"},
      )
      self.matrix_loop(
        "amx.inner",
        inner,
        {"inner_extent": f"{selected}?({n}):({m})", "inner_step": f"{selected}?{columns * 16}:{rows * 16}"},
      )
      self.line(f"const auto {mi}={selected}?{outer}:{inner}, {ni}={selected}?{inner}:{outer};")
    tag = self.matrix_tag(dtype, lanes=16)
    direct_output = statement.value is product and statement.mask is None and statement.destination.dtype == dtype
    for r in range(rows):
      for c in range(columns):
        tile = r * columns + c
        roworigin, colorigin = f"({mi}+{r * 16})", f"({ni}+{c * 16})"
        self.line(f"vecops::vec::amx::zero<{tile}>();")
        phase = getattr(self, "matrix_phase", None)
        if phase is not None:
          self.opening(f"if ({phase[0]}) /* prologue exactly once */")
        if initial:
          seed = initial[0]
          direct_seed = (
            seed.op == "load"
            and not seed.args
            and seed.data.dtype == dtype
            and len(seed.data.shape) == 2
            and (
              I.same_expr(seed.data.shape[1], product.type.shape[1])
              or I.constant(seed.data.shape[1]) == I.constant(product.type.shape[1]) == 1
            )
          )
          if direct_seed:
            guard = f"{roworigin}+{row_counts[r]}<=({m}) && {colorigin}+{col_counts[c]}<=({n}) && ({self.scalar(seed.data.strides[1])})==1"
            self.opening(f"if ({guard})")
            coordinates = self.coordinates(seed.data.shape, product.type.shape, (roworigin, colorigin))
            pointer = self.pointer(seed.data, coordinates)
            broadcast = I.constant(seed.data.shape[0]) == 1
            stride = "0" if broadcast else self.scalar(seed.data.strides[0])
            if I.constant(seed.data.shape[0]) is None and not I.same_expr(seed.data.shape[0], product.type.shape[0]):
              stride = f"(({self.scalar(seed.data.shape[0])})==1 ? 0 : ({stride}))"
            if id(seed) in self.snapshots:
              saved, alias = self.snapshots[id(seed)]
              pointer = f"({alias} ? {self.pointer(saved, coordinates)} : {pointer})"
              saved_stride = "0" if broadcast else self.scalar(saved.strides[0])
              if I.constant(seed.data.shape[0]) is None and not I.same_expr(seed.data.shape[0], product.type.shape[0]):
                saved_stride = f"(({self.scalar(seed.data.shape[0])})==1 ? 0 : ({saved_stride}))"
              stride = f"({alias} ? {saved_stride} : {stride})"
            self.line(f"vecops::vec::amx::load<{tile}>({pointer}, ({stride})*4);")
            self.close()
            self.opening("else")
          row = self.unique("amx_seed_row")
          self.opening(f"for (vecops::nint_t {row}=0; {row}<16; ++{row})")
          value = self.unique("amx_seed")
          self.line(f"auto {value}=vecops::vec::zeros({tag}{{}});")
          self.opening(f"if ({roworigin}+{row}<({m}) && {colorigin}<({n}))")
          active = f"std::min<vecops::nint_t>(16,({n})-{colorigin})"
          self.chunk_cache = {}
          loaded = self.vector(seed, tag, product.type.shape, (f"({roworigin}+{row})", colorigin), active, 1)
          self.line(f"{value}={loaded};")
          self.close()
          self.line(f"vecops::vec::store({tag}{{}}, {ctmp[tile]}+{row}*16,{value});")
          self.close()
          self.line(f"vecops::vec::amx::load<{tile}>({ctmp[tile]},64);")
          if direct_seed:
            self.close()
        if phase is not None:
          self.close()
          self.opening("else /* native accumulator continuation */")
          self.amx_partial_load(phase[2], tile, roworigin, colorigin, m, n, tag, ctmp[tile])
          self.close()

    def multiply(full):
      previous = getattr(self, "matrix_k_full", False)
      self.matrix_k_full = full
      for r in range(rows):
        self.amx_operand_tile(a, "A", f"({mi}+{r * 16})", ki, m, k, atmp[r], acc_count + r, tile_rows=row_counts[r])
      for c in range(columns):
        self.amx_operand_tile(b, "B", f"({ni}+{c * 16})", ki, n, k, btmp[c], acc_count + rows + c)
      for r in range(rows):
        for c in range(columns):
          self.line(
            f"vecops::vec::amx::dot<{CPP_TYPES[dtype]}, {CPP_TYPES[a.type.dtype]}, {CPP_TYPES[b.type.dtype]}, {r * columns + c}, {acc_count + r}, {acc_count + rows + c}>();"
          )
      self.matrix_k_full = previous

    def k_loop(spatial_full):
      previous_n = getattr(self, "matrix_n_full", False)
      previous_cursors = getattr(self, "matrix_amx_cursors", {})
      self.matrix_n_full = spatial_full
      cursors = {}
      for operand, side, count, index, spatial, guaranteed in (
        (a, "A", rows, mi, m, full_m),
        (b, "B", columns, ni, n, spatial_full),
      ):
        if known_k == 0:
          continue
        packed = operand.op == "packed_load" and operand.data.format in {"amx", "auto"}
        direct_a = (
          side == "A"
          and guaranteed
          and operand.op == "load"
          and not operand.args
          and operand.data.dtype == operand.type.dtype
          and I.constant(operand.data.strides[1]) == 1
        )
        if not packed and not direct_a:
          continue
        for group in range(count):
          origin = f"({index}+{group * 16})"
          pointer = self.unique("amx_k_cursor")
          tile = acc_count + group + (rows if side == "B" else 0)
          if packed:
            offset = self.packed_offset(operand.data, origin, "0")
            # An inactive tile's cursor starts in panel zero, not beyond the
            # allocation. Its unused K walk remains valid C++ pointer arithmetic.
            offset = offset if guaranteed else f"(({origin})<({spatial}) ? ({offset}) : 0)"
            address = f"{self.pointer(operand.data.view)}+({offset})"
            stride = "64"
          else:
            address = self.pointer(operand.data, (origin, "0"))
            stride = self.scalar(operand.data.strides[0])
            if id(operand) in self.snapshots:
              saved, alias = self.snapshots[id(operand)]
              address = f"({alias} ? {self.pointer(saved, (origin, '0'))} : {address})"
              if not I.same_expr(saved.strides[0], operand.data.strides[0]):
                stride = f"({alias} ? {self.scalar(saved.strides[0])} : {stride})"
            if I.constant(self.active_matmul_plan.expression.args[0].type.shape[1]) is None:
              # Zero-length tensors permit arbitrary spatial strides. Do not
              # even form a row-offset pointer when the cursor will be unused.
              address = f"(({k})>0 ? {address} : {self.storage_pointer(operand.data.storage)})"
            stride = f"({stride})*{ELEMENT_BYTES[operand.type.dtype]}"
          self.line(f"auto* {pointer}={address}; /* invariant AMX K cursor */")
          cursors[tile] = (pointer, stride, guaranteed)
      self.matrix_amx_cursors = cursors
      self.line(f"vecops::nint_t {ki}=0;")
      # Extents are nonnegative and this loop never advances beyond K; a
      # subtraction comparison avoids signed-128 arithmetic in every step.
      padded_k = all(operand.op == "packed_load" and operand.data.format in {"amx", "auto"} for operand in (a, b))
      self.line("/* full K: unmasked load/convert */")
      self.matrix_loop(
        "amx.k", ki, {"K": k, "AKR": kr}, initialize=False, bound="extent" if full_k or padded_k else "full-tiles"
      )
      if self.active_matmul_plan.schedule.pipeline != "off":
        distance = self.active_matmul_plan.schedule.loop("amx.k").schedule.distance
        self.opening(f"if (({k})-{ki}>{distance * kr}) /* AMX hints; no ninth tile register */")
        for tile, (pointer, stride, guaranteed) in cursors.items():
          advance = 16 * kr if tile >= acc_count + rows or a.op == "packed_load" else kr
          count = self.active_matmul_plan.schedule.prefetch_lines if guaranteed else 1
          for row in range(count):
            self.line(
              f"__builtin_prefetch(reinterpret_cast<const char*>({pointer}+{distance * advance})+({row})*({stride}),0,3);"
            )
        self.close()
      multiply(True)
      for tile, (pointer, _, _) in cursors.items():
        step = 16 * kr if tile >= acc_count + rows or a.op == "packed_load" else kr
        self.line(f"{pointer}+={step};")
      self.close()
      if not full_k and not padded_k:
        self.opening(f"if ({ki}<({k})) /* one zero-padded K tail */")
        multiply(False)
        self.close()
      self.matrix_n_full = previous_n
      self.matrix_amx_cursors = previous_cursors

    # Unswitch N-tail checks before K. GCC otherwise keeps 16 predicates per
    # B tile load in the hot full-K loop of a dynamic small-M product.
    if full_n:
      k_loop(True)
    elif b.op != "packed_load" and self.amx_b_dense_condition(b) is not None:
      self.opening(f"if (({n})-{ni}>={columns * 16}) /* full AMX N group */")
      k_loop(True)
      self.close()
      self.opening("else /* partial AMX N group */")
      k_loop(False)
      self.close()
    else:
      k_loop(False)
    for r in range(rows):
      for c in range(columns):
        tile = r * columns + c
        roworigin, colorigin = f"({mi}+{r * 16})", f"({ni}+{c * 16})"
        self.opening(f"if ({roworigin}<({m}) && {colorigin}<({n}))")
        if phase is not None:
          self.opening(f"if ({phase[1]}) /* epilogue exactly once */")
        if direct_output:
          guard = f"{roworigin}+{row_counts[r]}<=({m}) && {colorigin}+{col_counts[c]}<=({n}) && ({self.scalar(statement.destination.strides[1])})==1"
          self.opening(f"if ({guard})")
          self.line(
            f"vecops::vec::amx::store<{tile}>({self.pointer(statement.destination, (roworigin, colorigin))}, ({self.scalar(statement.destination.strides[0])})*4);"
          )
          self.close()
          self.opening("else")
        self.line(f"vecops::vec::amx::store<{tile}>({ctmp[tile]},64);")
        row = self.unique("amx_store_row")
        self.opening(f"for (vecops::nint_t {row}=0; {row}<16 && {roworigin}+{row}<({m}); ++{row})")
        value = self.unique("amx_result")
        active = f"std::min<vecops::nint_t>(16,({n})-{colorigin})"
        self.line(
          f"auto {value}=vecops::vec::load({tag}{{}}, {ctmp[tile]}+{row}*16,vecops::vec::opt::first({active}));"
        )
        self.matrix_epilogue(statement, product, tag, (f"({roworigin}+{row})", colorigin), active, value)
        self.close()
        if direct_output:
          self.close()
        if phase is not None:
          self.close()
          self.opening("else")
          self.amx_partial_store(phase[2], tile, roworigin, colorigin, m, n, tag, ctmp[tile])
          self.close()
        self.close()
    self.close()
    self.close()
    for _ in range(cache_loops):
      self.close()
    self.indent -= 1
    self.line("});")
    self.indent -= 1
    self.line("});")

  def amx_partial_load(self, view, tile, ro, co, m, n, tag, scratch):
    # Output-backed split-K accumulators inherit the caller's layout. AMX
    # memory tiles support a row byte stride, not a column element stride.
    # Use the vector staging path for nonunit/reversed columns just as for
    # spatial fringes, without changing the partial accumulator's ownership.
    stride = self.scalar(view.strides[1])
    extra = "" if I.constant(view.strides[1]) == 1 else f",vecops::vec::opt::strided({stride})"
    self.opening(f"if ({ro}+16<=({m}) && {co}+16<=({n}) && ({stride})==1)")
    self.line(f"vecops::vec::amx::load<{tile}>({self.pointer(view, (ro, co))},({self.scalar(view.strides[0])})*4);")
    self.close()
    self.opening("else")
    row = self.unique("partial_row")
    self.opening(f"for (vecops::nint_t {row}=0; {row}<16; ++{row})")
    value = self.unique("partial_value")
    self.line(f"auto {value}=vecops::vec::zeros({tag}{{}});")
    self.opening(f"if ({ro}+{row}<({m}) && {co}<({n}))")
    active = f"std::min<vecops::nint_t>(16,({n})-{co})"
    self.line(
      f"{value}=vecops::vec::load({tag}{{}},{self.pointer(view, (f'({ro}+{row})', co))},vecops::vec::opt::first({active}){extra});"
    )
    self.close()
    self.line(f"vecops::vec::store({tag}{{}},{scratch}+{row}*16,{value});")
    self.close()
    self.line(f"vecops::vec::amx::load<{tile}>({scratch},64);")
    self.close()

  def amx_partial_store(self, view, tile, ro, co, m, n, tag, scratch):
    stride = self.scalar(view.strides[1])
    extra = "" if I.constant(view.strides[1]) == 1 else f",vecops::vec::opt::strided({stride})"
    self.opening(f"if ({ro}+16<=({m}) && {co}+16<=({n}) && ({stride})==1)")
    self.line(f"vecops::vec::amx::store<{tile}>({self.pointer(view, (ro, co))},({self.scalar(view.strides[0])})*4);")
    self.close()
    self.opening("else")
    self.line(f"vecops::vec::amx::store<{tile}>({scratch},64);")
    row = self.unique("partial_row")
    self.opening(f"for (vecops::nint_t {row}=0; {row}<16 && {ro}+{row}<({m}); ++{row})")
    value = self.unique("partial_value")
    self.line(f"auto {value}=vecops::vec::load({tag}{{}},{scratch}+{row}*16);")
    active = f"std::min<vecops::nint_t>(16,({n})-{co})"
    self.line(
      f"vecops::vec::store({tag}{{}},{self.pointer(view, (f'({ro}+{row})', co))},{value},vecops::vec::opt::first({active}){extra});"
    )
    self.close()
    self.close()

  @staticmethod
  def slice_matrix_axis(extent, origin, size):
    """Keep runtime singleton broadcasts in their original coordinates.

    A cache panel does not turn a singleton input into a panel-sized load.
    Preserve both origin zero and extent one before rebinding the local DAG,
    including its captured snapshots/materializations. Known nonsingletons
    retain the original branch-free slicing representation.
    """
    constant = I.constant(extent)
    if constant == 1:
      return I.literal(0), extent
    if constant is not None:
      return origin, size
    singleton = I.binary("eq", extent, I.literal(1))
    return (
      I.Expr(origin.type, "where", (singleton, I.literal(0), origin)),
      I.Expr(size.type, "where", (singleton, I.literal(1), size)),
    )

  def slice_matrix_view(self, view, origins, shape):
    offset = view.offset
    sizes = []
    for extent, stride, origin, size in zip(view.shape, view.strides, origins, shape):
      origin, size = self.slice_matrix_axis(extent, origin, size)
      offset = I.binary("add", offset, I.binary("mul", origin, stride))
      sizes.append(size)
    return replace(view, offset=offset, shape=tuple(sizes))

  def slice_matrix_value(self, value, origins, shape, memo=None):
    if not isinstance(value.type, I.TileType):
      return value
    memo = {} if memo is None else memo
    if id(value) in memo:
      return memo[id(value)]
    original = value
    if len(value.type.shape) == 1:
      promoted_shape = (I.literal(1), value.type.shape[0])
      data = (
        replace(value.data, shape=promoted_shape, strides=(I.literal(0), value.data.strides[0]))
        if value.op == "load"
        else 1
        if value.op == "arange"
        else value.data
      )
      value = replace(value, type=I.TileType(value.type.dtype, promoted_shape, value.type.mask), data=data)
    if len(value.type.shape) != 2:
      raise TypeError("cache panel slicing requires rank-two DAGs")
    axes = tuple(
      self.slice_matrix_axis(extent, origin, size) for extent, origin, size in zip(value.type.shape, origins, shape)
    )
    sizes = tuple(size for _, size in axes)
    data = self.slice_matrix_view(value.data, origins, shape) if value.op == "load" else value.data
    if value.op == "packed_load":
      old = value.data.origins or (I.literal(0), I.literal(0))
      data = replace(value.data, origins=tuple(I.binary("add", old_axis, axis[0]) for old_axis, axis in zip(old, axes)))
    args = tuple(self.slice_matrix_value(a, origins, shape, memo) for a in value.args)
    if value.op == "arange":
      args = (I.binary("add", args[0], axes[data if data is not None else 1][0]),)
    result = replace(value, type=I.TileType(value.type.dtype, sizes, value.type.mask), data=data, args=args)
    self.retain_expression(result)
    memo[id(original)] = result
    if id(original) in self.materialized:
      view, enabled = self.materialized[id(original)]
      if len(view.shape) == 1:
        view = replace(view, shape=(I.literal(1), view.shape[0]), strides=(I.literal(0), view.strides[0]))
      self.materialized[id(result)] = (self.slice_matrix_view(view, origins, shape), enabled)
    if id(original) in self.snapshots:
      view, enabled = self.snapshots[id(original)]
      if len(view.shape) == 1:
        view = replace(view, shape=(I.literal(1), view.shape[0]), strides=(I.literal(0), view.strides[0]))
      self.snapshots[id(result)] = (self.slice_matrix_view(view, origins, shape), enabled)
    return result

  def sme_native_a_pack(self, operand, columns):
    """Conditionally pack native A, dispatching its load path before K.

    Only the library's automatic SME policy reaches this entry. Raw GEMV,
    spatially contiguous/broadcast loads and high-padding skinny matrices
    retain their existing route. The cache driver supplies NC, not full N:
    this copy is deliberately not credited with reuse across other N panels.
    """
    plan = self.active_matmul_plan
    policy = plan.sme_native_a_packing
    if (
      policy is None
      or not plan.options.tiled
      or plan.options.packing != "auto"
      or operand.op != "load"
      or operand.args
      or operand.data.dtype != operand.type.dtype
      or I.constant(operand.data.strides[1]) != 1
      or I.constant(operand.data.strides[0]) in (0, 1)
    ):
      return operand
    m_expr, k_expr = operand.type.shape
    dtype, acc_dtype = operand.type.dtype, plan.expression.type.dtype
    kp, element_bytes = max(1, 4 // ELEMENT_BYTES[dtype]), ELEMENT_BYTES[dtype]
    known = tuple(I.constant(v) for v in (m_expr, columns, k_expr))
    if any(isinstance(v, int) and v <= minimum for v, minimum in zip(known, (1, 1, 0))):
      return operand
    svl = self.environment.vector_state.sme_bytes
    # Exact tiny/small metadata incurs neither scratch nor runtime dispatch.
    if all(isinstance(v, int) for v in known) and svl is not None:
      if not policy.profitable(
        *known,
        panel=2 * svl // ELEMENT_BYTES[acc_dtype],
        lanes=svl // ELEMENT_BYTES[acc_dtype],
        k_group=kp,
        element_bytes=element_bytes,
      ):
        return operand
      if I.constant(operand.data.strides[0]) is not None:
        return self.emit_matrix_pack(operand, "A", "sme", "A")

    enabled = self.unique("sme_native_a_pack_profitable")
    self.line(f"bool {enabled}=false;")
    m, n, k = (self.scalar(v) for v in (m_expr, columns, k_expr))
    stride = self.scalar(operand.data.strides[0])
    self.opening(f"if (({m})>1 && ({n})>1 && ({k})>0 && ({stride})!=0 && ({stride})!=1)")
    names = tuple(self.unique(f"sme_pack_cost_{name}") for name in ("m", "n", "k", "panel", "lanes"))
    cm, cn, ck, panel, lanes = names
    for name, value in zip(
      names,
      (
        m,
        n,
        k,
        f"vecops_dsl::pack_panel<{CPP_TYPES[dtype]},2>()",
        f"vecops::vec::sme::streaming_lanes<{CPP_TYPES[acc_dtype]}>()",
      ),
    ):
      self.line(f"const __int128 {name}=({value});")
    logical, physical, output, pm, pk = (
      self.unique(f"sme_pack_cost_{name}") for name in ("logical", "physical", "output", "padded_m", "padded_k")
    )
    self.line(f"const __int128 {logical}={cm}*{ck}, {output}={cm}*{cn};")
    self.line(f"const __int128 {pm}=(({cm}+{panel}-1)/{panel})*{panel}, {pk}=(({ck}+{kp - 1})/{kp})*{kp};")
    self.line(f"const __int128 {physical}={pm}*{pk};")
    limit = "std::numeric_limits<vecops::nint_t>::max()"
    self.opening(
      f"if ({logical}<={limit} && {output}<={limit} && {cn}*{ck}<={limit} && {physical}<={limit}/{element_bytes})"
    )
    span, reuse, groups, saving, traffic = (
      self.unique(f"sme_pack_cost_{name}") for name in ("column_span", "reuse", "groups", "saving", "traffic")
    )
    expanded_columns = f" || ({cm}>=2*{lanes} && {cm}<4*{lanes} && {cn}>=4*{lanes})" if element_bytes == 8 else ""
    self.line(f"const __int128 {span}=({cm}<={lanes}{expanded_columns}?4:({cn}<={lanes}?1:2))*{lanes};")
    self.line(f"const __int128 {reuse}=({cn}+{span}-1)/{span};")
    self.line(f"const __int128 {groups}=(({cm}+{lanes}-1)/{lanes})*(({ck}+{kp - 1})/{kp});")
    self.line(
      f"const __int128 {saving}={groups}*{policy.gather_cycles - policy.contiguous_cycles}*{policy.packing_bytes_per_cycle};"
    )
    self.line(
      f"const __int128 {traffic}=({logical}+{physical})*{element_bytes}+{policy.packing_setup_cycles * policy.packing_bytes_per_cycle};"
    )
    self.line(
      f"{enabled}=({output}>={policy.minimum_spatial_pairs} && {output}*{ck}>={policy.minimum_work} "
      f"&& {reuse}>={policy.minimum_reuse_groups} && {physical}<={logical}*{policy.maximum_padding_ratio} "
      f"&& {reuse}-1>=({traffic}+{saving}-1)/{saving}); /* SME native A: NC-local gather versus padded pack cost */"
    )
    self.close()
    self.close()
    # A false decision must not resize a buffer or enter a packing SM owner.
    # Pack's internal alias checks are scoped to this branch; preserve the raw
    # operand's original snapshot map for the false K-body as well.
    snapshots = dict(self.snapshots)
    self.opening(f"if ({enabled}) /* pack A only when the actual panel amortizes it */")
    packed = self.emit_matrix_pack(operand, "A", "sme", "A")
    self.close()
    self.snapshots = snapshots
    self.retain_expression(packed)
    if not hasattr(self, "sme_optional_packs"):
      self.sme_optional_packs = {}
    self.sme_optional_packs[id(packed)] = operand, enabled
    return packed

  def emit_matrix_pack(self, operand, side, format_, policy, *, native_a_columns=None):
    direct = operand.op == "load" and not operand.args and operand.data.dtype == operand.type.dtype
    contiguous = I.constant(operand.data.strides[1 if side == "A" else 0]) == 1 if direct else False
    if operand.op == "packed_load" or policy == "none" or policy != "auto" and side not in policy:
      return operand
    if policy == "auto" and side == "A" and direct and contiguous:
      return (
        self.sme_native_a_pack(operand, native_a_columns)
        if format_ == "sme" and native_a_columns is not None
        else operand
      )
    storage = self.unique("panel_packed")
    count = self.packing_size(operand.type.shape, operand.type.dtype, format_, side)
    self.allocate_buffer(operand.type.dtype, storage, count)
    view = I.View(
      storage,
      operand.type.dtype,
      (I.Expr(I.ScalarType("int64"), "raw", data=count),),
      (I.literal(1),),
      I.literal(0),
      owned=True,
    )
    packed = I.PackedView(view, operand.type.shape, side, format_)
    self.pack_matrix(
      I.Pack(operand, packed), dense_plan=self.active_matmul_plan.dense_transposes[0 if side == "A" else 1]
    )
    return I.Expr(operand.type, "packed_load", data=packed)

  def amx_automatic_b_pack(self, operand, single_m_tile, spatial_alignment=(32, 32)):
    """A native contiguous B needs enough M reuse to pay for a cache-panel copy.

    Conversion/transform and non-K-contiguous operands keep packing. Use the
    original problem extent, not the shortened cache panel's M; otherwise a
    full-size product would repeatedly re-enter the marginal-reuse path.
    Incomplete spatial groups retain packing to avoid repeating masked row
    staging in each K iteration of a ragged product.
    Small unblocked products keep their resident packed-buffer schedule;
    this crossover applies only to the cache-panel driver.
    """
    if single_m_tile:
      return False
    known_m = I.constant(self.active_matmul_plan.expression.type.shape[0])
    known_n = I.constant(self.active_matmul_plan.expression.type.shape[1])
    direct = (
      operand.op == "load"
      and not operand.args
      and operand.data.dtype == operand.type.dtype
      and I.constant(operand.data.strides[0]) == 1
    )
    minimum = self.active_matmul_plan.amx_native_b_pack_min_m
    full_groups = all(
      isinstance(extent, int) and extent % alignment == 0
      for extent, alignment in zip((known_m, known_n), spatial_alignment)
    )
    return not (direct and full_groups and known_m < minimum)

  def blocked_matrix_matmul(
    self,
    statement,
    product,
    cache,
    format_,
    *,
    spatial_alignment=(4, 16),
    k_alignment=32,
    amx_family=None,
    single_m_tile=False,
    amx_full_extents=(False, False, False),
    runtime_quant_family=False,
  ):
    """NC/KC/MC traversal: bounded panels, native partial C, phased fusion."""
    a, b = self.matrix_operands(product)
    old_materialized, old_snapshots = dict(self.materialized), dict(self.snapshots)
    m, n = product.type.shape
    k = a.type.shape[1]
    bm, bn, bk = cache
    mb, nb, kb = (self.unique(v) for v in ("block_m", "block_n", "block_k"))
    mm, nn, kk = (self.unique(v) for v in ("panel_m", "panel_n", "panel_k"))

    def raw(text):
      return I.Expr(I.ScalarType("int64"), "raw", data=text)

    def panel_extent(extent, name, alignment, requested):
      known = I.constant(extent)
      # Cache budgets may be dynamic, but every block is at least its native
      # aligned extent. Retain known skinny dimensions through panel slicing
      # so the leaf can still select a small-M/N family and exact TILECFG.
      bound = (requested + alignment - 1) // alignment * alignment if requested is not None else alignment
      return extent if isinstance(known, int) and known <= bound else raw(name)

    ms = panel_extent(m, mm, spatial_alignment[0], product.data.block_m)
    ns = panel_extent(n, nn, spatial_alignment[1], product.data.block_n)
    ks = panel_extent(k, kk, k_alignment, product.data.block_k)
    self.matrix_loop("cache.n", nb, {"N": self.scalar(n), "BN": bn})
    self.line(f"const auto {nn}=std::min<vecops::nint_t>({bn},({self.scalar(n)})-{nb});")
    output_backed = statement.destination.dtype == product.type.dtype and statement.mask is None
    if output_backed:
      partial_root = self.slice_matrix_view(statement.destination, (I.literal(0), raw(nb)), (m, raw(nn)))
    else:
      partial = self.unique("matrix_partial")
      count = f"vecops_dsl::checked_shape({{{self.scalar(m)},{nn}}})"
      self.allocate_buffer(product.type.dtype, partial, count)
      partial_root = I.View(
        partial, product.type.dtype, (m, raw(nn)), (raw(nn), I.literal(1)), I.literal(0), owned=True
      )
    self.matrix_loop("cache.k", kb, {"K": self.scalar(k), "BK": bk})
    self.line(f"const auto {kk}=std::min<vecops::nint_t>({bk},({self.scalar(k)})-{kb});")
    bv = self.slice_matrix_value(b, (raw(kb), raw(nb)), (ks, ns))
    future_k = f"({kb})+({bk})"
    self.matrix_prefetch(b, (future_k, nb), f"({self.scalar(k)})-({kb})>({bk})")
    if not (
      format_ == "amx"
      and product.data.packing == "auto"
      and not self.amx_automatic_b_pack(bv, single_m_tile, spatial_alignment)
    ):
      bv = self.emit_matrix_pack(bv, "B", format_, product.data.packing)
    self.matrix_loop("cache.m", mb, {"M": self.scalar(m), "BM": bm})
    self.line(f"const auto {mm}=std::min<vecops::nint_t>({bm},({self.scalar(m)})-{mb});")
    av = self.slice_matrix_value(a, (raw(mb), raw(kb)), (ms, ks))
    self.matrix_prefetch(a, (mb, future_k), f"({self.scalar(k)})-({kb})>({bk})")
    if not runtime_quant_family:
      av = self.emit_matrix_pack(av, "A", format_, product.data.packing, native_a_columns=ns)
    initial = tuple(self.slice_matrix_value(seed, (raw(mb), raw(nb)), (ms, ns)) for seed in product.args[2:])
    local = replace(
      product,
      type=I.TileType(product.type.dtype, (ms, ns)),
      args=(av, bv, *initial),
      data=replace(product.data, tiled=False),
    )
    sink = self.slice_matrix_view(statement.destination, (raw(mb), raw(nb)), (raw(mm), raw(nn)))
    value = self.slice_matrix_value(statement.value, (raw(mb), raw(nb)), (raw(mm), raw(nn)), {id(product): local})
    mask = self.slice_matrix_value(statement.mask, (raw(mb), raw(nb)), (raw(mm), raw(nn))) if statement.mask else None
    partial_view = self.slice_matrix_view(partial_root, (raw(mb), I.literal(0)), (raw(mm), raw(nn)))
    old_phase = getattr(self, "matrix_phase", None)
    self.matrix_phase = (f"{kb}==0", f"{kb}+{kk}==({self.scalar(k)})", partial_view)
    emit = self.amx_matmul if format_ == "amx" else self.sme_matmul if format_ == "sme" else self.vector_matrix_leaf
    if format_ == "amx":
      emit(
        replace(statement, destination=sink, value=value, mask=mask),
        local,
        family=amx_family,
        single_m_tile=single_m_tile,
        full_extents=amx_full_extents,
      )
    else:
      local_statement = replace(statement, destination=sink, value=value, mask=mask)
      if runtime_quant_family:
        # The original packed-B origin and whole-problem gate were accepted
        # before entering the cache driver. BK uses the four-byte dot quantum,
        # so every sliced K origin remains aligned and the final tail is the
        # original matrix's end. Prior/correction and epilogue keep their
        # existing first/final phase guards; there is no leaf redispatch.
        self.sve_runtime_quant_rows(
          local_statement,
          local,
          self.scalar(ms),
          self.scalar(ns),
          self.scalar(ks),
          self.scalar(bv.data.origins[1]),
        )
      elif format_ != "sme" or not self.sve_packed_mmla(local_statement, local, fallback=emit):
        emit(local_statement, local)
    self.matrix_phase = old_phase
    self.close()
    self.close()
    self.close()
    self.materialized, self.snapshots = old_materialized, old_snapshots

  def matrix_cache_blocks(self, product, mr, nr, kr, *, format_=None):
    if not product.data.tiled:
      return None
    a, b = product.args[:2]
    sizes = tuple(I.constant(v) for v in (*product.type.shape, a.type.shape[1]))
    # A known small problem has no cache traversal overhead unless requested.
    options = product.data
    if (
      all(v is None for v in (options.block_m, options.block_n, options.block_k))
      and all(isinstance(v, int) for v in sizes)
      and sizes[0] * sizes[1] * ELEMENT_BYTES[product.type.dtype]
      + sizes[2] * (sizes[0] * ELEMENT_BYTES[a.type.dtype] + sizes[1] * ELEMENT_BYTES[b.type.dtype])
      <= self.active_matmul_plan.unblocked_bytes
    ):
      return None
    name = self.unique("matrix_cache")
    divisor = self.active_matmul_plan.amx_cache_b_panel_divisor if format_ == "amx" and options.block_n is None else 0
    maximum = self.active_matmul_plan.amx_cache_b_panel_max_m
    if divisor and maximum:
      known_m = I.constant(product.type.shape[0])
      divisor = (
        (divisor if known_m <= maximum else 0)
        if isinstance(known_m, int)
        else f"(({self.scalar(product.type.shape[0])})<={maximum} ? {divisor} : 0)"
      )
    self.line(
      f"const auto {name}=vecops_dsl::matmul_blocks({mr},{nr},{kr},{ELEMENT_BYTES[a.type.dtype]},{ELEMENT_BYTES[b.type.dtype]},{ELEMENT_BYTES[product.type.dtype]},{self.scalar(a.type.shape[1])},{divisor});"
    )
    values = []
    for axis, value, alignment in (("m", options.block_m, mr), ("n", options.block_n, nr), ("k", options.block_k, kr)):
      values.append(
        (
          str((value + alignment - 1) // alignment * alignment)
          if isinstance(alignment, int)
          else f"(({value}+({alignment})-1)/({alignment}))*({alignment})"
        )
        if value
        else f"{name}.{axis}"
      )
    return tuple(values)

  def matrix_cache_region(self, cache, m, n):
    if cache is None:
      return m, n, ("0", "0"), 0
    bm, bn, _ = cache
    mi, ni = (self.unique(v) for v in ("cache_m", "cache_n"))
    self.opening(f"for (vecops::nint_t {ni}=0; {ni}<({n}); {ni}+={bn})")
    self.opening(f"for (vecops::nint_t {mi}=0; {mi}<({m}); {mi}+={bm})")
    return f"std::min<vecops::nint_t>({m},{mi}+{bm})", f"std::min<vecops::nint_t>({n},{ni}+{bn})", (mi, ni), 2

  def automatic_matrix_packing(self, product, format_, *, single_m_tile=False):
    operands = list(self.matrix_operands(product))
    if not product.data.tiled or product.data.packing == "none":
      return tuple(operands)
    for index, side in enumerate(("A", "B")):
      operand = operands[index]
      if operand.op == "packed_load":
        continue
      policy = product.data.packing
      if format_ == "sme" and policy == "auto" and side == "A":
        direct = operand.op == "load" and not operand.args and operand.data.dtype == operand.type.dtype
        if direct and I.constant(operand.data.strides[1]) == 1:
          operands[index] = self.sme_native_a_pack(operand, product.type.shape[1])
          continue
      skip_b = (
        single_m_tile
        if format_ == "amx"
        else isinstance(I.constant(product.type.shape[0]), int) and I.constant(product.type.shape[0]) <= 32
      )
      if policy == "auto" and side == "B" and skip_b:
        # Insufficient cross-M reuse: raw instruction staging already does
        # the required pack; a second packed buffer adds marginal traffic.
        continue
      direct = operand.op == "load" and not operand.args and operand.data.dtype == operand.type.dtype
      need = (
        side in policy if policy != "auto" else side == "B" or not direct or I.constant(operand.data.strides[1]) != 1
      )
      if not need:
        continue
      storage = self.unique("matmul_packed")
      dtype = operand.type.dtype
      count = self.packing_size(operand.type.shape, dtype, format_, side)
      self.allocate_buffer(dtype, storage, count)
      view = I.View(
        storage, dtype, (I.Expr(I.ScalarType("int64"), "raw", data=count),), (I.literal(1),), I.literal(0), owned=True
      )
      packed = I.PackedView(view, operand.type.shape, side, format_)
      self.pack_matrix(I.Pack(operand, packed), dense_plan=self.active_matmul_plan.dense_transposes[index])
      operands[index] = I.Expr(operand.type, "packed_load", data=packed)
    return tuple(operands)

  def sme_operand_vector(self, operand, side, origin, ki, spatial, k):
    dtype = operand.type.dtype
    kp = max(1, 4 // ELEMENT_BYTES[dtype])
    tag = self.matrix_tag(dtype, power=-(kp.bit_length() - 1))
    native = self.matrix_tag(dtype)
    if operand.op == "packed_load" and operand.data.format in {"sme", "auto"}:
      value = self.unique("sme_packed")
      cursor = getattr(self, "sme_packed_cursors", {}).get((id(operand), origin))
      pointer = (
        f"{cursor[0]}+(({ki})/{kp})*({cursor[1]})"
        if cursor is not None
        else f"{self.pointer(operand.data.view)}+({self.packed_offset(operand.data, origin, ki)})"
      )
      if getattr(self, "sme_blocks_exist", False):
        self.line(f"auto {value}=vecops::vec::load({native}{{}},{pointer});")
      else:
        self.line(f"auto {value}=vecops::vec::zeros({native}{{}});")
        self.opening(f"if (({origin})<({spatial}))")
        self.line(f"{value}=vecops::vec::load({native}{{}}, {pointer});")
        self.close()
      return native, value
    spatial_axis = 0 if side == "A" else 1
    if (
      self.active_matmul_plan.sme_narrow_singleton_slices
      and dtype == "bfloat16"
      and I.constant(operand.type.shape[spatial_axis]) == 1
    ):
      # Each logical K slot contains one spatial value. Interleave only those
      # values before MOPA; do not construct wide signed-index predicates for
      # a strided gather whose logical width is statically one. The packed
      # load branch above still loads a complete native word, and the public
      # MOPA wrapper receives ordinary native-word Vec/Mask representations.
      self.line("/* proven singleton spatial input: one value per K slot */")
      tag = self.matrix_tag(dtype, lanes=1)
    spatial_full = getattr(self, "sme_spatial_full", False)
    width = self.sme_scalar_lanes(dtype, divisor=kp)
    active = width if spatial_full else self.sme_clamp_extent(f"({spatial})-({origin})", width)
    full_k = getattr(self, "matrix_k_full", False)

    def scalar_groups():
      values = []
      for group in range(kp):
        value = self.unique("sme_operand")
        guarded = not (full_k and spatial_full)
        if guarded:
          self.line(f"auto {value}=vecops::vec::zeros({tag}{{}});")
          condition = f"({active})>0" if full_k else f"({ki})+{group}<({k}) && ({active})>0"
          self.opening(f"if ({condition})")
        coords = (origin, f"({ki})+{group}") if side == "A" else (f"({ki})+{group}", origin)
        loaded = self.matrix_value(operand, tag, coords, active, 0 if side == "A" else 1)
        if not spatial_full:
          loaded = self.zero_matrix_tail(tag, loaded, active)
        self.line(f"{'auto ' if not guarded else ''}{value}={loaded};")
        if guarded:
          self.close()
        values.append(value)
      return self.interleave_group(tag, values, dtype)

    # A whole K-group already has the interleaved lane order of BFMOPA /
    # SMOPA. One u32 spatial gather replaces 2/4 narrow gathers and shuffles.
    # Keep the partial-K fallback: loading a whole word past K is incorrect.
    k_axis, spatial_axis = (1, 0) if side == "A" else (0, 1)
    grouped = (
      kp > 1
      and operand.op == "load"
      and not operand.args
      and operand.data.dtype == dtype
      and len(operand.data.shape) == 2
      and I.constant(operand.data.strides[k_axis]) == 1
    )
    if not grouped:
      return scalar_groups()
    result = self.unique("sme_grouped_operand")
    self.line(f"auto {result}=vecops::vec::zeros({native}{{}});")
    stride = self.scalar(operand.data.strides[spatial_axis])
    # The SVE memory request uses signed byte-scaled gather indices. Prove
    # their range before multiplication or address formation, including
    # negative/broadcast strides and extreme dynamic strides.
    upper = 2147483647 // ELEMENT_BYTES[dtype]
    lower = -2147483648 // ELEMENT_BYTES[dtype]
    coords = (origin, ki) if side == "A" else (ki, origin)
    pointer = self.pointer(operand.data, coords)
    if id(operand) in self.snapshots:
      saved, alias = self.snapshots[id(operand)]
      # Snapshot layout may differ from the source. Its addressing and byte
      # stride must switch together, not only the base pointer.
      pointer = f"({alias} ? {self.pointer(saved, coords)} : {pointer})"
      stride = f"({alias} ? {self.scalar(saved.strides[spatial_axis])} : ({stride}))"
    fits = f"({stride})>={lower} && ({stride})<={upper}"
    complete = "true" if full_k else f"({k})-({ki})>={kp}"
    self.opening(f"if ({complete} && {fits} && ({active})>0) /* whole K-group u32 gather */")
    words = self.matrix_tag("uint32")
    mask = "vecops::vec::opt::unmasked" if spatial_full else f"vecops::vec::opt::first({active})"
    self.line(
      f"{result}=vecops::vec::bitcast({native}{{}},{words}{{}},vecops::vec::load({words}{{}},"
      f"reinterpret_cast<const std::uint32_t*>({pointer}),{mask},"
      f"vecops::vec::opt::strided(({stride})*{ELEMENT_BYTES[dtype]},vecops::vec::opt::scale<1>)));"
    )
    self.close()
    self.opening("else")
    _, value = scalar_groups()
    self.line(f"{result}=static_cast<vecops::vec::Vec<{native}>>({value});")
    self.close()
    return native, result

  @staticmethod
  def sme_direct_axis(view, dtype):
    """Return the contiguous ZA slice axis; conversions need Z registers."""
    if len(view.shape) != 2 or view.dtype != dtype:
      return None
    # A single-column tensor can be contiguous along both axes. Use one
    # vertical ZA slice rather than one horizontal slice per logical row.
    if I.constant(view.shape[1]) == 1 and I.constant(view.strides[0]) == 1:
      return 0
    if I.constant(view.strides[1]) == 1:
      return 1
    if I.constant(view.strides[0]) == 1:
      return 0
    return None

  def sme_slice_pointer(self, view, axis, origin, index, rows, columns, *, seed=None):
    """Hoist ZA-slice address invariants past the memory-clobber barrier.

    Inactive catalog blocks must not form an out-of-bounds tile pointer.
    Snapshot selection preserves its own row stride rather than borrowing
    the original layout's stride. A snapshot with another contiguous axis
    retains the existing per-slice route; it is not a contiguous cursor.
    """
    saved = self.snapshots.get(id(seed)) if seed is not None else None
    if saved is not None and I.constant(saved[0].strides[axis]) != 1:
      return None
    pointer = self.pointer(view, origin)
    inactive = self.pointer(view)
    stride = self.scalar(view.strides[1 - axis])
    if saved is not None:
      snapshot, alias = saved
      pointer = f"({alias} ? {self.pointer(snapshot, origin)} : {pointer})"
      inactive = f"({alias} ? {self.pointer(snapshot)} : {inactive})"
      stride = f"({alias} ? {self.scalar(snapshot.strides[1 - axis])} : {stride})"
    base, step = self.unique("sme_slice_base"), self.unique("sme_slice_stride")
    self.line(f"auto* const {base}=(({rows})>0 && ({columns})>0 ? {pointer} : {inactive});")
    self.line(f"const vecops::nint_t {step}={stride};")
    return f"{base}+({index})*{step}"

  def sme_memory_slice(self, operation, tile, index, predicate, tag, dtype, view, coords, *, seed=None, pointer=None):
    """Emit a bit-preserving ZA-memory instruction, not a Z-register move."""
    bits = "std::uint64_t" if ELEMENT_BYTES[dtype] == 8 else "std::uint32_t"
    bitstag = f"vecops::vec::ScalableTag<{bits},0>"
    if pointer is None:
      pointer = self.pointer(view, coords)
      if seed is not None and id(seed) in self.snapshots:
        saved, alias = self.snapshots[id(seed)]
        pointer = f"({alias} ? {self.pointer(saved, coords)} : {pointer})"
    const = "const " if operation.startswith("load") else ""
    self.line(
      f"vecops::vec::sme::{operation}<{tile}>(static_cast<std::uint32_t>({index}),"
      f"static_cast<vecops::vec::Mask<{bitstag}>>({predicate}),reinterpret_cast<{const}{bits}*>({pointer}));"
    )

  def sme_slice_value_tag(self, product, axis, tag):
    """Keep ISA tags native; narrow only proven one-element memory slices.

    An arbitrary short/tail slice is not a singleton proof. In particular a
    vertical column of an Mx1 result still contains M elements. FixedTag<T,1>
    keeps one physical SVE word but avoids wide-index predicate tuples for
    scalar-width strided accesses, without scalarizing the ZA instruction.
    """
    if not self.active_matmul_plan.sme_narrow_singleton_slices or I.constant(product.type.shape[axis]) != 1:
      return tag
    self.line("/* proven singleton ZA slice: scalar-width vector memory/epilogue */")
    return self.matrix_tag(product.type.dtype, lanes=1)

  def sme_seed_tile(self, seed, product, tile, ro, co, rows, columns, tag):
    """Initialize ZA directly when possible, retaining fused prologue DAGs.

    Row broadcasts are a single outer product / ADDHA, not one reload per
    output row. Raw column-contiguous seeds use LD1 into vertical ZA slices.
    """
    dtype = product.type.dtype
    raw = seed.op == "load" and not seed.args and seed.data.dtype == dtype
    broadcast = raw and (
      len(seed.data.shape) == 1
      or len(seed.data.shape) == 2
      and (I.constant(seed.data.shape[0]) == 1 or I.constant(seed.data.strides[0]) == 0)
    )
    if broadcast:
      self.opening(f"if (({rows})>0 && ({columns})>0) /* broadcast C seed: one ZA instruction */")
      pg_rows, pg_columns = (self.unique(v) for v in ("sme_seed_rows", "sme_seed_columns"))
      self.line(f"auto {pg_rows}=vecops::vec::mwhilelt({tag}{{}},0,{rows});")
      self.line(f"auto {pg_columns}=vecops::vec::mwhilelt({tag}{{}},0,{columns});")
      self.chunk_cache = {}
      value_tag = self.sme_slice_value_tag(product, 1, tag)
      value = self.vector(seed, value_tag, product.type.shape, (ro, co), columns, 1)
      if dtype in {"int32", "uint32"}:
        self.line(f"vecops::vec::sme::addha<{tile},{CPP_TYPES[dtype]}>({pg_rows},{pg_columns},{value});")
      else:
        self.line(
          f"vecops::vec::sme::mopa<{tile},{CPP_TYPES[dtype]},{CPP_TYPES[dtype]},{CPP_TYPES[dtype]}>("
          f"{pg_rows},{pg_columns},vecops::vec::fill({tag}{{}},{CPP_TYPES[dtype]}{{1}}),{value});"
        )
      self.close()
      return
    same_shape = raw and all(I.same_expr(x, y) for x, y in zip(seed.data.shape, product.type.shape))
    axis = self.sme_direct_axis(seed.data, dtype) if same_shape else None
    axis = 1 if axis is None else axis
    active, extent = (columns, rows) if axis == 1 else (rows, columns)
    pg, index = (self.unique(v) for v in ("sme_seed_mask", "sme_seed_slice"))
    self.line(f"auto {pg}=vecops::vec::mwhilelt({tag}{{}},0,{active});")
    direct = same_shape and self.sme_direct_axis(seed.data, dtype) is not None
    pointer = self.sme_slice_pointer(seed.data, axis, (ro, co), index, rows, columns, seed=seed) if direct else None
    self.opening(f"for (vecops::nint_t {index}=0;{index}<({extent}) && ({active})>0;++{index})")
    coords = (f"({ro}+{index})", co) if axis == 1 else (ro, f"({co}+{index})")
    if direct:
      self.sme_memory_slice(
        "load_hor" if axis == 1 else "load_ver",
        tile,
        index,
        pg,
        tag,
        dtype,
        seed.data,
        coords,
        seed=seed,
        pointer=pointer,
      )
    else:
      self.chunk_cache = {}
      value_tag = self.sme_slice_value_tag(product, axis, tag)
      value = self.vector(seed, value_tag, product.type.shape, coords, active, axis)
      self.line(f"vecops::vec::sme::write_hor<{tile}>(static_cast<std::uint32_t>({index}),{pg},{value});")
    self.close()

  def sme_store_tile(self, statement, product, tile, ro, co, rows, columns, tag, *, _axis=None, _single_column=False):
    """Drain native C/partial accumulators directly; vectorize only transforms."""
    dtype = product.type.dtype
    destination = statement.destination
    value = statement.value
    if value.op == "cast" and value.type.dtype == destination.dtype:
      value = value.args[0]
    direct = value is product and statement.mask is None and destination.dtype == dtype
    axis = self.sme_direct_axis(destination, dtype) if direct else None
    if _axis is None and direct and axis == 1 and I.constant(destination.strides[0]) is None:
      # The row-major leading stride may only become one at invocation.
      # Keep this outside K and retain the original route for negative or
      # non-unit strides, multi-column tiles, transforms and masked stores.
      self.opening(
        f"if (({self.scalar(destination.strides[0])})==1 && ({columns})==1 && ({rows})>1) "
        "/* runtime contiguous column: one vertical ZA store */"
      )
      self.sme_store_tile(statement, product, tile, ro, co, rows, columns, tag, _axis=0, _single_column=True)
      self.close()
      self.opening("else /* original ZA slice orientation */")
      self.sme_store_tile(statement, product, tile, ro, co, rows, columns, tag, _axis=1)
      self.close()
      return
    if _axis is not None:
      axis = _axis
    # Transformed column-contiguous C also uses vertical slices: no scatter.
    if axis is None:
      axis = 0 if I.constant(destination.strides[0]) == 1 else 1
    active, extent = (columns, rows) if axis == 1 else (rows, columns)
    pg, index = (self.unique(v) for v in ("sme_store_mask", "sme_store_slice"))
    self.line(f"auto {pg}=vecops::vec::mwhilelt({tag}{{}},0,{active});")
    phase = getattr(self, "matrix_phase", None)
    pointer = (
      self.sme_slice_pointer(destination, axis, (ro, co), index, rows, columns)
      if direct and self.sme_direct_axis(destination, dtype) is not None
      else None
    )
    partial_axis = self.sme_direct_axis(phase[2], dtype) if phase is not None else None
    partial_pointer = (
      self.sme_slice_pointer(phase[2], axis, (ro, co), index, rows, columns) if partial_axis == axis else None
    )
    if _single_column:
      # The dispatch proved extent == 1 and active > 1. Preserve that proof
      # explicitly: the C++ optimizer need not rediscover it across ZA inline
      # assembly memory barriers and repeated clamp expressions.
      self.opening("/* exactly one active vertical slice */")
      self.line(f"constexpr vecops::nint_t {index}=0;")
    else:
      self.opening(f"for (vecops::nint_t {index}=0;{index}<({extent}) && ({active})>0;++{index})")
    coords = (f"({ro}+{index})", co) if axis == 1 else (ro, f"({co}+{index})")
    if phase is not None:
      self.opening(f"if ({phase[1]}) /* epilogue exactly once */")
    if direct and self.sme_direct_axis(destination, dtype) is not None:
      self.sme_memory_slice(
        "store_hor" if axis == 1 else "store_ver", tile, index, pg, tag, dtype, destination, coords, pointer=pointer
      )
    else:
      result = self.unique("sme_result")
      self.line(
        f"auto {result}=vecops::vec::sme::read_{'hor' if axis == 1 else 'ver'}<{tile}>("
        f"{tag}{{}},static_cast<std::uint32_t>({index}),{pg});"
      )
      value_tag = self.sme_slice_value_tag(product, axis, tag)
      self.matrix_epilogue(statement, product, value_tag, coords, active, result, axis)
    if phase is not None:
      self.close()
      self.opening("else /* native partial C: direct ZA store */")
      if partial_axis == axis:
        self.sme_memory_slice(
          "store_hor" if axis == 1 else "store_ver",
          tile,
          index,
          pg,
          tag,
          dtype,
          phase[2],
          coords,
          pointer=partial_pointer,
        )
      else:
        result = self.unique("sme_partial_result")
        self.line(
          f"auto {result}=vecops::vec::sme::read_{'hor' if axis == 1 else 'ver'}<{tile}>("
          f"{tag}{{}},static_cast<std::uint32_t>({index}),{pg});"
        )
        stride = phase[2].strides[axis]
        extra = "" if I.constant(stride) == 1 else f",vecops::vec::opt::strided({self.scalar(stride)})"
        value_tag = self.sme_slice_value_tag(product, axis, tag)
        self.line(
          f"vecops::vec::store({value_tag}{{}},{self.pointer(phase[2], coords)},{result},vecops::vec::opt::first({active}){extra});"
        )
      self.close()
    self.close()

  def sme_scalar_lanes(self, dtype, *, divisor=1):
    """Use captured SVL, never ordinary VL or a cost-profile vector width.

    Native compilation keys and the adapter entry guard retain this thread's
    independent SVL. Literal integer arithmetic also avoids leaving metadata
    Const division calls in a large manual streaming owner. Cross lowering
    without a vector-state snapshot still uses the public runtime SVL reader.
    """
    svl = self.environment.vector_state.sme_bytes
    if svl is not None:
      return str(svl // ELEMENT_BYTES[dtype] // divisor)
    lanes = f"vecops::vec::sme::streaming_lanes<{CPP_TYPES[dtype]}>()"
    return lanes if divisor == 1 else f"({lanes})/{divisor}"

  @staticmethod
  def sme_clamp_extent(value, upper):
    """Force-inline scalar extents: ordinary std min/max may remain calls."""
    return f"vecops::vec::sme::clamp_value<vecops::nint_t>({value},0,{upper})"

  @contextmanager
  def sme_loader_metadata(self, operands):
    """Keep immutable producer addressing in this selected owner's SSA.

    The manual state trampoline exposes its captures by reference. ZA's
    memory clobbers can therefore reload an otherwise immutable ABI stride,
    alias predicate or Buffer::data() in every K group. Copy only existing
    ABI descriptor leaves and producer buffer bases after entering the owner;
    retain the original load identities, masks and historical read captures.

    No offset pointer is formed here. An inactive conditional Buffer may
    supply a null data() base, but an unknown/optional storage expression and
    arbitrary raw/local/state expressions retain their original evaluation
    scope. Every map is restored even if a nested emitter raises.
    """
    old_metadata, old_pointers = self.parameter_metadata, self.parameter_pointers
    old_snapshots, old_materialized = self.snapshots, self.materialized
    old_optional = getattr(self, "sme_optional_packs", None)
    old_cache = self.chunk_cache
    self.parameter_metadata, self.parameter_pointers = dict(old_metadata), dict(old_pointers)
    self.snapshots, self.materialized = dict(old_snapshots), dict(old_materialized)
    if old_optional is not None:
      self.sme_optional_packs = dict(old_optional)
    self.chunk_cache = {}
    try:
      views, metadata, expressions = {}, [], {}
      roots = list(operands)
      for operand in operands:
        optional = (old_optional or {}).get(id(operand))
        if optional is not None:
          roots.append(optional[0])
      for root in roots:
        for value in walk_unique(root):
          expressions[id(value)] = value
          if value.op == "load":
            views[id(value.data)] = value.data
          elif value.op == "packed_load":
            views[id(value.data.view)] = value.data.view
            metadata.extend((*value.data.shape, *(value.data.origins or ())))
          for mapping in (old_snapshots, old_materialized):
            saved = mapping.get(id(value))
            if saved is not None:
              views[id(saved[0])] = saved[0]
      self.line("/* selected SME owner: immutable producer addressing, not tensor contents */")
      for view in views.values():
        metadata.extend((view.offset, *view.shape, *view.strides))
        storage = view.storage
        # ABI entry pointers and named owned buffers have a pure data() base.
        # Never hoist an optional.value() or another unknown storage accessor.
        if storage not in old_pointers and not (view.owned and storage.isidentifier()):
          continue
        if self.parameter_pointers.get(storage) != old_pointers.get(storage):
          continue  # Several views can share one storage without another read.
        local = self.unique("sme_loader_base")
        self.line(f"[[maybe_unused]] auto* const {local}={self.storage_pointer(storage)};")
        self.parameter_pointers[storage] = local

      seen = set()

      def descriptor(value):
        if id(value) in seen:
          return
        seen.add(id(value))
        if value.op == "symbol":
          bound = self.ir.bindings.get(value.data)
          if bound is not None:
            descriptor(bound)
        elif value.op == "raw" and value.data in old_metadata:
          # The exact ABI mapping, not a spelling/opcode heuristic, proves
          # this is immutable metadata. Do not freeze an induction/state raw.
          if self.parameter_metadata[value.data] == old_metadata[value.data]:
            local = self.unique("sme_loader_metadata")
            self.line(f"[[maybe_unused]] const vecops::nint_t {local}={old_metadata[value.data]};")
            self.parameter_metadata[value.data] = local
        for argument in value.args:
          descriptor(argument)

      for value in metadata:
        descriptor(value)

      predicates = {}

      def predicate(condition):
        # These are already-decided generated bool locals. More general
        # expressions, including guarded optional access, are not speculated.
        if not isinstance(condition, str) or not condition.isidentifier() or condition in {"true", "false"}:
          return condition
        if condition not in predicates:
          local = self.unique("sme_loader_enabled")
          self.line(f"const bool {local}={condition};")
          predicates[condition] = local
        return predicates[condition]

      for identity in expressions:
        for mapping in (self.snapshots, self.materialized):
          entry = mapping.get(identity)
          if entry is not None:
            mapping[identity] = (entry[0], predicate(entry[1]))
      if old_optional is not None:
        for operand in operands:
          entry = old_optional.get(id(operand))
          if entry is not None:
            self.sme_optional_packs[id(operand)] = (entry[0], predicate(entry[1]))
      yield
    finally:
      self.parameter_metadata, self.parameter_pointers = old_metadata, old_pointers
      self.snapshots, self.materialized = old_snapshots, old_materialized
      if old_optional is not None:
        self.sme_optional_packs = old_optional
      self.chunk_cache = old_cache

  def sme_family_owner(self, statement, product, a, b, session, extents, nr_m, nr_n, *, exact_blocks=False):
    """One selected geometry per manual SM owner, not one giant family tree.

    The outer geometry branch executes before entering SM. Its callback must
    inline into the state trampoline, but the trampoline itself remains the
    public state owner's ABI boundary. Splitting the tree prevents unrelated
    gather/tail families from exhausting the optimizer's local inline budget.
    """
    scope = self.unique("sme_dot_scope")
    self.opening_hardware_owner("sme", session, f"[&](auto& {scope})")
    previous_scope, self.hardware_scope = self.hardware_scope, ("sme", scope)
    # Each outlined state trampoline reads escaped captures exactly once.
    # Keep immutable, callback-owned SSA extents throughout its native loops.
    held_extents = tuple(self.unique(f"sme_extent_{axis}_value") for axis in ("m", "n", "k"))
    for name, value in zip(held_extents, extents):
      self.line(f"const vecops::nint_t {name}={value};")
    tag = self.matrix_tag(product.type.dtype)
    lanes = self.unique("sme_matrix_lanes")
    self.line(f"const vecops::nint_t {lanes}={self.sme_scalar_lanes(product.type.dtype)};")
    try:
      with self.sme_loader_metadata((a, b)):
        self.sme_matrix_family(
          statement, product, a, b, tag, lanes, nr_m, nr_n, held_extents, exact_blocks=exact_blocks
        )
    finally:
      self.hardware_scope = previous_scope
    self.indent -= 1
    self.line("});")

  def sme_packed_ab_cache_bypass(self, product, a, b, m, n, k, kp):
    """Select a resident full-K packed traversal outside any streaming owner.

    This is intentionally narrow, not a promise that every packed
    product should skip caches. Unsupported dtype/format and explicit block
    requests emit no extra code. Quotients avoid nint overflow even at INT64_MAX.
    The existing leaf owns accumulator, fusion, alias and signed-layout rules.
    """
    plan = self.active_matmul_plan
    if not plan.sme_packed_ab_cache_bypass or product.type.dtype != "float32":
      return None
    if not all(
      value.op == "packed_load"
      and value.type.dtype == "bfloat16"
      and value.data.side == side
      and value.data.format in {"sme", "auto"}
      for value, side in ((a, "A"), (b, "B"))
    ):
      return None
    policy = plan.sme_packed_ab_traversal
    known = tuple(I.constant(value) for value in (*product.type.shape, a.type.shape[1]))
    captured = self.environment.vector_state.sme_bytes
    if (
      captured is not None
      and all(type(value) is int for value in known)
      and not policy.shape_candidate(*known, lanes=captured // 4, k_group=kp)
    ):
      return None
    lanes = self.unique("sme_packed_ab_lanes")
    groups = self.unique("sme_packed_ab_k_groups")
    selected = self.unique("sme_packed_ab_whole_problem")
    self.line(f"const vecops::nint_t {lanes}={self.sme_scalar_lanes('float32')};")
    self.line(f"bool {selected}=false;")
    self.opening(
      f"if (({m})>0 && ({m})<={policy.maximum_row_panels}*{lanes}"
      f" && ({n})/{lanes}>={policy.minimum_column_panels} && ({k})/{kp}>={policy.minimum_k_groups})"
      " /* SME packed-AB whole-problem geometry */"
    )
    self.line(f"const vecops::nint_t {groups}=({k})/{kp}+(({k})%{kp}!=0);")
    self.line(
      f"{selected}={groups}<=vecops::platform::cache_info().l2_bytes"
      f"/{policy.private_cache_divisor}/(4*{lanes}*{ELEMENT_BYTES[a.type.dtype]})/{kp};"
      " /* complete padded two-by-two micro-panel stays in private L2 */"
    )
    self.close()
    return selected

  def sme_matmul(self, statement, product, *, skip_cache=False):
    a, b = self.matrix_operands(product)
    dtype = product.type.dtype
    m, n = (self.scalar(v) for v in product.type.shape)
    k = self.scalar(a.type.shape[1])
    kp = max(1, 4 // ELEMENT_BYTES[a.type.dtype])
    span = f"2*({self.sme_scalar_lanes(dtype)})"
    automatic = product.data.tiled and all(
      value is None for value in (product.data.block_m, product.data.block_n, product.data.block_k)
    )
    unknown_extent = any(I.constant(value) is None for value in (*product.type.shape, a.type.shape[1]))
    packed_bypass = (
      self.sme_packed_ab_cache_bypass(product, a, b, m, n, k, kp) if not skip_cache and automatic else None
    )
    bypass = not skip_cache and automatic and (unknown_extent or packed_bypass is not None)
    if bypass:
      # Dynamic metadata is not evidence that the actual working set is large.
      # The shared quotient-based budget predicate cannot overflow; this test
      # also precedes cache_info(), the panel loops and partial-C phase setup.
      # Explicit cache tiles are never bypassed, and the leaf retains its own
      # packing/fusion policies just like the known-small Const path.
      limit = self.active_matmul_plan.runtime_unblocked_bytes
      small = (
        f"vecops_dsl::matmul_fits({m},{n},{k},{ELEMENT_BYTES[a.type.dtype]},{ELEMENT_BYTES[b.type.dtype]},{ELEMENT_BYTES[dtype]},{limit})"
        if unknown_extent
        else "false"
      )
      condition = f"{packed_bypass} || {small}" if packed_bypass else small
      self.opening(f"if ({condition}) /* bypass cache panels for small runtime SME products */")
      self.sme_matmul(statement, product, skip_cache=True)
      self.close()
      self.opening("else")
    cache = None if skip_cache else self.matrix_cache_blocks(product, span, span, kp)
    if cache is not None:
      # Cache BK is rounded to this operand's actual MOPA K quantum (1 for
      # FP32/FP64, 2 for FP16/BF16, 4 for bytes). The generic driver's default
      # 32 belongs to other families: using it to preserve Const K can claim
      # a whole K=5 panel even though an explicit BK=3 produces only 3/4 lanes.
      self.blocked_matrix_matmul(statement, product, cache, "sme", k_alignment=kp)
      if bypass:
        self.close()
      return
    a, b = self.automatic_matrix_packing(product, "sme")
    # Capture validated immutable metadata before any noexcept SM callback.
    # Geometry selection stays ordinary scalar code, with SVL-derived lanes
    # rather than CNT from the current ordinary SVE execution domain.
    extents = tuple(self.unique(f"sme_extent_{axis}") for axis in ("m", "n", "k"))
    for name, value in zip(extents, (m, n, k)):
      self.line(f"const vecops::nint_t {name}=({value});")
    m, n, k = extents
    session = self.unique("sme_dot_session")
    self.line(f"vecops::execution::ExecutionSession {session};")
    lanes = self.unique("sme_matrix_lanes")
    self.line(f"const vecops::nint_t {lanes}={self.sme_scalar_lanes(dtype)};")
    # ZA has four f32/i32 tiles or eight f64 tiles. A fixed 2x2 schedule
    # wastes half/all its outer products for narrow-M/N problems; select a
    # rectangular family outside K. All registers remain named variables.
    self.opening(f"if (({m})<={lanes}) /* narrow M: 1x4 ZA family */")
    for blocks in (1, 2, 3):
      self.opening(f"{'if' if blocks == 1 else 'else if'} (({n})<={blocks}*{lanes}) /* exact 1x{blocks} ZA blocks */")
      self.sme_family_owner(statement, product, a, b, session, extents, 1, blocks, exact_blocks=True)
      self.close()
    self.opening("else")
    self.sme_family_owner(statement, product, a, b, session, extents, 1, 4)
    self.close()
    self.close()
    self.opening(f"else if (({n})<={lanes}) /* narrow N: 4x1 ZA family */")
    for blocks in (2, 3):
      self.opening(f"{'if' if blocks == 2 else 'else if'} (({m})<={blocks}*{lanes}) /* exact {blocks}x1 ZA blocks */")
      self.sme_family_owner(statement, product, a, b, session, extents, blocks, 1, exact_blocks=True)
      self.close()
    self.opening("else")
    self.sme_family_owner(statement, product, a, b, session, extents, 4, 1)
    self.close()
    self.close()
    self.opening(f"else if (({m})<=2*{lanes} && ({n})<=2*{lanes}) /* exact 2x2 ZA blocks */")
    self.sme_family_owner(statement, product, a, b, session, extents, 2, 2, exact_blocks=True)
    self.close()
    if dtype == "float64" and not all(
      e.op == "packed_load" and id(e) not in getattr(self, "sme_optional_packs", {}) for e in (a, b)
    ):
      self.opening(f"else if (({m})>=4*{lanes} && ({n})>=2*{lanes}) /* expanded ZA64 catalog */")
      self.sme_family_owner(statement, product, a, b, session, extents, 4, 2)
      self.close()
      self.opening(f"else if (({m})>=2*{lanes} && ({n})>=4*{lanes}) /* expanded ZA64 catalog */")
      self.sme_family_owner(statement, product, a, b, session, extents, 2, 4)
      self.close()
    self.opening("else /* balanced 2x2 ZA family */")
    self.sme_family_owner(statement, product, a, b, session, extents, 2, 2)
    self.close()
    if bypass:
      self.close()

  def sme_matrix_family(self, statement, product, a, b, tag, lanes, nr_m, nr_n, extents, *, exact_blocks=False):
    """One rectangular ZA leaf, optionally proving all its blocks exist.

    Exact leaves are selected before K only for a single macro-tile whose
    block count is known. Every non-last block has a full static slice extent;
    the last block retains its dynamic tail mask. This does not assume the
    tail rows/columns themselves are full or change accumulation order.
    """
    initial = product.args[2:]
    dtype = product.type.dtype
    m, n, k = extents
    optional_a = getattr(self, "sme_optional_packs", {}).get(id(a))
    kp = max(1, 4 // ELEMENT_BYTES[a.type.dtype])
    mi, ni, ki = (self.unique(v) for v in ("sme_m", "sme_n", "sme_k"))
    self.line(f"/* SME {nr_m}x{nr_n} outer-product schedule; all vectors are born inside SM */")
    self.matrix_loop("sme.m", mi, {"M": m, "SMR": f"{nr_m}*{lanes}"})
    self.matrix_loop("sme.n", ni, {"N": n, "SNR": f"{nr_n}*{lanes}"})
    self.line("vecops::vec::sme::zero_za();")

    def active_extent(total, origin, block, blocks):
      if exact_blocks and block + 1 < blocks:
        return lanes
      return self.sme_clamp_extent(f"({total})-{origin}", lanes)

    for r in range(nr_m):
      for c in range(nr_n):
        tile = r * nr_n + c
        ro, co = f"({mi}+{r}*{lanes})", f"({ni}+{c}*{lanes})"
        rows = active_extent(m, ro, r, nr_m)
        columns = active_extent(n, co, c, nr_n)
        phase = getattr(self, "matrix_phase", None)
        if phase is not None:
          self.opening(f"if ({phase[0]}) /* prologue exactly once */")
        if initial:
          self.sme_seed_tile(initial[0], product, tile, ro, co, rows, columns, tag)
        if phase is not None:
          self.close()
          self.opening("else /* native ZA accumulator continuation */")
          partial = I.Expr(product.type, "load", data=phase[2])
          self.sme_seed_tile(partial, product, tile, ro, co, rows, columns, tag)
          self.close()

    old_cursors = getattr(self, "sme_packed_cursors", {})
    self.sme_packed_cursors = {}
    for operand, side, index, blocks, spatial in ((a, "A", mi, nr_m, m), (b, "B", ni, nr_n, n)):
      if operand.op != "packed_load" or operand.data.format not in {"sme", "auto"}:
        continue
      step = self.unique("sme_packed_k_step")
      self.line(
        f"const vecops::nint_t {step}=vecops_dsl::pack_panel<{CPP_TYPES[operand.type.dtype]},{FORMATS[operand.data.format]}>()*{kp};"
      )
      for block in range(blocks):
        origin = f"({index}+{block}*{lanes})"
        cursor = self.unique("sme_packed_k_cursor")
        offset = self.packed_offset(operand.data, origin, "0")
        base = self.pointer(operand.data.view)
        self.line(
          f"const {CPP_TYPES[operand.type.dtype]}* {cursor}=(({k})>0 && ({origin})<({spatial})"
          f"{' && (' + optional_a[1] + ')' if operand is a and optional_a is not None else ''} ? {base}+({offset}) : nullptr);"
        )
        self.sme_packed_cursors[id(operand), origin] = cursor, step

    def load_sme(index):
      previous_spatial, previous_vector = self.sme_spatial_full, self.full_vector

      def load(operand, side, base, block, blocks, spatial):
        prefix = exact_blocks and block + 1 < blocks
        self.sme_spatial_full = previous_spatial or prefix
        self.full_vector = previous_vector or prefix
        return self.sme_operand_vector(operand, side, f"({base}+{block}*{lanes})", index, spatial, k)

      av = [load(a, "A", mi, r, nr_m, m) for r in range(nr_m)]
      bv = [load(b, "B", ni, c, nr_n, n) for c in range(nr_n)]
      self.sme_spatial_full, self.full_vector = previous_spatial, previous_vector
      return (*av, *bv)

    def compute_sme(operands):
      for r in range(nr_m):
        for c in range(nr_n):
          atag, aval = operands[r]
          btag, bval = operands[nr_m + c]
          guarded = not getattr(self, "sme_blocks_exist", False) and (nr_m > 1 or nr_n > 1)
          if guarded:
            # Guard only block existence, not individual SIMD lanes. Tail
            # loads zero-fill, but empty blocks should not issue an MOPA.
            self.opening(f"if (({mi}+{r}*{lanes})<({m}) && ({ni}+{c}*{lanes})<({n}))")
          self.line(
            f"vecops::vec::sme::mopa<{r * nr_n + c},{CPP_TYPES[dtype]},{CPP_TYPES[a.type.dtype]},{CPP_TYPES[b.type.dtype]}>(vecops::vec::mtrue({atag}{{}}),vecops::vec::mtrue({btag}{{}}),{aval},{bval});"
          )
          if guarded:
            self.close()

    padded, stop = False, k

    def k_loop():
      previous_full = getattr(self, "matrix_k_full", False)
      self.matrix_k_full = True
      if self.active_matmul_plan.schedule.pipeline == "staged":
        full = stop if padded or kp == 1 else f"(({k})/{kp}*{kp})"
        self.matrix_lookahead("sme.k", ki, {"K": full, "SKR": kp}, load_sme, compute_sme)
      else:
        self.line(f"vecops::nint_t {ki}=0;")
        self.line("/* full K groups: no per-element tail tests */")
        self.line("VECOPS_LOOP_ALIGN(64)")
        self.matrix_loop("sme.k", ki, {"K": stop, "SKR": kp}, initialize=False, bound="full-tiles")
        distance = kp * self.active_matmul_plan.schedule.loop("sme.k").schedule.distance
        future = f"({ki})+{distance}"
        self.matrix_prefetch(a, (mi, future), f"({k})-({ki})>{distance}")
        self.matrix_prefetch(b, (future, ni), f"({k})-({ki})>{distance}")
        compute_sme(load_sme(ki))
        self.close()
      self.matrix_k_full = False
      known_k = I.constant(a.type.shape[1])
      if not padded and kp > 1 and (not isinstance(known_k, int) or known_k % kp):
        tail = f"(({k})/{kp}*{kp})"
        self.opening(f"if (({k})%{kp}) /* one partial K group */")
        compute_sme(load_sme(tail))
        self.close()
      self.matrix_k_full = previous_full

    def emit_k_path():
      nonlocal padded, stop
      padded = all(e.op == "packed_load" and e.data.format in {"sme", "auto"} for e in (a, b))
      stop = f"((({k})+{kp - 1})/{kp}*{kp})" if padded and kp > 1 else k
      previous_exists = getattr(self, "sme_blocks_exist", False)
      previous_spatial = getattr(self, "sme_spatial_full", False)
      previous_vector = getattr(self, "full_vector", False)
      condition = (
        f"({m})-({mi})>{nr_m - 1}*{lanes} && ({n})-({ni})>{nr_n - 1}*{lanes}"
        if padded
        else f"({m})-({mi})>={nr_m}*{lanes} && ({n})-({ni})>={nr_n}*{lanes}"
      )
      packed_axes = tuple(e.op == "packed_load" and e.data.format in {"sme", "auto"} for e in (a, b))
      whole_packed = all(not e.data.origins for e, is_packed in zip((a, b), packed_axes) if is_packed)
      if packed_axes[0] != packed_axes[1] and whole_packed:
        # A validated native packed panel has full physical spatial words,
        # even when the corresponding logical axis is a fringe. Prove full
        # raw lanes independently; do not make a padded packed-B column
        # force a full raw-A row block through the partial spatial K path.
        # Logical K padding still requires both operands to be packed.
        # Sliced packed views retain their prior conservative condition:
        # their physical panel/cursor origin needs a separate proof.
        conditions = []
        for total, index, blocks, native_packed in zip((m, n), (mi, ni), (nr_m, nr_n), packed_axes):
          relation, bound = (">", blocks - 1) if native_packed else (">=", blocks)
          conditions.append(f"({total})-({index}){relation}{bound}*{lanes}")
        condition = " && ".join(conditions)
      if exact_blocks:
        # The pre-K dispatch proved every block nonempty, not every lane active.
        # Packed panels are padded; raw inputs still mask the final block.
        self.line("/* exact blocks: no empty-block tests inside K */")
        self.sme_blocks_exist = True
        self.sme_spatial_full = False
        self.full_vector = False
        k_loop()
      else:
        self.opening(f"if ({condition}) /* full ZA family: branch-free spatial K body */")
        self.sme_blocks_exist = True
        self.sme_spatial_full = not padded
        self.full_vector = not padded
        k_loop()
        self.close()
        self.opening("else /* partial spatial family */")
        self.sme_blocks_exist = False
        self.sme_spatial_full = False
        self.full_vector = False
        k_loop()
        self.close()
      self.sme_blocks_exist = previous_exists
      self.sme_spatial_full = previous_spatial
      self.full_vector = previous_vector

    if optional_a is None:
      emit_k_path()
    else:
      self.opening(f"if ({optional_a[1]}) /* native A packing dispatch outside K */")
      emit_k_path()
      self.close()
      self.opening("else /* raw A: no scratch load or allocation */")
      packed_a, a = a, optional_a[0]
      emit_k_path()
      a = packed_a
      self.close()
    self.sme_packed_cursors = old_cursors
    for r in range(nr_m):
      for c in range(nr_n):
        tile = r * nr_n + c
        ro, co = f"({mi}+{r}*{lanes})", f"({ni}+{c}*{lanes})"
        rows = active_extent(m, ro, r, nr_m)
        columns = active_extent(n, co, c, nr_n)
        self.sme_store_tile(statement, product, tile, ro, co, rows, columns, tag)
    self.close()
    self.close()

  def vector_matmul(self, statement, product):
    a, b = self.matrix_operands(product)
    initial = product.args[2:]
    dtype = product.type.dtype
    tag = self.matrix_tag(dtype)
    mr = self.active_matmul_plan.vector_rows
    m, n = (self.scalar(v) for v in product.type.shape)
    k = self.scalar(a.type.shape[1])
    width = self.vector_size(tag)
    mi, ni, ki = (self.unique(v) for v in ("matmul_m", "matmul_n", "matmul_k"))
    self.line("/* vector outer-product family: named accumulators, fused input/C transforms */")
    self.matrix_loop("vector.m", mi, {"M": m, "MR": mr})
    self.matrix_loop("vector.n", ni, {"N": n, "VL": width})
    active = f"std::min<vecops::nint_t>({width}, ({n})-{ni})"
    accumulators = []
    for row in range(mr):
      accumulator = self.unique("matrix_acc")
      self.line(f"auto {accumulator} = vecops::vec::zeros({tag}{{}});")
      coords = (f"({mi}+{row})", ni)
      phase = getattr(self, "matrix_phase", None)
      if phase is not None:
        self.opening(f"if ({phase[0]})")
      if initial:
        self.opening(f"if ({mi}+{row} < ({m}))")
        value = self.vector(initial[0], tag, product.type.shape, coords, active, 1)
        self.line(f"{accumulator} = {value};")
        self.close()
      if phase is not None:
        self.close()
        self.opening(f"else if ({mi}+{row}<({m}))")
        stride = phase[2].strides[1]
        extra = "" if I.constant(stride) == 1 else f",vecops::vec::opt::strided({self.scalar(stride)})"
        self.line(
          f"{accumulator}=vecops::vec::load({tag}{{}},{self.pointer(phase[2], coords)},vecops::vec::opt::first({active}){extra});"
        )
        self.close()
      accumulators.append(accumulator)
    scalar_tag = self.matrix_tag(dtype, lanes=1)

    def load_vector(index):
      values = [(tag, self.matrix_value(b, tag, (index, ni), active, 1))]
      for row in range(mr):
        broadcast = self.unique("matrix_a")
        self.line(f"auto {broadcast}=vecops::vec::zeros({tag}{{}});")
        self.opening(f"if ({mi}+{row}<({m}))")
        av = self.matrix_value(a, scalar_tag, (f"({mi}+{row})", index), "1", 1)
        self.line(f"{broadcast}=vecops::vec::fill({tag}{{}},vecops::vec::get({scalar_tag}{{}},{av},0));")
        self.close()
        values.append((tag, broadcast))
      return tuple(values)

    def compute_vector(values):
      bv = values[0][1]
      for row, accumulator in enumerate(accumulators):
        self.opening(f"if ({mi}+{row}<({m}))")
        if len(values) == 1:
          av = self.matrix_value(a, scalar_tag, (f"({mi}+{row})", ki), "1", 1)
          broadcast = self.unique("matrix_a")
          self.line(f"auto {broadcast}=vecops::vec::fill({tag}{{}},vecops::vec::get({scalar_tag}{{}},{av},0));")
        else:
          broadcast = values[row + 1][1]
        if dtype.startswith("float"):
          self.line(f"{accumulator}=vecops::vec::fmadd({tag}{{}},{broadcast},{bv},{accumulator});")
        else:
          self.line(
            f"{accumulator}=vecops::vec::add({tag}{{}},{accumulator},vecops::vec::mul({tag}{{}},{broadcast},{bv}));"
          )
        self.close()

    if self.active_matmul_plan.schedule.pipeline == "staged":
      self.matrix_lookahead("vector.k", ki, {"K": k, "one": 1}, load_vector, compute_vector)
    else:
      self.matrix_loop("vector.k", ki, {"K": k, "one": 1})
      distance = self.active_matmul_plan.schedule.loop("vector.k").schedule.distance
      future = f"({ki})+{distance}"
      self.matrix_prefetch(a, (mi, future), f"({k})-({ki})>{distance}")
      self.matrix_prefetch(b, (future, ni), f"({k})-({ki})>{distance}")
      compute_vector(((tag, self.matrix_value(b, tag, (ki, ni), active, 1)),))
      self.close()
    for row, accumulator in enumerate(accumulators):
      self.opening(f"if ({mi}+{row} < ({m}))")
      self.vector_matrix_epilogue(statement, product, tag, (f"({mi}+{row})", ni), active, accumulator)
      self.close()
    self.close()
    self.close()
