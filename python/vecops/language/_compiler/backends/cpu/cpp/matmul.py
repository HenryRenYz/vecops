"""Print planned vector/FMA and matrix-instruction schedules.

The emitter owns no algorithm-selection passes and never calls ops::Matmul
or its private kernel. Registers are always independent C++ identifiers.
"""

from dataclasses import replace
from .... import ir as I
from ....ir.traversal import walk
from ....dtypes import ELEMENT_BYTES
from ..capabilities import CPP_TYPES

FORMATS = {"auto": -1, "dense": 0, "amx": 1, "sme": 2}


class MatmulEmitter:
  def matrix_permutation(self, network, inputs):
    """Print the prepared dword permutation with individually named registers."""
    names, tags, ordinal = {}, {}, 0

    def tag(lanes):
      if lanes not in tags:
        tags[lanes] = self.matrix_tag("uint32", lanes=lanes)
      return tags[lanes]

    for index, operation in enumerate(network.operations):
      own = tag(operation.lanes)
      if operation.kind == "input":
        names[index] = inputs[ordinal]
        ordinal += 1
        continue
      name = self.unique("pack_permute")
      args = [names[i] for i in operation.operands]
      if operation.kind.startswith("interleave_"):
        stage = f"vecops::vec::ViewAs<std::uint{operation.element_bytes * 8}_t,{own}>"
        a, b = (f"vecops::vec::bitcast({stage}{{}},{own}{{}},{v})" for v in args)
        self.line(
          f"auto {name}=vecops::vec::bitcast({own}{{}},{stage}{{}},vecops::vec::local_{operation.kind}({stage}{{}},{a},{b}));"
        )
      elif operation.kind == "concat":
        self.line(f"auto {name}=vecops::vec::concat({own}{{}},{','.join(args)});")
      elif operation.kind.startswith("x86_shuffle_128_"):
        immediate = "0x88" if operation.kind.endswith("lower") else "0xdd"
        self.line(
          f"auto {name}=vecops::vec::Vec<{own}>{{_mm512_shuffle_i32x4({args[0]}.value,{args[1]}.value,{immediate})}};"
        )
      else:
        parent = tag(network.operations[operation.operands[0]].lanes)
        self.line(f"auto {name}=vecops::vec::{operation.kind}({parent}{{}},{args[0]});")
      names[index] = name
    return tag(network.rows), tuple(names[i] for i in network.outputs)

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
    if along_k and packed.side == "B" and packed.format in {"amx", "sme"}:
      kp = max(1, 4 // ELEMENT_BYTES[packed.view.dtype])
      input_tag = f"vecops::vec::Rebind<{CPP_TYPES[packed.view.dtype]},{tag}>"
      bits = f"vecops::vec::ViewAs<std::uint32_t,{input_tag}>"
      panel = f"vecops_dsl::pack_panel<{CPP_TYPES[packed.view.dtype]},{FORMATS[packed.format]}>()"
      name = self.unique("packed_k_stream")
      self.line(
        f"auto {name}=vecops::vec::bitcast({input_tag}{{}},{bits}{{}},vecops::vec::load({bits}{{}},reinterpret_cast<const std::uint32_t*>({pointer}+({offset})),vecops::vec::opt::first((({active})+{kp}-1)/{kp}),vecops::vec::opt::strided({panel})));"
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
      halfwidth = f"vecops::vec::size({half}{{}})"
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
    width = f"vecops::vec::size({tag}{{}})"
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
    dtype, format_, side = packed.view.dtype, packed.format, packed.side
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
    self.pack_group_vectors(source, packed, spatial, logical_k, panel, block)
    if amx:
      self.line("#if defined(HAS_AMX_TILE)")
      self.close()
      self.line("#endif")

  def interleave_group(self, tag, values, dtype):
    if len(values) == 1:
      return tag, values[0]
    # Tags are named aliases. Use Twice rather than infer their declaration.
    pair = self.unique("pack_pair_tag")
    self.line(f"using {pair} = vecops::vec::Twice<{tag}>;")
    self.tag_dtypes[pair] = dtype
    v01 = self.unique("interleave_pair")
    self.line(f"auto {v01} = vecops::vec::interleave({pair}{{}}, {values[0]}, {values[1]});")
    if len(values) == 2:
      return pair, v01
    v23 = self.unique("interleave_pair")
    self.line(f"auto {v23} = vecops::vec::interleave({pair}{{}}, {values[2]}, {values[3]});")
    view = f"vecops::vec::ViewAs<std::uint16_t, {pair}>"
    quadview, quad = f"vecops::vec::Twice<{view}>", f"vecops::vec::Twice<{pair}>"
    result = self.unique("interleave_quad")
    self.line(
      f"auto {result} = vecops::vec::bitcast({quad}{{}}, {quadview}{{}}, vecops::vec::interleave({quadview}{{}}, vecops::vec::bitcast({view}{{}}, {pair}{{}}, {v01}), vecops::vec::bitcast({view}{{}}, {pair}{{}}, {v23}))); "
    )
    self.tag_dtypes[quad] = dtype
    return quad, result

  def pack_amx_vectors(self, source, packed, spatial, logical_k):
    dtype, side = packed.view.dtype, packed.side
    width = ELEMENT_BYTES[dtype]
    kr, kp = 64 // width, 4 // width
    s, k, row = (self.unique(v) for v in ("pack_panel_origin", "pack_k_origin", "pack_row"))
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
    lanes = f"vecops::vec::size({tag}{{}})"
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

  def pack_scalar_fallback(self, source, packed, spatial, logical_k, panel, block):
    cpp, side = CPP_TYPES[packed.view.dtype], packed.side
    dtype = packed.view.dtype
    # The architecture's packed loop is specialized below. This scalar
    # spelling is the correctness fallback for non-native permutations.
    s, k = self.unique("pack_s"), self.unique("pack_k")
    self.opening(f"for (vecops::nint_t {s}=0; {s} < (({spatial}+{panel}-1)/{panel})*{panel}; ++{s})")
    self.opening(f"for (vecops::nint_t {k}=0; {k} < (({logical_k}+{block}-1)/{block})*{block}; ++{k})")
    offset = self.packed_offset(packed, s, k)
    pointer = self.pointer(packed.view)
    self.opening(f"if ({s} < ({spatial}) && {k} < ({logical_k}))")
    tag = self.matrix_tag(dtype, lanes=1)
    coords = (s, k) if side == "A" else (k, s)
    value = self.matrix_value(source, tag, coords, "1", 1 if side == "A" else 0)
    self.line(f"{pointer}[{offset}] = vecops::vec::get({tag}{{}}, {value}, 0);")
    self.close()
    self.opening("else")
    self.line(f"{pointer}[{offset}] = {cpp}{{}};")
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
      yield value
      if value.op != "dot":
        for argument in value.args:
          yield from operands(argument)

    for root in (*product.args, statement.value):
      if not isinstance(root.type, I.TileType):
        continue
      for value in operands(root):
        if not isinstance(value.type, I.TileType):
          continue
        for source, destination in zip(reversed(value.type.shape), reversed(root.type.shape)):
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
    inputs = []
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
      inputs.append(expression)
    self.dot_inputs = {id(product): tuple(inputs)}
    a, b = self.matrix_operands(product)
    dtype, implementation = product.type.dtype, product.data.implementation
    native_half = a.type.dtype == b.type.dtype and a.type.dtype in {"bfloat16", "float16"} and dtype == "float32"
    native_int = a.type.dtype in {"int8", "uint8"} and b.type.dtype in {"int8", "uint8"} and dtype == "int32"
    amx = native_half or native_int
    sme = amx or a.type.dtype == b.type.dtype == dtype and dtype in {"float32", "float64"}
    if implementation == "amx" and not amx or implementation == "sme" and not sme:
      raise TypeError(f"unsupported {implementation} dot dtype combination")
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
      self.sme_matmul(statement, product)
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
      halfwidth = f"vecops::vec::size({half}{{}})"
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
    narrow = ELEMENT_BYTES[a.type.dtype] == ELEMENT_BYTES[b.type.dtype] < ELEMENT_BYTES[dtype] and (
      a.type.dtype == b.type.dtype == "bfloat16"
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
    width = f"vecops::vec::size({atag}{{}})"
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
    self.opening(f"for (vecops::nint_t {mi}=0; {mi}<({m}); ++{mi})")
    self.opening(f"for (vecops::nint_t {ni}={start}; {ni}<({end}); {ni} += {column_count})")
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
      self.opening(f"for (; {ki}<{bulk}; {ki}+={unroll}*({width}))")
      for bank in range(unroll):
        step(f"({ki})+{bank}*({width})", width, bank, True, full_columns)
      self.close()
      self.opening(f"for (; ({k})-{ki}>=({width}); {ki}+={width})")
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
    if separate_tail and not tail and column_count > 1:
      # Known narrowed groups divide N exactly and need no remainder.
      if column_count == self.active_matmul_plan.vector_columns:
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

  def amx_matmul(self, statement, product, *, family=None, skip_cache=False, single_m_tile=False):
    a, b = self.matrix_operands(product)
    initial = product.args[2:]
    dtype = product.type.dtype
    m, n = (self.scalar(v) for v in product.type.shape)
    k = self.scalar(a.type.shape[1])
    kr = 64 // ELEMENT_BYTES[a.type.dtype]
    cm, cn = (I.constant(v) for v in product.type.shape)
    if family is None and (cm is None or cn is None):
      if product.data.tiled and any(
        v is not None for v in (product.data.block_m, product.data.block_n, product.data.block_k)
      ):
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
        cache = self.matrix_cache_blocks(product, mr, nr, kr)
        minimum_m = 32 if isinstance(cm, int) and cm > 16 else 16
        minimum_n = 32 if isinstance(cn, int) and cn > 16 else 16
        self.blocked_matrix_matmul(
          statement, product, cache, "amx", spatial_alignment=(minimum_m, minimum_n), k_alignment=kr
        )
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
    cache = None if skip_cache else self.matrix_cache_blocks(product, rows * 16, columns * 16, kr)
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
      )
      if bypass:
        self.close()
      return
    a, b = self.automatic_matrix_packing(product, "amx", single_m_tile=single_m_tile)
    cfg, session, scope = (self.unique(v) for v in ("dot_tilecfg", "dot_session", "dot_scope"))
    self.line(f"vecops::execution::amx::TileConfiguration {cfg};")
    native_m = min(16, cm) if isinstance(cm, int) and cm > 0 else 16
    native_n = min(16, cn) if isinstance(cn, int) and cn > 0 else 16
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
    self.opening(f"for (vecops::nint_t {ni}={starts[1]}; {ni}<({n}); {ni}+={columns * 16})")
    self.opening(f"for (vecops::nint_t {mi}={starts[0]}; {mi}<({m}); {mi}+={rows * 16})")
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
      self.matrix_n_full = spatial_full
      self.line(f"vecops::nint_t {ki}=0;")
      # Extents are nonnegative and this loop never advances beyond K; a
      # subtraction comparison avoids signed-128 arithmetic in every step.
      self.opening(f"for (;({k})-{ki}>={kr};{ki}+={kr}) /* full K: unmasked load/convert */")
      multiply(True)
      self.close()
      if not (isinstance(I.constant(a.type.shape[1]), int) and I.constant(a.type.shape[1]) % kr == 0):
        self.opening(f"if ({ki}<({k})) /* one zero-padded K tail */")
        multiply(False)
        self.close()
      self.matrix_n_full = previous_n

    # Unswitch N-tail checks before K. GCC otherwise keeps 16 predicates per
    # B tile load in the hot full-K loop of a dynamic small-M product.
    if isinstance(cn, int) and cn % (columns * 16) == 0:
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
    self.opening(f"if ({ro}+16<=({m}) && {co}+16<=({n}))")
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
      f"{value}=vecops::vec::load({tag}{{}},{self.pointer(view, (f'({ro}+{row})', co))},vecops::vec::opt::first({active}));"
    )
    self.close()
    self.line(f"vecops::vec::store({tag}{{}},{scratch}+{row}*16,{value});")
    self.close()
    self.line(f"vecops::vec::amx::load<{tile}>({scratch},64);")
    self.close()

  def amx_partial_store(self, view, tile, ro, co, m, n, tag, scratch):
    self.opening(f"if ({ro}+16<=({m}) && {co}+16<=({n}))")
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
      f"vecops::vec::store({tag}{{}},{self.pointer(view, (f'({ro}+{row})', co))},{value},vecops::vec::opt::first({active}));"
    )
    self.close()
    self.close()

  def slice_matrix_view(self, view, origins, shape):
    offset = view.offset
    sizes = []
    for extent, stride, origin, size in zip(view.shape, view.strides, origins, shape):
      if I.constant(extent) == 1:
        sizes.append(extent)
      else:
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
    sizes = tuple(extent if I.constant(extent) == 1 else size for extent, size in zip(value.type.shape, shape))
    data = self.slice_matrix_view(value.data, origins, shape) if value.op == "load" else value.data
    if value.op == "packed_load":
      old = value.data.origins or (I.literal(0), I.literal(0))
      data = replace(value.data, origins=tuple(I.binary("add", a, b) for a, b in zip(old, origins)))
    args = tuple(self.slice_matrix_value(a, origins, shape, memo) for a in value.args)
    if value.op == "arange":
      args = (I.binary("add", args[0], origins[data if data is not None else 1]),)
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

  def emit_matrix_pack(self, operand, side, format_, policy):
    direct = operand.op == "load" and not operand.args and operand.data.dtype == operand.type.dtype
    contiguous = I.constant(operand.data.strides[1 if side == "A" else 0]) == 1 if direct else False
    if operand.op == "packed_load" or policy == "none" or policy != "auto" and side not in policy:
      return operand
    if policy == "auto" and side == "A" and direct and contiguous:
      return operand
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
    self.opening(f"for (vecops::nint_t {nb}=0; {nb}<({self.scalar(n)}); {nb}+={bn})")
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
    self.opening(f"for (vecops::nint_t {kb}=0; {kb}<std::max<vecops::nint_t>(1,{self.scalar(k)}); {kb}+={bk})")
    self.line(f"const auto {kk}=std::min<vecops::nint_t>({bk},({self.scalar(k)})-{kb});")
    bv = self.slice_matrix_value(b, (raw(kb), raw(nb)), (ks, ns))
    if not (single_m_tile and format_ == "amx" and product.data.packing == "auto"):
      bv = self.emit_matrix_pack(bv, "B", format_, product.data.packing)
    self.opening(f"for (vecops::nint_t {mb}=0; {mb}<({self.scalar(m)}); {mb}+={bm})")
    self.line(f"const auto {mm}=std::min<vecops::nint_t>({bm},({self.scalar(m)})-{mb});")
    av = self.slice_matrix_value(a, (raw(mb), raw(kb)), (ms, ks))
    av = self.emit_matrix_pack(av, "A", format_, product.data.packing)
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
      )
    else:
      emit(replace(statement, destination=sink, value=value, mask=mask), local)
    self.matrix_phase = old_phase
    self.close()
    self.close()
    self.close()
    self.materialized, self.snapshots = old_materialized, old_snapshots

  def matrix_cache_blocks(self, product, mr, nr, kr):
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
    self.line(
      f"const auto {name}=vecops_dsl::matmul_blocks({mr},{nr},{kr},{ELEMENT_BYTES[a.type.dtype]},{ELEMENT_BYTES[b.type.dtype]},{ELEMENT_BYTES[product.type.dtype]},{self.scalar(a.type.shape[1])});"
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
      if (
        policy == "auto"
        and side == "B"
        and (
          single_m_tile
          or isinstance(I.constant(product.type.shape[0]), int)
          and I.constant(product.type.shape[0]) <= 32
        )
      ):
        # No cross-M reuse: raw instruction staging is already the one pack
        # needed by this tile; a second packed buffer only adds traffic.
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
      self.line(f"auto {value}=vecops::vec::zeros({native}{{}});")
      self.opening(f"if (({origin})<({spatial}))")
      offset = self.packed_offset(operand.data, origin, ki)
      self.line(f"{value}=vecops::vec::load({native}{{}}, {self.pointer(operand.data.view)}+({offset}));")
      self.close()
      return native, value
    active = (
      f"std::max<vecops::nint_t>(0,std::min<vecops::nint_t>(vecops::vec::size({tag}{{}}),({spatial})-({origin})))"
    )
    values = []
    for group in range(kp):
      value = self.unique("sme_operand")
      self.line(f"auto {value}=vecops::vec::zeros({tag}{{}});")
      self.opening(f"if (({ki})+{group}<({k}) && ({active})>0)")
      coords = (origin, f"({ki})+{group}") if side == "A" else (f"({ki})+{group}", origin)
      loaded = self.matrix_value(operand, tag, coords, active, 0 if side == "A" else 1)
      loaded = self.zero_matrix_tail(tag, loaded, active)
      self.line(f"{value}={loaded};")
      self.close()
      values.append(value)
    outtag, value = self.interleave_group(tag, values, dtype)
    return outtag, value

  def sme_matmul(self, statement, product):
    a, b = self.matrix_operands(product)
    initial = product.args[2:]
    dtype = product.type.dtype
    m, n = (self.scalar(v) for v in product.type.shape)
    k = self.scalar(a.type.shape[1])
    kp = max(1, 4 // ELEMENT_BYTES[a.type.dtype])
    span = f"2*vecops::vec::sme::streaming_lanes<{CPP_TYPES[dtype]}>()"
    cache = self.matrix_cache_blocks(product, span, span, kp)
    if cache is not None:
      self.blocked_matrix_matmul(statement, product, cache, "sme")
      return
    a, b = self.automatic_matrix_packing(product, "sme")
    session, scope = (self.unique(v) for v in ("sme_dot_session", "sme_dot_scope"))
    self.line(f"vecops::execution::ExecutionSession {session};")
    self.opening(
      f"{session}.with_resources(vecops::execution::sme::Region{{}}, [&](auto& {scope}) VECOPS_INLINE_LAMBDA_NOEXCEPT"
    )
    tag = self.matrix_tag(dtype)
    lanes = self.unique("sme_matrix_lanes")
    self.line(f"const vecops::nint_t {lanes}=vecops::vec::size({tag}{{}});")
    mi, ni, ki = (self.unique(v) for v in ("sme_m", "sme_n", "sme_k"))
    self.line("/* SME 2x2 outer-product schedule; all vectors are born inside SM */")
    self.opening(f"for (vecops::nint_t {mi}=0; {mi}<({m}); {mi}+=2*{lanes})")
    self.opening(f"for (vecops::nint_t {ni}=0; {ni}<({n}); {ni}+=2*{lanes})")
    self.line("vecops::vec::sme::zero_za();")
    for r in range(2):
      for c in range(2):
        tile = r * 2 + c
        ro, co = f"({mi}+{r}*{lanes})", f"({ni}+{c}*{lanes})"
        active = f"std::max<vecops::nint_t>(0,std::min<vecops::nint_t>({lanes},({n})-{co}))"
        pg = self.unique("sme_columns")
        self.line(f"auto {pg}=vecops::vec::mwhilelt({tag}{{}},0,{active});")
        phase = getattr(self, "matrix_phase", None)
        if phase is not None:
          self.opening(f"if ({phase[0]}) /* prologue exactly once */")
        if initial:
          row = self.unique("sme_seed_row")
          self.opening(f"for (vecops::nint_t {row}=0; {row}<{lanes} && {ro}+{row}<({m}) && ({active})>0; ++{row})")
          seed = initial[0]
          direct = (
            seed.op == "load"
            and not seed.args
            and seed.data.dtype == dtype
            and len(seed.data.shape) == 2
            and I.constant(seed.data.strides[1]) == 1
            and all(I.same_expr(x, y) for x, y in zip(seed.data.shape, product.type.shape))
          )
          coords = (f"({ro}+{row})", co)
          if direct:
            bits = "std::uint64_t" if ELEMENT_BYTES[dtype] == 8 else "std::uint32_t"
            bitstag = f"vecops::vec::ScalableTag<{bits},0>"
            pointer = self.pointer(seed.data, coords)
            if id(seed) in self.snapshots:
              saved, alias = self.snapshots[id(seed)]
              pointer = f"({alias} ? {self.pointer(saved, coords)} : {pointer})"
            self.line(
              f"vecops::vec::sme::load_hor<{tile}>(static_cast<std::uint32_t>({row}),static_cast<vecops::vec::Mask<{bitstag}>>({pg}),reinterpret_cast<const {bits}*>({pointer}));"
            )
          else:
            self.chunk_cache = {}
            value = self.vector(seed, tag, product.type.shape, coords, active, 1)
            self.line(f"vecops::vec::sme::write_hor<{tile}>(static_cast<std::uint32_t>({row}),{pg},{value});")
          self.close()
        if phase is not None:
          self.close()
          self.opening("else /* native ZA accumulator continuation */")
          row = self.unique("sme_partial_row")
          self.opening(f"for (vecops::nint_t {row}=0; {row}<{lanes} && {ro}+{row}<({m}) && ({active})>0; ++{row})")
          bits = "std::uint64_t" if ELEMENT_BYTES[dtype] == 8 else "std::uint32_t"
          bitstag = f"vecops::vec::ScalableTag<{bits},0>"
          pointer = self.pointer(phase[2], (f"({ro}+{row})", co))
          self.line(
            f"vecops::vec::sme::load_hor<{tile}>(static_cast<std::uint32_t>({row}),static_cast<vecops::vec::Mask<{bitstag}>>({pg}),reinterpret_cast<const {bits}*>({pointer}));"
          )
          self.close()
          self.close()
    self.opening(f"for (vecops::nint_t {ki}=0; {ki}<({k}); {ki}+={kp})")
    av = [self.sme_operand_vector(a, "A", f"({mi}+{r}*{lanes})", ki, m, k) for r in range(2)]
    bv = [self.sme_operand_vector(b, "B", f"({ni}+{c}*{lanes})", ki, n, k) for c in range(2)]
    for r in range(2):
      for c in range(2):
        atag, aval = av[r]
        btag, bval = bv[c]
        self.line(
          f"vecops::vec::sme::mopa<{r * 2 + c},{CPP_TYPES[dtype]},{CPP_TYPES[a.type.dtype]},{CPP_TYPES[b.type.dtype]}>(vecops::vec::mtrue({atag}{{}}),vecops::vec::mtrue({btag}{{}}),{aval},{bval});"
        )
    self.close()
    for r in range(2):
      for c in range(2):
        tile = r * 2 + c
        ro, co = f"({mi}+{r}*{lanes})", f"({ni}+{c}*{lanes})"
        row = self.unique("sme_store_row")
        active = f"std::max<vecops::nint_t>(0,std::min<vecops::nint_t>({lanes},({n})-{co}))"
        self.opening(f"for (vecops::nint_t {row}=0; {row}<{lanes} && {ro}+{row}<({m}) && ({active})>0; ++{row})")
        pg = self.unique("sme_store_mask")
        self.line(f"auto {pg}=vecops::vec::mwhilelt({tag}{{}},0,{active});")
        value = self.unique("sme_result")
        self.line(f"auto {value}=vecops::vec::sme::read_hor<{tile}>({tag}{{}},static_cast<std::uint32_t>({row}),{pg});")
        if phase is not None:
          self.opening(f"if ({phase[1]}) /* epilogue exactly once */")
        self.matrix_epilogue(statement, product, tag, (f"({ro}+{row})", co), active, value)
        if phase is not None:
          self.close()
          self.opening("else")
          self.line(
            f"vecops::vec::store({tag}{{}},{self.pointer(phase[2], (f'({ro}+{row})', co))},{value},vecops::vec::opt::first({active}));"
          )
          self.close()
        self.close()
    self.close()
    self.close()
    self.indent -= 1
    self.line("});")

  def vector_matmul(self, statement, product):
    a, b = self.matrix_operands(product)
    initial = product.args[2:]
    dtype = product.type.dtype
    tag = self.matrix_tag(dtype)
    mr = self.active_matmul_plan.vector_rows
    m, n = (self.scalar(v) for v in product.type.shape)
    k = self.scalar(a.type.shape[1])
    width = f"vecops::vec::size({tag}{{}})"
    mi, ni, ki = (self.unique(v) for v in ("matmul_m", "matmul_n", "matmul_k"))
    self.line("/* vector outer-product family: named accumulators, fused input/C transforms */")
    self.opening(f"for (vecops::nint_t {mi}=0; {mi} < ({m}); {mi} += {mr})")
    self.opening(f"for (vecops::nint_t {ni}=0; {ni} < ({n}); {ni} += {width})")
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
        self.line(
          f"{accumulator}=vecops::vec::load({tag}{{}},{self.pointer(phase[2], coords)},vecops::vec::opt::first({active}));"
        )
        self.close()
      accumulators.append(accumulator)
    self.opening(f"for (vecops::nint_t {ki}=0; {ki} < ({k}); ++{ki})")
    bv = self.matrix_value(b, tag, (ki, ni), active, 1)
    scalar_tag = self.matrix_tag(dtype, lanes=1)
    for row, accumulator in enumerate(accumulators):
      self.opening(f"if ({mi}+{row} < ({m}))")
      av = self.matrix_value(a, scalar_tag, (f"({mi}+{row})", ki), "1", 1)
      broadcast = self.unique("matrix_a")
      self.line(f"auto {broadcast} = vecops::vec::fill({tag}{{}}, vecops::vec::get({scalar_tag}{{}}, {av}, 0));")
      op = "fmadd" if dtype.startswith("float") else "mul"
      if op == "fmadd":
        self.line(f"{accumulator} = vecops::vec::fmadd({tag}{{}}, {broadcast}, {bv}, {accumulator});")
      else:
        self.line(
          f"{accumulator} = vecops::vec::add({tag}{{}}, {accumulator}, vecops::vec::mul({tag}{{}}, {broadcast}, {bv}));"
        )
      self.close()
    self.close()
    for row, accumulator in enumerate(accumulators):
      self.opening(f"if ({mi}+{row} < ({m}))")
      self.vector_matrix_epilogue(statement, product, tag, (f"({mi}+{row})", ni), active, accumulator)
      self.close()
    self.close()
    self.close()
