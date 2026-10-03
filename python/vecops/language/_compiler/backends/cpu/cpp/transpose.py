"""Print prepared transpose DAGs. Algorithm selection lives in CPU planning."""

from .... import ir as I
from ....dtypes import ELEMENT_BYTES
from ..capabilities import CPP_TYPES


class TransposeEmitter:
  def transpose_store(self, statement, plan, *, fallback=None):
    """Return whether a prepared layout-conversion plan was emitted."""
    if plan is None:
      return False

    def stream():
      if fallback is None:
        self.streaming_store(statement)
      else:
        fallback()

    loads = tuple(I.loads(statement.value))
    guards = [f"({self.scalar(statement.destination.strides[-1])}) == 1"]
    guards.extend(f"({self.scalar(load.data.strides[plan.source_axis])}) == 1" for load in loads)
    self.opening("if (" + " && ".join(dict.fromkeys(guards)) + ") /* register layout transpose */")
    self.emitted_regions.append(plan)
    snapshots = self.snapshots
    aliases = [snapshots[id(load)][1] for load in loads if id(load) in snapshots]
    if aliases:
      self.opening("if (" + " || ".join(dict.fromkeys(aliases)) + ") /* overlapping operands */")
      stream()
      self.close()
      self.opening("else")
      # Alias is a region-level choice, not a ternary on every vector load.
      # Keep the hot DAG independent of scratch layout and snapshot branches.
      self.snapshots = {
        identity: entry for identity, entry in snapshots.items() if identity not in {id(load) for load in loads}
      }
    batch = []
    for extent in statement.destination.shape[:-2]:
      coordinate = self.unique("transpose_batch")
      self.opening(f"for (vecops::nint_t {coordinate} = 0; {coordinate} < ({self.scalar(extent)}); ++{coordinate})")
      batch.append(coordinate)
    implementation = statement.implementation
    if implementation == "sme" and not plan.sme:
      raise TypeError("SME transpose requires an unmasked load (optionally converted), not an elementwise expression")
    if plan.sme and (implementation == "sme" or implementation == "auto" and plan.sme_preferred):
      self.line(
        "#if defined(HAS_SME_FA64) && (!defined(HAS_FIXED_SVE_BITS) || (defined(FIXED_STREAMING_SVE_BITS) && FIXED_SVE_BITS == FIXED_STREAMING_SVE_BITS))"
      )
      self.sme_transpose(plan, tuple(batch))
      self.line("#else")
      if implementation == "sme":
        self.line('#error "explicit SME transpose requires an SME-FA64 target"')
      else:
        self.register_transpose(plan, tuple(batch))
      self.line("#endif")
    else:
      self.register_transpose(plan, tuple(batch))
    for _ in batch:
      self.close()
    self.snapshots = snapshots
    if aliases:
      self.close()
    self.close()
    self.opening("else /* unsupported strides: preserve gather/scatter */")
    stream()
    self.close()
    return True

  def register_transpose(self, plan, batch):
    for variant, (guard, network) in enumerate(plan.variants):
      self.line(f"#{'if' if variant == 0 else 'elif'} {guard}")
      self.register_transpose_variant(plan, network, batch)
    self.line("#endif")

  def register_transpose_variant(self, plan, network, batch):
    statement, dtype = plan.statement, network.dtype or plan.dtype
    destination = statement.destination
    source = statement.value
    if source.op == "cast" and source.type.dtype == destination.dtype:
      source = source.args[0]
    if dtype != source.type.dtype:
      source = I.Expr(I.TileType(dtype, source.type.shape), "load", data=source.data)
    rows, columns = network.rows, network.columns
    m, n = map(self.scalar, destination.shape[-2:][::-1])
    self.opening(f"if (({m}) == {rows} && ({n}) == {columns}) /* single register tile */")
    self.register_tile(plan, network, source, batch, "0", "0", None)
    self.close()
    self.opening("else")
    mb, nb, mi, ni = (self.unique(name) for name in ("cache_m", "cache_n", "tile_m", "tile_n"))
    mend, nend = self.unique("m_end"), self.unique("n_end")
    cache = plan.cache_block
    self.opening(f"for (vecops::nint_t {mb} = 0; {mb} < ({m}); {mb} += {cache})")
    self.line(f"const auto {mend} = std::min<vecops::nint_t>({mb} + {cache}, {m});")
    self.opening(f"for (vecops::nint_t {nb} = 0; {nb} < ({n}); {nb} += {cache})")
    self.line(f"const auto {nend} = std::min<vecops::nint_t>({nb} + {cache}, {n});")
    self.line(f"vecops::nint_t {ni} = {nb};")
    if plan.loop_order == "output":
      self.opening(f"for (; {ni} + {columns} <= {nend}; {ni} += {columns})")
      self.line(f"vecops::nint_t {mi} = {mb};")
      self.opening(f"for (; {mi} + {rows} <= {mend}; {mi} += {rows})")
      self.register_tile(plan, network, source, batch, mi, ni, None)
      self.close()
      self.opening(f"if ({mi} < {mend})")
      self.register_tile(plan, network, source, batch, mi, ni, f"({mend} - {mi})")
      self.close()
      self.close()
    else:
      self.line(f"vecops::nint_t {mi} = {mb};")
      self.opening(f"for (; {mi} + {rows} <= {mend}; {mi} += {rows})")
      self.opening(f"for ({ni} = {nb}; {ni} + {columns} <= {nend}; {ni} += {columns})")
      self.register_tile(plan, network, source, batch, mi, ni, None)
      self.close()
      self.close()
      self.opening(f"if ({mi} < {mend})")
      self.opening(f"for ({ni} = {nb}; {ni} + {columns} <= {nend}; {ni} += {columns})")
      self.register_tile(plan, network, source, batch, mi, ni, f"({mend} - {mi})")
      self.close()
      self.close()
      self.line(f"{ni} = {nb} + (({nend} - {nb}) / {columns}) * {columns};")
    # Only the narrow source-column fringe gathers; complete tiles use dense
    # loads, register permutations and dense converting stores.
    tag = self.unique("fringe_tag")
    self.line("#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)")
    self.line(
      f"using {tag} = vecops::vec::FixedTag<{CPP_TYPES[dtype]}, {min(rows, 4, 16 // (plan.lane_bytes or ELEMENT_BYTES[dtype]))}>;"
    )
    self.line("#else")
    self.line(f"using {tag} = vecops::vec::FixedTag<{CPP_TYPES[dtype]}, {rows}>;")
    self.line("#endif")
    self.tag_dtypes[tag] = dtype
    self.opening(f"for (; {ni} < {nend}; ++{ni})")
    self.opening(f"for (vecops::nint_t {mi} = {mb}; {mi} < {mend}; {mi} += vecops::vec::size({tag}{{}}))")
    active = f"std::min<vecops::nint_t>(vecops::vec::size({tag}{{}}), {mend} - {mi})"
    coords = batch + (ni, mi)
    before = self.chunk_cache
    self.chunk_cache = {}
    value = self.vector(source, tag, destination.shape, coords, active)
    self.line(
      f"vecops::vec::store_convert({tag}{{}}, {self.pointer(destination, coords)}, {value}, vecops::vec::opt::first({active}));"
    )
    self.chunk_cache = before
    self.close()
    self.close()
    self.close()
    self.close()
    self.close()

  def register_tile(self, plan, network, source, batch, mi, ni, active_m):
    dtype = network.dtype or plan.dtype
    names, tags = {}, {}
    before = self.chunk_cache
    self.chunk_cache = {}
    input_row = 0

    def tag(lanes):
      if lanes not in tags:
        name = self.unique("transpose_tag")
        tags[lanes] = name
        self.tag_dtypes[name] = dtype
        self.line(f"using {name} = vecops::vec::FixedTag<{CPP_TYPES[dtype]}, {lanes}>;")
      return tags[lanes]

    for index, operation in enumerate(network.operations):
      own_tag = tag(operation.lanes)
      name = self.unique("transpose_vec")
      args = [names[i] for i in operation.operands]
      if operation.kind == "input":
        coords = batch + (ni, f"({mi} + {input_row})")
        if active_m is not None:
          self.line(f"vecops::vec::Vec<{own_tag}> {name} = vecops::vec::zeros({own_tag}{{}});")
          self.opening(f"if ({input_row} < {active_m})")
        previous_full = getattr(self, "full_vector", False)
        # A complete source row has no inactive lanes, even in a row fringe.
        # Preserve that fact at the memory boundary: first(constant_lanes) can
        # select a scalar masked conversion rather than native FP16->FP64.
        self.full_vector = True
        value = self.vector(
          source, own_tag, plan.statement.destination.shape, coords, str(network.columns), plan.source_axis
        )
        self.full_vector = previous_full
        self.line(f"{'auto ' if active_m is None else ''}{name} = {value};")
        if active_m is not None:
          self.close()
        input_row += 1
      elif operation.kind.startswith("interleave_"):
        width = operation.element_bytes
        stage = f"vecops::vec::ViewAs<std::uint{width * 8}_t, {own_tag}>"
        a, b = (f"vecops::vec::bitcast({stage}{{}}, {own_tag}{{}}, {v})" for v in args)
        self.line(
          f"auto {name} = vecops::vec::bitcast({own_tag}{{}}, {stage}{{}}, vecops::vec::local_{operation.kind}({stage}{{}}, {a}, {b}));"
        )
      elif operation.kind == "concat":
        self.line(f"auto {name} = vecops::vec::concat({own_tag}{{}}, {', '.join(args)});")
      else:
        parent = tag(network.operations[operation.operands[0]].lanes)
        self.line(f"auto {name} = vecops::vec::{operation.kind}({parent}{{}}, {args[0]});")
      names[index] = name
    store_tag = tag(network.rows)
    options = f", vecops::vec::opt::first({active_m})" if active_m is not None else ""
    for column, output in enumerate(network.outputs):
      pointer = self.pointer(plan.statement.destination, batch + (f"({ni} + {column})", mi))
      self.line(f"vecops::vec::store_convert({store_tag}{{}}, {pointer}, {names[output]}{options});")
    self.chunk_cache = before

  def sme_transpose(self, plan, batch):
    # A private state-owning leaf, containing only public vector/ZA primitives.
    # Pointer/extent validation and alias snapshots have already run outside.
    statement, dtype = plan.statement, plan.dtype
    self.line("#if defined(HAS_FIXED_SVE_BITS)")
    self.line("#if !defined(FIXED_STREAMING_SVE_BITS) || FIXED_SVE_BITS != FIXED_STREAMING_SVE_BITS")
    self.line('#error "SME with sized SVE vectors requires matching fixed VL and SVL; use VLA otherwise"')
    self.line("#endif")
    self.line("#endif")
    source = statement.value
    if source.op == "cast" and source.type.dtype == statement.destination.dtype:
      source = source.args[0]
    view = source.data
    m, n = map(self.scalar, statement.destination.shape[-2:][::-1])
    own_region = self.hardware_scope is None
    if own_region:
      session = self.unique("sme_session")
      self.line(f"vecops::execution::ExecutionSession {session};")
      self.opening(
        f"{session}.with_resources(vecops::execution::sme::Region{{}}, [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT"
      )
    tag, bits, lanes = (self.unique(v) for v in ("sme_tag", "sme_bits", "svl"))
    width = ELEMENT_BYTES[dtype]
    self.line(f"using {tag} = vecops::vec::ScalableTag<{CPP_TYPES[dtype]}, 0>;")
    self.line(f"using {bits} = vecops::vec::ScalableTag<std::uint{width * 8}_t, 0>;")
    self.line(f"const auto {lanes} = vecops::vec::size({tag}{{}});")
    mi, ni, r, c = (self.unique(v) for v in ("sme_m", "sme_n", "za_row", "za_col"))
    am, an = self.unique("active_m"), self.unique("active_n")
    self.opening(f"for (vecops::nint_t {mi} = 0; {mi} < ({m}); {mi} += {lanes})")
    self.line(f"const auto {am} = std::min<vecops::nint_t>({lanes}, ({m}) - {mi});")
    self.opening(f"for (vecops::nint_t {ni} = 0; {ni} < ({n}); {ni} += {lanes})")
    self.line(f"const auto {an} = std::min<vecops::nint_t>({lanes}, ({n}) - {ni});")
    pgm, pgn = self.unique("pgm"), self.unique("pgn")
    self.line(f"auto {pgm} = vecops::vec::mwhilelt({bits}{{}}, 0, {am});")
    self.line(f"auto {pgn} = vecops::vec::mwhilelt({bits}{{}}, 0, {an});")

    def write_rows(offsets):
      values = []
      for offset in offsets:
        pointer = self.pointer(view, batch + (ni, f"({mi} + {offset})"))
        value = self.unique("za_input")
        self.line(f"vecops::vec::Vec<{tag}> {value};")
        self.opening(f"if ({an} == {lanes})")
        self.line(f"{value} = vecops::vec::load_convert({tag}{{}}, {pointer});")
        self.close()
        self.opening("else")
        self.line(f"{value} = vecops::vec::load_convert({tag}{{}}, {pointer}, vecops::vec::opt::first({an}));")
        self.close()
        values.append(value)
      for offset, value in zip(offsets, values):
        self.line(
          f"vecops::vec::sme::write_hor<0>(static_cast<std::uint32_t>({offset}), {pgn}, vecops::vec::bitcast({bits}{{}}, {tag}{{}}, {value}));"
        )

    def read_columns(offsets):
      values = []
      for offset in offsets:
        value = self.unique("za_output")
        self.line(
          f"auto {value} = vecops::vec::bitcast({tag}{{}}, {bits}{{}}, vecops::vec::sme::read_ver<0>({bits}{{}}, static_cast<std::uint32_t>({offset}), {pgm}));"
        )
        values.append(value)
      for offset, value in zip(offsets, values):
        pointer = self.pointer(statement.destination, batch + (f"({ni} + {offset})", mi))
        self.opening(f"if ({am} == {lanes})")
        self.line(f"vecops::vec::store_convert({tag}{{}}, {pointer}, {value});")
        self.close()
        self.opening("else")
        self.line(f"vecops::vec::store_convert({tag}{{}}, {pointer}, {value}, vecops::vec::opt::first({am}));")
        self.close()

    for index, extent, emit in ((r, am, write_rows), (c, an, read_columns)):
      self.line(f"vecops::nint_t {index} = 0;")
      if plan.sme_pairs:
        self.opening(f"for (; {index} + 1 < {extent}; {index} += 2)")
        emit((index, f"({index} + 1)"))
        self.close()
      self.opening(f"for (; {index} < {extent}; ++{index})")
      emit((index,))
      self.close()
    self.close()
    self.close()
    if own_region:
      self.indent -= 1
      self.line("});")
