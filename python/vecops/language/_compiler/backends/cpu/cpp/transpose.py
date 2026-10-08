# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
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
    self.opening(f"for (vecops::nint_t {mi} = {mb}; {mi} < {mend}; {mi} += {self.vector_size(tag)})")
    active = f"std::min<vecops::nint_t>({self.vector_size(tag)}, {mend} - {mi})"
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
    before = self.chunk_cache
    self.chunk_cache = {}

    def load(row, own_tag):
      name = self.unique("transpose_vec")
      coords = batch + (ni, f"({mi} + {row})")
      if active_m is not None:
        self.line(f"vecops::vec::Vec<{own_tag}> {name} = vecops::vec::zeros({own_tag}{{}});")
        self.opening(f"if ({row} < {active_m})")
      previous_full = getattr(self, "full_vector", False)
      # A complete source row has no inactive lanes, even in a row fringe.
      # Preserve this proof to select the same fused native load_convert.
      self.full_vector = True
      value = self.vector(
        source, own_tag, plan.statement.destination.shape, coords, str(network.columns), plan.source_axis
      )
      self.full_vector = previous_full
      self.line(f"{'auto ' if active_m is None else ''}{name} = {value};")
      if active_m is not None:
        self.close()
      return name

    store_tag, outputs = self.register_program(network, load, prefix="transpose")
    options = f", vecops::vec::opt::first({active_m})" if active_m is not None else ""
    for column, output in enumerate(outputs):
      pointer = self.pointer(plan.statement.destination, batch + (f"({ni} + {column})", mi))
      self.line(f"vecops::vec::store_convert({store_tag}{{}}, {pointer}, {output}{options});")
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
    dynamic_unroll = self.vector_ir.manager.options.get("cpu_transpose_sme_dynamic_unroll", False)
    if not isinstance(dynamic_unroll, bool):
      raise TypeError("cpu_transpose_sme_dynamic_unroll must be bool")
    m, n = map(self.scalar, statement.destination.shape[-2:][::-1])
    own_region = self.hardware_scope is None
    if own_region:
      session = self.unique("sme_session")
      self.line(f"vecops::execution::ExecutionSession {session};")
      self.opening_hardware_owner("sme", session, "[&](auto&)")
    # Const entry snapshots are still captured by reference in an outlined
    # streaming trampoline. ZA assembly clobbers memory, so accessing those
    # captures in every slice can reload the closure after each instruction.
    # Copy only this invocation's scalar geometry and origins into SSA inside
    # the owner. This is also fresh after an owned buffer's most recent resize.
    # Keep the copies local: another view/region may have different offsets or
    # may resize the same scratch storage before its next invocation.
    extent_m, extent_n = self.unique("transpose_extent_m"), self.unique("transpose_extent_n")
    source_base, target_base = self.unique("transpose_source_base"), self.unique("transpose_target_base")
    source_m, source_n = self.unique("transpose_source_m_stride"), self.unique("transpose_source_n_stride")
    target_m, target_n = self.unique("transpose_target_m_stride"), self.unique("transpose_target_n_stride")
    self.line(f"const vecops::nint_t {extent_m}=({m});")
    self.line(f"const vecops::nint_t {extent_n}=({n});")
    for name, view_ in ((source_base, view), (target_base, statement.destination)):
      self.line(
        f"const auto {name}=({extent_m}>0 && {extent_n}>0 ? "
        f"{self.pointer(view_, batch)} : {self.storage_pointer(view_.storage)});"
      )
    # Both callers already guard these contiguous axes before entering SME.
    # An outlined trampoline cannot otherwise inherit that unit-stride proof.
    # Keep only the independent row pitches dynamic, including negative ones.
    for name, stride in (
      (source_m, view.strides[-1]),
      (source_n, I.literal(1)),
      (target_m, I.literal(1)),
      (target_n, statement.destination.strides[-2]),
    ):
      self.line(f"const vecops::nint_t {name}={self.scalar(stride)};")
    m, n = extent_m, extent_n
    if plan.sme_program is not None:
      # The memory planner staged the executable DSL before emission. The
      # checked caller already established shape, unit axes and nonaliasing.
      # Bind its virtual boundary to fresh callback-owned SSA; no passes or
      # source parsing occur in this terminal printer.
      self.hardware_program(
        plan.sme_program,
        metadata={
          "sme_library.m": m,
          "sme_library.n": n,
          "sme_library.source_pitch": source_m,
          "sme_library.target_pitch": target_n,
        },
        pointers={"sme_library_input": source_base, "sme_library_output": target_base},
      )
      if own_region:
        self.indent -= 1
        self.line("});")
      return

    def source_pointer(row):
      return f"({source_base} + ({ni}) * {source_n} + ({mi} + {row}) * {source_m})"

    def target_pointer(column):
      return f"({target_base} + ({ni} + {column}) * {target_n} + ({mi}) * {target_m})"

    tag, bits, lanes = (self.unique(v) for v in ("sme_tag", "sme_bits", "svl"))
    width = ELEMENT_BYTES[dtype]
    self.line(f"using {tag} = vecops::vec::ScalableTag<{CPP_TYPES[dtype]}, 0>;")
    self.line(f"using {bits} = vecops::vec::ScalableTag<std::uint{width * 8}_t, 0>;")
    self.line(f"const auto {lanes} = vecops::vec::sme::streaming_lanes<{CPP_TYPES[dtype]}>();")
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
      if view.dtype == dtype:
        # Native memory already has the ZA tile's element representation.
        # Moving it through a Z register adds a load + MOVA dependency (and
        # extends Z liveness) without performing a numeric conversion. ZA
        # memory instructions predicate the tail directly, including its
        # zero fill. Reinterpret only the pointer: floats/integers of the
        # same storage type keep every payload bit, including NaNs.
        for offset in offsets:
          pointer = source_pointer(offset)
          self.line(
            f"vecops::vec::sme::load_hor<0>(static_cast<std::uint32_t>({offset}), {pgn}, "
            f"reinterpret_cast<const std::uint{width * 8}_t*>({pointer}));"
          )
        return
      values = []
      for offset in offsets:
        pointer = source_pointer(offset)
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
      if statement.destination.dtype == dtype:
        # With no store conversion, one ST1 ZA vertical slice replaces
        # read_ver + a vector store and needs no temporary Z register.
        for offset in offsets:
          pointer = target_pointer(offset)
          self.line(
            f"vecops::vec::sme::store_ver<0>(static_cast<std::uint32_t>({offset}), {pgm}, "
            f"reinterpret_cast<std::uint{width * 8}_t*>({pointer}));"
          )
        return
      values = []
      for offset in offsets:
        value = self.unique("za_output")
        self.line(
          f"auto {value} = vecops::vec::bitcast({tag}{{}}, {bits}{{}}, vecops::vec::sme::read_ver<0>({bits}{{}}, static_cast<std::uint32_t>({offset}), {pgm}));"
        )
        values.append(value)
      for offset, value in zip(offsets, values):
        pointer = target_pointer(offset)
        self.opening(f"if ({am} == {lanes})")
        self.line(f"vecops::vec::store_convert({tag}{{}}, {pointer}, {value});")
        self.close()
        self.opening("else")
        self.line(f"vecops::vec::store_convert({tag}{{}}, {pointer}, {value}, vecops::vec::opt::first({am}));")
        self.close()

    def static_half_output():
      # Only complete columns exist in this branch. Form their origin inside
      # the guard and use independent constant offsets, not a mutable cursor
      # chain or a pointer after the final column. Pairs have two named Z
      # values and retain the original read/read/store/store ordering.
      origin = self.unique("za_half_output_origin")
      self.line(f"const auto {origin} = {target_pointer('0')};")
      pitch = I.constant(statement.destination.strides[-2])
      for first in range(0, plan.sme_const_half_output_columns, 2):
        self.opening("")
        values = []
        for column in (first, first + 1):
          value = self.unique("za_half_output")
          self.line(
            f"const auto {value} = vecops::vec::bitcast({tag}{{}}, {bits}{{}}, "
            f"vecops::vec::sme::read_ver<0>({bits}{{}}, std::uint32_t{{{column}}}, {pgm}));"
          )
          values.append((column, value))
        for column, value in values:
          self.line(f"vecops::vec::store_convert({tag}{{}}, ({origin} + std::int64_t{{{column * pitch}}}), {value});")
        self.close()

    def static_half_full_input():
      # Full am/an and captured SVL are proved by the caller. Same-type
      # public loads move payload bits without numeric conversion. Form
      # this invocation's origin inside the guard and keep only two named
      # Z values live per load/load/write/write pair; no aggregate Vec/Mask.
      origin = self.unique("za_half_input_origin")
      self.line(f"const auto {origin} = {source_pointer('0')};")
      pitch = I.constant(view.strides[-1])
      for first in range(0, plan.sme_const_half_full_input_rows, 2):
        self.opening("")
        values = []
        for row in (first, first + 1):
          value = self.unique("za_half_input")
          self.line(f"const auto {value} = vecops::vec::load({tag}{{}}, ({origin} + std::int64_t{{{row * pitch}}}));")
          values.append((row, value))
        for row, value in values:
          self.line(
            f"vecops::vec::sme::write_hor<0>(std::uint32_t{{{row}}}, {pgn}, "
            f"vecops::vec::bitcast({bits}{{}}, {tag}{{}}, {value}));"
          )
        self.close()

    for index, extent, emit, pitch in (
      (r, am, write_rows, view.strides[-1]),
      (c, an, read_columns, statement.destination.strides[-2]),
    ):
      static_output = plan.sme_const_half_output_columns is not None and emit is read_columns
      static_input = plan.sme_const_half_full_input_rows is not None and emit is write_rows
      if static_input:
        self.opening(
          f"if ({am} == {lanes} && {an} == {lanes} && {lanes} == {plan.sme_const_half_full_input_rows}) "
          "/* full ZA input: named constant-pitch F32 pairs */"
        )
        static_half_full_input()
        self.close()
        self.opening("else /* typed input fringe: retain predicated ZA loads */")
      if static_output:
        self.opening(
          f"if ({am} == {lanes} && {an} == {lanes} && {lanes} == {plan.sme_const_half_output_columns}) "
          "/* full ZA output: constant F32-to-F16 addresses */"
        )
        static_half_output()
        self.close()
        self.opening("else /* typed output fringe: keep cold addresses local */")

      def unroll_hint(pitch=pitch, static_output=static_output):
        if static_output:
          # Do not clone every cold column address into the full owner's
          # frame. This hint is scoped to this bounded constant-pitch output
          # fallback; no input or public serial/vector unroll is changed.
          self.line("#if defined(__clang__)")
          self.line("#pragma clang loop unroll(disable)")
          self.line("#elif defined(__GNUC__)")
          self.line("#pragma GCC unroll 1")
          self.line("#endif")
        elif dynamic_unroll and I.constant(pitch) is None:
          # The <=8-slice auto-unroll can keep seven multiples of each
          # runtime pitch live and spill them. Bound only that experimental
          # dynamic-address case; constant pitches retain native scheduling.
          self.line("#if defined(__clang__)")
          self.line("#pragma clang loop unroll_count(2)")
          self.line("#elif defined(__GNUC__)")
          self.line("#pragma GCC unroll 2")
          self.line("#endif")

      self.line(f"vecops::nint_t {index} = 0;")
      if plan.sme_pairs:
        unroll_hint()
        self.opening(f"for (; {index} + 1 < {extent}; {index} += 2)")
        emit((index, f"({index} + 1)"))
        self.close()
      unroll_hint()
      self.opening(f"for (; {index} < {extent}; ++{index})")
      emit((index,))
      self.close()
      if static_output or static_input:
        self.close()
    self.close()
    self.close()
    if own_region:
      self.indent -= 1
      self.line("});")
