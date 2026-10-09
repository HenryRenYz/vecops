# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""SME packed-layout traffic expressed entirely through public ZA/vector ops.

The raw K-contiguous route stages source rows horizontally and drains packed
K groups vertically. Full word panels ping-pong between two ZA tile pairs;
fringes retain exact storage bits and zero-pad both axes. No Vec arrays,
opaque native packer calls, or escaping streaming values are generated.
"""

from dataclasses import dataclass, replace

from .... import ir as I
from ....dtypes import ELEMENT_BYTES
from ....ir.traversal import walk_unique
from ....operations import Trait
from ..capabilities import CPP_TYPES


@dataclass(frozen=True)
class _FP32PackTransform:
  # Frontend load-convert folding can fuse the final storage conversion into
  # the selected load. Keep its identity for alias snapshots, but replace a
  # separate FP32 computation leaf when evaluating its value from ZA bits.
  primary: I.Expr
  expression: I.Expr
  replacement: I.Expr


class SMEPackingEmitter:
  def sme_packing_program(self, packed):
    shape = tuple(I.constant(v) for v in (packed.shape if packed.side == "A" else packed.shape[::-1]))
    programs = self.prepared.sme_packing_programs
    return programs.get((packed.view.dtype, shape), programs.get(packed.view.dtype))

  def sme_raw_pack(self, source, packed, spatial, logical_k, format_code):
    """Emit a guarded raw or FP32 pointwise-DAG route.

    Capacity checks and alias snapshots belong to pack_matrix's checked
    preamble. The actual (possibly snapshotted) strides select this route,
    so a negative spatial stride is valid but a non-unit K stride is not.
    """
    raw = (
      source.op == "load"
      and not source.args
      and source.data.dtype == packed.view.dtype
      and source.type.dtype == source.data.dtype
    )
    transformed = None if raw else self.sme_pack_transform(source, packed)
    inactive, guards = (), ()
    if not raw and transformed is None:
      transformed, inactive, guards = self.sme_inactive_pack_transform(source, packed)
    if not raw and transformed is None:
      return False
    primary = source if raw else transformed.primary
    view, dtype = primary.data, packed.view.dtype
    if dtype not in {"float32", "float64", "float16", "bfloat16", "int8", "uint8"}:
      return False
    k_axis, s_axis = (1, 0) if packed.side == "A" else (0, 1)
    pointer = self.pointer(view)
    strides = [self.scalar(view.strides[axis]) for axis in (s_axis, k_axis)]
    saved = self.snapshots.get(id(primary))
    alias_guard = ""
    if saved is not None:
      snapshot, alias = saved
      if snapshot.dtype == view.dtype:
        pointer = f"({alias} ? {self.pointer(snapshot)} : {pointer})"
        strides = [
          f"({alias} ? {self.scalar(snapshot.strides[axis])} : {stride})"
          for axis, stride in zip((s_axis, k_axis), strides)
        ]
      else:
        # A folded load-convert snapshots its already-converted values. It
        # is not FP32 input storage, so the alias case must use the ordinary
        # snapshot-aware vector route, never bitcast a narrow snapshot as FP32.
        alias_guard = f" && !({alias})"
    self.line(
      "#if defined(HAS_SME_FA64) && (!defined(HAS_FIXED_SVE_BITS) || "
      "(defined(FIXED_STREAMING_SVE_BITS) && FIXED_SVE_BITS == FIXED_STREAMING_SVE_BITS))"
    )
    base, row_step, k_step, m, k = (
      self.unique(name)
      for name in ("sme_pack_input", "sme_pack_stride", "sme_pack_kstride", "sme_pack_m", "sme_pack_k")
    )
    self.line(f"const auto* const {base}={pointer};")
    self.line(f"const vecops::nint_t {row_step}=({strides[0]}), {k_step}=({strides[1]});")
    self.line(f"const vecops::nint_t {m}=({spatial}), {k}=({logical_k});")
    self.opening(
      f"if (vecops_dsl::packing_format<{CPP_TYPES[dtype]}, {format_code}> == 2 "
      f"&& {k_step} == 1{alias_guard}"
      + (" && !(" + " || ".join(guards) + ")" if guards else "")
      + f") /* {'raw' if raw else 'FP32 DAG'} K-contiguous SME ZA packing */"
    )
    session, scope = self.unique("sme_pack_session"), self.unique("sme_pack_scope")
    self.line(f"vecops::execution::ExecutionSession {session};")
    self.opening_hardware_owner("sme", session, f"[&](auto& {scope})")
    previous_scope, self.hardware_scope = self.hardware_scope, ("sme", scope)
    # The streaming trampoline can outline the callback. Its reference
    # captures are address-exposed to ZA's memory clobbers; copy them into
    # uncaptured locals once at callback entry rather than reload every row.
    held_base, held_stride, held_m, held_k = (
      self.unique(name)
      for name in ("sme_pack_base_value", "sme_pack_stride_value", "sme_pack_m_value", "sme_pack_k_value")
    )
    self.line(f"const auto* const {held_base}={base};")
    own_spatial, own_k = packed.shape if packed.side == "A" else packed.shape[::-1]
    unchanged_stride = (
      saved is None
      or saved[0].dtype != view.dtype  # This owner is guarded by !alias.
      or I.same_expr(saved[0].strides[s_axis], view.strides[s_axis])
    )
    if raw and self.sme_packing_program(packed) is not None:
      singleton_snapshot = (
        saved is not None and saved[0].dtype == view.dtype and I.same_expr(saved[0].strides[k_axis], view.shape[s_axis])
      )
      # The checked entry selects unit K stride. A dense B snapshot's K
      # stride is its spatial extent, so an active snapshot here implies a
      # singleton spatial axis whose stride is never used. Restore the
      # original stride too: outlining otherwise loses this guard's fact.
      original_stride = unchanged_stride or singleton_snapshot
      own_stride = (
        self.owner_integer(view.strides[s_axis], self.scalar(view.strides[s_axis])) if original_stride else row_step
      )
      self.line(
        f"const vecops::nint_t {held_m}={self.owner_integer(own_spatial, m)}, {held_k}={self.owner_integer(own_k, k)};"
      )
      if original_stride and I.same_expr(view.strides[s_axis], own_k):
        own_stride = held_k
      self.line(f"const vecops::nint_t {held_stride}={own_stride};")
    else:
      # Production families stay unchanged until the whole library driver
      # passes its performance gate, including the transformed-input route.
      own_stride = self.owner_integer(view.strides[s_axis], row_step) if unchanged_stride else row_step
      self.line(
        f"const vecops::nint_t {held_stride}={own_stride}, "
        f"{held_m}={self.owner_integer(own_spatial, m)}, {held_k}={self.owner_integer(own_k, k)};"
      )
    try:
      if raw:
        self.sme_raw_pack_body(packed, held_base, held_stride, held_m, held_k)
      else:
        if inactive:
          # Only this branch proves every listed replacement absent. Keep all
          # historical READ identities/snapshots; the original fallback below
          # must still see its exact conditional buffers and capture state.
          materialized, snapshots, cache = self.materialized, self.snapshots, self.chunk_cache
          self.materialized = {identity: value for identity, value in materialized.items() if identity not in inactive}
          self.snapshots, self.chunk_cache = dict(snapshots), {}
          try:
            self.sme_transform_pack_body(packed, transformed, held_base, held_stride, held_m, held_k)
          finally:
            self.materialized, self.snapshots, self.chunk_cache = materialized, snapshots, cache
        else:
          self.sme_transform_pack_body(packed, transformed, held_base, held_stride, held_m, held_k)
    finally:
      self.hardware_scope = previous_scope
    self.indent -= 1
    self.line("});")
    self.close()
    self.opening("else /* non-contiguous K preserves vector gather packing */")
    # The caller emits one shared fallback in all unsupported branches.
    self.line("#endif")
    return True

  def sme_inactive_pack_transform(self, source, packed):
    """Version only proven-inactive conditional replacements of a pure DAG.

    A conditional map is not an unconditional materialization boundary:
    its false arm already recomputes the original value in vector lowering.
    Select that exact arm once outside packing loops, rather than discard
    the boundary globally. Unknown/unconditional replacements, reductions,
    state, masks and intermediate narrowing retain their existing fallback.
    Snapshot guards, including captures made before another write, must all
    be false too; their identities are never removed or storage-coalesced.
    """
    expressions = tuple(walk_unique(source))
    inactive = tuple(id(value) for value in expressions if id(value) in self.materialized)
    if not inactive:
      return None, (), ()
    entries = [self.materialized[identity] for identity in inactive]
    entries.extend(self.snapshots[id(value)] for value in expressions if id(value) in self.snapshots)
    if any(
      not isinstance(entry, tuple) or len(entry) != 2 or not isinstance(entry[1], str) or not entry[1].strip()
      for entry in entries
    ):
      return None, (), ()
    materialized = self.materialized
    try:
      self.materialized = {identity: value for identity, value in materialized.items() if identity not in inactive}
      transformed = self.sme_pack_transform(source, packed)
    finally:
      self.materialized = materialized
    if transformed is None:
      return None, (), ()
    guards = tuple(dict.fromkeys(f"({entry[1]})" for entry in entries))
    return transformed, inactive, guards

  def sme_pack_transform(self, source, packed):
    """Select one FP32 storage leaf in a pure, streaming-safe pointwise DAG.

    No reduction/materialized/state boundary is crossed. Additional FP32
    broadcasts retain their own logical coordinates and alias snapshots.
    Widening 16-bit captures or stateful/unsupported operations stay on the
    ordinary vector path rather than call a non-streaming helper inside SM.
    """
    if packed.view.dtype not in {"bfloat16", "float16", "int8", "uint8"}:
      return None
    if id(source) in self.materialized:
      return None
    expression = source.args[0] if source.op == "cast" and source.type.dtype == packed.view.dtype else source
    converted_load = None
    if (
      expression is source
      and source.op == "load"
      and source.data.dtype == "float32"
      and source.type.dtype == packed.view.dtype
    ):
      # This is only the final FP32 -> packed-storage conversion. Do not
      # discard an intermediate conversion such as FP32 -> FP16 -> BF16.
      converted_load = source
      expression = self.retain_expression(replace(source, type=replace(source.type, dtype="float32")))
    if expression.type.dtype != "float32":
      return None
    loads, pending, seen = [], [expression], set()
    catalog = self.environment.operations
    boundaries = (
      Trait.REDUCTION | Trait.SCALAR_REDUCTION | Trait.MASK_REDUCTION | Trait.STATE_REFERENCE | Trait.REGISTER_TRANSFORM
    )
    while pending:
      value = pending.pop()
      if id(value) in seen:
        continue
      seen.add(id(value))
      if id(value) in self.materialized or catalog.has(value, boundaries):
        return None
      if value.op == "load":
        if value.args or value.data.dtype != "float32" or value.type.dtype != "float32":
          return None
        loads.append(value)
      elif isinstance(value.type, I.TileType):
        if value.type.dtype not in {"float32", "bool"} or not catalog.has(value, Trait.ELEMENTWISE):
          return None
      elif catalog.get(value).effects != "none":
        return None
      pending.extend(value.args)
    primary = next(
      (
        load
        for load in loads
        if len(load.data.shape) == 2 and all(I.same_expr(a, b) for a, b in zip(load.data.shape, packed.shape))
      ),
      None,
    )
    if primary is None:
      return None
    return _FP32PackTransform(converted_load if converted_load is not None else primary, expression, primary)

  def sme_pack_rewrite(self, expression, primary, reference):
    """Replace a selected load, preserving sharing and all other coordinates."""
    memo = {}

    def rewrite(value):
      if value is primary:
        return reference
      if id(value) not in memo:
        arguments = tuple(rewrite(argument) for argument in value.args)
        memo[id(value)] = (
          value
          if all(a is b for a, b in zip(arguments, value.args))
          else self.retain_expression(replace(value, args=arguments))
        )
      return memo[id(value)]

    return rewrite(expression)

  def sme_transform_pack_body(self, packed, transformed, base, stride, spatial, logical_k):
    primary, expression = transformed.replacement, transformed.expression
    dtype, kp = packed.view.dtype, max(1, 4 // ELEMENT_BYTES[packed.view.dtype])
    bits_tag, float_tag = self.matrix_tag("uint32"), self.matrix_tag("float32")
    narrow_tag = self.matrix_tag(dtype, power=-(kp.bit_length() - 1))
    lanes, panel, output = (self.unique(name) for name in ("sme_dag_lanes", "sme_dag_panel", "sme_dag_output"))
    self.line(f"const vecops::nint_t {lanes}=vecops::vec::sme::streaming_lanes<float>();")
    self.line(f"const vecops::nint_t {panel}=2*{lanes};")
    self.line(f"auto* {output}={self.pointer(packed.view)};")
    origin, kb = self.unique("sme_dag_origin"), self.unique("sme_dag_k")
    self.opening(f"for (vecops::nint_t {origin}=0; {origin}<({spatial}); {origin}+={panel})")
    self.opening(f"for (vecops::nint_t {kb}=0; {kb}<({logical_k}); {kb}+={lanes})")
    active_k, load_pg = self.unique("sme_dag_active_k"), self.unique("sme_dag_load_mask")
    self.line(f"const vecops::nint_t {active_k}=std::min<vecops::nint_t>({lanes},({logical_k})-{kb});")
    self.line(f"const auto {load_pg}=vecops::vec::mwhilelt({bits_tag}{{}},0,{active_k});")
    halves = []
    for tile in (0, 1):
      active, pg, row = (self.unique(name) for name in ("sme_dag_active_rows", "sme_dag_rows_mask", "sme_dag_row"))
      self.line(
        f"const vecops::nint_t {active}=std::max<vecops::nint_t>(0,std::min<vecops::nint_t>({lanes},({spatial})-{origin}-{tile}*{lanes}));"
      )
      self.line(f"const auto {pg}=vecops::vec::mwhilelt({bits_tag}{{}},0,{active});")
      self.opening(f"for (vecops::nint_t {row}=0; {row}<{active}; ++{row})")
      pointer = f"{base}+({origin}+{tile}*{lanes}+{row})*({stride})+{kb}"
      self.line(
        f"vecops::vec::sme::load_hor<{tile}>(static_cast<std::uint32_t>({row}),{load_pg},reinterpret_cast<const std::uint32_t*>({pointer}));"
      )
      self.close()
      halves.append((tile, active, pg))
    group = self.unique("sme_dag_group")
    self.opening(f"for (vecops::nint_t {group}=0; {group}<({active_k}+{kp - 1})/{kp}; ++{group})")
    for tile, active, pg in halves:
      values = []
      for column in range(kp):
        slot, bits, memory = (self.unique(name) for name in ("sme_dag_slot", "sme_dag_bits", "sme_dag_memory"))
        self.line(f"auto {slot}=vecops::vec::zeros({narrow_tag}{{}});")
        self.opening(f"if ({group}*{kp}+{column}<{active_k} && {active}>0) /* do not evaluate padded K DAG */")
        self.line(
          f"const auto {bits}=vecops::vec::sme::read_ver<{tile}>({bits_tag}{{}},static_cast<std::uint32_t>({group}*{kp}+{column}),{pg});"
        )
        self.line(f"const auto {memory}=vecops::vec::bitcast({float_tag}{{}},{bits_tag}{{}},{bits});")
        reference = self.retain_expression(I.Expr(primary.type, "vector_ref", data=memory))
        value = self.sme_pack_rewrite(expression, primary, reference)
        own_s, own_k = f"({origin}+{tile}*{lanes})", f"({kb}+{group}*{kp}+{column})"
        coords = (own_s, own_k) if packed.side == "A" else (own_k, own_s)
        previous_cache, self.chunk_cache = self.chunk_cache, {}
        processed = self.vector(value, float_tag, packed.shape, coords, active, 0 if packed.side == "A" else 1)
        # An affine/quantization transform need not preserve zero. Sanitize
        # absent rows before the narrowing conversion as well as after math;
        # padded K slots skip the whole expression (and its parameter loads).
        processed = self.zero_matrix_tail(float_tag, processed, active)
        self.line(f"{slot}=vecops::vec::convert({narrow_tag}{{}},{float_tag}{{}},{processed});")
        self.chunk_cache = previous_cache
        self.close()
        values.append(slot)
      out_tag, result = self.interleave_group(narrow_tag, values, dtype)
      self.line(f"vecops::vec::store({out_tag}{{}},{output},{result});")
      self.line(f"{output}+={lanes}*{kp};")
    self.close()
    self.close()
    self.close()

  def close_sme_raw_pack(self):
    self.line(
      "#if defined(HAS_SME_FA64) && (!defined(HAS_FIXED_SVE_BITS) || "
      "(defined(FIXED_STREAMING_SVE_BITS) && FIXED_SVE_BITS == FIXED_STREAMING_SVE_BITS))"
    )
    self.close()
    self.line("#endif")

  def sme_raw_pack_body(self, packed, base, stride, spatial, logical_k):
    program = self.sme_packing_program(packed)
    if program is not None:
      self.hardware_program(
        program,
        metadata={
          "sme_packing.m": spatial,
          "sme_packing.k": logical_k,
          "sme_packing.pitch": stride,
        },
        pointers={"sme_packing_input": base, "sme_packing_output": self.pointer(packed.view)},
      )
      return
    width = ELEMENT_BYTES[packed.view.dtype]
    bits = f"std::uint{width * 8}_t"
    tag = self.matrix_tag(f"uint{width * 8}")
    lanes, panel, chunk, output = (
      self.unique(name) for name in ("sme_pack_lanes", "sme_pack_panel", "sme_pack_chunk", "sme_pack_output")
    )
    kp = max(1, 4 // width)
    self.line(f"const vecops::nint_t {lanes}=vecops::vec::sme::streaming_lanes<{bits}>();")
    self.line(
      f"const vecops::nint_t {panel}=2*vecops::vec::sme::streaming_lanes<{'std::uint64_t' if width == 8 else 'std::uint32_t'}>();"
    )
    self.line(f"const vecops::nint_t {chunk}=({lanes}/{kp})*{kp};")
    self.line(f"auto* {output}=reinterpret_cast<{bits}*>({self.pointer(packed.view)});")
    origin, kb = self.unique("sme_pack_panel_origin"), self.unique("sme_pack_k_origin")
    self.opening(f"for (vecops::nint_t {origin}=0; {origin}<({spatial}); {origin}+={panel})")
    self.line(f"vecops::nint_t {kb}=0;")
    if width <= 4:
      self.opening(f"if (({spatial})-{origin}>={panel}) /* full ZA32 panel: two-pair pipeline */")
      self.sme_pack_full_words(base, stride, logical_k, origin, chunk, output, bits, kb)
      self.close()
    self.opening(f"for (; ({logical_k})-{kb}>={chunk}; {kb}+={chunk})")
    self.sme_pack_fringe(base, stride, spatial, origin, kb, chunk, lanes, panel, tag, output, width, kp)
    self.close()
    self.opening(f"if ({kb}<({logical_k})) /* one masked K chunk */")
    self.sme_pack_fringe(
      base,
      stride,
      spatial,
      origin,
      kb,
      f"({logical_k})-{kb}",
      lanes,
      panel,
      tag,
      output,
      width,
      kp,
    )
    self.close()
    self.close()

  def sme_pack_full_words(self, base, stride, logical_k, origin, chunk, output, bits, consumed):
    tag = self.matrix_tag("uint32")
    lanes, pg, chunks, words = (
      self.unique(name) for name in ("sme_pack_words", "sme_pack_all", "sme_pack_chunks", "sme_pack_word_output")
    )
    self.line(f"const vecops::nint_t {lanes}=vecops::vec::sme::streaming_lanes<std::uint32_t>();")
    self.line(f"const auto {pg}=vecops::vec::mtrue({tag}{{}});")
    self.line(f"const vecops::nint_t {chunks}=({logical_k})/{chunk};")
    self.opening(f"if ({chunks}>0) /* pipeline prime */")
    self.line(f"auto* {words}=reinterpret_cast<std::uint32_t*>({output});")

    def transfer(load_base, kb, store_base=None, prefetch=None):
      row = self.unique("sme_pack_word_row")
      self.opening(f"for (vecops::nint_t {row}=0; {row}<{lanes}; ++{row})")
      for tile in (0, 1):
        pointer = f"({base}+(({origin})+{tile}*{lanes}+{row})*({stride})+({kb}))"
        if prefetch is not None:
          self.opening(f"if ({prefetch})")
          self.line(f"__builtin_prefetch({pointer}+4*{chunk},0,3);")
          self.close()
        self.line(
          f"vecops::vec::sme::load_hor<{load_base + tile}>(static_cast<std::uint32_t>({row}),"
          f"{pg},reinterpret_cast<const std::uint32_t*>({pointer}));"
        )
      if store_base is not None:
        for tile in (0, 1):
          self.line(
            f"vecops::vec::sme::store_ver<{store_base + tile}>(static_cast<std::uint32_t>({row}),{pg},{words});"
          )
          self.line(f"{words}+={lanes};")
      self.close()

    transfer(0, "0", prefetch=f"{chunks}>4")
    index = self.unique("sme_pack_chunk_index")
    self.opening(f"for (vecops::nint_t {index}=1; {index}<{chunks}; ++{index})")
    self.opening(f"if ({index}&1)")
    transfer(2, f"{index}*{chunk}", 0, f"{chunks}-{index}>4")
    self.close()
    self.opening("else")
    transfer(0, f"{index}*{chunk}", 2, f"{chunks}-{index}>4")
    self.close()
    self.close()
    self.opening(f"if ({chunks}&1) /* pipeline drain */")
    self.sme_pack_drain_words(0, pg, lanes, words)
    self.close()
    self.opening("else")
    self.sme_pack_drain_words(2, pg, lanes, words)
    self.close()
    self.line(f"{output}=reinterpret_cast<{bits}*>({words});")
    self.line(f"{consumed}={chunks}*{chunk};")
    self.close()

  def sme_pack_drain_words(self, base, pg, lanes, output):
    group = self.unique("sme_pack_word_group")
    self.opening(f"for (vecops::nint_t {group}=0; {group}<{lanes}; ++{group})")
    for tile in (0, 1):
      self.line(f"vecops::vec::sme::store_ver<{base + tile}>(static_cast<std::uint32_t>({group}),{pg},{output});")
      self.line(f"{output}+={lanes};")
    self.close()

  def sme_pack_fringe(self, base, stride, spatial, origin, kb, active_k, lanes, panel, tag, output, width, kp):
    active, pg, load_pg = (
      self.unique(name) for name in ("sme_pack_active_rows", "sme_pack_rows_mask", "sme_pack_k_mask")
    )
    self.line(
      f"const vecops::nint_t {active}=std::min<vecops::nint_t>({lanes},std::min<vecops::nint_t>({panel},({spatial})-{origin}));"
    )
    self.line(f"const auto {pg}=vecops::vec::mwhilelt({tag}{{}},0,{active});")
    self.line(f"const auto {load_pg}=vecops::vec::mwhilelt({tag}{{}},0,({active_k}));")

    def stage(tile, active_rows, offset):
      row = self.unique("sme_pack_row")
      self.opening(f"for (vecops::nint_t {row}=0; {row}<({active_rows}); ++{row})")
      pointer = f"{base}+(({origin})+({offset})+{row})*({stride})+({kb})"
      self.line(
        f"vecops::vec::sme::load_hor<{tile}>(static_cast<std::uint32_t>({row}),{load_pg},"
        f"reinterpret_cast<const std::uint{width * 8}_t*>({pointer}));"
      )
      self.close()

    stage(0, active, "0")
    if width >= 4:
      active2, pg2 = self.unique("sme_pack_active_rows"), self.unique("sme_pack_rows_mask")
      self.line(
        f"const vecops::nint_t {active2}=std::max<vecops::nint_t>(0,std::min<vecops::nint_t>({lanes},({spatial})-{origin}-{lanes}));"
      )
      self.line(f"const auto {pg2}=vecops::vec::mwhilelt({tag}{{}},0,{active2});")
      stage(1, active2, lanes)
    group = self.unique("sme_pack_k_group")
    self.opening(f"for (vecops::nint_t {group}=0; {group}<(({active_k})+{kp - 1})/{kp}; ++{group})")
    if width >= 4:
      for tile, mask in ((0, pg), (1, pg2)):
        value = self.unique("sme_pack_column")
        self.line(
          f"const auto {value}=vecops::vec::sme::read_ver<{tile}>({tag}{{}},static_cast<std::uint32_t>({group}),{mask});"
        )
        self.line(f"vecops::vec::store({tag}{{}},{output},{value});")
        self.line(f"{output}+={lanes};")
    else:
      values = []
      for column in range(kp):
        value = self.unique("sme_pack_column")
        self.line(
          f"const auto {value}=vecops::vec::sme::read_ver<0>({tag}{{}},static_cast<std::uint32_t>({group}*{kp}+{column}),{pg});"
        )
        values.append(value)
      self.sme_pack_zip_words(tag, values, output, lanes, width)
    self.close()

  def sme_pack_zip_words(self, tag, values, output, lanes, width):
    # Extract each native word directly from the public interleave expression.
    # No multi-word Vec variable/array is emitted; on SVE these expressions
    # become ZIP1/ZIP2 without a tuple round-trip. local_interleave_* instead
    # zips inside 16-byte blocks and would corrupt panels at larger SVLs.
    def zip_word(own_tag, first, second, index):
      pair = f"vecops::vec::Twice<{own_tag}>"
      return f"vecops::vec::get_word<{index}>({pair}{{}},vecops::vec::interleave({pair}{{}},{first},{second}))"

    if width == 2:
      for index in (0, 1):
        value = self.unique("sme_pack_zip")
        self.line(f"const auto {value}={zip_word(tag, values[0], values[1], index)};")
        self.line(f"vecops::vec::store({tag}{{}},{output},{value});")
        self.line(f"{output}+={lanes};")
      return
    pair_tag = self.matrix_tag("uint16")
    pairs = []
    for first, second in ((values[0], values[1]), (values[2], values[3])):
      pair = self.unique("sme_pack_pair")
      self.line(f"const auto {pair}=vecops::vec::bitcast({pair_tag}{{}},{tag}{{}},{zip_word(tag, first, second, 0)});")
      pairs.append(pair)
    for index in (0, 1):
      value = self.unique("sme_pack_quad")
      self.line(f"const auto {value}={zip_word(pair_tag, pairs[0], pairs[1], index)};")
      self.line(f"vecops::vec::store({pair_tag}{{}},reinterpret_cast<std::uint16_t*>({output}),{value});")
      self.line(f"{output}+={lanes};")
