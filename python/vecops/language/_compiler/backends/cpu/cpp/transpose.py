# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Print prepared transpose DAGs. Algorithm selection lives in CPU planning."""

from .... import ir as I


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
        self.streaming_store(statement)
      self.line("#endif")
    else:
      self.streaming_store(statement)
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

  def sme_transpose(self, plan, batch):
    # A private state-owning leaf, containing only public vector/ZA primitives.
    # Pointer/extent validation and alias snapshots have already run outside.
    statement = plan.statement
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
    own_n, own_m = statement.destination.shape[-2:]
    self.line(f"const vecops::nint_t {extent_m}=({self.owner_integer(own_m, m)});")
    self.line(f"const vecops::nint_t {extent_n}=({self.owner_integer(own_n, n)});")
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
    if plan.sme_program is None:
      raise RuntimeError("missing prepared SME transpose program")
    # The checked boundary binds the only implementation. All traversal,
    # conversion and full/fringe schedules live in the source-parsed helper.
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
