# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Lexical hardware ownership; all throwing preparation precedes entry."""

import json

from .... import ir as I
from ....ir.traversal import nodes, walk_unique
from ....dtypes import ELEMENT_BYTES
from .hardware_sme import SMEInstructionEmitter


class HardwareEmitter(SMEInstructionEmitter):
  def snapshot_hardware_inputs(self, program):
    """Snapshot invariant references inside the streaming callback.

    SME memory clobbers otherwise force LLVM to reload reference-captured
    pointers and loop metadata on every instruction. Region-local immutable
    values keep their SSA identity without changing storage or ownership.
    No allocation or scalar mutation is legal during this lexical interval.
    """
    statements = tuple(nodes(program))
    defined = {s.target for s in statements if isinstance(s, (I.Assign, I.For))}
    roots, views = [], []
    for statement in statements:
      if isinstance(statement, I.For):
        roots.extend((statement.start, statement.stop, statement.step))
      elif isinstance(statement, I.Assign):
        roots.append(statement.value)
      elif isinstance(statement, I.If):
        roots.append(statement.condition)
      elif isinstance(statement, I.ArchitectureCall):
        if statement.vector is not None:
          roots.append(statement.vector.extent)
        if statement.slice_index is not None:
          roots.append(statement.slice_index)
        if statement.view is not None:
          views.append(statement.view)
      elif isinstance(statement, I.Prefetch):
        views.append(statement.view)
    for view in views:
      roots.extend((view.offset, *view.shape, *view.strides))
    references = {}
    for root in roots:
      for value in walk_unique(root):
        if (isinstance(value.type, I.ScalarType) and value.op in {"raw", "symbol", "local", "frozen"}
            and not (value.op in {"local", "frozen"} and value.data in defined)):
          references.setdefault((value.op, value.data), value)
    for key, value in sorted(references.items()):
      owned = self.unique("hardware_metadata")
      self.line(f"const auto {owned} = {self.scalar(value)};")
      self.hardware_scalar_bindings[key] = owned
    for storage in sorted({view.storage for view in views}):
      owned = self.unique("hardware_pointer")
      self.line(f"auto* const {owned} = {self.storage_pointer(storage)};")
      self.parameter_pointers[storage] = owned

  def owner_integer(self, value, fallback):
    """Keep known scalar metadata literal inside an outlined state owner.

    Reference captures lose the caller's constant/range facts across the
    streaming trampoline. Only restore exact IR constants; runtime metadata
    and invocation-dependent snapshots retain their callback-local copies.
    """
    return self.scalar(value) if type(I.constant(value)) is int else fallback

  def hardware_program(self, program, *, metadata, pointers):
    """Bind a prepared instruction program to invocation-owned ABI values."""
    previous = self.parameter_metadata, self.parameter_pointers, self.hardware_scope
    self.parameter_metadata = {**self.parameter_metadata, **metadata}
    self.hardware_scope = ("sme", None)
    self.opening("/* library instruction program */")
    try:
      owned_pointers = {}
      for name, pointer in pointers.items():
        owned = self.unique("hardware_pointer")
        # A reference-captured ABI pointer is address-exposed across the
        # streaming trampoline. ZA memory clobbers must not force its reload
        # at each instruction. Keep an uncaptured, region-local pointer SSA
        # value; this preserves const pointees as well as writable outputs.
        self.line(f"auto* const {owned} = {pointer};")
        owned_pointers[name] = owned
      self.parameter_pointers = {**self.parameter_pointers, **owned_pointers}
      self.hardware_block(program)
    finally:
      self.close()
      self.parameter_metadata, self.parameter_pointers, self.hardware_scope = previous

  def hardware_region(self, region, future):
    kind = region.kind
    guard = "HAS_SME_FA64" if kind == "sme" else "HAS_AMX_TILE"
    self.line(f"#if defined({guard})")
    plan = self.prepared.hardware[id(region)]
    for condition, message in plan.checks:
      self.line(f"if (!({self.scalar(condition)})) throw std::invalid_argument({json.dumps(message)});")
    for view, active in plan.layouts:
      self.opening(f"if ({self.scalar(active)})")
      self.line(
        f"vecops_dsl::check_writable<{len(view.shape)}>({{{', '.join(self.scalar(v) for v in view.shape)}}}, {{{', '.join(self.scalar(v) for v in view.strides)}}});"
      )
      self.close()
    for view in plan.writes:
      root = self.storage_views.get(view.storage)
      if root is not None:
        for identity in set(self.active_loads) & set(future):
          self.snapshot(self.active_loads[identity], root, True)
    # Generic fusion may group stores; restore instruction order for a region
    # that explicitly owns destructive architectural state.
    statements = []
    for statement in region.body.statements:
      statements.extend(statement.stores if isinstance(statement, I.ElementwiseGroup) else (statement,))
    if kind == "sme":
      destinations = [s.destination for s in statements if isinstance(s, I.Store)]
      for statement in statements:
        if isinstance(statement, I.Store):
          plan = self.prepared.transposes.get(id(statement))
          if plan is None or not plan.sme or len(statement.destination.shape) != 2:
            raise TypeError("SME region only admits simple whole-view transposes")
          self.check_store(statement)
          loads = tuple(I.loads(statement.value))
          guards = [f"({self.scalar(statement.destination.strides[-1])}) == 1"]
          guards += [f"({self.scalar(load.data.strides[plan.source_axis])}) == 1" for load in loads]
          guards += [f"!({self.overlap(load.data, destination)})" for load in loads for destination in destinations]
          self.line(
            "if (!("
            + " && ".join(guards)
            + ')) throw std::invalid_argument("explicit SME region requires unit column strides and independent nonoverlapping operands");'
          )
    else:
      configuration = self.unique("tilecfg")
      self.line(f"vecops::execution::amx::TileConfiguration {configuration};")
      for index, (rows, columns) in enumerate(region.configuration):
        self.line(f"{configuration}.rows[{index}] = {rows};")
        self.line(f"{configuration}.column_bytes[{index}] = {columns};")
    session, scope = self.unique("hardware_session"), self.unique("hardware_scope")
    self.line(f"vecops::execution::ExecutionSession {session};")
    self.opening_hardware_owner(kind, session, f"[&](auto& {scope})")
    previous, self.hardware_scope = self.hardware_scope, (kind, scope)
    if kind == "amx":
      self.opening(f"{scope}.with_configuration({configuration}, [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT")
    prior_pointers, prior_scalars = self.parameter_pointers, self.hardware_scalar_bindings
    if kind == "sme" and not any(isinstance(s, I.Store) for s in nodes(I.Block(statements))):
      self.parameter_pointers = dict(prior_pointers)
      self.hardware_scalar_bindings = dict(prior_scalars)
      self.snapshot_hardware_inputs(I.Block(statements))
    try:
      self.hardware_block(I.Block(statements))
    finally:
      self.parameter_pointers, self.hardware_scalar_bindings = prior_pointers, prior_scalars
    if kind == "amx":
      self.indent -= 1
      self.line("});")
    self.hardware_scope = previous
    self.indent -= 1
    self.line("});")
    self.line("#else")
    self.line(f'#error "explicit {kind.upper()} region requires target feature {guard}"')
    self.line("#endif")

  def hardware_block(self, block):
    """Print already-validated loops without ordinary memory preparation."""
    for statement in block.statements:
      if isinstance(statement, I.For):
        step = I.constant(statement.step)
        comparison = "<" if step is None or step > 0 else ">"
        self.line("#if defined(__clang__)")
        self.line(f"#pragma clang loop unroll_count({statement.unroll})")
        self.line("#elif defined(__GNUC__)")
        self.line(f"#pragma GCC unroll {statement.unroll}")
        self.line("#endif")
        self.opening(
          f"for (vecops::nint_t {statement.target} = {self.scalar(statement.start)}; {statement.target} {comparison} ({self.scalar(statement.stop)}); {statement.target} += {self.scalar(statement.step)})"
        )
        self.hardware_block(statement.body)
        self.close()
      elif isinstance(statement, I.If):
        self.opening(f"if ({self.scalar(statement.condition)})")
        self.hardware_block(statement.body)
        self.close()
        if statement.otherwise.statements:
          self.opening("else")
          self.hardware_block(statement.otherwise)
          self.close()
      elif isinstance(statement, I.Assign):
        self.line(f"const auto {statement.target} = {self.scalar(statement.value)};")
      elif isinstance(statement, I.Prefetch):
        self.prefetch(statement)
      elif isinstance(statement, I.Store):
        plan = self.prepared.transposes[id(statement)]
        self.emitted_regions.append(plan)
        self.sme_transpose(plan, ())
      elif isinstance(statement, I.ArchitectureCall):
        if self.emit_sme_instruction(statement):
          pass
        elif statement.operation == "sme.zero_za":
          self.line("vecops::vec::sme::zero_za();")
        elif statement.operation == "amx.tdp":
          from ..capabilities import CPP_TYPES

          acc, a, b = statement.operands
          self.line(
            f"vecops::vec::amx::dot<{CPP_TYPES[acc.dtype]}, {CPP_TYPES[a.dtype]}, {CPP_TYPES[b.dtype]}, {acc.index}, {a.index}, {b.index}>();"
          )
        else:
          op = statement.operation.split(".")[1]
          args = (
            ""
            if statement.view is None
            else f"{self.pointer(statement.view)}, ({self.scalar(statement.view.strides[0])}) * {ELEMENT_BYTES[statement.view.dtype]}"
          )
          self.line(f"vecops::vec::amx::{op}<{statement.tile}>({args});")
