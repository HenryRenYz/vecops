"""Lexical hardware ownership; all throwing preparation precedes entry."""

from .... import ir as I
from ....dtypes import ELEMENT_BYTES
from .hardware_sme import SMEInstructionEmitter


class HardwareEmitter(SMEInstructionEmitter):
  def hardware_region(self, region, future):
    kind = region.kind
    # Generic fusion may group stores; restore instruction order for a region
    # that explicitly owns destructive architectural state.
    statements = []
    for statement in region.body.statements:
      statements.extend(statement.stores if isinstance(statement, I.ElementwiseGroup) else (statement,))
    if kind == "sme":
      destinations = [s.destination for s in statements if isinstance(s, I.Store)]
      for statement in statements:
        if isinstance(statement, I.ArchitectureCall):
          self.check_sme_instruction(statement)
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
      for statement in statements:
        view = statement.view
        if view is None:
          continue
        rows, columns = region.configuration[statement.tile]
        size = ELEMENT_BYTES[view.dtype]
        guards = (
          f"({self.scalar(view.shape[0])}) >= {rows}",
          f"({self.scalar(view.shape[1])}) >= {(columns + size - 1) // size}",
          f"({self.scalar(view.strides[1])}) == 1",
        )
        self.line(
          "if (!("
          + " && ".join(guards)
          + ')) throw std::invalid_argument("AMX memory view is smaller than its configured tile or is not row-contiguous");'
        )
        if statement.operation == "amx.store":
          self.line(
            f"vecops_dsl::check_writable<2>({{{', '.join(self.scalar(v) for v in view.shape)}}}, {{{', '.join(self.scalar(v) for v in view.strides)}}});"
          )
          for identity in set(self.active_loads) & set(future):
            self.snapshot(self.active_loads[identity], view, True)
    session, scope = self.unique("hardware_session"), self.unique("hardware_scope")
    guard = "HAS_SME_FA64" if kind == "sme" else "HAS_AMX_TILE"
    self.line(f"#if defined({guard})")
    self.line(f"vecops::execution::ExecutionSession {session};")
    self.opening_hardware_owner(kind, session, f"[&](auto& {scope})")
    previous, self.hardware_scope = self.hardware_scope, (kind, scope)
    if kind == "amx":
      self.opening(f"{scope}.with_configuration({configuration}, [&](auto&) VECOPS_INLINE_LAMBDA_NOEXCEPT")
    for statement in statements:
      if isinstance(statement, I.Store):
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
    if kind == "amx":
      self.indent -= 1
      self.line("});")
    self.hardware_scope = previous
    self.indent -= 1
    self.line("});")
    self.line("#else")
    self.line(f'#error "explicit {kind.upper()} region requires target feature {guard}"')
    self.line("#endif")
