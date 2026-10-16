# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""C++ emission: emitter."""

from __future__ import annotations

from .....expr import Dynamic
from .....types import TensorType
from .... import ir as I
from ....ir.traversal import nodes
from ....operations import use_catalog
from ..capabilities import CPP_TYPES
from .collectives import CollectiveEmitter, team_statements
from .control import ControlEmitter
from .expressions import ExpressionsEmitter
from .grid import GridEmitter
from .hardware import HardwareEmitter
from .memory import MemoryEmitter
from .packed_memory import PackedMemoryEmitter
from .packing_support import PREAMBLE as _PACKING_PREAMBLE
from .primitives import PrimitivesEmitter
from .printing import PrintEmitter
from .registers import RegistersEmitter
from .state import StateEmitter
from .support import _PREAMBLE, _SME_MATH_POLICY
from .transpose import TransposeEmitter
from .vectors import VectorsEmitter


class CPUEmitter(
  ExpressionsEmitter,
  MemoryEmitter,
  VectorsEmitter,
  ControlEmitter,
  TransposeEmitter,
  HardwareEmitter,
  PackedMemoryEmitter,
  GridEmitter,
  StateEmitter,
  PrimitivesEmitter,
  CollectiveEmitter,
  PrintEmitter,
  RegistersEmitter,
):
  def __init__(self, prepared):
    self.prepared = prepared
    self.environment = prepared.vectors.manager.options["_compiler_environment"]
    vector_ir = prepared.vectors
    self.vector_ir = vector_ir
    self.emitted_regions = []
    self.ir = vector_ir.kernel
    self.lines = []
    self.indent = 0
    self.counter = 0
    self.chunk_cache = {}
    self.allocated = set()
    self.tag_dtypes = {}
    self.fixed_tag_lanes = {}
    self.tag_geometry = {}
    self.active_loads = {}
    self.snapshots = {}
    self.storage_views = {}
    # Only ABI tensor descriptors are immutable for the invocation. Scratch
    # Buffer addresses can change on resize and must never enter these maps.
    self.parameter_metadata = {}
    self.parameter_pointers = {}
    self.materialized = {}
    self.reduction_materialized = {}
    self.hardware_scope = None
    self.hardware_scalar_bindings = {}
    # Only singly defined, immutable geometry may be reconstructed across
    # an outlined architectural owner. Runtime metadata stays snapshotted.
    self.hardware_geometry_definitions = prepared.immutable_geometry
    self.uses_streaming_math_policy = False
    self.buffer_declarations = {}
    storage = prepared.storage
    self.branch_buffer_scopes = storage.scopes if storage is not None else {}
    self.scratch_slots = storage.slots if storage is not None else {}
    self.active_buffer_branches = {}
    self.parallel_depth = 0
    self.verified_grid = None
    # Identity-indexed emission caches also see transient panel/proxy Exprs.
    # Hold strong references until emission ends: CPython can otherwise reuse
    # an id and attach another ISA branch's out-of-scope scratch/coordinates.
    self.expression_lifetimes = {}
    self.uses_executor = bool(prepared.grids) or any(isinstance(s, I.Team) for s in team_statements(self.ir.body))
    self.uses_workspace = any(
      isinstance(s, I.WorkspaceAllocate) or isinstance(s, I.Expr) and s.op == "worker_capacity"
      for s in nodes(self.ir.body)
    )
    self.traces_memory = vector_ir.manager.options.get("debug_memory", False)

  def retain_expression(self, value):
    self.expression_lifetimes[id(value)] = value
    return value

  def vector_size(self, tag):
    """Constant captured lanes without binding Vec to a fixed-width type."""
    streaming = self.hardware_scope is not None and self.hardware_scope[0] == "sme"
    domain = ", true" if streaming else ""
    return f"vecops_dsl::vector_size<{tag}{domain}>({tag}{{}})"

  def allocate_buffer(self, dtype, name, count):
    # Invocation-local scratch reuse across serial loops, never process/global
    # state. Each textual allocation site has its own independently sized slot.
    owner = self.active_buffer_branches.get(self.branch_buffer_scopes.get(name))
    slot = self.scratch_slots.get(name, name)
    if owner is None:
      declarations, indent = self.buffer_declarations, 1
    else:
      indent, declarations = owner
    declarations.setdefault(slot, f"{'  ' * indent}vecops_dsl::Buffer<{CPP_TYPES[dtype]}> {slot};")
    if slot != name:
      declarations[name] = f"{'  ' * indent}auto& {name} = {slot}; /* nonoverlapping scratch lifetimes */"
    self.line(f"{name}.resize({count});")

  def branch_block(self, block):
    """Construct branch-private descriptors only when that branch executes."""
    insertion = len(self.lines)
    declarations = {}
    key = id(block)
    previous = self.active_buffer_branches.get(key)
    self.active_buffer_branches[key] = self.indent, declarations
    self.block(block)
    self.lines[insertion:insertion] = list(declarations.values())
    if previous is None:
      del self.active_buffer_branches[key]
    else:
      self.active_buffer_branches[key] = previous

  def unique(self, prefix="v"):
    self.counter += 1
    return f"dsl_c_{prefix}_{self.counter}"

  def line(self, text):
    self.lines.append("  " * self.indent + text)

  def opening(self, text):
    self.line(text + " {")
    self.indent += 1

  def opening_hardware_owner(self, kind, session, callback):
    """Own architectural state and select its translation-unit call policy.

    ISA availability alone does not imply execution in streaming mode. Only
    an emitted SME owner requires vector-only exceptional helpers inline.
    Select the policy before any includes, including optional debug headers.
    """
    if kind == "sme":
      self.uses_streaming_math_policy = True
    self.opening(
      f"{session}.with_resources(vecops::execution::{kind}::Region{{}}, {callback} VECOPS_INLINE_LAMBDA_NOEXCEPT"
    )

  def close(self):
    self.indent -= 1
    self.line("}")

  def generate(self):
    self.line("#include <bit>")
    self.line('#include "vecops/util/ScalarConvert.h"')
    if (self.uses_executor or self.traces_memory) and not self.uses_workspace:
      self.line("#define VECOPS_KERNEL_EXECUTION_CONTEXT 1")
    parameters = []
    for name, annotation in self.ir.parameters.items():
      if isinstance(annotation, TensorType):
        parameters.append(f"vecops::tensor::TensorLike auto {name}")
        shape = tuple(
          I.Expr(I.ScalarType("int64"), "raw", data=f"{name}.size({axis})") for axis in range(len(annotation.shape))
        )
        strides = tuple(
          I.Expr(I.ScalarType("int64"), "raw", data=f"{name}.stride({axis})") for axis in range(len(annotation.shape))
        )
        from ....dtypes import dtype_name

        self.storage_views[name] = I.View(name, dtype_name(annotation.dtype), shape, strides, I.literal(0))
      else:
        from ....dtypes import dtype_name

        parameters.append(f"{CPP_TYPES[dtype_name(annotation)]} {name}")
    if self.uses_workspace:
      parameters.insert(0, "vecops::execution::WorkspaceContext& dsl_workspace")
    elif self.uses_executor or self.traces_memory:
      parameters.insert(0, "const VecopsExecutionContext* dsl_grid_context")
    self.opening(f"void __kernel__({', '.join(parameters)})")
    snapshot_parameters = self.vector_ir.manager.options.get("cpu_parameter_metadata_snapshot", True)
    if not isinstance(snapshot_parameters, bool):
      raise TypeError("cpu_parameter_metadata_snapshot must be bool")
    if snapshot_parameters:
      self.line("/* invocation-invariant ABI tensor metadata, not tensor contents */")
      # Materialize data pointers only for planned value captures. Keeping
      # ordinary inline accesses on the descriptor avoids an unnecessary
      # pointer/metadata materialization interaction in native optimization.
      captured_pointers = frozenset(name for plan in self.prepared.private_captures.values() for name in plan.pointers)
      for name, annotation in self.ir.parameters.items():
        if not isinstance(annotation, TensorType):
          continue
        if name in captured_pointers:
          pointer = self.unique(f"parameter_{name}_data")
          self.line(f"[[maybe_unused]] const auto {pointer} = {name}.data();")
          self.parameter_pointers[name] = pointer
        for operation in ("size", "stride"):
          dimensions = annotation.shape if operation == "size" else annotation.concrete_strides
          for axis, dimension in enumerate(dimensions):
            expression = f"{name}.{operation}({axis})"
            local = self.unique(f"parameter_{name}_{operation}_{axis}")
            # Runtime-index access reads PackedStorage directly. Reify a
            # Dynamic axis through its existing meta::Value conversion so
            # native optimizers retain the validated bounds/alignment that
            # kernel facts already used. Const axes already fold as written.
            value = (
              f"static_cast<std::int64_t>({name}.template {operation}<{axis}>())"
              if isinstance(dimension, Dynamic)
              else expression
            )
            self.line(f"[[maybe_unused]] const auto {local} = {value};")
            self.parameter_metadata[expression] = local
    if self.traces_memory:
      self.line(
        "const auto* dsl_allocation_observer = dsl_workspace.allocation_observer();"
        if self.uses_workspace
        else "const auto* dsl_allocation_observer = dsl_grid_context && dsl_grid_context->struct_size >= offsetof(VecopsExecutionContext,allocation_observer) + sizeof(dsl_grid_context->allocation_observer) ? dsl_grid_context->allocation_observer : nullptr;"
      )
    if self.vector_ir.manager.options.get("debug_timeline", False):
      import json

      self.line(f"vecops_dsl::TraceScope dsl_kernel_trace(dsl_allocation_observer,{json.dumps(self.ir.name)});")
    if self.uses_executor:
      if self.uses_workspace:
        self.line("const auto* dsl_grid_pool = dsl_workspace.thread_pool();")
        self.line("const auto dsl_grid_requested = dsl_workspace.parallelism();")
      else:
        self.line(
          "const auto* dsl_grid_pool = dsl_grid_context && dsl_grid_context->struct_size >= "
          "offsetof(VecopsExecutionContext,thread_pool)+sizeof(dsl_grid_context->thread_pool) "
          "? dsl_grid_context->thread_pool : nullptr;"
        )
        self.line(
          "const auto dsl_grid_requested = dsl_grid_context && dsl_grid_context->struct_size >= "
          "offsetof(VecopsExecutionContext,requested_threads)+sizeof(dsl_grid_context->requested_threads) "
          "? dsl_grid_context->requested_threads : 0;"
        )
      self.line("const auto dsl_grid_capacity=vecops::execution::grid_capacity(dsl_grid_pool);")
    for cached in self.ir.materializations.values():
      dtype = CPP_TYPES[cached.dtype]
      count = ", ".join(self.scalar(v) for v in cached.shape)
      lanes = self.vector_size(f"vecops::vec::ScalableTag<{dtype}, 0>")
      self.line(f"const bool {cached.storage}_enabled = {self.checked_shape(count)} >= 4 * {lanes};")
      self.line(f"vecops_dsl::Buffer<{dtype}> {cached.storage};")
    insertion = len(self.lines)
    self.block(self.ir.body)
    self.lines[insertion:insertion] = list(self.buffer_declarations.values())
    self.close()
    if self.traces_memory:
      import re

      self.lines = [
        re.sub(r"(vecops_dsl::Buffer<[^;]+> )(\w+);", r'\1\2(dsl_allocation_observer,"\2");', line)
        for line in self.lines
      ]
    headers = (_SME_MATH_POLICY if self.uses_streaming_math_policy else "") + '#include "vecops/execution/Atomic.h"\n'
    if self.traces_memory:
      headers = "#define VECOPS_DSL_MEMORY_TRACE 1\n" + headers
    if any(isinstance(n, I.Print) for n in nodes(self.ir.body)) and self.vector_ir.manager.options.get(
      "debug_prints", True
    ):
      headers += '#include "vecops/execution/Print.h"\n'
    if self.uses_workspace:
      headers += '#include "vecops/execution/WorkspaceContext.h"\n'
    notices = (
      "// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors\n"
      "// SPDX-License-Identifier: MIT\n"
      "// Generated support code is project-owned; user kernel logic retains its original terms.\n"
    )
    return notices + headers + _PREAMBLE + _PACKING_PREAMBLE + "\n".join(self.lines) + "\n"


def generate(prepared):
  emitter = CPUEmitter(prepared)
  with use_catalog(emitter.environment.operations):
    return emitter.generate(), emitter.emitted_regions
