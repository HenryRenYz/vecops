"""C++ emission: emitter."""

from __future__ import annotations

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
from .matmul import MatmulEmitter
from .matmul_support import PREAMBLE as _MATMUL_PREAMBLE
from .memory import MemoryEmitter
from .primitives import PrimitivesEmitter
from .printing import PrintEmitter
from .state import StateEmitter
from .support import _PREAMBLE
from .transpose import TransposeEmitter
from .vectors import VectorsEmitter


class CPUEmitter(
  ExpressionsEmitter,
  MemoryEmitter,
  VectorsEmitter,
  ControlEmitter,
  TransposeEmitter,
  HardwareEmitter,
  MatmulEmitter,
  GridEmitter,
  StateEmitter,
  PrimitivesEmitter,
  CollectiveEmitter,
  PrintEmitter,
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
    self.active_loads = {}
    self.snapshots = {}
    self.storage_views = {}
    self.materialized = {}
    self.reduction_materialized = {}
    self.hardware_scope = None
    self.buffer_declarations = {}
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

  def allocate_buffer(self, dtype, name, count):
    # Invocation-local scratch reuse across serial loops, never process/global
    # state. Each textual allocation site has its own independently sized slot.
    self.buffer_declarations[name] = f"  vecops_dsl::Buffer<{CPP_TYPES[dtype]}> {name};"
    self.line(f"{name}.resize({count});")

  def unique(self, prefix="v"):
    self.counter += 1
    return f"dsl_c_{prefix}_{self.counter}"

  def line(self, text):
    self.lines.append("  " * self.indent + text)

  def opening(self, text):
    self.line(text + " {")
    self.indent += 1

  def close(self):
    self.indent -= 1
    self.line("}")

  def generate(self):
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
      lanes = f"vecops::vec::size(vecops::vec::ScalableTag<{dtype}, 0>{{}})"
      self.line(f"const bool {cached.storage}_enabled = vecops_dsl::checked_shape({{{count}}}) >= 4 * {lanes};")
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
    headers = '#include "vecops/execution/Atomic.h"\n'
    if self.traces_memory:
      headers = "#define VECOPS_DSL_MEMORY_TRACE 1\n" + headers
    if any(isinstance(n, I.Print) for n in nodes(self.ir.body)) and self.vector_ir.manager.options.get(
      "debug_prints", True
    ):
      headers += '#include "vecops/execution/Print.h"\n'
    if self.uses_workspace:
      headers += '#include "vecops/execution/WorkspaceContext.h"\n'
    return headers + _PREAMBLE + _MATMUL_PREAMBLE + "\n".join(self.lines) + "\n"


def generate(prepared):
  emitter = CPUEmitter(prepared)
  with use_catalog(emitter.environment.operations):
    return emitter.generate(), emitter.emitted_regions
