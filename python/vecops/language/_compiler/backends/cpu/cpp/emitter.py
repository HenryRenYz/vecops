"""C++ emission: emitter."""
from __future__ import annotations
from .... import ir as I
from .....types import TensorType
from ..capabilities import CPP_TYPES
from .expressions import ExpressionsEmitter
from .memory import MemoryEmitter
from .vectors import VectorsEmitter
from .control import ControlEmitter
from .support import _PREAMBLE

class CPUEmitter(ExpressionsEmitter, MemoryEmitter, VectorsEmitter, ControlEmitter):
  def __init__(self, prepared):
    self.prepared = prepared
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
    self.active_loads = {}
    self.snapshots = {}
    self.storage_views = {}
    self.materialized = {}
    self.reduction_materialized = {}


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
    parameters = []
    for name, annotation in self.ir.parameters.items():
      if isinstance(annotation, TensorType):
        parameters.append(f"vecops::tensor::TensorLike auto {name}")
        shape = tuple(I.Expr(I.ScalarType("int64"), "raw", data=f"{name}.size({axis})") for axis in range(len(annotation.shape)))
        strides = tuple(I.Expr(I.ScalarType("int64"), "raw", data=f"{name}.stride({axis})") for axis in range(len(annotation.shape)))
        from ....dtypes import dtype_name
        self.storage_views[name] = I.View(name, dtype_name(annotation.dtype), shape, strides, I.literal(0))
      else:
        from ....dtypes import dtype_name
        parameters.append(f"{CPP_TYPES[dtype_name(annotation)]} {name}")
    self.opening(f"void __kernel__({', '.join(parameters)})")
    for cached in self.ir.materializations.values():
      dtype = CPP_TYPES[cached.dtype]
      count = ", ".join(self.scalar(v) for v in cached.shape)
      lanes = f"vecops::vec::size(vecops::vec::ScalableTag<{dtype}, 0>{{}})"
      self.line(f"const bool {cached.storage}_enabled = vecops_dsl::checked_shape({{{count}}}) >= 4 * {lanes};")
      self.line(f"vecops_dsl::Buffer<{dtype}> {cached.storage};")
    self.block(self.ir.body)
    self.close()
    return _PREAMBLE + "\n".join(self.lines) + "\n"


def generate(prepared):
  emitter = CPUEmitter(prepared)
  return emitter.generate(), emitter.emitted_regions
