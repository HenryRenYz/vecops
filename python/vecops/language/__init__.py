"""Tile-style Python metaprogramming frontend for vecops kernels."""

from . import amx, cost, library, passes, sme
from ._compiler.pipeline import load_builtin_passes as _load_builtin_passes
from .compile import CompiledKernel, compile, compile_pending
from .debug import DebugOptions, lower, stderr, stdout
from .expr import (
  Any,
  Const,
  DType,
  Dynamic,
  Predicate,
  any,
  dynamic,
  is_value,
  maximum,
  minimum,
)
from .jit import jit
from .template import TemplateFunction, template
from .types import Tensor, TensorType

_load_builtin_passes()
from .._dtype import bfloat16, bool_, float16, float32, float64, int32, int64
from .dsl import *
from .dsl import __all__ as _dsl_exports

__all__ = [
  "Any",
  "CompiledKernel",
  "Const",
  "DType",
  "DebugOptions",
  "Dynamic",
  "Predicate",
  "TemplateFunction",
  "Tensor",
  "TensorType",
  "any",
  "compile",
  "compile_pending",
  "dynamic",
  "is_value",
  "jit",
  "lower",
  "maximum",
  "minimum",
  "stderr",
  "stdout",
  "template",
]
__all__ += [*_dsl_exports, "float16", "bfloat16", "float32", "float64", "int32", "int64", "bool_"]
from .library import layernorm, matmul, softmax, transpose_copy

__all__ += ["layernorm", "matmul", "softmax", "transpose_copy"]
__all__ += ["amx", "cost", "library", "passes", "sme"]
