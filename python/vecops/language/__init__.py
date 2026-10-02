"""Tile-style Python metaprogramming frontend for vecops kernels."""

from .compile import CompiledKernel, compile, compile_pending
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
from . import passes
from ._compiler.pipeline import load_builtin_passes as _load_builtin_passes
_load_builtin_passes()
from .dsl import *
from .dsl import __all__ as _dsl_exports
from .._dtype import float16, bfloat16, float32, float64, int32, int64, bool_

__all__ = [
  "Any",
  "CompiledKernel",
  "Const",
  "DType",
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
  "maximum",
  "minimum",
  "template",
]
__all__ += [*_dsl_exports, "float16", "bfloat16", "float32", "float64", "int32", "int64", "bool_"]
from .library import layernorm, softmax
__all__ += ["layernorm", "softmax"]
__all__ += ["passes"]
