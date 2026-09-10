"""Pythonic compiler/runtime API; low-level bindings live in :mod:`vecops._C`."""

from . import _C, ops
from ._compiler import Compiler, default_cache_dir
from ._dtype import (
  Const,
  DType,
  bfloat16,
  bool_,
  float16,
  float32,
  float64,
  int8,
  int16,
  int32,
  int64,
  normalize_dtype,
  uint8,
  uint16,
  uint32,
  uint64,
)
from ._jit import JitKernel, jit
from ._operator import Operator
from ._precompile import (
  CompileRequest,
  CompileRequestCollector,
  PrecompileResult,
  collect_compile_requests,
  compile_batch,
  is_precompiling,
  precompile,
)
from ._schema import Dynamic, KernelDef, TensorAccess, TensorDef, TensorMeta, ValueDef

__all__ = [
  "_C",
  "Compiler",
  "CompileRequest",
  "CompileRequestCollector",
  "Const",
  "DType",
  "Dynamic",
  "JitKernel",
  "KernelDef",
  "Operator",
  "PrecompileResult",
  "TensorAccess",
  "TensorDef",
  "TensorMeta",
  "ValueDef",
  "bfloat16",
  "bool_",
  "collect_compile_requests",
  "compile_batch",
  "default_cache_dir",
  "float16",
  "float32",
  "float64",
  "int8",
  "int16",
  "int32",
  "int64",
  "is_precompiling",
  "jit",
  "normalize_dtype",
  "ops",
  "precompile",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
]
