# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Pythonic compiler/runtime API; low-level bindings live in :mod:`vecops._C`."""

import os

from . import _C, graph, planner

if hasattr(_C, "memory"):
  from . import memory as memory
from ._build_session import BuildSession
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
from ._operator import Operator
from ._precompile import (
  CompileRequest,
  CompileRequestCollector,
  PrecompileResult,
  collect_compile_requests,
  compile_batch,
  is_discovering_graph_artifacts,
  is_precompiling,
  precompile,
)
from ._schema import Dynamic, KernelDef, TensorAccess, TensorDef, TensorMeta, ValueDef
from ._thread_pool import cpu_thread_limit, cpu_thread_pool_capacity, native_thread_pool_region


def set_flush_subnormals(enabled: bool) -> None:
  """Set hardware subnormal flushing for this thread and vecops workers.

  Call before creating non-vecops worker pools so their new threads inherit
  the mode. Existing vecops workers apply the environment setting at each
  logical task entry. Other already-running thread pools need their own
  per-thread setup.
  """
  if not isinstance(enabled, bool):
    raise TypeError("enabled must be a bool")
  if not _C.flush_subnormals_supported():
    raise RuntimeError("hardware subnormal flushing is unsupported on this architecture")
  os.environ["VECOPS_FLUSH_SUBNORMALS"] = "1" if enabled else "0"
  _C.set_current_thread_flush_subnormals(enabled)


def current_thread_flush_subnormals() -> bool:
  """Return whether the calling thread has hardware subnormal flushing on."""
  return bool(_C.current_thread_flush_subnormals())

__all__ = [  # noqa: RUF022 - retain the established public API grouping
  "_C",
  "Compiler",
  "BuildSession",
  "CompileRequest",
  "CompileRequestCollector",
  "Const",
  "DType",
  "Dynamic",
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
  "current_thread_flush_subnormals",
  "default_cache_dir",
  "float16",
  "float32",
  "float64",
  "int8",
  "int16",
  "int32",
  "int64",
  "is_precompiling",
  "is_discovering_graph_artifacts",
  "normalize_dtype",
  "native_thread_pool_region",
  "cpu_thread_limit",
  "cpu_thread_pool_capacity",
  "graph",
  "planner",
  "precompile",
  "set_flush_subnormals",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
]

if hasattr(_C, "memory"):
  __all__.append("memory")
