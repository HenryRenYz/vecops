"""Compilation-core cache and source/compiler identity; no runtime dispatch."""

from __future__ import annotations

import threading
import weakref

from ..._compiler import Compiler
from ..template import TemplateIR

_compile_cache = weakref.WeakValueDictionary()
_compile_lock = threading.RLock()


def _source_identity(ir: TemplateIR) -> str:
  from ..._identity import digest, source_inputs

  return digest(source_inputs(ir.source, dependencies=ir.dependencies))


def _compiler_key(compiler: Compiler):
  return (
    compiler.target,
    str(compiler.cache_dir),
    str(compiler.build_dir),
    compiler.cache_mode,
    compiler.cache_namespace,
    tuple(map(str, compiler.include_dirs)),
    compiler.cflags,
    tuple(map(str, compiler.library_dirs)),
    compiler.libraries,
    compiler.ldflags,
    str(compiler.cc),
    str(compiler.cxx),
    str(compiler.cmake),
    compiler.generator,
    tuple(sorted(compiler.environment.items())),
  )
