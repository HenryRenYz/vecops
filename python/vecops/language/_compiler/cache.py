"""Compilation-core cache and source/compiler identity; no runtime dispatch."""

from __future__ import annotations
import hashlib
import threading
import weakref
from typing import Any
from ..._compiler import Compiler
from ..template import TemplateIR

_compile_cache = weakref.WeakValueDictionary()
_compile_lock = threading.RLock()


def _source_identity(ir: TemplateIR) -> str:
  digest = hashlib.sha256()
  digest.update(b"vecops-language-cpp-v1\0")
  if ir.source is not None:
    digest.update(ir.source.read_bytes())
  for dependency in ir.dependencies:
    digest.update(str(dependency).encode())
    digest.update(b"\0")
    digest.update(dependency.read_bytes())
  return digest.hexdigest()


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
