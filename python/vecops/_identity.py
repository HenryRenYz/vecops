# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Content-addressed build inputs, shared by source kernels and bridges.

Checks run at lookup/elaboration, never on a prepared kernel's execution path.
Directory inventories deliberately invalidate on new/shadowing headers. The
cache-only reader uses an existing toolchain attestation without discovering or
executing tools; ambiguous attestations require selecting an installed tool.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

try:
  import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
  fcntl = None


def digest(value):
  return hashlib.sha256(repr(value).encode()).hexdigest()


_trace_lock = threading.Lock()


def _reset_trace_lock_after_fork():
  global _trace_lock
  _trace_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
  os.register_at_fork(after_in_child=_reset_trace_lock_after_fork)


def _write_build_input_trace(event, *, compiler=None):
  """Append one opt-in, secret-free build-input diagnostic JSONL record."""
  if compiler is not None:
    from . import _C

    if compiler.cache_mode == _C.ArtifactCacheMode.cache_only:
      return
  configured = os.environ.get("VECOPS_BUILD_INPUT_TRACE")
  if not configured:
    return
  path = Path(configured).expanduser()
  if not path.is_absolute():
    raise ValueError("VECOPS_BUILD_INPUT_TRACE must be an absolute path")
  if not path.parent.is_dir():
    raise ValueError("VECOPS_BUILD_INPUT_TRACE parent directory must already exist")
  payload = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
  with _trace_lock:
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
      if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
      view = memoryview(payload)
      while view:
        view = view[os.write(descriptor, view) :]
    finally:
      if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
      os.close(descriptor)


def _input_records(inputs):
  """Flatten source_inputs output into path-to-identity records."""
  files, roots = inputs
  result = {path: identity for path, identity in files}
  for root_records in roots:
    result.update(root_records)
  return result


def _source_input_diff(before, after, *, limit=32):
  """Return a bounded path diff without re-reading or re-hashing inputs."""
  old, new = _input_records(before), _input_records(after)
  changed = [
    {"path": path, "before": old.get(path), "after": new.get(path)}
    for path in sorted(old.keys() | new.keys())
    if old.get(path) != new.get(path)
  ]
  return {"changed_count": len(changed), "paths": changed[:limit]}


def _code_identity(code):
  """Stable Python code content, independent of marshal reference/intern flags."""
  import types

  def constant(value):
    if isinstance(value, types.CodeType):
      return _code_identity(value)
    if isinstance(value, tuple):
      return tuple(constant(v) for v in value)
    if isinstance(value, frozenset):
      return tuple(sorted((constant(v) for v in value), key=repr))
    return value

  return (
    code.co_code,
    tuple(constant(v) for v in code.co_consts),
    code.co_names,
    code.co_varnames,
    code.co_freevars,
    code.co_cellvars,
    code.co_argcount,
    code.co_posonlyargcount,
    code.co_kwonlyargcount,
    code.co_flags,
    getattr(code, "co_exceptiontable", b""),
    code.co_filename,
    code.co_firstlineno,
  )


def file_identity(path):
  path = Path(path)
  from ._watch import content_hash

  return str(path.absolute()), content_hash(path)


def tree_identity(path, suffixes=None):
  from ._watch import tree_hash

  return tree_hash(path, suffixes)


def implementation_identity():
  from ._compiler import _sdk

  sdk, layout = _sdk()
  package = Path(__file__).parent
  import sys

  from . import _C

  items = [sys.version, file_identity(_C.__file__), tree_identity(package, {".py"}), tree_identity(sdk / "include")]
  if layout.name == "source_tree":
    items += [tree_identity(sdk / "src"), tree_identity(sdk / "cmake"), file_identity(sdk / "CMakeLists.txt")]
  else:
    items += [tree_identity(sdk / "lib")]
  return digest(items)


def callable_identity(function, *, include_state=True):
  """Actual callback bytecode as well as inspectable source; no qualname-only key."""
  code = getattr(function, "__code__", None)
  if code is None:
    if inspect.isbuiltin(function):
      import sys

      return (function.__module__, function.__qualname__, sys.version)
    token = getattr(function, "__vecops_cache_token__", None)
    if token is None:
      # Unknown native/plugin state cannot be promised cacheable.
      return (type(function).__module__, type(function).__qualname__, os.urandom(16).hex())
    return (type(function).__qualname__, token)
  source = inspect.getsourcefile(function)

  def freeze(value):
    if value is None or isinstance(value, (str, bool, int, float, bytes)):
      return value
    if isinstance(value, tuple):
      return tuple(freeze(v) for v in value)
    if isinstance(value, list):
      return tuple(freeze(v) for v in value) if include_state else ("host-list",)
    if isinstance(value, dict):
      return (
        tuple((freeze(k), freeze(v)) for k, v in sorted(value.items(), key=lambda pair: repr(pair[0])))
        if include_state
        else ("host-dict",)
      )
    if isinstance(value, (set, frozenset)):
      return tuple(sorted((freeze(v) for v in value), key=repr))
    if isinstance(value, Path):
      return file_identity(value)
    if hasattr(value, "canonical"):
      return value.canonical()
    if isinstance(value, type):
      return (value.__module__, value.__qualname__)
    if inspect.ismodule(value):
      return (value.__name__, getattr(value, "__version__", None))
    if inspect.isfunction(value):
      path = inspect.getsourcefile(value)
      return (digest(_code_identity(value.__code__)), file_identity(path) if path else None)
    if inspect.isbuiltin(value):
      return (value.__module__, value.__qualname__)
    if type(value).__module__ == "vecops.language.dsl" and hasattr(value, "function"):
      return callable_identity(value.function, include_state=include_state)
    from ._compiler import Compiler

    if isinstance(value, Compiler):
      return (
        value.target,
        value.cache_namespace,
        str(value.cc),
        str(value.cxx),
        value.cflags,
        value.ldflags,
        tuple(map(str, value.include_dirs)),
        tuple(sorted(value.environment.items())),
      )
    token = getattr(value, "__vecops_cache_token__", None)
    if token is not None:
      return token
    # DTypes/enums are immutable; a mutable closure without an explicit plugin
    # token disables memoization instead of silently caching a changing pass.
    import enum

    if isinstance(value, enum.Enum) or type(value).__module__ in ("torch", "vecops._C"):
      return (type(value).__qualname__, str(value))
    return (type(value).__qualname__, os.urandom(16).hex() if include_state else "host-object")

  variables = inspect.getclosurevars(function)
  captures = tuple((k, freeze(v)) for k, v in sorted({**variables.globals, **variables.nonlocals}.items()))
  return (
    digest(_code_identity(code)),
    file_identity(source) if source else None,
    getattr(function, "__vecops_cache_token__", None),
    captures,
    freeze(function.__defaults__),
    freeze(function.__kwdefaults__),
  )


def source_inputs(source, include_dirs=(), dependencies=(), *, definition_cwd=None, inventory_excludes=()):
  """Conservative quoted/angle include closure plus user include inventories.

  The SDK is tracked separately in implementation_identity. System headers are
  attested by the toolchain identity. Macro-generated external includes must be
  declared through dependencies or an include directory inventory. An implicit
  CWD search root can be excluded from whole-directory inventory while its
  resolved include closure remains content-addressed.
  """
  from ._compiler import _sdk

  sdk, _ = _sdk()
  base = Path(definition_cwd).resolve() if definition_cwd is not None else Path.cwd().resolve()
  roots = tuple((Path(p) if Path(p).is_absolute() else base / p).resolve() for p in include_dirs)
  visited, inputs = set(), []

  def visit(path):
    path = Path(path)
    if not path.is_absolute():
      path = base / path
    path = path.resolve()
    if path in visited:
      return
    visited.add(path)
    inputs.append(file_identity(path))
    if not path.is_file():
      return
    includes = re.findall(r'^\s*#\s*include\s*([<"])([^>"\n]+)[>"]', path.read_text(errors="replace"), re.MULTILINE)
    for delimiter, name in includes:
      # Only quoted includes search the including file's directory. In
      # particular, an unrelated sibling must not hide an angle include
      # selected from the implicit cwd search root whose inventory we omit.
      search = (path.parent, *roots, sdk / "include") if delimiter == '"' else (*roots, sdk / "include")
      for include_base in search:
        candidate = include_base / name
        if candidate.is_file():
          if not candidate.resolve().is_relative_to(sdk / "include"):
            visit(candidate)
          break

  visit(source)
  for dependency in dependencies:
    visit(dependency)
  excluded = {Path(path).resolve() for path in inventory_excludes}
  inventories = tuple(tree_identity(root) for root in roots if root not in excluded)
  return (tuple(sorted(inputs)), inventories)


def include_roots(compiler):
  from shlex import split

  from ._compiler import _absolute_include_path, _normalize_include_flags

  base = compiler.definition_cwd
  roots = list(compiler.include_dirs)
  flags = (*split(compiler.environment.get("CXXFLAGS", "")), *compiler.cflags)
  arguments = iter(_normalize_include_flags(flags, base))
  for arg in arguments:
    if arg in ("-I", "-isystem", "-iquote", "-idirafter"):
      roots.append(Path(next(arguments)))
    elif arg.startswith("-I") and len(arg) > 2:
      roots.append(Path(arg[2:]))
  for variable in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    value = compiler.environment[variable]
    if value:
      roots.extend(Path(_absolute_include_path(path, base)) for path in value.split(os.pathsep))
  return tuple(dict.fromkeys(roots))


def compiler_identity(compiler):
  """Persist the full build policy; output/jobs/build paths are presentation only."""
  from . import _C
  from ._compiler import _tool

  policy = (
    compiler.target,
    compiler.vector_state.env_token,
    compiler.cache_namespace,
    tuple(map(str, compiler.include_dirs)),
    compiler.cflags,
    tuple(map(str, compiler.library_dirs)),
    compiler.libraries,
    compiler.ldflags,
    compiler.generator or os.environ.get("CMAKE_GENERATOR") or "auto",
    tuple(sorted(compiler.environment.items())),
    tuple(
      (k, compiler.environment.get(k, os.environ.get(k)))
      for k in (
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "C_INCLUDE_PATH",
        "LIBRARY_PATH",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "SDKROOT",
      )
    ),
  )
  sdk_identity = implementation_identity()
  libraries = []
  for name in compiler.libraries:
    p = Path(name)
    if p.is_file():
      libraries.append(file_identity(p))
    else:
      for root in compiler.library_dirs:
        for ext in (".so", ".a", ".dylib"):
          candidate = root / f"lib{name}{ext}"
          if candidate.is_file():
            libraries.append(file_identity(candidate))
  config_key = digest((policy, sdk_identity, libraries))
  attestation = compiler.cache_dir / "identities" / f"{config_key}.json"
  if compiler.cache_mode == _C.ArtifactCacheMode.cache_only:
    if not attestation.is_file():
      raise ImportError("cache-only execution has no matching build-input attestation")
    records = json.loads(attestation.read_text())
    if len(records) != 1:
      raise ImportError("cache-only build identity is ambiguous across toolchains; select a unique cache namespace")
    return records[0]["identity"]
  tools = tuple(
    file_identity(_tool(getattr(compiler, attribute), variable, choices))
    for attribute, variable, choices in (
      ("cc", "CC", ("clang", "gcc")),
      ("cxx", "CXX", ("clang++", "g++")),
      ("cmake", "CMAKE", ("cmake",)),
    )
  )
  # Include the compiler's actual standard-library search roots. No source is
  # compiled here. This catches toolchain headers being updated in-place.
  import shlex
  import subprocess

  environment = {**os.environ, **compiler.environment}
  # CMake initializes CMAKE_CXX_FLAGS from CXXFLAGS. Compiler drivers do not
  # consume that variable themselves: pass it explicitly when attesting the
  # actual headers/runtime (e.g. a private --gcc-toolchain or -stdlib choice).
  flags = (*shlex.split(environment.get("CXXFLAGS", "")), *compiler.cflags)
  command = [tools[1][0], *flags, "-E", "-x", "c++", "-v", "-"]
  search = subprocess.run(
    command, input="", text=True, capture_output=True, check=True, env=environment, cwd=compiler.definition_cwd
  ).stderr
  match = re.search(r"#include <\.\.\.> search starts here:\n(.*?)End of search list", search, re.DOTALL)
  user_roots = {str(root) for root in include_roots(compiler)}
  definition_cwd = compiler.definition_cwd
  system_roots = (
    tuple(
      (
        line.strip(),
        tree_identity(Path(line.strip()) if Path(line.strip()).is_absolute() else definition_cwd / line.strip()),
      )
      for line in match.group(1).splitlines()
      if str((Path(line.strip()) if Path(line.strip()).is_absolute() else definition_cwd / line.strip()).resolve())
      not in user_roots
    )
    if match
    else ()
  )
  system = tuple(records for _, records in system_roots)
  linked_runtime = []
  for name in ("libstdc++.so", "libgcc_s.so", "libc.so"):
    location = subprocess.run(
      [tools[1][0], *flags, f"-print-file-name={name}"],
      capture_output=True,
      text=True,
      check=True,
      env=environment,
      cwd=compiler.definition_cwd,
    ).stdout.strip()
    runtime_path = Path(location)
    if not runtime_path.is_absolute():
      runtime_path = compiler.definition_cwd / runtime_path
    if runtime_path.is_file():
      linked_runtime.append(file_identity(runtime_path))
  import platform

  hardware = platform.machine()
  if compiler.target.lower().startswith("native"):
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
      hardware += "\n" + "\n".join(
        line for line in cpuinfo.read_text().splitlines() if line.startswith(("flags", "Features", "model name"))
      )
  identity = digest(
    (
      config_key,
      tools,
      system,
      linked_runtime,
      hardware,
      compiler.generator or os.environ.get("CMAKE_GENERATOR") or ("Ninja" if shutil.which("ninja") else None),
    )
  )
  generator = compiler.generator or os.environ.get("CMAKE_GENERATOR") or ("Ninja" if shutil.which("ninja") else None)
  components = {
    "config_key": config_key,
    "tools": [{"path": path, "content": content} for path, content in tools],
    "system_roots": [
      {
        "path": root,
        "tree_digest": digest(records),
        "unreadable": [
          {"path": path, "identity": value}
          for path, value in records
          if isinstance(value, str) and value.startswith("unreadable:")
        ][:16],
      }
      for root, records in system_roots
    ],
    "linked_runtime": [{"path": path, "content": content} for path, content in linked_runtime],
    "hardware_digest": digest(hardware),
    "generator": generator,
  }
  _write_build_input_trace(
    {"event": "compiler_identity", "identity": identity, "components": components}, compiler=compiler
  )
  if getattr(compiler, "_build_identity", None) != identity:
    compiler._native_compiler = None
    compiler._build_identity = identity
  attestation.parent.mkdir(parents=True, exist_ok=True)
  # Serialize read/append/replace across processes, not just the final rename.
  # Cache-only readers never create this lock or any directories.
  from ._cache_lock import file_lock

  with file_lock(attestation.with_suffix(".lock")):
    records = json.loads(attestation.read_text()) if attestation.is_file() else []
    matching = next((record for record in records if record["identity"] == identity), None)
    changed = False
    if matching is None:
      records.append({"identity": identity, "tools": tools})
      matching = records[-1]
      changed = True
    if "components" not in matching:
      matching["components"] = components
      changed = True
    if changed:
      descriptor, temporary = tempfile.mkstemp(prefix="identity-", dir=attestation.parent)
      try:
        with os.fdopen(descriptor, "w") as stream:
          json.dump(records, stream)
        os.replace(temporary, attestation)
      finally:
        Path(temporary).unlink(missing_ok=True)
  return identity
