# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Python-facing compiler configuration and lazy native-tool discovery.

The public :class:`Compiler` owns cache policy and reproducible CMake inputs;
it does not start a compiler until an artifact cache miss requires one.  This
keeps ``cache-only`` usable in runtime-only installations without CMake or a
C/C++ compiler.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from . import _C

_BUILD_API_VERSION = 10


def require_native_build_api() -> None:
  """Reject an editable Python/native extension version mismatch clearly."""
  actual = getattr(_C, "build_api_version", 0)
  required_symbols = (
    "KernelBuildBatchRequest",
    "BuildBatchResult",
    "prepare_batch",
    "native_thread_pool_address",
    "native_thread_pool_begin_active",
    "native_thread_pool_end_active",
    "cpu_thread_pool_capacity",
    "cpu_thread_limit",
    "grid_workers_available",
    "memory_event_callback_address",
    "content_watch_epoch",
  )
  missing = [name for name in required_symbols if not hasattr(_C, name)]
  if not hasattr(getattr(_C, "KernelDef", None), "runtime_parallelism"):
    missing.append("KernelDef.runtime_parallelism")
  if actual == _BUILD_API_VERSION and not missing:
    return
  package = Path(__file__).resolve().parent
  source = package.parents[1]
  if (source / "CMakeLists.txt").is_file():
    repair = f"python -m pip install -e {source} --no-build-isolation"
  else:
    repair = "reinstall vecops so its Python package and native extension come from the same build"
  detail = f"build API {actual}, expected {_BUILD_API_VERSION}"
  if missing:
    detail += f"; missing {', '.join(missing)}"
  raise RuntimeError(
    f"vecops Python sources and vecops._C are out of sync ({_C.__file__}; {detail}). Repair with: {repair}"
  )


def default_cache_dir() -> Path:
  """Return the platform user-cache root used when no explicit path is given.

  ``VECOPS_CACHE_DIR`` wins.  This function only calculates a path; it never
  creates the directory.
  """
  override = os.environ.get("VECOPS_CACHE_DIR")
  if override:
    return Path(override).expanduser()
  if sys.platform == "darwin":
    return Path.home() / "Library" / "Caches" / "vecops"
  if os.name == "nt":
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "vecops" / "Cache"
  return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vecops"


def _tool(explicit: str | os.PathLike[str] | None, environment: str, candidates: tuple[str, ...]) -> Path:
  """Locate one executable without dereferencing its driver-name symlink.

  Compiler drivers use ``argv[0]`` semantics: resolving ``clang++`` to the
  underlying ``clang`` binary can suppress automatic C++ runtime linkage.
  Return an absolute path, but deliberately preserve the selected basename.
  """
  requested = str(explicit) if explicit is not None else os.environ.get(environment)
  if requested:
    resolved = shutil.which(requested)
    if resolved:
      return Path(resolved).absolute()
    path = Path(requested).expanduser()
    if path.is_file():
      return path.absolute()
    raise FileNotFoundError(f"cannot find {environment} compiler {requested!r}")
  for candidate in candidates:
    if resolved := shutil.which(candidate):
      return Path(resolved).absolute()
  raise FileNotFoundError(f"cannot find any of {', '.join(candidates)}; pass {environment.lower()}= explicitly")


def _sdk() -> tuple[Path, _C.SdkLayout]:
  """Locate the wheel SDK first, then the source tree used by editable installs."""
  package = Path(__file__).resolve().parent
  installed = package / "_sdk"
  if (installed / "lib" / "cmake" / "Vecops" / "VecopsConfig.cmake").is_file():
    return installed, _C.SdkLayout.package
  source = package.parents[1]
  if (source / "CMakeLists.txt").is_file() and (source / "include" / "vecops" / "Kernel.h").is_file():
    return source, _C.SdkLayout.source_tree
  raise RuntimeError("the vecops compiler SDK is missing from this installation")


_CACHE_MODES = {
  "cache-only": _C.ArtifactCacheMode.cache_only,
  "read-write": _C.ArtifactCacheMode.read_write,
  "compile-only": _C.ArtifactCacheMode.compile_only,
}

_INCLUDE_PATH_ENV = ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH")
_INCLUDE_PATH_FLAGS = {"-I", "-iquote", "-isystem", "-idirafter"}


def _absolute_include_path(value: str, base: Path) -> str:
  path = Path(value or ".").expanduser()
  return str((path if path.is_absolute() else base / path).resolve())


def _normalize_include_flags(flags, base: Path):
  normalized = []
  index = 0
  while index < len(flags):
    flag = flags[index]
    if flag in _INCLUDE_PATH_FLAGS:
      if index + 1 >= len(flags):
        raise ValueError(f"include flag {flag!r} requires a path argument")
      normalized.extend((flag, _absolute_include_path(flags[index + 1], base)))
      index += 2
      continue
    if flag.startswith("-I") and len(flag) > 2:
      normalized.append("-I" + _absolute_include_path(flag[2:], base))
    else:
      normalized.append(flag)
    index += 1
  return tuple(normalized)


def _normalize_include_environment(environment, process_environment, base: Path):
  from shlex import join, split

  result = dict(environment)
  for variable in (*_INCLUDE_PATH_ENV, "CXXFLAGS"):
    value = result.get(variable, process_environment.get(variable, ""))
    if variable == "CXXFLAGS":
      result[variable] = join(_normalize_include_flags(split(value), base))
    else:
      result[variable] = (
        os.pathsep.join(_absolute_include_path(entry, base) for entry in value.split(os.pathsep)) if value else ""
      )
  return result


def _include_flag_paths(flags, base: Path):
  paths = []
  arguments = iter(_normalize_include_flags(tuple(flags), base))
  for flag in arguments:
    if flag in _INCLUDE_PATH_FLAGS:
      paths.append(Path(next(arguments)))
    elif flag.startswith("-I") and len(flag) > 2:
      paths.append(Path(flag[2:]))
  return tuple(paths)


class Compiler:
  """Shared JIT toolchain, source-SDK, and artifact-cache policy.

  Args:
    target: Vecops kernel target architecture; ``"native"`` selects ``Native``
      unless ``VECOPS_TARGET_ARCH`` overrides the process default.
    cache_dir: Persistent artifact root. Defaults below :func:`default_cache_dir`.
    cache_mode: ``"cache-only"``, ``"read-write"``, or ``"compile-only"``.
    build_dir: Root for disposable CMake attempts, separate from artifacts.
    cc, cxx, cmake: Explicit native tools; otherwise environment then PATH wins.
    generator: CMake generator. Defaults to ``CMAKE_GENERATOR``, then Ninja
      when available, otherwise CMake's platform default.
    jobs: Maximum CMake parallel jobs, or a conservative host-derived default.
    verbose: Compatibility alias for output="verbose".
    output: "quiet", concise "progress" (default), or full "verbose" logs.
    environment: Additional environment variables inherited by CMake children.
    cache_namespace: Separates incompatible target/toolchain/SDK cache populations.
    include_dirs, cflags: Extra compile inputs for user kernel source.
    library_dirs, libraries, ldflags: Extra link inputs for user kernel source.

  Cache-only access neither resolves tools nor creates a cache/build directory.
  The object is reusable, but concurrent first compilation through one instance
  is delegated to the native provider's synchronization policy.
  """

  def __init__(
    self,
    *,
    target: str = "native",
    cache_dir: str | os.PathLike[str] | None = None,
    cache_mode: str = "read-write",
    build_dir: str | os.PathLike[str] | None = None,
    cc: str | os.PathLike[str] | None = None,
    cxx: str | os.PathLike[str] | None = None,
    cmake: str | os.PathLike[str] | None = None,
    generator: str | None = None,
    jobs: int | None = None,
    verbose: bool = False,
    output: str = "progress",
    environment: Mapping[str, str] | None = None,
    cache_namespace: str | None = None,
    include_dirs=(),
    cflags=(),
    library_dirs=(),
    libraries=(),
    ldflags=(),
  ) -> None:
    from shlex import split

    try:
      self.cache_mode = _CACHE_MODES[cache_mode]
    except KeyError as error:
      raise ValueError(f"unknown cache mode {cache_mode!r}") from error
    self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir() / "artifacts"
    self.definition_cwd = Path.cwd().resolve()
    default_build = Path(os.environ.get("VECOPS_BUILD_DIR", default_cache_dir() / "builds"))
    self.build_dir = Path(build_dir).expanduser() if build_dir is not None else default_build
    target_override = os.environ.get("VECOPS_TARGET_ARCH")
    selected_target = target_override if target.lower() == "native" and target_override else target
    self.target = "Native" if selected_target.lower() == "native" else selected_target
    self.cc = cc
    self.cxx = cxx
    self.cmake = cmake
    self.generator = generator
    self.jobs = jobs
    self.verbose = verbose
    if output not in ("quiet", "progress", "verbose"):
      raise ValueError("output must be quiet, progress or verbose")
    self.output = "verbose" if verbose else output
    self.build_events = []
    supplied_environment = dict(environment or {})
    self.environment = _normalize_include_environment(supplied_environment, os.environ, self.definition_cwd)
    # Retain the legacy namespace as a user-visible population label. Artifact
    # compatibility is now governed by full content/build identities, not bumps.
    self.cache_namespace = cache_namespace or f"python-sdk-v12;target={self.target}"
    self.include_dirs = tuple(
      Path(path).expanduser()
      if Path(path).expanduser().is_absolute()
      else self.definition_cwd / Path(path).expanduser()
      for path in include_dirs
    )
    self.include_dirs = tuple(path.resolve() for path in self.include_dirs)
    self.cflags = _normalize_include_flags(tuple(cflags), self.definition_cwd)
    explicit_include_roots = set(self.include_dirs)
    explicit_include_roots.update(_include_flag_paths(self.cflags, self.definition_cwd))
    explicit_include_roots.update(_include_flag_paths(split(self.environment["CXXFLAGS"]), self.definition_cwd))
    implicit_cwd_include = False
    effective_environment = {**os.environ, **supplied_environment}
    for variable in _INCLUDE_PATH_ENV:
      value = effective_environment.get(variable, "")
      if not value:
        continue
      for entry in value.split(os.pathsep):
        if entry:
          explicit_include_roots.add(Path(_absolute_include_path(entry, self.definition_cwd)))
        else:
          implicit_cwd_include = True
    self._source_include_inventory_excludes = (
      (self.definition_cwd,) if implicit_cwd_include and self.definition_cwd not in explicit_include_roots else ()
    )
    self.library_dirs = tuple(Path(path).expanduser().resolve() for path in library_dirs)
    self.libraries = tuple(libraries)
    self.ldflags = tuple(ldflags)
    self._native_compiler: _C.Compiler | None = None
    self._vector_state = None
    self._is_snapshot = False

  @property
  def vector_state(self):
    """Current calling-thread lengths, or the immutable elaboration snapshot."""
    from ._arm_vector_state import get_vector_state

    return self._vector_state if self._vector_state is not None else get_vector_state()

  def snapshot(self):
    """Freeze per-thread ISA state without changing the user's compiler object.

    A compiler can be reused by factories in processes/threads with different
    VL/SVL. Operators retain a frozen copy, so pending batches never silently
    pick up a later setting. Build event storage remains shared for inspection.
    """
    if self._is_snapshot:
      return self
    state = self.vector_state
    import copy

    result = copy.copy(self)
    result.environment = dict(self.environment)
    result._vector_state = state
    result._native_compiler = None
    result._is_snapshot = True
    return result

  def is_environment_current(self):
    if self._vector_state is None:
      return True
    from ._arm_vector_state import get_vector_state

    return get_vector_state() == self._vector_state

  def _native_config(self) -> _C.KernelCompilerConfig:
    """Resolve and return the native build policy shared by all task kinds."""
    sdk_path, sdk_layout = _sdk()
    sdk = _C.SdkSpec()
    sdk.path = sdk_path
    sdk.layout = sdk_layout
    toolchain = _C.ToolchainSpec()
    toolchain.c_compiler = _tool(self.cc, "CC", ("clang", "gcc"))
    toolchain.cxx_compiler = _tool(self.cxx, "CXX", ("clang++", "g++"))
    toolchain.cmake_program = _tool(self.cmake, "CMAKE", ("cmake",))
    selected_generator = self.generator or os.environ.get("CMAKE_GENERATOR")
    if selected_generator is None and shutil.which("ninja"):
      selected_generator = "Ninja"
    toolchain.generator = selected_generator
    affinity = getattr(os, "sched_getaffinity", None)
    available = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
    toolchain.parallel_jobs = self.jobs or max(1, min(available, 8))
    toolchain.environment = self.environment
    toolchain.output_mode = self.output
    import weakref

    reference = weakref.ref(self)

    def observe(event):
      selected = reference()
      if selected is not None:
        selected._observe_build(event)

    toolchain.observer = observe
    config = _C.KernelCompilerConfig()
    config.sdk = sdk
    config.toolchain = toolchain
    config.work_directory = self.build_dir
    config.target_arch = self.target
    config.include_directories = self.include_dirs
    config.compile_options = self.cflags
    state = self.vector_state
    if self.target.lower() != "scalar" and state.sve_bytes is not None:
      # Manual SM regions must remain sizeless when VL and SVL differ. Keep
      # Vec representations single-register; loop bounds use domain-specific
      # captured constants rather than a global fixed-SVE type assumption.
      config.compile_options = (*self.cflags, "-msve-vector-bits=scalable")
      definitions = [f"VECOPS_KERNEL_SVE_BYTES={state.sve_bytes}"]
      if state.sme_bytes is not None:
        definitions += [
          f"VECOPS_KERNEL_SME_BYTES={state.sme_bytes}",
          f"VECOPS_TARGET_FIXED_STREAMING_SVE_BITS={state.sme_bytes * 8}",
        ]
      config.compile_definitions = definitions
      # The actual widths come from the Python caller, not the child CMake
      # process (which may reset VL at exec) or the maximum supported width.
      if self.target in {"NativeFixedSVE", "NativeFixedStreamingSVE"}:
        config.target_arch = "Native"
    config.link_directories = self.library_dirs
    config.link_libraries = self.libraries
    config.link_options = self.ldflags
    return config

  def _observe_build(self, event):
    import time

    self.build_events.append(
      {
        "phase": event.phase,
        "state": event.state,
        "log_path": str(event.log_path),
        "elapsed_seconds": event.elapsed_seconds,
        "exit_code": event.exit_code,
        "timestamp_ns": time.time_ns(),
      }
    )

  @property
  def build_backend(self) -> _C.Compiler:
    """Return the native batch builder even for cache-only kernel providers."""
    state = self.vector_state
    if not self._is_snapshot and (state.sve_bytes is not None or state.sme_bytes is not None):
      return self.snapshot().build_backend
    require_native_build_api()
    if self._native_compiler is None:
      self._native_compiler = _C.Compiler(self._native_config())
    return self._native_compiler

  @property
  def native(self) -> _C.Compiler | None:
    """Return the lazily-created raw compiler, or ``None`` for cache-only mode.

    Accessing this property in a build-capable mode resolves the SDK and native
    tools and may raise :class:`FileNotFoundError`; it still does not compile.
    """
    if self.cache_mode == _C.ArtifactCacheMode.cache_only:
      return None
    return self.build_backend

  def operator(self, source, kernel_def, *, recipe_id: str | None = None):
    """Create an :class:`~vecops.Operator` backed by one ``__kernel__`` source file."""
    from ._operator import Operator

    return Operator(source, kernel_def, self, recipe_id=recipe_id)


__all__ = ["Compiler", "default_cache_dir"]
