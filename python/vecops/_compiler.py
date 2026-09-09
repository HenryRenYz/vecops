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


class Compiler:
  """Shared JIT toolchain, source-SDK, and artifact-cache policy.

  Args:
    target: Vecops kernel target architecture; ``"native"`` selects ``Native``.
    cache_dir: Persistent artifact root. Defaults below :func:`default_cache_dir`.
    cache_mode: ``"cache-only"``, ``"read-write"``, or ``"compile-only"``.
    build_dir: Root for disposable CMake attempts, separate from artifacts.
    cc, cxx, cmake: Explicit native tools; otherwise environment then PATH wins.
    jobs: Maximum CMake parallel jobs, or a conservative host-derived default.
    verbose: Reserved presentation option retained on this public object.
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
    jobs: int | None = None,
    verbose: bool = False,
    environment: Mapping[str, str] | None = None,
    cache_namespace: str | None = None,
    include_dirs=(),
    cflags=(),
    library_dirs=(),
    libraries=(),
    ldflags=(),
  ) -> None:
    try:
      self.cache_mode = _CACHE_MODES[cache_mode]
    except KeyError as error:
      raise ValueError(f"unknown cache mode {cache_mode!r}") from error
    self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir() / "artifacts"
    default_build = Path(os.environ.get("VECOPS_BUILD_DIR", default_cache_dir() / "builds"))
    self.build_dir = Path(build_dir).expanduser() if build_dir is not None else default_build
    self.target = "Native" if target.lower() == "native" else target
    self.cc = cc
    self.cxx = cxx
    self.cmake = cmake
    self.jobs = jobs
    self.verbose = verbose
    self.environment = dict(environment or {})
    self.cache_namespace = cache_namespace or f"python-sdk-v2;target={self.target}"
    self.include_dirs = tuple(Path(path).expanduser().resolve() for path in include_dirs)
    self.cflags = tuple(cflags)
    self.library_dirs = tuple(Path(path).expanduser().resolve() for path in library_dirs)
    self.libraries = tuple(libraries)
    self.ldflags = tuple(ldflags)
    self._native_compiler: _C.Compiler | None = None

  @property
  def native(self) -> _C.Compiler | None:
    """Return the lazily-created raw compiler, or ``None`` for cache-only mode.

    Accessing this property in a build-capable mode resolves the SDK and native
    tools and may raise :class:`FileNotFoundError`; it still does not compile.
    """
    if self.cache_mode == _C.ArtifactCacheMode.cache_only:
      return None
    if self._native_compiler is None:
      sdk_path, sdk_layout = _sdk()
      sdk = _C.SdkSpec()
      sdk.path = sdk_path
      sdk.layout = sdk_layout
      toolchain = _C.ToolchainSpec()
      toolchain.c_compiler = _tool(self.cc, "CC", ("clang", "gcc"))
      toolchain.cxx_compiler = _tool(self.cxx, "CXX", ("clang++", "g++"))
      toolchain.cmake_program = _tool(self.cmake, "CMAKE", ("cmake",))
      toolchain.parallel_jobs = self.jobs or max(1, min(os.cpu_count() or 1, 8))
      toolchain.environment = self.environment
      config = _C.KernelCompilerConfig()
      config.sdk = sdk
      config.toolchain = toolchain
      config.work_directory = self.build_dir
      config.target_arch = self.target
      config.include_directories = self.include_dirs
      config.compile_options = self.cflags
      config.link_directories = self.library_dirs
      config.link_libraries = self.libraries
      config.link_options = self.ldflags
      self._native_compiler = _C.Compiler(config)
    return self._native_compiler

  def operator(self, source, kernel_def, *, recipe_id: str | None = None):
    """Create an :class:`~vecops.Operator` backed by one ``__kernel__`` source file."""
    from ._operator import Operator

    return Operator(source, kernel_def, self, recipe_id=recipe_id)


__all__ = ["Compiler", "default_cache_dir"]
