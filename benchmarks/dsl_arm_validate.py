# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Run real DSL pytest kernels on a scheduled ARM compute allocation.

Source SDK is the default and should pass a separate smoke before a full
installed-SDK run. ``--reuse-installed-sdk`` explicitly reuses the SDK next
to the loaded native extension, avoiding repeated source-core builds. It does
not bypass kernel compilation or change vecops's installed SDK policy.

Examples (inside an exclusive lx-run compute job, after activating af3)::

  python benchmarks/dsl_arm_validate.py python/tests/test_dsl.py::test_control_flow
  python benchmarks/dsl_arm_validate.py --reuse-installed-sdk -- --maxfail=3 -q
  python benchmarks/dsl_arm_validate.py --reuse-installed-sdk --benchmark dsl_norm_catalog -- --list

For an authorized standalone ARM test host (such as 920f-4), first verify its
load and then pass ``--standalone-arm``. The default still requires Donau;
neither mode admits a login node. ``--allow-local`` is a separate x86/mock opt-in.

Pytest options such as ``-k`` go after ``--``. Explicit selectors go before it.
An explicitly authorized scratch allocation can use ``--scratch-root`` with
an existing, current-UID-owned, mode-0700 ``/tmp/vecops-*`` or
``/dev/shm/vecops-*`` directory. The default continues to use the user's home.
Default compiler and Torch-bridge caches are always fresh and scoped to this
validation run, rather than inherited from the user's application environment.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import runpy
import shutil
import stat
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_machine(*, allow_local=False, standalone_arm=False):
  if allow_local and standalone_arm:
    raise ValueError("select either local/mocked validation or standalone ARM, not both")
  if not allow_local and (platform.machine().lower() not in {"aarch64", "arm64"} or sys.platform != "linux"):
    raise RuntimeError("ARM validation must run on Linux AArch64 in a scheduled compute job")
  if not allow_local:
    if platform.node().split(".", 1)[0].lower().startswith("login"):
      raise RuntimeError("run ARM validation on a compute host, never on the login node")
    if standalone_arm:
      return  # Explicit opt-in after checking the standalone host's load/access.
    allocation = os.environ.get("CCS_ALLOC_FILE")
    if not os.environ.get("CCS_JOB_ID") or not allocation or not Path(allocation).is_file():
      raise RuntimeError("run ARM validation inside lx-run/Donau, never on the login node")


def compiler_path(requested):
  """Preserve clang++'s driver basename rather than resolving its symlink."""
  selected = shutil.which(requested)
  if selected is None:
    path = Path(requested).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
      raise FileNotFoundError(f"cannot find requested compiler {requested!r}")
    selected = str(path)
  return str(Path(selected).absolute())


def installed_sdk(module):
  root = (Path(module._C.__file__).absolute().parent / "_sdk").resolve()
  required = (
    root / "lib/cmake/Vecops/VecopsConfig.cmake",
    root / "lib/cmake/Vecops/VecopsTargets.cmake",
    root / "include/vecops/Kernel.h",
    root / "include/vecops/vec/Vec.h",
  )
  missing = [str(path) for path in required if not path.is_file()]
  libraries = [path for path in (root / "lib").glob("libvecops*") if path.is_file() and path.stat().st_size]
  if missing or not libraries:
    detail = ", ".join(missing) if missing else "nonempty libvecops library under lib/"
    raise RuntimeError(f"installed SDK is incomplete at {root}: missing {detail}")
  return root, module._C.SdkLayout.package


@contextlib.contextmanager
def sdk_selection(module, mode):
  """Override only this validation process, and always restore on pytest exit."""
  original = module._sdk
  if mode == "source":
    root, layout = original()
    if layout != module._C.SdkLayout.source_tree:
      raise RuntimeError("source smoke requires an editable source SDK; use --sdk installed only after that smoke")
    root = Path(root).resolve()
    if not (root / "CMakeLists.txt").is_file() or not (root / "include/vecops/Kernel.h").is_file():
      raise RuntimeError(f"source SDK is incomplete at {root}")
    yield root, layout
    return
  root, layout = installed_sdk(module)
  module._sdk = lambda: (root, layout)
  try:
    yield root, layout
  finally:
    module._sdk = original


def sdk_identity(root, *, installed):
  """Hash relevant SDK file contents, not timestamps or absolute paths."""
  roots = ("include", "lib") if installed else ("include", "src", "cmake")
  files = {path for name in roots for path in (root / name).rglob("*") if path.is_file()}
  if not installed:
    files.add(root / "CMakeLists.txt")
  digest = hashlib.sha256()
  total_bytes = 0
  for path in sorted(files):
    digest.update(str(path.relative_to(root)).encode())
    digest.update(b"\0")
    with path.open("rb") as stream:
      while block := stream.read(1024 * 1024):
        digest.update(block)
        total_bytes += len(block)
    digest.update(b"\0")
  return {"sha256": digest.hexdigest(), "files": len(files), "bytes": total_bytes}


def test_paths():
  return [str(path) for path in sorted((ROOT / "python/tests").glob("test_dsl*.py"))]


def benchmark_path(name):
  stem = name.removesuffix(".py")
  if not re.fullmatch(r"[A-Za-z0-9_]+", stem):
    raise ValueError("--benchmark accepts only a basename from benchmarks/*.py, not an arbitrary path")
  script = (ROOT / "benchmarks" / (stem + ".py")).resolve()
  if script.parent != ROOT / "benchmarks" or not script.is_file() or script.name == Path(__file__).name:
    raise ValueError(f"unknown or invalid benchmark {name!r}")
  return script


def run_benchmark(script, arguments):
  previous_argv, previous_path = sys.argv, list(sys.path)
  sys.argv = [str(script), *arguments]
  sys.path.insert(0, str(script.parent))
  try:
    runpy.run_path(str(script), run_name="__main__")
    return 0
  except SystemExit as exit_:
    if exit_.code is None or isinstance(exit_.code, int):
      return int(exit_.code or 0)
    raise
  finally:
    sys.argv = previous_argv
    sys.path[:] = previous_path


def reject_symlinks(path):
  """Do not follow scratch paths through aliases, including ancestor aliases."""
  if any(component.is_symlink() for component in (path, *path.parents)):
    raise ValueError(f"scratch paths must not contain symlinks: {path}")


def validate_scratch_root(requested):
  """Accept only an existing private, directly named task scratch directory."""
  path = Path(requested)
  if (
    not path.is_absolute()
    or ".." in path.parts
    or path.parent not in {Path("/tmp"), Path("/dev/shm")}
    or not path.name.startswith("vecops-")
    or path.name == "vecops-"
  ):
    raise ValueError("scratch root must be a task-private /tmp/vecops-* or /dev/shm/vecops-* directory")
  reject_symlinks(path)
  if not path.is_dir():
    raise ValueError(f"scratch root must be an existing directory: {path}")
  metadata = path.stat()
  if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
    raise ValueError(f"scratch root must be owned by the current UID with permissions 0700: {path}")
  return path


def resolve_output_path(requested, *, scratch_root=None):
  path = Path(requested).expanduser().absolute()
  if scratch_root is not None and (path.is_relative_to(Path("/tmp")) or path.is_relative_to(Path("/dev/shm"))):
    reject_symlinks(path)
  return path.resolve()


def output_path_allowed(path, *, scratch_root=None):
  home = Path.home().resolve()
  return path.is_relative_to(home) or (
    scratch_root is not None and path != scratch_root and path.is_relative_to(scratch_root)
  )


def fresh_basetemp(requested, *, allow_local=False, scratch_root=None):
  # pytest clears --basetemp recursively. Never silently give it a directory
  # holding prior results, source, or someone else's shared-cluster files.
  run = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"
  if scratch_root is not None:
    scratch_root = validate_scratch_root(scratch_root)
  default_root = scratch_root if scratch_root is not None else Path.home() / "tmp/vecops-arm-validation"
  path = resolve_output_path(requested or default_root / run, scratch_root=scratch_root)
  home = Path.home().resolve()
  if path in {Path(path.anchor), home, ROOT, home / "tmp", Path("/tmp"), Path("/dev/shm"), scratch_root}:
    raise ValueError("refusing a broad directory as pytest basetemp")
  if not allow_local and not output_path_allowed(path, scratch_root=scratch_root):
    raise ValueError("ARM pytest basetemp must stay under the user's home or the explicit private scratch root")
  if path.exists() and (not path.is_dir() or any(path.iterdir())):
    raise ValueError(f"pytest would erase existing data in basetemp {path}; select a fresh directory")
  path.parent.mkdir(parents=True, exist_ok=True)
  return path


@contextlib.contextmanager
def validation_cache_selection(basetemp):
  """Keep implicit compiler/bridge writes outside shared application caches.

  The sibling directory is not inside pytest's recursively cleared basetemp.
  Keep its contents after the run for inspection, and restore the caller's
  environment even if compilation or pytest fails. Explicit per-test compiler
  paths remain unaffected.
  """
  root = Path(basetemp).with_name(Path(basetemp).name + "-cache")
  reject_symlinks(root)
  root.mkdir(mode=0o700, exist_ok=False)
  paths = {
    "VECOPS_CACHE_DIR": root / "artifacts",
    "VECOPS_TORCH_BRIDGE_DIR": root / "torch-bridges",
    "VECOPS_BUILD_DIR": root / "builds",
  }
  previous = {name: os.environ.get(name) for name in paths}
  try:
    for name, path in paths.items():
      path.mkdir(mode=0o700)
      os.environ[name] = str(path)
    yield {name: str(path) for name, path in paths.items()}
  finally:
    for name, value in previous.items():
      if value is None:
        os.environ.pop(name, None)
      else:
        os.environ[name] = value


class FailureRecorder:
  """Persist failures immediately without flooding a long run's console.

  The usual pytest/JUnit report remains authoritative. Each JSONL event keeps
  the full diagnostic and frozen-run provenance, even before the suite finishes
  or if its scheduler later terminates the process.
  """

  def __init__(self, path, provenance):
    self.path = Path(path)
    self.provenance = dict(provenance)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to merge a previous run's failures into this run's evidence.
    with self.path.open("x", encoding="utf-8"):
      pass

  def record(self, report, phase):
    if not report.failed:
      return
    event = {
      "nodeid": report.nodeid,
      "phase": phase,
      "diagnostic": report.longreprtext,
      "provenance": self.provenance,
    }
    try:
      with self.path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=True) + "\n")
        stream.flush()
    except OSError as error:
      # A full filesystem or FD table must not turn the original kernel
      # failure into an unrelated pytest INTERNALERROR and discard its trace.
      print(f"[vecops] diagnostic file unavailable: {error}\n{report.longreprtext}", file=sys.stderr, flush=True)
    print(f"[vecops] test failed ({phase}): {report.nodeid}; diagnostics: {self.path}", flush=True)

  def pytest_runtest_logreport(self, report):
    self.record(report, report.when)

  def pytest_collectreport(self, report):
    self.record(report, "collection")


def main(argv=None):
  arguments = list(sys.argv[1:] if argv is None else argv)
  split = arguments.index("--") if "--" in arguments else len(arguments)
  own, extra = arguments[:split], arguments[split + 1 :]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sdk", choices=("source", "installed"), default="source")
  parser.add_argument("--reuse-installed-sdk", action="store_true")
  parser.add_argument("--cc", default="clang")
  parser.add_argument("--cxx", default="clang++")
  parser.add_argument("--benchmark", help="Run this checked benchmarks/*.py basename instead of pytest")
  parser.add_argument("--scratch-root", type=Path, help="Explicit private 0700 /tmp/vecops-* or /dev/shm/vecops-* root")
  parser.add_argument("--basetemp", type=Path)
  parser.add_argument("--junitxml", type=Path)
  parser.add_argument("--failure-log", type=Path, help="Optional immediate failed-test diagnostics as JSONL")
  parser.add_argument("--allow-local", action="store_true", help="Explicit opt-in for local/mocked runner tests")
  parser.add_argument(
    "--standalone-arm", action="store_true", help="Explicit opt-in for a verified standalone ARM host"
  )
  parser.add_argument("selectors", nargs="*")
  args = parser.parse_args(own)
  if args.benchmark and args.selectors:
    parser.error("benchmark arguments go after --; pytest selectors cannot be combined with --benchmark")
  if args.benchmark and args.failure_log:
    parser.error("--failure-log is for pytest, not benchmark execution")
  if not args.benchmark and any(
    option in {"--basetemp", "--junitxml"} or option.startswith(("--basetemp=", "--junitxml=")) for option in extra
  ):
    parser.error("pass --basetemp/--junitxml to the runner before -- so their output paths can be validated")
  benchmark = benchmark_path(args.benchmark) if args.benchmark else None
  check_machine(allow_local=args.allow_local, standalone_arm=args.standalone_arm)
  cc, cxx = compiler_path(args.cc), compiler_path(args.cxx)
  os.environ.update(CC=cc, CXX=cxx, VECOPS_TEST_TARGET="Native")
  scratch_root = validate_scratch_root(args.scratch_root) if args.scratch_root else None
  basetemp = fresh_basetemp(args.basetemp, allow_local=args.allow_local, scratch_root=scratch_root)
  junit = resolve_output_path(args.junitxml or basetemp.parent / (basetemp.name + ".xml"), scratch_root=scratch_root)
  if junit in {Path(junit.anchor), Path.home().resolve(), ROOT, Path("/tmp"), Path("/dev/shm"), scratch_root}:
    raise ValueError("refusing a broad directory as ARM junitxml")
  if not args.allow_local and not output_path_allowed(junit, scratch_root=scratch_root):
    raise ValueError("ARM junitxml must stay under the user's home or the explicit private scratch root")
  junit.parent.mkdir(parents=True, exist_ok=True)
  failure_log = resolve_output_path(args.failure_log, scratch_root=scratch_root) if args.failure_log else None
  if failure_log is not None:
    if not args.allow_local and not output_path_allowed(failure_log, scratch_root=scratch_root):
      raise ValueError("ARM failure log must stay under the user's home or explicit private scratch root")
    if failure_log.exists() or failure_log == junit:
      raise ValueError("failure log must be a fresh, independent output file")
  import pytest
  import vecops._compiler as compiler
  from vecops._arm_vector_state import get_vector_state
  from vecops._identity import implementation_identity

  mode = "installed" if args.reuse_installed_sdk else args.sdk
  with validation_cache_selection(basetemp) as caches, sdk_selection(compiler, mode) as (root, layout):
    vector_state = get_vector_state()
    provenance = {
      "mode": mode,
      "sdk_root": str(root),
      "sdk_layout": str(layout),
      "sdk_content": sdk_identity(root, installed=mode == "installed"),
      "source_implementation_identity": implementation_identity(),
      "working_directory": str(Path.cwd()),
      "test_selectors": args.selectors or test_paths(),
      "python_compiler": str(Path(compiler.__file__).resolve()),
      "native_extension": str(Path(compiler._C.__file__).resolve()),
      "cc": cc,
      "cxx": cxx,
      "target": "Native",
      "vector_state": {
        "sve_bytes": vector_state.sve_bytes,
        "sme_bytes": vector_state.sme_bytes,
        "env_token": vector_state.env_token,
      },
      "machine": platform.machine(),
      "hostname": platform.node(),
      "job_id": os.environ.get("CCS_JOB_ID"),
      "affinity": sorted(os.sched_getaffinity(0)),
      "scratch_root": str(scratch_root) if scratch_root is not None else None,
      "basetemp": str(basetemp),
      "validation_caches": caches,
      "junitxml": str(junit),
    }
    print("ARM validation provenance: " + json.dumps(provenance, sort_keys=True), flush=True)
    if benchmark:
      return run_benchmark(benchmark, extra)
    arguments = [*(args.selectors or test_paths()), f"--basetemp={basetemp}", f"--junitxml={junit}", *extra]
    if failure_log is not None:
      return int(pytest.main(arguments, plugins=[FailureRecorder(failure_log, provenance)]))
    return int(pytest.main(arguments))


if __name__ == "__main__":
  raise SystemExit(main())
