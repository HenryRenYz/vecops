# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Materialize a hash-pinned ARM C++ header environment before Python imports.

This intentionally imports only the Python standard library. In particular,
do not add NumPy, Torch, vecops, or the project compiler modules here: callers
must set LD_PRELOAD before those libraries can select another libgcc runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def assert_no_symlink_path(path: Path) -> None:
  absolute = Path(os.path.abspath(path))
  for component in (absolute, *absolute.parents):
    if component.is_symlink():
      raise RuntimeError(f"refusing symlink path component: {component}")


def safe_archive_path(raw: str) -> PurePosixPath:
  path = PurePosixPath(raw)
  if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
    raise ValueError(f"unsafe toolchain bundle member: {raw}")
  return path


def _check_readable_regular(path: Path, description: str) -> Path:
  assert_no_symlink_path(path)
  resolved = Path(path).resolve(strict=True)
  info = resolved.stat()
  if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.R_OK):
    raise RuntimeError(f"{description} must be a readable regular file: {resolved}")
  return resolved


def _private_root() -> Path:
  temp_root = Path("/tmp")
  assert_no_symlink_path(temp_root)
  root = Path(tempfile.mkdtemp(prefix="vecops-arm-job-", dir=temp_root))
  root.chmod(0o700)
  info = root.stat()
  if root.is_symlink() or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
    raise RuntimeError(f"temporary toolchain root must be UID-owned and mode 0700: {root}")
  return root


def _extract_bundle(
  bundle: Path,
  expected_sha256: str,
  destination: Path,
  *,
  allowed_runtime_links: dict[str, tuple[Path, str]] | None = None,
) -> tuple[Path, ...]:
  assert_no_symlink_path(bundle)
  assert_no_symlink_path(destination)
  if sha256_file(bundle) != expected_sha256:
    raise RuntimeError("standard-library bundle SHA-256 mismatch")
  if destination.exists() or destination.is_symlink():
    raise FileExistsError(destination)
  verified_links: dict[str, str] = {}
  for member_name, (runtime_path, expected_runtime_sha256) in (allowed_runtime_links or {}).items():
    relative = safe_archive_path(member_name).as_posix()
    runtime = _check_readable_regular(runtime_path, "allowed durable runtime link target")
    if sha256_file(runtime) != expected_runtime_sha256:
      raise RuntimeError(f"allowed runtime link target SHA-256 mismatch: {runtime}")
    verified_links[relative] = str(runtime)
  with tarfile.open(bundle, "r:gz") as archive:
    members = list(archive)
  names: set[str] = set()
  symlink_names: set[str] = set()
  for member in members:
    path = safe_archive_path(member.name).as_posix()
    if path in names:
      raise ValueError(f"duplicate toolchain bundle member: {path}")
    names.add(path)
    if member.issym():
      expected_target = verified_links.get(path)
      if expected_target is None or member.linkname != expected_target:
        raise ValueError(f"toolchain bundle links/special files include an unapproved runtime symlink: {member.name}")
      symlink_names.add(path)
    elif not (member.isdir() or member.isfile()):
      raise ValueError(f"toolchain bundle links/special files are forbidden: {member.name}")
  for path in names:
    if any(path.startswith(link_name + "/") for link_name in symlink_names):
      raise ValueError(f"toolchain bundle member is nested below a runtime symlink: {path}")
  if verified_links and symlink_names != set(verified_links):
    raise ValueError(
      "toolchain bundle runtime symlink set differs from the pinned layout: "
      f"missing={sorted(set(verified_links) - symlink_names)}, unexpected={sorted(symlink_names - set(verified_links))}"
    )
  destination.mkdir(parents=True, mode=0o700)
  extracted = []
  with tarfile.open(bundle, "r:gz") as archive:
    for member in members:
      relative = safe_archive_path(member.name)
      target = destination.joinpath(*relative.parts)
      if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        continue
      target.parent.mkdir(parents=True, exist_ok=True)
      if member.issym():
        os.symlink(verified_links[relative.as_posix()], target)
        extracted.append(target)
        continue
      payload = archive.extractfile(member)
      if payload is None:
        raise ValueError(f"cannot read toolchain bundle member: {member.name}")
      with target.open("xb") as output:
        shutil.copyfileobj(payload, output)
      target.chmod(member.mode & 0o755)
      extracted.append(target)
  return tuple(extracted)


def _validate_af3(python: str, prefix: str) -> dict[str, str]:
  selected = Path(python).resolve(strict=True)
  actual = Path(sys.executable).resolve(strict=True)
  actual_prefix = Path(sys.prefix).resolve(strict=True)
  selected_prefix = Path(prefix).resolve(strict=True)
  if selected != actual or selected_prefix != actual_prefix:
    raise RuntimeError(
      f"active interpreter mismatch: running={actual} prefix={actual_prefix}; "
      f"selected={selected} prefix={selected_prefix}"
    )
  if not actual_prefix.name.startswith("af3"):
    raise RuntimeError(f"active environment is not af3: {actual_prefix}")
  return {
    "executable": str(actual),
    "prefix": str(actual_prefix),
    "version": sys.version,
    "executable_sha256": sha256_file(actual),
  }


def _run(command: list[str], *, environment: dict[str, str], log: Path) -> None:
  result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
  log.with_suffix(".stdout.log").write_text(result.stdout)
  log.with_suffix(".stderr.log").write_text(result.stderr)
  if result.returncode:
    raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)


def prepare(
  *,
  python: str,
  conda_prefix: str,
  clang: Path,
  clangxx: Path,
  bundle: Path,
  bundle_sha256: str,
  libgcc: Path,
  libgcc_sha256: str,
  libstdcxx: Path,
  libstdcxx_sha256: str,
  shell_path: Path,
  probe_path: Path,
) -> dict[str, object]:
  """Verify inputs, extract headers, run Clang compile/link probes, emit env."""
  identity = _validate_af3(python, conda_prefix)
  selected_shell = _check_readable_regular(shell_path, "sourceable project shell")
  selected_probe = _check_readable_regular(probe_path, "C++ toolchain probe")
  selected_helper = _check_readable_regular(Path(__file__), "job-environment helper")
  project_root = selected_probe.parent.parent
  print_header = _check_readable_regular(project_root / "include/vecops/execution/Print.h", "project print header")
  compiler_paths = {}
  for label, raw in (("clang", clang), ("clangxx", clangxx)):
    invocation = Path(os.path.abspath(raw))
    executable = invocation.resolve(strict=True)
    if not invocation.is_file() or not os.access(invocation, os.X_OK):
      raise RuntimeError(f"explicit {label} is not executable: {invocation}")
    version = subprocess.run([str(invocation), "--version"], check=True, capture_output=True, text=True).stdout
    if "clang" not in version.lower():
      raise RuntimeError(f"explicit compiler is not Clang: {invocation}")
    compiler_paths[label] = {
      # Keep the driver invocation path: clang++ may be a symlink whose target
      # is clang-19, but invoking that target directly changes driver mode.
      "path": str(invocation),
      "resolved_path": str(executable),
      "version": version.splitlines()[0],
      "sha256": sha256_file(executable),
    }

  selected_bundle = _check_readable_regular(bundle, "GCC14 header bundle")
  selected_libgcc = _check_readable_regular(libgcc, "durable libgcc runtime")
  selected_libstdcxx = _check_readable_regular(libstdcxx, "durable libstdc++ runtime")
  for label, path, expected in (
    ("GCC14 bundle", selected_bundle, bundle_sha256),
    ("libgcc runtime", selected_libgcc, libgcc_sha256),
    ("libstdc++ runtime", selected_libstdcxx, libstdcxx_sha256),
  ):
    actual = sha256_file(path)
    if actual != expected:
      raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")
  if selected_libgcc.parent != selected_libstdcxx.parent:
    raise RuntimeError("durable libgcc/libstdc++ must share one runtime directory")

  root = _private_root()
  installation = root / "installation"
  extracted = _extract_bundle(
    selected_bundle,
    bundle_sha256,
    installation,
    allowed_runtime_links={
      "installation/lib/libgcc_s.so": (selected_libgcc, libgcc_sha256),
      "installation/lib/libgcc_s.so.1": (selected_libgcc, libgcc_sha256),
      "installation/lib/libstdc++.so": (selected_libstdcxx, libstdcxx_sha256),
      "installation/lib/libstdc++.so.6": (selected_libstdcxx, libstdcxx_sha256),
    },
  )
  installations = sorted(
    {
      item.parent
      for item in extracted
      if item.name == "libgcc.a"
      and re.fullmatch(r"14(?:\.\d+)*", item.parent.name)
      and (item.parent / "include/c++/format").is_file()
    }
  )
  if len(installations) != 1:
    raise RuntimeError(f"expected one GCC14 libgcc selector/header tree, found {installations}")
  gcc_install = installations[0]
  include_root = gcc_install / "include/c++"
  target_include = include_root / gcc_install.parent.name
  backward_include = include_root / "backward"
  if not (include_root / "format").is_file() or not target_include.is_dir() or not backward_include.is_dir():
    raise RuntimeError("bundle lacks format, target-specific, or backward C++ headers")

  runtime_dir = selected_libgcc.parent
  if root == runtime_dir or root in runtime_dir.parents:
    raise RuntimeError("durable C++ runtimes must not live below the ephemeral job root")
  base_env = dict(os.environ)
  old_cxxflags = shlex.split(base_env.get("CXXFLAGS", ""))
  old_ldflags = shlex.split(base_env.get("LDFLAGS", ""))
  if any(flag.startswith(("--gcc-install-dir", "--gcc-toolchain")) for flag in old_cxxflags):
    raise RuntimeError("inherited CXXFLAGS select another GCC tree; remove stale selector before sourcing")
  if any("/tmp/" in flag or "/dev/shm/" in flag for flag in old_cxxflags + old_ldflags):
    raise RuntimeError("inherited flags contain ephemeral toolchain paths")
  private_flags = [
    f"--gcc-install-dir={gcc_install}",
    "-nostdinc++",
    "-isystem",
    str(include_root),
    "-isystem",
    str(target_include),
    "-isystem",
    str(backward_include),
    f"-L{runtime_dir}",
    f"-Wl,-rpath,{runtime_dir}",
    f"-L{gcc_install}",
  ]
  env = dict(base_env)
  env["CC"] = compiler_paths["clang"]["path"]
  env["CXX"] = compiler_paths["clangxx"]["path"]
  env["VECOPS_TORCH_BRIDGE_CC"] = env["CC"]
  env["VECOPS_TORCH_BRIDGE_CXX"] = env["CXX"]
  env["CXXFLAGS"] = shlex.join(old_cxxflags + private_flags)
  env["LDFLAGS"] = shlex.join(old_ldflags + [f"-L{runtime_dir}", f"-Wl,-rpath,{runtime_dir}"])
  env["LD_PRELOAD"] = ":".join(
    (str(selected_libgcc), str(selected_libstdcxx), *filter(None, (base_env.get("LD_PRELOAD", ""),)))
  )
  old_library_path = base_env.get("LD_LIBRARY_PATH", "")
  env["LD_LIBRARY_PATH"] = ":".join((str(runtime_dir), *filter(None, (old_library_path,))))

  probe_root = root / "probe"
  probe_root.mkdir(mode=0o700)
  syntax_log = probe_root / "syntax"
  syntax_argv = [
    env["CXX"],
    "-std=c++20",
    "-pthread",
    *shlex.split(env["CXXFLAGS"]),
    f"-I{project_root / 'include'}",
    "-fsyntax-only",
    str(selected_probe),
  ]
  _run(
    syntax_argv,
    environment=env,
    log=syntax_log,
  )
  executable = probe_root / "stdlib-probe"
  link_log = probe_root / "link"
  link_argv = [
    env["CXX"],
    "-std=c++20",
    "-O2",
    "-pthread",
    *shlex.split(env["CXXFLAGS"]),
    f"-I{project_root / 'include'}",
    str(selected_probe),
    *shlex.split(env["LDFLAGS"]),
    "-o",
    str(executable),
  ]
  _run(
    link_argv,
    environment=env,
    log=link_log,
  )
  run_log = probe_root / "run"
  run_argv = [str(executable)]
  _run(run_argv, environment=env, log=run_log)
  readelf = shutil.which("readelf")
  probe_dynamic = None
  if readelf is not None:
    probe_dynamic = subprocess.run([readelf, "-d", str(executable)], check=True, capture_output=True, text=True).stdout

  manifest = {
    "schema": "vecops-arm-job-env:v1",
    "active_interpreter": identity,
    "clang": compiler_paths["clang"]["path"],
    "clangxx": compiler_paths["clangxx"]["path"],
    "compiler_identity": compiler_paths,
    "helper_path": str(selected_helper),
    "helper_sha256": sha256_file(selected_helper),
    "source_shell_path": str(selected_shell),
    "source_shell_sha256": sha256_file(selected_shell),
    "probe_path": str(selected_probe),
    "probe_sha256": sha256_file(selected_probe),
    "print_header_path": str(print_header),
    "print_header_sha256": sha256_file(print_header),
    "bundle_path": str(selected_bundle),
    "bundle_sha256": bundle_sha256,
    "private_root": str(root),
    "gcc_install": str(gcc_install),
    "private_gcc_install": str(gcc_install),
    "include_dirs": [str(include_root), str(target_include), str(backward_include)],
    "runtime": {
      "libgcc": {"path": str(selected_libgcc), "sha256": libgcc_sha256},
      "libstdcxx": {"path": str(selected_libstdcxx), "sha256": libstdcxx_sha256},
    },
    "probe": {"syntax": "passed", "link": "passed", "run": "passed"},
    "probe_commands": {"syntax": syntax_argv, "link": link_argv, "run": run_argv},
    "probe_executable": str(executable),
    "probe_executable_sha256": sha256_file(executable),
    "probe_readelf_dynamic": probe_dynamic,
  }
  manifest_path = root / "job-env-manifest.json"
  manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
  manifest_path.chmod(0o600)
  env_path = root / "job-env.sh"
  exports = {
    **{
      key: env[key]
      for key in (
        "CC",
        "CXX",
        "VECOPS_TORCH_BRIDGE_CC",
        "VECOPS_TORCH_BRIDGE_CXX",
        "CXXFLAGS",
        "LDFLAGS",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
      )
    },
    "VECOPS_ARM_PRIVATE_STDLIB_ROOT": str(root),
    "VECOPS_ARM_PRIVATE_GCC_INSTALL": str(gcc_install),
    "VECOPS_ARM_JOB_ENV_MANIFEST": str(manifest_path),
  }
  env_path.write_text("\n".join(f"export {key}={shlex.quote(value)}" for key, value in exports.items()) + "\n")
  env_path.chmod(0o600)
  print(f"[arm-job-env] verified private toolchain at {root}", file=sys.stderr)
  return {"private_root": root, "job_env": env_path, "manifest": manifest_path, "identity": manifest}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  prepare_parser = sub.add_parser("prepare")
  for option in (
    "python",
    "conda-prefix",
    "clang",
    "clangxx",
    "bundle",
    "bundle-sha256",
    "libgcc",
    "libgcc-sha256",
    "libstdcxx",
    "libstdcxx-sha256",
    "shell-path",
    "probe-path",
  ):
    prepare_parser.add_argument(f"--{option}", required=True)
  args = parser.parse_args()
  if args.command == "prepare":
    result = prepare(
      python=args.python,
      conda_prefix=args.conda_prefix,
      clang=Path(args.clang),
      clangxx=Path(args.clangxx),
      bundle=Path(args.bundle),
      bundle_sha256=args.bundle_sha256,
      libgcc=Path(args.libgcc),
      libgcc_sha256=args.libgcc_sha256,
      libstdcxx=Path(args.libstdcxx),
      libstdcxx_sha256=args.libstdcxx_sha256,
      shell_path=Path(args.shell_path),
      probe_path=Path(args.probe_path),
    )
    print(result["job_env"])


if __name__ == "__main__":
  main()
