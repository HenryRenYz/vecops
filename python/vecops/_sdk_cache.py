# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Content/toolchain addressed source SDK installation for JIT build sessions."""

import json
import os
import subprocess
import tempfile
import time
from copy import copy
from pathlib import Path


def prepared_sdk(compiler, sdk, toolchain):
  """Build and publish one immutable SDK package under an interprocess lock.

  The generated kernel's private options do not affect the common runtime.
  CMake's inherited flags, compiler, SDK bytes and native target do. A package
  is consumed only after its exported targets and installed file hashes verify.
  Failed attempts are never published; source-tree fallback remains available
  by selecting BuildSession(reuse_sdk=False).
  """
  from ._build_session import snapshot_value
  from ._identity import compiler_identity, digest

  policy = copy(compiler)
  policy.environment = dict(compiler.environment)
  policy.target = "Native"
  policy.cache_namespace = "shared-sdk-v1"
  policy.cflags = policy.ldflags = policy.include_dirs = policy.library_dirs = policy.libraries = ()
  identity = (
    str(sdk.path),
    compiler_identity(policy),
    toolchain.build_type,
    str(toolchain.toolchain_file),
    str(toolchain.generator),
  )
  key = digest(identity)
  return snapshot_value("sdk-package", key, lambda: _prepare(compiler, sdk, toolchain, identity, key))


def _prepare(compiler, sdk, toolchain, identity, key):
  from . import _C
  from ._build_session import build_frontier
  from ._cache_lock import file_lock
  from ._identity import digest, tree_identity

  root = compiler.cache_dir.parent / "sdk-packages-v1"
  root.mkdir(parents=True, exist_ok=True)
  package = root / key
  manifest = package / "manifest.json"

  def installed_digest(prefix):
    # Inventory identity is relative to the publication directory, so an
    # atomic rename cannot change it. Every file's actual bytes are hashed.
    from ._watch import tree_hash

    return digest(tuple((str(Path(path).relative_to(prefix)), value) for path, value in tree_hash(prefix)))

  def verify():
    try:
      record = json.loads(manifest.read_text())
      return (
        record["key"] == key
        and record["installed_digest"] == installed_digest(package / "prefix")
        and any((package / "prefix").rglob("VecopsTargets.cmake"))
      )
    except (OSError, ValueError, KeyError):
      return False

  with file_lock(root / (key + ".lock")):
    if not verify():
      if package.exists():
        # An invalid publication is retained for diagnosis, never trusted.
        package.rename(root / (key + ".invalid-" + str(time.time_ns())))
      attempt = Path(tempfile.mkdtemp(prefix=key + ".attempt-", dir=root))
      # Configure against the final prefix, but publish only when install and
      # hashes succeed. The attempt is not visible as a usable package.
      prefix = attempt / "prefix"
      env = {**os.environ, **toolchain.environment}

      def run(command, stage):
        cpus = env.get("VECOPS_BUILD_CPU_LIST")
        if cpus:
          command = ["taskset", "-c", cpus, *command]
        with (attempt / (stage + ".log")).open("w") as log:
          subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)

      with build_frontier("sdk", key=key, parallelism=toolchain.parallel_jobs):
        command = [
          str(toolchain.cmake_program),
          "-S",
          str(sdk.path),
          "-B",
          str(attempt / "build"),
          "-DCMAKE_BUILD_TYPE=" + toolchain.build_type,
          "-DCMAKE_C_COMPILER=" + str(toolchain.c_compiler),
          "-DCMAKE_CXX_COMPILER=" + str(toolchain.cxx_compiler),
          "-DCMAKE_INSTALL_LIBDIR=lib",
          "-DCMAKE_INSTALL_PREFIX=" + str(prefix),
          "-DVECOPS_BUILD_TESTING=OFF",
          "-DVECOPS_BUILD_BENCHMARKS=OFF",
          "-DVECOPS_BUILD_COMPILER=OFF",
          "-DVECOPS_BUILD_PYTHON=OFF",
          "-DVECOPS_BUILD_COMMUNICATION=OFF",
          "-DVECOPS_BUILD_MEMORY=OFF",
          "-DVECOPS_BUILD_MEMORY_TOOLS=OFF",
        ]
        if toolchain.generator:
          command += ["-G", toolchain.generator]
        if toolchain.toolchain_file:
          command += ["-DCMAKE_TOOLCHAIN_FILE=" + str(toolchain.toolchain_file)]
        run(command, "configure")
        run(
          [str(toolchain.cmake_program), "--build", str(attempt / "build"), "--parallel", str(toolchain.parallel_jobs)],
          "build",
        )
        run([str(toolchain.cmake_program), "--install", str(attempt / "build")], "install")
        from ._build_session import current_build_session

        current_build_session().checkpoint()
        # Avoid retaining a snapshot of the temporary prefix after rename.
        (attempt / "manifest.json").write_text(
          json.dumps({"key": key, "identity": repr(identity), "installed_digest": installed_digest(prefix)}, indent=2)
          + "\n"
        )
        attempt.rename(package)
  if not verify():
    raise RuntimeError(f"shared SDK package verification failed: {package}")
  # Keep published bytes in the session's revalidation inventory too.
  tree_identity(package / "prefix")
  result = _C.SdkSpec()
  result.path = package / "prefix"
  result.layout = _C.SdkLayout.package
  return result
