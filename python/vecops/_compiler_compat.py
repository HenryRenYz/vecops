# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Probe scalable-vector sinking and retain its exact build/cache policy."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

SINK_OPTION = "-mllvm=-simplifycfg-sink-common=false"
_PROBE = """#include <arm_sve.h>
void repro(bool cond, char* p) {
  if (cond) *reinterpret_cast<svint16_t*>(p) = svdup_s16(0);
  else *reinterpret_cast<svint16_t*>(p + 320) = svdup_s16(0);
}
"""


def scalable_sink_flags(compiler) -> tuple[str, ...]:
  """Use the workaround only after this executable fails the exact probe.

  Both the original failure and the successful narrowly modified compilation
  are required. Offline readers replay an attestation without finding tools.
  The selected option becomes an ordinary compiler flag before artifact keys
  are computed; it applies equally to generated DSL and source-C++ kernels.
  """
  from . import _C
  from ._cache_lock import file_lock
  from ._compiler import _tool
  from ._identity import digest, file_identity

  if compiler.target.lower() == "scalar" or compiler.vector_state.sve_bytes is None:
    return ()
  if SINK_OPTION in compiler.cflags:
    return ()
  policy = ("scalable-sink-v2", compiler.target, str(compiler.cxx), compiler.cflags,
            tuple(sorted(compiler.environment.items())), compiler.vector_state.env_token, _PROBE)
  directory = compiler.cache_dir / "identities"
  path = directory / f"scalable-sink-{digest(policy)}.json"
  if compiler.cache_mode == _C.ArtifactCacheMode.cache_only:
    if not path.is_file():
      raise ImportError("cache-only ARM execution has no matching scalable-sinking policy attestation")
    records = json.loads(path.read_text())
    choices = {tuple(record["flags"]) for record in records}
    if len(choices) != 1 or not choices <= {(), (SINK_OPTION,)}:
      raise ImportError("cache-only scalable-sinking policy is ambiguous; select a unique cache namespace")
    return next(iter(choices))
  tool = _tool(compiler.cxx, "CXX", ("clang++", "g++"))
  environment = {**os.environ, **compiler.environment}
  version = subprocess.run([str(tool), "--version"], capture_output=True, text=True,
                           timeout=5, check=True, env=environment, cwd=compiler.definition_cwd).stdout
  # A private wrapper may remain unchanged when its underlying Clang changes.
  # Attest the driver's reported installed binary as well as wrapper content.
  match = re.search(r"^InstalledDir:\s*(.+)$", version, re.MULTILINE)
  installed = Path(match.group(1).strip()) / "clang" if match else None
  identity = digest((file_identity(tool), version,
                     file_identity(installed) if installed is not None and installed.is_file() else None))
  directory.mkdir(parents=True, exist_ok=True)
  with file_lock(path.with_suffix(".lock")):
    records = json.loads(path.read_text()) if path.is_file() else []
    for record in records:
      if record["identity"] == identity:
        return tuple(record["flags"])
    with tempfile.TemporaryDirectory(prefix="scalable-sink-", dir=directory) as temporary:
      source = Path(temporary) / "probe.cpp"
      source.write_text(_PROBE)
      command = [str(tool), *shlex.split(environment.get("CXXFLAGS", "")), *compiler.cflags, "-O3", "-march=armv8-a+sve", "-msve-vector-bits=scalable",
                 "-c", str(source), "-o", str(Path(temporary) / "probe.o")]
      base = subprocess.run(command, capture_output=True, text=True, timeout=15,
                            env=environment, cwd=compiler.definition_cwd)
      flags = ()
      fixed = None
      if base.returncode and "Invalid size request on a scalable vector" in base.stderr:
        fixed = subprocess.run([*command, SINK_OPTION], capture_output=True, text=True, timeout=15,
                               env=environment, cwd=compiler.definition_cwd)
        if fixed.returncode:
          raise RuntimeError("scalable-vector compiler compatibility probe still fails with common sinking disabled")
        flags = (SINK_OPTION,)
      elif base.returncode:
        raise RuntimeError(f"ARM scalable-vector compatibility probe failed unexpectedly: {base.stderr}")
    record = {"identity": identity, "flags": flags, "base_exit": base.returncode,
              "base_stderr": base.stderr, "fixed_exit": None if fixed is None else fixed.returncode}
    records.append(record)
    descriptor, temporary = tempfile.mkstemp(prefix="scalable-sink-policy-", dir=directory)
    try:
      with os.fdopen(descriptor, "w") as stream:
        json.dump(records, stream)
      os.replace(temporary, path)
    finally:
      Path(temporary).unlink(missing_ok=True)
    return flags
