# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Repair a corrected full-suite plan without trusting old execution IDs alone.

DSL reuse requires exact generated text (apart from the one compiler environment
comment) and the complete adapter ABI. Explicit --allow-codegen-change permits
only affine analysis, CPU recipes, and source DSL packing helper changes;
each reused role additionally needs
complete archived capture, task flags and toolchain policy proof. All other
runtime/SDK bytes remain identical. Old binaries and archives stay untouched.
CPP references are conservatively rebuilt. Native chunks retain partial successes;
only missing groups are split/retried. Lowering runs in separate timeout processes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import dsl_matmul_full_suite as suite

NORMALIZATION = "Remove only a first line matching '// compiler environment: [0-9a-f]+\\n'; preserve all other bytes."


def digest(data):
  return hashlib.sha256(data).hexdigest()


def normalized(data):
  return re.sub(rb"\A// compiler environment: [0-9a-f]+\n", b"", data, count=1)


def canonical(value):
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


CODEGEN_CHANGE_PATHS = frozenset(
  {
    "python/vecops/language/_compiler/analysis/affine.py",
    "python/vecops/language/_compiler/backends/cpu/recipes.py",
    # Source DSL helper bodies are elaborated into the exact generated TU;
    # this exception does not admit runtime APIs or library export changes.
    "python/vecops/language/_library/packing.py",
  }
)


def proof_identity(plan, runtime=None):
  result = {key: plan[key] for key in ("runner_hashes", "pass_options", "target", "toolchain", "compiler_cflags")}
  if runtime is not None:
    result["runtime_sha256"] = digest(canonical(runtime).encode())
  return result


def validate_runtime_change(old, new, allowed):
  """Only explicitly named existing pure-codegen files may change bytes."""
  allowed = set(allowed)
  if not allowed <= CODEGEN_CHANGE_PATHS:
    raise ValueError("unsupported --allow-codegen-change path")
  if old.keys() != new.keys():
    raise ValueError("runtime file inventory differs; missing/extra files cannot be allowed")
  changed = {name: {"old_sha256": old[name], "new_sha256": new[name]} for name in old if old[name] != new[name]}
  if not set(changed) <= allowed:
    raise ValueError("undeclared runtime Python/native extension or SDK change; reuse disabled")
  return changed


def guard_runtime(args):
  old_plan = suite.read(args.old_suite / "plan.json")
  old = runtime_identity(Path(old_plan["snapshot"]))
  new = runtime_identity(args.snapshot)
  changes = validate_runtime_change(old, new, getattr(args, "allow_codegen_change", ()))
  return {"old": old, "new": new, "allowed_changes": changes}


def runtime_identity(snapshot):
  """Exact bytes of all runtime Python/extensions and public SDK header inputs."""
  paths = [
    p
    for p in (snapshot / "python/vecops").rglob("*")
    if p.is_file() and (p.suffix in (".py", ".so") or ".so." in p.name)
  ]
  for folder in ("include", "src", "cmake", "python/vecops/_sdk"):
    paths += [p for p in (snapshot / folder).rglob("*") if p.is_file()]
  if (snapshot / "CMakeLists.txt").is_file():
    paths.append(snapshot / "CMakeLists.txt")
  return {str(p.relative_to(snapshot)): digest(p.read_bytes()) for p in sorted(paths)}


def archive_members(record):
  archive = Path(record["evidence_archive"])
  if digest(archive.read_bytes()) != record["evidence_sha256"]:
    raise ValueError(f"evidence archive changed: {archive}")
  with tarfile.open(archive, "r:gz") as stream:
    return {item.name: stream.extractfile(item).read() for item in stream.getmembers() if item.isfile()}


def archived_path(members, original):
  # retain() preserves paths relative to each group's owned work directory.
  suffix = "work/" + str(original).split("/work/", 1)[1] if "/work/" in str(original) else None
  if suffix is None or suffix not in members:
    raise ValueError(f"exact original evidence path missing: {original}")
  return members[suffix]


def candidates(old_suite, ids):
  """Return strictly task-associated sources/adapters/binaries, including failures."""
  wanted = set(ids)
  plan = suite.read(old_suite / "plan.json")
  for group in plan["groups"]:
    if not wanted.intersection(group["execution_ids"]):
      continue
    directory = old_suite / "groups" / group["id"]
    record = suite.read(directory / "index.json")
    if not record or not record.get("evidence_archive"):
      record = suite.read(directory / "failure-evidence.json", {})
    if not record.get("evidence_archive"):
      continue
    members = archive_members(record)
    mapping = record.get("retained_binary_mapping", {})
    for original, retained in mapping.items():
      # Cache aliases lack a task/adapter relationship and are deliberately ignored.
      match = re.search(r"(.*?/kernel-batch-[^/]+)/artifacts/([^/]+)/[^/]+\.so$", original)
      if not match:
        continue
      adapter = match[1] + "/tasks/" + match[2] + "/input/vecops_adapter.cpp"
      try:
        adapter_data = archived_path(members, adapter)
        source = re.search(rb'#include "([^"\n]+\.cpp)"', adapter_data).group(1).decode()
        if Path(source).name == "dsl_matmul_source_reference.cpp":
          continue
        source_data = archived_path(members, source)
      except (ValueError, AttributeError):
        continue
      binary = Path(retained)
      expected = record.get("retained_binary_sha256", {}).get(str(binary))
      if expected is None or not binary.is_file() or digest(binary.read_bytes()) != expected:
        raise ValueError(f"retained binary changed or unattested: {binary}")
      yield {
        "binary": str(binary),
        "binary_sha256": expected,
        "source": source_data,
        "adapter": adapter_data.decode(),
        "original_binary": original,
        "archive": record["evidence_archive"],
        "archive_sha256": record["evidence_sha256"],
        "execution_ids": sorted(wanted.intersection(group["execution_ids"])),
        "spec": members.get("work/" + adapter.split("/work/", 1)[1].rsplit("/", 1)[0] + "/vecops_spec.h"),
        "build_evidence": archived_build_evidence(members, adapter),
      }


def archived_build_evidence(members, adapter):
  """Associate a command with this exact adapter; retain complete identity components."""
  commands, identities = [], []
  for name, data in members.items():
    if name.endswith("/compile_commands.json"):
      commands.extend(item for item in json.loads(data) if item.get("file") == adapter)
    elif "/identities/" in name and name.endswith(".json"):
      identities.extend(json.loads(data))
  return {"commands": commands, "identities": identities}


def spec_captures(data):
  """Parse known emitted captures only; unknown spec syntax fails closed."""
  if data is None:
    return None
  result = {}
  for line in data.decode().splitlines():
    line = line.strip()
    if not line or line.startswith(("#", "namespace ", "} //")):
      continue
    value = re.fullmatch(r"inline constexpr auto (\w+) = ::vecops::meta::cint<(-?\d+)>;", line)
    dtype = re.fullmatch(r"using (\w+) = (?:::vecops::)?(\w+_t);", line)
    if value:
      result[value[1]] = int(value[2])
    elif dtype:
      result[dtype[1]] = dtype[2]
    elif re.fullmatch(r"using \w+ = ::vecops::meta::Dynamic<[-\w:, ]+>;", line):
      # Runtime dimension constraints and relations are covered by the full ABI.
      continue
    else:
      return None
  return result


def task_command_matches(fresh, candidate, command):
  """Reject task flags not accounted for by policy or the fixed CMake scaffold."""
  arguments = list(command.get("arguments") or shlex.split(command["command"]))
  if not arguments or not any(tool["path"] == arguments[0] for tool in fresh["old_build_components"]["tools"]):
    return False
  remaining = arguments[1:]
  for flag in [*fresh.get("environment_cxxflags", ()), *fresh["effective_cflags"]]:
    if flag not in remaining:
      return False
    remaining.remove(flag)
  adapter = command["file"]
  roots = {str(Path(adapter).parent), *fresh.get("compile_include_roots", ())}
  scaffold = {
    "-O3",
    "-DNDEBUG",
    "-std=c++20",
    "-fPIC",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
    "-march=native",
    "-mcpu=native",
    "-mtune=native",
    "-Wno-ignored-attributes",
    "-falign-functions=64",
    "-falign-loops=64",
    "-fno-math-errno",
  }
  source_count = output_count = 0
  iterator = iter(remaining)
  for flag in iterator:
    if flag == "-c":
      if next(iterator, None) != adapter:
        return False
      source_count += 1
    elif flag == "-o":
      output = next(iterator, "")
      if not output.endswith("vecops_adapter.cpp.o"):
        return False
      output_count += 1
    elif flag.startswith("-I"):
      if flag[2:] not in roots:
        return False
    elif re.fullmatch(r"-Dvecops_kernel_[0-9a-f]+_\d+_EXPORTS", flag):
      continue
    elif flag not in scaffold:
      return False
  return source_count == output_count == 1


def codegen_role_evidence(fresh, candidate):
  """Require captures, full effective build policy and task-specific command evidence."""
  if spec_captures(candidate.get("spec")) != fresh.get("captures") or "captures" not in fresh:
    return None
  expected = fresh.get("old_build_components")
  evidence = candidate.get("build_evidence", {})
  if not expected or not any(item.get("components") == expected for item in evidence.get("identities", ())):
    return None
  commands = evidence.get("commands", ())
  if len(commands) != 1:
    return None
  if not task_command_matches(fresh, candidate, commands[0]):
    return None
  return {"captures": fresh["captures"], "build_components": expected, "compile_command": commands[0]}


def old_build_components(compiler, snapshot):
  """Recompute the unchanged build policy against the old source implementation.

  _identity.py is outside the permitted changes. Its exact policy tuple is mirrored
  here; its current attested components still prove tools/system/runtime/hardware.
  Unsupported SDK layouts conservatively disable reuse rather than guess identity.
  """
  from vecops import _identity as identity

  extensions = list((snapshot / "python/vecops").glob("_C*.so"))
  if len(extensions) != 1 or (snapshot / "python/vecops/_sdk").exists():
    return None
  items = [
    sys.version,
    identity.file_identity(extensions[0]),
    identity.tree_identity(snapshot / "python/vecops", {".py"}),
    identity.tree_identity(snapshot / "include"),
    identity.tree_identity(snapshot / "src"),
    identity.tree_identity(snapshot / "cmake"),
    identity.file_identity(snapshot / "CMakeLists.txt"),
  ]
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
  libraries = []
  for name in compiler.libraries:
    path = Path(name)
    if path.is_file():
      libraries.append(identity.file_identity(path))
    else:
      for root in compiler.library_dirs:
        for extension in (".so", ".a", ".dylib"):
          path = root / f"lib{name}{extension}"
          if path.is_file():
            libraries.append(identity.file_identity(path))
  current = identity.compiler_identity(compiler)
  records = [
    item
    for path in (compiler.cache_dir / "identities").glob("*.json")
    for item in json.loads(path.read_text())
    if item["identity"] == current
  ]
  if len(records) != 1 or "components" not in records[0]:
    return None
  return records[0]["components"] | {"config_key": identity.digest((policy, identity.digest(items), libraries))}


def prove_reuse(fresh, candidate, contract):
  import dsl_matmul_artifact_ab as ab

  if normalized(fresh) != normalized(candidate["source"]):
    return None
  if canonical(ab.adapter_contract(candidate["adapter"])) != canonical(contract):
    return None
  return {
    key: candidate[key]
    for key in ("binary", "binary_sha256", "original_binary", "archive", "archive_sha256", "execution_ids")
  } | {
    "old_raw_sha256": digest(candidate["source"]),
    "new_raw_sha256": digest(fresh),
    "normalized_sha256": digest(normalized(fresh)),
    "normalization": NORMALIZATION,
    "abi_sha256": digest(canonical(contract).encode()),
  }


def source_bytes(proof):
  path = Path(proof["source"])
  data = path.read_bytes()
  if proof.get("compression") == "gzip":
    if digest(data) != proof["compressed_sha256"]:
      raise ValueError("compressed prelowered source changed")
    data = gzip.decompress(data)
  elif proof.get("compression") is not None:
    raise ValueError("unknown prelowered compression")
  if digest(data) != proof["raw_sha256"]:
    raise ValueError("prelowered source changed")
  return data


def materialize_proof(proof, work):
  """Keep only the active chunk's exact raw C++ source on disk."""
  data = source_bytes(proof)
  path = work / "dsl-sources" / proof["raw_sha256"] / "kernel.cpp"
  path.parent.mkdir(parents=True, exist_ok=True)
  if not path.exists():
    path.write_bytes(data)
  elif path.read_bytes() != data:
    raise ValueError("materialized prelowered source changed")
  return {key: value for key, value in proof.items() if key not in ("compression", "compressed_sha256")} | {
    "source": str(path)
  }


class BatchInputAttestationError(RuntimeError):
  """A private compilation batch cannot publish any newly built artifacts."""


class BatchInputAttestation:
  """Opt-in harness guard: full SDK/toolchain checks surround a private batch.

  The native source fingerprint, namespace, recipe and parameter definition are
  identical to normal Operator. Each native callback still validates the full
  source/dependency/user-include closure. Shared SDK/system trees are checked at
  first policy use, before native build, and after it, never replaced by a label.
  """

  def __init__(self, work):
    self.work = Path(work).resolve()
    self.policies = {}
    self.sources = []
    self.invalid_reason = None
    self.events = []

  @staticmethod
  def _policy_key(compiler):
    from vecops.language._compiler.cache import _compiler_key

    external = tuple(
      (name, compiler.environment.get(name, os.environ.get(name)))
      for name in (
        "CC",
        "CXX",
        "CMAKE",
        "PATH",
        "CMAKE_GENERATOR",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "C_INCLUDE_PATH",
        "LIBRARY_PATH",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "SDKROOT",
      )
    )
    return repr(
      (
        _compiler_key(compiler),
        str(compiler.definition_cwd),
        tuple(map(str, compiler._source_include_inventory_excludes)),
        external,
      )
    )

  @staticmethod
  def _configuration(compiler):
    fields = ("target", "cc", "cxx", "cmake", "generator", "jobs", "cache_namespace", "environment")
    result = {name: getattr(compiler, name) for name in fields}
    for name in ("cc", "cxx", "cmake"):
      if result[name] is not None:
        result[name] = str(result[name])
    for name in ("cache_dir", "build_dir"):
      result[name] = str(getattr(compiler, name))
    for name in ("include_dirs", "library_dirs"):
      result[name] = list(map(str, getattr(compiler, name)))
    for name in ("cflags", "libraries", "ldflags"):
      result[name] = list(getattr(compiler, name))
    return result

  def _ensure_private(self, compiler):
    import vecops

    if compiler.cache_mode != vecops._C.ArtifactCacheMode.read_write:
      raise ValueError("batch input attestation is restricted to private read-write harness caches")
    if not compiler.cache_dir.resolve().is_relative_to(self.work / "cache"):
      raise ValueError("batch input attestation requires this chunk's private cache")
    if not compiler.build_dir.resolve().is_relative_to(self.work / "build"):
      raise ValueError("batch input attestation requires this chunk's private build directory")

  def _inputs(self, record):
    from vecops._identity import source_inputs

    selected = self.policies[record["policy"]]["compiler"]
    return source_inputs(
      record["source"],
      record["include_roots"],
      record["dependencies"],
      definition_cwd=selected.definition_cwd,
      inventory_excludes=selected._source_include_inventory_excludes,
    )

  def operator(self, source, definition, compiler, *, recipe_id=None, dependencies=()):
    import vecops
    from vecops._identity import compiler_identity, include_roots, source_inputs
    from vecops._identity import digest as identity_digest
    from vecops._operator import Operator

    selected = compiler.snapshot()
    self._ensure_private(selected)
    key = self._policy_key(selected)
    if key not in self.policies:
      identity = compiler_identity(selected)
      self.policies[key] = {"compiler": selected, "identity": identity}
      self.events.append({"phase": "first_policy", "policy": key, "identity": identity})
    policy = self.policies[key]
    path = Path(source).expanduser().resolve()
    roots = include_roots(selected)
    dependencies = tuple(dependencies)
    inputs = source_inputs(
      path,
      roots,
      dependencies,
      definition_cwd=selected.definition_cwd,
      inventory_excludes=selected._source_include_inventory_excludes,
    )
    record = {
      "source": str(path),
      "policy": key,
      "include_roots": list(map(str, roots)),
      "dependencies": list(map(str, dependencies)),
      "inputs": inputs,
    }
    self.sources.append(record)
    fingerprint = identity_digest(inputs)

    def validate_source():
      try:
        equal = canonical(self._inputs(record)) == canonical(inputs)
      except Exception as exc:
        self.invalid_reason = f"source callback could not validate {path}: {exc!r}"
        self.checkpoint()
        raise
      if not equal:
        self.invalid_reason = f"source inputs changed during native callback: {path}"
        self.checkpoint()
      return equal

    operation = Operator.__new__(Operator)
    operation.source, operation.kernel_def, operation.compiler = path, definition, selected
    operation._native = vecops._C.Operator(
      path,
      definition._native(),
      selected.native,
      selected.cache_mode,
      selected.cache_dir,
      f"{selected.cache_namespace};inputs={policy['identity']}",
      recipe_id or f"{definition.name}:{fingerprint}",
      fingerprint,
      validate_source,
    )
    return operation

  def guard(self, phase):
    from vecops._identity import compiler_identity

    try:
      if self.invalid_reason:
        raise BatchInputAttestationError(self.invalid_reason)
      for key, record in self.policies.items():
        compiler = record["compiler"]
        self._ensure_private(compiler)
        if self._policy_key(compiler) != key or compiler_identity(compiler) != record["identity"]:
          raise BatchInputAttestationError(f"compiler/SDK inputs changed at {phase}: {key}")
      for record in self.sources:
        if canonical(self._inputs(record)) != canonical(record["inputs"]):
          raise BatchInputAttestationError(f"source inputs changed at {phase}: {record['source']}")
    except Exception as exc:
      self.invalid_reason = self.invalid_reason or f"batch validation failed at {phase}: {exc!r}"
      self.checkpoint()
      raise BatchInputAttestationError(self.invalid_reason) from exc
    self.events.append({"phase": phase, "policies": len(self.policies), "sources": len(self.sources), "verified": True})
    self.checkpoint()

  def checkpoint(self):
    from dataclasses import asdict

    document = {
      "version": 1,
      "work": str(self.work),
      "invalid_reason": self.invalid_reason,
      "events": self.events,
      "sources": self.sources,
      "policies": [
        {
          "key": key,
          "identity": record["identity"],
          "configuration": self._configuration(record["compiler"]),
          "vector_state": asdict(record["compiler"].vector_state),
          "definition_cwd": str(record["compiler"].definition_cwd),
          "inventory_excludes": list(map(str, record["compiler"]._source_include_inventory_excludes)),
        }
        for key, record in self.policies.items()
      ],
    }
    suite.save(self.work / "batch-input-attestation.json", document)

  @classmethod
  def restore(cls, work):
    import vecops
    from vecops._arm_vector_state import VectorState

    result = cls(work)
    document = suite.read(result.work / "batch-input-attestation.json")
    if not document or document.get("version") != 1 or document.get("work") != str(result.work):
      raise BatchInputAttestationError("missing or incompatible batch input attestation checkpoint")
    result.sources, result.events = document["sources"], document["events"]
    result.invalid_reason = document["invalid_reason"]
    for record in document["policies"]:
      compiler = vecops.Compiler(**record["configuration"]).snapshot()
      compiler._vector_state = VectorState(**record["vector_state"])
      compiler.definition_cwd = Path(record["definition_cwd"])
      compiler._source_include_inventory_excludes = tuple(Path(p) for p in record["inventory_excludes"])
      result._ensure_private(compiler)
      if result._policy_key(compiler) != record["key"]:
        raise BatchInputAttestationError("restored compiler policy differs from original batch")
      result.policies[record["key"]] = {"compiler": compiler, "identity": record["identity"]}
    return result


def reject_attestation_batch(args, groups, work, error):
  """Archive untrusted evidence; never index or retain standalone new binaries."""
  archive = work.parent / "invalid-attestation-evidence.tar.gz"
  temporary = archive.with_suffix(".gz.partial")
  with tarfile.open(temporary, "w:gz", compresslevel=6) as stream:
    for path in sorted(work.rglob("*")):
      if path.is_file() and (
        path.suffix in (".so", ".cpp", ".h", ".json", ".log", ".sh") or path.name == "CMakeCache.txt"
      ):
        stream.add(path, arcname=str(Path("untrusted-work") / path.relative_to(work)), recursive=False)
  os.replace(temporary, archive)
  suite.save(
    work.parent / "invalid-attestation.json",
    {"usable": False, "error": str(error), "archive": str(archive), "archive_sha256": digest(archive.read_bytes())},
  )
  for group in groups:
    suite.save(
      args.suite_dir / "groups" / group["id"] / "build.json",
      {
        "status": "batch_attestation_failed",
        "stage": "batch_input_attestation",
        "error": str(error),
        "execution_ids": group["execution_ids"],
        "untrusted_evidence": str(archive),
      },
    )
  suite.clean_work(work)


def compile_prelowered(template, proof, compiler, *, attestation=None):
  """Build exact proved source with the original DSL ABI and compiler policy.

  Source-backed T.compile would drop runtime_parallelism and strict DSL flags.
  Reuse the same definition and Operator entry used by the normal DSL driver,
  retaining specialization captures, dependencies and normal SDK attestation.
  """
  from copy import copy
  from dataclasses import replace
  from types import SimpleNamespace

  import dsl_matmul_artifact_ab as ab
  from vecops._operator import Operator
  from vecops.language._compiler.backends.cpu.source import _build_definition
  from vecops.language._compiler.cache import _source_identity

  source = Path(proof["source"])
  source_bytes(proof)
  if proof.get("compression") is not None:
    raise ValueError("compressed source must be materialized in owned chunk work first")
  contract, _ = ab.template_contract(template)
  if canonical(contract) != canonical(proof["contract"]):
    raise ValueError("prelowered source ABI differs from original DSL template")
  selected = compiler.snapshot()
  if canonical(selected.vector_state.env_token) != canonical(proof["vector_state_token"]):
    raise ValueError("prelowered source vector state differs from compile thread")
  effective = copy(selected)
  effective.environment = dict(selected.environment)
  effective.cflags = tuple(proof["effective_cflags"])
  effective._native_compiler = None
  definition, values, order = _build_definition(template.ir, (), runtime_parallelism=True)
  ir = replace(template.ir, source=source)
  identity = _source_identity(ir)
  recipe = ir.recipe_id or f"{ir.name}:{identity[:16]}"
  recipe = f"language-v1:{recipe}:{identity[:16]}"
  operation = (
    attestation.operator(source, definition, effective, recipe_id=recipe, dependencies=ir.dependencies)
    if attestation is not None
    else Operator(source, definition, effective, recipe_id=recipe, dependencies=ir.dependencies)
  )
  return SimpleNamespace(
    source=source,
    operator=operation,
    _core=SimpleNamespace(specialization_values=values, specialization_order=order),
    kernel_def=definition,
  )


def compile_cpp_source(template, compiler, *, attestation=None):
  """Use the normal source-C++ definition without an unused DSL debug/core cache.

  Operator retains normal toolchain, SDK and input attestation. No identity is
  supplied or bypassed; only T.compile's redundant Python cache lookup is absent.
  """
  from types import SimpleNamespace

  from vecops._operator import Operator
  from vecops.language._compiler.backends.cpu.source import _build_definition
  from vecops.language._compiler.cache import _source_identity

  ir = template.ir
  if ir.source is None:
    raise ValueError("CPP source-backed compilation requires an explicit source")
  selected = compiler.snapshot()
  definition, values, order = _build_definition(ir, (), runtime_parallelism=False)
  identity = _source_identity(ir)
  recipe = ir.recipe_id or f"{ir.name}:{identity[:16]}"
  recipe = f"language-v1:{recipe}:{identity[:16]}"
  operation = (
    attestation.operator(ir.source, definition, selected, recipe_id=recipe, dependencies=ir.dependencies)
    if attestation is not None
    else Operator(ir.source, definition, selected, recipe_id=recipe, dependencies=ir.dependencies)
  )
  return SimpleNamespace(
    source=ir.source,
    operator=operation,
    _core=SimpleNamespace(specialization_values=values, specialization_order=order),
    kernel_def=definition,
  )


def lower_worker(args, source, plan):
  import dsl_matmul_artifact_ab as ab
  import vecops
  import vecops.language as T
  from vecops.language._compiler.driver import _sme_grid_compiler, _strict_dsl_compiler

  selected = vecops.Compiler(
    target=plan["target"],
    cc=plan["toolchain"]["cc"],
    cxx=plan["toolchain"]["cxx"],
    cflags=plan["compiler_cflags"],
    cache_dir=args.suite_dir / "policy-cache",
  ).snapshot()
  before_runtime = guard_runtime(args)
  group = plan["groups"][args.group_index]
  directory = args.suite_dir / "groups" / group["id"]
  directory.mkdir(parents=True, exist_ok=True)
  document = suite.subset_manifest(plan, group["execution_ids"])
  scenario = ab.source_scenarios(document)[0]
  result = {}
  for packing, pair in enumerate(source.source_templates(scenario)):
    if pair is None:
      continue
    role = "pack_dsl" if packing else "dsl"
    lowered = T.lower(
      pair[0], target=plan["target"], pass_options={**plan["pass_options"], "_vector_state": selected.vector_state}
    )
    effective = _sme_grid_compiler(_strict_dsl_compiler(selected), lowered.artifact)
    text = lowered.artifact.source.encode()
    path = directory / "dsl-sources" / digest(text) / "kernel.cpp"
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = {}
    if getattr(args, "compress_prelower", False):
      path = path.with_suffix(".cpp.gz")
      data = gzip.compress(text, compresslevel=6, mtime=0)
      path.write_bytes(data)
      compressed = {"compression": "gzip", "compressed_sha256": digest(data)}
    else:
      path.write_bytes(text)
    contract, captures = ab.template_contract(pair[0])
    result[role] = {
      **compressed,
      "source": str(path),
      "effective_cflags": list(effective.cflags),
      "environment_cxxflags": shlex.split(effective.environment.get("CXXFLAGS", "")),
      "compile_include_roots": [
        str(Path(suite.read(args.old_suite / "plan.json")["snapshot"]) / suffix)
        for suffix in ("include", "include/vecops")
      ]
      + list(map(str, effective.include_dirs))
      if getattr(args, "allow_codegen_change", ())
      else [],
      "captures": {**captures, "Parallelism": 1},
      "old_build_components": old_build_components(
        effective, Path(suite.read(args.old_suite / "plan.json")["snapshot"])
      )
      if getattr(args, "allow_codegen_change", ())
      else None,
      "vector_state_token": selected.vector_state.env_token,
      "runtime_parallelism": True,
      "contract": contract,
      "raw_sha256": digest(text),
      "normalized_sha256": digest(normalized(text)),
    }
  if guard_runtime(args) != before_runtime:
    raise ValueError("runtime inputs changed during lowering")
  suite.save(directory / "lower-proof.json", result)
  suite.save(directory / "lower-proof-identity.json", proof_identity(plan, before_runtime["new"]))


def index_partial(args, source, plan, groups, work, manifests):
  import dsl_matmul_artifact_ab as ab
  import vecops

  source.source_arguments = lambda scenario: (source.source_metadata(scenario), None)
  ab.arguments = suite.schema_probe_arguments
  for group, document in zip(groups, manifests):
    directory = args.suite_dir / "groups" / group["id"]
    path = directory / "manifest.json"
    suite.save(path, document)
    index_path = directory / "index.json"
    ab.index_artifacts(
      SimpleNamespace(
        manifest=path, build_dir=[work], target=plan["target"], index_output=index_path, snapshot=args.snapshot
      )
    )
    index = suite.read(index_path)
    reused = suite.read(directory / "reuse.json", {})
    completed = suite.read(directory / "completed-roles.json", {})
    reused = {**reused, **completed}
    for execution_id in group["execution_ids"]:
      for role, proof in reused.items():
        scenario = next(s for s in ab.source_scenarios(document) if s.execution_id == execution_id)
        executable = vecops._C.Executable.load(proof["binary"])
        if not executable.can_invoke(suite.schema_probe_arguments(source.source_metadata(scenario), 1)):
          raise ValueError(f"reused artifact rejects exact metadata: {execution_id}/{role}")
        index["artifacts"][execution_id][role] = proof["binary"]
        index["schema_audit"][execution_id][role] = {"reuse_proof": proof}
    suite.save(index_path, index)


def compile_policies_keep_going(args, policies, attestation, work):
  """Attempt every actual compiler policy; continue configured Ninja graphs.

  No build graph, flags, adapter or specialization is rewritten. A failed graph
  retains its original SDK error log; a separate log captures Ninja keep-going.
  A full guard precedes continuation and the caller performs the full postguard
  before any artifact can enter the index or retained-binary inventory.
  """
  import vecops
  from vecops._compiler import _tool

  if attestation is None:
    raise ValueError("keep-going requires batch input attestation")
  errors = []
  events = []
  attempt = time.time_ns()
  for index, record in enumerate(policies.values()):
    if attestation.invalid_reason:
      raise BatchInputAttestationError(attestation.invalid_reason)
    before = set((work / "build").glob("kernel-batch-*/build/CMakeCache.txt"))
    try:
      vecops.compile_batch(record["requests"], parallelism=args.jobs, execution_parallelism=1)
      events.append({"policy": record["key"], "status": "compiled"})
    except Exception as exc:  # noqa: BLE001 -- keep original error and attempt independent policy groups
      errors.append(repr(exc))
      suite.save(
        work.parent / f"policy-{index}-first-error-{attempt}.json",
        {"error": repr(exc), "traceback": traceback.format_exc(), "policy": record["key"]},
      )
      events.append({"policy": record["key"], "status": "compile_failed", "error": repr(exc)})
      after = set((work / "build").glob("kernel-batch-*/build/CMakeCache.txt"))
      compiler = record["compiler"]
      for graph_index, cache in enumerate(sorted(after - before)):
        text = cache.read_text()
        generator = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", text, re.MULTILINE)
        build = cache.parent
        if not generator or generator.group(1) != "Ninja" or not (build / "build.ninja").is_file():
          events.append(
            {"build": str(build), "status": "continuation_not_applicable", "reason": "not a configured Ninja graph"}
          )
          continue
        attestation.guard("pre_continue_build")
        cmake = _tool(compiler.cmake, "CMAKE", ("cmake",))
        kind = re.search(r"^CMAKE_BUILD_TYPE:STRING=(.*)$", text, re.MULTILINE)
        command = [
          str(cmake),
          "--build",
          str(build),
          "--target",
          "vecops_batch_all",
          "--config",
          kind.group(1) if kind and kind.group(1) else "Release",
          "--parallel",
          str(args.jobs),
          "--",
          "-k",
          "0",
        ]
        log = work.parent / f"policy-{index}-ninja-keepgoing-{graph_index}-{attempt}.log"
        with log.open("w") as stream:
          result = subprocess.run(
            command,
            cwd=compiler.definition_cwd,
            env={**os.environ, **compiler.environment},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
          )
        events.append(
          {
            "build": str(build),
            "status": "continued",
            "returncode": result.returncode,
            "command": command,
            "log": str(log),
          }
        )
  suite.save(work.parent / "keepgoing-policy-results.json", events)
  return "\n".join(errors) if errors else None


def apply_keepgoing_marker(args):
  """Opt in future compile children only; running parents/compilers are untouched."""
  if args.worker != "compile" or args.keep_going or not args.batch_input_attestation:
    return
  marker = Path(args.suite_dir).resolve() / ".repair-keepgoing-opt-in.json"
  if not marker.exists():
    return
  if (
    not marker.is_file() or marker.is_symlink() or marker.stat().st_uid != os.getuid() or marker.stat().st_mode & 0o077
  ):
    raise ValueError("keep-going marker must be a caller-owned private regular file")
  raw = marker.read_bytes()
  policy = json.loads(raw)
  expected = digest(Path(__file__).resolve().read_bytes())
  if (
    policy.get("version") != 1
    or policy.get("driver_sha256") != expected
    or policy.get("scope") != "compile_workers_with_batch_input_attestation"
    or policy.get("keep_going") is not True
  ):
    raise ValueError("keep-going marker does not match this driver and guarded worker scope")
  args.keep_going = True
  args.keep_going_marker = {"path": str(marker), "sha256": digest(raw), "policy": policy}


def compile_worker(args, source, plan):
  """One frontend collects a small chunk; preserve successes even on exception."""
  import dsl_matmul_artifact_ab as ab
  import vecops

  groups = [plan["groups"][i] for i in sorted(suite.ranges(args.chunk))]
  work = args.suite_dir / "chunks" / args.chunk.replace(",", "_") / "work"
  if work.exists():
    suite.clean_work(work)
  work.mkdir(parents=True)
  (work / ".full-suite-owned-work").write_text("repair-chunk")
  if getattr(args, "keep_going_marker", None):
    suite.save(work.parent / f"keepgoing-worker-policy-{time.time_ns()}.json", args.keep_going_marker)
  (work / "compiler-tmp").mkdir()
  os.environ["TMPDIR"] = str(work / "compiler-tmp")
  os.environ["CLANG_CRASH_DIAGNOSTICS_DIR"] = str(work / "compiler-tmp")
  compiler = vecops.Compiler(
    target=plan["target"],
    cc=plan["toolchain"]["cc"],
    cxx=plan["toolchain"]["cxx"],
    cflags=plan["compiler_cflags"],
    jobs=args.jobs,
    cache_dir=work / "cache",
    build_dir=work / "build",
  )
  attestation = BatchInputAttestation(work) if getattr(args, "batch_input_attestation", False) else None
  if attestation is not None:
    (work / ".batch-input-attestation-required").write_text("1")
  policies = {}
  documents = []
  error = None
  try:
    with vecops.collect_compile_requests() as collector:
      for group in groups:
        directory = args.suite_dir / "groups" / group["id"]
        directory.mkdir(parents=True, exist_ok=True)
        document = suite.subset_manifest(plan, group["execution_ids"]) | {"artifact_sources": {}}
        reused = {**suite.read(directory / "reuse.json", {}), **suite.read(directory / "completed-roles.json", {})}
        scenarios = ab.source_scenarios(document)
        for packing, pair in enumerate(source.source_templates(scenarios[0])):
          if pair is None:
            continue
          for role, template in zip(("dsl", "cpp"), pair):
            label = ("pack_" if packing else "") + role
            if label in reused:
              saved = reused[label]
              if digest(Path(saved["binary"]).read_bytes()) != saved["binary_sha256"]:
                raise ValueError("completed/reused binary changed")
              if (
                "corrected_plan_runner_hashes" in saved
                and saved["corrected_plan_runner_hashes"] != plan["runner_hashes"]
              ):
                raise ValueError("completed artifact belongs to another corrected plan")
              continue
            if role == "dsl":
              proof = materialize_proof(suite.read(directory / "lower-proof.json")[label], work)
              kernel = compile_prelowered(template, proof, compiler, attestation=attestation)
            else:
              kernel = compile_cpp_source(template, compiler, attestation=attestation)
            for scenario in scenarios:
              count_before = len(collector.requests)
              kernel.operator(*source.source_metadata(scenario), 1, **kernel._core.specialization_values)
              if getattr(args, "keep_going", False):
                selected = kernel.operator.compiler
                key = BatchInputAttestation._policy_key(selected)
                policy = policies.setdefault(key, {"key": key, "compiler": selected, "requests": []})
                policy["requests"].extend(collector.requests[count_before:])
              document["artifact_sources"].setdefault(scenario.execution_id, {})[label] = str(kernel.source)
        proof = suite.read(directory / "lower-proof.json", {})
        for scenario in scenarios:
          for role in reused:
            if role in proof:
              document["artifact_sources"].setdefault(scenario.execution_id, {})[role] = materialize_proof(
                proof[role], work
              )["source"]
        documents.append(document)
        suite.save(directory / "manifest.json", document)
    suite.save(work / "manifest-ready.json", documents)
    if attestation is not None:
      attestation.guard("pre_build")
    if getattr(args, "keep_going", False):
      error = compile_policies_keep_going(args, policies, attestation, work)
    else:
      vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  except Exception as exc:  # noqa: BLE001 -- preserve partial native successes before isolating failures
    error = repr(exc)
  if attestation is not None:
    try:
      attestation.guard("post_build")
    except Exception as exc:  # noqa: BLE001 -- no new artifact can escape uncertain batch input state
      reject_attestation_batch(args, groups, work, exc)
      return
  finalize_chunk(args, source, plan, groups, work, documents, error)


def persist_batch_attestation(work):
  """Keep full validated input evidence after owned temporary work is removed."""
  if not (work / ".batch-input-attestation-required").exists():
    return None
  checkpoint = work / "batch-input-attestation.json"
  document = suite.read(checkpoint)
  events = document.get("events", []) if document else []
  if (
    not document
    or document.get("invalid_reason")
    or not events
    or events[-1].get("phase") not in ("post_build", "recovery_post_build")
    or events[-1].get("verified") is not True
    or not any(event.get("phase") == "pre_build" and event.get("verified") is True for event in events)
  ):
    raise BatchInputAttestationError("cannot publish a batch without complete successful pre/post attestation")
  data = checkpoint.read_bytes()
  sha256 = digest(data)
  target = work.parent / f"batch-input-attestation.{sha256}.json"
  if target.exists() and target.read_bytes() != data:
    raise BatchInputAttestationError("immutable persisted batch input evidence changed")
  if not target.exists():
    temporary = target.with_suffix(".json.partial")
    temporary.write_bytes(data)
    os.replace(temporary, target)
  return {"path": str(target), "sha256": sha256, "scope": "full compiler/SDK and source inputs pre/post"}


def finalize_chunk(args, source, plan, groups, work, documents, error):
  attestation_evidence = persist_batch_attestation(work)
  if len(documents) == len(groups):
    index_partial(args, source, plan, groups, work, documents)
    # One archive per chunk, not one copy of a large chunk for every group.
    combined = {"artifacts": {}, "schema_audit": {}}
    for group in groups:
      row = suite.read(args.suite_dir / "groups" / group["id"] / "index.json")
      combined["artifacts"].update(
        {eid: {role: path for role, path in paths.items() if path} for eid, paths in row["artifacts"].items()}
      )
      combined["schema_audit"].update(row["schema_audit"])
    retained = suite.retain(work.parent, combined, work)
    for group in groups:
      directory = args.suite_dir / "groups" / group["id"]
      row = suite.read(directory / "index.json")
      if attestation_evidence is not None:
        row["batch_input_attestation"] = attestation_evidence
      row.update({key: value for key, value in retained.items() if key not in ("artifacts", "schema_audit")})
      for eid, paths in row["artifacts"].items():
        paths.update(retained["artifacts"][eid])
      suite.save(directory / "index.json", row)
      complete = all(path for paths in row["artifacts"].values() for path in paths.values())
      role_paths = list(row["artifacts"].values())
      completed = {
        role: {
          "binary": path,
          "binary_sha256": digest(Path(path).read_bytes()),
          "corrected_plan_runner_hashes": plan["runner_hashes"],
        }
        for role, path in role_paths[0].items()
        if path and all(p.get(role) == path for p in role_paths)
      }
      suite.save(directory / "completed-roles.json", completed)
      suite.save(
        directory / "build.json",
        {"status": "built" if complete else "failed", "error": error, "execution_ids": group["execution_ids"]},
      )
  else:
    suite.save(work.parent / "failure-evidence.json", suite.retain(work.parent, {}, work))
  suite.clean_work(work)


def interrupted_recovery_policy(args):
  marker = args.suite_dir / ".repair-interrupted-recovery-opt-in.json"
  if args.worker != "recover" or not marker.exists():
    return None
  info = marker.stat()
  if marker.is_symlink() or not marker.is_file() or info.st_uid != os.getuid() or info.st_mode & 0o077:
    raise ValueError("recovery opt-in must be a caller-owned private regular file")
  policy = suite.read(marker)
  if policy.get("driver_sha256") != digest(Path(__file__).read_bytes()) or policy.get("version") != 1:
    raise ValueError("recovery opt-in does not match this exact driver")
  if args.chunk not in policy.get("chunks", []):
    return None
  return policy


def work_readers(work):
  """Conservatively include any same-UID process referring to the owned graph."""
  root = str(work.resolve())
  readers = []
  for path in Path("/proc").iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
      continue
    try:
      if path.stat().st_uid != os.getuid():
        continue
      arguments = (path / "cmdline").read_bytes().decode(errors="replace").split("\0")
      cwd = os.readlink(path / "cwd")
      references = [value for value in (*arguments, cwd) if value == root or value.startswith(root + "/")]
      if "--worker" in arguments and "--suite-dir" in arguments and "--chunk" in arguments:
        selected_suite = Path(arguments[arguments.index("--suite-dir") + 1])
        if not selected_suite.is_absolute():
          selected_suite = Path(cwd) / selected_suite
        selected_chunk = arguments[arguments.index("--chunk") + 1].replace(",", "_")
        if selected_suite.resolve() == work.parents[2].resolve() and selected_chunk == work.parent.name:
          references.append("same-suite same-chunk worker")
      if references:
        readers.append({"identity": suite.process_identity(int(path.name)), "references": references})
    except (FileNotFoundError, ProcessLookupError, PermissionError):
      continue
  return readers


def validate_configured_policy(compiler, build, adapters):
  config = compiler._native_config()
  cache = (build / "CMakeCache.txt").read_text()
  for name, expected in (
    ("CMAKE_C_COMPILER", config.toolchain.c_compiler),
    ("CMAKE_CXX_COMPILER", config.toolchain.cxx_compiler),
  ):
    match = re.search(rf"^{name}:(?:FILEPATH|STRING)=(.*)$", cache, re.MULTILINE)
    if not match or Path(match.group(1)).resolve() != Path(expected).resolve():
      raise ValueError("configured compiler differs from restored policy")
  environment_flags = re.search(r"^CMAKE_CXX_FLAGS:STRING=(.*)$", cache, re.MULTILINE)
  if not environment_flags or shlex.split(environment_flags.group(1)) != shlex.split(
    compiler.environment.get("CXXFLAGS", os.environ.get("CXXFLAGS", ""))
  ):
    raise ValueError("configured environment compiler flags differ from restored policy")
  project_hashes = {}
  for adapter in adapters:
    project = build.parent / "project/tasks" / adapter.parents[1].name / "CMakeLists.txt"
    text = project.read_text()
    target = re.search(r'TARGET_ARCH "([^"\n]+)"', text)
    if not target or target.group(1) != config.target_arch:
      raise ValueError("configured target differs from restored policy")
    for section, expected in (
      ("COMPILE_OPTIONS", config.compile_options),
      ("COMPILE_DEFINITIONS", config.compile_definitions),
      ("LINK_OPTIONS", config.link_options),
      ("LINK_LIBRARIES", config.link_libraries),
    ):
      match = re.search(rf"^    {section}\n((?:        .*\n)*)", text, re.MULTILINE)
      actual = re.findall(r'^        "([^"\n]*)"$', match.group(1), re.MULTILINE) if match else []
      if actual != list(expected):
        raise ValueError(f"configured {section} differs from restored policy")
    project_hashes[str(project)] = digest(project.read_bytes())
  return project_hashes


def continue_interrupted_graphs(args, work, attestation):
  """Resume exact configured graphs only after policy and ownership validation."""
  import fcntl

  from vecops._compiler import _tool

  claim = (work.parent / ".interrupted-recovery.lock").open("a")
  try:
    fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
    readers = work_readers(work)
    if readers:
      raise ValueError(f"owned graph still has live readers: {readers}")
    mapping = {}
    for record in attestation.sources:
      mapping.setdefault(str(Path(record["source"]).resolve()), set()).add(record["policy"])
    graphs = []
    for cache in sorted((work / "build").glob("kernel-batch-*/build/CMakeCache.txt")):
      text = cache.read_text()
      if not re.search(r"^CMAKE_GENERATOR:INTERNAL=Ninja$", text, re.MULTILINE):
        raise ValueError("interrupted graph is not Ninja")
      build = cache.parent
      commands = suite.read(build / "compile_commands.json", [])
      tasks, keys = [], set()
      for adapter in sorted(build.parent.glob("tasks/*/input/vecops_adapter.cpp")):
        match = re.search(r'#include "([^"\n]+\.cpp)"', adapter.read_text())
        if not match:
          raise ValueError("adapter has no exact source binding")
        source = str(Path(match.group(1)).resolve())
        selected = mapping.get(source, set())
        if len(selected) != 1:
          raise ValueError("configured task source has ambiguous attested policy")
        keys.update(selected)
        spec = adapter.with_name("vecops_spec.h")
        entry = next((x for x in commands if Path(x["file"]).resolve() == adapter.resolve()), None)
        if entry is None:
          raise ValueError("configured task lacks exact compile command")
        words = entry.get("arguments") or shlex.split(entry["command"])
        obj = Path(entry["directory"]) / words[words.index("-o") + 1]
        tasks.append(
          {
            "adapter": str(adapter),
            "adapter_sha256": digest(adapter.read_bytes()),
            "spec_sha256": digest(spec.read_bytes()),
            "source": source,
            "policy": next(iter(selected)),
            "object": str(obj),
            "object_sha256": digest(obj.read_bytes()) if obj.exists() else None,
          }
        )
      if not tasks or len(keys) != 1:
        raise ValueError("configured graph lacks one exact compiler policy")
      key = next(iter(keys))
      compiler = attestation.policies[key]["compiler"]
      project_hashes = validate_configured_policy(compiler, build, [Path(t["adapter"]) for t in tasks])
      kind = re.search(r"^CMAKE_BUILD_TYPE:STRING=(.*)$", text, re.MULTILINE)
      if (
        not kind
        or kind.group(1) != compiler._native_config().toolchain.build_type
        or "vecops_batch_all" not in (build / "build.ninja").read_text()
      ):
        raise ValueError("configured graph target/build type unknown")
      immutable_graph = [
        build / "CMakeCache.txt",
        build / "build.ninja",
        build / "CMakeFiles/rules.ninja",
        *(build.parent / "project").rglob("CMakeLists.txt"),
      ]
      graphs.append(
        {
          "graph_sha256": {str(path): digest(path.read_bytes()) for path in immutable_graph},
          "build": str(build),
          "policy": key,
          "tasks": tasks,
          "project_sha256": project_hashes,
          "compile_commands_sha256": digest((build / "compile_commands.json").read_bytes()),
          "command": [
            str(_tool(compiler.cmake, "CMAKE", ("cmake",))),
            "--build",
            str(build),
            "--target",
            "vecops_batch_all",
            "--config",
            kind.group(1),
            "--parallel",
            "2",
            "--",
            "-k",
            "0",
          ],
        }
      )
    if not graphs:
      raise ValueError("no configured graphs to resume")
    suite.save(work.parent / "interrupted-recovery-graph-proof.json", graphs)
    for i, graph in enumerate(graphs):
      attestation.guard("pre_continue_build")
      if work_readers(work):
        raise ValueError("live graph reader appeared before continuation")
      compiler = attestation.policies[graph["policy"]]["compiler"]
      log = work.parent / f"interrupted-recovery-{i}-{time.time_ns()}.log"
      with log.open("w") as stream:
        result = subprocess.run(
          graph["command"],
          cwd=compiler.definition_cwd,
          env={**os.environ, **compiler.environment},
          stdout=stream,
          stderr=subprocess.STDOUT,
          check=False,
        )
      for path, expected in {**graph["graph_sha256"], **graph["project_sha256"]}.items():
        if digest(Path(path).read_bytes()) != expected:
          raise ValueError("configured project changed during continuation")
      if digest((Path(graph["build"]) / "compile_commands.json").read_bytes()) != graph["compile_commands_sha256"]:
        raise ValueError("compile commands changed during continuation")
      for task in graph["tasks"]:
        if (
          digest(Path(task["adapter"]).read_bytes()) != task["adapter_sha256"]
          or digest(Path(task["adapter"]).with_name("vecops_spec.h").read_bytes()) != task["spec_sha256"]
        ):
          raise ValueError("adapter/spec changed during continuation")
      graph.update(returncode=result.returncode, log=str(log))
      suite.save(work.parent / "interrupted-recovery-graph-proof.json", graphs)
    attestation.guard("recovery_post_build")
  finally:
    claim.close()


def child(args, options, log, timeout):
  command = [
    sys.executable,
    str(Path(__file__).resolve()),
    "--snapshot",
    str(args.snapshot),
    "--suite-dir",
    str(args.suite_dir),
    "--old-suite",
    str(args.old_suite),
    "--jobs",
    str(args.jobs),
    "--min-free-gb",
    str(args.min_free_gb),
    *options,
  ]
  for path in getattr(args, "allow_codegen_change", ()):
    command.extend(("--allow-codegen-change", path))
  if args.compress_prelower:
    command.append("--compress-prelower")
  if args.batch_input_attestation:
    command.append("--batch-input-attestation")
  if args.keep_going:
    command.append("--keep-going")
  with log.open("a") as stream:
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    process._full_suite_identity = suite.process_identity(process.pid)
    try:
      return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      suite.stop(process, audit_path=log.with_name(f"process-cleanup-{process.pid}-{time.time_ns()}.json"))
      return 124


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--snapshot", type=Path, required=True)
  parser.add_argument("--suite-dir", type=Path, required=True)
  parser.add_argument("--old-suite", type=Path, required=True)
  parser.add_argument("--groups")
  parser.add_argument(
    "--allow-codegen-change",
    action="append",
    default=[],
    metavar="PATH",
    help="Explicit existing codegen file allowed to change (affine.py, CPU recipes.py, or DSL _library/packing.py only)",
  )
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--chunk-size", type=int, default=8)
  parser.add_argument(
    "--keep-going",
    action="store_true",
    help="Opt-in policy-isolated compile and Ninja continuation; requires batch input attestation",
  )
  parser.add_argument(
    "--batch-input-attestation",
    action="store_true",
    help="Opt-in private batch full pre/post SDK/toolchain attestation",
  )
  parser.add_argument(
    "--compress-prelower", action="store_true", help="Compress proved source; use only with an updated coordinator"
  )
  parser.add_argument(
    "--lower-workers",
    type=int,
    default=1,
    help="Independent pure-lower processes; native chunk frontend remains one per parent",
  )
  parser.add_argument("--min-free-gb", type=float, default=2)
  parser.add_argument("--lower-timeout", type=float, default=1800)
  parser.add_argument("--build-timeout", type=float, default=1800)
  parser.add_argument("--phase", choices=("reuse", "build", "all"), default="all")
  parser.add_argument(
    "--build-missing",
    action="store_true",
    help="Admit absent DSL only after isolated successful lowering; do not use while old groups still build",
  )
  parser.add_argument("--retry-lowering", action="store_true")
  parser.add_argument("--streaming-child", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--worker", choices=("lower", "compile", "recover"))
  parser.add_argument("--group-index", type=int)
  parser.add_argument("--chunk")
  args = parser.parse_args()
  apply_keepgoing_marker(args)
  if args.keep_going and not args.batch_input_attestation:
    parser.error("--keep-going requires --batch-input-attestation")
  if (
    min(args.jobs, args.chunk_size, args.lower_workers, args.lower_timeout, args.build_timeout) <= 0
    or args.min_free_gb < 0
  ):
    parser.error("jobs/chunk size/timeouts must be positive; free-space threshold must be nonnegative")
  args.snapshot = args.snapshot.resolve()
  args.suite_dir = args.suite_dir.resolve()
  args.old_suite = args.old_suite.resolve()
  plan = suite.read(args.suite_dir / "plan.json")
  suite.verify_plan_sources(args.snapshot, plan)
  guard_runtime(args)
  if args.worker:
    source = suite.catalog(args.snapshot)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.worker == "lower":
      lower_worker(args, source, plan)
    elif args.worker == "compile":
      compile_worker(args, source, plan)
    else:
      groups = [plan["groups"][i] for i in sorted(suite.ranges(args.chunk))]
      work = args.suite_dir / "chunks" / args.chunk.replace(",", "_") / "work"
      if work.exists():
        policy = None
        if (work / ".batch-input-attestation-required").exists():
          try:
            policy = interrupted_recovery_policy(args)
            if policy is not None and policy.get("mode") == "preserve_only":
              suite.save(
                work.parent / "interrupted-recovery-pending.json",
                {
                  "work_preserved": True,
                  "policy": policy,
                  "reason": "explicit preserve-only; no build or publication attempted",
                },
              )
              time.sleep(60)
              return 0
            attestation = BatchInputAttestation.restore(work)
            if policy is not None:
              continue_interrupted_graphs(args, work, attestation)
            else:
              attestation.guard("recovery_post_build")
          except Exception as exc:  # noqa: BLE001 -- uncertain opt-in recovery preserves configured objects
            if (args.suite_dir / ".repair-interrupted-recovery-opt-in.json").exists():
              suite.save(
                work.parent / "interrupted-recovery-pending.json", {"error": repr(exc), "work_preserved": True}
              )
              return 0
            reject_attestation_batch(args, groups, work, exc)
            return 0
        documents = suite.read(work / "manifest-ready.json", [])
        finalize_chunk(
          args,
          source,
          plan,
          groups,
          work,
          documents if all(documents) else [],
          "continued_after_worker_interruption" if policy is not None else "worker_interrupted",
        )
    return 0
  groups = [plan["groups"][i] for i in sorted(suite.ranges(args.groups))] if args.groups else plan["groups"]
  if args.phase == "all" and len(groups) > args.chunk_size and not args.streaming_child:
    for offset in range(0, len(groups), args.chunk_size):
      part = groups[offset : offset + args.chunk_size]
      options = [
        "--phase",
        "all",
        "--streaming-child",
        "--groups",
        ",".join(str(g["index"]) for g in part),
        "--chunk-size",
        str(args.chunk_size),
        "--lower-workers",
        str(args.lower_workers),
        "--lower-timeout",
        str(args.lower_timeout),
        "--build-timeout",
        str(args.build_timeout),
      ]
      if args.build_missing:
        options.append("--build-missing")
      if args.retry_lowering:
        options.append("--retry-lowering")
      log = args.suite_dir / "repair-progress.log"
      child(args, options, log, args.lower_timeout * len(part) + args.build_timeout * (2 * len(part) + 1))
      print(f"repair chunk {part[0]['index']}..{part[-1]['index']} completed; {log}", flush=True)
    return 0
  old_plan = suite.read(args.old_suite / "plan.json")
  # Runtime and SDK must be identical; changed benchmark harness is intentional.
  if plan["source_hashes"] != old_plan["source_hashes"]:
    raise ValueError("benchmark registration source hashes differ; reuse disabled")
  if any(
    plan.get(key) != old_plan.get(key)
    for key in ("architecture", "pass_options", "target", "toolchain", "compiler_cflags")
  ):
    raise ValueError("reuse requires identical target/toolchain/pass options/compiler flags")
  runtime_evidence = guard_runtime(args)
  runtime_manifest = runtime_evidence["new"]
  suite.save(args.suite_dir / "repair-runtime-identity.json", runtime_evidence)
  if args.phase in ("reuse", "all"):

    def reuse_group(group):
      directory = args.suite_dir / "groups" / group["id"]
      directory.mkdir(parents=True, exist_ok=True)
      if suite.read(directory / "build.json", {}).get("status") == "built":
        if suite.read(directory / "lower-proof-identity.json") != proof_identity(plan, runtime_manifest):
          raise ValueError("completed build belongs to different runtime inputs; use a new suite directory")
        return
      print(f"reuse group {group['index']} {group['id']}", flush=True)
      found = list(candidates(args.old_suite, group["execution_ids"]))
      prior = suite.read(directory / "reuse-status.json", {})
      if prior.get("status") in ("failed", "timeout") and not args.retry_lowering:
        return
      if not found and not args.build_missing:
        suite.save(directory / "reuse-status.json", {"status": "waiting_old_artifacts"})
        return
      saved_proof_matches = (directory / "lower-proof.json").is_file() and suite.read(
        directory / "lower-proof-identity.json"
      ) == proof_identity(plan, runtime_manifest)
      status = (
        0
        if saved_proof_matches
        else child(
          args, ["--worker", "lower", "--group-index", str(group["index"])], directory / "lower.log", args.lower_timeout
        )
      )
      if status:
        suite.save(
          directory / "reuse-status.json", {"status": "timeout" if status == 124 else "failed", "returncode": status}
        )
        suite.save(
          directory / "build.json",
          {
            "status": "timeout" if status == 124 else "lowering_failed",
            "stage": "lowering",
            "returncode": status,
            "log": str(directory / "lower.log"),
            "execution_ids": group["execution_ids"],
          },
        )
        return
      if guard_runtime(args) != runtime_evidence:
        raise ValueError("runtime inputs changed while proving reuse")
      result = {}
      for role, fresh in suite.read(directory / "lower-proof.json").items():
        proofs = []
        for item in found:
          role_evidence = codegen_role_evidence(fresh, item) if args.allow_codegen_change else {}
          if role_evidence is None:
            continue
          proof = prove_reuse(source_bytes(fresh), item, fresh["contract"])
          if proof:
            proofs.append(proof | {"codegen_change_evidence": role_evidence})
        if proofs:
          result[role] = proofs[0]
      suite.save(directory / "reuse.json", result)
      print(f"reuse group {group['index']}: {list(result)}", flush=True)
      suite.save(
        directory / "reuse-status.json", {"status": "proved" if result else "not_equivalent", "roles": list(result)}
      )

    with ThreadPoolExecutor(max_workers=args.lower_workers) as pool:
      # Consume every outcome; metadata/evidence integrity failures abort visibly.
      list(pool.map(reuse_group, groups))
  if args.phase in ("build", "all"):
    admitted = [
      group
      for group in groups
      if suite.read(args.suite_dir / "groups" / group["id"] / "reuse-status.json", {}).get("status")
      in (("proved", "not_equivalent") if args.build_missing else ("proved",))
      and suite.read(args.suite_dir / "groups" / group["id"] / "build.json", {}).get("status") != "built"
    ]

    def compile_chunk(chunk):
      if not chunk:
        return
      if shutil.disk_usage(args.suite_dir).free < args.min_free_gb * 1024**3:
        for group in chunk:
          suite.save(
            args.suite_dir / "groups" / group["id"] / "build.json",
            {"status": "insufficient_disk", "execution_ids": group["execution_ids"]},
          )
        return
      name = ",".join(str(group["index"]) for group in chunk)
      log = args.suite_dir / "chunks" / name.replace(",", "_") / "dispatch.log"
      log.parent.mkdir(parents=True, exist_ok=True)
      print(f"compile chunk {name}", flush=True)
      status = child(args, ["--worker", "compile", "--chunk", name], log, args.build_timeout)
      if status:
        child(args, ["--worker", "recover", "--chunk", name], log, args.build_timeout)
      for group in chunk:
        directory = args.suite_dir / "groups" / group["id"]
        if not (directory / "build.json").exists():
          suite.save(
            directory / "build.json",
            {
              "status": "timeout" if status == 124 else "failed",
              "returncode": status,
              "execution_ids": group["execution_ids"],
            },
          )
      missing = [
        group
        for group in chunk
        if suite.read(args.suite_dir / "groups" / group["id"] / "build.json", {}).get("status") != "built"
      ]
      if len(missing) > 1:
        midpoint = len(missing) // 2
        compile_chunk(missing[:midpoint])
        compile_chunk(missing[midpoint:])

    for offset in range(0, len(admitted), args.chunk_size):
      compile_chunk(admitted[offset : offset + args.chunk_size])
  return 0


if __name__ == "__main__":
  for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
  raise SystemExit(main())
