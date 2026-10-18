# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Resumable full catalog: isolated grouped builds, then quiet DSL/C++ timing.

--execution-ids-file optionally selects an exact architecture-scoped ID set
before grouping. Its path and content are immutable on resume, including workers.
Full source catalog hashes remain attested even for a subset.

No workload cap is applied. A group is exactly one compilation_key, while every
registered execution ID retains its own oracle and timing record. Build workers
never allocate catalog payloads or calculate reference products. Compact mode
archives source/adapter/commands/logs and retains executable shared libraries;
only this runner's marked temporary work directory is removed.

--prelowered-manifest consumes a completed source-equivalence emission from
the selected compiler snapshot. Native construction preserves the DSL ABI,
strict flags and original source provenance; it does not repeat Python lowering.
The native build driver may change between emission and construction; its current
path and bytes are separately fixed in the native plan, while other runner hashes
and the complete compiler frontend remain fixed to the emitted inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

VERSION = 1


def save(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
  try:
    with os.fdopen(fd, "w") as stream:
      json.dump(value, stream, indent=2)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    Path(temporary).unlink(missing_ok=True)


def read(path, default=None):
  return json.loads(path.read_text()) if path.exists() else default


def ranges(value):
  result = set()
  for part in value.split(","):
    pair = part.split("-")
    begin, end = (int(pair[0]), int(pair[-1]))
    if begin < 0 or end < begin or len(pair) > 2:
      raise ValueError("ranges must contain nonnegative increasing integers")
    result.update(range(begin, end + 1))
  return result


def grouped(scenarios):
  groups = {}
  for scenario in scenarios:
    key = repr(scenario.compilation_key)
    groups.setdefault(key, []).append(scenario.execution_id)
  return [
    {"index": index, "id": hashlib.sha256(key.encode()).hexdigest()[:24], "key": key, "execution_ids": ids}
    for index, (key, ids) in enumerate(groups.items())
  ]


def failure(returncode, text, *, timed_out=False):
  if timed_out:
    return "timeout"
  lower = text.lower()
  if "out of memory" in lower or "bad_alloc" in lower or "memoryerror" in lower or "cannot allocate memory" in lower:
    return "oom"
  if returncode < 0:
    return "signal_killed"  # SIGKILL alone does not establish OOM.
  if "notimplementederror" in lower or "unsupported" in lower:
    return "unsupported"
  return "failed"


def selected(groups, args):
  wanted = ranges(args.groups) if args.groups else {g["index"] for g in groups}
  unknown = wanted - {g["index"] for g in groups}
  if unknown:
    raise ValueError(f"unknown group indexes: {sorted(unknown)}")
  return [g for g in groups if g["index"] in wanted and g["index"] % args.shard_count == args.shard_index]


def catalog(snapshot):
  sys.path.insert(0, str(snapshot / "benchmarks"))
  from dsl_snapshot_entry import select_snapshot

  select_snapshot(snapshot)
  import dsl_matmul_source_catalog as source
  import vecops

  if Path(vecops.__file__).resolve() != snapshot / "python/vecops/__init__.py":
    raise RuntimeError("vecops escaped the selected snapshot")
  return source


def make_plan(args, source):
  from dsl_benchmark_selection import select_rows, selection_identity
  from dsl_matmul_catalog import load_cases

  cases = load_cases(
    architecture=args.architecture, features=args.feature, complete_registration=args.complete_registration
  )
  executions = tuple(source.scenarios(cases, levels=args.level))
  identity = selection_identity(getattr(args, "execution_ids_file", None), args.architecture, "execution_ids")
  executions = select_rows(executions, identity, "execution_ids", lambda row: row.execution_id)
  document = source.audit_document(cases, executions)
  document["selection"] = identity
  document.update(
    version=VERSION,
    snapshot=str(args.snapshot),
    architecture=args.architecture,
    features=args.feature,
    pass_options=json.loads(args.extra_options),
    label=args.label,
    compiler_cflags=args.cflag,
    complete_registration=args.complete_registration,
    target=args.target,
    toolchain={"cc": args.cc, "cxx": args.cxx},
    levels=args.level,
    selected_execution_ids=[s.execution_id for s in executions],
    groups=grouped(executions),
    timing={"cpu": args.cpu, "rounds": args.rounds, "calibration_ns": 25_000_000, "stages": args.stages, "batch_size": getattr(args, "timing_batch_size", 1)},
  )
  if not isinstance(document["pass_options"], dict):
    raise TypeError("extra-options must be a JSON object")
  if document["pass_options"] and args.label == "default":
    raise ValueError("nondefault pass options require an explicit nondefault label")
  document["prelowered"] = prelowered_identity(getattr(args, "prelowered_manifest", None))
  if document["prelowered"] is not None:
    document["native_build_driver"] = {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)}
  return document


def file_sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prelowered_identity(path):
  if path is None:
    return None
  path = Path(path).resolve()
  return {"path": str(path), "sha256": file_sha(path)}


def read_prelowered(plan, snapshot):
  """Bind the emission to its actual frontend, templates and native SDK.

  Backend compiler environment flags may differ: these are emitted C++ inputs
  to a new native compilation. Their explicit DSL flags and vector geometry
  remain part of each program's checked proof and normal artifact identity.
  """
  identity = plan.get("prelowered")
  if identity is None:
    return None
  from dsl_matmul_source_equivalence import native_environment, tree_sha

  if prelowered_identity(identity["path"]) != identity:
    raise ValueError("prelowered manifest changed")
  document = read(Path(identity["path"]))
  if not document.get("completed"):
    raise ValueError("prelowered source emission is incomplete")
  source_plan = Path(document["plan"])
  if file_sha(source_plan) != document["plan_sha256"]:
    raise ValueError("prelowered source plan changed")
  original = read(source_plan)
  for field in ("source_hashes", "pass_options", "target", "compiler_cflags"):
    if original[field] != plan[field]:
      raise ValueError(f"prelowered {field} differs from selected templates or policy")
  # This file orchestrates native construction; it does not produce the emitted
  # DSL body. Permit a driver revision only when its actual current bytes are
  # independently bound. Keep the entire runner inventory and all other hashes.
  driver_name = "dsl_matmul_full_suite.py"
  before, after = original["runner_hashes"], plan["runner_hashes"]
  if before != after:
    driver = plan.get("native_build_driver", {})
    if (
      before.keys() != after.keys()
      or driver_name not in before
      or any(before[name] != after[name] for name in before if name != driver_name)
      or driver != {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)}
    ):
      raise ValueError("prelowered runner_hashes differs from selected templates or policy")
  if tree_sha(snapshot / "python/vecops", {".py"}) != document["compiler_python_sha256"]:
    raise ValueError("prelowered frontend differs from selected compiler snapshot")
  current = native_environment(snapshot, plan)
  for field in ("sdk_sha256", "extension_sha256", "target", "machine", "compiler_cflags"):
    if current[field] != document["native_environment"][field]:
      raise ValueError(f"prelowered native {field} differs from selected snapshot")
  groups = {group["id"]: group for group in document["groups"]}
  if len(groups) != len(document["groups"]) or set(groups) != set(document["group_ids"]):
    raise ValueError("prelowered group inventory contains duplicates or omissions")
  return document, groups


def prelowered_program(document, groups, group, label, template):
  from dsl_matmul_artifact_ab import template_contract
  from dsl_matmul_repair_suite import canonical
  from vecops._arm_vector_state import VectorState

  validate_prelowered_scope(groups, [group])
  origin = groups[group["id"]]
  program = origin["programs"].get(label)
  if program is None or program["name"] != template.ir.name or program["runtime_parallelism"] is not True:
    raise ValueError("prelowered program identity or DSL runtime policy differs")
  contract, captures = template_contract(template)
  if canonical((contract, captures)) != canonical((program["contract"], program["captures"])):
    raise ValueError("prelowered program ABI or specialization captures differ")
  source = Path(document["_manifest_directory"]) / "sources" / (program["source_sha256"] + ".cpp")
  if source.stat().st_size != program["source_bytes"] or file_sha(source) != program["source_sha256"]:
    raise ValueError("prelowered C++ source changed")
  state = origin["vector_state"]
  return {
    "source": str(source), "raw_sha256": program["source_sha256"], "contract": contract,
    "effective_cflags": program["effective_cflags"], "runtime_parallelism": True,
    "vector_state_token": VectorState(state["sve_bytes"], state["sme_bytes"]).env_token,
  }


def validate_prelowered_scope(emitted, requested):
  """Reject an unavailable selection before spawning native workers."""
  for group in requested:
    origin = emitted.get(group["id"])
    if origin is None or not set(group["execution_ids"]) <= set(origin["execution_ids"]):
      raise ValueError("prelowered group does not cover selected execution IDs")


def subset_manifest(plan, ids):
  wanted = set(ids)
  return {
    key: plan[key]
    for key in (
      "source_hashes",
      "runner_hashes",
      "pass_options",
      "snapshot",
      "architecture",
      "label",
      "compiler_cflags",
    )
    if key in plan
  } | {
    "selected_execution_ids": list(ids),
    "executions": [row for row in plan["executions"] if row["execution_id"] in wanted],
  }


def verify_plan_sources(snapshot, plan):
  """Reject content drift under a reused snapshot path before starting work."""
  for section, prefix in (("source_hashes", snapshot), ("runner_hashes", snapshot / "benchmarks")):
    for name, expected in plan[section].items():
      path = prefix / name
      if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"immutable plan source changed: {path}")
  if plan.get("prelowered") is not None:
    if prelowered_identity(plan["prelowered"]["path"]) != plan["prelowered"]:
      raise ValueError("prelowered manifest changed since planning")
    driver = plan["native_build_driver"]
    if file_sha(driver["path"]) != driver["sha256"] or Path(driver["path"]).resolve() != Path(__file__).resolve():
      raise ValueError("prelowered native build driver changed")


def timing_deadline_base(directory, process_pid, started):
  """A fresh row/progress marker resets the deadline, not elapsed group time."""
  now = time.monotonic_ns()
  events = [
    record.get(field, 0)
    for record in read(directory / "timing.json", {}).values()
    if record.get("worker_pid") == process_pid
    for field in ("started_monotonic_ns", "finished_monotonic_ns")
  ]
  return max([started, *(value / 1e9 for value in events if 0 < value <= now)])


def unrun_stage(build, timing):
  if build["status"] not in ("built", "pending", "running"):
    return {"status": "not_run", "reason": "build_" + build["status"], "build_log": build.get("log")}
  if timing["status"] not in ("timed", "pending", "running"):
    return {"status": "not_run", "reason": "timing_" + timing["status"]}
  return {"status": "pending"}


def clean_work(work):
  if work.name != "work" or not (work / ".full-suite-owned-work").is_file():
    raise ValueError("refusing to remove an unmarked work directory")
  shutil.rmtree(work)


def retain(group_dir, index, work):
  """Commit binaries and compressed evidence before removing temporary builds."""
  retained = group_dir / "retained"
  retained.mkdir(exist_ok=True)
  binaries = {}
  for original in work.rglob("*.so"):
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    target = retained / f"{digest}.so"
    if not target.exists():
      shutil.copy2(original, target)
    binaries[str(original)] = str(target)
  for row in index.get("artifacts", {}).values():
    for role, path in row.items():
      if path is None:
        raise RuntimeError(f"missing compiled {role} artifact")
      original = Path(path)
      if str(original) not in binaries:
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        target = retained / f"{digest}.so"
        same_target = original.resolve() == target.resolve() or (target.exists() and original.samefile(target))
        if same_target:
          if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("same retained binary changed during SHA validation")
        else:
          shutil.copy2(original, target)
        binaries[str(original)] = str(target)
      row[role] = binaries[str(original)]
  archive = group_dir / f"evidence.{time.time_ns()}.tar.gz"
  temporary = archive.with_suffix(".gz.partial")
  wanted = []
  for path in work.rglob("*"):
    if path.is_file() and (
      path.suffix in (".cpp", ".h")
      and ("dsl-sources" in path.parts or "input" in path.parts)
      or "compiler-tmp" in path.parts
      and path.suffix in (".cpp", ".sh")
      or "identities" in path.parts and path.suffix == ".json"
      or path.name in ("compile_commands.json", "vecops-build.log", "CMakeCache.txt", "flags.make", "attestation.txt")
    ):
      wanted.append(path)
  with tarfile.open(temporary, "w:gz", compresslevel=6) as archive_stream:
    for path in sorted(wanted):
      archive_stream.add(path, arcname=str(Path("work") / path.relative_to(work)), recursive=False)
  with tarfile.open(temporary) as archive_stream:
    if len(archive_stream.getmembers()) != len(wanted):
      raise RuntimeError("evidence archive verification failed")
  os.replace(temporary, archive)
  index["evidence_archive"] = str(archive)
  index["evidence_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
  index["retained_binary_mapping"] = binaries
  index["retained_binary_sha256"] = {
    path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in set(binaries.values())
  }
  return index


def schema_probe_arguments(values, repeat):
  """Nonempty metadata-only views for can_invoke; never execute these views.

  Executable.can_invoke requires nonnull data even for schema validation.
  Each view therefore owns one aligned element and advertises the catalog's
  exact shape/strides. No kernel invocation or payload access occurs here.
  Timing constructs independent full-size storage using source_arguments.
  """
  import numpy as np
  import vecops
  from dsl_transpose import aligned_empty
  from vecops._dtype import dtype_name

  views = []
  for i, metadata in enumerate(values):
    storage = aligned_empty((1,), dtype_name(metadata.dtype))
    phantom = np.lib.stride_tricks.as_strided(
      storage,
      shape=metadata.shape,
      strides=tuple(stride * storage.dtype.itemsize for stride in metadata.strides),
    )
    views.append(vecops._C.TensorView.from_numpy(phantom, writable=i in (2, 3, 11, 12)))
  return vecops._C.ArgumentMetadata([*views, repeat])


def build_worker(args, source, plan, group, directory):
  import dsl_matmul_artifact_ab as ab
  import vecops
  import vecops.language as T

  work = directory / "work"
  if work.exists():
    clean_work(work)
  work.mkdir(parents=True)
  (work / ".full-suite-owned-work").write_text(group["id"])
  (work / "compiler-tmp").mkdir()
  os.environ["TMPDIR"] = str(work / "compiler-tmp")
  os.environ["CLANG_CRASH_DIAGNOSTICS_DIR"] = str(work / "compiler-tmp")
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cflags=plan["compiler_cflags"],
    cache_dir=work / "cache",
    build_dir=work / "build",
  )
  document = {**subset_manifest(plan, group["execution_ids"]), "artifact_sources": {}}
  scenarios = ab.source_scenarios(document)
  pairs = source.source_templates(scenarios[0])
  compiled = []
  prelowered = read_prelowered(plan, args.snapshot)
  if prelowered is not None:
    from dsl_matmul_repair_suite import compile_cpp_source, compile_prelowered, materialize_proof

    emitted, emitted_groups = prelowered
    emitted["_manifest_directory"] = str(Path(plan["prelowered"]["path"]).parent)
    save(directory / "prelowered-inputs.json", {
      "emission": plan["prelowered"], "origin_root": emitted["root"],
      "origin_frontend_sha256": emitted["compiler_python_sha256"],
      "source_plan": emitted["plan"], "source_plan_sha256": emitted["plan_sha256"],
      "native_build_driver": plan["native_build_driver"], "fresh_python_lowering": False,
    })
  with vecops.collect_compile_requests() as collector:
    for packing, pair in enumerate(pairs):
      if pair is None:
        continue
      for role, template in zip(("dsl", "cpp"), pair):
        label = ("pack_" if packing else "") + role
        if prelowered is None:
          kernel = T.compile(template, compiler=compiler, pass_options=plan["pass_options"] if role == "dsl" else None)
        elif role == "dsl":
          proof = prelowered_program(emitted, emitted_groups, group, label, template)
          proof = materialize_proof(proof, work)
          kernel = compile_prelowered(template, proof, compiler)
        else:
          kernel = compile_cpp_source(template, compiler)
        compiled.append((label, kernel))
    for scenario in scenarios:
      for label, kernel in compiled:
        kernel.operator(*source.source_metadata(scenario), 1, **kernel._core.specialization_values)
        document["artifact_sources"].setdefault(scenario.execution_id, {})[label] = str(kernel.source)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  manifest = directory / "manifest.json"
  save(manifest, document)
  # Index only metadata. Actual tensor storage and independent numerical/ABI
  # validation belong to the subsequent quiet timing phase, never a build job.
  source.source_arguments = lambda scenario: (source.source_metadata(scenario), None)
  ab.arguments = schema_probe_arguments
  index_path = directory / "index.json"
  ab.index_artifacts(
    SimpleNamespace(
      manifest=manifest, build_dir=[work], target=args.target, index_output=index_path, snapshot=args.snapshot
    )
  )
  index = retain(directory, read(index_path), work)
  save(index_path, index)
  if not args.keep_work:
    clean_work(work)
  verify_plan_sources(args.snapshot, plan)
  if prelowered is not None:
    for packing, pair in enumerate(pairs):
      if pair is not None:
        prelowered_program(emitted, emitted_groups, group, "pack_dsl" if packing else "dsl", pair[0])
  return {
    "status": "built", "index": str(index_path), "execution_ids": group["execution_ids"],
    "prelowered_inputs": str(directory / "prelowered-inputs.json") if prelowered is not None else None,
  }


def row_stage_availability(plan, execution_id, paths, stage):
  import dsl_matmul_artifact_ab as ab

  row = next((r for r in plan.get("executions", ()) if r["execution_id"] == execution_id), {})
  packing = row.get("packing")
  expected = packing.endswith(("packeda", "packedb", "packedab")) if packing is not None else "pack_cpp" in paths
  return ab.stage_availability(
    paths, paths, packing_only=stage == "packing", packing_expected=expected, reference_only=True
  )


def timing_worker(args, source, plan, group, directory):
  import dsl_matmul_artifact_ab as ab

  index = read(directory / "index.json")
  records = read(directory / "timing.json", {})
  for execution_id in group["execution_ids"]:
    for stage in args.stages:
      key = execution_id + "/" + stage
      if key in records and (
        records[key]["status"] in ("passed", "not_applicable")
        or (
          records[key].get("reason") != "missing_artifacts" and (not args.retry_failed or args.resume_after_row_failure)
        )
      ):
        continue
      paths = index.get("artifacts", {}).get(execution_id, {})
      availability = row_stage_availability(plan, execution_id, paths, stage)
      if availability["status"] != "available":
        records[key] = {"execution_id": execution_id, "stage": stage, **availability}
        save(directory / "timing.json", records)
        continue
      output = directory / "timing" / stage / execution_id
      manifest = output / "manifest.json"
      one_index = output / "index.json"
      save(manifest, subset_manifest(plan, [execution_id]))
      save(one_index, {**index, "manifest": str(manifest), "artifacts": {execution_id: paths}})
      (output / "results.json").unlink(missing_ok=True)
      records[key] = {
        "execution_id": execution_id,
        "stage": stage,
        "status": "running",
        "started_ns": time.time_ns(),
        "started_monotonic_ns": time.monotonic_ns(),
        "worker_pid": os.getpid(),
      }
      save(directory / "timing.json", records)
      try:
        ab.compare(
          SimpleNamespace(
            baseline_index=None,
            candidate_index=one_index,
            allow_baseline_subset=False,
            output_dir=output,
            cpu=args.cpu,
            rounds=args.rounds,
            packing_only=stage == "packing",
          )
        )
        rows = read(output / "results.json")["rows"]
        if len(rows) != 1:
          raise RuntimeError("expected one exact execution result")
        records[key] = {**rows[0], "status": "passed", "raw_clocks": str(output / "raw.jsonl")}
      except Exception as error:  # noqa: BLE001 -- ledger must retain numerical/runtime failures
        records[key] = {
          "execution_id": execution_id,
          "stage": stage,
          "status": failure(1, traceback.format_exc()),
          "error": str(error),
          "traceback": traceback.format_exc(),
        }
      records[key].update(worker_pid=os.getpid(), finished_monotonic_ns=time.monotonic_ns())
      save(directory / "timing.json", records)
  return {
    "status": "timed",
    "records": len(records),
    "failures": sum(r["status"] not in ("passed", "not_applicable") for r in records.values()),
  }


def process_identity(pid):
  """Linux identity used only for an explicitly launched child and descendants."""
  try:
    path = Path("/proc") / str(pid)
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    return {
      "pid": pid,
      "uid": path.stat().st_uid,
      "state": fields[0],
      "ppid": int(fields[1]),
      "pgid": int(fields[2]),
      "sid": int(fields[3]),
      "start_ticks": int(fields[19]),
    }
  except (FileNotFoundError, ProcessLookupError):
    return None


def same_process(expected, current):
  return current is not None and all(current[key] == expected[key] for key in ("pid", "uid", "start_ticks"))


def owned_process_tree(root):
  if not same_process(root, process_identity(root["pid"])) or root["uid"] != os.getuid():
    return {}
  all_processes = {int(p.name): process_identity(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()}
  owned = {root["pid"]: root}
  while True:
    children = {
      pid: record
      for pid, record in all_processes.items()
      if record and record["uid"] == root["uid"] and record["ppid"] in owned and pid not in owned
    }
    if not children:
      return owned
    owned.update(children)


def open_process_descriptor(pid):
  """Use Linux pidfds even when Python was built against pre-pidfd headers."""
  if hasattr(os, "pidfd_open"):
    return os.pidfd_open(pid)
  import ctypes
  import platform

  if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
    return None
  libc = ctypes.CDLL(None, use_errno=True)
  # Linux x86-64 and AArch64 both assign pidfd_open syscall number 434.
  descriptor = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
  if descriptor < 0:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))
  return descriptor


def send_process_descriptor(descriptor, signum):
  if hasattr(signal, "pidfd_send_signal"):
    signal.pidfd_send_signal(descriptor, signum)
    return
  import ctypes

  libc = ctypes.CDLL(None, use_errno=True)
  result = libc.syscall(
    ctypes.c_long(424), ctypes.c_int(descriptor), ctypes.c_int(signum), ctypes.c_void_p(), ctypes.c_uint(0)
  )
  if result < 0:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))


def signal_owned_processes(owned, signum):
  """Signal captured processes individually; pidfds pin Linux process identity."""
  events = []
  for pid, expected in owned.items():
    descriptor = None
    try:
      descriptor = open_process_descriptor(pid)
      current = process_identity(pid)
      if not same_process(expected, current) or current["state"] == "Z":
        continue
      if descriptor is not None:
        send_process_descriptor(descriptor, signum)
        method = "pidfd"
      else:
        # Platforms without pidfds retain a conservative identity check. This
        # fallback cannot close the check-to-signal PID reuse window atomically.
        os.kill(pid, signum)
        method = "pid_checked"
      events.append({"target": ("pid", pid), "signal": signum, "method": method})
    except ProcessLookupError:
      pass
    finally:
      if descriptor is not None:
        os.close(descriptor)
  return events


def stop(process, *, grace=5, audit_path=None):
  """Freeze and terminate the owned subtree, including separate compiler groups."""
  root = getattr(process, "_full_suite_identity", None) or process_identity(process.pid)
  if root is None:
    process.poll()
    return
  owned = owned_process_tree(root)
  events = []
  # Stop launchers before recapturing children so Ninja cannot race into a new TU.
  for _ in range(3):
    events.extend(signal_owned_processes(owned, signal.SIGSTOP))
    refreshed = owned_process_tree(root)
    if refreshed.keys() <= owned.keys():
      break
    owned.update(refreshed)
  events.extend(signal_owned_processes(owned, signal.SIGTERM))
  events.extend(signal_owned_processes(owned, signal.SIGCONT))
  deadline = time.monotonic() + grace
  while time.monotonic() < deadline:
    process.poll()
    if not any(
      same_process(record, current := process_identity(pid)) and current["state"] != "Z"
      for pid, record in owned.items()
    ):
      break
    time.sleep(0.02)
  events.extend(signal_owned_processes(owned, signal.SIGKILL))
  process.wait(timeout=max(1, grace))
  if audit_path is not None:
    save(
      audit_path,
      {
        "root": root,
        "captured": list(owned.values()),
        "signals": events,
        "remaining": [
          r
          for pid, expected in owned.items()
          if same_process(expected, r := process_identity(pid)) and r["state"] != "Z"
        ],
      },
    )


def timing_group_pending(args, plan, group):
  """Keep the single-group resume and stage-availability rules for batches."""
  directory = args.suite_dir / "groups" / group["id"]
  state_path = directory / "time.json"
  state = read(state_path, {})
  records = read(directory / "timing.json", {})
  current = [records.get(eid + "/" + stage) for eid in group["execution_ids"] for stage in args.stages]
  if state.get("status") == "timed" and all(current) and (
    (not args.retry_failed and not any(r.get("reason") == "missing_artifacts" for r in current))
    or all(r["status"] in ("passed", "not_applicable") for r in current)
  ):
    return False
  if state.get("status") not in (None, "running", "timed") and not args.retry_failed:
    return False
  if read(directory / "build.json", {}).get("status") != "built":
    index = read(directory / "index.json", {})
    available = False
    for eid in group["execution_ids"]:
      paths = index.get("artifacts", {}).get(eid, {})
      for stage in args.stages:
        readiness = row_stage_availability(plan, eid, paths, stage)
        if readiness["status"] == "available":
          available = True
        else:
          records.setdefault(eid + "/" + stage, {"execution_id": eid, "stage": stage, **readiness})
    save(directory / "timing.json", records)
    if not available:
      save(state_path, {"status": "timed", "records": len(records),
                        "failures": sum(r["status"] == "not_run" for r in records.values())})
      return False
  return True


def timing_batch_worker(args, source, plan, request_path):
  """Reuse imports, retaining independent exact-row compare calls and ledgers."""
  request = read(request_path)
  groups = {group["id"]: group for group in plan["groups"]}
  for entry in request["groups"]:
    group = groups[entry["id"]]
    directory = args.suite_dir / "groups" / group["id"]
    marker = {"group_id": group["id"], "worker_pid": os.getpid(),
              "started_monotonic_ns": time.monotonic_ns()}
    save(request_path.with_suffix(".progress.json"), marker)
    save(directory / "time.json", {"status": "running", "pid": os.getpid(),
                                  "execution_ids": group["execution_ids"]})
    args.resume_after_row_failure = entry.get("recovering", False)
    try:
      result = timing_worker(args, source, plan, group, directory)
    except Exception:  # noqa: BLE001 -- A setup exception affects only this group, not the next.
      text = traceback.format_exc()
      print(text, flush=True)
      records = read(directory / "timing.json", {})
      for record in records.values():
        if record.get("status") == "running" and record.get("worker_pid") == os.getpid():
          record.update(status="group_setup_failed", error=text,
                        finished_monotonic_ns=time.monotonic_ns())
      for eid in group["execution_ids"]:
        for stage in args.stages:
          records.setdefault(eid + "/" + stage, {"execution_id": eid, "stage": stage,
                                                "status": "group_setup_failed", "error": text})
      save(directory / "timing.json", records)
      result = {"status": "failed", "error": text, "execution_ids": group["execution_ids"]}
    save(directory / "time.json", result)
    save(request_path.with_suffix(".progress.json"), {**marker, "completed": True,
                                                       "finished_monotonic_ns": time.monotonic_ns()})
  return 0


def dispatch_timing_batches(args, plan, groups):
  """One quiet worker at a time; restart only unfinished work after a crash."""
  pending = list(groups)
  batch_dir = args.suite_dir / "timing-batches"
  batch_dir.mkdir(parents=True, exist_ok=True)
  while pending:
    batch = []
    while pending and len(batch) < args.timing_batch_size:
      group = pending.pop(0)
      if timing_group_pending(args, plan, group):
        batch.append(group)
    if not batch:
      continue
    request = batch_dir / f"batch-{time.time_ns()}.json"
    save(request, {"groups": [{"id": g["id"], "recovering": g.get("recovering", False)} for g in batch]})
    log_path = request.with_suffix(".log")
    progress_path = request.with_suffix(".progress.json")
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    command = [sys.executable, str(Path(__file__).resolve()), *args.runner_arguments,
               "--worker", "time", "--worker-batch-file", str(request)]
    with log_path.open("a") as log:
      process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=environment)
      process._full_suite_identity = process_identity(process.pid)
      begin = time.monotonic()
      expired = False
      try:
        while True:
          marker = read(progress_path, {})
          current = next((g for g in batch if g["id"] == marker.get("group_id")), batch[0])
          directory = args.suite_dir / "groups" / current["id"]
          start = max(begin, marker.get("finished_monotonic_ns", marker.get("started_monotonic_ns", 0)) / 1e9)
          deadline = timing_deadline_base(directory, process.pid, start)
          expired = time.monotonic() - deadline >= args.timing_timeout
          if expired:
            stop(process)
          code = process.poll()
          if code is not None:
            break
          time.sleep(0.2)
      finally:
        if process.poll() is None:
          stop(process)
    marker = read(progress_path, {})
    current_index = next((i for i, g in enumerate(batch) if g["id"] == marker.get("group_id")), 0)
    current = batch[current_index]
    directory = args.suite_dir / "groups" / current["id"]
    if not expired and code == 0:
      for group in batch:
        print("time", group["index"], read(args.suite_dir / "groups" / group["id"] / "time.json", {}).get("status"), flush=True)
      continue
    # A completed-group marker attributes a between-group crash to the next
    # group's setup. Successfully persisted groups never become failures.
    if marker.get("completed"):
      current_index += 1
      if current_index == len(batch):
        continue
      current = batch[current_index]
      directory = args.suite_dir / "groups" / current["id"]
    text = log_path.read_text(errors="replace")
    status = failure(code, text, timed_out=expired)
    records = read(directory / "timing.json", {})
    had_running = False
    for record in records.values():
      if record.get("status") == "running" and record.get("worker_pid") == process.pid:
        had_running = True
        record.update(status=status, returncode=code, log=str(log_path))
    complete = all(eid + "/" + stage in records for eid in current["execution_ids"] for stage in args.stages)
    state = {"status": status, "returncode": code, "log": str(log_path),
             "execution_ids": current["execution_ids"], "error": text[-12000:]}
    remaining = batch[current_index + 1:]
    if had_running and not complete:
      state.update(status="running", resume_after_row_failure=True)
      remaining.insert(0, {**current, "recovering": True})
    elif not had_running:
      for eid in current["execution_ids"]:
        for stage in args.stages:
          records.setdefault(eid + "/" + stage, {"execution_id": eid, "stage": stage,
                                                "status": "group_setup_failed", "error": text[-12000:]})
    save(directory / "timing.json", records)
    save(directory / "time.json", state)
    pending = remaining + pending
    print("time", current["index"], state["status"], flush=True)


def dispatch(args, plan, groups, phase):
  if phase == "time" and getattr(args, "timing_batch_size", 1) > 1:
    return dispatch_timing_batches(args, plan, groups)
  pending, active = list(groups), {}
  workers = args.build_workers if phase == "build" else 1
  timeout = args.build_timeout if phase == "build" else args.timing_timeout
  try:
    while pending or active:
      while pending and len(active) < workers:
        group = pending.pop(0)
        directory = args.suite_dir / "groups" / group["id"]
        state_path = directory / (phase + ".json")
        state = read(state_path, {})
        if phase == "build" and state.get("status") == "built":
          continue
        if phase == "time" and state.get("status") == "timed":
          records = read(directory / "timing.json", {})
          current = [records.get(eid + "/" + stage) for eid in group["execution_ids"] for stage in args.stages]
          if all(current) and (
            (not args.retry_failed and not any(r.get("reason") == "missing_artifacts" for r in current))
            or all(r["status"] in ("passed", "not_applicable") for r in current)
          ):
            continue
        elif state and state.get("status") != "running" and not args.retry_failed:
          continue
        if phase == "time" and read(directory / "build.json", {}).get("status") != "built":
          index = read(directory / "index.json", {})
          records = read(directory / "timing.json", {})
          available = False
          for eid in group["execution_ids"]:
            paths = index.get("artifacts", {}).get(eid, {})
            for stage in args.stages:
              readiness = row_stage_availability(plan, eid, paths, stage)
              if readiness["status"] == "available":
                available = True
              else:
                records.setdefault(eid + "/" + stage, {"execution_id": eid, "stage": stage, **readiness})
          save(directory / "timing.json", records)
          if not available:
            save(
              state_path,
              {
                "status": "timed",
                "records": len(records),
                "failures": sum(r["status"] == "not_run" for r in records.values()),
              },
            )
            continue
        if phase == "build" and shutil.disk_usage(args.suite_dir).free < args.min_free_gb * 2**30:
          save(state_path, {"status": "insufficient_disk", "execution_ids": group["execution_ids"]})
          continue
        directory.mkdir(parents=True, exist_ok=True)
        log = (directory / (phase + ".log")).open("a")
        command = [
          sys.executable,
          str(Path(__file__).resolve()),
          *args.runner_arguments,
          "--worker",
          phase,
          "--worker-group",
          group["id"],
        ]
        if group.get("recovering"):
          command.append("--resume-after-row-failure")
        environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        process = subprocess.Popen(
          command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment
        )
        process._full_suite_identity = process_identity(process.pid)
        save(
          state_path,
          {
            "status": "running",
            "pid": process.pid,
            "started_ns": time.time_ns(),
            "execution_ids": group["execution_ids"],
          },
        )
        active[process.pid] = (process, time.monotonic(), group, directory, log)
      for pid, (process, begin, group, directory, log) in list(active.items()):
        deadline = timing_deadline_base(directory, process.pid, begin) if phase == "time" else begin
        expired = time.monotonic() - deadline >= timeout
        if expired:
          stop(process)
        code = process.poll()
        if code is None:
          continue
        log.close()
        state_path = directory / (phase + ".json")
        state = read(state_path, {})
        if expired or code != 0 or state.get("status") == "running":
          text = (directory / (phase + ".log")).read_text(errors="replace")
          save(
            state_path,
            {
              "status": failure(code, text, timed_out=expired),
              "returncode": code,
              "execution_ids": group["execution_ids"],
              "log": str(directory / (phase + ".log")),
              "error": text[-12000:],
            },
          )
        if phase == "build" and (directory / "work").exists() and not args.keep_work:
          try:
            evidence = retain(directory, {}, directory / "work")
            save(directory / "failure-evidence.json", evidence)
            clean_work(directory / "work")
          except Exception:  # noqa: BLE001 -- retain failed child/archive diagnostics
            traceback.print_exc()
        if phase == "time" and (expired or code != 0):
          records = read(directory / "timing.json", {})
          had_running = any(record.get("status") == "running" for record in records.values())
          for record in records.values():
            if record.get("status") == "running":
              record.update(
                status=failure(code, text, timed_out=expired), returncode=code, log=str(directory / "time.log")
              )
          save(directory / "timing.json", records)
          complete = all(eid + "/" + stage in records for eid in group["execution_ids"] for stage in args.stages)
          if not complete and had_running:
            save(
              state_path,
              {"status": "running", "execution_ids": group["execution_ids"], "resume_after_row_failure": True},
            )
            pending.append({**group, "recovering": True})
          if not complete and not had_running:
            for eid in group["execution_ids"]:
              for stage in args.stages:
                records.setdefault(
                  eid + "/" + stage,
                  {"execution_id": eid, "stage": stage, "status": "group_setup_failed", "error": text[-12000:]},
                )
            save(directory / "timing.json", records)
        print(phase, group["index"], read(state_path).get("status"), flush=True)
        del active[pid]
      if active:
        time.sleep(0.2)
  finally:
    for process, _, group, directory, log in active.values():
      stop(process)
      log.close()
      save(directory / (phase + ".json"), {"status": "interrupted", "execution_ids": group["execution_ids"]})


def ledger(args, plan):
  rows = []
  for group in plan["groups"]:
    directory = args.suite_dir / "groups" / group["id"]
    build = read(directory / "build.json", {"status": "pending"})
    timing = read(directory / "time.json", {"status": "pending"})
    records = read(directory / "timing.json", {})
    for execution_id in group["execution_ids"]:
      rows.append(
        {
          "execution_id": execution_id,
          "group": group["id"],
          "group_index": group["index"],
          "build": build["status"],
          "timing": timing["status"],
          "stages": {
            stage: records.get(execution_id + "/" + stage, unrun_stage(build, timing))
            for stage in plan["timing"]["stages"]
          },
        }
      )
  save(
    args.suite_dir / "ledger.json",
    {
      "snapshot": plan["snapshot"],
      "architecture": plan["architecture"],
      "label": plan["label"],
      "requested": len(rows),
      "rows": rows,
    },
  )


def main(argv=None):
  from dsl_benchmark_selection import selection_identity, verify_selection_resume
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--snapshot", type=Path, required=True)
  parser.add_argument("--suite-dir", type=Path, required=True)
  parser.add_argument("--architecture", choices=("arm", "x86"), required=True)
  parser.add_argument("--feature", action="append")
  parser.add_argument("--execution-ids-file", type=Path, help="exact architecture/execution_ids JSON selection; content-bound on resume")
  parser.add_argument("--prelowered-manifest", type=Path, help="completed, snapshot-matching source-equivalence manifest for native-only construction")
  parser.add_argument("--complete-registration", action="store_true")
  parser.add_argument("--level", nargs="+", choices=("matmul", "dot"), default=["matmul"])
  parser.add_argument("--phase", choices=("plan", "build", "time", "all", "report"), default="all")
  parser.add_argument("--plan-only", action="store_true", help="validate and save exact plan without building or timing")
  parser.add_argument("--groups", help="group indexes, e.g. 0-31,40")
  parser.add_argument("--shard-index", type=int, default=0)
  parser.add_argument("--shard-count", type=int, default=1)
  parser.add_argument("--build-workers", type=int, default=4)
  parser.add_argument("--build-cpus", help="allowed build CPU indexes/ranges")
  parser.add_argument("--jobs", type=int, default=1)
  parser.add_argument("--build-timeout", type=float, default=1800)
  parser.add_argument("--timing-timeout", type=float, default=1800)
  parser.add_argument("--timing-batch-size", type=int, default=1, help="groups per timing process; row timing is unchanged")
  parser.add_argument("--min-free-gb", type=float, default=2)
  parser.add_argument("--cpu", type=int, required=True)
  parser.add_argument("--rounds", type=int, default=12)
  parser.add_argument("--stages", nargs="+", choices=("compute", "packing"), default=["compute"])
  parser.add_argument("--extra-options", default="{}")
  parser.add_argument("--label", default="default")
  parser.add_argument("--target", default="Native")
  parser.add_argument("--cc")
  parser.add_argument("--cxx")
  parser.add_argument("--cflag", action="append", default=["-DVECOPS_MATH_ASSUME_VALID_INPUTS"])
  parser.add_argument("--retry-failed", action="store_true")
  parser.add_argument("--keep-work", action="store_true")
  parser.add_argument("--worker", choices=("build", "time"), help=argparse.SUPPRESS)
  parser.add_argument("--worker-group", help=argparse.SUPPRESS)
  parser.add_argument("--worker-batch-file", type=Path, help=argparse.SUPPRESS)
  parser.add_argument("--resume-after-row-failure", action="store_true", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)
  if args.plan_only and (args.worker or args.worker_group or args.worker_batch_file or args.phase not in ("all", "plan")):
    parser.error("plan-only cannot be combined with workers or execution phases")
  if args.prelowered_manifest is not None and (not args.cc or not args.cxx):
    parser.error("prelowered construction requires explicit --cc and --cxx")
  args.runner_arguments = list(sys.argv[1:] if argv is None else argv)
  if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count or min(args.jobs, args.build_workers, args.timing_batch_size) < 1:
    parser.error("positive workers/jobs/shard-count and valid shard-index required")
  if args.rounds < 2 or args.rounds % 2 or min(args.build_timeout, args.timing_timeout) <= 0:
    parser.error("positive timeouts and positive even rounds required")
  args.snapshot, args.suite_dir = args.snapshot.resolve(), args.suite_dir.resolve()
  args.suite_dir.mkdir(parents=True, exist_ok=True)
  if args.worker == "build":
    allowed = ranges(args.build_cpus) if args.build_cpus else set(os.sched_getaffinity(0)) - {args.cpu}
    if not allowed:
      parser.error("no build CPU remains")
    os.sched_setaffinity(0, allowed)
  os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
  source = catalog(args.snapshot)
  import torch

  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)
  plan_path = args.suite_dir / "plan.json"
  plan = read(plan_path)
  if plan is None:
    plan = make_plan(args, source)
    save(plan_path, plan)
  elif any(
    (
      plan["snapshot"] != str(args.snapshot),
      plan["architecture"] != args.architecture,
      plan["label"] != args.label,
      plan["pass_options"] != json.loads(args.extra_options),
      plan["compiler_cflags"] != args.cflag,
      plan["target"] != args.target,
      plan["complete_registration"] != args.complete_registration,
      plan["levels"] != args.level,
      plan["features"] != args.feature,
      plan["toolchain"] != {"cc": args.cc, "cxx": args.cxx},
      plan.get("prelowered") != prelowered_identity(args.prelowered_manifest),
      plan["timing"]["cpu"] != args.cpu,
      plan["timing"]["rounds"] != args.rounds,
      plan["timing"].get("batch_size", 1) != args.timing_batch_size,
    )
  ):
    raise ValueError("resume configuration differs from immutable plan; use a different suite directory")
  verify_selection_resume(plan.get("selection"), selection_identity(args.execution_ids_file, args.architecture, "execution_ids"))
  if not args.worker:
    plan["timing"]["stages"] = list(dict.fromkeys((*plan["timing"]["stages"], *args.stages)))
    save(plan_path, plan)
  verify_plan_sources(args.snapshot, plan)
  if not args.worker and plan.get("prelowered") is not None:
    _, emitted_groups = read_prelowered(plan, args.snapshot)
    validate_prelowered_scope(emitted_groups, selected(plan["groups"], args))
  if args.plan_only:
    selected(plan["groups"], args)
    return 0
  if args.worker and args.worker_batch_file is not None:
    if args.worker != "time":
      parser.error("batch workers are only supported for timing")
    return timing_batch_worker(args, source, plan, args.worker_batch_file)
  if args.worker:
    group = next(g for g in plan["groups"] if g["id"] == args.worker_group)
    directory = args.suite_dir / "groups" / group["id"]
    try:
      result = (build_worker if args.worker == "build" else timing_worker)(args, source, plan, group, directory)
      save(directory / (args.worker + ".json"), result)
      return 0
    except Exception:  # noqa: BLE001 -- retain failed child/archive diagnostics
      traceback.print_exc()
      return 1
  groups = selected(plan["groups"], args)
  print(
    f"{len(plan['executions'])} real executions, {len(plan['groups'])} compilation groups; selected {len(groups)}",
    flush=True,
  )
  if args.phase in ("build", "all"):
    dispatch(args, plan, groups, "build")
  if args.phase in ("time", "all"):
    dispatch(args, plan, groups, "time")
  ledger(args, plan)
  return 0


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
  raise SystemExit(main())
