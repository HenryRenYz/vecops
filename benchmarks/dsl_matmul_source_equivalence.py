# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Attest exact generated C++/ABI equivalence before replaying native artifacts.

Emission uses fresh snapshot processes and never compiles a native kernel.
Comparison requires every planned group, matching SDK/toolchain/flags, and
identical generated source and template contracts. Optional native admission
also checks the retained build archive's actual source bytes. A changed group
must be rebuilt; this tool never rewrites an artifact key or an old suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


def read(path):
  return json.loads(Path(path).read_text())


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(dir=path.parent)
  try:
    with os.fdopen(descriptor, "w") as stream:
      json.dump(data, stream, indent=2, allow_nan=False)
      stream.write("\n")
    os.replace(temporary, path)
  finally:
    Path(temporary).unlink(missing_ok=True)


def tree_sha(root, suffixes=None):
  data = [
    (str(path.relative_to(root)), sha(path))
    for path in sorted(root.rglob("*"))
    if path.is_file() and (suffixes is None or path.suffix in suffixes)
  ]
  return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def native_environment(root, plan):
  tools = {}
  for role, command in plan["toolchain"].items():
    executable = Path(shutil.which(command) or command).resolve()
    version = subprocess.check_output([str(executable), "--version"], text=True)
    tools[role] = {"path": str(executable), "sha256": sha(executable), "version": version}
  return {
    "sdk_sha256": tree_sha(root / "python/vecops/_sdk"),
    "extension_sha256": sorted(sha(path) for path in (root / "python/vecops").glob("_C*.so")),
    "target": plan["target"],
    "toolchain": tools,
    "machine": platform.machine(),
    "host": platform.node(),
    "compiler_cflags": plan["compiler_cflags"],
    "environment": {
      name: os.environ.get(name)
      for name in (
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "C_INCLUDE_PATH",
        "LIBRARY_PATH",
        "SDKROOT",
      )
    },
  }


def emit_worker(args, plan):
  sys.path.insert(0, str(args.root / "benchmarks"))
  from dsl_snapshot_entry import select_snapshot

  select_snapshot(args.root)
  import dsl_matmul_artifact_ab as ab
  import dsl_matmul_source_catalog as source
  import vecops
  import vecops.language as T
  from dsl_matmul_full_suite import subset_manifest

  compiler = vecops.Compiler(
    target=plan["target"],
    cc=plan["toolchain"]["cc"],
    cxx=plan["toolchain"]["cxx"],
    cflags=plan["compiler_cflags"],
    cache_dir=args.output / "policy-cache",
    build_dir=args.output / "unused-native-build",
  )
  for group in plan["groups"]:
    if group["index"] % args.workers != args.worker_index:
      continue
    output = args.output / "groups" / f"{group['id']}.json"
    if output.exists():
      continue
    scenario = ab.source_scenarios(subset_manifest(plan, group["execution_ids"]))[0]
    programs = {}
    begin = time.perf_counter()
    for packing, pair in enumerate(source.source_templates(scenario)):
      if pair is None:
        continue
      template = pair[0]
      compiled = T.compile(template, compiler=compiler, pass_options=plan["pass_options"])
      text = compiled.generated_source
      digest = hashlib.sha256(text.encode()).hexdigest()
      destination = args.output / "sources" / f"{digest}.cpp"
      destination.parent.mkdir(parents=True, exist_ok=True)
      if destination.exists() and sha(destination) != digest:
        raise ValueError("content-addressed emitted source changed")
      if not destination.exists():
        destination.write_text(text)
      contract, captures = ab.template_contract(template)
      programs["pack_dsl" if packing else "dsl"] = {
        "source_sha256": digest,
        "source_bytes": len(text.encode()),
        "contract": contract,
        "captures": captures,
        "name": template.ir.name,
        "runtime_parallelism": True,
        "effective_cflags": list(compiled.operator.compiler.cflags),
      }
    save(
      output,
      {
        "id": group["id"],
        "index": group["index"],
        "execution_ids": group["execution_ids"],
        "programs": programs,
        "vector_state": {"sve_bytes": compiler.vector_state.sve_bytes, "sme_bytes": compiler.vector_state.sme_bytes},
        "lowering_seconds": time.perf_counter() - begin,
      },
    )
    print(group["index"], group["id"], flush=True)


def emit(args):
  plan = read(args.plan)
  if args.worker_index is not None:
    emit_worker(args, plan)
    return 0
  args.output.mkdir(parents=True, exist_ok=True)
  admission = {
    "root": str(args.root),
    "plan": str(args.plan),
    "plan_sha256": sha(args.plan),
    "driver_sha256": sha(__file__),
    "native_environment": native_environment(args.root, plan),
    "compiler_python_sha256": tree_sha(args.root / "python/vecops", {".py"}),
    "group_ids": [group["id"] for group in plan["groups"]],
  }
  path = args.output / "admission.json"
  if path.exists() and read(path) != admission:
    raise ValueError("source-emission inputs changed; use a new output directory")
  save(path, admission)
  children = []
  for index in range(args.workers):
    log = (args.output / f"worker-{index}.log").open("w")
    command = [
      sys.executable,
      str(Path(__file__).resolve()),
      "--phase",
      "emit",
      "--root",
      str(args.root),
      "--plan",
      str(args.plan),
      "--output",
      str(args.output),
      "--workers",
      str(args.workers),
      "--worker-index",
      str(index),
    ]
    children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
  codes = []
  for process, log in children:
    codes.append(process.wait())
    log.close()
  if any(codes):
    raise RuntimeError(f"source emission failed: {codes}")
  groups = [read(args.output / "groups" / f"{identity}.json") for identity in admission["group_ids"]]
  save(args.output / "manifest.json", {**admission, "completed": True, "groups": groups})
  return 0


def compare_documents(before, after):
  if not before.get("completed") or not after.get("completed"):
    raise ValueError("source equivalence requires complete emissions")
  if before["native_environment"] != after["native_environment"]:
    raise ValueError("SDK/toolchain/native policy changed")
  if before["group_ids"] != after["group_ids"] or len(set(before["group_ids"])) != len(before["group_ids"]):
    raise ValueError("source group scopes differ or contain duplicates")
  old = {group["id"]: group for group in before["groups"]}
  new = {group["id"]: group for group in after["groups"]}
  if set(old) != set(before["group_ids"]) or old.keys() != new.keys():
    raise ValueError("missing emitted groups")
  changed = []
  for identity in before["group_ids"]:
    fields = ("execution_ids", "programs", "vector_state")
    if any(old[identity][field] != new[identity][field] for field in fields):
      changed.append(identity)
  return {"all_programs_identical": not changed, "groups": len(old), "changed_groups": changed}


def compare(args):
  before, after = read(args.baseline / "manifest.json"), read(args.candidate / "manifest.json")
  for document in (before, after):
    if tree_sha(Path(document["root"]) / "python/vecops", {".py"}) != document["compiler_python_sha256"]:
      raise ValueError("emitted compiler source snapshot changed")
  report = compare_documents(before, after)
  source_inputs = {}
  for directory, document in ((args.baseline, before), (args.candidate, after)):
    for group in document["groups"]:
      for program in group["programs"].values():
        path = directory / "sources" / f"{program['source_sha256']}.cpp"
        if sha(path) != program["source_sha256"]:
          raise ValueError("emitted source no longer matches its manifest")
        source_inputs[str(path)] = program["source_sha256"]
  native_sources = {}
  if args.native_suite:
    from dsl_matmul_artifact_ab import adapter_contract

    for group in before["groups"]:
      folder = args.native_suite / "groups" / group["id"]
      if read(folder / "build.json").get("status") != "built":
        raise ValueError(f"native group is not built: {group['id']}")
      index = read(folder / "index.json")
      manifest = read(index["manifest"])
      archive = Path(index["evidence_archive"])
      if sha(archive) != index["evidence_sha256"]:
        raise ValueError("retained native evidence archive changed")
      with tarfile.open(archive) as stream:
        for role, program in group["programs"].items():
          for execution, paths in manifest["artifact_sources"].items():
            original = Path(paths[role])
            member = "work/dsl-sources/" + original.parent.name + "/kernel.cpp"
            content = stream.extractfile(member)
            if content is None or hashlib.sha256(content.read()).hexdigest() != program["source_sha256"]:
              raise ValueError(f"native build source differs from emitted source: {group['id']}/{role}")
            audit = index["schema_audit"][execution][role]
            adapter = Path(audit["adapter"])
            relative = adapter.relative_to(folder / "work")
            adapter_text = stream.extractfile("work/" + str(relative)).read().decode()
            actual_contract = json.loads(json.dumps(adapter_contract(adapter_text)))
            if actual_contract != program["contract"]:
              raise ValueError("retained native adapter does not match emitted ABI")
            specialization = stream.extractfile("work/" + str(relative.with_name("vecops_spec.h"))).read()
            if hashlib.sha256(specialization).hexdigest() != audit["specialization_sha256"]:
              raise ValueError("retained specialization header changed")
            text = specialization.decode()
            selectors = {
              name: int(value)
              for name, value in re.findall(r"inline constexpr auto (\w+) = ::vecops::meta::cint<(-?\d+)>;", text)
            }
            selectors.update(dict(re.findall(r"using (\w+) = ::vecops::(\w+);", text)))
            if any(selectors.get(name) != value for name, value in program["captures"].items()):
              raise ValueError("retained specialization captures differ")
            flags = shlex.split(audit["compile_command"]["command"])
            if any(flag not in flags for flag in program["effective_cflags"]):
              raise ValueError("retained native compile policy differs")
      native_sources[str(archive)] = index["evidence_sha256"]
  report.update(
    baseline=str(args.baseline),
    candidate=str(args.candidate),
    native_suite=str(args.native_suite) if args.native_suite else None,
    frontend_snapshot=after["root"],
    frontend_python_sha256=after["compiler_python_sha256"],
    native_environment=after["native_environment"],
    native_snapshot=before["root"],
    input_sha256={
      str(args.baseline / "manifest.json"): sha(args.baseline / "manifest.json"),
      str(args.candidate / "manifest.json"): sha(args.candidate / "manifest.json"),
    },
    source_inputs=source_inputs,
    native_evidence=native_sources,
  )
  save(args.output, report)
  return int(not report["all_programs_identical"])


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--phase", choices=("emit", "compare"), required=True)
  parser.add_argument("--root", type=Path)
  parser.add_argument("--plan", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--workers", type=int, default=16)
  parser.add_argument("--worker-index", type=int)
  parser.add_argument("--baseline", type=Path)
  parser.add_argument("--candidate", type=Path)
  parser.add_argument("--native-suite", type=Path)
  args = parser.parse_args(argv)
  for key in ("root", "plan", "output", "baseline", "candidate", "native_suite"):
    if getattr(args, key) is not None:
      setattr(args, key, getattr(args, key).resolve())
  if args.phase == "emit":
    if args.root is None or args.plan is None or args.workers < 1:
      parser.error("emission requires root, plan and positive workers")
    return emit(args)
  if args.baseline is None or args.candidate is None:
    parser.error("comparison requires both source-emission directories")
  return compare(args)


if __name__ == "__main__":
  raise SystemExit(main())
