# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Remeasure retained old/new kernels together after host frequency changes.

Both suites must have completed native builds. Historical timing values are
never used: the existing artifact A/B runner independently validates and times
baseline, candidate and C++ in alternating order on the same CPU. Original
suite directories and binaries remain untouched. A new output directory is
required; every selected execution and applicable packing stage is mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace

from dsl_benchmark_selection import selection_identity
from dsl_matmul_full_suite import subset_manifest, verify_plan_sources


def read(path):
  return json.loads(Path(path).read_text())


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frequency(cpu):
  directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
  result = {}
  for name in ("scaling_cur_freq", "cpuinfo_cur_freq", "scaling_min_freq", "scaling_max_freq", "scaling_governor"):
    path = directory / name
    try:
      result[name] = path.read_text().strip()
    except OSError:
      pass
  return result


def suite_indexes(directory, plan, wanted):
  result, inputs = {}, {}
  for group in plan["groups"]:
    selected = wanted.intersection(group["execution_ids"])
    if not selected:
      continue
    folder = directory / "groups" / group["id"]
    build_path, index_path = folder / "build.json", folder / "index.json"
    if read(build_path).get("status") != "built":
      raise ValueError(f"unbuilt group: {folder}")
    index = read(index_path)
    inputs[str(build_path)] = digest(build_path)
    inputs[str(index_path)] = digest(index_path)
    manifest = Path(index["manifest"])
    inputs[str(manifest)] = digest(manifest)
    if index.get("evidence_archive"):
      archive = index["evidence_archive"]
      if digest(archive) != index["evidence_sha256"]:
        raise ValueError(f"native build evidence changed: {archive}")
      inputs[archive] = index["evidence_sha256"]
    for execution in selected:
      if execution in result:
        raise ValueError(f"duplicate execution: {execution}")
      artifacts = index["artifacts"][execution]
      for binary in artifacts.values():
        expected = index.get("retained_binary_sha256", {}).get(binary)
        if expected is None or digest(binary) != expected:
          raise ValueError(f"retained binary is not attested or changed: {binary}")
        inputs[binary] = expected
      result[execution] = index
  if result.keys() != wanted:
    raise ValueError(f"missing native executions: {sorted(wanted - result.keys())}")
  return result, inputs


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--baseline-suite", type=Path, required=True)
  parser.add_argument("--candidate-suite", type=Path, required=True)
  parser.add_argument("--selection", type=Path, required=True)
  parser.add_argument(
    "--source-equivalence", type=Path, help="admitted exact C++/ABI replay proof for a newer frontend snapshot"
  )
  parser.add_argument("--architecture", choices=("x86", "arm"), required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--cpu", type=int, required=True)
  parser.add_argument("--rounds", type=int, default=12)
  parser.add_argument("--stages", nargs="+", choices=("compute", "packing"), default=("compute", "packing"))
  args = parser.parse_args(argv)
  if args.rounds < 1 or args.cpu not in os.sched_getaffinity(0):
    parser.error("rounds must be positive and CPU must be allowed")
  if args.output.exists():
    parser.error("use a new output directory to retain independent measurement evidence")
  args.root = args.root.resolve()
  directories = (args.baseline_suite.resolve(), args.candidate_suite.resolve())
  plans = tuple(read(directory / "plan.json") for directory in directories)
  identity = selection_identity(args.selection, args.architecture, "execution_ids")
  wanted = set(identity["execution_ids"])
  metadata = []
  for plan in plans:
    if plan["architecture"] != args.architecture:
      raise ValueError("suite architecture mismatch")
    verify_plan_sources(Path(plan["snapshot"]), plan)
    records = {row["execution_id"]: row for row in plan["executions"]}
    metadata.append({execution: records[execution] for execution in wanted})
  if metadata[0] != metadata[1] or plans[0]["source_hashes"] != plans[1]["source_hashes"]:
    raise ValueError("old/new workload or C++ catalog source changed")
  replay = None
  if args.source_equivalence:
    from dsl_matmul_source_equivalence import tree_sha

    replay = read(args.source_equivalence)
    if (
      not replay.get("all_programs_identical")
      or replay.get("changed_groups")
      or not replay.get("native_evidence")
      or replay["groups"] != len(plans[1]["groups"])
      or Path(replay["native_suite"]).resolve() != directories[1]
      or Path(replay["native_snapshot"]).resolve() != Path(plans[1]["snapshot"]).resolve()
      or Path(replay["frontend_snapshot"]).resolve() != args.root
      or tree_sha(args.root / "python/vecops", {".py"}) != replay["frontend_python_sha256"]
      or tree_sha(args.root / "python/vecops/_sdk") != replay["native_environment"]["sdk_sha256"]
      or sorted(digest(path) for path in (args.root / "python/vecops").glob("_C*.so"))
      != replay["native_environment"]["extension_sha256"]
    ):
      raise ValueError("source equivalence does not admit this frontend/native suite")
    for section in ("input_sha256", "source_inputs", "native_evidence"):
      if any(digest(path) != expected for path, expected in replay[section].items()):
        raise ValueError("source-equivalence evidence changed")
  elif args.root != Path(plans[1]["snapshot"]).resolve():
    raise ValueError("a different frontend snapshot requires an admitted source-equivalence proof")
  indexes, inputs = [], {str(directory / "plan.json"): digest(directory / "plan.json") for directory in directories}
  for directory, plan in zip(directories, plans):
    index, evidence = suite_indexes(directory, plan, wanted)
    indexes.append(index)
    inputs.update(evidence)

  from dsl_snapshot_entry import select_snapshot

  select_snapshot(args.root)
  import dsl_matmul_artifact_ab as ab
  import vecops

  if Path(vecops.__file__).resolve() != args.root / "python/vecops/__init__.py":
    raise RuntimeError("native ABI loader escaped the selected source snapshot")
  args.output.mkdir(parents=True)
  proof = {
    "selection": identity,
    "cpu": args.cpu,
    "rounds": args.rounds,
    "host": platform.uname()._asdict(),
    "input_sha256": inputs,
    "frequency_before": frequency(args.cpu),
    "driver_sha256": digest(__file__),
    "historical_timings_used": False,
    "source_equivalence": {"path": str(args.source_equivalence.resolve()), "sha256": digest(args.source_equivalence)}
    if replay
    else None,
  }
  save(args.output / "admission.json", proof)
  results = []
  for execution in identity["execution_ids"]:
    pair = [index[execution]["artifacts"][execution] for index in indexes]
    for stage in args.stages:
      packing = metadata[1][execution].get("packing", "")
      readiness = ab.stage_availability(
        *pair,
        packing_only=stage == "packing",
        packing_expected=packing.endswith(("packeda", "packedb", "packedab")),
        reference_only=False,
      )
      if readiness["status"] == "not_applicable":
        results.append({"execution_id": execution, "stage": stage, **readiness})
        continue
      if readiness["status"] != "available":
        raise ValueError(f"missing paired artifacts: {execution}/{stage}: {readiness}")
      directory = args.output / execution / stage
      paths = []
      for role, plan, index in zip(("baseline", "candidate"), plans, indexes):
        manifest = directory / f"{role}-manifest.json"
        save(manifest, subset_manifest(plan, [execution]))
        path = directory / f"{role}-index.json"
        save(path, {**index[execution], "manifest": str(manifest), "artifacts": {execution: pair[len(paths)]}})
        paths.append(path)
      before = frequency(args.cpu)
      ab.compare(
        SimpleNamespace(
          baseline_index=paths[0],
          candidate_index=paths[1],
          allow_baseline_subset=False,
          output_dir=directory,
          cpu=args.cpu,
          rounds=args.rounds,
          packing_only=stage == "packing",
        )
      )
      rows = read(directory / "results.json")["rows"]
      if len(rows) != 1:
        raise ValueError("paired comparison did not return exactly one execution")
      results.append(
        {**rows[0], "status": "passed", "frequency_before": before, "frequency_after": frequency(args.cpu)}
      )
      save(args.output / "results.json", {**proof, "completed": False, "rows": results})
  if any(digest(path) != expected for path, expected in inputs.items()):
    raise ValueError("admitted input changed during timing")
  save(
    args.output / "results.json", {**proof, "completed": True, "frequency_after": frequency(args.cpu), "rows": results}
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
