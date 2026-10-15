# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Remeasure exact supplemental old/new native artifacts at the current frequency.

The original checkpoints and binaries remain immutable. Fresh SDK/toolchain
pre/post guards write to the new output directory. Four implementations share
inputs and balanced execution order; no historical timing is read.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from dsl_benchmark_selection import selection_identity
from dsl_matmul_paired_regression import frequency


def read(path):
  return json.loads(Path(path).read_text())


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, document):
  path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def checkpoint_environment(document):
  """Recover recorded external policy without interpreting executable reprs."""
  environments = []
  for policy in document["policies"]:
    key = policy["key"]
    external = ast.literal_eval(key[key.index("(('CC',") : -1])
    environment = dict(external)
    if len(environment) != len(external) or not all(
      isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in external
    ):
      raise ValueError("invalid recorded compiler environment")
    environments.append(environment)
  if not environments or any(value != environments[0] for value in environments):
    raise ValueError("batch policies require different external environments")
  return environments[0]


def guard_batch(root, directory, destination, phase):
  """Check the original frontend/SDK identity in its own fresh process."""
  checkpoint = directory / "batch-input-attestation.json"
  environment = dict(os.environ)
  for name, value in checkpoint_environment(read(checkpoint)).items():
    if value is None:
      environment.pop(name, None)
    else:
      environment[name] = value
  subprocess.run(
    [sys.executable, str(Path(__file__).resolve()), "_guard", str(root),
     str(directory), str(destination), phase],
    env=environment, check=True,
  )
  return read(destination)


def guard_main(arguments):
  root, directory, destination = (Path(p).resolve() for p in arguments[:3])
  phase = arguments[3]
  sys.path.insert(0, str(root / "benchmarks"))
  from dsl_snapshot_entry import select_snapshot

  select_snapshot(root)
  from dsl_matmul_repair_suite import BatchInputAttestation
  from dsl_matmul_source_equivalence import tree_sha

  checkpoint = directory / "batch-input-attestation.json"
  before = sha(checkpoint)
  guard = BatchInputAttestation.restore(directory)
  # Full original guard checks remain active; only its output goes elsewhere.
  guard.checkpoint = lambda: None
  guard.guard(phase)
  if sha(checkpoint) != before:
    raise ValueError("original checkpoint changed during read-only guard")
  save(destination, {"root": str(root), "directory": str(directory), "phase": phase,
                     "checkpoint_sha256": before, "events": guard.events,
                     "frontend_python_sha256": tree_sha(root / "python/vecops", {".py"})})


def measure(calls, rounds):
  repeats = {}
  for label, call in calls.items():
    repeat = 1
    while True:
      start = time.perf_counter_ns()
      call(repeat)
      elapsed = time.perf_counter_ns() - start
      if elapsed >= 25_000_000 or repeat >= 2**30:
        break
      repeat *= max(2, min(16, 25_000_000 // max(1, elapsed)))
    repeats[label] = repeat
  labels = tuple(calls)
  # Every implementation occupies every ordinal position in each four rounds.
  samples, raw = {label: [] for label in labels}, []
  for round_ in range(rounds):
    order = labels[round_ % 4:] + labels[:round_ % 4]
    if round_ // 4 % 2:
      order = tuple(reversed(order))
    for label in order:
      start = time.perf_counter_ns()
      calls[label](repeats[label])
      elapsed = time.perf_counter_ns() - start
      samples[label].append(elapsed / repeats[label])
      raw.append({"round": round_, "label": label, "repeat": repeats[label], "elapsed_ns": elapsed})
  return {"median_ns": {label: statistics.median(values) for label, values in samples.items()}, "samples": raw}


def source_admission(proof_path, root, candidate_root, candidate_suite):
  """Require a replay proof only when executing a different frontend snapshot."""
  if proof_path is None:
    if root.resolve() != candidate_root.resolve():
      raise ValueError("a different frontend snapshot requires an admitted source-equivalence proof")
    return None, {}
  proof = read(proof_path)
  if (not proof.get("all_programs_identical") or proof.get("changed_groups")
      or Path(proof["frontend_snapshot"]).resolve() != root.resolve()
      or Path(proof["old_native_suite"]).resolve() != candidate_suite.resolve()):
    raise ValueError("source replay proof does not admit this frontend/native suite")
  inputs = {str(proof_path.resolve()): sha(proof_path), **proof["input_sha256"]}
  if any(sha(path) != expected for path, expected in inputs.items()):
    raise ValueError("source replay evidence changed")
  return proof, inputs


def verify_reference_root(source, original_root):
  """Bind a retained suite to its original frozen declaration snapshot."""
  if not Path(source).resolve().is_relative_to(original_root.resolve()):
    raise ValueError("native reference does not belong to its original frontend snapshot")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("root", "baseline-suite", "candidate-suite", "selection", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
  parser.add_argument("--source-equivalence", type=Path,
                      help="Admitted replay proof when root differs from the native-build candidate root")
  parser.add_argument("--baseline-root", type=Path, required=True, help="Original native-build frontend snapshot")
  parser.add_argument("--candidate-root", type=Path, required=True, help="Original native-build frontend snapshot")
  parser.add_argument("--architecture", choices=("x86", "arm"), required=True)
  parser.add_argument("--cpu", type=int, required=True)
  parser.add_argument("--rounds", type=int, default=12)
  args = parser.parse_args()
  if args.rounds <= 0 or args.rounds % 4 or args.cpu not in os.sched_getaffinity(0):
    parser.error("rounds must be a positive multiple of four and CPU must be available")
  if args.output.exists():
    parser.error("retain independent evidence in a new output directory")
  args.output = args.output.resolve()
  args.root = args.root.resolve()
  wanted = selection_identity(args.selection, args.architecture, "registration_ids")
  proof, inputs = source_admission(
    args.source_equivalence, args.root, args.candidate_root, args.candidate_suite
  )
  sys.path.insert(0, str(args.root / "benchmarks"))
  from dsl_snapshot_entry import select_snapshot

  select_snapshot(args.root)
  import numpy as np
  from dsl_matmul_source_equivalence import tree_sha
  from dsl_matmul_supplemental_artifacts import artifact_inventory, bind_artifact
  from dsl_matmul_supplemental_lifecycle import lifecycle_arguments, scenario_for, templates
  from dsl_matmul_supplemental_pack import packing_arguments
  from dsl_matmul_supplemental_pack import templates as packing_templates
  from dsl_matmul_supplemental_validate import batch_lock
  from dsl_timing import reserve_cpu

  def frontend_identity():
    return {"frontend_python_sha256": tree_sha(args.root / "python/vecops", {".py"}),
            "sdk_sha256": tree_sha(args.root / "python/vecops/_sdk"),
            "extension_sha256": sorted(sha(path) for path in (args.root / "python/vecops").glob("_C*.so"))}

  frontend_fingerprint = frontend_identity()
  if proof and any(frontend_fingerprint[key] != proof[key] for key in frontend_fingerprint):
    raise ValueError("frontend or native SDK changed after source emission")
  suites = (args.baseline_suite.resolve(), args.candidate_suite.resolve())
  plans = [read(suite / "plan.json") for suite in suites]
  if plans[0]["suite"] != plans[1]["suite"]:
    raise ValueError("supplemental suites differ")
  packing = plans[0]["suite"] == "packing"
  selected = set(wanted["registration_ids"])
  metadata = [{row["registration_id"]: row for group in plan["groups"] for row in group["rows"]
               if row["registration_id"] in selected} for plan in plans]
  if metadata[0] != metadata[1] or set(metadata[0]) != selected:
    raise ValueError("old/new registration scope or metadata changed")
  args.output.mkdir(parents=True)
  guards, indexes = [], []
  with ExitStack() as stack:
    stack.enter_context(reserve_cpu(args.cpu))
    os.sched_setaffinity(0, {args.cpu})
    for variant, suite, original_root in zip(
      ("baseline", "candidate"), suites, (args.baseline_root, args.candidate_root)
    ):
      index = {}
      inputs[str(suite / "plan.json")] = sha(suite / "plan.json")
      for directory in sorted((suite / "native").glob("batch-*")):
        manifest_path = directory / "manifest.json"
        manifest = read(manifest_path)
        rows = [row for row in manifest["artifacts"] if row["registration_id"] in selected]
        if not rows:
          continue
        for row in rows:
          verify_reference_root(row["cpp_source"], original_root)
        stack.enter_context(batch_lock(directory))
        status = read(directory / "build-status.json")
        checkpoint = Path(status["attestation_checkpoint"])
        if not status["admitted"] or sha(checkpoint) != status["attestation_checkpoint_sha256"]:
          raise ValueError("native checkpoint changed or was not admitted")
        inputs[str(checkpoint)] = sha(checkpoint)
        inputs[str(manifest_path)] = sha(manifest_path)
        guard_file = args.output / f"guard-{variant}-{directory.name}.json"
        guarded = guard_batch(original_root, directory, guard_file, "supplemental_paired_pre")
        if variant == "candidate" and not proof and guarded["frontend_python_sha256"] != frontend_fingerprint["frontend_python_sha256"]:
          raise ValueError("native guard used a different candidate frontend")
        guards.append((original_root, directory, guard_file.with_name(guard_file.stem + "-post.json")))
        inputs[str(guard_file)] = sha(guard_file)
        records = artifact_inventory(directory)
        oracle_path = suite / f"validation-{directory.name}.json"
        oracle = read(oracle_path)
        inputs[str(oracle_path)] = sha(oracle_path)
        validated = {row["registration"]["registration_id"]: row for row in oracle["rows"] if row.get("validated")}
        for row in rows:
          identity = row["registration_id"]
          metadata_row = metadata[0][identity]
          pair = packing_templates(metadata_row) if packing else templates(metadata_row)[1]
          bound = [bind_artifact(template, row[label + "_source"], records)
                   for label, template in zip(("dsl", "cpp"), pair)]
          for label, (_, evidence) in zip(("dsl", "cpp"), bound):
            expected = validated[identity]["artifacts"][label]
            if (evidence["binary"], evidence["binary_sha256"]) != (expected["binary"], expected["binary_sha256"]):
              raise ValueError("native binary differs from the independent oracle")
            inputs[evidence["binary"]] = evidence["binary_sha256"]
          if identity in index:
            raise ValueError("duplicate native registration")
          index[identity] = bound
      if set(index) != selected:
        raise ValueError("missing selected native registrations")
      indexes.append(index)
    admission = {"inputs": inputs, "selection": wanted, "historical_timings_used": False,
                 "frequency_before": frequency(args.cpu), "cpu": args.cpu, "rounds": args.rounds,
                 "driver_sha256": sha(__file__), "frontend_snapshot": str(args.root),
                 "frontend_identity": frontend_fingerprint,
                 "source_equivalence": str(args.source_equivalence.resolve()) if proof else None}
    save(args.output / "admission.json", admission)
    results = []
    for identity in wanted["registration_ids"]:
      row = metadata[0][identity]
      if packing:
        values, expected, correction = packing_arguments(row)
      else:
        values, expected = lifecycle_arguments(scenario_for(row))
        correction = None
      if row["suite"] == "StatefulMatmul":
        values[0].fill(1)
        values[1].fill(1)
        expected.fill(row["shape"][2])
      calls, artifacts = {}, {}
      output = values[1] if packing else values[2]
      tolerance = 0.02 if output.dtype.itemsize <= 2 else (8e-4 if output.dtype.itemsize == 4 else 2e-10)
      for variant, index in zip(("baseline", "candidate"), indexes):
        for label, (invoke, evidence) in zip(("dsl", "cpp"), index[identity]):
          name = variant + "_" + label
          calls[name] = lambda repeat, invoke=invoke, values=values: invoke(values, repeat)
          artifacts[name] = evidence
          for poison in (0xA5, 0x5A):
            output.view(np.uint8).fill(poison)
            if correction is not None:
              values[2].view(np.uint8).fill(poison)
            calls[name](1)
            if packing:
              np.testing.assert_array_equal(output.view(np.uint8), expected.view(np.uint8))
              if correction is not None:
                np.testing.assert_array_equal(values[2], correction)
            elif output.dtype.kind in "iu":
              np.testing.assert_array_equal(output, expected)
            else:
              np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance)
      result = {"registration": row, "artifacts": artifacts, "validated": True, **measure(calls, args.rounds)}
      results.append(result)
      save(args.output / (identity + ".json"), result)
      print(identity, result["median_ns"], flush=True)
    for root, directory, guard_file in guards:
      guard_batch(root, directory, guard_file, "supplemental_paired_post")
    if any(sha(path) != expected for path, expected in inputs.items()):
      raise ValueError("admitted inputs changed during measurement")
    if frontend_identity() != frontend_fingerprint:
      raise ValueError("frontend or native SDK changed during measurement")
    if sha(__file__) != admission["driver_sha256"]:
      raise ValueError("timing driver changed during measurement")
    save(args.output / "results.json", {"all_passed": len(results) == len(selected), "rows": results,
                                       "frequency_after": frequency(args.cpu),
                                       "post_guards_sha256": {str(path): sha(path) for _, _, path in guards},
                                       "admission_sha256": sha(args.output / "admission.json")})


if __name__ == "__main__":
  if sys.argv[1:2] == ["_guard"]:
    guard_main(sys.argv[2:])
  else:
    main()
