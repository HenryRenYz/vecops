"""Refresh all 30 projection metas and A/B only DSL online packing.

The reference is the unchanged source-catalog public C++ Matmul, including
its original family, tile policy and offline packed inputs. The second DSL
variant changes only library.matmul's packing request to ``AB``. Its extra
packing stays inside the native repeat loop. A packed-AB source observation
is an intentional no-repack control, not an online packing measurement.

This entry must run in a fresh process: capture the rank's affinity and turn
off OpenMP placement before importing torch/vecops. No environment, SDK or
affinity outside this process is changed. --list is build-free; execution
requires an immutable snapshot and retained manifests, and may run one bounded
chunk per process. No missing source metadata is silently filtered away.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

# torch/OpenMP may pin its importing master thread to one CPU. Capture this
# before any heavy import, not in main() after importing the source catalog.
ENTRY_AFFINITY = frozenset(os.sched_getaffinity(0))
ENTRY_ENVIRONMENT = {
  name: os.environ.get(name)
  for name in (
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_NUM_THREADS",
    "CC",
    "CXX",
    "CXXFLAGS",
    "LDFLAGS",
    "VECOPS_BUILD_CPU_LIST",
    "LX_WORLD_RANK",
    "LX_LOCAL_RANK",
    "CCS_JOB_ID",
  )
}
SHAPES = ((1, 1152, 896), (128, 1024, 768))
SOURCE_MODES = {
  SHAPES[0]: frozenset(("raw", "packedb", "packedab")),
  SHAPES[1]: frozenset(("raw", "packedab")),
}
VARIANTS = ("auto", "online_ab")
CACHE_FIELDS = ("line_bytes", "l1d_bytes", "l2_bytes", "l3_bytes", "llc_bytes", "llc_level", "l3_present", "fallback")
ROOT = Path(__file__).resolve().parents[1]


def projection_scenarios(svl_bytes):
  from dsl_matmul_source_catalog import load_cases, scenarios

  selected = tuple(
    scenario
    for scenario in scenarios(load_cases(svl_bytes=svl_bytes))
    if scenario.case.catalog == "MatmulPolicyExpanded"
    and (scenario.case.m, scenario.case.n, scenario.case.k) in SOURCE_MODES
    and scenario.packing in SOURCE_MODES[scenario.case.m, scenario.case.n, scenario.case.k]
  )
  expected = Counter(
    (shape, packing, mode, policy)
    for shape, packing_modes in SOURCE_MODES.items()
    for packing in packing_modes
    for mode in ("Const", "Dynamic")
    for policy in ("ExactCover", "FourRegions", "Automatic")
  )
  observed = Counter(((s.case.m, s.case.n, s.case.k), s.packing, s.mode, s.reference[1]) for s in selected)
  if observed != expected or len(selected) != 30 or len({s.execution_id for s in selected}) != 30:
    raise ValueError(f"projection source catalog changed: expected exact 30 metas; observed {dict(observed)}")
  for scenario in selected:
    scenario.validate()
    if (
      scenario.batched
      or scenario.has_prior
      or scenario.reference[0] != "Automatic"
      or scenario.types != ("bfloat16", "bfloat16", "float32", "float32", "bfloat16", "bfloat16", "float32")
      or scenario.orientation != (False, True, False)
      or not scenario.swap
      or any(
        pipeline != "Convert"
        for pipeline in (scenario.case.input_pipeline_a, scenario.case.input_pipeline_b, scenario.case.output_pipeline)
      )
    ):
      raise ValueError(f"projection runner does not define changed semantics: {scenario.key}")
  return selected


def online_ab_template(scenario, original):
  """Reuse exact signature/layout descriptors; alter only the DSL pack request."""
  import vecops.language as T
  from dsl_benchmark_utils import packing_geometry

  annotations = original.ir.annotations
  AT, BT, CT, DT, ST, SCT, SQT, RQT, RDT, WT, ZT, APT, BPT = (
    annotations[name] for name in ("A", "B", "C", "D", "S", "SC", "SQ", "RQ", "RD", "W", "Z", "AP", "BP")
  )
  na, nb, acc = scenario.types[4:]
  pa, pb, swap = scenario.pack_a, scenario.pack_b, T.Const(scenario.swap)
  format_ = packing_geometry(na)[0]

  @T.template
  def dsl_online_ab(
    A: AT, B: BT, C: CT, D: DT, S: ST, SC: SCT, SQ: SQT, RQ: RQT, RD: RDT, W: WT, Z: ZT, AP: APT, BP: BPT, repeat: int
  ):
    for _iteration in T.serial(repeat, unroll=1):
      a = T.packed(AP, (A.shape[0], A.shape[1]), side="A", format=format_) if pa else T.load(A, dtype=na)
      b = T.packed(BP, (B.shape[0], B.shape[1]), side="B", format=format_) if pb else T.load(B, dtype=nb)
      result = T.library.matmul((a, b), C, acc_dtype=acc, swap=swap, packing="AB")
      T.store(C, result)
      T.compiler_fence()

  return dsl_online_ab


def modeled_blocks(cache, logical_k, svl):
  """Record the current SME block formula, explicitly NOT an observed branch."""
  mr = nr = svl // 2  # BF16 primary, two F32 ZA tiles per panel.
  kr, a_bytes, b_bytes, c_bytes = 2, 2, 2, 4

  def align(value, extent):
    return max(extent, value // extent * extent)

  kc = align(max(cache["l1d_bytes"] // 2, 1) // (mr * a_bytes + nr * b_bytes), kr)
  if logical_k * (mr * a_bytes + nr * b_bytes) <= cache["l2_bytes"] // 2:
    kc = align((logical_k + kr - 1) // kr * kr, kr)
  budget = max(cache["l2_bytes"] * 3 // 4, cache["line_bytes"])
  mc = align(budget // 3 // (kc * a_bytes), mr)
  nc = align((budget - budget // 3) // (kc * b_bytes), nr)
  outer = cache["llc_bytes"] or (cache["l3_bytes"] if cache["l3_present"] else 0)
  if outer:
    output_budget = max(mr * nr * c_bytes, outer // 4)
    while mc * nc * c_bytes > output_budget and (mc > mr or nc > nr):
      if nc >= mc and nc > nr:
        nc = align(nc // 2, nr)
      else:
        mc = align(mc // 2, mr)
  return {
    "status": "model_only_not_executed_branch",
    "assumed_geometry": {"mr": mr, "nr": nr, "kr": kr},
    "automatic_block_m": mc,
    "automatic_block_n": nc,
    "automatic_block_k": kc,
  }


def kernel_evidence(kernel):
  """Bounded planning information plus actual commands, not huge IR dumps."""
  plans = getattr(kernel.backend_ir, "matmuls", {})
  result = []
  for plan in plans.values():
    result.append(
      {
        "options": asdict(plan.options),
        "vector_rows": plan.vector_rows,
        "vector_columns": plan.vector_columns,
        "k_unroll": plan.k_unroll,
        "unblocked_bytes": plan.unblocked_bytes,
        "runtime_unblocked_bytes": plan.runtime_unblocked_bytes,
        "pass_trace": plan.pass_trace,
        "schedule": asdict(plan.schedule) if plan.schedule is not None else None,
        "operand_fusion": [
          {
            "materialized": operand.materialization is not None,
            "conditional": operand.condition is not None,
            "operation_cost": operand.operation_cost,
            "reason": operand.reason,
          }
          for operand in plan.operands
        ],
      }
    )
  commands = [artifact["command"] for artifact in kernel.debug.artifacts()]
  if not commands:
    raise RuntimeError("cold experiment needs retained actual compile commands")
  source = kernel.generated_source
  compiler = kernel.debug.compiler
  native = compiler._native_config()
  return {
    "cache": kernel.debug.get_cache_info(),
    "actual_compile_commands": commands,
    "target_arch": native.target_arch,
    "compile_options": list(native.compile_options),
    "compile_definitions": list(native.compile_definitions),
    "environment": dict(compiler.environment),
    "generated_source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
    "generated_source_bytes": len(source.encode()) if source else None,
    "plans": result,
    "executed_family_and_runtime_blocks": "not_observable_without_instrumenting_timed_kernel",
  }


def probe_template():
  import vecops.language as T

  @T.template(source=ROOT / "benchmarks/dsl_matmul_projection_cache.cpp")
  def probe(Information: T.Tensor((8,), dtype="int64", access="output")):
    pass

  return probe


def verify_documents(paths, identity, expected_ids):
  """Reject missing/duplicate/stale observations; an empty partition is not pass."""
  if not expected_ids:
    raise ValueError("cannot verify zero observations")
  results, duplicates, stale = {}, [], []
  for path in paths:
    document = json.loads(Path(path).read_text())
    if any(
      document.get(key) != identity.get(key)
      for key in (
        "source_label",
        "source_archive_sha256",
        "source_hashes",
        "runner_hashes",
        "runner_sha256",
        "pass_options",
        "implementation_identity",
      )
    ):
      stale.append(str(path))
    permitted = set(document["selected_observation_ids"])
    for key, value in document["results"].items():
      if key not in permitted or value.get("observation_id") != key:
        raise ValueError(f"result does not belong to the selected partition: {path}: {key}")
      if key in results:
        duplicates.append(key)
      results[key] = value
  missing = sorted(set(expected_ids) - results.keys())
  unexpected = sorted(results.keys() - set(expected_ids))
  failed = sorted(key for key in results.keys() & set(expected_ids) if results[key].get("status") != "passed")
  return {
    "complete": not any((missing, unexpected, failed, duplicates, stale)),
    "expected": len(expected_ids),
    "executed": len(results),
    "missing": missing,
    "unexpected": unexpected,
    "failed": failed,
    "duplicates": duplicates,
    "stale": stale,
  }


def main(argv=None):
  global ROOT
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--snapshot-root", type=Path)
  parser.add_argument("--sdk", type=Path, help="Copied SDK with matching frozen header overlay; never installs it")
  parser.add_argument("--source-label", default="source18")
  parser.add_argument(
    "--source-archive-sha256", required=True, help="Parent-attested immutable source archive identity"
  )
  parser.add_argument(
    "--svl-bytes", type=int, help="Build-free audit only; execution queries actual current VL and SVL"
  )
  parser.add_argument("--list", action="store_true")
  parser.add_argument("--verify", nargs="+", type=Path)
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--build-dir", type=Path)
  parser.add_argument("--chunk-size", type=int, default=6)
  parser.add_argument("--chunk-index", type=int)
  parser.add_argument("--jobs", type=int, default=8)
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--cpu", type=int)
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--target", default=os.environ.get("VECOPS_TEST_TARGET", "Native"))
  args = parser.parse_args(argv)
  if args.chunk_size < 1 or args.jobs < 1 or args.rounds < 1:
    raise ValueError("chunk size, jobs and rounds must be positive")
  if len(args.source_archive_sha256) != 64 or any(
    char not in "0123456789abcdef" for char in args.source_archive_sha256
  ):
    raise ValueError("source archive SHA256 must be 64 lowercase hex digits")
  if args.snapshot_root:
    from dsl_snapshot_entry import select_snapshot

    ROOT = select_snapshot(args.snapshot_root)
  elif not args.list and not args.verify:
    raise ValueError("execution requires --snapshot-root; do not benchmark a mutable editable source tree")
  sys.path.insert(0, str(ROOT / "benchmarks"))
  if args.build_dir is None:
    args.build_dir = ROOT / "build-dsl-matmul-projection"
  if args.svl_bytes is not None and not (16 <= args.svl_bytes <= 256 and args.svl_bytes % 16 == 0):
    raise ValueError("SVL bytes must be a legal 16-byte multiple between 16 and 256")

  # Child-only build placement is rank-local, never inferred from /sys online
  # CPUs and never broadened beyond the wrapper's existing taskset allocation.
  os.environ["OMP_PROC_BIND"] = "false"
  cpu = min(ENTRY_AFFINITY) if args.cpu is None else args.cpu
  if cpu not in ENTRY_AFFINITY:
    raise ValueError("timing CPU is outside the Python entry rank allocation")
  build_cpus = ENTRY_AFFINITY - {cpu}
  if not build_cpus:
    build_cpus = ENTRY_AFFINITY
  os.environ["VECOPS_BUILD_CPU_LIST"] = ",".join(map(str, sorted(build_cpus)))

  import numpy as np
  import torch
  import vecops
  import vecops.language as T
  from dsl_matmul_source_catalog import (
    audit_document,
    result_metadata,
    source_arguments,
    source_metadata,
    source_templates,
  )
  from dsl_timing import reserve_cpu
  from dsl_transpose import measure
  from vecops._arm_vector_state import get_vector_state
  from vecops._identity import implementation_identity

  if args.sdk:
    import vecops._compiler as compiler_module

    sdk = args.sdk.resolve()
    if not (sdk / "lib/cmake/Vecops/VecopsConfig.cmake").is_file():
      raise ValueError("--sdk needs a complete copied package SDK")
    compiler_module._sdk = lambda: (sdk, vecops._C.SdkLayout.package)
  state = get_vector_state()
  svl = args.svl_bytes if args.list or args.verify else state.sme_bytes
  if svl is None:
    raise RuntimeError("actual SME streaming vector length is required; use --svl-bytes for an offline audit")
  if args.svl_bytes is not None and not (args.list or args.verify) and args.svl_bytes != state.sme_bytes:
    raise ValueError("requested audit SVL differs from the executing thread")
  selected = projection_scenarios(svl)
  chunks = [selected[start : start + args.chunk_size] for start in range(0, len(selected), args.chunk_size)]
  if args.chunk_index is not None and not 0 <= args.chunk_index < len(chunks):
    raise ValueError(f"chunk index must lie in [0,{len(chunks)}); zero observations is not success")
  chosen = selected if args.chunk_index is None else chunks[args.chunk_index]
  options = {"cpu_matmul_fusion": "auto", "cpu_matmul_pipeline": "auto"}
  original_audit = audit_document(tuple(s.case for s in selected), selected)
  audit = {
    "experiment": "projection_original_reference_vs_DSL_auto_and_online_AB",
    "source_label": args.source_label,
    "source_archive_sha256": args.source_archive_sha256,
    "implementation_identity": implementation_identity(),
    "source_hashes": original_audit["source_hashes"],
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "runner_hashes": {
      **original_audit["runner_hashes"],
      "cache_probe": hashlib.sha256((ROOT / "benchmarks/dsl_matmul_projection_cache.cpp").read_bytes()).hexdigest(),
    },
    "pass_options": options,
    "expected_source_metas": 30,
    "expected_observations": 60,
    "variants": VARIANTS,
    "source_metas": [dict(result_metadata(s), declaration=s.case.declaration) for s in selected],
    "selected_observation_ids": [f"{s.execution_id}:{v}" for s in chosen for v in VARIANTS],
    "requested_observation_ids": [f"{s.execution_id}:{v}" for s in selected for v in VARIANTS],
    "chunk_index": args.chunk_index,
    "chunks": len(chunks),
    "results": {},
    "executed": False,
    "completed": 0,
    "failures": 0,
  }

  def save():
    if args.manifest:
      args.manifest.parent.mkdir(parents=True, exist_ok=True)
      args.manifest.write_text(json.dumps(audit, indent=2) + "\n")

  if args.verify:
    verified = verify_documents(args.verify, audit, audit["requested_observation_ids"])
    print(json.dumps(verified, indent=2))
    return 0 if verified["complete"] else 1
  save()
  if args.list:
    print(json.dumps(audit, indent=2))
    return 0
  if platform.machine().lower() not in {"aarch64", "arm64"} or state.sve_bytes is None:
    raise RuntimeError("paired projection experiment requires actual ARM SVE and SME")
  if not args.output or not args.manifest:
    raise ValueError("execution needs --output and --manifest to retain all 60 observations")
  if Path(vecops.__file__).resolve() != (ROOT / "python/vecops/__init__.py").resolve():
    raise RuntimeError("vecops import escaped the frozen snapshot")
  os.sched_setaffinity(0, build_cpus)
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  audit["environment"] = {
    "cpu": cpu,
    "hostname": platform.node(),
    "machine": platform.machine(),
    "entry_affinity": sorted(ENTRY_AFFINITY),
    "build_affinity": sorted(build_cpus),
    "entry_environment": ENTRY_ENVIRONMENT,
    "effective_environment": {name: os.environ.get(name) for name in ENTRY_ENVIRONMENT},
    "vector_state": asdict(state),
    "implementation_identity": implementation_identity(),
    "python_package": vecops.__file__,
    "native_extension": vecops._C.__file__,
    "cpuinfo": Path("/proc/cpuinfo").read_text(),
  }
  probe = T.compile(probe_template(), compiler=compiler)
  information = np.empty(8, dtype="int64")
  with vecops.collect_compile_requests() as collector:
    probe.operator(vecops.TensorMeta((8,), dtype=vecops.int64), **probe._core.specialization_values)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  with reserve_cpu(cpu):
    os.sched_setaffinity(0, {cpu})
    try:
      probe.operator.native(information, **probe._core.specialization_values)
    finally:
      os.sched_setaffinity(0, build_cpus)
  cache = dict(zip(CACHE_FIELDS, map(int, information), strict=True))
  audit["environment"]["runtime_public_cache_info"] = cache
  audit["environment"]["cache_probe_compilation"] = kernel_evidence(probe)
  audit["executed"] = True
  save()
  fields = (
    *result_metadata(selected[0]),
    "observation_id",
    "variant",
    "extra_online_packing",
    "dsl_us",
    "cpp_us",
    "dsl_over_cpp",
    "status",
    "error",
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for start in range(0, len(chosen), args.chunk_size):
      chunk, compiled = chosen[start : start + args.chunk_size], {}
      try:
        with vecops.collect_compile_requests() as collector:
          for scenario in chunk:
            pair, packing = source_templates(scenario)
            templates = (pair[0], online_ab_template(scenario, pair[0]), pair[1])
            kernels = tuple(
              T.compile(t, compiler=compiler, pass_options=options if i < 2 else None) for i, t in enumerate(templates)
            )
            packers = (
              tuple(
                T.compile(t, compiler=compiler, pass_options=options if i == 0 else None) for i, t in enumerate(packing)
              )
              if packing
              else ()
            )
            for kernel in (*kernels, *packers):
              kernel.operator(*source_metadata(scenario), 1, **kernel._core.specialization_values)
            compiled[scenario.execution_id] = kernels, packers
        vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
      except Exception as error:  # noqa: BLE001 -- ledger retains every failed meta
        for scenario in chunk:
          for variant in VARIANTS:
            key = f"{scenario.execution_id}:{variant}"
            row = dict(
              result_metadata(scenario),
              observation_id=key,
              variant=variant,
              extra_online_packing="unexecuted",
              status="compile_failed",
              error=str(error),
            )
            writer.writerow(row)
            audit["results"][key] = row
        audit.update(
          completed=len(audit["results"]), failures=sum(r["status"] != "passed" for r in audit["results"].values())
        )
        stream.flush()
        save()
        continue
      with reserve_cpu(cpu):
        os.sched_setaffinity(0, {cpu})
        try:
          for scenario in chunk:
            kernels, packers = compiled[scenario.execution_id]
            evidence = {}
            try:
              evidence = {
                "compilations": [kernel_evidence(k) for k in kernels],
                "packing_compilations": [kernel_evidence(k) for k in packers],
                "block_model": modeled_blocks(cache, scenario.case.k, svl),
              }
              values, expected = source_arguments(scenario)
              if packers:
                snapshots = []
                for packer in packers:
                  packer.operator.native(*values, 1, **packer._core.specialization_values)
                  snapshots.append((values[11].copy(), values[12].copy()))
                for index, active in enumerate((scenario.pack_a, scenario.pack_b)):
                  if active:
                    np.testing.assert_array_equal(snapshots[0][index], snapshots[1][index])
              c = values[2]
              calls = tuple(
                lambda kernel=k, values=values: kernel.operator.native(*values, 1, **kernel._core.specialization_values)
                for k in kernels
              )
              calls[2]()
              reference = c.copy()
              np.testing.assert_allclose(reference, expected, rtol=8e-4, atol=8e-4)
              for call in calls[:2]:
                call()
                np.testing.assert_allclose(c, reference, rtol=8e-4, atol=8e-4)
              repeat = max(2, min(4000, 20_000_000 // (scenario.case.m * scenario.case.n * scenario.case.k)))
              timed = tuple(
                lambda kernel=k, values=values, repeat=repeat: kernel.operator.native(
                  *values, repeat, **kernel._core.specialization_values
                )
                for k in kernels
              )
              t_auto, t_ab, t_cpp = measure(timed, repeat, args.rounds)
              evidence.update(
                repeat=repeat,
                rounds=args.rounds,
                timing_affinity=sorted(os.sched_getaffinity(0)),
                vector_state_after=asdict(get_vector_state()),
              )
              status, error, times = "passed", "", (t_auto, t_ab)
            except Exception as error:  # noqa: BLE001 -- no source case is discarded
              status, error, times, t_cpp = "validation_failed", str(error), (None, None), None
            for variant, timing in zip(VARIANTS, times, strict=True):
              key = f"{scenario.execution_id}:{variant}"
              added = (
                ("none_already_packed_control" if scenario.packing == "packedab" else "A" if scenario.pack_b else "AB")
                if variant == "online_ab"
                else "original_auto"
              )
              row = dict(
                result_metadata(scenario),
                observation_id=key,
                variant=variant,
                extra_online_packing=added,
                dsl_us=timing,
                cpp_us=t_cpp,
                dsl_over_cpp=timing / t_cpp if timing is not None else None,
                status=status,
                error=error,
              )
              writer.writerow(row)
              audit["results"][key] = {**row, "evidence": evidence}
              print(f"{key} {scenario.key}: {status}", flush=True)
            stream.flush()
            audit.update(
              completed=len(audit["results"]), failures=sum(r["status"] != "passed" for r in audit["results"].values())
            )
            save()
        finally:
          os.sched_setaffinity(0, build_cpus)
  audit["partition_complete"] = (
    set(audit["results"]) == set(audit["selected_observation_ids"]) and not audit["failures"]
  )
  audit["full_experiment_complete"] = audit["partition_complete"] and audit["completed"] == 60
  save()
  return 0 if audit["partition_complete"] else 1


if __name__ == "__main__":
  sys.exit(main())
