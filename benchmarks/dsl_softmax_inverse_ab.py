# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Frozen-snapshot strict softmax versus unchanged C++ with explicit schedules.

This external experiment calls only the public T.library.softmax interface,
so it can select an older snapshot without requiring the new inverse helper.
Both DSL row and direct-rank-two scopes include the whole operator inside the
native repeat loop. C++ receives the same dense input/output dtype and shape.
Use a fresh namespace/build directory per snapshot; compilation is real.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
  "source_archive_sha256",
  "runner_sha256",
  "cpp_source_sha256",
  "namespace",
  "scope",
  "shape",
  "dtype",
  "meta",
  "accuracy",
  "unroll",
  "dsl_us",
  "cpp_us",
  "dsl_over_cpp",
)
DEFAULT_SHAPES = ("8x128", "1x64", "16x1000", "1", "31", "33", "129")


def shape(text):
  if not re.fullmatch(r"[1-9][0-9]*(?:x[1-9][0-9]*)?", text):
    raise argparse.ArgumentTypeError("shape must have one or two positive dimensions, for example 31 or 8x128")
  return tuple(map(int, text.split("x")))


def sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection(shapes, dtypes, metas, scopes, unrolls):
  return tuple(
    (scope, dimensions, dtype, meta, unroll)
    for dimensions in shapes
    for dtype in dtypes
    for meta in metas
    for scope in scopes
    if scope == "row-rank1" or len(dimensions) == 2
    for unroll in unrolls
  )


def snapshot_modules(root):
  expected = root / "python/vecops/__init__.py"
  if "vecops" in sys.modules:
    if Path(sys.modules["vecops"].__file__).resolve() != expected:
      raise RuntimeError("softmax A/B must use a fresh process; vecops belongs to another source snapshot")
  else:
    from dsl_snapshot_entry import select_snapshot

    select_snapshot(root)
  # Benchmark factories and the C++ reference must belong to the selected
  # snapshot too, rather than accidentally resolving from this external runner.
  sys.path.insert(0, str(root / "benchmarks"))
  import dsl_arm_validate as validation
  import dsl_norm_catalog as catalog
  import dsl_norm_library as library
  import vecops

  for module in (validation, catalog, library):
    if not Path(module.__file__).resolve().is_relative_to(root / "benchmarks"):
      raise RuntimeError(f"benchmark factory escaped the selected snapshot: {module.__file__}")
  return vecops, validation, catalog, library


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--snapshot-root", type=Path, default=ROOT)
  parser.add_argument("--source-archive-sha256")
  parser.add_argument("--source-archive", type=Path, help="optionally verify the supplied archive hash")
  parser.add_argument("--namespace")
  parser.add_argument("--shape", type=shape, nargs="+", default=tuple(map(shape, DEFAULT_SHAPES)))
  parser.add_argument("--dtype", nargs="+", choices=("float32", "float64"), default=("float32", "float64"))
  parser.add_argument("--meta", nargs="+", choices=("Const", "Any"), default=("Const", "Any"))
  parser.add_argument(
    "--scope", nargs="+", choices=("row-rank1", "direct-rank2"), default=("row-rank1", "direct-rank2")
  )
  parser.add_argument("--unroll", type=int, nargs="+", choices=(1, 2), default=(1, 2))
  parser.add_argument("--list", action="store_true")
  parser.add_argument("--sdk", choices=("source", "installed"), default="source")
  parser.add_argument("--reuse-installed-sdk", action="store_true")
  parser.add_argument("--build-dir", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--target", default=os.environ.get("VECOPS_TEST_TARGET", "Native"))
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--rounds", type=int, default=9)
  parser.add_argument("--cpu", type=int)
  args = parser.parse_args(argv)
  selected = selection(args.shape, args.dtype, args.meta, args.scope, args.unroll)
  if not selected or len(set(selected)) != len(selected):
    parser.error("selection must contain at least one case and no duplicate schedule metadata")
  if args.list:
    writer = csv.writer(sys.stdout)
    writer.writerow(("scope", "shape", "dtype", "meta", "accuracy", "unroll"))
    writer.writerows(
      (scope, "x".join(map(str, dimensions)), dtype, meta, "strict", unroll)
      for scope, dimensions, dtype, meta, unroll in selected
    )
    return
  if not args.source_archive_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.source_archive_sha256):
    parser.error("--source-archive-sha256 requires the explicit frozen archive SHA256")
  if args.source_archive and sha256(args.source_archive) != args.source_archive_sha256:
    parser.error("source archive SHA256 mismatch")
  if not args.namespace or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.namespace):
    parser.error("--namespace requires a distinct simple label for this snapshot experiment")
  if args.build_dir is None or args.output is None:
    parser.error("--build-dir and --output are required for a preserved experiment")
  if args.jobs < 1 or args.rounds < 1:
    parser.error("--jobs and --rounds must be positive")
  root = args.snapshot_root.resolve(strict=True)
  build = args.build_dir.resolve() / args.namespace
  if build.exists() and any(build.iterdir()):
    parser.error("namespace build directory is not empty; do not reuse a warm-cache experiment")
  if args.output.exists():
    parser.error("output already exists; do not overwrite or merge another experiment")
  allowed = os.sched_getaffinity(0)
  cpu = min(allowed) if args.cpu is None else args.cpu
  if cpu not in allowed:
    parser.error(f"CPU {cpu} is outside the allowed affinity {sorted(allowed)}")
  vecops, validation, catalog, library = snapshot_modules(root)
  reference = root / "benchmarks/dsl_norm_catalog_reference.cpp"
  runner_sha, cpp_sha = sha256(__file__), sha256(reference)
  from vecops import _compiler, _identity
  from vecops._arm_vector_state import get_vector_state

  mode = "installed" if args.reuse_installed_sdk else args.sdk
  build.mkdir(parents=True, exist_ok=True)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("VECOPS_BUILD_INPUT_TRACE", str(build / "build-inputs.jsonl"))
  with validation.sdk_selection(_compiler, mode) as (sdk, layout):
    state = get_vector_state()
    provenance = {
      "source_root": str(root),
      "source_archive_sha256": args.source_archive_sha256,
      "archive_hash_verified": args.source_archive is not None,
      "source_implementation_identity": _identity.implementation_identity(),
      "runner_sha256": runner_sha,
      "cpp_source_sha256": cpp_sha,
      "namespace": args.namespace,
      "comparison": "same-dense-input/output-meta; whole operator inside native repeat loop",
      "sdk_root": str(sdk),
      "sdk_layout": str(layout),
      "sdk_content": validation.sdk_identity(sdk, installed=mode == "installed"),
      "cc": args.cc,
      "cxx": args.cxx,
      "target": args.target,
      "compiler_environment": {key: os.environ.get(key) for key in ("CXXFLAGS", "LDFLAGS", "VECOPS_BUILD_CPU_LIST")},
      "cpu": cpu,
      "affinity_before_compilation": sorted(allowed),
      "rounds": args.rounds,
      "vector_state": {"sve_bytes": state.sve_bytes, "sme_bytes": state.sme_bytes, "env_token": state.env_token},
    }
    (build / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print("Strict softmax A/B provenance: " + json.dumps(provenance, default=str), flush=True)
    compiler = vecops.Compiler(
      target=args.target, cc=args.cc, cxx=args.cxx, jobs=args.jobs, cache_dir=build / "cache", build_dir=build / "build"
    )
    prepared, rows, direct, references = [], {}, {}, {}
    with vecops.collect_compile_requests() as collector:
      for scope, dimensions, dtype, meta, unroll in selected:
        case = catalog.Case("softmax", dimensions, dtype, meta, "none", "strict", "n/a")
        key = (case.schema_key, unroll)
        if key not in rows:
          row, cpp = catalog.kernels(compiler, case, unroll=unroll)
          rows[key] = row
          references.setdefault(case.schema_key, cpp)
        if scope == "direct-rank2" and key not in direct:
          direct[key] = library.library_kernel(compiler, case, unroll=unroll)
        dsl = rows[key] if scope == "row-rank1" else direct[key]
        cpp = references[case.schema_key]
        for kernel in (dsl, cpp):
          kernel.operator(*catalog.metadata(case), 1, **kernel._core.specialization_values)
        prepared.append((scope, case, unroll, (dsl, cpp)))
    print(
      f"Preparing {len(collector)} unique artifacts for {len(prepared)} strict schedule metadata executions", flush=True
    )
    vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
    try:
      os.sched_setaffinity(0, {cpu})
      with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for scope, case, unroll, pair in prepared:
          operands = catalog.validate(case, pair)
          repeat = max(2, min(10000, 4_000_000 // math.prod(case.shape)))
          dsl_us, cpp_us = (
            duration / repeat for duration in catalog.measure(catalog.calls(pair, operands, repeat), args.rounds)
          )
          writer.writerow(
            (
              args.source_archive_sha256,
              runner_sha,
              cpp_sha,
              args.namespace,
              scope,
              case.key[1],
              case.dtype,
              case.meta,
              "strict",
              unroll,
              f"{dsl_us:.4f}",
              f"{cpp_us:.4f}",
              f"{dsl_us / cpp_us:.4f}",
            )
          )
          stream.flush()
    finally:
      os.sched_setaffinity(0, allowed)
  print(f"Completed {len(prepared)} strict softmax schedule metadata executions", flush=True)


if __name__ == "__main__":
  main()
