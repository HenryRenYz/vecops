# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Same-source BF16 normalization math-chunk diagnostic, not full catalog.

The six 8x128 keys pair Const/Any for moments LayerNorm (none/gamma_beta)
and strict Softmax. LayerNorm's original benchmark has Any metadata and does
not contain 8x128: these are maintained operators + diagnostic metadata
extensions, not original-catalog coverage. All variants retain the existing
library/row/C++ factories, precision, input generators and validation oracle.
Only cpu_math_chunk_pow changes; requested operator unroll stays two.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path

import vecops
from dsl_benchmark_utils import add_toolchain_arguments
from dsl_norm_catalog import Case, calls, kernels, measure, metadata, validate
from dsl_norm_library import library_kernel

ROOT = Path(__file__).resolve().parents[1]
KEY_FIELDS = ("operation", "shape", "dtype", "meta", "affine", "accuracy", "variance")
FIELDS = (
  *KEY_FIELDS,
  "math_chunk_pow",
  "library_us",
  "row_us",
  "cpp_us",
  "library_over_cpp",
  "row_over_cpp",
)


def cases():
  return tuple(
    Case(operation, (8, 128), "bfloat16", meta, affine, "strict", variance)
    for operation, affine, variance in (
      ("layernorm", "none", "moments"),
      ("layernorm", "gamma_beta", "moments"),
      ("softmax", "none", "n/a"),
    )
    for meta in ("Const", "Any")
  )


def pass_options(chunk):
  return {"cpu_math_chunk_pow": chunk, "cpu_pointwise_register_layout_policy": "off"}


def parser():
  result = argparse.ArgumentParser(description=__doc__)
  add_toolchain_arguments(result)
  result.add_argument("--chunk-pow", nargs="+", type=int, choices=(0, 1, 2), default=(1, 0, 2))
  result.add_argument("--rounds", type=int, default=9)
  result.add_argument("--jobs", type=int, default=16)
  result.add_argument("--cpu", type=int)
  result.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-norm-chunk-ab")
  result.add_argument("--output", type=Path)
  result.add_argument("--source-archive-sha256", required=True)
  result.add_argument("--list", action="store_true")
  return result


def main(arguments=None):
  argument_parser = parser()
  args = argument_parser.parse_args(arguments)
  if not re.fullmatch(r"[0-9a-f]{64}", args.source_archive_sha256):
    argument_parser.error("--source-archive-sha256 requires a verified source archive SHA256")
  if len(args.chunk_pow) != len(set(args.chunk_pow)) or args.rounds < 1 or args.jobs < 1:
    argument_parser.error("chunk powers must be unique; rounds/jobs must be positive")
  selected = cases()
  if args.list:
    writer = csv.writer(sys.stdout)
    writer.writerow((*KEY_FIELDS, "math_chunk_pow"))
    writer.writerows((*case.key, chunk) for case in selected for chunk in args.chunk_pow)
    return
  if args.output is None:
    argument_parser.error("--output is required to retain diagnostic timings and provenance")
  manifest_path = args.output.with_suffix(".manifest.json")
  if args.output.exists() or manifest_path.exists():
    argument_parser.error("choose fresh timing/manifest output paths; evidence is not overwritten")
  allowed = os.sched_getaffinity(0)
  cpu = min(allowed) if args.cpu is None else args.cpu
  if cpu not in allowed:
    argument_parser.error(f"CPU {cpu} is outside the current allowed affinity {sorted(allowed)}")
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  prepared = []
  with vecops.collect_compile_requests() as collector:
    for case in selected:
      variants = []
      cpp_shared = None
      for chunk in args.chunk_pow:
        options = pass_options(chunk)
        row, cpp = kernels(compiler, case, unroll=2, pass_options=options)
        direct = library_kernel(compiler, case, unroll=2, pass_options=options)
        for compiled in (row, cpp, direct):
          compiled.operator(*metadata(case), 1, **compiled._core.specialization_values)
        cpp_shared = cpp if cpp_shared is None else cpp_shared
        variants.append((chunk, direct, row))
      prepared.append((case, variants, cpp_shared))
  sources = {
    "row_factory": (inspect.getfile(kernels), inspect.getsource(kernels)),
    "library_factory": (inspect.getfile(library_kernel), inspect.getsource(library_kernel)),
    "validation": (inspect.getfile(validate), inspect.getsource(validate)),
  }
  manifest = {
    "source_root": str(ROOT),
    "source_archive_sha256": args.source_archive_sha256,
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "factory_inputs": {
      name: {
        "path": path,
        "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "function_sha256": hashlib.sha256(function.encode()).hexdigest(),
      }
      for name, (path, function) in sources.items()
    },
    "cpp_reference": {
      "path": str(ROOT / "benchmarks/dsl_norm_catalog_reference.cpp"),
      "sha256": hashlib.sha256((ROOT / "benchmarks/dsl_norm_catalog_reference.cpp").read_bytes()).hexdigest(),
    },
    "compiler": {"cc": args.cc, "cxx": args.cxx, "target": args.target},
    "environment": {key: os.environ.get(key) for key in ("CXXFLAGS", "LDFLAGS", "CCS_JOB_ID")},
    "cpu": cpu,
    "rounds": args.rounds,
    "unroll": 2,
    "pass_options": {chunk: pass_options(chunk) for chunk in args.chunk_pow},
    "cases": [list(case.key) for case in selected],
    "logical_requests": len(collector),
    "timing_order": "All library/row variants and one same-meta C++ reference interleaved each round",
    "coverage": "Maintained operators + diagnostic metadata extension; not original catalog coverage",
    "arithmetic": "Moments versus moments/strict Softmax; no change to FMA or accuracy contracts",
  }
  print("Math-chunk diagnostic manifest: " + json.dumps(manifest, sort_keys=True), flush=True)
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  with args.output.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(FIELDS)
    try:
      os.sched_setaffinity(0, {cpu})
      for case, variants, cpp in prepared:
        operands = None
        for _, direct, row in variants:
          validate(case, (direct, cpp))
          operands = validate(case, (row, cpp))
        repeat = max(2, min(10000, 4_000_000 // math.prod(case.shape)))
        compiled = tuple(kernel for _, direct, row in variants for kernel in (direct, row)) + (cpp,)
        timings = tuple(duration / repeat for duration in measure(calls(compiled, operands, repeat), args.rounds))
        cpp_us = timings[-1]
        for index, (chunk, _, _) in enumerate(variants):
          direct_us, row_us = timings[2 * index : 2 * index + 2]
          writer.writerow(
            (
              *case.key,
              chunk,
              f"{direct_us:.6f}",
              f"{row_us:.6f}",
              f"{cpp_us:.6f}",
              f"{direct_us / cpp_us:.6f}",
              f"{row_us / cpp_us:.6f}",
            )
          )
        stream.flush()
    finally:
      os.sched_setaffinity(0, allowed)


if __name__ == "__main__":
  main()
