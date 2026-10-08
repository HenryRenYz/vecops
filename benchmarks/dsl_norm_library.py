# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Measure the actual rank-two normalization library path, not a row wrapper.

Each case compares a direct ``T.library.layernorm/softmax(T.load(A))`` with
the existing row-wise DSL catalog kernel and the exact C++ operator. All
three use the same rank-two inputs, output, affine metadata and accuracy.
Rank-three/four inputs are deliberately excluded: the current library does
not support those ranks, and silently reshaping them would hide that limit.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import vecops
import vecops.language as T
from dsl_catalog_utils import parse_partition, validate_catalog_filters
from dsl_norm_catalog import (
  ACCURACY,
  AFFINE,
  DTYPES,
  ROOT,
  calls,
  collect_kernel_pairs,
  measure,
  metadata,
  validate,
)
from dsl_norm_catalog import (
  catalog as norm_catalog,
)

SCOPE = "direct-rank2-library"
KEY_FIELDS = ("scope", "operation", "shape", "dtype", "meta", "affine", "accuracy", "variance")
FIELDS = (*KEY_FIELDS, "library_us", "row_us", "cpp_us", "library_over_cpp", "library_over_row")


def catalog(*, variance=("moments",)):
  return tuple(case for case in norm_catalog(variance=variance) if len(case.shape) == 2)


def library_kernel(compiler, case, *, unroll=2, pass_options=None):
  if len(case.shape) != 2:
    raise ValueError("direct normalization library benchmark requires a rank-two case")
  Shape = case.shape if case.meta == "Const" else tuple(T.dynamic(lower=0) for _ in case.shape)
  N = Shape[-1]
  DType = T.DType(case.dtype)
  ComputeType = T.DType("float64" if case.dtype == "float64" else "float32")
  Kind, Affine, Mode, Variance = case.operation, case.affine, case.accuracy, case.variance

  @T.template
  def direct_library(
    A: T.Tensor(Shape, dtype=DType),
    G: T.Tensor((N,), dtype=DType),
    B: T.Tensor((N,), dtype=DType),
    C: T.Tensor(Shape, dtype=DType, access="output"),
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      x = T.load(A, dtype=ComputeType)
      if Kind == "layernorm":
        y = T.library.layernorm(x, variance=Variance, unroll=unroll)
        if Affine == "gamma_beta":
          y = y * T.load(G, dtype=ComputeType) + T.load(B, dtype=ComputeType)
        elif Affine == "gamma":
          y = y * T.load(G, dtype=ComputeType)
        elif Affine == "beta":
          y = y + T.load(B, dtype=ComputeType)
      else:
        y = T.library.softmax(x, unroll=unroll, accuracy=Mode)
      T.store(C, y)
      T.compiler_fence()

  return T.compile(direct_library, compiler=compiler, pass_options=pass_options)


def collect_kernel_triples(compiler, cases, *, unroll=2, pass_options=None):
  """Reuse Any schemas; still validate and time every selected metadata."""
  prepared = collect_kernel_pairs(compiler, cases, unroll=unroll, pass_options=pass_options)
  direct_by_schema = {}
  triples = []
  for case, (row, cpp) in prepared:
    if case.schema_key not in direct_by_schema:
      direct = library_kernel(compiler, case, unroll=unroll, pass_options=pass_options)
      direct.operator(*metadata(case), 1, **direct._core.specialization_values)
      direct_by_schema[case.schema_key] = direct
    triples.append((case, (direct_by_schema[case.schema_key], row, cpp)))
  return triples


def load_completed(path):
  if not path.exists():
    return set()
  with path.open(newline="") as stream:
    reader = csv.DictReader(stream)
    if tuple(reader.fieldnames or ()) != FIELDS:
      raise ValueError("resume CSV does not match the direct rank-two library benchmark")
    return {tuple(row[name] for name in KEY_FIELDS) for row in reader}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--operation", nargs="+", choices=("layernorm", "softmax"))
  parser.add_argument("--shape", nargs="+")
  parser.add_argument("--dtype", nargs="+", choices=DTYPES)
  parser.add_argument("--meta", nargs="+", choices=("Const", "Any"))
  parser.add_argument("--affine", nargs="+", choices=AFFINE)
  parser.add_argument("--accuracy", nargs="+", choices=ACCURACY)
  parser.add_argument("--variance", nargs="+", choices=("centered", "moments"), default=["moments"])
  parser.add_argument("--list", action="store_true")
  parser.add_argument("--partition", default="0/1")
  parser.add_argument("--limit", type=int)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-norm-library")
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--target", default=os.environ.get("VECOPS_TEST_TARGET", "Native"))
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--cpu", type=int)
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--unroll", type=int, default=2)
  parser.add_argument("--math-chunk-pow", type=int, default=1)
  parser.add_argument("--register-layout-policy", choices=("off", "auto"), default="off")
  parser.add_argument(
    "--register-layout-force",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Explicit experimental bool overrides the policy; omitted preserves off/auto selection",
  )
  args = parser.parse_args()
  part, count = parse_partition(parser, args.partition)
  if args.limit is not None and args.limit < 0:
    parser.error("--limit must be nonnegative")
  if args.rounds < 1:
    parser.error("--rounds must be positive")
  if args.resume and args.output is None:
    parser.error("--resume requires --output")
  all_cases = catalog(variance=args.variance)
  matching = validate_catalog_filters(
    parser,
    all_cases,
    (
      ("--operation", args.operation, lambda case: case.operation),
      ("--shape", args.shape, lambda case: case.key[1]),
      ("--dtype", args.dtype, lambda case: case.dtype),
      ("--meta", args.meta, lambda case: case.meta),
      ("--affine", args.affine, lambda case: case.affine),
      ("--accuracy", args.accuracy, lambda case: case.accuracy),
    ),
    allow_empty=args.list,
  )
  completed = load_completed(args.output) if args.resume else set()
  cases = [
    case
    for index, case in enumerate(all_cases)
    if index % count == part and case in matching and (SCOPE, *case.key) not in completed
  ]
  if args.limit is not None:
    cases = cases[: args.limit]
  if args.list:
    writer = csv.writer(sys.stdout)
    writer.writerow(KEY_FIELDS)
    writer.writerows((SCOPE, *case.key) for case in cases)
    return
  if not cases:
    print("0 executions: no metadata remains after partition/resume/limit; no correctness or timing checks ran")
    return
  if any(case.dtype == "bfloat16" for case in cases) and importlib.util.find_spec("ml_dtypes") is None:
    parser.error("BF16 cases require the optional ml_dtypes package")
  allowed = os.sched_getaffinity(0)
  cpu = min(allowed) if args.cpu is None else args.cpu
  if cpu not in allowed:
    parser.error(f"CPU {cpu} is not in the current allowed affinity {sorted(allowed)}")
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  pass_options = {
    "cpu_math_chunk_pow": args.math_chunk_pow,
    "cpu_pointwise_register_layout_policy": args.register_layout_policy,
  }
  if args.register_layout_force is not None:
    pass_options["cpu_pointwise_register_layout"] = args.register_layout_force
  print("Normalization pass options: " + json.dumps(pass_options, sort_keys=True), flush=True)
  with vecops.collect_compile_requests() as collector:
    prepared = collect_kernel_triples(compiler, cases, unroll=args.unroll, pass_options=pass_options)
  print(f"Preparing {len(collector)} unique artifacts for {len(prepared)} direct rank-two metadata cases", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  append = args.resume and args.output.exists() and args.output.stat().st_size > 0
  stream = args.output.open("a" if append else "w", newline="") if args.output else sys.stdout
  try:
    os.sched_setaffinity(0, {cpu})
    writer = csv.writer(stream)
    if not append:
      writer.writerow(FIELDS)
    for case, (direct, row, cpp) in prepared:
      # Both DSL paths independently pass C++ and FP64 reference checks.
      # Reuse the final operands for all three timings, not Python-level loops.
      validate(case, (direct, cpp))
      operands = validate(case, (row, cpp))
      repeat = max(2, min(10000, 4_000_000 // math.prod(case.shape)))
      library_us, row_us, cpp_us = (
        duration / repeat for duration in measure(calls((direct, row, cpp), operands, repeat), args.rounds)
      )
      writer.writerow(
        (
          SCOPE,
          *case.key,
          f"{library_us:.4f}",
          f"{row_us:.4f}",
          f"{cpp_us:.4f}",
          f"{library_us / cpp_us:.4f}",
          f"{library_us / row_us:.4f}",
        )
      )
      stream.flush()
  finally:
    os.sched_setaffinity(0, allowed)
    if args.output:
      stream.close()


if __name__ == "__main__":
  main()
