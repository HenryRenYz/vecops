# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact LayerNormBench/SoftmaxBench metadata with paired native-ABI timing.

Run in the af3 environment on an allocated compute CPU. All compilers are
explicit; compile_batch prepares artifacts before binding the timing process
to one allowed CPU. C++ and DSL share input/output buffers and accuracy tiers.
--list only prints metadata; --resume skips successful CSV rows. LayerNorm's
original catalog uses Any metadata; Softmax has both Const and Any.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T

ROOT = Path(__file__).resolve().parents[1]
DTYPES = ("float32", "float64", "float16", "bfloat16")
AFFINE = ("gamma_beta", "gamma", "beta", "none")
ACCURACY = ("strict", "fast", "estimate")
FIELDS = (
  "operation",
  "shape",
  "dtype",
  "meta",
  "affine",
  "accuracy",
  "variance",
  "dsl_us",
  "cpp_us",
  "dsl_over_cpp",
)


@dataclass(frozen=True)
class Case:
  operation: str
  shape: tuple[int, ...]
  dtype: str
  meta: str
  affine: str
  accuracy: str
  variance: str = "moments"

  @property
  def key(self):
    return (
      self.operation,
      "x".join(map(str, self.shape)),
      self.dtype,
      self.meta,
      self.affine,
      self.accuracy,
      self.variance,
    )

  @property
  def schema_key(self):
    """Only unconstrained Any extents share a rank-specific template pair.

    Const extents keep per-shape specialization, including C++ Softmax's
    shape-dependent algorithm selection. Metadata cases still get distinct
    input validation and paired timing even when their compiled pair is shared.
    """
    shape = self.shape if self.meta == "Const" else len(self.shape)
    return (self.operation, shape, self.dtype, self.meta, self.affine, self.accuracy, self.variance)


def catalog(*, variance=("moments",)):
  """Extract shapes from maintained C++ catalogs rather than duplicating them."""
  layernorm = (ROOT / "benchmarks/ops/LayerNormBench.cpp").read_text()
  body = re.search(r"constexpr LayerNormCase kCases\[\] = \{(.*?)\n\};", layernorm, re.DOTALL).group(1)
  layernorm_shapes = []
  for dimensions, rank in re.findall(r'\{"\w+", \{([^}]+)\}, (\d+)\}', body):
    layernorm_shapes.append(tuple(map(int, dimensions.split(",")))[: int(rank)])
  softmax = (ROOT / "benchmarks/ops/SoftmaxBench.cpp").read_text()
  body = re.findall(r"using SoftmaxCases = CaseList<(.*?)>;", softmax, re.DOTALL)[-1]
  softmax_shapes = [
    tuple(map(int, dimensions.split(","))) for dimensions in re.findall(r"SoftmaxCase<\w+, ([\d, ]+)>", body)
  ]
  return tuple(
    Case("layernorm", shape, dtype, "Any", affine, "strict", algorithm)
    for shape in layernorm_shapes
    for dtype in DTYPES
    for affine in AFFINE
    for algorithm in variance
  ) + tuple(
    Case("softmax", shape, dtype, meta, "none", accuracy, "n/a")
    for shape in softmax_shapes
    for dtype in DTYPES
    for meta in ("Const", "Any")
    for accuracy in ACCURACY
  )


def kernels(compiler, case, *, unroll=2, pass_options=None):
  Shape = case.shape if case.meta == "Const" else tuple(T.dynamic() for _ in case.shape)
  Rows = math.prod(Shape[:-1])
  N = Shape[-1]
  DType = T.DType(case.dtype)
  ComputeType = T.DType("float64" if case.dtype == "float64" else "float32")
  Kind, Affine, Mode, Variance = case.operation, case.affine, case.accuracy, case.variance

  @T.template
  def dsl(
    A: T.Tensor(Shape, dtype=DType),
    G: T.Tensor((N,), dtype=DType),
    B: T.Tensor((N,), dtype=DType),
    C: T.Tensor(Shape, dtype=DType, access="output"),
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      a = T.reshape(A, (Rows, N))
      c = T.reshape(C, (Rows, N))
      for row in T.serial(Rows, unroll=1):
        x = T.load(a[row, :], dtype=ComputeType)
        if Kind == "layernorm":
          y = T.layernorm(x, variance=Variance, unroll=unroll)
          if Affine == "gamma_beta":
            y = y * T.load(G, dtype=ComputeType) + T.load(B, dtype=ComputeType)
          elif Affine == "gamma":
            y = y * T.load(G, dtype=ComputeType)
          elif Affine == "beta":
            y = y + T.load(B, dtype=ComputeType)
        else:
          y = T.softmax(x, unroll=unroll, accuracy=Mode)
        T.store(c[row, :], y)
      T.compiler_fence()

  Op = T.Const(1 if Kind == "layernorm" else 2)
  Accuracy = T.Const(ACCURACY.index(Mode))
  HasGamma = T.Const(Affine in ("gamma_beta", "gamma"))
  HasBeta = T.Const(Affine in ("gamma_beta", "beta"))

  @T.template(source=ROOT / "benchmarks/dsl_norm_catalog_reference.cpp")
  def cpp(
    A: T.Tensor(Shape, dtype=DType),
    G: T.Tensor((N,), dtype=DType),
    B: T.Tensor((N,), dtype=DType),
    C: T.Tensor(Shape, dtype=DType, access="output"),
    repeat: int,
  ):
    Op  # noqa: B018 - source-kernel specialization capture
    Accuracy  # noqa: B018 - source-kernel specialization capture
    ComputeType  # noqa: B018 - source-kernel specialization capture
    HasGamma  # noqa: B018 - source-kernel specialization capture
    HasBeta  # noqa: B018 - source-kernel specialization capture

  return T.compile(dsl, compiler=compiler, pass_options=pass_options), T.compile(cpp, compiler=compiler)


def numpy_dtype(name):
  if name == "bfloat16":
    import ml_dtypes

    return np.dtype(ml_dtypes.bfloat16)
  return np.dtype(name)


def aligned_empty(shape, dtype):
  dtype = numpy_dtype(dtype)
  storage = np.empty(math.prod(shape) * dtype.itemsize + 4096, dtype=np.uint8)
  return np.ndarray(shape, dtype=dtype, buffer=storage, offset=(-storage.ctypes.data) % 4096)


def metadata(case):
  gamma = vecops.TensorMeta((case.shape[-1],), dtype=case.dtype)
  beta = vecops.TensorMeta((case.shape[-1],), dtype=case.dtype)
  return vecops.TensorMeta(case.shape, dtype=case.dtype), gamma, beta, vecops.TensorMeta(case.shape, dtype=case.dtype)


def calls(pair, operands, repeat):
  return tuple(
    lambda kernel=kernel: kernel.operator.native(*operands, repeat, **kernel._core.specialization_values)
    for kernel in pair
  )


def measure(calls_, rounds):
  samples = [[] for _ in calls_]
  order = random.Random(42)
  for call in calls_:
    call()
  for _ in range(rounds):
    indices = list(range(len(calls_)))
    order.shuffle(indices)
    for index in indices:
      begin = time.perf_counter_ns()
      calls_[index]()
      samples[index].append((time.perf_counter_ns() - begin) / 1000)
  return tuple(statistics.median(sample) for sample in samples)


def validate(case, pair):
  total = math.prod(case.shape)
  input_ = aligned_empty(case.shape, case.dtype)
  # Reproduce the C++ benchmark's input families. Values matter on ARM:
  # some processors have input-dependent FEXPA throughput, so substituting
  # an unrelated random/narrower score distribution can hide a regression.
  indices = np.arange(total, dtype=np.int32)
  coefficient = np.float32(0.07 if case.operation == "layernorm" else 0.17)
  input_[:] = (
    (indices % 37 - 18).astype(np.float32) * coefficient + (indices % 11).astype(np.float32) * np.float32(0.013)
  ).reshape(case.shape)
  # Keep both signatures identical: unused affine operands are real tensors
  # removed by compile-time dispatch. The DSL does not yet bind optional
  # tensors in a parsed body; these unused arguments are never read.
  gamma = aligned_empty((case.shape[-1],), case.dtype)
  beta = aligned_empty((case.shape[-1],), case.dtype)
  gamma[:] = np.float32(0.75) + np.float32(0.001) * (np.arange(len(gamma)) % 257).astype(np.float32)
  beta[:] = np.float32(-0.2) + np.float32(0.0007) * (np.arange(len(beta)) % 263).astype(np.float32)
  output = aligned_empty(case.shape, case.dtype)
  operands = (input_, gamma, beta, output)
  ds, cp = calls(pair, operands, 1)
  cp()
  reference = output.astype(np.float64)
  ds()
  if case.operation == "layernorm":
    tolerance = {"float32": 2e-5, "float64": 2e-8, "float16": 2e-2, "bfloat16": 4e-2}[case.dtype]
  elif case.accuracy == "estimate":
    tolerance = {"float16": 1e-2, "bfloat16": 2e-2}.get(case.dtype, 8e-3)
  else:
    tolerance = {"float32": 3e-5, "float64": 2e-12, "float16": 2e-3, "bfloat16": 8e-3}[case.dtype]
  np.testing.assert_allclose(output.astype(np.float64), reference, rtol=tolerance, atol=tolerance)
  x = input_.astype(np.float64)
  if case.operation == "layernorm":
    centered = x - x.mean(axis=-1, keepdims=True)
    expected = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)
    if case.affine in ("gamma_beta", "gamma"):
      expected *= gamma.astype(np.float64)
    if case.affine in ("gamma_beta", "beta"):
      expected += beta.astype(np.float64)
  else:
    exponent = np.exp(x - x.max(axis=-1, keepdims=True))
    expected = exponent / exponent.sum(axis=-1, keepdims=True)
  np.testing.assert_allclose(output.astype(np.float64), expected, rtol=tolerance, atol=tolerance)
  return operands


def load_completed(path):
  if not path.exists():
    return set()
  with path.open(newline="") as stream:
    reader = csv.DictReader(stream)
    if tuple(reader.fieldnames or ()) != FIELDS:
      raise ValueError("resume CSV header does not match the current normalization benchmark format")
    return {tuple(row[name] for name in FIELDS[:7]) for row in reader}


def collect_kernel_pairs(compiler, cases, *, unroll=2, pass_options=None):
  """Collect each dynamic schema once, while preserving every timed case.

  The C++ reference always implements moments variance; a centered DSL case
  intentionally compares against the same C++ artifact rather than compiling
  another identical reference. Const schemas remain shape-specialized.
  """
  prepared = []
  pairs_by_schema = {}
  cpp_by_schema = {}
  for case in cases:
    if case.schema_key not in pairs_by_schema:
      dsl, cpp = kernels(compiler, case, unroll=unroll, pass_options=pass_options)
      cpp_key = case.schema_key[:-1] if case.operation == "layernorm" else case.schema_key
      if cpp_key not in cpp_by_schema:
        cpp_by_schema[cpp_key] = cpp
        cpp.operator(*metadata(case), 1, **cpp._core.specialization_values)
      dsl.operator(*metadata(case), 1, **dsl._core.specialization_values)
      pairs_by_schema[case.schema_key] = (dsl, cpp_by_schema[cpp_key])
    prepared.append((case, pairs_by_schema[case.schema_key]))
  return prepared


def main():
  from dsl_catalog_utils import parse_partition, validate_catalog_filters

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--operation", nargs="+", choices=("layernorm", "softmax"))
  parser.add_argument("--shape", nargs="+", help="Filter logical shapes, for example 64x768")
  parser.add_argument("--dtype", nargs="+", choices=DTYPES)
  parser.add_argument("--meta", nargs="+", choices=("Const", "Any"))
  parser.add_argument("--affine", nargs="+", choices=AFFINE)
  parser.add_argument("--accuracy", nargs="+", choices=ACCURACY)
  parser.add_argument(
    "--variance",
    nargs="+",
    default=["moments"],
    choices=("centered", "moments"),
    help="LayerNorm algorithm; moments matches C++, centered keeps the stable DSL default",
  )
  parser.add_argument("--list", action="store_true", help="Only print selected metadata, no compilation")
  parser.add_argument("--partition", default="0/1", help="Stable metadata partition INDEX/COUNT")
  parser.add_argument("--limit", type=int)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-norm-catalog")
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--target", default=os.environ.get("VECOPS_TEST_TARGET", "Native"))
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--cpu", type=int, help="Timing CPU, default first CPU in the current allowed affinity")
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--unroll", type=int, default=2)
  parser.add_argument("--math-chunk-pow", type=int, default=1)
  args = parser.parse_args()
  part, count = parse_partition(parser, args.partition)
  if args.limit is not None and args.limit < 0:
    parser.error("--limit must be nonnegative")
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
    if index % count == part and case.key not in completed and case in matching
  ]
  if args.limit is not None:
    cases = cases[: args.limit]
  if args.list:
    writer = csv.writer(__import__("sys").stdout)
    writer.writerow(FIELDS[:7])
    writer.writerows(case.key for case in cases)
    return
  if not cases:
    print(
      "0 executions: no metadata remains after partition/resume/limit; no correctness or timing checks ran", flush=True
    )
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
  with vecops.collect_compile_requests() as collector:
    prepared = collect_kernel_pairs(
      compiler, cases, unroll=args.unroll, pass_options={"cpu_math_chunk_pow": args.math_chunk_pow}
    )
  print(f"Preparing {len(collector)} unique artifacts for {len(prepared)} metadata cases", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  os.sched_setaffinity(0, {cpu})
  append = args.resume and args.output.exists() and args.output.stat().st_size > 0
  stream = args.output.open("a" if append else "w", newline="") if args.output else __import__("sys").stdout
  try:
    writer = csv.writer(stream)
    if not append:
      writer.writerow(FIELDS)
    for case, pair in prepared:
      operands = validate(case, pair)
      repeat = max(2, min(10000, 4_000_000 // math.prod(case.shape)))
      td, tc = (duration / repeat for duration in measure(calls(pair, operands, repeat), args.rounds))
      writer.writerow((*case.key, f"{td:.4f}", f"{tc:.4f}", f"{td / tc:.4f}"))
      stream.flush()
      if args.output:
        print(f"{'/'.join(case.key)}: {td / tc:.3f}x", flush=True)
  finally:
    if args.output:
      stream.close()


if __name__ == "__main__":
  main()
