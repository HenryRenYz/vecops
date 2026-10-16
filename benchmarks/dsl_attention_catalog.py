# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""AttentionBench shapes/metadata, paired DSL/C++ native-ABI timings.

Dense SDPA only: sparse/IBS compute different problems. Inputs, BF16 atom
conversion, probability dtype, Fast exponential and scale match AttentionBench.
The caller tiles Q by 32. K blocks 32 and 64 compare matching C++ blocks and
the library default. C++ scratch is preallocated; compilation is untimed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import statistics
import time

from dsl_timing import warmup_native
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T
from dsl_norm_catalog import aligned_empty, calls

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
  group: str
  name: str
  shape: tuple[int, int, int, int]
  metadata: tuple[str, str, str, str]
  qk: str = "bfloat16"
  value: str = "bfloat16"
  output: str = "float32"

  @property
  def schema(self):
    return self.metadata, self.qk, self.value, self.output


def catalog():
  source = (ROOT / "benchmarks/ops/AttentionBench.cpp").read_text()
  body = re.search(r"constexpr Case Cases\[\] = \{(.*?)\n\};", source, re.DOTALL).group(1)
  records = []
  for group, name, m, n, d, v, am, an in re.findall(
      r'\{"([^"]+)", "([^"]+)", (\d+), (\d+), (\d+), (\d+), (true|false), (true|false)\}', body):
    records.append(Case(group, name, tuple(map(int, (m, n, d, v))),
                        ("aligned32" if am == "true" else "Any",
                         "aligned32" if an == "true" else "Any", "Any", "Any")))
  coverage = re.search(r'Case MetaCase\{\s*"coverage", "[^"]+", (\d+), (\d+), (\d+), (\d+)', source)
  shape = tuple(map(int, coverage.groups()))
  records.extend((
    Case("coverage", "fixed_n", shape, (str(shape[0]), str(shape[1]), "Any", "Any"), "float32", "bfloat16", "float32"),
    Case("coverage", "fixed_all_dims", shape, tuple(map(str, shape)), "float32", "bfloat16", "float32"),
    Case("coverage", "aligned32_n", shape, ("bounded32", "bounded32", "Any", "Any"), "float16", "float32", "float16"),
    Case("coverage", "any_n", shape, ("Any",) * 4, "bfloat16", "float16", "bfloat16"),
  ))
  if not records:
    raise ValueError("AttentionBench catalog is empty")
  return records


def dimensions(case):
  return tuple(T.dynamic(32) if s == "aligned32" else T.dynamic(32, 32, 128) if s == "bounded32"
               else T.dynamic() if s == "Any" else int(s) for s in case.metadata)


def kernels(compiler, case, block_sizes):
  M, N, D, V = dimensions(case)
  QType, VType, OType = map(T.DType, (case.qk, case.value, case.output))
  W = T.dynamic(lower=0)
  result = {}
  for block in block_sizes:
    Block = block

    @T.template
    def dsl(Q: T.Tensor((M, D), dtype=QType), K: T.Tensor((N, D), dtype=QType),
            Value: T.Tensor((N, V), dtype=VType), O: T.Tensor((M, V), dtype=OType, access="output"),
            Scratch: T.Tensor((W,), dtype="uint8", access="inout"), Info: T.Tensor((1,), dtype="int64", access="output"),
            scale: float, repeat: int):
      for iteration in T.serial(repeat, unroll=1):
        for row in T.serial(0, M, 32, unroll=1):
          end = T.min(row + 32, M)
          # AttentionBench fixes the native atom to BF16, independently of
          # memory dtype. Match its conversion before both matrix products.
          T.library.scaled_dot_product_attention(
            T.cast(T.load(Q[row:end, :]), "bfloat16"), T.cast(T.load(K), "bfloat16"),
            T.cast(T.load(Value), "bfloat16"), O[row:end, :],
            scale=scale, block_k=Block, probability_dtype="bfloat16", accuracy="fast")  # noqa: B023 -- T.template freezes captures now
        T.compiler_fence()

    result[f"dsl_k{block}"] = T.compile(dsl, compiler=compiler)
  for name, index in (("automatic", 0), ("streaming", 2), ("materialized", 1)):
    Strategy = T.Const(index)

    @T.template(source=ROOT / "benchmarks/dsl_attention_reference.cpp")
    def cpp(Q: T.Tensor((M, D), dtype=QType), K: T.Tensor((N, D), dtype=QType),
            Value: T.Tensor((N, V), dtype=VType), O: T.Tensor((M, V), dtype=OType, access="output"),
            Scratch: T.Tensor((W,), dtype="uint8", access="inout"), Info: T.Tensor((1,), dtype="int64", access="output"),
            scale: float, repeat: int):
      Strategy  # noqa: B018,B023 -- T.template freezes source specialization now

    result[f"cpp_{name}"] = T.compile(cpp, compiler=compiler)
  return result


def metadata(case):
  m, n, d, v = case.shape
  return (vecops.TensorMeta((m, d), dtype=case.qk), vecops.TensorMeta((n, d), dtype=case.qk),
          vecops.TensorMeta((n, v), dtype=case.value), vecops.TensorMeta((m, v), dtype=case.output),
          vecops.TensorMeta((64,), dtype="uint8"), vecops.TensorMeta((1,), dtype="int64"))


def operands(case):
  m, n, d, v = case.shape
  tensors = []
  for shape, dtype, modulus in (((m, d), case.qk, 17), ((n, d), case.qk, 19), ((n, v), case.value, 23)):
    value = aligned_empty(shape, dtype)
    indices = np.arange(math.prod(shape), dtype=np.int64)
    value[:] = ((indices % modulus - modulus // 2).astype(np.float32) / 32).reshape(shape)
    tensors.append(value)
  return (*tensors, aligned_empty((m, v), case.output))


def reference(q, k, v, block):
  # Scalar-library composition reference with BF16 rounded unnormalized P
  # for every streamed block, matching each implementation's block schedule.
  import ml_dtypes
  q, k, v = (x.astype(ml_dtypes.bfloat16).astype(np.float64) for x in (q, k, v))
  maximum = np.full((len(q), 1), -np.inf)
  denominator = np.zeros_like(maximum)
  out = np.zeros((len(q), v.shape[1]))
  scale = float(np.float32(1) / np.sqrt(np.float32(q.shape[1])))
  for start in range(0, len(k), block):
    scores = q @ k[start:start + block].T * scale
    next_max = np.maximum(maximum, scores.max(axis=1, keepdims=True))
    correction = np.exp(maximum - next_max)
    probability = np.exp(scores - next_max)
    denominator = denominator * correction + probability.sum(axis=1, keepdims=True)
    out = out * correction + probability.astype(ml_dtypes.bfloat16).astype(np.float64) @ v[start:start + block]
    maximum = next_max
  return out / denominator


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--list", action="store_true")
  parser.add_argument("--output", type=Path, default=ROOT / "build-attention-audit")
  parser.add_argument("--case", nargs="*")
  parser.add_argument("--blocks", nargs="+", type=int, default=[32, 64])
  parser.add_argument("--cpu", type=int)
  parser.add_argument("--cc", default="gcc")
  parser.add_argument("--cxx", default="g++")
  parser.add_argument("--sdk", type=Path, help="explicit installed SDK; source tree otherwise")
  parser.add_argument("--jobs", type=int, default=8)
  parser.add_argument("--rounds", type=int, default=12)
  parser.add_argument("--prepare-only", action="store_true", help="Build and validate without timing")
  parser.add_argument("--cache-only", action="store_true", help="Require retained native artifacts")
  args = parser.parse_args()
  cases = [c for c in catalog() if not args.case or c.name in args.case or c.group in args.case]
  if args.list:
    print(json.dumps([asdict(c) for c in cases], indent=2))
    return
  args.output = args.output.resolve()
  args.output.mkdir(parents=True, exist_ok=True)

  class Compiler(vecops.Compiler):
    def _native_config(self):
      config = super()._native_config()
      config.sdk.path = args.sdk.resolve() if args.sdk else ROOT
      config.sdk.layout = vecops._C.SdkLayout.package if args.sdk else vecops._C.SdkLayout.source_tree
      return config

  compiler = Compiler(cc=args.cc, cxx=args.cxx, jobs=args.jobs, cache_dir=args.output / "cache",
                      build_dir=args.output / "build", cflags=("-ffp-contract=off",),
                      cache_mode="cache-only" if args.cache_only else "read-write")
  groups = {}
  with vecops.collect_compile_requests() as collector:
    for case in cases:
      if case.schema not in groups:
        group = groups[case.schema] = kernels(compiler, case, args.blocks)
        source_dir = args.output / "sources"
        source_dir.mkdir(exist_ok=True)
        for label, kernel in group.items():
          if kernel.generated_source is not None:
            (source_dir / f"schema{len(groups)}-{label}.cpp").write_text(kernel.generated_source)
          kernel.operator(*metadata(case), 1.0, 1, **kernel._core.specialization_values)
  print(f"Preparing {len(collector.requests)} kernels for {len(cases)} catalog cases", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  allowed = os.sched_getaffinity(0)
  cpu = args.cpu if args.cpu is not None else min(allowed)
  if cpu not in allowed:
    raise ValueError("benchmark CPU is outside allocation")
  os.sched_setaffinity(0, {cpu})
  rows = []
  with (args.output / "raw.jsonl").open("w") as log:
    for case in cases:
      group = groups[case.schema]
      q, k, v, o = operands(case)
      scale = float(np.float32(1) / np.sqrt(np.float32(case.shape[2])))
      info = np.zeros((1,), np.int64)
      scratch = aligned_empty((64,), "uint8")
      sizes = {}
      for name, kernel in group.items():
        if name.startswith("cpp_"):
          kernel.operator.native(q, k, v, o, scratch, info, scale, 0, **kernel._core.specialization_values)
          sizes[name] = int(info[0])
      scratch = aligned_empty((max(64, *sizes.values()),), "uint8")
      buffers = (q, k, v, o, scratch, info, scale)
      # All implementations also approximate ordinary dense SDPA. Validate
      # absolute error since the deterministic benchmark output crosses zero.
      exact_q, exact_k, exact_v = (x.astype(np.float64) for x in (q, k, v))
      scores = exact_q @ exact_k.T * scale
      weights = np.exp(scores - scores.max(axis=1, keepdims=True))
      dense = weights @ exact_v / weights.sum(axis=1, keepdims=True)
      errors = {}
      for name, kernel in group.items():
        kernel.operator.native(*buffers, 1, **kernel._core.specialization_values)
        error = float(np.max(np.abs(o.astype(np.float64) - dense)))
        if error > 0.003:
          raise AssertionError(f"{case.name} {name}: dense SDPA max error {error}")
        if name.startswith("dsl_"):
          expected = reference(q, k, v, int(name[5:]))
          np.testing.assert_allclose(o.astype(np.float64), expected, rtol=0.02, atol=0.0006)
        errors[name] = error
      if args.prepare_only:
        rows.append({"case": asdict(case), "max_abs_error": errors, "validated": True, "timed": False})
        (args.output / "prepared.json").write_text(json.dumps(rows, indent=2))
        continue
      # Native ABI dispatch has fixed overhead. Calibrate each implementation
      # separately to >=25 ms: equal tiny repeat counts severely overstate the
      # cost of a fast C++ kernel relative to the original AttentionBench.
      warmup = warmup_native([calls((kernel,), buffers, 1)[0] for kernel in group.values()])
      repeats, invokes = {}, {}
      for name, kernel in group.items():
        repeat = 1
        while True:
          invoke = calls((kernel,), buffers, repeat)[0]
          start = time.perf_counter_ns()
          invoke()
          elapsed = time.perf_counter_ns() - start
          if elapsed >= 25_000_000 or repeat >= 65536:
            break
          repeat *= 2
        repeats[name], invokes[name] = repeat, invoke
      samples = {name: [] for name in group}
      rng = random.Random(42)
      for round_ in range(args.rounds):
        order = list(group)
        rng.shuffle(order)
        for name in order:
          start = time.perf_counter_ns()
          invokes[name]()
          ns = (time.perf_counter_ns() - start) / repeats[name]
          samples[name].append(ns)
          log.write(json.dumps({"case": asdict(case), "implementation": name, "round": round_, "repeat": repeats[name], "ns": ns}) + "\n")
      log.flush()
      medians = {name: statistics.median(values) for name, values in samples.items()}
      row = {"case": asdict(case), "warmup": warmup, "median_ns": medians, "dsl_over_cpp_auto":
             {name: ns / medians["cpp_automatic"] for name, ns in medians.items() if name.startswith("dsl_")},
             "max_abs_error": errors, "cpp_workspace_bytes": sizes, "repeats": repeats}
      rows.append(row)
      report = {"host": platform.node(), "machine": platform.machine(), "cpu": cpu,
                "compiler": args.cxx, "vector_state": str(compiler.vector_state), "rows": rows}
      (args.output / "results.json").write_text(json.dumps(report, indent=2))
      print(case.group, case.name, {name: round(ns / 1000, 2) for name, ns in medians.items()}, flush=True)


if __name__ == "__main__":
  main()
