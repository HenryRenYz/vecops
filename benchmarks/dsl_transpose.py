"""Exact TransposeBench metadata, shared I/O, native inner-loop A/B timing.

Run in af3, pinned to one CPU. JIT is excluded; the two kernels use the same
ABI, page-aligned buffers and compiler barriers. CSV times are per transpose,
not per Python invocation. The metadata is extracted from the C++ catalog to
avoid drifting from its shapes, dtype triples and const/Any/aligned cases.
"""
import argparse
import csv
import os
import random
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import vecops
import vecops.language as T

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "benchmarks/ops/TransposeBenchCommon.h"


def catalog():
  text = CATALOG.read_text()
  def shapes(name):
    body = re.search(rf"using {name} = CaseList<(.*?)>;", text, re.S).group(1)
    return tuple((int(m), int(n)) for m, n in re.findall(r"TransposeCase<\w+, (\d+), (\d+)>", body))
  pairs = []
  dtype = {"int8_t": "int8", "float16_t": "float16", "bfloat16_t": "bfloat16",
           "float32_t": "float32", "float64_t": "float64"}
  for name in ("Int8", "Fp16", "Fp32", "Fp64"):
    body = re.search(rf"using {name}ConversionPairs = TypePairList<(.*?)>;", text, re.S).group(1)
    for triple in re.findall(r"TypePair<([^>]+)>", body):
      pairs.append(tuple(dtype[value.strip().removeprefix("vecops::")] for value in triple.split(",")))
  return (tuple((m, n, d, d, d) for m, n in shapes("TransposeCases")
                for d in ("int8", "float16", "bfloat16", "float32", "float64"))
          + tuple((m, n, a, b, c) for m, n in shapes("ConversionCases") for a, b, c in pairs))


def alignment(extent, maximum):
  result = 1
  while result * 2 <= maximum and extent % (result * 2) == 0:
    result *= 2
  return result


def meta_shape(m, n, mode):
  if mode == "Const":
    return m, n
  if mode == "Any":
    return T.dynamic(), T.dynamic()
  return T.dynamic(alignment=alignment(m, 16)), T.dynamic(alignment=alignment(n, 32))


def kernels(compiler, m, n, mode, input_dtype, output_dtype, compute_dtype, implementation="auto", pass_options=None):
  M, N = meta_shape(m, n, mode)
  input_dtype, output_dtype = getattr(vecops, input_dtype), getattr(vecops, output_dtype)
  @T.template
  def dsl(A: T.Tensor((M, N), (N, 1), dtype=input_dtype),
          C: T.Tensor((N, M), (M, 1), dtype=output_dtype, access="output"), repeat: int):
    for iteration in T.serial(repeat, unroll=1):
      T.transpose_copy(A, C, compute_dtype=compute_dtype, implementation=implementation)
      T.compiler_fence()
  Compute = T.DType(getattr(vecops, compute_dtype))
  @T.template(source=ROOT / "benchmarks/dsl_transpose_reference.cpp")
  def cpp(A: T.Tensor((M, N), (N, 1), dtype=input_dtype),
          C: T.Tensor((N, M), (M, 1), dtype=output_dtype, access="output"), repeat: int):
    Compute
  return T.compile(dsl, compiler=compiler, pass_options=pass_options), T.compile(cpp, compiler=compiler)


def numpy_dtype(dtype):
  if dtype == "bfloat16":
    import ml_dtypes
    return np.dtype(ml_dtypes.bfloat16)
  return np.dtype(dtype)


def aligned_empty(shape, dtype):
  dtype = numpy_dtype(dtype)
  size = int(np.prod(shape)) * dtype.itemsize
  storage = np.empty(size + 4096, dtype=np.uint8)
  offset = (-storage.ctypes.data) % 4096
  return np.ndarray(shape, dtype=dtype, buffer=storage, offset=offset)


def calls(kernels_, a, c, repeat):
  return tuple(lambda k=k: k.operator.native(a, c, repeat, **k._core.specialization_values)
               for k in kernels_)


def measure(calls_, repeat, rounds):
  samples = [[] for _ in calls_]
  rng = random.Random(42)
  for call in calls_:
    call()
  for _ in range(rounds):
    order = list(range(len(calls_)))
    rng.shuffle(order)
    for index in order:
      begin = time.perf_counter_ns()
      calls_[index]()
      samples[index].append((time.perf_counter_ns() - begin) / repeat / 1000)
  return tuple(statistics.median(sample) for sample in samples)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-transpose")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--mode", nargs="+", default=["Const", "Any", "Aligned"])
  parser.add_argument("--dtype", nargs="+", help="Filter by input dtype")
  parser.add_argument("--shape", nargs="+", help="Filter shapes, e.g. 1024x1024")
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--cpu", type=int, default=2, help="Pin timing only, not compiler workers")
  parser.add_argument("--cxx", default="g++")
  parser.add_argument("--cc", default="gcc")
  parser.add_argument("--implementation", default="auto", choices=["auto", "vector", "gather", "sme"])
  parser.add_argument("--cache-block", type=int, help="Override CPU memory-planning cache block")
  parser.add_argument("--loop-order", choices=["input", "output"], help="Override microtile loop order")
  parser.add_argument("--no-defer-conversion", action="store_true", help="Disable narrow-bit permutation/wide store motion")
  args = parser.parse_args()
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)
  compiler = vecops.Compiler(target="Native", cc=args.cc, cxx=args.cxx, jobs=args.jobs,
                            cache_dir=args.build_dir / "cache", build_dir=args.build_dir / "build")
  # Compile once per shape-meta/dtype combination; storage-free collection
  # keeps the complete catalog from allocating hundreds of large tensors.
  cases = []
  options = {}
  if args.cache_block is not None:
    options["cpu_transpose_cache_block"] = args.cache_block
  if args.loop_order is not None:
    options["cpu_transpose_loop_order"] = args.loop_order
  if args.no_defer_conversion:
    options["cpu_transpose_defer_conversion"] = False
  with vecops.collect_compile_requests() as collector:
    for m, n, a, c, compute in catalog():
      if args.dtype and a not in args.dtype or args.shape and f"{m}x{n}" not in args.shape:
        continue
      for mode in args.mode:
        pair = kernels(compiler, m, n, mode, a, c, compute, args.implementation, options)
        for kernel in pair:
          kernel.operator(vecops.TensorMeta((m, n), dtype=getattr(vecops, a)),
                          vecops.TensorMeta((n, m), dtype=getattr(vecops, c)), 1,
                          **kernel._core.specialization_values)
        cases.append((m, n, a, c, compute, mode, pair))
  print(f"Preparing {len(collector)} unique artifacts for {len(cases)} cases", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  os.sched_setaffinity(0, {args.cpu})
  stream = args.output.open("w") if args.output else __import__("sys").stdout
  writer = csv.writer(stream)
  writer.writerow(("m", "n", "input_dtype", "output_dtype", "compute_dtype", "meta", "dsl_us", "cpp_us", "dsl_over_cpp"))
  for m, n, ad, cd, compute, mode, pair in cases:
    a = aligned_empty((m, n), ad)
    c = aligned_empty((n, m), cd)
    a[:] = ((np.arange(m*n, dtype=np.int32) % 251) - 125).reshape(m, n)
    ds, cp = calls(pair, a, c, 1)
    cp()
    reference = c.copy()
    ds()
    np.testing.assert_array_equal(c, reference)
    expected = a.astype(numpy_dtype(compute)).astype(c.dtype).T
    np.testing.assert_array_equal(c, expected)
    repeat = max(2, min(10000, 4_000_000 // (m * n)))
    td, tc = measure(calls(pair, a, c, repeat), repeat, args.rounds)
    writer.writerow((m, n, ad, cd, compute, mode, f"{td:.4f}", f"{tc:.4f}", f"{td/tc:.4f}"))
    stream.flush()
    if args.output:
      print(f"{mode} {m}x{n} {ad}->{cd}/{compute}: {td/tc:.3f}x", flush=True)
  if args.output:
    stream.close()


if __name__ == "__main__":
  main()
