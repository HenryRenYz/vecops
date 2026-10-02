"""Single-thread CPU DSL / existing C++ ops comparison, excluding JIT time.

Run: conda run -n af3 python benchmarks/dsl_norm.py
Pin the process with taskset for reproducible measurements. Both vecops paths
use the same native ABI dispatcher (not Python output allocation). This is a
microbenchmark, not a claim about multi-threaded model inference.
"""
import argparse
import statistics
import random
import timeit
from pathlib import Path

import numpy as np
import torch
import vecops
import vecops.language as T

ROOT = Path(__file__).resolve().parents[1]


def dsl_kernel(compiler, M, N, which, variance="moments", serial_unroll=2, reduction_unroll=2, full_chunks=True, chunk_pow=1):
  @T.template
  def kernel(A: T.Tensor((M, N), dtype=T.float32),
             C: T.Tensor((M, N), dtype=T.float32, access="output")):
    for row in T.serial(M, unroll=serial_unroll):
      x = T.load(A[row, :])
      if which == "layernorm":
        y = T.layernorm(x, variance=variance, unroll=reduction_unroll)
      else:
        y = T.softmax(x, unroll=reduction_unroll)
      T.store(C[row, :], y)
  return T.compile(kernel, compiler=compiler, pass_options={"cpu_full_chunks": full_chunks, "cpu_math_chunk_pow": chunk_pow})


def cpp_kernel(compiler, M, N, which):
  @T.jit(compiler=compiler)
  def factory(B, D, Op, ComputeType):
    @T.template(source=ROOT / "benchmarks/dsl_norm_reference.cpp", dependencies=[ROOT / "tests/runtime/NormOperatorKernel.cpp"])
    def kernel(input: T.Tensor((B, D), dtype=T.float32),
               scale: T.Tensor((D,), dtype=T.float32) | None,
               bias: T.Tensor((D,), dtype=T.float32) | None,
               output: T.Tensor((B, D), dtype=T.float32, access="output"), eps: float):
      Op
      ComputeType
    return kernel
  return factory(M, N, 1 if which == "layernorm" else 2, T.float32)


def measure(call, iterations):
  for _ in range(20):
    call()
  return statistics.median(timeit.repeat(call, number=iterations, repeat=5)) / iterations * 1e6


def aligned_empty(shape, byte_offset=0):
  """Control page/cache-line alignment without comparing different buffers."""
  storage = np.empty(int(np.prod(shape)) * 4 + 4096 + byte_offset, dtype=np.uint8)
  offset = (-storage.ctypes.data) % 4096 + byte_offset
  return np.ndarray(shape, dtype=np.float32, buffer=storage, offset=offset)


def measure_interleaved(calls, iterations, rounds=7):
  """Warm all paths, then randomize order to reduce frequency/thermal bias."""
  samples = [[] for _ in calls]
  order = random.Random(42)
  for call in calls:
    for _ in range(20):
      call()
  for _ in range(rounds):
    indices = list(range(len(calls)))
    order.shuffle(indices)
    for index in indices:
      samples[index].append(timeit.timeit(calls[index], number=iterations) / iterations * 1e6)
  return [statistics.median(values) for values in samples]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--rows", type=int, default=64)
  parser.add_argument("--widths", type=int, nargs="+", default=[128, 512, 1024, 4096])
  parser.add_argument("--iterations", type=int, default=300)
  parser.add_argument("--input-offset", type=int, choices=[0, 16], default=0,
                      help="input byte offset from a page-aligned buffer")
  parser.add_argument("--output-offset", type=int, choices=[0, 16], default=0,
                      help="shared vecops output byte offset from a page-aligned buffer")
  parser.add_argument("--serial-unroll", type=int, default=2)
  parser.add_argument("--reduction-unroll", type=int, default=2)
  parser.add_argument("--no-full-chunks", action="store_true")
  parser.add_argument("--math-chunk-pow", type=int, default=1)
  parser.add_argument("--variance", choices=["moments", "centered"], default="moments",
                      help="moments matches existing C++ LayerNorm; centered is the stable DSL default")
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-bench")
  args = parser.parse_args()
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)
  compiler = vecops.Compiler(target="Native", cc="gcc", cxx="g++", jobs=4,
                            cache_dir=args.build_dir / "cache", build_dir=args.build_dir / "build")
  rng = np.random.default_rng(42)
  print("operation,rows,width,input_offset,output_offset,dsl_us,cpp_us,torch_us,dsl_over_cpp")
  for which in ("layernorm", "softmax"):
    for width in args.widths:
      a = aligned_empty((args.rows, width), args.input_offset)
      a[:] = rng.normal(size=a.shape)
      y = aligned_empty(a.shape, args.output_offset)
      dsl = dsl_kernel(compiler, args.rows, width, which, args.variance,
                       args.serial_unroll, args.reduction_unroll, not args.no_full_chunks, args.math_chunk_pow)
      cpp = cpp_kernel(compiler, args.rows, width, which)
      specs = cpp._core.specialization_values
      call_dsl = lambda: dsl.operator.native(a, y)
      call_cpp = lambda: cpp.operator.native(a, None, None, y, 1e-5, **specs)
      call_cpp()
      reference = y.copy()
      call_dsl()
      np.testing.assert_allclose(y, reference, atol=5e-6, rtol=5e-5)
      ta = torch.from_numpy(a)
      call_torch = (lambda: torch.nn.functional.layer_norm(ta, (width,))) if which == "layernorm" else (lambda: torch.softmax(ta, -1))
      td, tc, tt = measure_interleaved((call_dsl, call_cpp, call_torch), args.iterations)
      label = f"{which}-{args.variance}" if which == "layernorm" else which
      print(f"{label},{args.rows},{width},{args.input_offset},{args.output_offset},{td:.3f},{tc:.3f},{tt:.3f},{td / tc:.3f}", flush=True)


if __name__ == "__main__":
  main()
