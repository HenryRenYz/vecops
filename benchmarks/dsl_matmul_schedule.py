"""Same-round cache/online-packing schedule experiments against public C++.

Every variant includes online packing inside the native repeat loop. This is
not a prepacked throughput benchmark. Generated sources are recorded so a
winning schedule can be inspected and reproduced before changing defaults.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments, default_cpu, packing_geometry
from dsl_matmul import ROOT, kernels
from dsl_timing import isolate_compilation, reserve_cpu
from dsl_transpose import aligned_empty, measure


def variant(compiler, m, n, k, mode, schedule, blocks):
  M, N, K = (m, n, k) if mode == "Const" else (T.dynamic(lower=0) for _ in range(3))
  BM, BN, BK = (T.Const(v) if v is not None else None for v in blocks)
  format_ = packing_geometry("bfloat16")[0]

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), "bfloat16"),
    B: T.Tensor((K, N), (1, K), "bfloat16"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
    D: T.Tensor((M, N), dtype="float32"),
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      a = T.load(A)
      b = T.load(B)
      if schedule == "pack_ab":
        a = T.pack(a, side="A", format=format_)
      if schedule == "pack_b" or schedule == "pack_ab":
        b = T.pack(b, side="B", format=format_)
      if schedule == "dot" or schedule == "pack_b" or schedule == "pack_ab":
        T.dot((a, b), C)
      else:
        T.library.matmul((a, b), C, BM=BM, BN=BN, BK=BK)
      T.compiler_fence()

  return T.compile(kernel, compiler=compiler)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shape", nargs="+", default=["128x4096x4096"])
  parser.add_argument("--mode", choices=("Const", "Dynamic"), default="Const")
  parser.add_argument("--rounds", type=int, default=25)
  parser.add_argument("--repeat", type=int, default=8)
  parser.add_argument("--jobs", type=int, default=6)
  parser.add_argument("--cpu", type=int, default=default_cpu())
  add_toolchain_arguments(parser)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--compact", action="store_true", help="Compare only auto, C++ and the previous B-panel budget")
  parser.add_argument("--block-m", nargs="+", type=int, default=[32, 64, 128])
  parser.add_argument("--block-n", nargs="+", type=int, default=[32, 64, 128, 256])
  args = parser.parse_args()
  isolate_compilation(args.cpu)
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
  cases, sources = [], {}
  with vecops.collect_compile_requests() as collector:
    for shape in args.shape:
      m, n, k = map(int, shape.split("x"))
      pair, _ = kernels(compiler, m, n, k, args.mode, "bf16", "plain", "matmul")
      schedules = [("auto", pair[0]), ("cpp", pair[1])]
      if args.compact:
        previous = T.cost.CostProfile("previous-panel-budget", policies=(("amx_cache_b_panel_divisor", 0),))
        old_pair, _ = kernels(compiler, m, n, k, args.mode, "bf16", "plain", "matmul", {"cost_profiles": (previous,)})
        schedules.append(("previous_budget", old_pair[0]))
      else:
        for bm in args.block_m:
          for bn in args.block_n:
            name = f"blocks_{bm}_{bn}_{k}"
            schedules.append((name, variant(compiler, m, n, k, args.mode, "blocks", (bm, bn, k))))
        for bk in (512, 1024):
          name = f"blocks_128_128_{bk}"
          schedules.append((name, variant(compiler, m, n, k, args.mode, "blocks", (128, 128, bk))))
        for name in ("dot", "pack_b", "pack_ab"):
          schedules.append((name, variant(compiler, m, n, k, args.mode, name, (None, None, None))))
        schedules.append(("full", variant(compiler, m, n, k, args.mode, "blocks", (m, n, k))))
      meta = (
        vecops.TensorMeta((m, k), dtype=vecops.bfloat16),
        vecops.TensorMeta((k, n), strides=(1, k), dtype=vecops.bfloat16),
        vecops.TensorMeta((m, n), dtype=vecops.float32),
        vecops.TensorMeta((m, n), dtype=vecops.float32),
      )
      for name, kernel in schedules:
        kernel.operator(*meta, 1, **kernel._core.specialization_values)
        sources[f"{shape}/{args.mode}/{name}"] = str(kernel.source)
      cases.append((m, n, k, schedules))
  args.output.with_suffix(".json").write_text(json.dumps(sources, indent=2) + "\n")
  print(f"{len(collector)} artifacts", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  with reserve_cpu(args.cpu), args.output.open("w") as stream:
    os.sched_setaffinity(0, {args.cpu})
    writer = csv.writer(stream)
    writer.writerow(("m", "n", "k", "mode", "schedule", "us", "over_auto", "over_cpp"))
    rng = np.random.default_rng(17)
    for m, n, k, schedules in cases:
      a, b = aligned_empty((m, k), "bfloat16"), aligned_empty((n, k), "bfloat16").T
      c, d = aligned_empty((m, n), "float32"), aligned_empty((m, n), "float32")
      a[:] = rng.integers(-3, 4, size=a.shape) / 8
      b[:] = rng.integers(-3, 4, size=b.shape) / 8
      d.fill(0)

      def calls(repeat, a=a, b=b, c=c, d=d, schedules=schedules):
        return tuple(
          lambda kernel=kernel: kernel.operator.native(a, b, c, d, repeat, **kernel._core.specialization_values)
          for _, kernel in schedules
        )

      checks = calls(1)
      checks[1]()
      expected = c.copy()
      for check in checks:
        check()
        np.testing.assert_allclose(c, expected, rtol=5e-3, atol=3e-4)
      np.testing.assert_allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=5e-3, atol=3e-4)
      times = measure(calls(args.repeat), args.repeat, args.rounds)
      for (name, _), elapsed in zip(schedules, times):
        writer.writerow((m, n, k, args.mode, name, elapsed, elapsed / times[0], elapsed / times[1]))
        stream.flush()
        print(
          f"{m}x{n}x{k} {name}: {elapsed:.1f} us, {elapsed / times[0]:.3f} auto, {elapsed / times[1]:.3f} C++",
          flush=True,
        )


if __name__ == "__main__":
  main()
