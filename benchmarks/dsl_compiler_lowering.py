# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Measure fixed dot/matmul source lowering, including outlined SIMD fallbacks.

Each sample starts with a fresh TemplateIR. Native compilation and execution
are excluded. This is a compiler regression benchmark, not a configuration
search; compare the same case/options/CPU in immutable source snapshots.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path


def declaration(T, tiled):
  m, n, k = T.Dynamic(name="m"), T.Dynamic(name="n"), T.Dynamic(name="k")

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), "bfloat16"),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), "bfloat16"),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), "float32", access="output"),
  ):
    if tiled:
      value = T.matmul((A, B), C)
    else:
      value = T.dot((A, B), C)
    T.store(C, value)

  return kernel.ir


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--snapshot", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--target", default="AVX512", choices=("AVX512", "SME", "Native"))
  parser.add_argument("--case", default="dot", choices=("dot", "matmul"))
  parser.add_argument("--outline-vector-fallback", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--repeat", type=int, default=1)
  parser.add_argument("--cpu", type=int, help="pin lowering to one allowed CPU")
  args = parser.parse_args()
  if args.repeat < 1:
    parser.error("--repeat must be positive")
  if args.snapshot is not None:
    from dsl_snapshot_entry import select_snapshot
    select_snapshot(args.snapshot)
  if args.cpu is not None:
    os.sched_setaffinity(0, {args.cpu})

  import vecops.language as T
  from vecops._identity import implementation_identity
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  affinity = sorted(os.sched_getaffinity(0))
  def frequency():
    if len(affinity) != 1:
      return None
    root = Path(f"/sys/devices/system/cpu/cpu{affinity[0]}/cpufreq")
    return {name: (root / name).read_text().strip() for name in
            ("scaling_max_freq", "scaling_min_freq", "scaling_governor") if (root / name).exists()}

  identity, frequency_before = implementation_identity(), frequency()
  samples = []
  for _ in range(args.repeat):
    template = declaration(T, args.case == "matmul")
    begin = time.perf_counter_ns()
    artifact = CPUBackend().lower(template, (), target=args.target,
                                  options={"cpu_outline_vector_fallback": args.outline_vector_fallback})
    elapsed = time.perf_counter_ns() - begin
    source = artifact.source.encode()
    samples.append(dict(lowering_ns=elapsed, source_bytes=len(source), source_sha256=hashlib.sha256(source).hexdigest()))
  result = dict(case=args.case, target=args.target, outline_vector_fallback=args.outline_vector_fallback,
                machine=platform.machine(), affinity=affinity, implementation=identity,
                frequency_before=frequency_before, frequency_after=frequency(), native_compilation=False,
                median_lowering_ns=statistics.median(row["lowering_ns"] for row in samples), samples=samples)
  text = json.dumps(result, indent=2) + "\n"
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
  print(text, end="")


if __name__ == "__main__":
  main()
