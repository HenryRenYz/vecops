# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compare the source-level FMA recipe with the existing vector dot family.

Both generated sources use one frozen package/SDK, explicit vector lowering,
identical inputs and epilogues. This is not a matrix-instruction benchmark or
evidence about cache-sized drivers. Compile everything before paired timing;
retain every raw sample and validate against an independent oracle twice.
"""

import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dsl_snapshot_entry import select_snapshot


def make_template(T, recipe_function, shape, dtype, pipeline, *, recipe, streaming=False):
  m, n, k = shape

  @T.func
  def finish(value, row, column):
    return T.max(value * 0.5 + 1, 0)

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype=dtype),
    B: T.Tensor((k, n), dtype=dtype),
    D: T.Tensor((m, n), dtype="float32"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
    repeat: int,
  ):
    for _ in T.serial(repeat, unroll=1):
      a, b = T.load(A), T.load(B)
      if pipeline == "affine":
        a, b = a * 2 + 1, b * 2 + 1
      seed = T.load(D) if pipeline != "plain" else None
      if recipe and streaming:
        recipe_function((a, b), C, acc=seed, epilogue=finish)
      else:
        if recipe:
          value = recipe_function((a, b), acc=seed)
        else:
          value = T.dot((a, b), C, acc=seed, acc_dtype="float32", implementation="vector")
        T.store(C, T.max(value * 0.5 + 1, 0))

  return kernel


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--shape", nargs="+", default=["4x16x128", "4x19x17", "1x16x257", "3x8x17"])
  parser.add_argument("--dtype", nargs="+", choices=("float32", "float16"), default=["float32", "float16"])
  parser.add_argument("--pipeline", nargs="+", choices=("plain", "seed", "affine"), default=["plain", "seed", "affine"])
  parser.add_argument("--target", default="Native")
  parser.add_argument("--cc", default="gcc")
  parser.add_argument("--cxx", default="g++")
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--rounds", type=int, default=24)
  parser.add_argument("--prepare-only", action="store_true")
  parser.add_argument(
    "--streaming", action="store_true", help="Measure the experimental row/column streaming DSL driver"
  )
  args = parser.parse_args()
  if args.rounds < 4 or args.rounds % 4:
    parser.error("rounds must be a positive multiple of four")
  root = select_snapshot(args.root)
  sys.path.insert(0, str(root / "benchmarks"))
  import numpy as np
  import vecops
  import vecops.language as T
  from dsl_timing import reserve_cpu
  from vecops.language._library.matmul import vector_dot_into, vector_tile_dot

  recipe_function = vector_dot_into if args.streaming else vector_tile_dot

  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=True)
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cflags=("-ffp-contract=off",),
    cache_dir=output / "cache",
    build_dir=output / "build",
  )
  records = []
  with vecops.collect_compile_requests() as collector:
    for text, dtype, pipeline in itertools.product(args.shape, args.dtype, args.pipeline):
      shape = tuple(map(int, text.split("x")))
      if len(shape) != 3 or any(n <= 0 for n in shape):
        parser.error("microtile shape must be positive MxNxK")
      m, n, k = shape
      rng = np.random.default_rng(127)
      a = (rng.integers(-3, 4, size=(m, k)) / 8).astype(dtype)
      b = (rng.integers(-3, 4, size=(k, n)) / 8).astype(dtype)
      d = (rng.integers(-3, 4, size=(m, n)) / 8).astype(np.float32)
      c = np.empty((m, n), np.float32)
      x, y = a, b
      if pipeline == "affine":
        x, y = (a * 2 + 1).astype(dtype), (b * 2 + 1).astype(dtype)
      expected = x.astype(np.float32) @ y.astype(np.float32)
      if pipeline != "plain":
        expected += d
      expected = np.maximum(expected * 0.5 + 1, 0)
      pair, sources = [], []
      for recipe in (False, True):
        kernel = T.compile(
          make_template(T, recipe_function, shape, dtype, pipeline, recipe=recipe, streaming=args.streaming),
          compiler=compiler,
          pass_options={"cpu_ordered_loop_unroll": recipe, "cpu_state_vector_reuse": recipe},
        )
        source = kernel.generated_source
        digest = hashlib.sha256(source.encode()).hexdigest()
        (output / (digest + ".cpp")).write_text(source)
        sources.append(digest)
        kernel.operator(a, b, d, c, 1, **kernel._core.specialization_values)
        pair.append(kernel)
      records.append((shape, dtype, pipeline, (a, b, d, c), expected, pair, sources))
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  if args.prepare_only:
    print(f"Prepared {len(records)} paired vector microtiles; no timing", flush=True)
    return

  cpu = min(os.sched_getaffinity(0))
  rows = []
  with reserve_cpu(cpu):
    os.sched_setaffinity(0, {cpu})
    for shape, dtype, pipeline, operands, expected, pair, sources in records:
      repeats = max(2, min(4000, 2_000_000 // (shape[0] * shape[1] * shape[2])))
      calls = [
        lambda kernel=kernel, operands=operands, repeats=repeats: kernel.operator.native(
          *operands, repeats, **kernel._core.specialization_values
        )
        for kernel in pair
      ]
      for call in calls:
        call()
        np.testing.assert_array_equal(operands[-1], expected)
      samples, raw = [[], []], []
      for round_ in range(args.rounds):
        for label in ((0, 1), (1, 0), (1, 0), (0, 1))[round_ % 4]:
          begin = time.perf_counter_ns()
          calls[label]()
          end = time.perf_counter_ns()
          samples[label].append(end - begin)
          raw.append({"round": round_, "label": label, "begin_ns": begin, "end_ns": end})
      for call in calls:
        call()
        np.testing.assert_array_equal(operands[-1], expected)
      medians = [statistics.median(sample) for sample in samples]
      row = {
        "shape": shape,
        "dtype": dtype,
        "pipeline": pipeline,
        "sources": sources,
        "repeat": repeats,
        "median_ns": medians,
        "recipe_over_legacy": medians[1] / medians[0],
        "raw": raw,
      }
      rows.append(row)
      print(shape, dtype, pipeline, row["recipe_over_legacy"], flush=True)
  (output / "results.json").write_text(
    json.dumps(
      {
        "root": str(root),
        "native": vecops._C.__file__,
        "target": args.target,
        "streaming": args.streaming,
        "cpu": cpu,
        "rows": rows,
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
