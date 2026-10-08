# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Controlled two/three/four-load CPU pointwise register-layout A/B.

Off/auto/force use one source, operator, schema, strict arithmetic and inputs.
The repeat loop lives inside each compiled kernel and includes the complete
pointwise operator plus a compiler fence. This is a policy diagnostic, not a
comparison to an opaque C++ operator; dsl_norm_library supplies matched C++
LayerNorm comparisons using the maintained benchmark metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments
from dsl_norm_catalog import aligned_empty, calls, measure

ROOT = Path(__file__).resolve().parents[1]
KEY_FIELDS = ("shape", "dtype", "meta", "inputs")


def pass_options(mode):
  if mode == "force":
    return {"cpu_pointwise_register_layout": True}
  return {"cpu_pointwise_register_layout_policy": mode}


def kernel(compiler, shape, dtype, meta, inputs, mode):
  Shape = shape if meta == "Const" else (T.dynamic(lower=0), T.dynamic(lower=0))
  N = Shape[-1]

  @T.template
  def controlled(
    A: T.Tensor(Shape, dtype=dtype),
    G: T.Tensor((N,), dtype=dtype),
    B: T.Tensor((N,), dtype=dtype),
    D: T.Tensor((N,), dtype=dtype),
    C: T.Tensor(Shape, dtype=dtype, access="output"),
    repeat: int,
  ):
    for iteration in T.serial(repeat, unroll=1):
      y = (T.load(A, dtype="float32") - 0.25) * 0.5
      y = y * T.load(G, dtype="float32")
      if inputs >= 3:
        y = y + T.load(B, dtype="float32")
      if inputs >= 4:
        y = y * T.load(D, dtype="float32")
      T.store(C, y)
      T.compiler_fence()

  return T.compile(controlled, compiler=compiler, pass_options=pass_options(mode))


def operands(shape, dtype, inputs):
  a = aligned_empty(shape, dtype)
  g, b, d = (aligned_empty((shape[-1],), dtype) for _ in range(3))
  indices = np.arange(math.prod(shape), dtype=np.int32)
  a[:] = ((indices % 37 - 18).astype(np.float32) * np.float32(0.07)).reshape(shape)
  axis = np.arange(shape[-1], dtype=np.int32)
  g[:] = np.float32(0.75) + (axis % 257).astype(np.float32) * np.float32(0.001)
  b[:] = np.float32(-0.2) + (axis % 263).astype(np.float32) * np.float32(0.0007)
  d[:] = np.float32(0.5) + (axis % 241).astype(np.float32) * np.float32(0.002)
  y = (a.astype(np.float32) - np.float32(0.25)) * np.float32(0.5)
  y = y * g.astype(np.float32)
  if inputs >= 3:
    y = y + b.astype(np.float32)
  if inputs >= 4:
    y = y * d.astype(np.float32)
  expected = y.astype(a.dtype)
  c = aligned_empty(shape, dtype)
  return (a, g, b, d, c), expected


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_toolchain_arguments(parser)
  parser.add_argument("--shape", nargs="+", default=("1x4096", "128x4096", "4096x4096"))
  parser.add_argument("--dtype", nargs="+", choices=("float16", "bfloat16"), default=("float16", "bfloat16"))
  parser.add_argument("--inputs", nargs="+", type=int, choices=(2, 3, 4), default=(2, 3, 4))
  parser.add_argument("--meta", nargs="+", choices=("Any", "Const"), default=("Any",))
  parser.add_argument("--mode", nargs="+", choices=("off", "auto", "force"), default=("off", "auto"))
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--rounds", type=int, default=9)
  parser.add_argument("--cpu", type=int)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-register-layout")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args()
  try:
    shapes = tuple(tuple(map(int, shape.split("x"))) for shape in args.shape)
  except ValueError:
    parser.error("--shape requires rows x width, both positive integer extents")
  if any(len(shape) != 2 or any(extent < 1 for extent in shape) for shape in shapes):
    parser.error("--shape requires two positive extents")
  for name in ("shape", "dtype", "inputs", "meta", "mode"):
    if len(getattr(args, name)) != len(set(getattr(args, name))):
      parser.error(f"--{name} contains duplicate metadata")
  if "off" not in args.mode or args.rounds < 1 or args.jobs < 1:
    parser.error("include the off baseline and positive --rounds/--jobs")
  cases = tuple(
    (shape, dtype, meta, count)
    for shape in shapes
    for dtype in args.dtype
    for meta in args.meta
    for count in args.inputs
  )
  if args.list:
    writer = csv.writer(sys.stdout)
    writer.writerow(KEY_FIELDS)
    writer.writerows(("x".join(map(str, shape)), dtype, meta, count) for shape, dtype, meta, count in cases)
    return
  if "bfloat16" in args.dtype and importlib.util.find_spec("ml_dtypes") is None:
    parser.error("BF16 requires the optional ml_dtypes package")
  allowed = os.sched_getaffinity(0)
  cpu = min(allowed) if args.cpu is None else args.cpu
  if cpu not in allowed:
    parser.error(f"CPU {cpu} is not in the current allowed affinity {sorted(allowed)}")
  if args.output is not None and args.output.exists():
    parser.error("select a fresh output path; existing timing evidence is not overwritten")
  manifest_path = args.manifest or (args.output.with_suffix(".manifest.json") if args.output else None)
  if manifest_path is not None and manifest_path.exists():
    parser.error("select a fresh manifest path")
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  prepared, shared = [], {}
  with vecops.collect_compile_requests() as collector:
    for shape, dtype, meta, count in cases:
      variants = []
      for mode in args.mode:
        schema = (shape if meta == "Const" else len(shape), dtype, meta, count, mode)
        if schema not in shared:
          compiled = kernel(compiler, shape, dtype, meta, count, mode)
          compiled.operator(
            vecops.TensorMeta(shape, dtype=dtype),
            *(vecops.TensorMeta((shape[-1],), dtype=dtype) for _ in range(3)),
            vecops.TensorMeta(shape, dtype=dtype),
            1,
            **compiled._core.specialization_values,
          )
          shared[schema] = compiled
        variants.append(shared[schema])
      prepared.append(((shape, dtype, meta, count), variants))
  manifest = {
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_root": str(ROOT),
    "build_dir": str(args.build_dir.resolve()),
    "compiler": {"cc": args.cc, "cxx": args.cxx, "target": args.target},
    "cpu": cpu,
    "rounds": args.rounds,
    "modes": args.mode,
    "pass_options": {mode: pass_options(mode) for mode in args.mode},
    "logical_requests": len(collector),
    "cases": [list(case) for case in cases],
    "comparison": "same-source DSL off/auto; not a C++ operator timing",
  }
  print("Register-layout manifest: " + json.dumps(manifest, sort_keys=True), flush=True)
  if manifest_path is not None:
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  stream = args.output.open("w", newline="") if args.output else sys.stdout
  fields = (
    *KEY_FIELDS,
    *(f"{mode}_us" for mode in args.mode),
    *(f"{mode}_over_off" for mode in args.mode if mode != "off"),
  )
  try:
    os.sched_setaffinity(0, {cpu})
    writer = csv.writer(stream)
    writer.writerow(fields)
    for (shape, dtype, meta, count), variants in prepared:
      own, expected = operands(shape, dtype, count)
      for execute in calls(variants, own, 1):
        execute()
        np.testing.assert_array_equal(
          own[-1].view(np.uint16), expected.view(np.uint16), err_msg=f"{shape=} {dtype=} {meta=} {count=}"
        )
      repeat = max(2, min(10000, 4_000_000 // math.prod(shape)))
      timings = tuple(duration / repeat for duration in measure(calls(variants, own, repeat), args.rounds))
      off = timings[args.mode.index("off")]
      writer.writerow(
        (
          "x".join(map(str, shape)),
          dtype,
          meta,
          count,
          *(f"{duration:.6f}" for duration in timings),
          *(f"{duration / off:.6f}" for mode, duration in zip(args.mode, timings) if mode != "off"),
        )
      )
      stream.flush()
  finally:
    os.sched_setaffinity(0, allowed)
    if args.output:
      stream.close()


if __name__ == "__main__":
  main()
