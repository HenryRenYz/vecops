# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compare frozen DSL math sources and C++ operators with their own SDKs.

Emit each snapshot in a fresh process, then compile both SDK/source pairs and
time them in alternating order on one CPU. Softmax cases come from the
maintained dsl_norm_catalog/SoftmaxBench metadata, including its input family.
No editable installation or global environment is replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--emit", type=Path)
  parser.add_argument("--baseline", type=Path)
  parser.add_argument("--candidate", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--shape", nargs="+", default=["64", "8x128", "64x513", "4096x1024", "1x154880"])
  parser.add_argument("--dtype", nargs="+", default=["float32"])
  parser.add_argument("--accuracy", nargs="+", default=["estimate"])
  parser.add_argument("--sigmoid", action="store_true")
  parser.add_argument("--cc", default="gcc")
  parser.add_argument("--cxx", default="g++")
  parser.add_argument("--cflag", action="append", default=[])
  parser.add_argument("--target", default="Native")
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--rounds", type=int, default=12)
  parser.add_argument("--prepare-only", action="store_true")
  parser.add_argument("--cpu", type=int, required=True)
  args = parser.parse_args()
  args.output = args.output.resolve()
  from dsl_snapshot_entry import select_snapshot

  root = select_snapshot(args.root)
  sys.path.insert(0, str(root / "benchmarks"))
  import numpy as np
  import vecops
  import vecops.language as T
  from dsl_norm_catalog import Case, aligned_empty, calls, catalog, kernels, metadata, validate
  from dsl_norm_library import library_kernel

  args.output.mkdir(parents=True, exist_ok=True)
  if args.emit:
    compiler = vecops.Compiler(target=args.target, cc=args.cc, cxx=args.cxx, cflags=tuple(args.cflag))
    records = []
    for case in catalog():
      if (case.operation != "softmax" or case.meta != "Const"
          or "x".join(map(str, case.shape)) not in args.shape
          or case.dtype not in args.dtype or case.accuracy not in args.accuracy):
        continue
      for scope in ("row", "direct") if len(case.shape) == 2 else ("row",):
        kernel = library_kernel(compiler, case) if scope == "direct" else kernels(compiler, case)[0]
        records.append({"case": asdict(case), "scope": scope, "source": kernel.generated_source})
    if args.sigmoid:
      for dtype in args.dtype:
        for accuracy in args.accuracy:
          for algorithm in ("exp", "polynomial"):
            DType, Mode, Algorithm = T.DType(dtype), accuracy, algorithm
            Shape = (4096,)

            @T.template
            def sigmoid(A: T.Tensor(Shape, dtype=DType), G: T.Tensor((4096,), dtype=DType),
                        B: T.Tensor((4096,), dtype=DType),
                        C: T.Tensor(Shape, dtype=DType, access="output"), repeat: int):
              for i in T.serial(repeat, unroll=1):
                T.store(C, T.library.stable_sigmoid(T.load(A), accuracy=Mode, algorithm=Algorithm))
                T.compiler_fence()

            kernel = T.compile(sigmoid, compiler=compiler)
            case = Case("sigmoid", Shape, dtype, "Const", "none", accuracy, algorithm)
            records.append({"case": asdict(case), "scope": "elementwise", "source": kernel.generated_source})
    if not records:
      parser.error("no original benchmark cases selected")
    args.emit.write_text(json.dumps({"root": str(root), "records": records}))
    print(f"Emitted {len(records)} math sources from {root}", flush=True)
    return

  if not args.baseline or not args.candidate or args.rounds < 4 or args.rounds % 4:
    parser.error("comparison requires two catalogs and rounds divisible by four")
  snapshots = [json.loads(p.read_text()) for p in (args.baseline, args.candidate)]
  before, after = [s["records"] for s in snapshots]
  assert [(r["case"], r["scope"]) for r in before] == [(r["case"], r["scope"]) for r in after]

  class SnapshotCompiler(vecops.Compiler):
    def _native_config(self):
      config = super()._native_config()
      installed = self.sdk_root / "python/vecops/_sdk"
      if (installed / "lib/cmake/Vecops/VecopsConfig.cmake").is_file():
        config.sdk.path = installed.resolve()
        config.sdk.layout = vecops._C.SdkLayout.package
      else:
        config.sdk.path = self.sdk_root
        config.sdk.layout = vecops._C.SdkLayout.source_tree
      return config

  compilers = []
  for label, snapshot in zip(("baseline", "candidate"), snapshots):
    compiler = SnapshotCompiler(target=args.target, cc=args.cc, cxx=args.cxx, jobs=args.jobs,
                                cflags=("-ffp-contract=off", *args.cflag),
                                cache_dir=args.output / label / "cache", build_dir=args.output / label / "build")
    compiler.sdk_root = Path(snapshot["root"])
    compilers.append(compiler)

  def cpp_kernel(compiler, case):
    # Use the catalog's exact reference and captures without lowering an
    # unused DSL program. Large source recipes make that wasted work costly.
    Shape, DType = case.shape, T.DType(case.dtype)
    Width = Shape[-1]
    ComputeType = T.DType("float64" if case.dtype == "float64" else "float32")
    Op = T.Const(2)
    Accuracy = T.Const(("strict", "fast", "estimate").index(case.accuracy))
    HasGamma, HasBeta = T.Const(False), T.Const(False)

    @T.template(source=root / "benchmarks/dsl_norm_catalog_reference.cpp")
    def cpp(A: T.Tensor(Shape, dtype=DType), G: T.Tensor((Width,), dtype=DType),
            B: T.Tensor((Width,), dtype=DType), C: T.Tensor(Shape, dtype=DType, access="output"), repeat: int):
      Op  # noqa: B018 - source-kernel specialization captures
      Accuracy  # noqa: B018
      ComputeType  # noqa: B018
      HasGamma  # noqa: B018
      HasBeta  # noqa: B018

    return T.compile(cpp, compiler=compiler)

  prepared = []
  with vecops.collect_compile_requests() as collector:
    for old, new in zip(before, after):
      case = Case(**{**old["case"], "shape": tuple(old["case"]["shape"])})
      pair = []
      for record, compiler, label in zip((old, new), compilers, ("baseline", "candidate")):
        digest = hashlib.sha256(record["source"].encode()).hexdigest()
        path = args.output / f"{label}-{digest}.cpp"
        path.write_text(record["source"])
        Shape, DType = case.shape, T.DType(case.dtype)
        Width = Shape[-1]

        @T.template(source=path)
        def frozen(A: T.Tensor(Shape, dtype=DType), G: T.Tensor((Width,), dtype=DType),
                   B: T.Tensor((Width,), dtype=DType), C: T.Tensor(Shape, dtype=DType, access="output"), repeat: int):
          pass

        dsl = T.compile(frozen, compiler=compiler)
        dsl.operator(*metadata(case), 1, **dsl._core.specialization_values)
        pair.append(dsl)
      prepared.append((case, old["scope"], "dsl", pair))
      if old["scope"] == "row" and case.operation == "softmax":
        pair = [cpp_kernel(c, case) for c in compilers]
        for kernel in pair:
          kernel.operator(*metadata(case), 1, **kernel._core.specialization_values)
        prepared.append((case, old["scope"], "cpp", pair))
  print(f"Building {len(collector.requests)} snapshot-specific kernels", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  if args.prepare_only:
    print("Prepared both snapshots; no timings collected", flush=True)
    return
  allowed = os.sched_getaffinity(0)
  if args.cpu not in allowed:
    parser.error("requested CPU is outside the process affinity")
  os.sched_setaffinity(0, {args.cpu})
  rows = []
  with (args.output / "raw.jsonl").open("w") as stream:
    for case, scope, language, pair in prepared:
      if case.operation == "softmax":
        operands = validate(case, pair)
      else:
        a = aligned_empty(case.shape, case.dtype)
        a[:] = np.linspace(-80, 80, math.prod(case.shape)).reshape(case.shape)
        operands = (a, np.empty_like(a), np.empty_like(a), np.empty_like(a))
        expected = 1 / (1 + np.exp(-a.astype(np.float64)))
        for invoke in calls(pair, operands, 1):
          invoke()
          tolerance = 8e-3 if case.accuracy == "estimate" else 5e-5
          if case.dtype in {"float16", "bfloat16"}:
            tolerance = max(tolerance, 8e-3)
          np.testing.assert_allclose(operands[-1], expected, rtol=tolerance, atol=tolerance)
      repeat = max(1, min(4096, 2_000_000 // math.prod(case.shape)))
      invocations = calls(pair, operands, repeat)
      for invoke in invocations:
        invoke()
      samples = [[], []]
      for round_ in range(args.rounds):
        for label in ((0, 1), (1, 0), (1, 0), (0, 1))[round_ % 4]:
          begin = time.perf_counter_ns()
          invocations[label]()
          end = time.perf_counter_ns()
          ns = (end - begin) / repeat
          samples[label].append(ns)
          stream.write(json.dumps({"case": asdict(case), "scope": scope, "language": language,
                                   "round": round_, "label": label, "repeat": repeat,
                                   "begin_ns": begin, "end_ns": end, "ns": ns}) + "\n")
          stream.flush()
      medians = [statistics.median(s) for s in samples]
      result = {"case": asdict(case), "scope": scope, "language": language,
                "median_ns": medians, "new_over_old": medians[1] / medians[0]}
      rows.append(result)
      print(case.key, scope, language, f'{result["new_over_old"]:.3f}', flush=True)
      (args.output / "results.json").write_text(json.dumps({"cpu": args.cpu, "snapshots": [s["root"] for s in snapshots],
                                                           "rows": rows}, indent=2))


if __name__ == "__main__":
  main()
