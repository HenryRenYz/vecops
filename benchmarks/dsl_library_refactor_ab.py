"""Paired normalization refactor checks from two frozen source snapshots.

Emit source/schema catalogs in separate fresh Python processes, then build
and time both generated C++ variants together on the same allocated CPU.
No editable install is replaced. All individual timing observations are saved.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from dsl_snapshot_entry import select_snapshot


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--emit", type=Path)
  parser.add_argument("--baseline", type=Path)
  parser.add_argument("--candidate", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--shape", nargs="+", default=["1x4096", "128x4096", "4096x4096"])
  parser.add_argument("--dtype", nargs="+", default=["float32", "float16", "bfloat16", "float64"])
  parser.add_argument("--affine", nargs="+", default=["gamma_beta", "none"])
  parser.add_argument("--target", default="Native")
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--rounds", type=int, default=12)
  parser.add_argument("--prepare-only", action="store_true", help="Build both variants without starting any timing")
  args = parser.parse_args()
  args.output = args.output.resolve()
  root = select_snapshot(args.root)
  sys.path.insert(0, str(root / "benchmarks"))
  import numpy as np
  import vecops
  import vecops.language as T
  from dsl_norm_catalog import Case, calls, catalog, metadata, validate
  from dsl_norm_library import library_kernel

  assert Path(vecops.__file__).resolve() == root / "python/vecops/__init__.py"
  args.output.mkdir(parents=True, exist_ok=True)
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cflags=("-ffp-contract=off",),
    cache_dir=args.output / "cache",
    build_dir=args.output / "build",
  )
  if args.emit:
    selected = [
      c
      for c in catalog(variance=("moments", "centered"))
      if "x".join(map(str, c.shape)) in args.shape
      and c.dtype in args.dtype
      and c.affine in args.affine
      and len(c.shape) == 2
      and c.accuracy == "strict"
    ]
    records = []
    for case in selected:
      kernel = library_kernel(compiler, case)
      records.append({"case": case.__dict__, "source": kernel.generated_source})
    if not records:
      parser.error("no original catalog metadata selected")
    args.emit.write_text(json.dumps({"root": str(root), "native": vecops._C.__file__, "records": records}))
    print(f"Emitted {len(records)} original metadata from {root}", flush=True)
    return
  if not args.baseline or not args.candidate or args.rounds < 4 or args.rounds % 4:
    parser.error("timing requires two catalogs and a positive multiple of four rounds")
  before, after = (json.loads(p.read_text())["records"] for p in (args.baseline, args.candidate))
  assert [r["case"] for r in before] == [r["case"] for r in after]
  compiled, by_source = [], {}
  with vecops.collect_compile_requests() as collector:
    for old, new in zip(before, after):
      case = Case(**{**old["case"], "shape": tuple(old["case"]["shape"])})
      pair = []
      for record in (old, new):
        digest = hashlib.sha256(record["source"].encode()).hexdigest()
        if digest not in by_source:
          source = args.output / (digest + ".cpp")
          source.write_text(record["source"])
          shape = case.shape if case.meta == "Const" else tuple(T.dynamic(lower=0) for _ in case.shape)
          dtype = T.DType(case.dtype)
          width = shape[-1]

          @T.template(source=source)
          def frozen(
            A: T.Tensor(shape, dtype=dtype),
            G: T.Tensor((width,), dtype=dtype),
            B: T.Tensor((width,), dtype=dtype),
            C: T.Tensor(shape, dtype=dtype, access="output"),
            repeat: int,
          ):
            pass

          kernel = T.compile(frozen, compiler=compiler)
          kernel.operator(*metadata(case), 1, **kernel._core.specialization_values)
          by_source[digest] = kernel
        pair.append(by_source[digest])
      compiled.append((case, tuple(pair)))
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  if args.prepare_only:
    print(f"Prepared {len(compiled)} original cases / {len(by_source)} generated sources; no timing", flush=True)
    return
  affinity = sorted(os.sched_getaffinity(0))
  os.sched_setaffinity(0, {affinity[0]})
  rows = []
  for case, pair in compiled:
    # Preserve the original input family and the independent NumPy oracle.
    operands = validate(case, (pair[1], pair[0]))
    invocations = calls(pair, operands, max(1, min(512, 4_000_000 // math.prod(case.shape))))
    for invoke in invocations:
      invoke()
    samples = [[], []]
    raw = []
    for round_ in range(args.rounds):
      for label in ((0, 1), (1, 0), (1, 0), (0, 1))[round_ % 4]:
        begin = time.perf_counter_ns()
        invocations[label]()
        end = time.perf_counter_ns()
        samples[label].append(end - begin)
        raw.append({"round": round_, "label": label, "begin_ns": begin, "end_ns": end})
    medians = [statistics.median(s) for s in samples]
    rows.append({"case": case.__dict__, "median_ns": medians, "new_over_old": medians[1] / medians[0], "raw": raw})
    print(case.key, rows[-1]["new_over_old"], flush=True)
    # Both variants must still agree after all repeated invocations.
    invocations[0]()
    expected = operands[-1].copy()
    invocations[1]()
    np.testing.assert_allclose(operands[-1].astype(np.float64), expected.astype(np.float64), rtol=4e-5, atol=4e-5)
  (args.output / "results.json").write_text(json.dumps({"cpu": affinity[0], "rows": rows}, indent=2))


if __name__ == "__main__":
  main()
