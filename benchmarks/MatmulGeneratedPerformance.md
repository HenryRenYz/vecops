<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->

# Generated Matmul measurements

Fixed nested regression cohorts (80 / 320 / 1280 / full), runner commands and strict
comparison rules are documented in [MatmulRegressionSubsets.md](MatmulRegressionSubsets.md).

## Full catalog optimization run, 2026-10-15

This completed run compares generated `T.dot` / `T.library.matmul` and the public
C++ Matmul reference across the enabled registrations in `benchmarks/ops/Matmul*Bench.cpp`.
The immutable historical baseline is `build-matmul-full-20261014/source-final`;
the candidate is `build-matmul-opt-20261015/source-packed-checked`.
Correctness checks and input generation are unchanged. All comparable registrations
passed; unsupported Prepared APIs remain explicit gaps.

| Platform | Main compute | Main packing stages | WeightReuse | Independent MatmulPack | Stateful Legacy | Prepared API gaps |
|---|---:|---:|---:|---:|---:|---:|
| Local x86 | 2,622/2,622 | 842/842 | 394 | 220 | 5 | 5 |
| ARM 920f-4 | 3,856/3,856 | 1,276/1,276 | 442 | 252 | 5 | 5 |

Main packing stages are additional measurements of the same registrations, not
additional independent matrix shapes. The ARM MMLA experiment remains disabled in
the source configuration and is not included in the enabled catalog.

### Main performance

Ratios mean DSL time / C++ time; lower is better. Speedup means old time / new time.
Before/after comparisons require exact execution ID, stage and workload metadata,
and both runs must have passed. Newly recovered cases enter current coverage but
have no invented baseline time. Means below are registration-weighted geometric
means, not application-weighted throughput.

| Platform / stage | Common passed pairs | DSL speedup | C++ speedup | Paired DSL/C++ old → new | All current DSL/C++ geo / median |
|---|---:|---:|---:|---:|---:|
| Local x86 / compute | 2,591 | 2.097x | 0.981x | 5.548 → 2.594 | 2.563 / 2.238 |
| Local x86 / packing | 842 | 1.050x | 0.963x | 4.428 → 4.058 | 4.058 / 3.918 |
| ARM 920f-4 / compute | 3,660 | 1.576x | 0.998x | 7.652 → 4.843 | 4.652 / 2.938 |
| ARM 920f-4 / packing | 1,274 | 0.991x | 0.997x | 1.039 → 1.045 | 1.046 / 1.116 |

### Supplemental performance

Packing, weight-reuse lifecycle and Stateful Legacy are kept separate. WeightReuse
retains the actual R matrix products, with results also grouped by `calls_per_pack`.

| Platform / category | Common passed pairs | DSL speedup | C++ speedup | Paired DSL/C++ old → new | All current DSL/C++ geo |
|---|---:|---:|---:|---:|---:|
| Local x86 / MatmulWeightReuse | 394 | 4.506x | 0.976x | 55.870 → 12.095 | 12.095 |
| Local x86 / MatmulPack | 220 | 1.233x | 0.994x | 6.439 → 5.194 | 5.194 |
| Local x86 / StatefulMatmul | 5 | 1.005x | 1.006x | 2.242 → 2.245 | 2.245 |
| ARM 920f-4 / MatmulWeightReuse | 442 | 4.391x | 1.003x | 40.772 → 9.316 | 9.316 |
| ARM 920f-4 / MatmulPack | 247 | 0.997x | 1.006x | 1.219 → 1.230 | 1.213 |
| ARM 920f-4 / StatefulMatmul | 5 | 1.021x | 1.011x | 3.038 → 3.008 | 3.008 |

### Improvements and remaining limits

This revision restores saturating converting-scatter semantics, preserves valid
online packed bindings, fuses byte B packing with its column compensation, refines
small-problem native dispatch, fuses safe batched result/epilogue stores, and caches
individual affine proof queries. ARM uses a probe-gated workaround for the actual
BiSheng scalable-vector SimplifyCFG crash; both C++ and DSL receive the same policy.
The compiler binary is unchanged, but the policy is an explicit baseline change.

Local x86: 783 common compute registrations have old/new speedup > 1.1; 261 have new/old time > 1.1. The median speedup is 1.007x. ARM 920f-4: 976 common compute registrations have old/new speedup > 1.1; 447 have new/old time > 1.1. The median speedup is 1.004x. These results do not establish universal non-regression or
C++ parity. See the per-case data for large remaining gaps and slowdowns.

An isolated x86 diagnostic explains one quantized packing regression: the same
generated kernel for execution `2997be040868918d5e155f6d`, with only the historical
conversion header substituted, measures 2.100 microseconds versus 5.140 with the
current header. Both pass the unchanged packing-byte oracle. This locates that
case's slowdown in the corrected conversion backend. The historical header lacks
the required exceptional-value semantics and is not a shipping candidate. The
separate diagnostic did not overwrite formal scores, and does not explain every
regression. Full evidence is in `x86-conversion-regression-diagnostic/`.

### Measurement and reproduction

Local x86 uses GCC 13.3 on CPU54. ARM uses `920f-4`, project
`/home/ryz/vecops-neo`, BiSheng Clang 19.1.7 through
`build-math-unify/clangxx-gcc15`, CPU181, SVE VL = SME SVL = 64 B.
Both use the matching API10 extension and frozen source SDK. This task's build and validation
workers exit before each platform's formal timing; resource audits and pre/post
input identities are retained. Each implementation calibrates to at least 25 ms,
then runs 12 balanced native timing rounds. JIT compilation is excluded. ARM is a
shared host: ongoing load observations are retained, including external compiler
activity on other cores; this is a pinned-core run, not an exclusive-machine run.

Build entry scripts are `build-{x86,arm}-packed-checked.sh`; supplemental preparation
uses `supplemental-prepare-{x86,arm}-packed-checked.sh`, under the candidate artifact
root. Exact resumptions and ownership-preserving build-only reschedules are retained
there. After all groups and supplemental preguards pass, `finish_platform.py x86`
or `finish_platform.py arm` performs serial timing, postguards and strict summaries.
On ARM, source `~/.bashrc` and activate the verified `af3` environment first; compiler
paths are explicit in each frozen plan. Do not run the timing controller over a live
build or mix its frozen source tree with the working tree's Python package.

Across platforms, 1,470 identical Automatic logical workloads are
matched, weighting each once and taking the median across registration aliases.
The current DSL/C++ geometric means are 2.327
(x86) and 4.673 (ARM).

![Matched Automatic workload distribution](/home/renyz/project/vecops-neo/build-matmul-opt-20261015/final-matched-matmul.png)

- [Main before/after data, exact paired workloads](/home/renyz/project/vecops-neo/build-matmul-opt-20261015/optimization-comparison.json)
- [Supplemental before/after data and individual registrations](/home/renyz/project/vecops-neo/build-matmul-opt-20261015/supplemental-optimization-comparison.json)
- [Current platform comparison and coverage by geometry/dtype/packing](/home/renyz/project/vecops-neo/build-matmul-opt-20261015/final-platform-comparison.json)
- [Priority-related subsets, restored coverage and regressions](/home/renyz/project/vecops-neo/build-matmul-opt-20261015/priority-results.json)
- [x86 strict summary](/home/renyz/project/vecops-neo/build-matmul-opt-20261015/x86-final-summary/summary.json)
- [ARM strict summary](/home/renyz/project/vecops-neo/build-matmul-opt-20261015/arm-final-summary/summary.json)
- [x86 full timings CSV](/home/renyz/project/vecops-neo/build-matmul-opt-20261015/x86-final-summary/timings.csv)
- [ARM full timings CSV](/home/renyz/project/vecops-neo/build-matmul-opt-20261015/arm-final-summary/timings.csv)

The detailed implementation, targeted regressions, compiler diagnosis, limitations
and exact operational evidence are in
[the analysis report](/home/renyz/project/vecops-neo/docs/internal/matmul-cpp-dsl-analysis-20261013.md).

## Historical selected-registration experiment, 2026-10-11

The starting implementation is commit `f8ebb69`. This experiment compares
generated `T.dot` / `T.library.matmul` with the public C++ `ops::Matmul`, using
the actual registrations parsed from `benchmarks/ops/Matmul*Bench.cpp`.
It covers selected registrations, not the complete catalog.

### Machines and measurement

| Platform | Source root | Compiler | Geometry / timing CPUs |
|---|---|---|---|
| Local Sapphire Rapids | `/home/renyz/project/vecops-neo` | GCC 13.3 | AMX; CPUs 24–26 |
| `920f-4` | `/home/ryz/vecops-neo` | `build-math-unify/clangxx-gcc15`, BiSheng Clang 19.1.7 with GCC 15 libstdc++ | SVE VL = SME SVL = 64 B; CPUs 72–76 |

`dsl_matmul_source_catalog.py` supplies the source shape, metadata, input/output
pipeline, accumulator and packing selectors. Both native implementations first
pass its independent NumPy oracle; integer comparisons are exact. Offline
packers must agree byte-for-byte, including padding. Offline packing is excluded
from compute timing; necessary packing of raw inputs is included.

`dsl_matmul_artifact_ab.py` calibrates each implementation's native repeat to
at least 25 ms, then measures 12 rounds in balanced order. JIT compilation is
excluded. C++ sources, headers, selectors and toolchain are held fixed. Raw
clock readings and executable paths are retained; no frequency correction is
applied. C++ temporary workspace planning occurs once per repeated native call,
matching the existing operator benchmark's amortization over iterations.

The initial 30-row baseline could not compile eight converted-packing rows
because it selected between BF16 and FP32 pointers. Those rows are explicitly
excluded from before/after claims, and pass candidate-versus-C++ validation.
There are 22 usable BF16 before/after rows per platform. Additional measured
coverage is eight mixed-byte rows locally, twenty FP16/FP32/FP64/mixed-byte rows
on ARM, and sixteen accumulator rows covering all four signed/unsigned byte
pairs on each platform, with both Const and Dynamic metadata and both APIs.

### Representative results

Times are microseconds per product. All rows below use `T.library.matmul`;
BF16 rows use Dynamic metadata and FP32 accumulation/output.
The ratio is generated time / C++ time; smaller is better.

| Platform | Shape M×N×K and pipeline | Inputs | Before | Optimized | C++ | Speedup | Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| x86 | 128×128×128, Convert | packed A/B | 157.14 | 2.31 | 2.40 | 67.99× | 0.96× |
| x86 | 128×128×128, Convert | raw | 8.00 | 6.40 | 3.34 | 1.25× | 1.92× |
| x86 | 64×256×256, BiasRelu | packed A/B | 204.26 | 9.37 | 4.70 | 21.80× | 1.99× |
| x86 | 64×256×256, BiasRelu | raw | 41.19 | 23.33 | 8.89 | 1.77× | 2.62× |
| 920f-4 | 128×128×128, Convert | packed A/B | 473.14 | 7.28 | 5.84 | 64.99× | 1.25× |
| 920f-4 | 128×128×128, Convert | raw | 471.31 | 18.42 | 7.91 | 25.58× | 2.33× |
| 920f-4 | 64×256×256, BiasRelu | packed A/B | 1693.26 | 67.96 | 10.85 | 24.92× | 6.26× |
| 920f-4 | 64×256×256, BiasRelu | raw | 1725.14 | 93.47 | 15.74 | 18.46× | 5.94× |
| 920f-4 | 64×256×256, FP64, Dynamic | raw | 6545.39 | 384.65 | 120.84 | 17.02× | 3.18× |
| 920f-4 | 64×256×256, FP64, Const | raw | 2173.47 | 300.07 | 121.94 | 7.24× | 2.46× |

FP64's "Before" is the already-corrected pre-SSA/cache version (`candidate-v9`),
not the original commit, whose Const overflow guard did not compile.
The final BF16 datasets are `ab-final-local` and `arm-results/ab-final-arm`;
FP64 is `arm-results/cache-ab-arm-v12`, under `build-matmul-perf` locally.

### Changes and limits

AMX consumes validated external packed A/B directly for complete K groups.
SME uses ZA storage-word transposes to construct panel-major operands and
reuses external packed operands. Complete K-contiguous spatial panels omit
intermediate dense copies. Fractional K is still privately zero-padded;
arbitrary external padding is never assumed numerically zero. I16/I64 retains
its separate four-halfword layout rather than misusing the public I16 ABI.

Whole-row epilogues share vector layout checks. Complete initializers omit
redundant zero fills. SME callbacks snapshot invariant pointers and metadata
inside each ownership region; assembly inspection found repeated reference
reloads after MOPA memory clobbers. This last change alone improves the thirty
BF16 observations by 1.08–4.29×. Larger default cache panels reuse register
tiles; moderate raw FP64 working sets use one strided resident leaf. Explicit
BM/BN/BK policies retain their original semantics.

Correctness fixes retain transpose coordinates, select converting scalar-load
values rather than incompatible pointers, avoid unrepresentable signed guard
literals, and capture inline epilogue loads at the original whole-matrix store.
An epilogue broadcast aliasing the first output row must preserve its old value
for subsequent rows. These protections must not be removed for timing gains.

This is not universal parity or universal non-regression: the local 16×16×8
and 32×32×17 packed library cases are approximately 5% and 3% slower than the
starting implementation in the final paired run. ARM 19×23×17 tails remain
roughly 40–50× C++ time. Medium fusion, raw packing, entry/preflight overhead
and small-problem path selection still need further work. The sixteen byte-pair
accumulator observations precede the final ARM SSA/cache changes and are
retained as diagnostic coverage, not final-speed claims.

### Reproduction and artifact provenance

For a new representative build, run in an immutable checkout / package snapshot
and pass compilers explicitly:

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false python benchmarks/dsl_matmul_source_catalog.py \
  --cohort base packed --mode Dynamic --level dot matmul --native bf16 \
  --shape 16x16x8 32x32x17 128x128x128 19x23x17 64x256x256 \
  --packing raw packedab --pipeline Convert BiasRelu Relu --rank 2 --cc gcc --cxx g++ --target Native \
  --jobs 8 --cpu 24 --build-dir build-matmul-perf/rebuild \
  --manifest build-matmul-perf/rebuild.json --output build-matmul-perf/rebuild.csv
```

On `920f-4`, first `source ~/.bashrc` and `conda activate af3`; use
`--cc clang --cxx /home/ryz/vecops-neo/build-math-unify/clangxx-gcc15`, CPU 73,
`CXXFLAGS=''`, `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`, and
`TMPDIR=/home/ryz/vecops-neo/build-matmul-perf/tmp`. The installed immutable SDK
is `/home/ryz/vecops-neo/build-matrix-recipes/sdk-v3`.

Index a published build using `--snapshot CHECKOUT --manifest MANIFEST
--build-dir BUILD --index-output INDEX`. Then time existing native artifacts:

```bash
python benchmarks/dsl_matmul_artifact_ab.py --snapshot build-matmul-perf/candidate-v14 \
  --baseline-index build-matmul-perf/baseline-local-index.json \
  --candidate-index build-matmul-perf/candidate-local-v7-index.json \
  --output-dir build-matmul-perf/ab-final-local --cpu 24
```

Final ARM BF16 uses `baseline-valid-arm-index.json` and `ssa-v10-index.json`,
CPU 73, and `--allow-baseline-subset` to retain the eight excluded baseline
rows in its ledger. Typed/cache measurements use `types-arm-v9-index.json`
and `cache-v11-index.json`; FP64 uses `fp64-base-index.json` and
`cache-v12-index.json`, CPU 76. Paths are relative to the remote source root.

The final ARM timing variants were compiled from freshly lowered DSL source
with the previously validated adapter ABI/spec, recorded compiler command,
unchanged SDK headers and runtime. Their runtime-linked DSOs and commands are
under `build-matmul-perf/{ssa-v10,cache-v11,cache-v12}` remotely. This diagnostic
compilation avoids the slow SDK attestation/publication step; it is not a claim
that these variants were published through the normal JIT artifact flow.
Every timed variant still passes the independent oracle. Normal JIT validation
also passes the targeted SME ownership, cache, external-packing and alias tests.
The final compiler/catalog regression run passes 510 tests. Local native matrix
coverage passes 22 tests, followed by eight row-epilogue/packing regressions.
ARM native coverage passes 31 tests with one AMX-only skip, followed by eight
row-epilogue/packing and four SSA ownership/cache regressions. The final inline
broadcast/output-alias regression passes for both APIs on both machines;
the default FP64 producer/accumulator/epilogue alias case also passes on ARM.

The machine-readable ledger and links to raw datasets are in
`build-matmul-perf/performance-summary.json`. Build products are git-ignored.
