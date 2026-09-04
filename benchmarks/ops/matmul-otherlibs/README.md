# BF16 GEMM cross-library benchmark

This directory compares vecops with OpenBLAS, oneDNN, LIBXSMM, Arm Compute
Library (ACL), and KUPL MMA on one pinned CPU core.  The common contract is:

```text
Y[N,M] = X[N,K] * W[M,K]^T
X/W: BF16, accumulation and Y: FP32
```

All timed implementations are checked against sampled scalar references before
measurement.  Dependency revisions are fixed in `dependencies.json`.

## Workload catalog

`Cases.h` is the single C++ catalog.  It contains 86 `shape × operation`
records over 31 shapes:

| Group | Shapes and purpose |
|---|---|
| General | square `256×256×256`; ragged/tail `1025×127×769`; 1×1-conv-like `64×3136×576`; rank reduce/expand `64×1024×4096`, `1024×64×4096` |
| LLM | QKV decode/batch/prefill; MLP up/down decode and 128-token prefill; `128³` attention |
| AF3 projection | GridSelfAttention projection at residue counts 128/512/1536/2048, using AF3 chunk 128 or 32 |
| AF3 pair bias | `[C·R,128] × [4,128]^T` at R=128/512/1536 |
| AF3 attention | four-head QK and PV (`head_dim=32`) at R=128/512/1536/2048 |
| AF3 triangle multiplication | batch 128 at R=128/512 and a one-channel representative at R=1536; `call_multiplicity` records the full graph multiplicity |

AF3 dimensions come from the pinned AlphaFold 3 `GridSelfAttention` defaults:
pair channel 128, four heads, head dimension 32, and inference chunks 128/32.
The benchmark measures one representative operator invocation; it does not
multiply elapsed time by `call_multiplicity`.

Operations are plain GEMM, `C + GEMM`, bias, bias+ReLU, and bias+SiLU.  The
catalog only enables semantically relevant epilogues.  `epilogue` in the result
states whether the provider fused it or ran a separate pass.

## Meta.h requirement

vecops registrations are deliberately compile-time separate:

- General cases: `AllDynamic` and `AllConst`.
- LLM and AF3 cases: `TokenDynamic` (only logical N/num_tokens is `meta::Any`;
  M, K, and batch are Const) and `AllConst`.

`audit_catalog.py` verifies both the provider coverage and these exact pairs.
The vecops source is split into five workload shards plus three epilogue shards
to keep ARM template compilation memory bounded; all shards still report
`provider:vecops` and are merged into one CSV.

## Measurement phases and fairness

| Provider | Phase | Packing/setup treatment | Fusion |
|---|---|---|---|
| vecops | `raw_e2e` | raw X/W; any online packing performed by the selected plan is timed | native input/output transforms |
| OpenBLAS | `raw_e2e` | raw X/W and `Trans` flag; no external packing | epilogues are separate |
| oneDNN | `prepared_execute` | rank-2 where possible; oneDNN-selected blocked-W reorder is outside timing | sum/bias native; ReLU and SiLU separate because optimized Arm BRGEMM rejects those post-ops |
| LIBXSMM | `prepared_execute` | W layout conversion and JIT dispatch are outside timing | epilogues are separate |
| ACL arm_gemm | `prepared_execute` | ACL pretransposed W is outside timing | bias/ReLU native; SiLU separate |
| KUPL MMA | `raw_e2e` | KUPL only supplies a 16×64×2 microkernel; vecops transpose kernels produce its A16/B64 layouts online, and tails/macro traversal remain timed | accumulation/bias seed native; activations separate |

Do not compare `raw_e2e` and `prepared_execute` as if they were the same
contract.  `packing_included`, `weights_prepacked`, and `native_batch` are
explicit CSV columns.

Every benchmark uses one OS thread, is pinned to one core by `run_suite.py`,
warms up once for correctness, then defaults to seven repetitions (the recorded
ARM snapshot uses three).  The CSV
contains median, p10, p90, CV, logical GFLOP/s, and a same-phase speedup against
OpenBLAS when such a baseline exists.

Thread limiting is redundant by design: the runner sets `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`; OpenBLAS additionally calls
`openblas_set_num_threads(1)`, oneDNN is built with `ONEDNN_CPU_RUNTIME=SEQ`,
ACL calls `set_nthreads(1)`, and vecops/LIBXSMM/KUPL adapters execute serially.
Google Benchmark itself registers no threaded benchmark, and the process is
affinity-pinned to one logical CPU.

The normalized CSV also contains two deliberately distinct OpenBLAS columns:

- `speedup_vs_openblas` is populated only when the row and OpenBLAS have the
  same measurement phase.
- `speedup_vs_openblas_raw_e2e` always uses the raw-E2E OpenBLAS row.  It makes
  prepared ACL/LIBXSMM rows easy to inspect, but is a cross-phase ratio rather
  than a like-for-like speedup.

For every row, `vecops_dynamic_speedup_vs_row` and
`vecops_const_speedup_vs_row` are `row_median / vecops_median`; values above
one mean the corresponding vecops extent is faster.  The associated reference
times and the actual dynamic extent name are retained in
`vecops_dynamic_us`, `vecops_const_us`, and `vecops_dynamic_extent`.

## Cache-tiling policy

The first run is always `tuning=default`.  A fixed tiling may be added only
after a stable comparison and profiling identify cache blocking, including K
blocking, as the cause of a loss.  Both default and tuned rows must remain in
the output.  Benchmark names and CSV rows reserve `tuning`, `mc`, `nc`, and
`kc` for this audit trail.

- vecops exposes `CacheTiling<MC,NC,KC>` for its GenericTiled family.
- ACL arm_gemm exposes `GemmConfig::inner_block_size` (K block) and
  `outer_block_size`; the adapter also records the automatically selected
  values.
- OpenBLAS and oneDNN do not expose a supported per-call public block-size API.
- LIBXSMM dispatches a full-shape JIT kernel and KUPL is a microkernel-only
  interface here; no artificial cache-tiler is attributed to either library.

The accepted evidence threshold is a repeatable median improvement of at least
5%, with overlapping correctness and no material p90/CV regression.  A tuned
result is hardware-specific and must include host, CPU, compiler, dependency
revision, and the exact blocks.

## Build

Enable this suite explicitly; normal benchmark builds are unaffected.

```bash
cmake -S . -B cmake-build-matmul-otherlibs-x86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(which gcc)" \
  -DCMAKE_CXX_COMPILER="$(which g++)" \
  -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=ON \
  -DVECOPS_ENABLE_MATMUL_OTHERLIBS_BENCHMARKS=ON \
  -DVECOPS_MATMUL_OTHERLIBS_OPENBLAS_ROOT=/path/to/openblas-bf16 \
  -DVECOPS_MATMUL_OTHERLIBS_OPENBLAS_BUILD_TARGET=SAPPHIRERAPIDS \
  -DVECOPS_MATMUL_OTHERLIBS_ONEDNN=ON \
  -DVECOPS_MATMUL_OTHERLIBS_ONEDNN_SOURCE_DIR=/path/to/onednn \
  -DVECOPS_MATMUL_OTHERLIBS_LIBXSMM=ON \
  -DVECOPS_MATMUL_OTHERLIBS_LIBXSMM_SOURCE_DIR=/path/to/libxsmm
cmake --build cmake-build-matmul-otherlibs-x86 -j8
```

The system OpenBLAS is not sufficient merely because `cblas.h` declares
`cblas_sbgemm`; the linked library must export that symbol and must have been
built with `BUILD_BFLOAT16=ON`.  Set
`VECOPS_MATMUL_OTHERLIBS_OPENBLAS_BUILD_TARGET` to the OpenBLAS `TARGET` used
for that install; it is emitted as the result's `implementation` label.

On ARM, use explicit compilers and enable ACL/KUPL with their pinned source and
library paths.  KUPL itself must be configured with `ENABLE_KUPL_MMA=ON`.

```bash
cmake -S . -B cmake-build-matmul-otherlibs-arm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(which clang)" \
  -DCMAKE_CXX_COMPILER="$(which clang++)" \
  -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=ON \
  -DVECOPS_ENABLE_MATMUL_OTHERLIBS_BENCHMARKS=ON \
  -DVECOPS_MATMUL_OTHERLIBS_OPENBLAS_BUILD_TARGET=ARMV8SVE \
  -DVECOPS_MATMUL_OTHERLIBS_KUPL=ON \
  -DVECOPS_MATMUL_OTHERLIBS_KUPL_SOURCE_DIR=/path/to/kupl \
  -DVECOPS_MATMUL_OTHERLIBS_KUPL_LIB=/path/to/libkupl.so \
  -DVECOPS_MATMUL_OTHERLIBS_ACL=ON \
  -DVECOPS_MATMUL_OTHERLIBS_ACL_SOURCE_DIR=/path/to/ComputeLibrary \
  -DVECOPS_MATMUL_OTHERLIBS_ACL_LIB_DIR=/path/to/acl/build
```

## Run, audit, and CSV

```bash
python3 benchmarks/ops/matmul-otherlibs/run_suite.py \
  --build-dir cmake-build-matmul-otherlibs-x86 \
  --output-dir benchmarks/ops/matmul-otherlibs/results/raw/x86 \
  --host local-x86 --cpu 4 \
  --providers vecops openblas onednn libxsmm \
  --compiler "gcc 13.3.0" --backend-isa "AVX-512+AMX-BF16"
```

For ARM add `acl kupl` to the providers and ensure the ACL/KUPL runtime library
directories are in `LD_LIBRARY_PATH`.  `run_suite.py` writes one JSON and log
per executable, a metadata manifest, and one merged CSV.  It sets common
single-thread environment variables, but it does not change frequency,
governor, NUMA placement, or system services.

Use `--benchmark_list_tests` indirectly through `audit_catalog.py` before a
formal run.  Repeat `--vecops` for all eight vecops shard executables and
`--provider` for every external provider being audited.

The checked-in result snapshots are:

- `results/x86-baseline-default.csv`: all 31 shapes and 86 semantic cases on
  the local x86 host.
- `results/arm-baseline-common.csv`: 11 cross-provider shapes on 920f-4.
- `results/benchmark-results.csv`: a complete 1,204-row matrix: both hosts,
  all 31 shapes/86 semantic cases, and all providers.  Missing measurements
  have blank timing/speedup fields and an explicit `not_measured`,
  `unsupported_arch`, or correctness-failure status.
- `results/benchmark-results-measured.csv`: only actually measured rows,
  including the two rejected ARM OpenBLAS G02 rows.
- `results/cross-arch-common.csv`: only case IDs present on both hosts, for a
  balanced x86/ARM view generated from the same measurements.
- `results/vecops-performance-summary.csv`: geometric-mean vecops throughput
  and provider-relative speedups, split overall, by case group, and by
  epilogue operation.
- `results/x86-tiling-probe.csv`: the cache-tiling audit, kept separate from
  the accepted baseline rows.

`MatmulOtherlibsVecopsPhaseProbe-Native` is a diagnostic-only executable for
rank-two cases.  It reports `pack_a`, `pack_b`, `prepared_b_execute`, and
`prepared_ab_compute` separately, using the public `ops::matmul_pack` API.
The actual phase name is `prepared_b_execute`: B preparation is excluded but
the planner may still pack the changing A inside the timed execution.  Only
`prepared_ab_compute` excludes packing on both sides.
General cases are emitted as AllDynamic and AllConst; LLM/AF3 cases are
emitted as TokenDynamic and AllConst, matching the Meta contract of the
formal suite.  The last phase is the closest estimate of the architecture
microkernel/traversal cost with packing removed; none of these rows belongs
in the formal raw-E2E provider aggregate.

`MatmulOtherlibsVecopsTilingProbe-Native` also registers `probe_prepared_b`
rows for L03/L05/L07. They prepack B before timing, then sweep the same
MC/NC/KC profiles (including large/full-K candidates) through GenericTiled.
This separates cache-loop/accumulator round trips from the one-time weight
packing cost and guards against replacing the faster whole-problem prepared
path with an apparently more sophisticated but slower split-K plan.

On Arm, `MatmulOtherlibsVecopsBatchPhaseProbe-Native` decomposes A08 into a
native rank-3 call, four explicit rank-2 calls, prepared-B/prepared-AB calls,
and matching variants whose whole batch is wrapped by one Streaming+ZA
region.  It isolates rank-3 dispatch, packing state, and streaming-region
lifetime from the shallow-K GEMMAdd cost.

On x86, `MatmulOtherlibsVecopsSpecialCaseProbe-Native` compares the existing
AVX-512 small-N leaf with the default AMX path for A05--A07 and sweeps aligned,
M-tail, N-tail, K-tail, and all-tail shapes against OpenBLAS.  These are
diagnostic rows and are not mixed into the formal provider aggregate.

## Repeated optimization cycles

`run_cycle.sh` automates the expensive iteration used by this report: rebuild
all formal providers, run every executable pinned to one CPU with all thread
environment variables set to one, sync/build/run on 920f-4, pull both result
sets, merge them, and generate regression/remaining-slower/summary CSVs.
It uses the existing configured dependency revisions; remote CMake is always
given explicit clang compiler paths.
AF3 attention registration is split into shallow QK, large QK, and PV
executables/TUs.  This keeps the optimizer peak bounded when SME plans are
instantiated across dynamic/constant and GEMM/GEMMAdd variants;
`run_suite.py` merges all three transparently.

```bash
benchmarks/ops/matmul-otherlibs/run_cycle.sh my-change-01
```

Useful environment overrides include `VECOPS_BENCH_FILTER`,
`VECOPS_X86_FILTER`, `VECOPS_ARM_VECOPS_FILTER`, `VECOPS_ARM_PEER_FILTER`,
`VECOPS_X86_BUILD`, `VECOPS_ARM_HOST`, `VECOPS_ARM_ROOT`,
`VECOPS_ARM_BUILD`, `VECOPS_X86_CPU`, `VECOPS_ARM_CPU`,
`VECOPS_BUILD_JOBS`, `VECOPS_CYCLE_OUTPUT`, `VECOPS_DISABLE_ASLR=1`, and
`VECOPS_BENCH_BASELINE`.

`run_suite.py --disable-aslr` wraps each process with
`setarch <machine> -R` and records `aslr_disabled=true` in metadata. Use it
for adjacent old/new regression gates on short kernels: A08 in particular can
switch between two cache/address modes even at a stable CPU frequency. It does
not replace an interleaved A/B order or an address-skew sweep, but prevents
unrelated executable relocation from dominating the comparison.

The ARM vecops pass covers the full catalog by default.  Its peer pass uses
the practical cross-library subset by default; override
`VECOPS_ARM_PEER_FILTER='.*'` for a deliberately exhaustive (and potentially
very slow) external-library sweep.  The two passes are merged before analysis,
so shapes with no peer datum are retained.

Re-running the same run ID reuses a completed per-platform CSV and resumes at
the missing platform, while the remote `run_suite.py --resume` pass reuses
each valid provider JSON after a partial failure, then regenerates the merged
CSV and analysis.

For a backend-only refresh, pass the old aggregate followed by the refreshed
CSV to `combine_csv.py --latest-wins`; unchanged providers are retained and
the refreshed result keys replace their old rows.

Each run writes a self-contained directory under
`results/raw/cycles/<run-id>/` with `combined.csv` plus:

- `analysis/vecops-regressions.csv`: exact host/case/op/extent/phase matches
  against the selected baseline;
- `analysis/vecops-comparison.csv`: one row for every vecops measurement,
  including per-provider times/phases and `vecops_speedup_vs_*` columns;
  providers without data for that shape stay blank;
- `analysis/vecops-slower.csv`: every provider row faster than vecops, with
  an explicit `phase_aligned` field;
- `analysis/vecops-summary.csv`: geometric means overall, by case group, and
  by operation. `aligned_peer_speedup_geomean` uses every row with a
  phase-aligned peer and follows the detailed table's direction
  (`peer_us / vecops_us`, so values above one favor vecops).

For already collected CSVs, `analyze_cycle.py` can be run independently:

```bash
python3 benchmarks/ops/matmul-otherlibs/analyze_cycle.py \
  --current current.csv --peers previous-complete.csv \
  --baseline previous-complete.csv --output-dir analysis
```

`analyze_splitk_binary.py` is the companion static-code audit for cache-K
work. It reports `.text`, de-duplicated matrix-kernel symbol ranges, and static
AMX dot/SME MOPA/tile load/store occurrences. On x86 split-K probes it also
classifies the legacy unsplit/first/middle/last C endpoint signatures; unknown
or compiler-inlined hosts remain explicit instead of being dropped.

```bash
python3 benchmarks/ops/matmul-otherlibs/analyze_splitk_binary.py \
  path/to/MatmulOtherlibsVecopsTilingProbe-Native \
  --output splitk-binary.csv
```

`G02_ragged` is not in the ARM common snapshot because OpenBLAS 0.3.34's BF16
GEMM returned incorrect values for its odd K/tail combination with both the
NEOVERSEV1 and ARMV8SVE builds.  The case remains in the catalog and in the
x86 result.  See `RESULTS.md` for exact run conditions and the tiling decision.
