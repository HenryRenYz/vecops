# Recorded benchmark snapshots

The CSV files in `results/` are reproducible benchmark snapshots, not claimed
cross-machine peak numbers.  Every accepted row passed the adapter's sampled
FP32 reference check.

## Hosts and scope

| CSV | Host | CPU/ISA | Pinning | Repetitions | Coverage |
|---|---|---|---|---:|---|
| `x86-baseline-default.csv` | local-x86 | Intel Xeon w9-3495X, AVX-512 + AMX-BF16 | one logical CPU | 7 | 31 shapes, 86 semantic cases; vecops + OpenBLAS + oneDNN + LIBXSMM |
| `arm-baseline-common.csv` | 920f-4 | ARM SVE/SVE2/SME, VL=SVL=512b | CPU 4 | 3 | 11 common shapes; all six providers |

`cross-arch-common.csv` is a postprocessed 372-row view containing those same
11 case IDs from both hosts.  `benchmark-results-measured.csv` contains the
661 measured rows (659 correct and two rejected OpenBLAS G02 rows).
`benchmark-results.csv` expands that data into a 1,204-row rectangular matrix:
602 rows per host, covering all 31 shapes/86 semantic cases and every provider.
Unmeasured or unsupported cells retain metadata but have blank timings and
speedups.  `vecops-performance-summary.csv` aggregates only `status=ok` rows.

ARM used BiSheng clang 19.1.7.  The formal OpenBLAS ARM rows use an
`ARMV8SVE`, `BUILD_BFLOAT16=1`, single-thread build and carry `ARMV8SVE` in
the `implementation` column.  The local run used GCC 13.3.0.  Exact library
revisions are recorded in every CSV row and in `dependencies.json`.

The ARM common set is G01, G03, L01, L02, L04, L08, A01, A05, A08, A12, and
A16.  `G02_ragged` was tested but excluded: OpenBLAS produced wrong BF16 GEMM
values with both NEOVERSEV1 and ARMV8SVE targets, including after zero-padding
odd K.  Keeping a known-wrong row would violate the common-case contract.

## Cache-tiling decision

The original `x86-tiling-probe.csv` exposed a GenericTiled output-origin bug;
that bug is now fixed and the current 90/90-case correctness probe passes.
The 2026-09-04 rerun swept ten MC/NC/KC profiles on G05, L03, L05, L07, and
A18.  Even each case's best profile was 1.90x--2.70x slower than the default
WholeProblem path.  This is not evidence that cache blocking is unnecessary:
the probe changes both the loop nest and the implementation family, and its
panel packing/accumulator lifecycle is still expensive.  No manual block size
is accepted into the formal rows.

ACL records the cache block selected by `arm_gemm` in `selected_kc` and
`selected_outer_block`.  OpenBLAS and oneDNN expose no supported per-call block
override, while LIBXSMM is a full-shape JIT kernel and KUPL supplies only the
microkernel used by this adapter.  Consequently all formal rows retain
`tuning=default`; no result was improved by silently changing only one side.

## Interpretation

Compare rows only when `phase` matches.  In particular, vecops, OpenBLAS, and
KUPL are `raw_e2e`, whereas oneDNN, ACL, and LIBXSMM are `prepared_execute`;
the latter exclude documented weight preparation/JIT work.
The `extent` dimension applies only to vecops and encodes the required Meta.h
comparison (`AllDynamic`/`AllConst` or `TokenDynamic`/`AllConst`).

`speedup_vs_openblas` remains a same-phase metric.  The additional
`speedup_vs_openblas_raw_e2e` is explicitly a raw-OpenBLAS reference and can
therefore be populated for prepared rows.  Likewise, the two
`vecops_*_speedup_vs_row` columns compare every row against the raw-E2E vecops
Dynamic and Const rows.  They are convenient ratios, but a phase mismatch must
still be considered when interpreting ACL or LIBXSMM.

## Why the recorded ARM ratios are unusually large

The six ARM rows do not all use SME, despite all running on an SME-capable
host.  The recorded implementations and inspected binaries show:

- vecops uses its `SME_BF16F32` atom, and ACL selected kernels whose names
  start with `sme_`.
- KUPL uses the SME `KP36_16x64x2_BF16BF16F32` microkernel, but KUPL does not
  provide the surrounding full GEMM in this comparison.  The revised timed
  adapter uses vecops transpose kernels to produce KUPL's distinct A16/B64
  layouts, then performs tile initialization/copy-out, tails, and one
  microkernel call per 16x64 output tile.  Its padding ratio remains 16 for
  N=1 and 4 for N=4.
- The OpenBLAS `ARMV8SVE` build is SVE-targeted, not SME-targeted.  Its archive
  selected generic `sbgemm_kernel.c.o`; disassembly showed repeated calls to
  `sbf16tos_` and no BF16 dot/matrix or SME instructions.  This explains the
  observed 0.07--0.46 logical GFLOP/s and makes OpenBLAS the source of most
  hundred-fold ratios.
- oneDNN now uses rank-2 descriptors where possible and preorders W into the
  library-selected blocked format.  It reports `brg:sve_512`: oneDNN 3.13 has
  SME BRGEMM source, but its available SME kernel is FP32 FMOPA rather than a
  BF16 BFMOPA path.  ReLU/SiLU are separate because optimized Arm BRGEMM
  rejects those post-ops.  LIBXSMM did not expose a selected ARM ISA label in
  this run and must not be described as SME based on host capability alone.

The ARM v3 snapshot replaces the previous fallback oneDNN and scalar-pack KUPL
rows.  Summary numbers in `vecops-performance-summary.csv` are regenerated from
the v3 data; old raw JSON remains ignored build evidence only.

## 2026-09-04 current-worktree rerun

The current-worktree rerun is stored under `results/raw/rebench-current/`.
The x86 merged CSV contains 172 correct vecops rows and 86 correct rows for
each of OpenBLAS, oneDNN, and LIBXSMM; no row was excluded for a compile or
numerical failure.  Relative to the post-merge `0b7a801` vecops snapshot, the
172-row wall-time geometric mean is 0.994x (`old/new`), i.e. effectively flat
within the scaling-enabled host's run-to-run noise.  AllConst is 1.004x,
AllDynamic 0.971x, and TokenDynamic 0.986x.

The accompanying `*-phase-probe.csv` files separate pack A, pack B,
prepared-B compute, and prepared-A/B compute.  On x86, G05 becomes 528 us when
both operands are prepared versus 961 us raw, but L05 remains 14.0 ms prepared
versus OpenBLAS's 9.8 ms raw including its own packing.  On ARM, prepared-B
compute for L01/L04 is 1.63/4.41 ms while standalone B packing is 3.15/8.41 ms:
the large raw decode gap is a weight-lifecycle mismatch, not an absent SME
compute kernel.  Detailed perf and disassembly conclusions are maintained in
`docs/internal/gemm-issues.md`.

The ARM common rerun contains 217/217 correct rows (62 vecops plus 31 for each
of the five peers).  A second vecops-only pass fills all 31 shapes, so
`benchmark-results-current-20260904.csv` is again a 1,204-row rectangular
matrix: 757 measured-correct rows, 275 unmeasured ARM peer rows, and 172
architecture-unsupported x86 ACL/KUPL rows.  The registered benchmark setting
won over an attempted command-line override, so both current formal runs use
seven repetitions; this is reflected in the raw aggregates.

## 2026-09-05 split-K routed-kernel rerun

The routed-kernel cycle is under
`results/raw/cycles/splitk-routed-20260905/`. Its `combined.csv` contains 757
measured rows, all `ok`: 172 vecops rows on each host plus the retained peer
rows from the preceding full provider cycle. The stable exports are:

- `results/benchmark-results-current-20260905.csv`: measured long form;
- `results/vecops-comparison-current-20260905.csv`: all 344 vecops
  shape/op/extent rows, with one column group per provider and blanks where a
  provider was not measured;
- `results/vecops-performance-current-20260905-summary.csv`: overall,
  case-group, and operation geometric means.

The x86 formal cycle has `old/new=1.0286x` overall (2.78% lower wall-time
geometric mean); General is `1.1609x`, AF3 `1.0261x`, and LLM is effectively
flat at `0.9937x`. Apparent 13--53% regressions in
four short/batched cases did not reproduce under CPU-4, single-thread,
ASLR-disabled old/new/new/old interleaving: G01/A05/A16 stayed within 0.83%,
while A08 improved 1.8--5.6%. The authoritative A/B measurements are in
`validation/x86-aslr-disabled-stage2-vs-stage3.csv`; independent-process CSV
medians must not override that adjacent comparison for these bimodal cases.

The ARM 172-row formal cycle is neutral overall (`old/new=0.99988x`). Its
strengthened SME suite passes 29/29, including a three-phase K=9/KC=4
workspace-accumulator test with ragged M/N and transformed C input/output.
The disassembly gate also passes all 106 StreamingZA regions with no
unexpected nested call; only the intentional shared split-K traversal and
existing predicate helpers are allowlisted, and the traversal was separately
checked to contain no nested call.

Static split-K probes show the intended structural result. On x86, executable
`.text` falls 20.1%, matrix-kernel symbol bytes 43.1%, and static AMX dots 50%;
first/middle/last become one `split_shared` body. On ARM, outlining the whole
per-KC Tile2D traversal leaves its selected ZA cases inline while reducing the
probe `.text` 14.3%, kernel bytes 37.7%, and static MOPA count 46.5%; two
`split_shared` traversal symbols replace the phase-specific bodies. G05/KC256
is 1.95--2.25% slower than the pre-route probe, within the explicit 5% gate;
the default formal WholeProblem targets contain no routed symbol.
