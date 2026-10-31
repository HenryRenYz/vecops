<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->

# NUMA-aware memory measurements

`vecops-memory-bench` measures FP32 application bandwidth, randomized dependent
pointer-load latency, latency under FP32 bandwidth load, and atomic handoff
round-trip latency. It is a Linux tool for native AVX-capable x86 and SVE ARM
targets. It never changes governors, NUMA balancing, global huge page pools,
or system configuration. All code and output are English.

## Build

From the repository root:

```bash
cmake -S . -B build/memory-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v gcc)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DVECOPS_ENABLE_OPENMP=OFF -DVECOPS_BUILD_COMPILER=OFF \
  -DVECOPS_BUILD_TESTING=OFF -DVECOPS_BUILD_BENCHMARKS=OFF \
  -DVECOPS_BUILD_MEMORY_TOOLS=ON
cmake --build build/memory-bench --target vecops-memory-bench -j 2
build/memory-bench/tools/memory/vecops-memory-bench --help
```

On ARM substitute explicit `clang` and `clang++` paths. The target reuses the
repository's Native ISA configuration and SVE compiler compatibility flags.
Native binaries are host-specific. No hwloc/libnuma dependency is required:
Linux sysfs, affinity, `mbind`, and query-only `move_pages` are used directly.
Binding or residence-query failures are fatal, with no unbound fallback.

## Placement and CPU selection

`--numa 3,4` starts two simultaneous groups with private arrays and workers.
`--numa 3:4,5:6` starts two simultaneous source-to-destination routes. For
copy, input is on the source and output on the destination. For triad,
**both inputs** are on the source and output is on the destination.
Read/write accept single targets only; an explicit colon is rejected.

`--interleave` distributes **each private array** cyclically over the sorted,
unique memory nodes across all routes. Source and destination pools are formed
separately: `--numa 8:24,9:25,10:26,11:27 --interleave` interleaves each input
over DDR 8-11 and each output over HBM 24-27. Triad's two inputs each use the
source pool. Execution-node selection and worker ownership do not change.
Every array starts at the lowest node (fixed phase zero), including arrays
shorter than a complete cycle; one unique node degenerates to single-node bind.
There is no implicit grouping by socket or die.

Interleaving supports bandwidth, unloaded chase and loaded chase. Use explicit
`--pages base` or `--pages huge`: the unit is the OS base page or one 2 MiB
hugetlb page, respectively. `--pages auto` and ping-pong reject this option.
The allocator installs one range interleave policy, probes its initial phase,
retains a phase-zero slice and releases unused prefix/suffix pages before full
initialization. Temporary reservation overhead is at most (node count - 1)
pages per array. No per-page policies or global policy changes are used.
Every page, including padding, is checked against its **exact cyclic node**
before and after timing. Allocation fallback or migration is a fatal placement
failure, never accepted as a partially interleaved measurement.

Schema version 2 adds `memory_policy`, `source_memory_nodes`,
`destination_memory_nodes`, `interleave_bytes`, `source_page_counts` and
`destination_page_counts` to CSV/JSON. Page-count strings use `node:count|...`;
counts cover mapped pages across all arrays of that role in the row (both
Triad inputs count), including zero counts and allocation padding. Group/total
rows sum page counts, never bandwidths. `source_node`/`destination_node` remain
the original route labels; they do not describe exclusive memory ownership
when interleaving is enabled. Group bandwidth belongs to its executing workers,
not an individual memory controller. Chase reports a mixed-placement average.

Example: four execution NUMAs interleaving their private arrays within die B:

```sh
vecops-memory-bench bandwidth --numa 28,29,30,31 \
  --exec-node 12,13,14,15 --cpu 0-36 --size 16MiB \
  --interleave --pages huge --kernel vec4 --access temporal,stream \
  --csv interleave-die-b.csv
```

`--exec-node auto` and `dst` execute on each destination node; `src` executes
on each source. An explicit list supplies one execution node per group.
Memory-only targets require an explicit execution node with CPUs.

`--cpu 0-3` selects offsets 0 through 3 in each execution node's sorted online
CPU list. Offsets are resolved before applying process affinity; excluded CPUs
are errors, never silently renumbered. `--cpu-kind physical` first retains the
lowest OS CPU ID for each physical core. The default is `logical`, retaining
SMT siblings. `--cpu-absolute` selects OS CPU IDs and requires one group.
Overlapping worker CPUs are rejected. Absolute mappings are printed/exported.
For heterogeneous groups, `--group-cpus '0-1,6-37|0-37'` specifies one offset
list per group, overriding `--cpu`. This supports excluding occupied cores in
one NUMA domain while retaining every core in another. It cannot be combined
with a semicolon CPU sweep. Reserved loaded-latency and ping-pong peer offsets
are resolved independently of this override.

Quote semicolons: `--cpu '0;0-1;0-3;0-7'` runs 1, 2, 4, and 8 workers **per
group**, sequentially. This is a CPU sweep, not four concurrent groups.

## Kernels and bandwidth accounting

| Operation | FP32 expression | Read bytes/element | Write bytes/element |
|---|---|---:|---:|
| read | Load `a[i]` | 4 | 0 |
| write | `d[i] = 1.25f` | 0 | 4 |
| copy | `d[i] = a[i]` | 4 | 4 |
| triad | `d[i] = a[i] + 0.5f * b[i]` | 8 | 4 |

Triad uses explicit FP32 FMA by default. `--fma off` uses separate operations
with implicit contraction disabled. Inputs are bounded, exactly representable
values. All inputs and outputs are checked after timing, including sentinels
in any unwritten portion of an incomplete scan.

`scalar16/32/64` advance **each array** by 16/32/64 bytes per iteration using
4-byte scalar accesses. `vec1/2/4` use public
`vec::Vec<ScalableTag<float, P>>`, with `P=0,1,2`. The actual native vector
width is recorded. Multi-word tags represent multiple hardware registers.

Explicit scalar instruction wrappers preserve scalar accesses. Vec reads use
register-only compiler barriers, without an artificial arithmetic dependency
chain or extra output traffic. Automatic vectorization, SLP, additional loop
unrolling, LTO, and applicable pattern-replacement/peeling transforms are
disabled. Inspect assembly after changing kernels or compilers. Short-array
rates include function/loop overhead; no estimated overhead is subtracted.

`total_GB_s = (read_bytes + write_bytes) / elapsed_seconds / 1e9`.
Copy also exports `payload_GB_s`, counting copied bytes once. These are
**application rates**, not measured DRAM/controller/interconnect traffic.
RFO, writeback, prefetching, and coherence traffic are not guessed or counted
as physical traffic. Completion fences do not force all temporal dirty lines
to DRAM. Cache-resident writes measure cache access throughput.

## Streaming access

`--access temporal,stream` runs both policies. Stream means NT hints, not a
standard STREAM score or SME Streaming SVE mode. `--load-policy temporal|nt`
and `--store-policy temporal|nt` independently override the preset; `auto`
follows it. Requested and effective policies are exported separately.

| Backend | NT read | NT write |
|---|---|---|
| x86 Vec | Aligned `VMOVNTDQA` | Aligned `VMOVNTDQ` |
| ARM Vec | `LDNT1W` | `STNT1W` |
| x86 scalar | Unavailable | `MOVNTI` of the FP32 32-bit bit pattern |
| ARM scalar | `LDNP S,S` | `STNP S,S` |

x86 scalar NT read is exported as `unsupported`. Scalar streaming copy/triad
on x86 explicitly report temporal reads plus NT writes. Ordinary WB memory
does not acquire guaranteed cache bypass from `VMOVNTDQA`. ARM NT accesses are
hints too. Stream is not proof of physical memory traffic or complete bypass.

BiSheng 5.1 was observed to lower even direct NT ACLE intrinsics to ordinary
SVE loads/stores. The tool therefore forces `LDNT1W/STNT1W` with small inline
assembly wrappers at the public Vec native-word boundary on ARM, preserving
ScalableTag grouping and Vec arithmetic. Both the instruction count and absence
of unintended stack traffic must be checked in the generated kernels.

## Working sets and pages

`--size 16KiB,1MiB,64MiB` selects exact per-thread, per-array sizes by default.
For 8 workers at 64 MiB, triad occupies 1.5 GiB per group. With
`--size-scope group-array`, size is per group per array and divided equally.
Per-worker sizes must be positive multiples of the selected kernel step;
no application buffer is silently enlarged or truncated. Binary suffixes
`KiB/MiB/GiB` and `K/M/G` are accepted.

Cache comparisons use **all active arrays**, including shared-cache usage
across workers. Repeated scans intentionally measure warm-cache throughput
when data fits. For memory bandwidth, exceed relevant caches and check that
further size increases give a stable rate. Remote NUMA placement alone does
not prevent repeated accesses hitting the executing CPU's caches.

`--pages base` uses OS base pages and disables THP for the mapping. `auto`
requests THP without promising it; its exported policy explicitly says
`auto(THP-eligible)`. `huge` requires explicit 2 MiB hugetlb mappings and
fails if unavailable. Global huge page pools are never modified. Mappings
are page-rounded, but byte counts use the exact application size.

Mappings are bound before initialization, fully write-touched, and every
page's residence is queried before and after timed phases, including mapping
padding. Post-run placement/arithmetic failure invalidates all retained rows
for that configuration. Admission checks total demand against 90% of observed
free memory per target; free memory can still change on a shared machine.
Strict huge-page requests check the free 2 MiB hugetlb pool instead of ordinary
`MemFree`, because reserved hugetlb memory is not part of the ordinary pool.

## Timing and aggregation

Pinned workers persist across concurrent calibration, warmup, and samples of
one configuration. A control barrier outside timing wakes them before a
shared future start. All use the same deadline. Only completed batches count;
The start lead is at least 5 ms and scales to 0.1 ms per worker for large
broadcasts; this lead is outside the measured interval.
batches can end partway through an array scan. There are no shared counters,
locks, timer queries, or hardware fences in element loops.

Concurrent 20 ms calibration selects a batch budget up to 20 microseconds,
reduced for tighter skew limits. Defaults: 50 ms warmup, seven 200 ms samples.
Resource accounting excludes the start gate. The end timestamp follows
x86 `MFENCE; LFENCE` or ARM `DSB ISH; ISB`.

Each group counts private bytes and uses the envelope from its earliest
worker start to latest finish. Total bandwidth uses the same formula across
all bandwidth workers. Independently timed rates/best scores are never added.
Thread, group, and aggregate raw rows are exported. Do not add these scopes
together: they describe the same work.

Start/end spread and common overlap are exported. Samples exceeding
`--max-skew` times target duration, or observing page faults, migration, or
worker context switches, are rejected with bounded `--retries`. Rejected
attempts remain in raw output. The terminal shows accepted-sample median,
minimum, maximum, and spread. Insufficient accepted samples produce exit 1.
Do not loosen skew limits to hide competition during final measurements.

## Latency

```bash
build/memory-bench/tools/memory/vecops-memory-bench latency \
  --numa 0 --cpu 0 --size 16KiB,256KiB,4MiB,256MiB --csv chase.csv
```

The chase follows one reproducibly shuffled cycle covering every slot, one
native pointer per cache-line-sized slot. Every address depends on the previous
load. `--stride` increases spacing in cache-line multiples; `--seed` selects
the permutation. Size must divide by stride and hold at least two slots.
Each group uses one CPU; semicolon CPU lists run other CPUs sequentially.

The metric is batch-average **ns/dependent native-pointer load**, including
loop and timing-boundary overhead. Pointer metadata avoids FP32 address
encoding/conversion artifacts. This is not an FP32 bandwidth test or a
distribution of individually timed loads. Page-size/stride comparisons can
study translation effects, but do not isolate pure TLB latency.

```bash
build/memory-bench/tools/memory/vecops-memory-bench latency --loaded \
  --numa 0 --latency-cpu 0 --cpu 1-7 --op triad --kernel vec4 \
  --size 64MiB --csv loaded.csv
```

Loaded mode reserves one `--latency-cpu` offset per execution node, disjoint
from traffic workers. Each chase has its own size-byte buffer; `--size-scope`
controls background arrays only. Both workloads share the deadline. Chase
samples are rejected if background traffic is invalid or their intervals do
not match within the skew limit. Configuration/sample/attempt link the rows.

```bash
build/memory-bench/tools/memory/vecops-memory-bench pingpong \
  --numa 0 --cpu 0 --peer-cpu 1 --csv pingpong.csv
```

Two pinned CPUs exchange a lock-free atomic token on one private cache line,
placed on the source node. Peer offsets refer to the destination node.
Use `--exec-node src` for a source-side cross-node initiator. The metric is
**ns/round-trip**, including coherence, polling, and atomic costs, without an
assumed RTT/2 directional latency. Initiator resource counters are exported;
responder scheduling can also increase RTT. Size sweeps do not apply here.

## Socket 1 on 920f-4

Recheck topology and load before each run. At initial validation node 12 has
CPUs 456-493, node 13 has CPUs 494-531, both on socket 1. Node 28 is memory-only.
Do not infer memory types from IDs. Project remote measurements use socket 1
because socket 0 may be occupied. After sourcing `~/.bashrc` and building with
explicit clang paths, run from `~/vecops-neo`:

```bash
build/memory-bench/tools/memory/vecops-memory-bench bandwidth \
  --numa 12,13 --cpu '0;0-3;0-7;0-15;0-37' \
  --op read,write,copy,triad --kernel scalar16,scalar32,scalar64,vec1,vec2,vec4 \
  --access temporal,stream --size 64MiB --csv build/memory-bench/socket1.csv

build/memory-bench/tools/memory/vecops-memory-bench bandwidth \
  --numa 12:13 --exec-node dst --cpu 0-7 --op copy,triad --kernel vec4 \
  --size 64MiB --csv build/memory-bench/socket1-cross.csv

build/memory-bench/tools/memory/vecops-memory-bench latency \
  --numa 28 --exec-node 12 --cpu 0 --size 256MiB \
  --csv build/memory-bench/socket1-chase.csv
```

These are potentially long/full-load runs, not smoke tests. Check selected
CPUs' `physical_package_id`, utilization, and free memory first. `--dry-run`
checks mappings and sizes without allocating; it does not verify residence.

## Exports and correctness tests

`--csv PATH` writes a header and raw rows. `--json PATH` writes JSON Lines
with identical keys and string-valued fields for lossless CSV parity. Files
are overwritten; use new paths to preserve older runs. Schema version 1
records build flags/compiler, Git description, tool-source SHA-256, host,
CPU/NUMA mapping, sizes, access policies, exact byte counts/intervals,
placement checks, resource counters, and rejection reasons. The Git description
may identify a remote checkout: compare tool-source hashes for rsynced edits.

Exit codes: 0 completed (including explicit unsupported combinations),
1 insufficient accepted samples, 2 configuration/allocation/verification/output
failure. There are no performance thresholds in the correctness suite.

The standard-library smoke suite covers 96 bandwidth combinations (4 ops,
6 steps, 2 policies, 2 sizes), FMA-off, group sizing, idle/loaded chase,
ping-pong, malformed arguments, and CSV/JSON/envelope accounting. x86 scalar
NT reads are expected skips. Disturbed raw samples may yield exit 1 while
their arithmetic/accounting checks still pass.

```bash
python3 tools/memory/SmokeTest.py \
  build/memory-bench/tools/memory/vecops-memory-bench \
  --node 0 --cpu 54 --peer-cpu 55 --output-dir build/memory-bench/smoke

# On 920f-4: socket 1, cross-node paths, aggregation, memory-only target.
python3 tools/memory/SmokeTest.py \
  build/memory-bench/tools/memory/vecops-memory-bench \
  --node 12 --cpu 0 --peer-cpu 1 --second-node 13 --memory-only-node 28 \
  --output-dir build/memory-bench/smoke

# On either build host: reject extra stack traffic, lost NT hints, accidental
# scalar SIMD widening, and unexpected read/NT instruction counts.
python3 tools/memory/CheckAssembly.py build/memory-bench
```

## Socket-1 campaign and report

`RunSocketCampaign.py` is a checkpointed experiment driver for the confirmed
920f-4 topology: CPU NUMA 8-11 and 12-15 are the two dies; HBM targets are
24-27 and 28-31. It keeps commands, raw CSV, logs, placement results, resource
counters and host snapshots. Review current CPU usage before running it on
the shared host. The default excludes the occupied CPU range 306-309.
`--phase full` is separate and must only run once all 304 socket-1 CPUs are
available. No host-wide settings or other processes are modified.

The `scaling` phase includes 1, 2, 4, 8, 15, 24, 32, 36, 37 and 38 cores on
CPU NUMA 12, fixed-size page controls, matching idle-latency baselines, and
source:destination copy/triad transfers, and fixed-working-set cache page
controls. CPU selection is deterministic:
offsets 4-37 followed by 0-3, filtering any excluded physical CPU IDs.

```bash
python3 tools/memory/RunSocketCampaign.py \
  --binary build/memory-bench/tools/memory/vecops-memory-bench \
  --output build/memory-bench/campaign/main --phase all \
  --samples 5 --seconds .15 --retries 0 --exclude-cpus 306-309
python3 tools/memory/SummarizeSocketCampaign.py build/memory-bench/campaign
python3 tools/memory/AuditSocketCampaign.py build/memory-bench/campaign
python3 tools/memory/BuildSocketReport.py build/memory-bench/campaign
```

The report requires NumPy, Matplotlib and the Noto Sans CJK font; the benchmark,
driver, summarizer and audit use no such Python dependencies. It emits a
self-contained Chinese HTML document and standalone PNG figures. The report
uses medians of at least three placement-verified, fault-free, synchronized
observations. Context-switched observations may contribute with their complete
elapsed time; their flags and the separate strict zero-switch count remain
visible. This does not change the executable's stricter acceptance rule.
The start/end skew limit remains 0.1% and the common-window limit 99.8%.
Missing or rejected cases are explicitly marked, never replaced by a peak
from a different configuration. Targeted reruns accept a JSON ID array via
`--case-ids`; keep them in separate output directories with their actual
measurement duration recorded.

`RepairSocketCampaign.py` waits for an optional predecessor, then reruns only
undersampled cases in prescribed order: 5 x 0.5 s, followed (if necessary) by
3 x 20 s. It retains the 0.1% skew threshold, excludes the occupied CPUs, and
does not rerun full-304 cases with fewer cores. Longer windows amortize a
measured completion tail without changing the synchronization acceptance
rule. Report cells include the actual duration. The final prescribed experimental conditions replace earlier conditions.
Repeats with identical duration, default memory policy, complete CPU mapping,
page size and kernel are pooled without selecting by speed; different
conditions are never pooled. `selected-summary.csv` exposes `pooled_runs`,
`source_csvs` and a `qualified_median` only when at least three samples qualify.

Use `--replay-manifest PATH` to replay the exact retained case definitions in
a fresh output directory, including cases from earlier matrix revisions.
Explicitly retain the original duration, sample count and excluded CPU list.

For long many-core runs on a host with automatic NUMA balancing, ordinary
thread stacks and runtime allocations may still produce timed hinting faults
even when every benchmark array has an explicit VMA policy. The campaign
driver can wrap only the benchmark process with `--default-memory-bind 8-15`.
This uses `numactl --membind=8-15`; the arrays' explicit DDR/HBM bindings still
override that default and are verified page by page. It does not change any
global policy. The default mask is retained in manifests and summary CSVs.
For direct executable use, prefix the command with the same numactl policy.

`CaptureSocketBuild.py` retains the actual ELF, compiler flags, project header
inputs, hashes and original license notices. `PackageSocketCampaign.py` creates
a compressed artifact with exact-path data attribution and SHA-256 hashes.
PMU probes are diagnostic, cover whole processes rather than gated samples,
and must run serially with the primary benchmark.

Explicit hugetlb reservations happen at mmap before the array VMA binding.
For huge-page runs the process default mask must include targets with free
hugetlb pages, e.g. `--default-memory-bind 8-15,24-31` on socket 1. A DDR-only
mask can reject mmap even when the requested HBM node has free huge pages.


## Interleave validation and paired campaign

`InterleaveTest.py` checks exported role pools, exact per-array page counts,
aggregate page/byte sums, CSV/JSON agreement, incomplete cycles, repeated
memory nodes, separate input/output pools, and unloaded/loaded pointer chases.
The executable itself verifies every page's exact cyclic residence.
For example, on 920f-4 with idle test cores and free HBM huge-page pools:

```sh
numactl --membind=8-15,24-31 python3 tools/memory/InterleaveTest.py \
  build/memory-interleave/tools/memory/vecops-memory-bench \
  --nodes 28,29,30,31 --execution-nodes 12,13,14,15 \
  --destination-nodes 24,25,26,27 --pages base,huge \
  --output-dir build/memory-interleave/interleave-test
```

`MakeInterleaveCampaign.py --output PATH` predeclares die-local DDR/HBM
comparisons at 16 and 37 workers per group. Replay its manifest with
`RunSocketCampaign.py --replay-manifest PATH`. Each memory configuration has
fresh local-bind and interleave controls; base-page HBM also has a ring-bind
control. `RepairInterleaveCampaign.py` repeats undersampled cases and, if
necessary, increases the window for **all placements in the same comparison**
while retaining the skew threshold. It checks for occupied socket-1 cores
before each heavy repair phase and defers instead of overlapping a new job.

Keep this campaign in its own `interleave-20261030` directory below the original
campaign root. Summarize it independently because the tool source fingerprint
changed. `BuildSocketReport.py` discovers its `analysis.json` and adds a section
without changing historical measurements. Bandwidth ratios use the fresh
same-binary controls, never pooled observations from different tool versions.

## SDMA copy backend

Build with `-DVECOPS_MEMORY_SDMA=ON` and the target's installed `sdma-dk`
headers (AUTO detects headers; OFF disables the adapter). The library is loaded
only when SDMA is requested. No SDK implementation is vendored. CPU operation
remains available without this SDK, including on x86.

`topology --engine sdma` enumerates driver-visible devices and cross-checks the
SDK count. Physical engine count remains unknown. NUMA locality is reported only
from explicit device sysfs attributes; missing locality is `unknown`. Device IDs
never imply socket, die or NUMA IDs. The driver `share_chns` parameter is recorded,
not changed. Enumerating does not allocate every channel to discover capacity.

`bandwidth --engine sdma` defaults to copy. Specify `--sdma-device 0` or one
explicit device per route separated by `|`. One device broadcasts to all routes;
all groups using that device get distinct shared-channel indexes. The default
mode is shared. Exclusive mode must allocate successfully; no fallback occurs.
CPU kernels, access hints and FMA flags are rejected for SDMA.

`--sdma-channels`, `--sdma-depth`, `--sdma-batch` and `--sdma-chunk` accept comma
sweeps. Channels are counted per group, independently of controller CPUs. Each
controller owns at least one channel and private source/destination arrays.
Channels partition each worker's arrays; `--size` retains its normal per-worker,
per-array meaning. Batch divides depth; each channel's array segment must have
at least depth chunks. Partial final chunks are supported. Descriptor sizes and
page-rounded pin lengths must fit the SDK uint32 fields. Software depth is capped
at 1024; allocation and driver configuration determine usable channel capacity.

Each channel maintains a bounded FIFO of batch requests. Outstanding destination
chunks do not overlap; reuse requires completion. Allocate, pin, channel setup
and a full validation copy occur outside timing. Every timed phase uses a new
input generation and poisoned output. All input/output words, untouched suffix
sentinels, and every mapped page's exact NUMA residence are verified. Base and
hugetlb pages and existing `--interleave` policies use the same allocator as CPU
measurements. SDK completion functions govern device visibility; a CPU fence
alone never establishes DMA completion.

All controllers share a future start and deadline. The timed envelope includes
submissions, polling, and draining every outstanding request. Only successfully
completed descriptors contribute bytes. Each worker and group is recorded;
aggregate bandwidth divides total bytes by the earliest-start/latest-completion
envelope. A failure that might leave DMA active terminates without unmapping its
buffers in user space. No global driver configuration, MPAM, IRQ affinity or huge
page pool is changed.

SDMA CSV schema 3 adds device/channel identities, SDK path, channels/depth/batch/
chunk, completed descriptor count, submission count, drain time, setup time and
controller CPU seconds from CLOCK_THREAD_CPUTIME_ID. CPU CSV remains schema 2 with additive columns. Primary
SDMA `payload_GB_s` counts copied bytes once; `total_GB_s` counts the application
read and write (twice payload). Neither represents measured physical fabric
traffic. Bidirectional tests use separate opposite routes and private buffers.
`ns_per_access` remains unused: completion time per descriptor is not a pointer
load latency. Polling may occupy an entire controller core; CPU time is reported.

```sh
vecops-memory-bench bandwidth --engine sdma --numa 8:9 --exec-node 8 --cpu 0 \
  --sdma-device 0 --sdma-channels 4 --sdma-depth 16 --sdma-batch 8 \
  --sdma-chunk 256KiB --size 256MiB --seconds 1 --samples 5 --csv sdma.csv
```

On the tested SDK, shared completion query may return `SDMA_RNDCNT_ERR` while
the requested CQ generation has not arrived. The runner retains that request
until completion, rejects other errors and bounds the final drain. This behavior
is checked against the installed SDK disassembly and generation/sentinel tests.
`getrusage` CPU totals were unreliable on the isolated compute cores; only the
thread CPU clock is used for DMA CPU efficiency. Fault/switch counters still use
getrusage. Setup time and per-worker CPU time have distinct CSV columns.

### Scheduled SDMA campaign and report

`RunSdmaCampaign.py BINARY OUTPUT` targets the verified socket-1 CPU/NUMA map
(296 isolated compute cores, nodes 8-15 and HBM 24-31), queries driver-recommended
devices and tests every enumerated device explicitly. Request one exclusive
920f-lx compute node, 296 CPUs and 32 GiB through lx-run; pass the explicit socket-1
compute CPU list. Builds run on compute with explicit clang/clang++ paths.
The campaign retains all raw results and per-case commands and never changes
system settings. Its default full plan includes edge-copy checks, channel and
request/queue sweeps, fixed-group-working-set saturation, all 256 memory pairs,
bidirectional traffic, die/socket bind and interleave aggregation, all-device pool
sharing, matched CPU copies and high-concurrency socket saturation.

`SummarizeSdmaCampaign.py OUTPUT` independently audits accounting and writes
analysis.json and summary.csv. `--repair` invokes same-node/same-ELF/same-SDK
repairs for cases with fewer than three synchronized fault-free observations.
The scheduled completion stage also adds symmetric local-memory saturation
controls for recommended devices whose corresponding source group differs from
the fixed 8/24 sweep. Repairs use longer predeclared windows derived from observed timing spread, not
performance, keep every original CSV, and select the last repair condition
without pooling unlike windows. A compute-side `repair.enabled` marker opts into
this behavior when the final scheduled summary runs. Local summarization never
launches repair work merely because this marker was fetched.

`BuildSdmaReport.py OUTPUT` creates a Chinese HTML section and standalone plots.
The socket report builder includes the final supplement at
`sdma-20261031/run-03/report-section.html`. Its existing 920f-4 data remains
separate from 920f-lx measurements. Retained runtime conditions and hashes are
part of the report; physical engine counts and NUMA/device ownership are never
inferred from device IDs or bandwidth.
