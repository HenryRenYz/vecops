# Instruction microbenchmarks

The instruction benchmarks measure concrete x86 or SVE instructions,
explicitly named dependency sequences, and cache-sized memory working sets.
They do not try to assign one architecture's instruction names to another.

## Build

```bash
cmake -S . -B cmake-build-release-inst \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DBUILD_BENCHMARKS=ON
cmake --build cmake-build-release-inst \
  --target InstructionBench-AVX2 InstructionBench-AVX512 InstructionBench-Native -j2
```

On ARM the corresponding explicit targets are `InstructionBench-SVE`,
`InstructionBench-SVE2`, and `InstructionBench-Native`.  Every instruction
target is compiled with `-O2`, its target-specific `-march`, and without LTO.
The generated kernel symbols are `noinline` and have stable C names for
disassembly.

## Coverage

- AVX2: immediate and scalar-count packed shifts, AVX2 per-element shifts,
  AND/OR/XOR/ANDN/NOT, and `vblendvps`/`vblendvpd`/`vpblendvb`.
- AVX-512: immediate, scalar-count, and per-element B/W/D/Q shifts,
  AND/OR/XOR/ANDN/ternary-logic NOT, and half-density B/W/D/Q mask blends.
- SVE: immediate and per-element B/H/S/D LSL/LSR/ASR, AND/OR/EOR/BIC/NOT,
  and half-predicate B/H/S/D `sel`.
- Memory: aligned sequential load/store throughput at each reported cache
  level and beyond the last reported cache, dependent pointer-chase load
  latency, store-to-load forwarding, cross-cache-line accesses, and
  half-mask/predicate contiguous load/store.
- AVX2 memory conversions: signed and unsigned widening loads for every
  8/16/32-bit source to wider 16/32/64-bit destination pair.
- AVX-512 memory conversions: the same widening loads plus all 16/32/64-bit
  to narrower 8/16/32-bit truncating, signed-saturating, and
  unsigned-destination-saturating stores.
- SVE memory conversions: all signed/unsigned widening `ld1sb`/`ld1sh`/`ld1sw`
  and `ld1b`/`ld1h`/`ld1w` forms, plus all truncating `st1b`/`st1h`/`st1w`
  forms. SVE has no fused saturating narrow-store instruction, so a software
  narrow-plus-store sequence is intentionally not presented as a
  single-instruction result.

## Inspect and run

List machine-readable case metadata:

```bash
cmake-build-release-inst/benchmarks/InstructionBench-Native --inst_list_json
```

Run one Google Benchmark case with a fixed inner-loop count:

```bash
cmake-build-release-inst/benchmarks/InstructionBench-Native \
  --benchmark_filter='^Instruction/Native/vaddps_zmm_zmm_zmm/latency$' \
  --benchmark_min_time=1x \
  --inst_loops=10000000
```

Run the perf-based driver on one logical CPU:

```bash
benchmarks/inst/run_instruction_bench.py \
  --binary cmake-build-release-inst/benchmarks/InstructionBench-Native \
  --cpu 1 \
  --loops 10000000 \
  --repetitions 7 \
  --dump-asm
```

The driver automatically adds every baseline required by the selected cases,
runs each case in a separate `perf stat` process under `numactl`, and writes
`context.json`, raw perf CSV, summary CSV/Markdown, per-run logs, and
optionally a disassembly to `benchmarks/results/inst/`.  Register cases use a
core event preset; memory cases use a cache-oriented preset.  `--events`
overrides both.

`cycles_per_sequence` contains whole-process perf counts normalized by the
known asm work. `net_cycles_per_sequence` subtracts the matching control
baseline. Stream cases use a same-stride address-generation control;
conversion index-chase cases use a same-source-width scalar extending-load
chain; other cases use the empty loop. Use sufficiently large `--loops` for
final measurements and inspect both values.

Memory throughput rows report raw and baseline-adjusted bytes/cycle, logical
GB/s, and, for conversions, raw and adjusted elements/cycle plus Gelem/s.
GB/s and Gelem/s use the benchmark kernel time, while perf cycle fields count
the whole process. Masked rows count active bytes only. Working-set sizes are
discovered from the cache topology of the CPU selected by `--cpu`:

- L1 and L2 use half of the reported capacity, rounded down to a power of two.
- L3 uses one quarter of the reported shared capacity.
- `beyond_last_reported_cache` uses at least 256 MiB or twice the largest
  reported cache, whichever is larger.

Absent cache levels are omitted.  In particular, a machine that exposes only
L1/L2 does not get a synthetic L3 result.  Large random pointer-chase results
include TLB effects and any unreported system cache.

Sequential load/store cases measure sustainable traffic with hardware
prefetching enabled.  A store has no directly observable "completed in L2/L3"
instruction latency, so latency is represented by a separately named
store-to-load-forwarding round trip.  Dependent cache latency is represented
by a vector load followed by extraction of the next address.

Conversion-load latency is a compressed-index pointer chase. Every source
lane contains the next cache-line index; the vector instruction widens the
load, one widened lane is extracted, and that value determines the next
address. Its raw cycle value is the latency of the complete dependent chain.
Subtracting a matched scalar extending-load index chase produces a relative
incremental estimate for vector widening and extraction. It is not a
standalone or architecture-defined "pure load latency".

Narrow-store latency is measured only as a closed
`narrow store -> matching extending load` forwarding round trip. A store can
retire before data reaches a lower cache, so the benchmark does not claim a
standalone L2/L3 store latency.

## Adding a case

- Put x86 cases in `InstructionBenchX86.cpp` and SVE cases in
  `InstructionBenchSVE.cpp`.
- Name a case after the exact mnemonic, operand widths, predicate/mask mode,
  and relevant immediate.
- A latency case must contain a real dependency chain.  If one instruction
  cannot feed itself safely, register its single-instruction throughput and a
  separately named closed sequence for latency.
- Keep initialization outside the aligned hot loop, update
  `sequences_per_loop`, `instructions_per_sequence`, and `dependency_chains`,
  then verify the generated symbol with `objdump`.
- For a memory case, also set its access pattern, working-set level,
  logical bytes per sequence, and matching address-generation baseline.
