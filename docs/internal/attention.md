# Attention optimization log

This document is the working record for Attention performance engineering.
It intentionally records unsuccessful experiments as well as retained changes,
so later tuning does not repeat disproven ideas.

## Environment

- x86: local 4th Gen Intel Xeon, AVX-512 + AMX, GCC 13, `-O3 -march=native`.
- ARM: `920f-4`, 512-bit SVE/SVE2/SME, BiSheng Clang 19.1.7,
  `-O3 -march=native+bf16+i8mm+sme+sme-fa64+sme-f64f64`.
- Benchmark mode: one logical attention head, BF16 Q/K/V and FP32 output
  unless the case name states otherwise; Google Benchmark CPU time is used.

## Shape provenance and coverage design

The expanded matrix is based on:

- AlphaFold 3's public model configuration and parameter manifest: pair
  attention uses four 32-wide heads, Pairformer single attention uses sixteen
  24-wide heads, and the diffusion transformer uses sixteen 48-wide heads.
  Pair attention selects query chunks of 128 through N=1536 and 32 above that.
  Sources:
  <https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/test_data/model_config.json>
  and
  <https://github.com/google-deepmind/alphafold3/blob/main/docs/model_parameters.md>.
- Llama/Qwen use 128-wide heads and exercise GQA decode/cache workloads;
  Gemma 2 uses 256-wide heads; DeepSeek V3 uses 192 Q/K dimensions
  (128 no-PE + 64 RoPE) and 128 value dimensions. Sources:
  <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json>,
  <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/configuration_gemma2.py>,
  and
  <https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json>.
- Sequence lengths deliberately bracket 32/128/512/1536/2048/4096/8192
  boundaries with 127/255/257/511/1025/1537/4097 tails. `meta::dyn<32>` is
  used independently for each guaranteed-aligned Q or K/V axis; arbitrary
  axes use `meta::Any`.

The benchmark now measures SDPA automatic/materialized/streaming strategies,
Sparse FlashAttention, IBS base indexing alone, and complete IBS Attention.

## Baseline before this iteration

These are the last stable numbers after the initial four-vector `fold` rewrite
and Matmul orientation merge (`c4d4daf`):

| host | case | CPU time |
|---|---|---:|
| x86 | LLM prefill `128x512x128x128`, automatic | 54.3 us |
| x86 | AF3 pair `128x512x32x32`, automatic | 33.3 us |
| ARM | LLM prefill `128x512x128x128`, automatic | 228.4 us |
| ARM | AF3 pair `128x512x32x32`, automatic | 99.3 us |
| ARM | AF3 sparse pair, four blocks | 66.8 us |
| ARM | AF3 dynamic pair, four blocks | 48.1 us |

## Iteration 1: expanded baseline

The x86 expanded baseline contains 78 median aggregates. Representative
results (ns, median of three repetitions):

| case | automatic | materialized | streaming |
|---|---:|---:|---:|
| decode `1x127x128x128` | 1,924 | 1,917 | 3,577 |
| decode `1x2048x128x128` | 20,160 | 19,892 | 56,610 |
| decode `1x8192x128x128` | 285,036 | 206,811 | 284,454 |
| speculative `17x4097x128x128` | 210,016 | 123,489 | 212,207 |
| prefill `128x512x128x128` | 55,386 | 55,259 | 113,105 |
| AF3 pair `128x512x32x32` | 34,677 | 34,514 | 88,061 |
| AF3 pair `128x1536x32x32` | 92,660 | 92,725 | 241,356 |
| AF3 diffusion tail `33x257x48x48` | 9,242 | 9,265 | 18,852 |

Sparse/dynamic component baselines:

| case | direct sparse | dynamic mask | composed dynamic |
|---|---:|---:|---:|
| AF3 pair `128x512`, S=4 | 87,160 | 3,187 | 36,640 |
| AF3 pair `128x1536`, S=8 | 170,869 | 9,673 | 104,950 |
| AF3 diffusion `33x257`, S=4 | 25,166 | 1,247 | 4,041 |

The automatic threshold of 2048 is wrong on x86: at 4097 and 8192 keys it
selects streaming even though materialized is respectively 1.72x and 1.38x
faster. ARM must be measured before moving the cross-platform threshold.

### Profiling and generated code

`perf stat` on x86 AF3 materialized reports 1.77 instructions/cycle and only
0.10% branch misses. LBR sampling attributes 38.2% to the two AMX exact
microkernels and 4.0% to partial-B packing; the remaining 57.7% is the inlined
score decoration/softmax orchestration.

Direct sparse attention attributes roughly 86% to its inlined online kernel
and only about 10% to separately visible AMX microkernels. Disassembly of the
hot body shows a scalar `expf@plt` call for every row/block rescale, including
the common `exp(0)` case, and uses the accumulating PV Matmul variant even for
the first selected block.

Commands:

```sh
perf stat -r 3 -e cycles,instructions,branches,branch-misses,\
cache-references,cache-misses -- AttentionBench-Native \
  --benchmark_filter='Attention/af3_pair/medium/.*/materialized'
perf record -e cycles:u -g --call-graph lbr -- AttentionBench-Native \
  --benchmark_filter='Attention/af3_pair/medium/.*/materialized'
perf report --stdio --no-children --sort=symbol
objdump -d -C --no-show-raw-insn AttentionBench-Native
```

## Iteration 2: remove redundant work

Retained changes:

1. Reuse `decorate_score_row`'s maximum in materialized softmax, eliminating a
   full score-row max scan.
2. Sparse online rescaling calls scalar `exp` only when the block maximum
   actually increases; it skips output scaling on the first block and on rows
   where the scale is exactly one or the previous sum is zero.
3. The first sparse PV product uses non-accumulating Matmul; later blocks use
   the C-input accumulating form.
4. Unmasked dynamic block means omit the row-active initialization and tests
   at compile time.

Representative x86 measurements after the change:

| case | before | after | result |
|---|---:|---:|---:|
| materialized AF3 pair `32x255` | 5.87 us | 5.33 us | -9.3% |
| materialized AF3 pair `128x512` | 34.51 us | 33.86 us | -1.9% |
| materialized AF3 diffusion tail | 9.27 us | 8.71 us | -6.0% |
| sparse AF3 pair, S=4 | 87.16 us | 85.47 us | -1.9% |
| dynamic-mask AF3 pair | 3.19 us | 2.82 us | -11.6% |
| dynamic-mask AF3 diffusion tail | 1.25 us | 1.00 us | -19.9% |
| composed dynamic AF3 pair N=1536 | 104.95 us | 98.49 us | -6.2% |

The long-cache measurements varied substantially with CPU frequency and were
not used to attribute the row-scan change. They will be remeasured alongside
the strategy-threshold experiment.

### Rejected: dense-mask sparse materialization

To recover the old repository's distinction between `SparseAttention` and
`SparseFlashAttention`, an experimental fast path expanded the block index map
to a dense byte mask and invoked materialized dense attention. This retained
correct duplicate-index fallback semantics but was decisively slower because
the masked dense row decoration dominates and all nominally sparse QK/PV work
is still executed:

| case | online sparse | dense-mask experiment |
|---|---:|---:|
| AF3 pair `128x512`, S=4 | 83.90 us | 272.66 us |
| AF3 pair `128x1536`, S=8 | 167.20 us | 807.13 us |
| AF3 diffusion `33x257`, S=4 | 24.77 us | 42.55 us |

The experiment was reverted. A future materialized sparse implementation must
pack only selected K/V blocks into a contiguous panel, as the old repository
did; expanding sparsity into a full-size mask is not viable.

## Iteration 3: retain FP32 exponentials in the score buffer

The first materialized implementation wrote exponentials to the configured
Matmul probability type and immediately loaded them again for PV. With the
usual BF16 probability type this both rounded unnecessarily and spent an
extra conversion pass. Softmax now overwrites its FP32 score buffer with
exponentials, normalizes from that buffer, and converts only once when writing
the probability panel.

Representative strict-exponential x86 results:

| materialized case | before | after | result |
|---|---:|---:|---:|
| LLM prefill `128x512x128x128` | 55.26 us | 52.79 us | -4.5% |
| AF3 pair `128x512x32x32` | 33.86 us | 32.40 us | -4.3% |
| AF3 pair `128x1536x32x32` | 92.73 us | 86.16 us | -7.1% |
| AF3 diffusion tail | 8.71 us | 8.64 us | neutral |

## Iteration 4: explicit exponential accuracy policy

Attention's normalized probabilities do not require the strict libm-style
contract for the normal inference path. `AttentionConfig` therefore exposes
`ExpAccuracy` at compile time, defaults it to `vec::Accuracy::Fast`, and keeps
`Strict` available to accuracy-sensitive callers and a dedicated reference
test. This removes accuracy dispatch and slow strict approximation work from
the hot loop without adding a runtime branch.

Against the FP32-score-buffer strict build on x86:

| case | Strict | Fast | result |
|---|---:|---:|---:|
| LLM prefill `128x512x128x128` | 52.79 us | 50.81 us | -3.8% |
| AF3 pair `128x512x32x32` | 32.40 us | 30.08 us | -7.2% |
| AF3 pair `128x1536x32x32` | 86.16 us | 81.44 us | -5.5% |
| AF3 diffusion tail | 8.64 us | 8.22 us | -4.8% |
| composed dynamic pair `128x512`, S=4 | 36.88 us | 34.77 us | -5.7% |

## Iteration 5: backend-specific automatic dense strategy

The expanded ARM baseline showed that a single key-length threshold cannot
serve AMX and SME. On AMX, materialized remains faster through the measured
8192-key cases. On SME, streaming wins for single-query short decode and for
large key counts, while materialization wins for multi-query problems through
about 2048 keys.

Representative ARM measurements before changing the policy (ns):

| case | materialized | streaming | preferred |
|---|---:|---:|---|
| decode `1x127x128x128` | 6,844 | 4,528 | streaming |
| decode `1x2048x128x128` | 159,560 | 165,419 | materialized/noisy |
| decode `1x8192x128x128` | 735,270 | 514,418 | streaming |
| speculative `17x4097x128x128` | 720,561 | 691,637 | streaming |
| prefill `128x512x128x128` | 221,358 | 347,917 | materialized |
| AF3 pair `128x1536x32x32` | 278,022 | 553,351 | materialized |

The retained automatic policy is:

- x86/AMX: materialize through 8192 keys;
- ARM/SVE: stream for `Lq=1, Lkv<=512`, materialize through 2048 keys, and
  stream above 2048;
- other backends: retain the conservative 2048-key threshold.

The x86 policy alone changes representative automatic results as follows:

| case | old automatic | new automatic | result |
|---|---:|---:|---:|
| decode `1x8192x128x128` | 285.04 us | 214.62 us | -24.7% |
| speculative `17x4097x128x128` | 210.02 us | 119.90 us | -42.9% |
| DeepSeek MLA `1x4096x192x128` | 145.97 us | 76.85 us | -47.3% |

### Rejected: stream every ARM single-query decode

The final unpinned sweep initially suggested extending the ARM single-query
streaming rule from 512 to 2048 keys (137.32 us streaming versus 147.62 us
materialized). Immediate reruns reversed that ordering: one run measured
171.57 versus 151.63 us, while a run pinned to CPU 300 measured 165.42 versus
161.38 us. Automatic-path code layout also changed the result relative to the
explicit strategies. Earlier expanded data had materialized slightly ahead at
2048 as well. The crossover is therefore not stable enough to specialize;
the conservative `Lq=1, Lkv<=512` rule is retained and the 2048 point stays
materialized.

## Iteration 6: `fold` tail-factor sweep

Changing every attention traversal from `fold<4, 1>` to `fold<4, 2>` was
tested as a generated-code tuning experiment. It helps loops whose length is
usually an exact multiple of the vector width, but increases masked-tail and
code-size cost in dense attention:

| x86 case | one-vector tail | two-vector tail | result |
|---|---:|---:|---:|
| short prompt materialized | 1.55 us | 1.64 us | +5.6% |
| AF3 pair boundary materialized | 81.26 us | 83.69 us | +3.0% |
| AF3 atom materialized | 2.09 us | 2.56 us | +22.3% |
| dynamic mask `128x512` | 2.86 us | 2.05 us | -28.4% |
| dynamic mask `128x1536` | 8.68 us | 6.41 us | -26.1% |
| dynamic mask tail `33x257` | 1.01 us | 1.09 us | +7.6% |

The global change was rejected. The retained hybrid uses a two-vector tail
only for x86 dynamic block means and coarse mask/bias traversal; dense
softmax, sparse rescaling, coarse softmax, normalization, and every SVE loop
stay at one vector. A fresh x86 combined run measures 1.98 us and 6.24 us for
the aligned dynamic-mask cases, while the arbitrary-tail case returns to
1.00 us. The first ARM trial confirmed the need for backend specialization:
two-vector tails regressed the same aligned cases from 20.16/59.57 us to
21.83/64.21 us and the arbitrary-tail case from 4.50 us to 7.45 us.
After specializing SVE back to one vector, the final medians are
20.07/59.22/4.67 us respectively.

## Iteration 7: compact selected-block sparse materialization

The viable `sparse_attention` fast path packs only selected contiguous K/V
blocks, then runs a single materialized dense attention for each query block.
It preserves index order, tail-block row counts, and duplicate indices. The
fully general `sparse_flash_attention` remains the fallback for Specs,
non-contiguous tensors, optional masks/bias, sampled maps, or an incompatible
K element type.

Final x86 medians:

| case | online sparse | compact materialized | result | workspace |
|---|---:|---:|---:|---:|
| AF3 pair `128x512`, S=4 | 83.64 us | 72.22 us | -13.7% | 18.9 / 49.2 KiB |
| AF3 pair `128x1536`, S=8 | 166.43 us | 139.84 us | -16.0% | 18.9 / 90.2 KiB |
| AF3 diffusion `33x257`, S=4 | 24.65 us | 21.40 us | -13.2% | 20.9 / 57.4 KiB |

The additional workspace is intentional: it buys fewer Matmul invocations,
eliminates online output-rescale passes, and avoids scalar row/block `expf`
calls. `SparseFlashAttention` remains available when workspace is the binding
constraint.

The same implementation is more favorable on ARM/SME:

| case | online sparse | compact materialized | result |
|---|---:|---:|---:|
| AF3 pair `128x512`, S=4 | 55.70 us | 40.50 us | -27.3% |
| AF3 pair `128x1536`, S=8 | 108.31 us | 74.09 us | -31.6% |
| AF3 diffusion `33x257`, S=4 | 18.64 us | 16.04 us | -13.9% |

An ARM `perf stat` run of the final medium dynamic-mask kernel reports 2.72
instructions/cycle and a 1.41% cache-miss/reference ratio. Branch counting is
not supported by the host PMU configuration, but branch misses were available.
This agrees with the tail sweep: the retained SVE kernel is not front-end
stalled enough to justify doubling the masked tail body.

## Validation artifacts

- Expanded x86 dense result: `benchmarks/results/attention_final_x86.json`.
- Hybrid sparse/dynamic result:
  `benchmarks/results/attention_sparse_hybrid_x86.json`.
- ARM pre-policy expanded result:
  `benchmarks/results/attention_expanded_iteration2_arm.json`.
- ARM full result before the final SVE tail specialization:
  `benchmarks/results/attention_final_arm_pre_tail_specialization.json`;
  corrected dynamic-mask medians are in
  `benchmarks/results/attention_dynamic_tail_final_arm.json`.
- Release native test: 11/11 passed.
- Debug ASan test: 12/12 passed, including the non-contiguous dynamic-buffer
  rejection death test; leak detection enabled.

No new Matmul-specific missing branch was isolated in this iteration. The
observed bottlenecks were Attention orchestration, strict/scalar exponential
work, strategy selection, and repeated sparse Matmul setup, so
`docs/internal/matmul-issues.md` did not need a new entry.

## Iteration 8: public API and IBS algorithm boundaries

The initial API mixed mathematical semantics (`dense`/`sparse`), internal
strategies (`materialized`/`flash`), and the IBS routing stage (`dynamic
mask`). It was replaced with four non-aliasing public factories:

- `scaled_dot_product_attention`: exact SDPA with internal automatic,
  materialized, or streaming strategy;
- `sparse_flash_attention`: online attention over an already-final selected
  index sequence `J`;
- `ibs_attention_indexer`: the SparseFold paper's reusable base index
  computation, producing index/weight maps `I/W`;
- `ibs_attention`: optional base indexing, weighted sampling `I/W -> J`, and
  Sparse FlashAttention execution.

The materialized SDPA class moved to `attention_details::MaterializedSDPA` and
its public factory was removed. Benchmarks continue to force it through
`SDPAStrategy::materialized`. Weighted sampling and its scratch/validation
were moved out of `SparseFlashAttention` into `IBSAttention`, matching the
paper's separation between base index computation, index sampling, and
attention computation.

The experimental compact materialized `SparseAttention` and its factory were
removed by API design despite the speedups recorded in Iteration 7. This
intentionally leaves `SparseFlashAttention` as the single general
caller-indexed sparse primitive; the Iteration 7 measurements remain above as
historical data rather than a currently shipped path.

The post-refactor x86 benchmark is stored in
`benchmarks/results/attention_ibs_api_x86.json`. Moving weighted sampling did
not regress complete IBS Attention: the three medians are 33.99, 95.51, and
3.35 us versus the preceding 33.85, 95.67, and 3.46 us. Base indexing is
1.94, 6.31, and 1.06 us. The current suite registers 78 benchmarks after
removing the three compact-materialized sparse cases.

The ARM result is stored in
`benchmarks/results/attention_ibs_api_arm.json`. Complete IBS Attention is
42.89, 118.25, and 8.05 us, matching the pre-refactor 42.98, 118.44, and
7.95 us within normal run variation. Base indexing is 20.28, 59.97, and
4.64 us. The final API passed 11/11 Release tests on both x86 and ARM and
12/12 x86 ASan tests with leak detection enabled.

## Iteration 9: filename alignment and scalar-loop audit

The internal files now match the public algorithm names:

- `Operation.h` became `SDPA.h`;
- `Sparse.h` became `SparseFlashAttention.h`;
- `Dynamic.h` became `IBSAttention.h`.

Every remaining `for`/`while` loop in these files was reviewed by iteration
axis and dependency structure. The useful changes are:

- query/key mask rows are loaded and counted in vector batches once per block,
  replacing repeated `DataAccess::load_scalar` calls in SDPA, Sparse
  FlashAttention, and both IBS block-mean/coarse-score stages;
- the no-key-mask/no-attention-mask/no-bias/non-causal IBS coarse row uses one
  vector scale/store/max traversal. It removes both the key-block scalar score
  loop and the nested query-row pair-count loop from the common inference path;
- block-mean reduction remains vectorized across features, but its row
  reduction now uses four independent vector accumulators. This shortens the
  dependency chain without requiring a row/feature transpose or gather.

The loops deliberately left scalar fall into three groups:

| loop | reason |
|---|---|
| SDPA/Sparse Flash query rows | each row invokes an already-vectorized contiguous key/feature traversal; vectorizing rows would require strided gather or a layout transpose |
| selected K/V block traversal and online-softmax recurrence | the next block consumes the preceding row maximum, normalizer, and output accumulator |
| IBS Top-S insertion and CDF/RNG sampling | ordered tie-breaking, prefix state, and sampling-without-replacement are loop-carried; typical capacities are only 4 or 8 |

An explicit vector-iota replacement for SDPA streaming's dense index-map
initialization was also tested. Alternating builds produced no stable gain:
the two representative scalar medians were 301.86 us (`1x8192` decode) and
97.42 us (`128x512` prefill), while explicit-vector runs varied between
298.12--305.47 us and 97.12--97.60 us. The scalar spelling was retained; this
loop is tiny relative to QK/PV and is a normal compiler auto-vectorization
candidate.

Same-binary x86 medians before and after the retained changes:

| case | before | after | result |
|---|---:|---:|---:|
| IBS indexer, AF3 pair `128x512`, S=4 | 1.937 us | 1.363 us | -29.6% |
| IBS indexer, AF3 pair `128x1536`, S=8 | 6.316 us | 4.301 us | -31.9% |
| IBS indexer, AF3 diffusion `33x257`, S=4 | 1.058 us | 0.939 us | -11.2% |
| complete IBS, AF3 pair `128x512`, S=4 | 34.006 us | 33.684 us | -0.9% |
| complete IBS, AF3 pair `128x1536`, S=8 | 95.529 us | 93.738 us | -1.9% |
| complete IBS, AF3 diffusion `33x257`, S=4 | 3.353 us | 3.301 us | -1.5% |

Sparse Flash medians changed by -0.6%, -0.6%, and -1.4%, respectively, so
the shared activity preload introduced no measured regression. `perf stat` on
the `128x1536` IBS indexer reports 2.51 instructions/cycle and a 0.04% branch
miss rate. The retained vector hot path emits AVX-512 masked loads/stores and
vector FP32 arithmetic in the benchmark object; tail lanes are explicitly
blended to negative infinity after scaling, so zero or negative user scales
cannot turn inactive lanes into NaNs or positive infinity.

The same fast path is substantially more important on SVE, where it removes
the scalar nested query/key traversal that dominated the small coarse map:

| ARM case | before | after | result |
|---|---:|---:|---:|
| IBS indexer, AF3 pair `128x512`, S=4 | 20.298 us | 3.479 us | -82.9% |
| IBS indexer, AF3 pair `128x1536`, S=8 | 60.031 us | 9.414 us | -84.3% |
| IBS indexer, AF3 diffusion `33x257`, S=4 | 4.642 us | 2.230 us | -52.0% |
| complete IBS, AF3 pair `128x512`, S=4 | 42.949 us | 25.780 us | -40.0% |
| complete IBS, AF3 pair `128x1536`, S=8 | 118.366 us | 66.862 us | -43.5% |
| complete IBS, AF3 diffusion `33x257`, S=4 | 8.059 us | 5.735 us | -28.8% |

ARM Sparse Flash changed by -1.8%, -2.9%, and -2.0%. Release validation is
11/11 on both x86 and ARM; x86 Debug ASan is 12/12 with leak detection.

The x86 and ARM results are stored in
`benchmarks/results/attention_vectorized_x86.json` and
`benchmarks/results/attention_vectorized_arm.json`. No missing Matmul special
case was isolated during this audit, so `docs/internal/matmul-issues.md` did
not need a new entry.
