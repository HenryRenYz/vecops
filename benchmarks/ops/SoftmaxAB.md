# Softmax structure A/B

Baseline: `69fd91a` (`Softmax.h`: 745 lines). Candidate: fold based
implementation (`Softmax.h`: 487 lines). Both binaries use the same compiler,
CPU affinity, benchmark cases, and interleaved execution.

## Structural changes

- Removed the public vector Tag from `SoftmaxConfig`; the operator now derives
  `ScalableTag<ComputeType, 0>` internally.
- Centralized input/output access policies and changed two-pass input planning
  to deferred materialization populated by the maximum pass.
- Replaced the handwritten regular max, exponential/sum, and normalization
  loops with vector `fold` recipes and invariant carries.
- Replaced the online full/tail tile, metadata, and output loops with the same
  primitives; fixed a duplicated output `commit()`.
- Removed the x86 raw contiguous access adapters. Equal-width reduction loads
  bypass `with_unordered_access` because unordered conversion has ordered lane
  provenance in this case; a direct-session copy is retained for GCC codegen.
- Changed vector `fold` tail emission to use unmasked complete tail-Tag
  blocks and `first(...)` only for the final partial block.
- Retained one narrow GCC/x86 BF16 store-factor recipe. GCC otherwise lowers
  the large-row normalization loop with lower IPC; this does not affect Clang
  or ARM recipe selection.
- Moved online, packed BF16, and AVX512 recipe selection entirely to type-level
  shape constraints. `required_workspace()` and execution use the same
  `consteval` selector; unsupported backends no longer instantiate an empty
  online callback.
- `SoftmaxConfig::allow_online` is now a static template choice. Unknown shape
  metadata conservatively disables online selection, while the measured
  unbounded AVX512 fallback keeps the four-word throughput recipe.
- `vec::exp/exp_neg` now lower `opt::first(count)` to a mask, allowing fold
  callbacks to forward `active` directly without a Softmax-only helper.
- Benchmark cases are type-level shape lists. Every definition registers a
  `shape_meta:compile_time` (`Const`) callback and an equal-valued
  `shape_meta:runtime` (`Any`) callback. Runtime values are passed through the
  benchmark registry to a rank-only instantiation, preventing `Dims...`
  constant propagation from making the `Any` side accidentally static.

## Local results

GCC 13.3 and Clang 19.1.3 both pass Debug/Release Loop, DataAccess,
LayerNorm, and Softmax tests. Representative x86 A/B results vary with host
frequency, but the repeatable ranges are:

- Medium and large rows are generally between 6% faster and 5% slower across
  FP32/FP16/BF16. The GCC-only BF16 recipe avoids a repeatable large-row
  lowering regression.
- FP32/FP16 64x513 remains the x86 outlier: long-run hardware counters are
  about 1% slower, while short benchmark medians vary from 5-10% slower. The
  first fold version was about 15% slower; using unmasked complete tail blocks
  and preserving the four-word recipe recovered most of the loss.
- BF16 medium/tail cases are typically unchanged or faster.
- Online FP32: `perf stat` on prefill estimate reports about 1.3% fewer cycles
  with essentially unchanged instruction count; benchmark-only swings were
  dominated by frequency/cache noise.
- With identical input values, compile-time metadata is measurably useful.
  In the final Clang harness, FP32 64x768/16x64x4096/64x513 is about
  1%/7%/5% faster than genuine runtime `Any`; FP16 is roughly 0%/1%/4%
  faster. BF16 varies by recipe but retains the same compile-time dispatch
  advantage on large and tail rows. The pair names make these differences
  visible without a runtime branch in the operator.

On 920f-4 with BiSheng Clang 19.1.7, interleaved SVE/SVE2 runs for the same
three shapes and data types remain within about 1% of baseline, except BF16
medium/tail which improve by roughly 2-4%.

On 920f-4, `.bashrc` aliases GCC to Arm GNU Toolchain 15.3.1 rather than the
system GCC 10.3.1. CMake is configured with the alias target's explicit path;
test execution uses the toolchain's bundled loader and libraries because its
GLIBC/libstdc++ requirements are newer than the host defaults.

GCC 15 initially regressed the static FP16 16x64x4096 case by about 12%: its
SVE lowering over-unrolled the constant trip count and expanded paired stores
into a longer dependency chain. Keeping the trip count in a register and using
the single-vector FP16 store recipe only for GCC/SVE restores 4173 us versus
4173-4176 us for the baseline; the same quick binary's genuine `Any` case is
4202 us. BiSheng keeps the generic paired recipe.

The source-line reduction is 258 lines (34.6%) without moving kernel code to a
different file.
