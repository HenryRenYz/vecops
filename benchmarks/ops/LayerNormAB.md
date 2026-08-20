# LayerNorm structure compression A/B (2026-08-21)

`LayerNorm.h` was reduced from 356 to 248 lines. Each performance comparison
used three interleaved process runs per binary and three registered repetitions
per benchmark. A candidate failed when any case regressed by more than
`max(1%, 2 * max(baseline CV, candidate CV))`.

| Experiment | Compiler / host | Result | Decision |
|---|---|---|---|
| unified `with_operands` vs direct fast path | GCC 13.3 / local x86 | 4/51 regressions, +0.262% geometric mean | lower all-direct scope elision into DataAccess |
| per-row params vs conditional | GCC 13.3 / local x86 | 3/51 regressions, +0.919% geometric mean | reject globally per-row; use hoisted binding |
| per-row params vs conditional | Clang 19.1.3 / local x86 | 0/24 regressions, -1.407% geometric mean | compiler behavior differs from GCC |
| hoisted params vs conditional | BiSheng Clang 19.1.7 / 920f-4 | 0/24 regressions, -0.157% geometric mean | use hoisted binding on every backend |
| per-row params vs conditional | BiSheng Clang 19.1.7 / 920f-4 | 0/24 regressions, -0.225% geometric mean | improvement is below noise and conflicts with GCC |
| forced inline vs x86 noinline | GCC 13.3 / local x86 | 14/24 regressions, +25.099% geometric mean | keep x86 noinline |
| forced inline vs x86 noinline | Clang 19.1.3 / local x86 | 1/24 regression, -1.494% geometric mean | keep cross-compiler x86 boundary |
| forced noinline vs ARM inline | BiSheng Clang 19.1.7 / 920f-4 | 14/24 regressions, +5.293% geometric mean | keep ARM inline |

The only GCC on 920f-4 is GCC 10.3.1. It cannot build VecOps' VLA SVE
representation, which requires GCC 13+ predicate tuples. Its fixed 512-bit SVE
fallback also lacks the BF16 tuple intrinsics used by the current headers.
Consequently no ARM GCC timing was fabricated, and compiler-sensitive inline
placement remains specialized.

The follow-up implementation moved the direct optimization into
`kernel::with_operands`: if every binding resolves to direct, the group creates
no workspace mark and invokes the callback through straight-line direct
bindings. A materialized group retains the original mark/rewind lifetime. The
callback contract now reserves workspace authority for operand sessions and
forbids direct mutation of the supplied `WorkspaceView`.

After this lowering, GCC emitted the same function size for every checked FP32
LayerNorm rank as the former operator-local direct branch (for example, 0x4fb
bytes for rank two), with no workspace rewind store in the all-direct path.
Three interleaved runs of representative FP32/FP16/BF16 medium, tail, and LLM
decode cases remained within run-to-run noise. LayerNorm and Softmax therefore
use the common operand path, while the independent x86 raw-contiguous Softmax
specialization remains intact.

Final functional validation passed locally with GCC 13.3 and Clang 19.1.3 in
Debug and Release for Scalar, AVX512, and Native. On 920f-4, the explicit
BiSheng Clang Release build passed Layout, Tensor, and the SVE/SVE2 DataAccess,
LayerNorm, and Softmax suites. The remote Debug build was stopped after the
shared `/home` filesystem ran out of space while linking; its regenerable build
artifacts were removed after local Debug and remote Release coverage passed.
