# Attention

`vecops/ops/Attention.h` provides exact dense attention, caller-selected
block-sparse attention, and dynamically selected block-sparse attention. The
API operates on one logical head; callers can schedule batch and head axes with
their existing outer execution policy without forcing a particular B/H layout.

## Operators

| Factory | Purpose | Score workspace |
|---|---|---|
| `dense_attention(config)` | Exact SDPA with automatic, streaming, or materialized strategy | `O(Bq*Bkv)` streaming or `O(Bq*Lkv)` materialized |
| `dense_materialized_attention(config)` | Explicit materialized baseline/small-problem path | `O(Bq*Lkv)` |
| `sparse_attention(config)` | Compact selected-block materialization with a general online fallback | `O(S*Bkv*(Dqk+Dv) + Bq*S*Bkv)` fast path |
| `sparse_flash_attention(config)` | Online exact softmax over selected K/V blocks | `O(Bq*Bkv)` |
| `dynamic_attention_mask(config)` | Build block index and weight maps from coarse Q/K scores | `O(Tq*Tkv + (Tq+Tkv)*D)` |
| `dynamic_attention(config)` | Build or consume maps and run sampled sparse attention | composition of the preceding workspaces |

All operators use caller-owned `kernel::Workspace`, expose
`required_workspace(...)`, and support an `ExecutionScope` overload so Matmul
resources can be retained across the whole operation.

## Shapes and element types

- Q: `[Lq, Dqk]`
- K: `[Lkv, Dqk]`
- V: `[Lkv, Dv]`
- output: `[Lq, Dv]`
- sparse index/weight map: `[ceil_div(Lq, Bq), S]`

`Dqk` and `Dv` may differ. Sequence lengths and the final blocks may be
arbitrary. Tensor memory types are independent of the Matmul atom types and
are converted by `tensor::DataAccess`; the validation matrix covers FP32,
FP16, and BF16 input/value/output combinations.

The sequence metadata can be fixed (`meta::cint<N>`), 32-aligned dynamic
(`meta::dyn<32,...>`), or unconstrained (`meta::Any`). Keeping constraints in
the input layouts lets Matmul retain compile-time tail/alignment information.

Score scaling, bias application, causal/key/attention masking, stable softmax,
online-softmax probability generation, output rescaling, block means, and
dynamic coarse reductions operate on `vec` vectors. Their full/tail traversal
uses `kernel::loop::fold` with four-vector full blocks. Most loops use a
one-vector masked tail; the dynamic block-mean and coarse-mask traversals use
a measured two-vector tail on x86 and one vector on SVE.
Per-row query-mask control values use DataAccess's native `load_scalar`.

Dynamic index and weight maps are algorithm-private buffers rather than model
operands. Every producer and consumer therefore accepts them only as direct,
rank-two `tensor::Tensor` objects with exact element types (`int32_t` and the
configured score type), and validates row-major contiguity. They do not enter
the DataAccess/InputSpec transformation or materialization pipeline. Writable
maps are required when building dynamic masks; callers may omit both maps at
compile time to let `dynamic_attention` allocate them in its workspace.

## Compile-time optional operands

The query mask, key mask, two-dimensional keep mask, and additive bias are
passed either as a Tensor/InputSpec or as `tensor::nullopt`. Their presence is
therefore a template property. No nullable pointer or presence test remains in
the hot loops. Boolean masks use SDPA keep semantics: nonzero/`true` means the
position participates in attention. A fully masked row produces zeros.

## Configuration

`AttentionConfig<MatmulConfig, Bq, Bkv, Causal, Strategy, ExpAccuracy>` fixes
the Matmul atom, sparse block sizes, causal alignment, dense strategy, and
vector exponential accuracy. `ExpAccuracy` defaults to `vec::Accuracy::Fast`;
callers that need the strict libm-style error contract can select
`vec::Accuracy::Strict` at compile time. `automatic` selects materialization
once per call using backend-tuned query/key bounds; longer key sequences use
streaming online softmax. Explicit `materialized` and `streaming` settings are
available for tuning and benchmarking.

`top_left` causal alignment is standard prefix attention. `bottom_right`
aligns the last query with the last key, which is useful for a KV cache when
`Lq != Lkv`.

Dynamic attention uses `selected_blocks`, `random_blocks`,
`static_probability`, `random_probability`, and `minimum_probability`.
Negative map weights identify statically retained blocks; positive weights
participate in weighted sampling without replacement; `(-1, 0)` terminates a
map row. Supplying index and weight Tensor buffers is an all-or-none
compile-time choice.

For direct contiguous K/V tensors whose K element type already matches the
Matmul B atom, `sparse_attention` copies only the selected K/V blocks into a
compact panel and invokes materialized attention once per query block. This
also preserves duplicate block indices as repeated probability mass. Specs,
non-contiguous tensors, optional masks/bias, sampled maps, and incompatible
element types fall back to `sparse_flash_attention`'s online implementation.

## Validation and performance coverage

`AttentionTest` checks the dense materialized and online results against a
scalar reference, compile-time optional masks/bias, fully masked rows, both
causal alignment behavior, sparse tail blocks, deterministic sampling,
dynamic ranking/composition, mixed dtypes, and fixed/aligned/arbitrary N
metadata.

`AttentionBench` contains LLM decode/prefill, AF3 pair/diffusion, arbitrary
tail, direct sparse, and dynamic sparse scenarios. Dense scenarios publish all
three strategies and workspace size; dedicated coverage cases isolate dtype
conversion and N-metadata variants.

The streaming recurrence is based on the exact tiled online softmax described
by [FlashAttention](https://arxiv.org/abs/2205.14135). Boolean keep masks and
causal alignment follow [PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).
