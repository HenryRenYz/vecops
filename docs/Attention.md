# Attention

`vecops/ops/Attention.h` provides scaled dot-product attention (SDPA),
caller-indexed Sparse FlashAttention, and Index-based Block Sparse (IBS)
Attention. The API operates on one logical head; callers schedule batch and
head axes with their outer execution policy without imposing a B/H layout.

## Public operators

| Factory | Purpose | Score workspace |
|---|---|---|
| `scaled_dot_product_attention(config)` | Exact SDPA with automatic, streaming, or materialized strategy | `O(Bq*Bkv)` streaming or `O(Bq*Lkv)` materialized |
| `sparse_flash_attention(config)` | Online exact softmax over a caller-supplied final block-index sequence | `O(Bq*Bkv + Bq*Dv)` |
| `ibs_attention_indexer(config)` | IBS base index computation: build reusable index and weight maps from coarse Q/K scores | `O(Tq*Tkv + (Tq+Tkv)*D)` |
| `ibs_attention(config)` | Complete IBS Attention: optionally index, sample dynamic candidates, then run Sparse FlashAttention | composition of the preceding workspaces |

All operators use caller-owned `kernel::Workspace`, expose
`required_workspace(...)`, and support an `ExecutionScope` overload so Matmul
resources can remain active across the operation.

Materialized SDPA is an internal strategy, not a separate public operator.
`attention_details::MaterializedSDPA` materializes at most `Bq*Lkv` scores and
is selected through `scaled_dot_product_attention` and `SDPAStrategy`.

The implementation filenames follow the same algorithm boundaries:
`details/attention/SDPA.h`, `SparseFlashAttention.h`, and `IBSAttention.h`;
shared masking, row-softmax, and attention-specific validation remain in
`Common.h`; generic workspace allocation and optional-operand plumbing live
in the tensor/kernel layers.

## Algorithm boundaries

Exact SDPA computes

```text
scores = scale * Q * K^T + bias
probabilities = softmax(mask(scores))
output = probabilities * V
```

The materialized strategy keeps one `[Bq,Lkv]` score panel. The streaming
strategy enumerates all K/V blocks through `SparseFlashAttention` and applies
the exact tiled online-softmax recurrence. They have the same mathematical
semantics, subject only to floating-point reassociation.

`SparseFlashAttention` consumes a final selected-index map `J` of shape
`[ceil_div(Lq,Bq), S]`. It owns neither IBS base maps nor weighted sampling;
it simply applies the online-softmax recurrence to the selected K/V blocks.

IBS Attention follows the two major phases in the SparseFold algorithm:

1. `IBSAttentionIndexer` performs base index computation. It average-pools Q/K
   blocks, computes and normalizes a coarse attention map, then writes the
   Top-S index map `I` and weight map `W`.
2. `IBSAttention` retains static entries, samples weighted dynamic candidates
   from `I/W`, produces the final selected sequence `J`, and invokes
   `SparseFlashAttention`.

Callers may provide previously computed `I/W` maps to `ibs_attention` to skip
base index computation while still resampling dynamic blocks. Omitting both
maps makes the complete operator build and consume them in its workspace.

## Shapes and element types

- Q: `[Lq, Dqk]`
- K: `[Lkv, Dqk]`
- V: `[Lkv, Dv]`
- output: `[Lq, Dv]`
- final Sparse Flash index map `J`: `[ceil_div(Lq, Bq), S]`
- IBS base index/weight maps `I/W`: `[ceil_div(Lq, Bq), S]`

`Dqk` and `Dv` may differ. Sequence lengths and final blocks may be arbitrary.
Tensor memory types are independent of Matmul atom types and are converted by
`tensor::DataAccess`; FP32, FP16, and BF16 input/value/output combinations are
covered by validation.

Sequence metadata may be fixed (`meta::cint<N>`), 32-aligned dynamic
(`meta::dyn<32,...>`), or unconstrained (`meta::Any`). These constraints flow
into Matmul so tail and alignment information remains available at compile
time.

The final Sparse Flash index map and IBS `I/W` maps are algorithm buffers.
They must be direct contiguous rank-two `tensor::Tensor` objects with exact
element types (`int32_t` for indices and the configured score type for
weights); they do not pass through DataAccess materialization or transforms.
The corresponding public constraints are `tensor::TensorOf<T, Rank>` and
`tensor::WritableTensorOf<T, Rank>`, rather than attention-specific wrapper
types.

All internal score panels, temporary index maps, and sampling buffers are
allocated as contiguous Tensor views through `WorkspaceView::allocate_tensor`.
The workspace applies at least `vec::DEFAULT_ALIGNMENT` to every typed
allocation, so attention does not carry a private aligned-allocation path.

## Compile-time optional operands

Query mask, key mask, two-dimensional keep mask, and additive bias are each a
Tensor/InputSpec or `tensor::nullopt`. Presence is therefore a compile-time
property with no nullable-pointer branch in hot loops. Boolean masks use SDPA
keep semantics: nonzero means active. A fully masked row produces zeros.
The common rank constraints and optional bind/workspace/access helpers live in
the tensor layer, so the same compile-time optionality is available to other
operators. Attention declares Q/K/V/output as `*OperandOf<2>` and its vector
and matrix optional operands as `OptionalInputOperandOf<1>` and
`OptionalInputOperandOf<2>`, respectively.

## Configuration

`AttentionConfig<MatmulConfig, Bq, Bkv, Causal, Strategy, ExpAccuracy>` fixes
the Matmul atom, block sizes, causal alignment, SDPA strategy, and vector
exponential accuracy. `ExpAccuracy` defaults to `vec::Accuracy::Fast`; use
`vec::Accuracy::Strict` when its stronger error contract is required.

`SDPAStrategy::automatic` selects an internal strategy once per invocation
using backend-tuned query/key bounds. `materialized` and `streaming` remain
available for deterministic workspace control, tuning, and benchmarking.

`top_left` causal alignment implements prefix attention. `bottom_right`
aligns the last query with the last key and supports KV-cache calls where
`Lq != Lkv`.

IBS Attention uses `selected_blocks`, `random_blocks`,
`static_probability`, `random_probability`, and `minimum_probability`.
Negative weights identify statically retained blocks; positive weights are
dynamic sampling candidates; `(-1, 0)` terminates a base-map row. Index and
weight maps are an all-or-none compile-time pair.

## Validation and performance coverage

`AttentionTest` compares materialized and streaming SDPA with a scalar
reference; covers compile-time masks/bias, fully masked rows, causal modes,
Sparse Flash tails and duplicate indices, deterministic IBS sampling and
indexing, mixed dtypes, and fixed/aligned/arbitrary N metadata.

`AttentionBench` contains LLM decode/prefill and AF3 pair/single/diffusion
shapes. It reports all three SDPA strategies plus dedicated Sparse Flash, IBS
indexer, and complete IBS Attention cases.

The online recurrence follows
[FlashAttention](https://arxiv.org/abs/2205.14135). Boolean keep masks and
causal alignment follow
[PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).
