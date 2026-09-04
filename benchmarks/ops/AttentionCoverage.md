# Attention benchmark and test coverage

The attention suite intentionally separates numerical correctness from timing.
`tests/ops/AttentionTest.cpp` uses small scalar-reference cases; this benchmark
uses model-shaped cases and publishes workspace alongside throughput.

The dense matrix spans LLM decode caches from 127 through 8192 tokens,
speculative-decode batches of 8 and 17 queries, prefill chunks from 32 through
2048 keys, 128/192/256-wide model heads, and AF3 pair/single/diffusion/atom
heads of width 24/32/48. Boundary pairs include 127/128, 255/257,
1536/1537, and 4096/4097. Sparse and dynamic coverage includes medium,
N=1536, and arbitrary-tail AF3 cases. Compact materialized sparse and online
sparse paths are reported separately, and dynamic-mask construction is timed
separately from the composed operator.

Every dense model shape registers `automatic`, `streaming`, and `materialized`
strategies. Each shape name states whether Q and K/V N metadata is guaranteed
32-aligned or arbitrary. The coverage group additionally instantiates:

| N metadata | Q/K memory | V memory | output memory |
|---|---|---|---|
| fixed | FP32 | BF16 | FP32 |
| dynamic, 32-aligned | FP16 | FP32 | FP16 |
| arbitrary | BF16 | FP16 | BF16 |

The correctness test also covers raw C++ `bool` masks, additive bias,
compile-time `tensor::nullopt`, all-masked rows, sparse tail maps, deterministic
weighted sampling, dynamic top-S selection, top-left-compatible and
bottom-right causal behavior, and strategy equivalence.

Build and run on the native host:

```sh
cmake -S . -B cmake-build-release \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=ON
cmake --build cmake-build-release \
  --target AttentionTest-Native AttentionBench-Native -j
cmake-build-release/tests/AttentionTest-Native
cmake-build-release/benchmarks/AttentionBench-Native \
  --benchmark_filter='Attention/'
```
