#include <benchmark/benchmark.h>

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "BenchmarkUtils.h"
#include "vecops/platform/Features.h"
#include "vecops/ops/Attention.h"

#if defined(ARCH_X86_FAMILY)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

using namespace vecops;
using bfloat16_t = vecops::bfloat16_t;
using float16_t = vecops::float16_t;
using float32_t = vecops::float32_t;

#if defined(ARCH_X86_FAMILY)
using BenchAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using BenchAtom = matmul::SME_BF16F32;
#endif

#if defined(ARCH_X86_FAMILY) || defined(HAS_SME)

template <ops::AttentionStrategy Strategy>
using Config = ops::AttentionConfig<
    ops::MatmulConfig<BenchAtom>, 32, 32,
    ops::AttentionCausalMode::none, Strategy>;

struct Case {
  const char* group;
  const char* name;
  nint_t lq;
  nint_t lkv;
  nint_t dqk;
  nint_t dv;
  bool lq_aligned32;
  bool lkv_aligned32;
};

constexpr Case Cases[] = {
    // Llama/Qwen-style 128-wide heads: decode, speculative decode, and
    // prompt chunks around common cache and paging boundaries.
    {"llm_decode", "short_cache_tail", 1, 127, 128, 128, false, false},
    {"llm_decode", "short_cache_aligned", 1, 128, 128, 128, false, true},
    {"llm_decode", "page_tail", 1, 511, 128, 128, false, false},
    {"llm_decode", "mqa_cache", 1, 2048, 128, 128, false, true},
    {"llm_decode", "long_cache", 1, 8192, 128, 128, false, true},
    {"llm_spec_decode", "eight_tokens", 8, 2048, 128, 128, false, true},
    {"llm_spec_decode", "tail_both", 17, 4097, 128, 128, false, false},
    {"llm_prefill", "short_prompt", 32, 32, 128, 128, true, true},
    {"llm_prefill", "chunked", 128, 512, 128, 128, true, true},
    {"llm_prefill", "long_chunk", 256, 2048, 128, 128, true, true},
    {"llm_prefill", "tail_both", 257, 1025, 128, 128, false, false},
    // Gemma 2 uses 256-wide heads; DeepSeek V3 has 192 Q/K dimensions and
    // 128 value dimensions after concatenating its no-PE and RoPE portions.
    {"llm_prefill", "gemma2_head", 64, 512, 256, 256, true, true},
    {"llm_decode", "deepseek_mla", 1, 4096, 192, 128, false, true},

    // AlphaFold 3 model parameters expose 32-wide pair heads, 24-wide
    // Pairformer single heads, and 48-wide diffusion-transformer heads.
    // Pair attention uses query chunks of 128 up to N=1536 and 32 beyond it.
    {"af3_pair", "small_tail", 32, 255, 32, 32, true, false},
    {"af3_pair", "medium", 128, 512, 32, 32, true, true},
    {"af3_pair", "chunk_boundary", 128, 1536, 32, 32, true, true},
    {"af3_pair", "above_boundary_tail", 32, 1537, 32, 32, true, false},
    {"af3_single", "pairformer", 128, 512, 24, 24, true, true},
    {"af3_diffusion", "token_transformer", 128, 512, 48, 48, true, true},
    {"af3_diffusion", "token_tail", 33, 257, 48, 48, false, false},
    {"af3_atom", "local_transformer", 32, 128, 32, 32, true, true},

    // Small fully arbitrary dimensions isolate vector and Matmul tails.
    {"tail", "arbitrary", 33, 65, 24, 48, false, false},
};

constexpr Case SparseCases[] = {
    {"af3_pair", "medium", 128, 512, 32, 32, true, true},
    {"af3_pair", "chunk_boundary", 128, 1536, 32, 32, true, true},
    {"af3_diffusion", "token_tail", 33, 257, 48, 48, false, false},
};

template <ops::AttentionStrategy Strategy,
          typename QK = bfloat16_t, typename Value = bfloat16_t,
          typename Output = float32_t,
          typename LqExtent = meta::Any, typename LkvExtent = meta::Any>
void run_attention(
    benchmark::State& state, Case test_case,
    LqExtent lq_extent = {}, LkvExtent lkv_extent = {}) {
  const nint_t q_elements = test_case.lq * test_case.dqk;
  const nint_t k_elements = test_case.lkv * test_case.dqk;
  const nint_t v_elements = test_case.lkv * test_case.dv;
  const nint_t o_elements = test_case.lq * test_case.dv;
  std::vector<QK> q(static_cast<std::size_t>(q_elements));
  std::vector<QK> k(static_cast<std::size_t>(k_elements));
  std::vector<Value> v(static_cast<std::size_t>(v_elements));
  std::vector<Output> out(static_cast<std::size_t>(o_elements));
  for (nint_t i = 0; i < q_elements; ++i)
    q[static_cast<std::size_t>(i)] = QK(float(i % 17 - 8) / 32);
  for (nint_t i = 0; i < k_elements; ++i)
    k[static_cast<std::size_t>(i)] = QK(float(i % 19 - 9) / 32);
  for (nint_t i = 0; i < v_elements; ++i)
    v[static_cast<std::size_t>(i)] = Value(float(i % 23 - 11) / 32);
  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      lq_extent, meta::Any{test_case.dqk}));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      lkv_extent, meta::Any{test_case.dqk}));
  auto vt = tensor::make_tensor(v.data(), tensor::make_shape(
      lkv_extent, meta::Any{test_case.dv}));
  auto ot = tensor::make_tensor(out.data(), tensor::make_shape(
      lq_extent, meta::Any{test_case.dv}));
  auto op = ops::dense_attention(Config<Strategy>{});
  const nint_t workspace_bytes = op.required_workspace(qt, kt, vt, ot);
  kernel::Workspace storage(workspace_bytes);
  auto workspace = storage.view();
  const float scale = 1.0f / std::sqrt(static_cast<float>(test_case.dqk));
  for (auto _ : state) {
    op(workspace, qt, kt, vt, ot, scale);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const double qk_flops = 2.0 * test_case.lq * test_case.lkv * test_case.dqk;
  const double pv_flops = 2.0 * test_case.lq * test_case.lkv * test_case.dv;
  state.counters["effective_gflops"] = benchmark::Counter(
      qk_flops + pv_flops, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::OneK::kIs1000);
  state.counters["tokens_per_second"] = benchmark::Counter(
      static_cast<double>(test_case.lq),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["workspace_bytes"] = static_cast<double>(workspace_bytes);
}

template <ops::AttentionStrategy Strategy>
void run_attention_metadata(benchmark::State& state, Case test_case) {
  if (test_case.lq_aligned32 && test_case.lkv_aligned32) {
    run_attention<Strategy>(
        state, test_case, meta::dyn<32>(test_case.lq),
        meta::dyn<32>(test_case.lkv));
  } else if (test_case.lq_aligned32) {
    run_attention<Strategy>(
        state, test_case, meta::dyn<32>(test_case.lq),
        meta::Any{test_case.lkv});
  } else if (test_case.lkv_aligned32) {
    run_attention<Strategy>(
        state, test_case, meta::Any{test_case.lq},
        meta::dyn<32>(test_case.lkv));
  } else {
    run_attention<Strategy>(
        state, test_case, meta::Any{test_case.lq},
        meta::Any{test_case.lkv});
  }
}

enum class SparseMode {
  materialized,
  direct,
  mask,
  dynamic,
};

template <SparseMode Mode>
void run_sparse_attention(
    benchmark::State& state, Case test_case, nint_t selected) {
  const nint_t tq = ceil_div(test_case.lq, nint_t{32});
  const nint_t tkv = ceil_div(test_case.lkv, nint_t{32});
  std::vector<bfloat16_t> q(test_case.lq * test_case.dqk);
  std::vector<bfloat16_t> k(test_case.lkv * test_case.dqk);
  std::vector<bfloat16_t> v(test_case.lkv * test_case.dv);
  std::vector<float32_t> out(test_case.lq * test_case.dv);
  std::vector<int32_t> index(tq * selected);
  std::vector<float32_t> weight(tq * selected);
  for (nint_t row = 0; row < tq; ++row) {
    for (nint_t slot = 0; slot < selected; ++slot) {
      index[static_cast<std::size_t>(row * selected + slot)] =
          static_cast<int32_t>((row * selected + slot) % tkv);
    }
  }
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = bfloat16_t(float(i % 17 - 8) / 32);
  for (std::size_t i = 0; i < k.size(); ++i)
    k[i] = bfloat16_t(float(i % 19 - 9) / 32);
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = bfloat16_t(float(i % 23 - 11) / 32);
  auto qt = tensor::make_tensor(q.data(), tensor::make_shape(
      meta::Any{test_case.lq}, meta::Any{test_case.dqk}));
  auto kt = tensor::make_tensor(k.data(), tensor::make_shape(
      meta::Any{test_case.lkv}, meta::Any{test_case.dqk}));
  auto vt = tensor::make_tensor(v.data(), tensor::make_shape(
      meta::Any{test_case.lkv}, meta::Any{test_case.dv}));
  auto ot = tensor::make_tensor(out.data(), tensor::make_shape(
      meta::Any{test_case.lq}, meta::Any{test_case.dv}));
  auto it = tensor::make_tensor(index.data(), tensor::make_shape(
      meta::Any{tq}, meta::Any{selected}));
  auto wt = tensor::make_tensor(weight.data(), tensor::make_shape(
      meta::Any{tq}, meta::Any{selected}));
  Config<ops::AttentionStrategy::automatic> config{};
  config.selected_blocks = selected;
  config.static_probability = 1.0f;
  config.random_probability = 1.0f;
  auto op = [&] {
    if constexpr (Mode == SparseMode::dynamic)
      return ops::dynamic_attention(config);
    else if constexpr (Mode == SparseMode::mask)
      return ops::dynamic_attention_mask(config);
    else if constexpr (Mode == SparseMode::materialized)
      return ops::sparse_attention(config);
    else
      return ops::sparse_flash_attention(config);
  }();
  const nint_t workspace_bytes = [&] {
    if constexpr (Mode == SparseMode::dynamic)
      return op.required_workspace(qt, kt, vt, ot);
    else if constexpr (Mode == SparseMode::mask)
      return op.required_workspace(qt, kt, it, wt);
    else
      return op.required_workspace(qt, kt, vt, it, ot);
  }();
  kernel::Workspace storage(workspace_bytes);
  auto workspace = storage.view();
  std::mt19937 rng(17);
  const float scale = 1.0f / std::sqrt(float(test_case.dqk));
  for (auto _ : state) {
    if constexpr (Mode == SparseMode::dynamic)
      op(workspace, qt, kt, vt, ot, scale, rng);
    else if constexpr (Mode == SparseMode::mask)
      op(workspace, qt, kt, it, wt, scale);
    else
      op(workspace, qt, kt, vt, it, ot, scale);
    if constexpr (Mode == SparseMode::mask) {
      benchmark::DoNotOptimize(index.data());
      benchmark::DoNotOptimize(weight.data());
    } else {
      benchmark::DoNotOptimize(out.data());
    }
    benchmark::ClobberMemory();
  }
  if constexpr (Mode != SparseMode::mask) {
    const double selected_keys = selected * 32.0;
    state.counters["effective_gflops"] = benchmark::Counter(
        2.0 * test_case.lq * selected_keys *
            (test_case.dqk + test_case.dv),
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
  }
  state.counters["workspace_bytes"] = static_cast<double>(workspace_bytes);
}

const char* metadata_label(const Case& test_case) {
  if (test_case.lq_aligned32 && test_case.lkv_aligned32)
    return "n:aligned32xaligned32";
  if (test_case.lq_aligned32) return "n:aligned32xany";
  if (test_case.lkv_aligned32) return "n:anyxaligned32";
  return "n:anyxany";
}

void register_benchmarks() {
  for (const auto& test_case : Cases) {
    const std::string base = std::string("Attention/") + test_case.group +
        "/" + test_case.name + "/shape:" +
        std::to_string(test_case.lq) + "x" +
        std::to_string(test_case.lkv) + "x" +
        std::to_string(test_case.dqk) + "x" +
        std::to_string(test_case.dv) + "/" + metadata_label(test_case);
#define VECOPS_REGISTER_ATTENTION_STRATEGY(Strategy, Label)                 \
    vecops::bench::configure_registered_benchmark(                         \
        benchmark::RegisterBenchmark(                                      \
            (base + "/" Label).c_str(),                                   \
            [test_case](benchmark::State& state) {                          \
              run_attention_metadata<ops::AttentionStrategy::Strategy>(     \
                  state, test_case);                                         \
            }),                                                             \
        0.2, 3)
    VECOPS_REGISTER_ATTENTION_STRATEGY(automatic, "automatic");
    VECOPS_REGISTER_ATTENTION_STRATEGY(streaming, "streaming");
    VECOPS_REGISTER_ATTENTION_STRATEGY(materialized, "materialized");
#undef VECOPS_REGISTER_ATTENTION_STRATEGY
  }

  constexpr Case MetaCase{
      "coverage", "metadata", 32, 64, 48, 32, true, true};
  vecops::bench::configure_registered_benchmark(
      benchmark::RegisterBenchmark(
          "Attention/coverage/fixed_n/fp32_bf16_fp32",
          [MetaCase](benchmark::State& state) {
            run_attention<ops::AttentionStrategy::automatic,
                          float32_t, bfloat16_t, float32_t>(
                state, MetaCase, meta::cint<32>, meta::cint<64>);
          }), 0.2, 3);
  vecops::bench::configure_registered_benchmark(
      benchmark::RegisterBenchmark(
          "Attention/coverage/aligned32_n/fp16_fp32_fp16",
          [MetaCase](benchmark::State& state) {
            run_attention<ops::AttentionStrategy::automatic,
                          float16_t, float32_t, float16_t>(
                state, MetaCase, meta::dyn<32, 32, 128>(32),
                meta::dyn<32, 32, 128>(64));
          }), 0.2, 3);
  vecops::bench::configure_registered_benchmark(
      benchmark::RegisterBenchmark(
          "Attention/coverage/any_n/bf16_fp16_bf16",
          [MetaCase](benchmark::State& state) {
            run_attention<ops::AttentionStrategy::automatic,
                          bfloat16_t, float16_t, bfloat16_t>(
                state, MetaCase, meta::Any{32}, meta::Any{64});
          }), 0.2, 3);
  for (const auto& test_case : SparseCases) {
    const nint_t selected = test_case.lkv >= 1024 ? 8 : 4;
    const std::string base = std::string("Attention/") + test_case.group +
        "/" + test_case.name + "/shape:" +
        std::to_string(test_case.lq) + "x" +
        std::to_string(test_case.lkv) + "x" +
        std::to_string(test_case.dqk) + "x" +
        std::to_string(test_case.dv) + "/selected:" +
        std::to_string(selected);
#define VECOPS_REGISTER_SPARSE_MODE(Mode, Label)                            \
    vecops::bench::configure_registered_benchmark(                         \
        benchmark::RegisterBenchmark(                                      \
            (base + "/" Label).c_str(),                                   \
            [test_case, selected](benchmark::State& state) {                \
              run_sparse_attention<SparseMode::Mode>(                      \
                  state, test_case, selected);                              \
            }),                                                             \
        0.2, 3)
    VECOPS_REGISTER_SPARSE_MODE(materialized, "sparse_materialized");
    VECOPS_REGISTER_SPARSE_MODE(direct, "sparse_flash");
    VECOPS_REGISTER_SPARSE_MODE(mask, "dynamic_mask");
    VECOPS_REGISTER_SPARSE_MODE(dynamic, "dynamic");
#undef VECOPS_REGISTER_SPARSE_MODE
  }
}

#endif

bool enable_architecture() {
#if defined(ARCH_X86_FAMILY)
  constexpr long ArchRequestXcompPerm = 0x1023;
  constexpr long XfeatureTileData = 18;
  return syscall(SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
#else
  return true;
#endif
}

} // namespace

int main(int argc, char** argv) {
  if (!enable_architecture()) {
    std::cerr << "Unable to enable attention matrix resources: "
              << std::strerror(errno) << '\n';
    return 1;
  }
#if defined(ARCH_X86_FAMILY) || defined(HAS_SME)
  register_benchmarks();
#endif
  const auto output = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, "attention", VECOPS_BENCH_ARCH_CODE, "csv");
  auto injected = vecops::bench::default_google_benchmark_output_args(
      argc, argv, output, "csv");
  std::vector<char*> args;
  args.reserve(static_cast<std::size_t>(argc) + injected.size());
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  for (auto& arg : injected) args.push_back(arg.data());
  int bench_argc = static_cast<int>(args.size());
  benchmark::Initialize(&bench_argc, args.data());
  if (benchmark::ReportUnrecognizedArguments(bench_argc, args.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  vecops::bench::print_default_output_path(argc, argv, output);
  return 0;
}
