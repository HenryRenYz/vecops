//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_BENCHMARKS_OPS_MATMUL_PACK_BENCH_COMMON_H
#define VECOPS_BENCHMARKS_OPS_MATMUL_PACK_BENCH_COMMON_H

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "BenchmarkUtils.h"
#include "vecops/matmul/Packing.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/ops/MatmulPack.h"

#ifndef VECOPS_BENCH_ARCH_CODE
#define VECOPS_BENCH_ARCH_CODE "unknown"
#endif

#ifndef VECOPS_SOURCE_DIR
#define VECOPS_SOURCE_DIR "."
#endif

namespace vecops::bench::matmul_pack {

using namespace ::vecops::meta;
using namespace ::vecops::tensor;

struct GenericGroup { static constexpr const char* name = "generic"; };
struct ActivationGroup { static constexpr const char* name = "llm_activation"; };
struct AttentionWeightGroup { static constexpr const char* name = "llm_attention_weight"; };
struct FFNWeightGroup { static constexpr const char* name = "llm_ffn_weight"; };
struct ExpertWeightGroup { static constexpr const char* name = "llm_expert_weight"; };
struct DTypeGroup { static constexpr const char* name = "dtype_coverage"; };

template <typename Group, typename Name,
          nint_t Spatial, nint_t K,
          bool PackA = true, bool PackB = true>
struct PackCase {
  using GroupType = Group;
  using NameType = Name;
  static constexpr nint_t spatial = Spatial;
  static constexpr nint_t k = K;
  static constexpr nint_t elements = Spatial * K;
  static constexpr bool pack_a = PackA;
  static constexpr bool pack_b = PackB;

  template <bool ConstShape>
  static constexpr auto shape() {
    if constexpr (ConstShape) {
      return make_shape(cint<Spatial>, cint<K>);
    } else {
      return make_shape(Any{Spatial}, Any{K});
    }
  }
};

template <typename... Cases>
struct CaseList {};

#define VECOPS_PACK_CASE_NAME(TypeName, Value) \
  struct TypeName { static constexpr const char* name = Value; }

VECOPS_PACK_CASE_NAME(TinyName, "tiny_tail");
VECOPS_PACK_CASE_NAME(TileTailName, "tile_tail");
VECOPS_PACK_CASE_NAME(SmallName, "small");
VECOPS_PACK_CASE_NAME(WideName, "wide");
VECOPS_PACK_CASE_NAME(MediumTailName, "medium_tail");
VECOPS_PACK_CASE_NAME(LargeTailName, "large_tail");
VECOPS_PACK_CASE_NAME(Qwen05DecodeName, "qwen2.5_0.5b_decode");
VECOPS_PACK_CASE_NAME(Qwen05PrefillName, "qwen2.5_0.5b_prefill");
VECOPS_PACK_CASE_NAME(Qwen7DecodeName, "qwen2.5_7b_decode");
VECOPS_PACK_CASE_NAME(Qwen7BatchName, "qwen2.5_7b_batch");
VECOPS_PACK_CASE_NAME(Qwen7PrefillName, "qwen2.5_7b_prefill");
VECOPS_PACK_CASE_NAME(DeepSeekBatchName, "deepseek_v3_batch");
VECOPS_PACK_CASE_NAME(Qwen72DecodeName, "qwen2.5_72b_decode");
VECOPS_PACK_CASE_NAME(Qwen72PrefillName, "qwen2.5_72b_prefill");
VECOPS_PACK_CASE_NAME(Qwen05QKVName, "qwen2.5_0.5b_qkv");
VECOPS_PACK_CASE_NAME(Qwen05UpName, "qwen2.5_0.5b_gate_up");
VECOPS_PACK_CASE_NAME(Qwen05DownName, "qwen2.5_0.5b_down");
VECOPS_PACK_CASE_NAME(Qwen7QKVName, "qwen2.5_7b_qkv");
VECOPS_PACK_CASE_NAME(Qwen7UpName, "qwen2.5_7b_gate_up");
VECOPS_PACK_CASE_NAME(Qwen7DownName, "qwen2.5_7b_down");
VECOPS_PACK_CASE_NAME(Qwen72QKVName, "qwen2.5_72b_qkv");
VECOPS_PACK_CASE_NAME(DeepSeekDenseName, "deepseek_v3_dense_gate_up");
VECOPS_PACK_CASE_NAME(DeepSeekExpertUpName, "deepseek_v3_expert_gate_up");
VECOPS_PACK_CASE_NAME(DeepSeekExpertDownName, "deepseek_v3_expert_down");
VECOPS_PACK_CASE_NAME(DTypeProbeName, "tail_probe");

#undef VECOPS_PACK_CASE_NAME

// Model dimensions come from the model publishers' config.json files:
//   https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json
//   https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
//   https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json
//   https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
//   Qwen2.5 0.5B: hidden=896, intermediate=4864, heads=14, kv_heads=2
//   Qwen2.5 7B:   hidden=3584, intermediate=18944, heads=28, kv_heads=4
//   Qwen2.5 72B:  hidden=8192, intermediate=29568, heads=64, kv_heads=8
//   DeepSeek-V3:  hidden=7168, intermediate=18432, expert_intermediate=2048
// QKV rows are hidden + 2 * kv_heads * head_dim. Activation cases represent
// decode, modest batching, and prefill. Weight cases are [output, input].
using RepresentativeCases = CaseList<
    PackCase<GenericGroup, TinyName, 1, 1>,
    PackCase<GenericGroup, TileTailName, 17, 65>,
    PackCase<GenericGroup, SmallName, 64, 256>,
    PackCase<GenericGroup, WideName, 1024, 256>,
    PackCase<GenericGroup, MediumTailName, 513, 1000>,
    PackCase<GenericGroup, LargeTailName, 1025, 1027>,

    PackCase<ActivationGroup, Qwen05DecodeName, 1, 896, true, false>,
    PackCase<ActivationGroup, Qwen05PrefillName, 2048, 896, true, false>,
    PackCase<ActivationGroup, Qwen7DecodeName, 1, 3584, true, false>,
    PackCase<ActivationGroup, Qwen7BatchName, 128, 3584, true, false>,
    PackCase<ActivationGroup, Qwen7PrefillName, 2048, 3584, true, false>,
    PackCase<ActivationGroup, DeepSeekBatchName, 128, 7168, true, false>,
    PackCase<ActivationGroup, Qwen72DecodeName, 1, 8192, true, false>,
    PackCase<ActivationGroup, Qwen72PrefillName, 2048, 8192, true, false>,

    PackCase<AttentionWeightGroup, Qwen05QKVName,
             1152, 896, false, true>,
    PackCase<FFNWeightGroup, Qwen05UpName,
             4864, 896, false, true>,
    PackCase<FFNWeightGroup, Qwen05DownName,
             896, 4864, false, true>,
    PackCase<AttentionWeightGroup, Qwen7QKVName,
             4608, 3584, false, true>,
    PackCase<FFNWeightGroup, Qwen7UpName,
             18944, 3584, false, true>,
    PackCase<FFNWeightGroup, Qwen7DownName,
             3584, 18944, false, true>,
    PackCase<AttentionWeightGroup, Qwen72QKVName,
             10240, 8192, false, true>,
    PackCase<FFNWeightGroup, DeepSeekDenseName,
             18432, 7168, false, true>,
    PackCase<ExpertWeightGroup, DeepSeekExpertUpName,
             2048, 7168, false, true>,
    PackCase<ExpertWeightGroup, DeepSeekExpertDownName,
             7168, 2048, false, true>>;

using DTypeProbeCases = CaseList<
    PackCase<DTypeGroup, DTypeProbeName, 17, 65>>;

using SecondaryCases = CaseList<
    PackCase<DTypeGroup, DTypeProbeName, 17, 65>,
    PackCase<GenericGroup, SmallName, 64, 256>,
    PackCase<GenericGroup, MediumTailName, 513, 1000>,
    PackCase<GenericGroup, LargeTailName, 1025, 1027>>;

template <typename T>
const char* dtype_name() {
  if constexpr (std::same_as<T, bfloat16_t>) return "bf16";
  if constexpr (std::same_as<T, float16_t>) return "fp16";
  if constexpr (std::same_as<T, float32_t>) return "fp32";
  if constexpr (std::same_as<T, float64_t>) return "fp64";
  if constexpr (std::same_as<T, int8_t>) return "int8";
  if constexpr (std::same_as<T, uint8_t>) return "uint8";
  return "unknown";
}

template <gemm::Operand Side>
const char* operand_name() {
  if constexpr (Side == gemm::Operand::A) return "A";
  return "B";
}

template <bool ConstShape>
const char* shape_metadata_name() {
  if constexpr (ConstShape) return "Const";
  return "Dynamic";
}

template <typename Atom>
struct AtomName;

template <typename Atom>
const char* atom_name() {
  return AtomName<Atom>::value;
}

template <typename Case>
std::string shape_name() {
  return std::to_string(Case::spatial) + "x" + std::to_string(Case::k);
}

template <typename T>
void fill_input(std::vector<T>& input) {
  for (std::size_t i = 0; i < input.size(); ++i) {
    if constexpr (std::same_as<T, int8_t> || std::same_as<T, uint8_t>) {
      input[i] = static_cast<T>((i * 13 + 7) % 127);
    } else {
      const float value =
          static_cast<float>(static_cast<int>(i % 251) - 125) * 0.015625f;
      input[i] = static_cast<T>(value);
    }
  }
}

template <typename Atom, gemm::Operand Side,
          typename Case, bool ConstShape>
void run_case(benchmark::State& state) {
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  std::vector<T> input(static_cast<std::size_t>(Case::elements));
  fill_input(input);
  auto input_shape = Case::template shape<ConstShape>();
  auto input_layout = make_layout(input_shape);
  auto input_tensor = make_tensor(input.data(), input_layout);
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  const nint_t output_elements = numel(output_layout);
  const nint_t output_bytes = output_elements * static_cast<nint_t>(sizeof(T));
  kernel::Workspace output_storage(output_bytes + 64);
  auto output_view = output_storage.view();
  auto* output_data = static_cast<T*>(output_view.allocate(output_bytes, 64));
  auto output_tensor = make_tensor(output_data, output_layout);
  auto operation = ops::make_matmul_pack<Atom, Side>(
      input_tensor, output_tensor);
  ExecutionSession execution{};

  operation(execution);
  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    operation(execution);
    benchmark::DoNotOptimize(output_data);
    benchmark::ClobberMemory();
  }

  const int64_t logical_bytes = static_cast<int64_t>(
      Case::elements * static_cast<nint_t>(sizeof(T)) + output_bytes);
  state.SetItemsProcessed(state.iterations() * Case::elements);
  state.SetBytesProcessed(state.iterations() * logical_bytes);
  state.counters["spatial"] = benchmark::Counter(double(Case::spatial));
  state.counters["k"] = benchmark::Counter(double(Case::k));
  state.counters["input_elements"] =
      benchmark::Counter(double(Case::elements));
  state.counters["packed_elements"] =
      benchmark::Counter(double(output_elements));
  state.counters["padding_ratio"] = benchmark::Counter(
      double(output_elements) / double(Case::elements));
}

template <typename Atom, gemm::Operand Side,
          typename Case, bool ConstShape>
std::string benchmark_name() {
  using T = typename gemm::packing_t<Atom, Side>::Element;
  return
      "MatmulPack/" + std::string(Case::GroupType::name) +
      "/case:" + Case::NameType::name +
      "/operand:" + operand_name<Side>() +
      "/shape:" + shape_name<Case>() +
      "/extent:" + shape_metadata_name<ConstShape>() +
      "/dtype:" + dtype_name<T>() +
      "/atom:" + atom_name<Atom>() +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
}

template <typename Atom, gemm::Operand Side,
          typename Case, bool ConstShape>
void register_shape_metadata() {
  const auto name = benchmark_name<Atom, Side, Case, ConstShape>();
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), &run_case<Atom, Side, Case, ConstShape>)
      ->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <typename Atom, gemm::Operand Side, typename Case>
void register_operand() {
  register_shape_metadata<Atom, Side, Case, true>();
  register_shape_metadata<Atom, Side, Case, false>();
}

template <typename Atom, typename Case>
void register_case() {
  if constexpr (Case::pack_a) {
    register_operand<Atom, gemm::Operand::A, Case>();
  }
  if constexpr (Case::pack_b) {
    register_operand<Atom, gemm::Operand::B, Case>();
  }
}

template <typename Atom, typename... Cases>
void register_cases(CaseList<Cases...>) {
  (register_case<Atom, Cases>(), ...);
}

inline int run_benchmarks(
    int argc, char** argv, const char* result_stem) {
  const auto default_csv = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, result_stem, VECOPS_BENCH_ARCH_CODE, "csv");
  auto injected = vecops::bench::default_google_benchmark_output_args(
      argc, argv, default_csv, "csv");
  std::vector<char*> args;
  args.reserve(static_cast<std::size_t>(argc) + injected.size());
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  for (auto& arg : injected) args.push_back(arg.data());
  int bench_argc = static_cast<int>(args.size());
  benchmark::Initialize(&bench_argc, args.data());
  if (benchmark::ReportUnrecognizedArguments(bench_argc, args.data())) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  vecops::bench::print_default_output_path(argc, argv, default_csv);
  return 0;
}

} // namespace vecops::bench::matmul_pack

#endif // VECOPS_BENCHMARKS_OPS_MATMUL_PACK_BENCH_COMMON_H
