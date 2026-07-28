#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>

#include "BenchmarkUtils.h"
#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace {

#ifndef VECOPS_BENCH_ARCH_CODE
#define VECOPS_BENCH_ARCH_CODE "unknown"
#endif

#ifndef VECOPS_SOURCE_DIR
#define VECOPS_SOURCE_DIR "."
#endif

constexpr nint_t kBlocks = 256;

template <typename T>
T input_value(nint_t i) {
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) {
    return T(float((i % 31) - 15) * 0.125f);
  } else if constexpr (std::is_floating_point_v<T>) {
    return T((i % 31) - 15) * T(0.125);
  } else if constexpr (std::is_unsigned_v<T>) {
    return T((i * 37 + 11) % 251);
  } else {
    return T((i * 29) % 251 - 125);
  }
}

template <typename Ei, typename To, bool Fused>
void bench_load(benchmark::State& state) {
  using OutTag = ScalableTag<To>;
  constexpr OutTag to{};
  using Ti = Rebind<Ei, OutTag>;
  constexpr Ti ti{};
  const nint_t lanes = size(to);
  const nint_t count = lanes * kBlocks;
  const nint_t padding = std::max<nint_t>(64, lanes * 4);
  std::vector<Ei> input(static_cast<size_t>(count + padding + 1));
  std::vector<To> output(static_cast<size_t>(count + padding + 1));
  for (nint_t i = 0; i < count; ++i) input[static_cast<size_t>(i + 1)] = input_value<Ei>(i);

  for (auto _ : state) {
    for (nint_t i = 0; i < count; i += lanes) {
      auto v = [&] {
        if constexpr (Fused) return xconvert_loadu(to, input.data() + i + 1);
        else return xconvert(to, loadu(ti, input.data() + i + 1));
      }();
      storeu(to, output.data() + i + 1, v);
    }
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(state.iterations() * count *
                          static_cast<int64_t>(sizeof(Ei) + sizeof(To)));
}

template <typename Ti, typename Eo, bool Fused>
void bench_store(benchmark::State& state) {
  using InputTag = ScalableTag<Ti>;
  constexpr InputTag ti{};
  using To = Rebind<Eo, InputTag>;
  constexpr To to{};
  const nint_t lanes = size(ti);
  const nint_t count = lanes * kBlocks;
  const nint_t padding = std::max<nint_t>(64, lanes * 4);
  std::vector<Ti> input(static_cast<size_t>(count + padding + 1));
  std::vector<Eo> output(static_cast<size_t>(count + padding + 1));
  for (nint_t i = 0; i < count; ++i) input[static_cast<size_t>(i + 1)] = input_value<Ti>(i);

  for (auto _ : state) {
    for (nint_t i = 0; i < count; i += lanes) {
      auto v = loadu(ti, input.data() + i + 1);
      if constexpr (Fused) {
        xconvert_storeu(ti, output.data() + i + 1, v);
      } else {
        storeu(to, output.data() + i + 1, xconvert(to, v));
      }
    }
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(state.iterations() * count *
                          static_cast<int64_t>(sizeof(Ti) + sizeof(Eo)));
}

template <typename Ei, typename To>
void register_load_pair(const char* pair) {
  const std::string prefix =
      "ConversionLoadStore/load/" + std::string(pair) + "/arch:" +
      VECOPS_BENCH_ARCH_CODE;
  benchmark::RegisterBenchmark((prefix + "/fused").c_str(),
                               &bench_load<Ei, To, true>)
      ->Unit(benchmark::kNanosecond)->MinTime(0.05);
  benchmark::RegisterBenchmark((prefix + "/separate").c_str(),
                               &bench_load<Ei, To, false>)
      ->Unit(benchmark::kNanosecond)->MinTime(0.05);
}

template <typename Ti, typename Eo>
void register_store_pair(const char* pair) {
  const std::string prefix =
      "ConversionLoadStore/store/" + std::string(pair) + "/arch:" +
      VECOPS_BENCH_ARCH_CODE;
  benchmark::RegisterBenchmark((prefix + "/fused").c_str(),
                               &bench_store<Ti, Eo, true>)
      ->Unit(benchmark::kNanosecond)->MinTime(0.05);
  benchmark::RegisterBenchmark((prefix + "/separate").c_str(),
                               &bench_store<Ti, Eo, false>)
      ->Unit(benchmark::kNanosecond)->MinTime(0.05);
}

void register_benchmarks() {
  register_load_pair<uint8_t, uint32_t>("u8-to-u32");
  register_store_pair<int32_t, int8_t>("s32-to-s8");
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  register_load_pair<vecops::bfloat16_t, float32_t>("bf16-to-fp32");
  register_store_pair<float32_t, vecops::bfloat16_t>("fp32-to-bf16");
#endif
}

} // namespace

int main(int argc, char** argv) {
  register_benchmarks();

  const auto default_csv = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, "conversion_load_store", VECOPS_BENCH_ARCH_CODE,
      "csv");
  auto injected = vecops::bench::default_google_benchmark_output_args(
      argc, argv, default_csv, "csv");

  std::vector<char*> args;
  args.reserve(static_cast<size_t>(argc) + injected.size());
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  for (auto& arg : injected) args.push_back(arg.data());
  int bench_argc = static_cast<int>(args.size());

  benchmark::Initialize(&bench_argc, args.data());
  if (benchmark::ReportUnrecognizedArguments(bench_argc, args.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  vecops::bench::print_default_output_path(argc, argv, default_csv);
  return 0;
}
