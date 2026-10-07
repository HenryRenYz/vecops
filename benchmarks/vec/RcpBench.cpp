// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <cmath>
#include <cstddef>
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

template <typename E>
const char* dtype_name() {
  if constexpr (std::is_same_v<E, float32_t>) return "fp32";
  if constexpr (std::is_same_v<E, float64_t>) return "f64";
  if constexpr (std::is_same_v<E, vecops::float16_t>) return "fp16";
  return "bf16";
}

template <Accuracy tier>
const char* mode_name() {
  if constexpr (tier == Accuracy::Strict) return "strict";
  if constexpr (tier == Accuracy::Fast) return "fast";
  return "estimate";
}

// "general" keeps every lane on the inline refinement path; "wide" mixes in
// tiny and huge magnitudes so the special routine is exercised as well.
constexpr double kGeneralLow = -10.0;
constexpr double kGeneralHigh = 6.0;
constexpr double kWideLow = -135.0;
constexpr double kWideHigh = 130.0;

const char* domain_name(bool wide) { return wide ? "wide" : "general"; }

template <Accuracy tier, VectorTag Tag>
Vec<Tag> apply_rcp(Tag tag, Vec<Tag> x) {
  return rcp(tag, x, opt::math::accuracy<tier>);
}

template <Accuracy tier, bool wide, typename E>
void bench_rcp(benchmark::State& state) {
  const ScalableTag<E> t;
  const nint_t lanes = size(t);
  const nint_t count = lanes * kBlocks;
  std::vector<E> input(static_cast<size_t>(count));
  std::vector<E> output(static_cast<size_t>(count));
  constexpr double low = wide ? kWideLow : kGeneralLow;
  constexpr double high = wide ? kWideHigh : kGeneralHigh;
  for (nint_t i = 0; i < count; ++i) {
    // Log-uniform magnitudes across the span; two lanes in every sixteen
    // sit on special-path magnitudes when the span crosses the tails.
    double exponent = low + (high - low) * double(i % 65537) / 65536.0;
    if constexpr (wide) {
      if (i % 16 == 3) exponent = -130.0 + 0.5 * double(i % 7);
      if (i % 16 == 11) exponent = 126.0 + 0.5 * double(i % 5);
    }
    input[static_cast<size_t>(i)] = E(std::pow(2.0, exponent));
  }

  for (auto _ : state) {
    for (nint_t i = 0; i < count; i += lanes) {
      const auto x = load(t, input.data() + i);
      store(t, output.data() + i, apply_rcp<tier>(t, x));
    }
    benchmark::ClobberMemory();
  }

  benchmark::DoNotOptimize(output.data());
  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(
      state.iterations() * count * static_cast<int64_t>(2 * sizeof(E)));
  state.counters["lanes"] = benchmark::Counter(double(lanes));
}

template <Accuracy tier, bool wide, typename E>
void register_one() {
  const ScalableTag<E> t;
  const std::string name =
      "Rcp/kernel/rank:1/shape:VLx256/dtype:" +
      std::string(dtype_name<E>()) + "/mode:" + mode_name<tier>() +
      "/domain:" + domain_name(wide) +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
  benchmark::RegisterBenchmark(
      name.c_str(), &bench_rcp<tier, wide, E>)
      ->Unit(benchmark::kNanosecond)
      ->MinTime(0.05)
      ->Repetitions(5)
      ->ReportAggregatesOnly(true);
}

template <typename E>
void register_tiers() {
  register_one<Accuracy::Strict, false, E>();
  register_one<Accuracy::Fast, false, E>();
  register_one<Accuracy::Estimate, false, E>();
  register_one<Accuracy::Strict, true, E>();
  register_one<Accuracy::Fast, true, E>();
  register_one<Accuracy::Estimate, true, E>();
}

void register_benchmarks() {
  register_tiers<float32_t>();
  register_tiers<float64_t>();
  register_tiers<vecops::float16_t>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  register_tiers<vecops::bfloat16_t>();
#endif
}

}  // namespace

int main(int argc, char** argv) {
  register_benchmarks();

  const auto default_csv = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, "rcp", VECOPS_BENCH_ARCH_CODE, "csv");
  auto injected = vecops::bench::default_google_benchmark_output_args(
      argc, argv, default_csv, "csv");

  std::vector<char*> args;
  args.reserve(static_cast<size_t>(argc) + injected.size());
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  for (auto& arg : injected) args.push_back(arg.data());
  int bench_argc = static_cast<int>(args.size());

  benchmark::Initialize(&bench_argc, args.data());
  if (benchmark::ReportUnrecognizedArguments(bench_argc, args.data()))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  vecops::bench::print_default_output_path(argc, argv, default_csv);
  return 0;
}
