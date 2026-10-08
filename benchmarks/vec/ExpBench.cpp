// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
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

template <ExpBase family>
const char* family_name() {
  if constexpr (family == ExpBase::E) return "Exp";
  if constexpr (family == ExpBase::Base2) return "Exp2";
  return "Exp10";
}

// Input spans stay inside the hot domain of every dtype, including the
// narrow ones (fp16 exp10 overflows just above 4.82).
template <ExpBase family>
constexpr double input_span() {
  if constexpr (family == ExpBase::E) return 10.0;
  else if constexpr (family == ExpBase::Base2) return 10.0;
  else return 4.0;
}

template <ExpBase family, Accuracy tier, bool negative_only, VectorTag Tag>
Vec<Tag> apply_exp(Tag tag, Vec<Tag> x) {
  return ExpCpo<family, negative_only>{}(
      tag, x, opt::math::accuracy<tier>);
}

template <ExpBase family, Accuracy tier, bool negative_only, typename E>
void bench_exp(benchmark::State& state) {
  const ScalableTag<E> t;
  const nint_t lanes = size(t);
  const nint_t count = lanes * kBlocks;
  std::vector<E> input(static_cast<size_t>(count));
  std::vector<E> output(static_cast<size_t>(count));
  constexpr double span = input_span<family>();
  for (nint_t i = 0; i < count; ++i) {
    const double unit = double(i % 257) / 256.0;
    const double value = negative_only ? -span * unit : 2.0 * span * unit - span;
    input[static_cast<size_t>(i)] = E(value);
  }

  for (auto _ : state) {
    for (nint_t i = 0; i < count; i += lanes) {
      const auto x = load(t, input.data() + i);
      store(t, output.data() + i, apply_exp<family, tier, negative_only>(t, x));
    }
    benchmark::ClobberMemory();
  }

  benchmark::DoNotOptimize(output.data());
  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(
      state.iterations() * count * static_cast<int64_t>(2 * sizeof(E)));
  state.counters["lanes"] = benchmark::Counter(double(lanes));
}

template <ExpBase family, Accuracy tier, bool negative_only, typename E>
void register_one() {
  const ScalableTag<E> t;
  const std::string name =
      std::string(family_name<family>()) +
      "/kernel/rank:1/shape:VLx256/dtype:" +
      std::string(dtype_name<E>()) + "/mode:" + mode_name<tier>() +
      "/domain:" + (negative_only ? "negative" : "general") +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
  benchmark::RegisterBenchmark(
      name.c_str(), &bench_exp<family, tier, negative_only, E>)
      ->Unit(benchmark::kNanosecond)
      ->MinTime(0.05)
      ->Repetitions(5)
      ->ReportAggregatesOnly(true);
}

template <ExpBase family, typename E>
void register_tiers() {
  register_one<family, Accuracy::Strict, false, E>();
  register_one<family, Accuracy::Fast, false, E>();
  register_one<family, Accuracy::Estimate, false, E>();
  register_one<family, Accuracy::Strict, true, E>();
  register_one<family, Accuracy::Fast, true, E>();
  register_one<family, Accuracy::Estimate, true, E>();
}

template <typename E>
void register_dtype() {
  register_tiers<ExpBase::E, E>();
  register_tiers<ExpBase::Base2, E>();
  register_tiers<ExpBase::Base10, E>();
}

void register_benchmarks() {
  register_dtype<float32_t>();
  register_dtype<float64_t>();
  register_dtype<vecops::float16_t>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  register_dtype<vecops::bfloat16_t>();
#endif
}

}  // namespace

int main(int argc, char** argv) {
  register_benchmarks();

  const auto default_csv = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, "exp", VECOPS_BENCH_ARCH_CODE, "csv");
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
