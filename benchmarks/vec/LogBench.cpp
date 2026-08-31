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

const char* base_name(LogBase base) {
  switch (base) {
    case LogBase::E: return "log";
    case LogBase::Base2: return "log2";
    default: return "log10";
  }
}

// "general" keeps every lane on the inline kernel path with log-uniform
// magnitudes across ten decades; "wide" mixes in subnormal, near-maximum,
// and tiny magnitudes so the special routine is exercised as well.
constexpr double kGeneralLow = -20.0;
constexpr double kGeneralHigh = 20.0;

const char* domain_name(bool wide) { return wide ? "wide" : "general"; }

template <Accuracy tier, LogBase Base, VectorTag Tag>
Vec<Tag> apply_log(Tag tag, Vec<Tag> x) {
  return LogCpo<Base>{}(tag, x, opt::math::accuracy<tier>);
}

template <Accuracy tier, bool wide, LogBase Base, typename E>
void bench_log(benchmark::State& state) {
  const ScalableTag<E> t;
  const nint_t lanes = size(t);
  const nint_t count = lanes * kBlocks;
  std::vector<E> input(static_cast<size_t>(count));
  std::vector<E> output(static_cast<size_t>(count));
  for (nint_t i = 0; i < count; ++i) {
    // Log-uniform magnitudes; two lanes in every sixteen sit on
    // special-path magnitudes in the wide domain.
    double exponent = kGeneralLow +
        (kGeneralHigh - kGeneralLow) * double(i % 65537) / 65536.0;
    if constexpr (wide) {
      if (i % 16 == 3) exponent = -140.0 + 0.5 * double(i % 7);
      if (i % 16 == 11) exponent = 120.0 + 0.5 * double(i % 5);
    }
    input[static_cast<size_t>(i)] = E(std::pow(2.0, exponent));
  }

  for (auto _ : state) {
    for (nint_t i = 0; i < count; i += lanes) {
      const auto x = load(t, input.data() + i);
      store(t, output.data() + i, apply_log<tier, Base>(t, x));
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
  const std::string prefix =
      "Log/kernel/rank:1/shape:VLx256/dtype:" +
      std::string(dtype_name<E>()) + "/mode:" + mode_name<tier>() +
      "/domain:" + domain_name(wide) + "/base:";
  const std::string suffix = std::string("/arch:") + VECOPS_BENCH_ARCH_CODE;
  benchmark::RegisterBenchmark(
      (prefix + "e" + suffix).c_str(),
      &bench_log<tier, wide, LogBase::E, E>)
      ->Unit(benchmark::kNanosecond)
      ->MinTime(0.05)
      ->Repetitions(5)
      ->ReportAggregatesOnly(true);
  benchmark::RegisterBenchmark(
      (prefix + "2" + suffix).c_str(),
      &bench_log<tier, wide, LogBase::Base2, E>)
      ->Unit(benchmark::kNanosecond)
      ->MinTime(0.05)
      ->Repetitions(5)
      ->ReportAggregatesOnly(true);
  benchmark::RegisterBenchmark(
      (prefix + "10" + suffix).c_str(),
      &bench_log<tier, wide, LogBase::Base10, E>)
      ->Unit(benchmark::kNanosecond)
      ->MinTime(0.05)
      ->Repetitions(5)
      ->ReportAggregatesOnly(true);
}

template <Accuracy tier, typename E>
void register_domains() {
  register_one<tier, false, E>();
  register_one<tier, true, E>();
}

template <typename E>
void register_tiers() {
  register_domains<Accuracy::Strict, E>();
  register_domains<Accuracy::Fast, E>();
  register_domains<Accuracy::Estimate, E>();
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
      VECOPS_SOURCE_DIR, "log", VECOPS_BENCH_ARCH_CODE, "csv");
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
