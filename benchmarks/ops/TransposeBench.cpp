// @vecops-target-shards: 9

#include <cstddef>
#include <vector>

#include <benchmark/benchmark.h>

#include "BenchmarkUtils.h"
#include "TransposeBenchCommon.h"

#ifndef VECOPS_SOURCE_DIR
#define VECOPS_SOURCE_DIR "."
#endif

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

static_assert(VECOPS_TARGET_SHARD_COUNT == 9);

namespace vecops::bench::transpose {

#if VECOPS_TARGET_SHARD_INDEX == 0
void register_int8_benchmarks() { register_same_dtype<int8_t>(); }
#elif VECOPS_TARGET_SHARD_INDEX == 1
void register_fp16_benchmarks() {
  register_same_dtype<vecops::float16_t>();
}
#elif VECOPS_TARGET_SHARD_INDEX == 2
void register_bf16_benchmarks() {
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  register_same_dtype<vecops::bfloat16_t>();
#endif
}
#elif VECOPS_TARGET_SHARD_INDEX == 3
void register_fp32_benchmarks() {
  register_same_dtype<vecops::float32_t>();
}
#elif VECOPS_TARGET_SHARD_INDEX == 4
void register_fp64_benchmarks() {
  register_same_dtype<vecops::float64_t>();
}
#elif VECOPS_TARGET_SHARD_INDEX == 5
void register_int8_conversion_benchmarks() {
  register_conversion_cases(ConversionCases{}, Int8ConversionPairs{});
}
#elif VECOPS_TARGET_SHARD_INDEX == 6
void register_fp16_conversion_benchmarks() {
  register_conversion_cases(ConversionCases{}, Fp16ConversionPairs{});
}
#elif VECOPS_TARGET_SHARD_INDEX == 7
void register_fp32_conversion_benchmarks() {
  register_conversion_cases(ConversionCases{}, Fp32ConversionPairs{});
}
#elif VECOPS_TARGET_SHARD_INDEX == 8
void register_fp64_conversion_benchmarks() {
  register_conversion_cases(ConversionCases{}, Fp64ConversionPairs{});
}
#endif

} // namespace vecops::bench::transpose

#else

int main(int argc, char** argv) {
  using namespace vecops::bench::transpose;
  register_int8_benchmarks();
  register_fp16_benchmarks();
  register_bf16_benchmarks();
  register_fp32_benchmarks();
  register_fp64_benchmarks();
  register_int8_conversion_benchmarks();
  register_fp16_conversion_benchmarks();
  register_fp32_conversion_benchmarks();
  register_fp64_conversion_benchmarks();

  const auto default_csv = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, "transpose", VECOPS_BENCH_ARCH_CODE, "csv");
  auto injected = vecops::bench::default_google_benchmark_output_args(
      argc, argv, default_csv, "csv");

  std::vector<char*> args;
  args.reserve(static_cast<std::size_t>(argc) + injected.size());
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

#endif
