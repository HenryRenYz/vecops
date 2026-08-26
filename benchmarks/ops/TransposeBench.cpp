#include <cstddef>
#include <vector>

#include <benchmark/benchmark.h>

#include "BenchmarkUtils.h"
#include "TransposeBenchCommon.h"

#ifndef VECOPS_SOURCE_DIR
#define VECOPS_SOURCE_DIR "."
#endif

int main(int argc, char** argv) {
  using namespace vecops::bench::transpose;
  register_int8_benchmarks();
  register_fp16_benchmarks();
  register_bf16_benchmarks();
  register_fp32_benchmarks();
  register_fp64_benchmarks();
  register_conversion_benchmarks();

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
