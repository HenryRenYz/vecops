//
// Shared allocation, verification, naming, and registration for the
// cross-library BF16 GEMM benchmarks.
//

#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../BenchmarkUtils.h"
#include "Cases.h"
#include "vecops/CoreTypes.h"

#ifndef VECOPS_BENCH_ARCH_CODE
#define VECOPS_BENCH_ARCH_CODE "unknown"
#endif

#ifndef VECOPS_MATMUL_OTHERLIBS_PROVIDER
#define VECOPS_MATMUL_OTHERLIBS_PROVIDER "unknown"
#endif

namespace vecops::bench::matmul_otherlibs {

inline std::size_t checked_elements(nint_t a, nint_t b, nint_t c = 1) {
  const auto value = static_cast<unsigned long long>(a) *
                     static_cast<unsigned long long>(b) *
                     static_cast<unsigned long long>(c);
  if (value > static_cast<unsigned long long>(
                  std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("matrix allocation size overflow");
  }
  return static_cast<std::size_t>(value);
}

struct Buffers {
  explicit Buffers(const Case& c)
      : x(checked_elements(c.batch, c.n, c.k)),
        weight(checked_elements(c.batch, c.m, c.k)),
        initial(checked_elements(c.batch, c.n, c.m)),
        output(initial.size()), bias(static_cast<std::size_t>(c.m)) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      const int value = static_cast<int>((i * 13 + 3) % 17) - 8;
      x[i] = bfloat16_t{static_cast<float>(value) / 32.0f};
    }
    for (std::size_t i = 0; i < weight.size(); ++i) {
      const int value = static_cast<int>((i * 7 + 11) % 19) - 9;
      weight[i] = bfloat16_t{static_cast<float>(value) / 32.0f};
    }
    for (std::size_t i = 0; i < initial.size(); ++i) {
      initial[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 64.0f;
    }
    for (std::size_t i = 0; i < bias.size(); ++i) {
      bias[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32.0f;
    }
    reset_output();
  }

  void prepare_output(Operation op) {
    if (op == Operation::Gemm) {
      std::fill(output.begin(), output.end(), 0.0f);
    } else if (op == Operation::GemmAdd) {
      output = initial;
    } else {
      std::fill(output.begin(), output.end(), 0.0f);
    }
  }

  void reset_output() { output = initial; }

  std::vector<bfloat16_t> x;
  std::vector<bfloat16_t> weight;
  std::vector<float> initial;
  std::vector<float> output;
  std::vector<float> bias;
};

inline float apply_epilogue(float value, Operation op, float bias) {
  if (op == Operation::Bias || op == Operation::BiasRelu ||
      op == Operation::BiasSilu) {
    value += bias;
  }
  if (op == Operation::BiasRelu) return std::max(value, 0.0f);
  if (op == Operation::BiasSilu) return value / (1.0f + std::exp(-value));
  return value;
}

inline bool verify_samples(
    const Case& c, Operation op, const Buffers& buffers,
    std::string* error = nullptr) {
  const std::array<std::array<nint_t, 3>, 6> samples{{
      {{0, 0, 0}},
      {{0, c.n - 1, c.m - 1}},
      {{0, c.n / 2, c.m / 2}},
      {{c.batch - 1, 0, c.m - 1}},
      {{c.batch - 1, c.n - 1, 0}},
      {{c.batch - 1, c.n / 2, c.m / 2}},
  }};
  for (const auto& sample : samples) {
    const nint_t batch = sample[0];
    const nint_t row = sample[1];
    const nint_t col = sample[2];
    const std::size_t x_base = checked_elements(batch, c.n, c.k);
    const std::size_t w_base = checked_elements(batch, c.m, c.k);
    const std::size_t y_index =
        checked_elements(batch, c.n, c.m) +
        static_cast<std::size_t>(row * c.m + col);
    float expected = op == Operation::GemmAdd ? buffers.initial[y_index] : 0.0f;
    for (nint_t kk = 0; kk < c.k; ++kk) {
      expected += static_cast<float>(buffers.x[
                      x_base + static_cast<std::size_t>(row * c.k + kk)]) *
                  static_cast<float>(buffers.weight[
                      w_base + static_cast<std::size_t>(col * c.k + kk)]);
    }
    expected = apply_epilogue(
        expected, op, buffers.bias[static_cast<std::size_t>(col)]);
    const float actual = buffers.output[y_index];
    const float tolerance = 8.0e-3f *
        std::max({1.0f, std::abs(expected), std::abs(actual)});
    if (std::abs(expected - actual) > tolerance) {
      if (error != nullptr) {
        *error = "sample mismatch at batch=" + std::to_string(batch) +
                 " row=" + std::to_string(row) +
                 " col=" + std::to_string(col) +
                 " expected=" + std::to_string(expected) +
                 " actual=" + std::to_string(actual);
      }
      return false;
    }
  }
  return true;
}

inline std::string benchmark_name(
    std::string_view provider, const Case& c, Operation op,
    std::string_view extent, std::string_view phase = "raw_e2e",
    std::string_view epilogue = "none",
    std::string_view tuning = "default",
    std::string_view mc = "auto", std::string_view nc = "auto",
    std::string_view kc = "auto") {
  return "MatmulOtherlibs/provider:" + std::string(provider) +
         "/case:" + std::string(c.id) +
         "/model:" + model_name(c.model) +
         "/subgraph:" + std::string(c.subgraph) +
         "/shape:" + std::to_string(c.m) + "x" +
             std::to_string(c.n) + "x" + std::to_string(c.k) +
         "/batch:" + std::to_string(c.batch) +
         "/tokens:" + std::to_string(c.num_tokens) +
         "/chunk:" + std::to_string(c.chunk_size) +
         "/op:" + operation_name(op) +
         "/dtype:bf16xbf16_acc_fp32" +
         "/extent:" + std::string(extent) +
         "/phase:" + std::string(phase) +
         "/epilogue:" + std::string(epilogue) +
         "/tuning:" + std::string(tuning) +
         "/mc:" + std::string(mc) +
         "/nc:" + std::string(nc) +
         "/kc:" + std::string(kc) +
         "/orientation:" + orientation_name(c.orientation) +
         "/arch:" + VECOPS_BENCH_ARCH_CODE;
}

inline double percentile(const std::vector<double>& samples, double p) {
  if (samples.empty()) return 0.0;
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double position = p * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

inline benchmark::internal::Benchmark* configure_comparison_benchmark(
    benchmark::internal::Benchmark* registered, double min_time,
    int repetitions) {
  ::vecops::bench::configure_registered_benchmark(
      registered, min_time, repetitions);
  registered->ComputeStatistics(
      "p10", [](const std::vector<double>& values) {
        return percentile(values, 0.10);
      });
  registered->ComputeStatistics(
      "p90", [](const std::vector<double>& values) {
        return percentile(values, 0.90);
      });
  return registered->ReportAggregatesOnly(true);
}

inline void set_counters(
    benchmark::State& state, const Case& c, bool native_batch,
    bool packing_included, bool weights_prepacked) {
  const double flops = 2.0 * static_cast<double>(c.batch) *
                       static_cast<double>(c.m) *
                       static_cast<double>(c.n) *
                       static_cast<double>(c.k);
  state.counters["FLOP/s"] = benchmark::Counter(
      flops, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["m"] = benchmark::Counter(static_cast<double>(c.m));
  state.counters["n"] = benchmark::Counter(static_cast<double>(c.n));
  state.counters["k"] = benchmark::Counter(static_cast<double>(c.k));
  state.counters["batch"] = benchmark::Counter(static_cast<double>(c.batch));
  state.counters["num_tokens"] =
      benchmark::Counter(static_cast<double>(c.num_tokens));
  state.counters["chunk_size"] =
      benchmark::Counter(static_cast<double>(c.chunk_size));
  state.counters["call_multiplicity"] =
      benchmark::Counter(static_cast<double>(c.call_multiplicity));
  state.counters["native_batch"] = benchmark::Counter(native_batch ? 1.0 : 0.0);
  state.counters["packing_included"] =
      benchmark::Counter(packing_included ? 1.0 : 0.0);
  state.counters["weights_prepacked"] =
      benchmark::Counter(weights_prepacked ? 1.0 : 0.0);
}

template <typename Run>
void run_external_benchmark(
    benchmark::State& state, const Case& c, Operation op,
    Run&& run, bool native_batch = false,
    bool packing_included = true, bool weights_prepacked = false) {
  Buffers buffers(c);
  buffers.prepare_output(op);
  run(buffers, op);
  std::string error;
  if (!verify_samples(c, op, buffers, &error)) {
    state.SkipWithError(error);
    return;
  }
  buffers.prepare_output(op);
  for (auto _ : state) {
    // beta=1 consumes C0 rather than the preceding benchmark output.
    // Restoring C0 is harness setup, not GEMM work, so all providers do it
    // with timing paused.
    if (op == Operation::GemmAdd) {
      state.PauseTiming();
      buffers.prepare_output(op);
      state.ResumeTiming();
    }
    benchmark::DoNotOptimize(buffers.x.data());
    benchmark::DoNotOptimize(buffers.weight.data());
    run(buffers, op);
    benchmark::DoNotOptimize(buffers.output.data());
    benchmark::ClobberMemory();
  }
  set_counters(
      state, c, native_batch, packing_included, weights_prepacked);
}

template <typename Factory>
void run_external_benchmark_factory(
    benchmark::State& state, const Case& c, Operation op,
    Factory&& factory, bool native_batch = false,
    bool packing_included = true, bool weights_prepacked = false) {
  Buffers buffers(c);
  auto run = factory(buffers, op);
  buffers.prepare_output(op);
  run();
  std::string error;
  if (!verify_samples(c, op, buffers, &error)) {
    state.SkipWithError(error);
    return;
  }
  buffers.prepare_output(op);
  for (auto _ : state) {
    // Keep every beta=1 invocation independent of Google Benchmark's prior
    // iteration and warm-up state; see run_external_benchmark above.
    if (op == Operation::GemmAdd) {
      state.PauseTiming();
      buffers.prepare_output(op);
      state.ResumeTiming();
    }
    benchmark::DoNotOptimize(buffers.x.data());
    benchmark::DoNotOptimize(buffers.weight.data());
    run();
    benchmark::DoNotOptimize(buffers.output.data());
    benchmark::ClobberMemory();
  }
  set_counters(
      state, c, native_batch, packing_included, weights_prepacked);
}

template <typename Callback>
void register_external_cases(
    std::string_view provider, Callback&& callback,
    double min_time = 0.1, int repetitions = 7,
    std::string_view phase = "raw_e2e", bool native_bias = false,
    bool native_relu = false, bool native_silu = false) {
  const auto register_catalog = [&](const auto& catalog) {
    for (const Case& c : catalog) {
      for (Operation op : {Operation::Gemm, Operation::GemmAdd,
                           Operation::Bias, Operation::BiasRelu,
                           Operation::BiasSilu}) {
        if ((c.operations & op_bit(op)) == 0) continue;
        const bool fused =
            (op == Operation::Bias && native_bias) ||
            (op == Operation::BiasRelu && native_relu) ||
            (op == Operation::BiasSilu && native_silu);
        const std::string epilogue =
            op == Operation::Gemm || op == Operation::GemmAdd
                ? "native"
                : fused ? "native_fused" : "separate";
        const auto name = benchmark_name(provider, c, op, "NA", phase, epilogue);
        auto* registered = benchmark::RegisterBenchmark(
            name.c_str(), [c, op, callback](benchmark::State& state) mutable {
              callback(state, c, op);
            });
        registered->Unit(benchmark::kMicrosecond);
        configure_comparison_benchmark(registered, min_time, repetitions);
      }
    }
  };
  register_catalog(CoreCases);
  register_catalog(AF3Cases);
}

inline int run_registered_benchmarks(
    int argc, char** argv, std::string_view result_stem) {
  const auto output = ::vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, std::string(result_stem),
      VECOPS_BENCH_ARCH_CODE, "json");
  auto defaults = ::vecops::bench::default_google_benchmark_output_args(
      argc, argv, output, "json");
  std::vector<std::string> owned;
  owned.reserve(static_cast<std::size_t>(argc) + defaults.size());
  for (int i = 0; i < argc; ++i) owned.emplace_back(argv[i]);
  for (auto& value : defaults) owned.push_back(std::move(value));
  std::vector<char*> args;
  args.reserve(owned.size());
  for (auto& value : owned) args.push_back(value.data());
  int adjusted_argc = static_cast<int>(args.size());
  benchmark::Initialize(&adjusted_argc, args.data());
  if (benchmark::ReportUnrecognizedArguments(adjusted_argc, args.data())) return 1;
  ::vecops::bench::print_default_output_path(argc, argv, output);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}

} // namespace vecops::bench::matmul_otherlibs
