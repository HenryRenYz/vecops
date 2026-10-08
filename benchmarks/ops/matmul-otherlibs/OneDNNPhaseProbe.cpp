// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Diagnostic decomposition of oneDNN raw-E2E into opaque-weight reorder and
// execution with a retained opaque weight.  These rows are intentionally not
// part of the formal provider aggregate.

#include "OneDNNRunner.h"

#include <stdexcept>

namespace vecops::bench::matmul_otherlibs {

enum class OneDNNPhase : std::uint8_t {
  PackWeights,
  PreparedWeightsCompute,
};

constexpr const char* phase_name(OneDNNPhase phase) {
  switch (phase) {
    case OneDNNPhase::PackWeights: return "pack_weights";
    case OneDNNPhase::PreparedWeightsCompute:
      return "prepared_weights_compute";
  }
  return "unknown";
}

void run_phase(
    benchmark::State& state, const Case& c, OneDNNPhase phase) {
  try {
    Buffers buffers(c);
    OneDNNRunner runner(c, buffers, Operation::Gemm);
    runner.prepare_weights();
    buffers.prepare_output(Operation::Gemm);
    runner.execute_prepared();
    std::string error;
    if (!verify_samples(c, Operation::Gemm, buffers, &error)) {
      state.SkipWithError(error);
      return;
    }

    buffers.prepare_output(Operation::Gemm);
    for (auto _ : state) {
      benchmark::DoNotOptimize(buffers.x.data());
      benchmark::DoNotOptimize(buffers.weight.data());
      if (phase == OneDNNPhase::PackWeights) {
        runner.prepare_weights();
      } else {
        runner.execute_prepared();
      }
      benchmark::DoNotOptimize(buffers.output.data());
      benchmark::ClobberMemory();
    }
    const bool packing = phase == OneDNNPhase::PackWeights;
    set_counters(state, c, c.batch > 1, packing, !packing);
    state.counters["weight_reorder_required"] = benchmark::Counter(
        runner.weights_reorder_required() ? 1.0 : 0.0);
    state.counters["prepared_weight_bytes"] = benchmark::Counter(
        static_cast<double>(runner.prepared_weight_bytes()));
    state.SetLabel(runner.implementation());
  } catch (const std::exception& error) {
    state.SkipWithError(error.what());
  }
}

void register_phase(const Case& c, OneDNNPhase phase) {
  const auto name = benchmark_name(
      "oneDNN", c, Operation::Gemm, "NA", phase_name(phase),
      "diagnostic", "phase_probe");
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [c, phase](benchmark::State& state) {
        run_phase(state, c, phase);
      });
  registered->Unit(benchmark::kMicrosecond);
  configure_comparison_benchmark(registered, 0.1, 7);
}

void register_case(const Case& c) {
  register_phase(c, OneDNNPhase::PackWeights);
  register_phase(c, OneDNNPhase::PreparedWeightsCompute);
}

void register_cases() {
#if defined(ARCH_X86_FAMILY)
  register_case(CoreCases[7]);   // L03
  register_case(CoreCases[9]);   // L05
  register_case(CoreCases[11]);  // L07
  register_case(AF3Cases[0]);    // A01
  register_case(AF3Cases[3]);    // A04
  register_case(AF3Cases[17]);   // A18
#else
  register_case(CoreCases[5]);   // L01
  register_case(CoreCases[8]);   // L04
  register_case(AF3Cases[0]);    // A01
#endif
}

} // namespace vecops::bench::matmul_otherlibs

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_otherlibs;
  register_cases();
  return run_registered_benchmarks(
      argc, argv, "matmul_otherlibs_onednn_phase_probe");
}
