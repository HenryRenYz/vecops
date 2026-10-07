// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>
#include <vector>

#if defined(ARCH_X86_FAMILY)
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

#include "BenchmarkUtils.h"
#include "vecops/matmul/Atom.h"
#include "vecops/ops/Matmul.h"

namespace {

using namespace vecops;
using namespace vecops::tensor;

#if defined(ARCH_X86_FAMILY)
using TestAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using TestAtom = matmul::SME_BF16F32;
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

#if defined(ARCH_X86_FAMILY) || defined(HAS_SME)

template <bool Prepared>
void run(benchmark::State& state) {
  const nint_t m = state.range(0);
  const nint_t n = state.range(1);
  const nint_t k = state.range(2);
  std::vector<vecops::bfloat16_t> a(
      static_cast<std::size_t>(m * k), vecops::bfloat16_t{1});
  std::vector<vecops::bfloat16_t> b(
      static_cast<std::size_t>(n * k), vecops::bfloat16_t{1});
  std::vector<vecops::float32_t> c(static_cast<std::size_t>(m * n));
  auto at = make_tensor(a.data(), make_shape(meta::Any{m}, meta::Any{k}));
  auto bt = make_tensor(b.data(), make_shape(meta::Any{n}, meta::Any{k}));
  auto ct = make_tensor(c.data(), make_shape(meta::Any{m}, meta::Any{n}));
  auto operation = ops::matmul(ops::MatmulConfig<TestAtom>{});
  const nint_t bytes = operation.required_workspace(m, n, k, at, bt, ct);
  kernel::Workspace storage(bytes + vec::DEFAULT_ALIGNMENT);
  auto workspace = storage.view();

  if constexpr (Prepared) {
    auto prepared = operation.prepare(workspace, m, n, k, at, bt, ct);
    prepared();
    for (auto _ : state) {
      benchmark::DoNotOptimize(a.data());
      benchmark::DoNotOptimize(b.data());
      prepared();
      benchmark::DoNotOptimize(c.data());
      benchmark::ClobberMemory();
    }
  } else {
    ExecutionSession execution{workspace};
    operation(execution, m, n, k, at, bt, ct);
    for (auto _ : state) {
      benchmark::DoNotOptimize(a.data());
      benchmark::DoNotOptimize(b.data());
      operation(execution, m, n, k, at, bt, ct);
      benchmark::DoNotOptimize(c.data());
      benchmark::ClobberMemory();
    }
  }

  if (c.front() != static_cast<float>(k) || c.back() != static_cast<float>(k)) {
    state.SkipWithError("matmul result verification failed");
  }
  const auto mnk = static_cast<int64_t>(m) * n * k;
  state.counters["FLOP/s"] =
    benchmark::Counter(static_cast<double>(2 * mnk), benchmark::Counter::kIsIterationInvariantRate);
  state.counters["workspace_bytes"] = benchmark::Counter(double(bytes));
}

void register_case(nint_t m, nint_t n, nint_t k) {
  const std::string shape = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
  auto* legacy = benchmark::RegisterBenchmark(("StatefulMatmul/Legacy/" + shape).c_str(), &run<false>);
  auto* prepared = benchmark::RegisterBenchmark(("StatefulMatmul/Prepared/" + shape).c_str(), &run<true>);
  for (auto* entry : {legacy, prepared}) {
    entry->Args({m, n, k})->Unit(benchmark::kNanosecond);
    vecops::bench::configure_registered_benchmark(entry, 0.05, 7)->ReportAggregatesOnly(true);
  }
}

#endif

} // namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (!enable_architecture())
    return 1;
#if defined(ARCH_X86_FAMILY) || defined(HAS_SME)
  register_case(1, 16, 32);
  register_case(16, 16, 32);
  register_case(16, 128, 128);
  register_case(64, 64, 128);
  register_case(256, 256, 256);
#endif
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
