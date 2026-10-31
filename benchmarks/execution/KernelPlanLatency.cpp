// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** Warm inference scheduling microbenchmark; excludes creation/prepare. */
#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <vector>
#include "vecops/execution/KernelPlan.h"
#include "vecops/execution/WorkspaceContext.h"
using namespace vecops;
using namespace vecops::execution;
using namespace vecops::tensor;
using Clock = std::chrono::steady_clock;
template <int P>
void measure(int rounds) {
  const auto* pool = vecops_native_thread_pool_v1(P);
  VecopsExecutionContext context{};
  context.struct_size = sizeof(context);
  context.thread_pool = pool;
  context.requested_threads = P;
  WorkspaceContext old("latency", nullptr, 0, nullptr, 0, &context);
  // Independent cache lines keep this a launch/barrier test rather than a
  // false-sharing test. Mixed widths exercise inactive-lane participation.
  std::array<int, P * 16> output{};
  KernelBuilder builder(P);
  auto out = builder.input(make_tensor(output.data(), make_shape(meta::cint<P * 16>)));
  for (int phase = 0; phase < 16; ++phase) {
    const nint_t tasks = phase % 2 ? std::max(1, P * 2 / 3) : P;
    builder.grid<P>("stage", nint_t{0}, tasks, meta::cint<1>, kernel_bind(out),
                    [](auto, const auto& item, auto out) { ++out(item.begin() * 16); });
  }
  KernelInstance instance(builder.finish(), pool);
  instance.set(out, output.data());
  auto legacy = [&] {
    for (int phase = 0; phase < 16; ++phase) {
      const nint_t tasks = phase % 2 ? std::max(1, P * 2 / 3) : P;
      old.parallel_for<P>(nint_t{0}, tasks, meta::cint<1>,
                          [&](auto, const auto& item) { ++output[item.begin() * 16]; });
    }
  };
  auto recorded = [&] { instance.replay(); };
  for (int i = 0; i < 10; ++i) {
    legacy();
    recorded();
  }
  std::vector<double> previous, planned;
  for (int sample = 0; sample < 7; ++sample) {
    auto run = [&](auto&& body) {
      const auto begin = Clock::now();
      for (int r = 0; r < rounds; ++r)
        body();
      return std::chrono::duration<double, std::micro>(Clock::now() - begin).count() / rounds;
    };
    if (sample % 2) {
      planned.push_back(run(recorded));
      previous.push_back(run(legacy));
    } else {
      previous.push_back(run(legacy));
      planned.push_back(run(recorded));
    }
  }
  std::sort(previous.begin(), previous.end());
  std::sort(planned.begin(), planned.end());
  const auto expected = (10 + 7 * rounds) * 2 * 16;
  if (output[0] != expected)
    throw std::runtime_error("benchmark result mismatch");
  std::cout << "{\"workers\":" << P << ",\"grids\":16,\"rounds\":" << rounds << ",\"legacy_us\":" << previous[3]
            << ",\"recorded_us\":" << planned[3] << ",\"speedup\":" << previous[3] / planned[3] << "}\n";
}
int main(int argc, char** argv) {
  const int workers = argc > 1 ? std::stoi(argv[1]) : 36;
  const int rounds = argc > 2 ? std::stoi(argv[2]) : 200;
  if (rounds <= 0)
    return 2;
  if (workers == 4)
    measure<4>(rounds);
  else if (workers == 36)
    measure<36>(rounds);
  else
    return 2;
}
