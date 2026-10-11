// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Standalone slow-path probe, without Google Benchmark dependencies:
// g++ -std=c++20 -O3 -march=native -ffp-contract=off -Iinclude -Iinclude/vecops
//     benchmarks/vec/ExpSpecialBench.cpp -o build-math-unify/exp-special
// ARM: replace g++ with clang++ and -march=native with -march=armv8-a+sve.
// Each row reports the median of five 10ms load/exp/store measurements.
// Keep nonfinite, finite-underflow and inactive-lane cases separate.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "vecops/vec/Vec.h"
#include "vecops/execution/FloatControl.h"

namespace {
using namespace vecops;
namespace v = vecops::vec;

template <typename T, v::Accuracy A>
void run(const char* name, double low, double high, bool masked = false) {
  const v::ScalableTag<T> tag;
  const auto lanes = v::size(tag);
  const auto count = lanes * 64;
  std::vector<T> input(count), output(count);
  for (nint_t i = 0; i < count; ++i)
    input[i] = static_cast<T>(i % 2 ? low : high);
  // Mask off high/NaN/Inf lanes; test that unpredicated backend instructions
  // do not accidentally consume those inactive inputs.
  const auto active = v::cmpeq(tag, v::load(tag, input.data()),
                             v::fill(tag, static_cast<T>(low)));
  auto invoke = [&] {
    for (nint_t i = 0; i < count; i += lanes) {
      const auto x = v::load(tag, input.data() + i);
      const auto y = masked
          ? v::exp_neg(tag, x, v::opt::math::accuracy<A>,
                       v::opt::masked(active), v::opt::zero)
          : v::exp_neg(tag, x, v::opt::math::accuracy<A>);
      v::store(tag, output.data() + i, y);
    }
    asm volatile("" : : "r"(output.data()) : "memory");
  };
  std::vector<double> samples;
  for (int round = 0; round < 5; ++round) {
    const auto begin = std::chrono::steady_clock::now();
    auto end = begin;
    int repeats = 0;
    do {
      invoke();
      ++repeats;
      end = std::chrono::steady_clock::now();
    } while (end - begin < std::chrono::milliseconds(10));
    samples.push_back(std::chrono::duration<double, std::nano>(end - begin).count()
                      / (repeats * count));
  }
  std::sort(samples.begin(), samples.end());
  std::printf("%s,%s,%s,%d,%.3f,%.9g\n", sizeof(T) == 4 ? "f32" : "f64",
              A == v::Accuracy::Strict ? "strict" :
              A == v::Accuracy::Fast ? "fast" : "estimate",
              name, masked, samples[2], static_cast<double>(output[0]));
  std::fflush(stdout);
}

template <typename T, v::Accuracy A>
void cases() {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double boundary = sizeof(T) == 4 ? -87.33654 : -708.396418532;
  run<T, A>("normal", -1, -10);
  run<T, A>("near_zero", -1e-30, 0);
  run<T, A>("subnormal_input", -std::numeric_limits<T>::denorm_min(), 0);
  run<T, A>("minus80", -80, -79.99999);
  run<T, A>("normal_boundary", boundary, boundary - 0.001);
  run<T, A>("underflow", -1000, -100);
  run<T, A>("mixed_underflow", -1, -1000);
  run<T, A>("minus_inf", -inf, -inf);
  run<T, A>("mixed_inf", -1, -inf);
  run<T, A>("nan", nan, nan);
  run<T, A>("mixed_nan", -1, nan);
  run<T, A>("inactive_nan", -1, nan, true);
  run<T, A>("inactive_inf", -1, inf, true);
  run<T, A>("inactive_underflow", -1, -1000, true);
}
template <typename T>
void tiers() {
  cases<T, v::Accuracy::Strict>();
  cases<T, v::Accuracy::Fast>();
  cases<T, v::Accuracy::Estimate>();
}
}

int main(int argc, char** argv) {
  const bool previous = vecops::execution::current_thread_flush_subnormals();
  if (argc > 1)
    vecops::execution::set_current_thread_flush_subnormals(argv[1][0] == '1');
  std::fprintf(stderr, "hardware_flush=%d\n",
               vecops::execution::current_thread_flush_subnormals());
  std::puts("dtype,accuracy,input,masked,ns_per_element,first_result");
  tiers<float>();
  tiers<double>();
  vecops::execution::set_current_thread_flush_subnormals(previous);
}
