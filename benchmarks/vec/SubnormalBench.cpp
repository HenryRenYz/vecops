// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Independent finite-input probe for true subnormal operands/results.
// Build like ExpSpecialBench.cpp; run with 0 (gradual) or 1 (hardware FTZ).
// Values come from runtime loads; output and elapsed time are both reported.
// No Inf/NaN is present in the subnormal cases. Those are separate controls.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "vecops/execution/FloatControl.h"
#include "vecops/vec/Vec.h"

namespace {
using namespace vecops;
namespace v = vecops::vec;
enum class Op { mul, fma, rcp, rsqrt, log, exp, scale };

template <typename T, Op O>
void run(const char* name, T a, T b) {
  const v::ScalableTag<T> tag;
  const auto lanes = v::size(tag);
  const auto count = lanes * 64;
  std::vector<T> input(count, a), factor(count, b), output(count);
  auto invoke = [&] {
    for (nint_t i = 0; i < count; i += lanes) {
      const auto x = v::load(tag, input.data() + i);
      const auto y = v::load(tag, factor.data() + i);
      const auto result = [&] {
        if constexpr (O == Op::mul) return v::mul(tag, x, y);
        if constexpr (O == Op::fma) return v::fmadd(tag, x, y, v::zeros(tag));
        if constexpr (O == Op::rcp) return v::rcp(tag, x);
        if constexpr (O == Op::rsqrt) return v::rsqrt(tag, x);
        if constexpr (O == Op::log) return v::log(tag, x);
        if constexpr (O == Op::exp) return v::exp(tag, x);
#if defined(CPU_CAPABILITY_SVE)
        if constexpr (O == Op::scale) {
          if constexpr (sizeof(T) == 4)
            return svscale_n_f32_x(svptrue_b32(), x, -1);
          else
            return svscale_n_f64_x(svptrue_b64(), x, -1);
        }
#endif
      }();
      v::store(tag, output.data() + i, result);
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
  constexpr const char* operation = O == Op::mul ? "mul" : O == Op::fma ? "fma" :
      O == Op::rcp ? "rcp" : O == Op::rsqrt ? "rsqrt" : O == Op::log ? "log" :
      O == Op::exp ? "exp" : "raw_svscale";
  std::printf("%s,%s,%s,%.3f,%a\n", sizeof(T) == 4 ? "f32" : "f64",
              operation, name, samples[2], static_cast<double>(output[0]));
  std::fflush(stdout);
}

template <typename T, Op O>
void arithmetic() {
  const T tiny = std::numeric_limits<T>::denorm_min();
  const T normal = std::numeric_limits<T>::min();
  const T scale = std::ldexp(T(1), sizeof(T) == 4 ? 126 : 1022);
  run<T, O>("normal", T(1.125), T(0.5));
  run<T, O>("subnormal_input_normal_output", tiny, scale);
  run<T, O>("normal_input_subnormal_output", normal, T(0.5));
  run<T, O>("subnormal_input_subnormal_output", normal / T(2), T(0.5));
  run<T, O>("inf", std::numeric_limits<T>::infinity(), T(0.5));
  run<T, O>("nan", std::numeric_limits<T>::quiet_NaN(), T(0.5));
}

template <typename T>
void cases() {
  arithmetic<T, Op::mul>();
  arithmetic<T, Op::fma>();
  const T normal = std::numeric_limits<T>::min();
  run<T, Op::rcp>("normal", T(1.125), T(0));
  run<T, Op::rcp>("normal_input_subnormal_output", std::numeric_limits<T>::max(), T(0));
  run<T, Op::rsqrt>("normal", T(1.125), T(0));
  run<T, Op::rsqrt>("subnormal_input_normal_output", normal / T(2), T(0));
  run<T, Op::log>("normal", T(1.125), T(0));
  run<T, Op::log>("subnormal_input_normal_output", normal / T(2), T(0));
  run<T, Op::exp>("normal", T(-1), T(0));
  run<T, Op::exp>("normal_input_subnormal_output", sizeof(T) == 4 ? T(-90) : T(-710), T(0));
#if defined(CPU_CAPABILITY_SVE)
  run<T, Op::scale>("normal", T(1.125), T(0));
  run<T, Op::scale>("normal_input_subnormal_output", normal, T(0));
  run<T, Op::scale>("subnormal_input_subnormal_output", normal / T(2), T(0));
#endif
}
}

int main(int argc, char** argv) {
  const bool previous = vecops::execution::current_thread_flush_subnormals();
  vecops::execution::set_current_thread_flush_subnormals(argc > 1 && argv[1][0] == '1');
  std::fprintf(stderr, "hardware_flush=%d\n",
               vecops::execution::current_thread_flush_subnormals());
  std::puts("dtype,operation,input,ns_per_element,first_result");
  cases<float>();
  cases<double>();
  vecops::execution::set_current_thread_flush_subnormals(previous);
}
