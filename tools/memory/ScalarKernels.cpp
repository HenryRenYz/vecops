// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Exact FP32 scalar accesses. Pack expansion specifies work within one loop
// iteration; compiler flags prohibit additional automatic loop unrolling.
#include "Kernels.h"
#include <bit>
#include <cmath>
#include <utility>

namespace memory_bench {
namespace {
inline float scalar_load(const float* p) {
  float value;
#if defined(__x86_64__)
  asm volatile("vmovss %1, %0" : "=x"(value) : "m"(*p));
#else
  asm volatile("ldr %s0, %1" : "=w"(value) : "m"(*p));
#endif
  return value;
}
template <bool NT> inline void scalar_store(float* p, float value) {
#if defined(__x86_64__)
  if constexpr (NT) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    asm volatile("movnti %1, %0" : "=m"(*p) : "r"(bits));
  } else {
    asm volatile("vmovss %1, %0" : "=m"(*p) : "x"(value));
  }
#else
  static_assert(!NT);
  asm volatile("str %s1, %0" : "=m"(*p) : "w"(value));
#endif
}
template <bool NT> inline void load_pair(const float* p, float& a, float& b) {
#if defined(__aarch64__)
  if constexpr (NT) {
    asm volatile("ldnp %s0, %s1, [%2]" : "=w"(a), "=w"(b) : "r"(p), "m"(*p), "m"(p[1]));
    return;
  }
#endif
  a = scalar_load(p); b = scalar_load(p + 1);
}
template <bool NT> inline void store_pair(float* p, float a, float b) {
#if defined(__aarch64__)
  if constexpr (NT) {
    asm volatile("stnp %s2, %s3, [%4]" : "=m"(*p), "=m"(p[1]) : "w"(a), "w"(b), "r"(p));
  } else
#endif
  { scalar_store<NT>(p, a); scalar_store<NT>(p + 1, b); }
}
template <Operation O, bool L, bool S, bool F, std::size_t I>
inline void pair(const float* a, const float* b, float* d) {
  float a0 = 1.25f, a1 = 1.25f;
  if constexpr (O != Operation::Write) load_pair<L>(a + I, a0, a1);
  if constexpr (O == Operation::Triad) {
    float b0, b1;
    load_pair<L>(b + I, b0, b1);
    if constexpr (F) { a0 = std::fma(0.5f, b0, a0); a1 = std::fma(0.5f, b1, a1); }
    else { a0 = a0 + 0.5f * b0; a1 = a1 + 0.5f * b1; }
  }
  if constexpr (O != Operation::Read) store_pair<S>(d + I, a0, a1);
}
template <Operation O, std::size_t N, bool L, bool S, bool F, std::size_t... I>
inline void block(const float* a, const float* b, float* d, std::index_sequence<I...>) {
  (pair<O, L, S, F, 2 * I>(a, b, d), ...);
}
template <Operation O, std::size_t N, bool L, bool S, bool F>
__attribute__((noinline)) void run(const float* a, const float* b, float* d, std::size_t n) {
  for (std::size_t i = 0; i < n; i += N) {
    block<O, N, L, S, F>(O == Operation::Write ? nullptr : a + i,
                         O == Operation::Triad ? b + i : nullptr,
                         O == Operation::Read ? nullptr : d + i,
                         std::make_index_sequence<N / 2>{});
  }
}
template <Operation O, bool L, bool S, bool F>
Kernel select(unsigned bytes) {
  Kernel k;
  if (bytes == 16) k.function = run<O, 4, L, S, F>;
  if (bytes == 32) k.function = run<O, 8, L, S, F>;
  if (bytes == 64) k.function = run<O, 16, L, S, F>;
  k.step_bytes = bytes;
  return k;
}
template <Operation O, bool L, bool S> Kernel select_fma(unsigned bytes, bool fma) {
  return fma ? select<O, L, S, true>(bytes) : select<O, L, S, false>(bytes);
}
template <Operation O> Kernel select_policy(unsigned bytes, bool l, bool s, bool fma) {
#if defined(__x86_64__)
  (void)l;
  return s ? select_fma<O, false, true>(bytes, fma) : select_fma<O, false, false>(bytes, fma);
#else
  if (l) return s ? select_fma<O, true, true>(bytes, fma) : select_fma<O, true, false>(bytes, fma);
  return s ? select_fma<O, false, true>(bytes, fma) : select_fma<O, false, false>(bytes, fma);
#endif
}
} // namespace
Kernel scalar_kernel(Operation op, unsigned bytes, bool l, bool s, bool fma) {
  Kernel k;
  switch (op) {
    case Operation::Read: k = select_policy<Operation::Read>(bytes, l, s, fma); break;
    case Operation::Write: k = select_policy<Operation::Write>(bytes, l, s, fma); break;
    case Operation::Copy: k = select_policy<Operation::Copy>(bytes, l, s, fma); break;
    case Operation::Triad: k = select_policy<Operation::Triad>(bytes, l, s, fma); break;
  }
  k.load_policy = op == Operation::Write ? "none" : l ? "nt-hint" : "temporal";
  k.store_policy = op == Operation::Read ? "none" : s ? "nt-hint" : "temporal";
#if defined(__x86_64__)
  if (op != Operation::Write) k.load_policy = "temporal";
  if (op == Operation::Read && l) k.unsupported = "Intel has no scalar FP32 non-temporal load";
  k.instructions = "VMOVSS";
  if (op != Operation::Read && s) k.instructions += "+MOVNTI32";
#else
  k.instructions = std::string(l ? "LDNP-S" : "LDR-S") + "/" + (s ? "STNP-S" : "STR-S");
#endif
  return k;
}
} // namespace memory_bench
