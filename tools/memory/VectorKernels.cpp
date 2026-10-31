// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Multi-word ScalableTags express the requested 1/2/4 native-vector step.
#include "Kernels.h"
#include "vecops/vec/Memory.h"
#include "vecops/vec/Arithmetic.h"
#include <utility>

namespace memory_bench {
namespace v = vecops::vec;
namespace {
#if defined(__aarch64__)
// BiSheng 5.1 lowers even direct svldnt1/svstnt1 ACLE calls to ordinary
// accesses. Force the promised instructions at the public native-word boundary,
// retaining Vec/ScalableTag for representation, grouping and arithmetic.
template <typename Tag>
inline v::NativeWordVec<Tag> nt_load_word(Tag, const float* pointer) {
  v::NativeWordVec<Tag> value;
  const auto active = svptrue_b32();
  asm volatile("ldnt1w { %0.s }, %1/z, [%2]" : "=w"(value)
               : "Upl"(active), "r"(pointer) : "memory");
  return value;
}
template <typename Tag, std::size_t... I>
inline v::Vec<Tag> nt_load(Tag tag, const float* pointer, std::index_sequence<I...>) {
  const auto lanes = v::native_word_size(tag);
  return v::from_words(tag, nt_load_word(tag, pointer + I * lanes)...);
}
template <typename Tag, std::size_t... I>
inline void nt_store(Tag tag, float* pointer, v::Vec<Tag> value, std::index_sequence<I...>) {
  const auto lanes = v::native_word_size(tag);
  const auto one = [](auto word, float* target) {
    const auto active = svptrue_b32();
    asm volatile("stnt1w { %0.s }, %1, [%2]" :
                 : "w"(word), "Upl"(active), "r"(target) : "memory");
  };
  (one(v::get_word<I>(tag, value), pointer + I * lanes), ...);
}
#endif
template <bool NT, int P>
inline v::Vec<v::ScalableTag<float, P>> load(v::ScalableTag<float, P> tag, const float* pointer) {
#if defined(__aarch64__)
  if constexpr (NT) return nt_load(tag, pointer, std::make_index_sequence<(1 << P)>{});
  else
#endif
  return v::load(tag, pointer, v::mem::aligned,
                 std::conditional_t<NT, v::mem::NonTemporal, v::mem::Temporal>{});
}
template <bool NT, int P>
inline void store(v::ScalableTag<float, P> tag, float* pointer, v::Vec<v::ScalableTag<float, P>> value) {
#if defined(__aarch64__)
  if constexpr (NT) nt_store(tag, pointer, value, std::make_index_sequence<(1 << P)>{});
  else
#endif
  v::store(tag, pointer, value, v::mem::aligned,
           std::conditional_t<NT, v::mem::NonTemporal, v::mem::Temporal>{});
}
template <int P, std::size_t... I>
inline void consume(v::Vec<v::ScalableTag<float, P>> value, std::index_sequence<I...>) {
  const v::ScalableTag<float, P> tag;
#if defined(__x86_64__)
  const auto one = [](auto word) { asm volatile("" : : "v"(word) : "memory"); };
#else
  const auto one = [](auto word) { asm volatile("" : : "w"(word) : "memory"); };
#endif
  (one(v::get_word<I>(tag, value)), ...);
}
template <Operation O, int P, bool L, bool S, bool F>
__attribute__((noinline)) void run(const float* a, const float* b, float* d, std::size_t n) {
  const v::ScalableTag<float, P> tag;
  for (std::size_t i = 0; i < n; i += v::size(tag)) {
    auto result = v::fill(tag, 1.25f);
    if constexpr (O != Operation::Write)
      result = load<L>(tag, a + i);
    if constexpr (O == Operation::Triad) {
      const auto alpha = v::fill(tag, 0.5f);
      auto bv = load<L>(tag, b + i);
      if constexpr (F) result = v::fmadd(tag, bv, alpha, result);
      else result = v::add(tag, result, v::mul(tag, bv, alpha));
    }
    if constexpr (O == Operation::Read) {
      consume<P>(result, std::make_index_sequence<(1 << P)>{});
    } else {
      store<S>(tag, d + i, result);
      // Repeated scans are opaque indirect calls from another translation
      // unit, with LTO disabled. A memory-clobbering asm here would force
      // multi-word constant Vecs onto the stack on GCC and add false traffic.
    }
  }
}
template <Operation O, bool L, bool S, bool F> Kernel select(unsigned power) {
  Kernel k;
  if (power == 0) k.function = run<O, 0, L, S, F>;
  if (power == 1) k.function = run<O, 1, L, S, F>;
  if (power == 2) k.function = run<O, 2, L, S, F>;
  k.vector_bytes = native_vector_bytes();
  k.step_bytes = k.vector_bytes << power;
  return k;
}
template <Operation O, bool L, bool S> Kernel select_fma(unsigned p, bool fma) {
  return fma ? select<O, L, S, true>(p) : select<O, L, S, false>(p);
}
template <Operation O> Kernel select_policy(unsigned p, bool l, bool s, bool fma) {
  if (l) return s ? select_fma<O, true, true>(p, fma) : select_fma<O, true, false>(p, fma);
  return s ? select_fma<O, false, true>(p, fma) : select_fma<O, false, false>(p, fma);
}
} // namespace
std::size_t native_vector_bytes() { return v::size(v::ScalableTag<float>{}) * sizeof(float); }
const char* vector_backend() {
#if defined(__aarch64__)
  return "SVE";
#elif defined(__AVX512F__)
  return "AVX-512";
#elif defined(__AVX2__)
  return "AVX2";
#else
  return "x86 SIMD";
#endif
}
Kernel vector_kernel(Operation op, unsigned p, bool l, bool s, bool fma) {
  Kernel k;
  switch (op) {
    case Operation::Read: k = select_policy<Operation::Read>(p, l, s, fma); break;
    case Operation::Write: k = select_policy<Operation::Write>(p, l, s, fma); break;
    case Operation::Copy: k = select_policy<Operation::Copy>(p, l, s, fma); break;
    case Operation::Triad: k = select_policy<Operation::Triad>(p, l, s, fma); break;
  }
  k.load_policy = op == Operation::Write ? "none" : l ? "nt-hint" : "temporal";
  k.store_policy = op == Operation::Read ? "none" : s ? "nt-hint" : "temporal";
#if defined(__x86_64__)
  k.instructions = std::string(l ? "VMOVNTDQA(WB-hint)" : "VMOVAPS") + "/" + (s ? "VMOVNTDQ" : "VMOVAPS");
#else
  k.instructions = std::string(l ? "LDNT1W" : "LD1(B/W)") + "/" + (s ? "STNT1W" : "ST1(B/W)");
#endif
  return k;
}
} // namespace memory_bench
