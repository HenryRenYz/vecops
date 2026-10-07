#ifndef VECOPS_VEC_DETAILS_SVE_SVE_H
#define VECOPS_VEC_DETAILS_SVE_SVE_H

// Ordinary-SVE single-register segment primitives. The public SVE.h owns
// their execution-domain/layout contract; no state changes or allocations.
#include "vecops/vec/Vec.h"

namespace vecops::vec::details::sve {

template <Element T>
using WordTag = ScalableTag<T, 0>;

template <Element T>
VECOPS_ALWAYS_INLINE Vec<WordTag<T>> join_k_groups(
    WordTag<T>, Vec<WordTag<T>> group0, Vec<WordTag<T>> group1) noexcept {
  using Bits = WordTag<uint32_t>;
  const auto a = bitcast(Bits{}, WordTag<T>{}, group0);
  const auto b = bitcast(Bits{}, WordTag<T>{}, group1);
  const auto joined = sve_basic_wrap_word<Bits>(svzip1_u32(
      sve_basic_raw_word(a), sve_basic_raw_word(b)));
  return bitcast(WordTag<T>{}, Bits{}, joined);
}

template <int Index, Element T>
VECOPS_ALWAYS_INLINE Vec<WordTag<T>> broadcast_segment(
    WordTag<T>, Vec<WordTag<T>> value) noexcept {
  static_assert(Index >= 0 && Index < 16,
                "SVE has at most sixteen 128-bit segments; check actual VL");
  using Bits = WordTag<uint64_t>;
  const auto bits = bitcast(Bits{}, WordTag<T>{}, value);
  const auto broadcast = sve_basic_wrap_word<Bits>(
      svdupq_lane_u64(sve_basic_raw_word(bits), Index));
  return bitcast(WordTag<T>{}, Bits{}, broadcast);
}

template <int Row, Element T>
  requires (sizeof(T) == 4)
VECOPS_ALWAYS_INLINE Vec<WordTag<T>> row_pair(
    WordTag<T>, Vec<WordTag<T>> value) noexcept {
  static_assert(Row == 0 || Row == 1, "MMLA segments have two rows");
  using Bits = WordTag<uint64_t>;
  const auto bits = bitcast(Bits{}, WordTag<T>{}, value);
  const auto raw = sve_basic_raw_word(bits);
  const auto row = sve_basic_wrap_word<Bits>(
      Row == 0 ? svuzp1_u64(raw, raw) : svuzp2_u64(raw, raw));
  return bitcast(WordTag<T>{}, Bits{}, row);
}

// Typed overloads make unsupported dtype combinations a compile-time error.
#if defined(__ARM_FEATURE_SVE_BF16)
VECOPS_ALWAYS_INLINE Vec<WordTag<float32_t>> mmla(
    WordTag<float32_t>, WordTag<bfloat16_t>, WordTag<bfloat16_t>,
    Vec<WordTag<bfloat16_t>> a, Vec<WordTag<bfloat16_t>> b,
    Vec<WordTag<float32_t>> accumulator) noexcept {
  return sve_basic_wrap_word<WordTag<float32_t>>(svbfmmla_f32(
      sve_basic_raw_word(accumulator), sve_basic_raw_word(a),
      sve_basic_raw_word(b)));
}
#endif

#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
VECOPS_ALWAYS_INLINE Vec<WordTag<int32_t>> mmla(
    WordTag<int32_t>, WordTag<int8_t>, WordTag<int8_t>,
    Vec<WordTag<int8_t>> a, Vec<WordTag<int8_t>> b,
    Vec<WordTag<int32_t>> accumulator) noexcept {
  return sve_basic_wrap_word<WordTag<int32_t>>(svmmla_s32(
      sve_basic_raw_word(accumulator), sve_basic_raw_word(a),
      sve_basic_raw_word(b)));
}

VECOPS_ALWAYS_INLINE Vec<WordTag<int32_t>> mmla(
    WordTag<int32_t>, WordTag<uint8_t>, WordTag<uint8_t>,
    Vec<WordTag<uint8_t>> a, Vec<WordTag<uint8_t>> b,
    Vec<WordTag<int32_t>> accumulator) noexcept {
  using Unsigned = WordTag<uint32_t>;
  const auto bits = bitcast(Unsigned{}, WordTag<int32_t>{}, accumulator);
  const auto sum = sve_basic_wrap_word<Unsigned>(svmmla_u32(
      sve_basic_raw_word(bits), sve_basic_raw_word(a), sve_basic_raw_word(b)));
  return bitcast(WordTag<int32_t>{}, Unsigned{}, sum);
}
#endif

} // namespace vecops::vec::details::sve

#endif // VECOPS_VEC_DETAILS_SVE_SVE_H
