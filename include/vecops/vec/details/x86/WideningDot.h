#ifndef VECOPS_VEC_DETAILS_X86_WIDENINGDOT_H
#define VECOPS_VEC_DETAILS_X86_WIDENINGDOT_H

/**
 * @file WideningDot.h
 * @brief x86 grouped widening-dot specializations. Native instruction paths
 * are added here; unsupported combinations deliberately use the shared
 * phase-selection fallback.
 */

#include "vecops/vec/details/WideningDot.h"

namespace vecops::vec::details {

template <typename To, typename From1, typename From2>
inline constexpr bool x86_has_native_widening_dot_v =
#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
    (std::same_as<To, float32_t> &&
     std::same_as<From1, bfloat16_t> &&
     std::same_as<From2, bfloat16_t>) ||
#endif
#if defined(HAS_AVX512VNNI) || \
    (defined(HAS_AVX_VNNI) && VEC_WIDTH <= 256)
    ((std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
     ((std::same_as<From1, int8_t> && std::same_as<From2, int8_t>) ||
      (std::same_as<From1, uint8_t> && std::same_as<From2, uint8_t>) ||
      (std::same_as<From1, uint8_t> && std::same_as<From2, int8_t>) ||
      (std::same_as<From1, int8_t> && std::same_as<From2, uint8_t>) ||
      (std::same_as<From1, int16_t> && std::same_as<From2, int16_t>))) ||
#endif
    false;

template <VectorTag ToTag, VectorTag FromTag1, VectorTag FromTag2>
VECOPS_ALWAYS_INLINE NativeWordVec<ToTag> x86_widening_dot_word(
    NativeWordVec<FromTag1> a, NativeWordVec<FromTag2> b,
    NativeWordVec<ToTag> c) {
  using To = ElementOf<ToTag>;
  using From1 = ElementOf<FromTag1>;
  using From2 = ElementOf<FromTag2>;
  using Raw = decltype(c.value);

#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  if constexpr (
      std::same_as<To, float32_t> &&
      std::same_as<From1, bfloat16_t> &&
      std::same_as<From2, bfloat16_t>) {
    if constexpr (sizeof(Raw) == 16) {
      return NativeWordVec<ToTag>{_mm_dpbf16_ps(
          c.value, x86_cast_si128_to_bfloat16(a.value),
          x86_cast_si128_to_bfloat16(b.value))};
    } else if constexpr (sizeof(Raw) == 32) {
      return NativeWordVec<ToTag>{_mm256_dpbf16_ps(
          c.value, x86_cast_si256_to_bfloat16(a.value),
          x86_cast_si256_to_bfloat16(b.value))};
    } else {
      return NativeWordVec<ToTag>{_mm512_dpbf16_ps(
          c.value, x86_cast_si512_to_bfloat16(a.value),
          x86_cast_si512_to_bfloat16(b.value))};
    }
  } else
#endif
#if defined(HAS_AVX512VNNI) || \
    (defined(HAS_AVX_VNNI) && VEC_WIDTH <= 256)
  if constexpr (
      (std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
      ((std::same_as<From1, uint8_t> && std::same_as<From2, int8_t>) ||
       (std::same_as<From1, int8_t> && std::same_as<From2, uint8_t>))) {
    const auto unsigned_value = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<From1, uint8_t>) return a.value;
      else return b.value;
    }();
    const auto signed_value = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<From1, int8_t>) return a.value;
      else return b.value;
    }();
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<ToTag>{
          _mm_dpbusd_epi32(c.value, unsigned_value, signed_value)};
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<ToTag>{
          _mm256_dpbusd_epi32(c.value, unsigned_value, signed_value)};
#if defined(HAS_AVX512VNNI)
    else
      return NativeWordVec<ToTag>{
          _mm512_dpbusd_epi32(c.value, unsigned_value, signed_value)};
#endif
  } else if constexpr (
      (std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
      std::same_as<From1, int8_t> && std::same_as<From2, int8_t>) {
    if constexpr (sizeof(Raw) == 16) {
      const auto bias = _mm_set1_epi8(static_cast<char>(0x80));
      const auto biased_a = _mm_xor_si128(a.value, bias);
      const auto main = _mm_dpbusd_epi32(c.value, biased_a, b.value);
      const auto correction =
          _mm_dpbusd_epi32(_mm_setzero_si128(), bias, b.value);
      return NativeWordVec<ToTag>{_mm_sub_epi32(main, correction)};
    } else if constexpr (sizeof(Raw) == 32) {
      const auto bias = _mm256_set1_epi8(static_cast<char>(0x80));
      const auto biased_a = _mm256_xor_si256(a.value, bias);
      const auto main = _mm256_dpbusd_epi32(c.value, biased_a, b.value);
      const auto correction =
          _mm256_dpbusd_epi32(_mm256_setzero_si256(), bias, b.value);
      return NativeWordVec<ToTag>{_mm256_sub_epi32(main, correction)};
#if defined(HAS_AVX512VNNI)
    } else {
      const auto bias = _mm512_set1_epi8(static_cast<char>(0x80));
      const auto biased_a = _mm512_xor_si512(a.value, bias);
      const auto main = _mm512_dpbusd_epi32(c.value, biased_a, b.value);
      const auto correction =
          _mm512_dpbusd_epi32(_mm512_setzero_si512(), bias, b.value);
      return NativeWordVec<ToTag>{_mm512_sub_epi32(main, correction)};
#endif
    }
  } else if constexpr (
      (std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
      std::same_as<From1, uint8_t> && std::same_as<From2, uint8_t>) {
    if constexpr (sizeof(Raw) == 16) {
      const auto flip = _mm_set1_epi8(static_cast<char>(0x80));
      const auto ones = _mm_set1_epi8(1);
      const auto biased_b = _mm_xor_si128(b.value, flip);
      const auto main = _mm_dpbusd_epi32(c.value, a.value, biased_b);
      const auto sums =
          _mm_dpbusd_epi32(_mm_setzero_si128(), a.value, ones);
      return NativeWordVec<ToTag>{
          _mm_add_epi32(main, _mm_slli_epi32(sums, 7))};
    } else if constexpr (sizeof(Raw) == 32) {
      const auto flip = _mm256_set1_epi8(static_cast<char>(0x80));
      const auto ones = _mm256_set1_epi8(1);
      const auto biased_b = _mm256_xor_si256(b.value, flip);
      const auto main = _mm256_dpbusd_epi32(c.value, a.value, biased_b);
      const auto sums =
          _mm256_dpbusd_epi32(_mm256_setzero_si256(), a.value, ones);
      return NativeWordVec<ToTag>{
          _mm256_add_epi32(main, _mm256_slli_epi32(sums, 7))};
#if defined(HAS_AVX512VNNI)
    } else {
      const auto flip = _mm512_set1_epi8(static_cast<char>(0x80));
      const auto ones = _mm512_set1_epi8(1);
      const auto biased_b = _mm512_xor_si512(b.value, flip);
      const auto main = _mm512_dpbusd_epi32(c.value, a.value, biased_b);
      const auto sums =
          _mm512_dpbusd_epi32(_mm512_setzero_si512(), a.value, ones);
      return NativeWordVec<ToTag>{
          _mm512_add_epi32(main, _mm512_slli_epi32(sums, 7))};
#endif
    }
  } else if constexpr (
      (std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
      std::same_as<From1, int16_t> && std::same_as<From2, int16_t>) {
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<ToTag>{
          _mm_dpwssd_epi32(c.value, a.value, b.value)};
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<ToTag>{
          _mm256_dpwssd_epi32(c.value, a.value, b.value)};
#if defined(HAS_AVX512VNNI)
    else
      return NativeWordVec<ToTag>{
          _mm512_dpwssd_epi32(c.value, a.value, b.value)};
#endif
  } else
#endif
  {
    static_assert(dispatch_dependent_false<To, From1, From2>);
  }
}

template <VectorTag ToTag>
struct NativeImpl<X86Backend, WideningDotOp, ToTag> {
  template <VectorTag FromTag1, VectorTag FromTag2>
    requires x86_has_native_widening_dot_v<
        ElementOf<ToTag>, ElementOf<FromTag1>, ElementOf<FromTag2>>
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      WideningDotOp op, ToTag to, FromTag1 from1, FromTag2 from2,
      Vec<FromTag1> a, Vec<FromTag2> b) {
    return call(op, to, from1, from2, a, b, zeros(to));
  }

  template <VectorTag FromTag1, VectorTag FromTag2>
    requires x86_has_native_widening_dot_v<
        ElementOf<ToTag>, ElementOf<FromTag1>, ElementOf<FromTag2>>
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      WideningDotOp, ToTag to, FromTag1 from1, FromTag2 from2,
      Vec<FromTag1> a, Vec<FromTag2> b, Vec<ToTag> c) {
    using ToTraits = RepresentationTraits<X86Backend, ToTag>;
    using FromTraits1 = RepresentationTraits<X86Backend, FromTag1>;
    using FromTraits2 = RepresentationTraits<X86Backend, FromTag2>;
    static_assert(ToTraits::word_count == FromTraits1::word_count);
    static_assert(ToTraits::word_count == FromTraits2::word_count);
    return construct_words<X86Backend>(
        to, [&]<nint_t Index>(ToTag) VECOPS_INLINE_LAMBDA {
      return x86_widening_dot_word<ToTag, FromTag1, FromTag2>(
          ::vecops::vec::get_word<Index>(from1, a),
          ::vecops::vec::get_word<Index>(from2, b),
          ::vecops::vec::get_word<Index>(to, c));
    });
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_WIDENINGDOT_H
