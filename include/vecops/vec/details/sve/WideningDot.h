// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_WIDENINGDOT_H
#define VECOPS_VEC_DETAILS_SVE_WIDENINGDOT_H

/**
 * @file WideningDot.h
 * @brief SVE grouped widening-dot specializations. Native instruction paths
 * are added here; unsupported combinations deliberately use the shared
 * phase-selection fallback.
 */

#include "vecops/vec/details/WideningDot.h"

namespace vecops::vec::details {

template <typename To, typename From1, typename From2>
inline constexpr bool sve_has_native_widening_dot_v =
#if defined(__ARM_FEATURE_SVE_BF16)
    (std::same_as<To, float32_t> &&
     std::same_as<From1, bfloat16_t> &&
     std::same_as<From2, bfloat16_t>) ||
#endif
#if defined(HAS_SVE2) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    (std::same_as<To, float32_t> &&
     std::same_as<From1, float16_t> &&
     std::same_as<From2, float16_t>) ||
#endif
#if defined(HAS_SVE2)
    ((sizeof(To) == 2 * sizeof(From1)) &&
     (std::same_as<To, int16_t> || std::same_as<To, uint16_t> ||
      std::same_as<To, int32_t> || std::same_as<To, uint32_t> ||
      std::same_as<To, int64_t> || std::same_as<To, uint64_t>) &&
     ((std::signed_integral<From1> && std::same_as<From1, From2>) ||
      (std::unsigned_integral<From1> && std::same_as<From1, From2>))) ||
#endif
    ((std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
     ((std::same_as<From1, int8_t> && std::same_as<From2, int8_t>) ||
      (std::same_as<From1, uint8_t> && std::same_as<From2, uint8_t>))) ||
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    ((std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
     ((std::same_as<From1, int8_t> && std::same_as<From2, uint8_t>) ||
      (std::same_as<From1, uint8_t> && std::same_as<From2, int8_t>))) ||
#endif
    ((std::same_as<To, int64_t> || std::same_as<To, uint64_t>) &&
     ((std::same_as<From1, int16_t> && std::same_as<From2, int16_t>) ||
      (std::same_as<From1, uint16_t> && std::same_as<From2, uint16_t>))) ||
    false;

template <VectorTag ToTag, VectorTag FromTag1, VectorTag FromTag2>
VECOPS_ALWAYS_INLINE NativeWordVec<ToTag> sve_widening_dot_word(
    NativeWordVec<FromTag1> a, NativeWordVec<FromTag2> b,
    NativeWordVec<ToTag> c) {
  using To = ElementOf<ToTag>;
  using From1 = ElementOf<FromTag1>;
  using From2 = ElementOf<FromTag2>;
  const auto raw_a = sve_basic_raw_word(a);
  const auto raw_b = sve_basic_raw_word(b);
  const auto raw_c = sve_basic_raw_word(c);

#if defined(__ARM_FEATURE_SVE_BF16)
  if constexpr (
      std::same_as<To, float32_t> &&
      std::same_as<From1, bfloat16_t> &&
      std::same_as<From2, bfloat16_t>) {
    return sve_basic_wrap_word<ToTag>(svbfdot_f32(raw_c, raw_a, raw_b));
  } else
#endif
#if defined(HAS_SVE2) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  if constexpr (
      std::same_as<To, float32_t> &&
      std::same_as<From1, float16_t> &&
      std::same_as<From2, float16_t>) {
#if defined(HAS_SVE2P1)
    return sve_basic_wrap_word<ToTag>(
        svdot_f32_f16(raw_c, raw_a, raw_b));
#else
    return sve_basic_wrap_word<ToTag>(
        svmlalt_f32(svmlalb_f32(raw_c, raw_a, raw_b), raw_a, raw_b));
#endif
  } else
#endif
#if defined(HAS_SVE2)
  if constexpr (
      sizeof(To) == 2 * sizeof(From1) &&
      std::same_as<From1, From2> && std::signed_integral<From1>) {
    const auto signed_c = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::signed_integral<To>) return raw_c;
      else if constexpr (sizeof(To) == 2) return svreinterpret_s16_u16(raw_c);
      else if constexpr (sizeof(To) == 4) return svreinterpret_s32_u32(raw_c);
      else return svreinterpret_s64_u64(raw_c);
    }();
    const auto result = [&]() VECOPS_INLINE_LAMBDA {
#if defined(HAS_SVE2P1)
      if constexpr (sizeof(From1) == 2)
        return svdot_s32_s16(signed_c, raw_a, raw_b);
      else
#endif
      if constexpr (sizeof(From1) == 1)
        return svmlalt_s16(
            svmlalb_s16(signed_c, raw_a, raw_b), raw_a, raw_b);
      else if constexpr (sizeof(From1) == 2)
        return svmlalt_s32(
            svmlalb_s32(signed_c, raw_a, raw_b), raw_a, raw_b);
      else
        return svmlalt_s64(
            svmlalb_s64(signed_c, raw_a, raw_b), raw_a, raw_b);
    }();
    if constexpr (std::signed_integral<To>)
      return sve_basic_wrap_word<ToTag>(result);
    else if constexpr (sizeof(To) == 2)
      return sve_basic_wrap_word<ToTag>(svreinterpret_u16_s16(result));
    else if constexpr (sizeof(To) == 4)
      return sve_basic_wrap_word<ToTag>(svreinterpret_u32_s32(result));
    else
      return sve_basic_wrap_word<ToTag>(svreinterpret_u64_s64(result));
  } else if constexpr (
      sizeof(To) == 2 * sizeof(From1) &&
      std::same_as<From1, From2> && std::unsigned_integral<From1>) {
    const auto unsigned_c = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::unsigned_integral<To>) return raw_c;
      else if constexpr (sizeof(To) == 2) return svreinterpret_u16_s16(raw_c);
      else if constexpr (sizeof(To) == 4) return svreinterpret_u32_s32(raw_c);
      else return svreinterpret_u64_s64(raw_c);
    }();
    const auto result = [&]() VECOPS_INLINE_LAMBDA {
#if defined(HAS_SVE2P1)
      if constexpr (sizeof(From1) == 2)
        return svdot_u32_u16(unsigned_c, raw_a, raw_b);
      else
#endif
      if constexpr (sizeof(From1) == 1)
        return svmlalt_u16(
            svmlalb_u16(unsigned_c, raw_a, raw_b), raw_a, raw_b);
      else if constexpr (sizeof(From1) == 2)
        return svmlalt_u32(
            svmlalb_u32(unsigned_c, raw_a, raw_b), raw_a, raw_b);
      else
        return svmlalt_u64(
            svmlalb_u64(unsigned_c, raw_a, raw_b), raw_a, raw_b);
    }();
    if constexpr (std::unsigned_integral<To>)
      return sve_basic_wrap_word<ToTag>(result);
    else if constexpr (sizeof(To) == 2)
      return sve_basic_wrap_word<ToTag>(svreinterpret_s16_u16(result));
    else if constexpr (sizeof(To) == 4)
      return sve_basic_wrap_word<ToTag>(svreinterpret_s32_u32(result));
    else
      return sve_basic_wrap_word<ToTag>(svreinterpret_s64_u64(result));
  } else
#endif
  if constexpr (
      (std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
      std::same_as<From1, int8_t> && std::same_as<From2, int8_t>) {
    const auto signed_c = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<To, int32_t>) return raw_c;
      else return svreinterpret_s32_u32(raw_c);
    }();
    const auto result = svdot_s32(signed_c, raw_a, raw_b);
    if constexpr (std::same_as<To, int32_t>)
      return sve_basic_wrap_word<ToTag>(result);
    else
      return sve_basic_wrap_word<ToTag>(svreinterpret_u32_s32(result));
  } else if constexpr (
      (std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
      std::same_as<From1, uint8_t> && std::same_as<From2, uint8_t>) {
    const auto unsigned_c = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<To, uint32_t>) return raw_c;
      else return svreinterpret_u32_s32(raw_c);
    }();
    const auto result = svdot_u32(unsigned_c, raw_a, raw_b);
    if constexpr (std::same_as<To, uint32_t>)
      return sve_basic_wrap_word<ToTag>(result);
    else
      return sve_basic_wrap_word<ToTag>(svreinterpret_s32_u32(result));
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
  } else if constexpr (
      (std::same_as<To, int32_t> || std::same_as<To, uint32_t>) &&
      ((std::same_as<From1, int8_t> && std::same_as<From2, uint8_t>) ||
       (std::same_as<From1, uint8_t> && std::same_as<From2, int8_t>))) {
    const auto signed_c = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<To, int32_t>) return raw_c;
      else return svreinterpret_s32_u32(raw_c);
    }();
    const auto result = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<From1, int8_t>)
        return svsudot_s32(signed_c, raw_a, raw_b);
      else
        return svusdot_s32(signed_c, raw_a, raw_b);
    }();
    if constexpr (std::same_as<To, int32_t>)
      return sve_basic_wrap_word<ToTag>(result);
    else
      return sve_basic_wrap_word<ToTag>(svreinterpret_u32_s32(result));
#endif
  } else if constexpr (
      (std::same_as<To, int64_t> || std::same_as<To, uint64_t>) &&
      std::same_as<From1, int16_t> && std::same_as<From2, int16_t>) {
    const auto signed_c = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<To, int64_t>) return raw_c;
      else return svreinterpret_s64_u64(raw_c);
    }();
    const auto result = svdot_s64(signed_c, raw_a, raw_b);
    if constexpr (std::same_as<To, int64_t>)
      return sve_basic_wrap_word<ToTag>(result);
    else
      return sve_basic_wrap_word<ToTag>(svreinterpret_u64_s64(result));
  } else if constexpr (
      (std::same_as<To, int64_t> || std::same_as<To, uint64_t>) &&
      std::same_as<From1, uint16_t> && std::same_as<From2, uint16_t>) {
    const auto unsigned_c = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<To, uint64_t>) return raw_c;
      else return svreinterpret_u64_s64(raw_c);
    }();
    const auto result = svdot_u64(unsigned_c, raw_a, raw_b);
    if constexpr (std::same_as<To, uint64_t>)
      return sve_basic_wrap_word<ToTag>(result);
    else
      return sve_basic_wrap_word<ToTag>(svreinterpret_s64_u64(result));
  } else {
    static_assert(dispatch_dependent_false<To, From1, From2>);
  }
}

template <VectorTag ToTag>
struct NativeImpl<SVEBackend, WideningDotOp, ToTag> {
  template <VectorTag FromTag1, VectorTag FromTag2>
    requires sve_has_native_widening_dot_v<
        ElementOf<ToTag>, ElementOf<FromTag1>, ElementOf<FromTag2>>
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      WideningDotOp op, ToTag to, FromTag1 from1, FromTag2 from2,
      Vec<FromTag1> a, Vec<FromTag2> b) {
    return call(op, to, from1, from2, a, b, zeros(to));
  }

  template <VectorTag FromTag1, VectorTag FromTag2>
    requires sve_has_native_widening_dot_v<
        ElementOf<ToTag>, ElementOf<FromTag1>, ElementOf<FromTag2>>
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      WideningDotOp, ToTag to, FromTag1 from1, FromTag2 from2,
      Vec<FromTag1> a, Vec<FromTag2> b, Vec<ToTag> c) {
    using ToTraits = RepresentationTraits<SVEBackend, ToTag>;
    using FromTraits1 = RepresentationTraits<SVEBackend, FromTag1>;
    using FromTraits2 = RepresentationTraits<SVEBackend, FromTag2>;
    static_assert(ToTraits::word_count == FromTraits1::word_count);
    static_assert(ToTraits::word_count == FromTraits2::word_count);
    return construct_words<SVEBackend>(
        to, [&]<nint_t Index>(ToTag) VECOPS_INLINE_LAMBDA {
      return sve_widening_dot_word<ToTag, FromTag1, FromTag2>(
          ::vecops::vec::get_word<Index>(from1, a),
          ::vecops::vec::get_word<Index>(from2, b),
          ::vecops::vec::get_word<Index>(to, c));
    });
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_WIDENINGDOT_H
