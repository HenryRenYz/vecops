// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_X86_ROUNDING_H
#define VECOPS_VEC_DETAILS_X86_ROUNDING_H

/**
 * @file Rounding.h
 * @brief x86 rounding using ROUNDPS/PD, VRNDSCALEPS/PD/PH, and widening
 * small-float fallbacks.
 */

#include "vecops/vec/details/Rounding.h"
#include "vecops/vec/details/x86/Arithmetic.h"

namespace vecops::vec::details {

template <typename Op>
consteval int x86_rounding_control() {
  if constexpr (std::same_as<Op, FloorOp>)
    return _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
  else if constexpr (std::same_as<Op, CeilOp>)
    return _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
  else if constexpr (std::same_as<Op, TruncOp>)
    return _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
  else if constexpr (std::same_as<Op, RoundEvenOp>)
    return _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  else if constexpr (std::same_as<Op, NearbyIntOp>)
    return _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC;
  else if constexpr (std::same_as<Op, RintOp>)
    return _MM_FROUND_CUR_DIRECTION;
  else
    static_assert(dispatch_dependent_false<Op>);
}

template <typename Op, Element T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_round_native_float(Raw value) {
  static_assert(
      std::same_as<T, float32_t> || std::same_as<T, float64_t>);
  constexpr int control = x86_rounding_control<Op>();
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16) {
#if defined(HAS_SSE4_1)
      return _mm_round_ps(value, control);
#else
      static_assert(dispatch_dependent_false<Raw>);
#endif
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) {
      return _mm256_round_ps(value, control);
    }
#endif
#if VEC_WIDTH >= 512
    else {
      return _mm512_roundscale_ps(value, control);
    }
#endif
  } else {
    if constexpr (sizeof(Raw) == 16) {
#if defined(HAS_SSE4_1)
      return _mm_round_pd(value, control);
#else
      static_assert(dispatch_dependent_false<Raw>);
#endif
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) {
      return _mm256_round_pd(value, control);
    }
#endif
#if VEC_WIDTH >= 512
    else {
      return _mm512_roundscale_pd(value, control);
    }
#endif
  }
}

template <Element T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_round_away_native(Raw value) {
  const auto integral = x86_round_native_float<TruncOp, T>(value);
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16) {
      const auto fraction = _mm_sub_ps(value, integral);
      const auto magnitude = _mm_and_ps(
          fraction, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));
      const auto adjust = _mm_cmpge_ps(magnitude, _mm_set1_ps(0.5F));
      const auto signed_one = _mm_or_ps(
          _mm_set1_ps(1.0F),
          _mm_and_ps(value, _mm_set1_ps(-0.0F)));
      return _mm_blendv_ps(
          integral, _mm_add_ps(integral, signed_one), adjust);
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) {
      const auto fraction = _mm256_sub_ps(value, integral);
      const auto magnitude = _mm256_and_ps(
          fraction, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
      const auto adjust =
          _mm256_cmp_ps(magnitude, _mm256_set1_ps(0.5F), _CMP_GE_OQ);
      const auto signed_one = _mm256_or_ps(
          _mm256_set1_ps(1.0F),
          _mm256_and_ps(value, _mm256_set1_ps(-0.0F)));
      return _mm256_blendv_ps(
          integral, _mm256_add_ps(integral, signed_one), adjust);
    }
#endif
#if VEC_WIDTH >= 512
    else {
      const auto fraction = _mm512_sub_ps(value, integral);
      const auto magnitude = _mm512_and_ps(
          fraction, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)));
      const auto adjust =
          _mm512_cmp_ps_mask(magnitude, _mm512_set1_ps(0.5F), _CMP_GE_OQ);
      const auto signed_one = _mm512_or_ps(
          _mm512_set1_ps(1.0F),
          _mm512_and_ps(value, _mm512_set1_ps(-0.0F)));
      return _mm512_mask_add_ps(
          integral, adjust, integral, signed_one);
    }
#endif
  } else {
    if constexpr (sizeof(Raw) == 16) {
      const auto fraction = _mm_sub_pd(value, integral);
      const auto magnitude = _mm_and_pd(
          fraction, _mm_castsi128_pd(_mm_set1_epi64x(
                        static_cast<long long>(0x7fffffffffffffffull))));
      const auto adjust = _mm_cmpge_pd(magnitude, _mm_set1_pd(0.5));
      const auto signed_one = _mm_or_pd(
          _mm_set1_pd(1.0),
          _mm_and_pd(value, _mm_set1_pd(-0.0)));
      return _mm_blendv_pd(
          integral, _mm_add_pd(integral, signed_one), adjust);
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) {
      const auto fraction = _mm256_sub_pd(value, integral);
      const auto magnitude = _mm256_and_pd(
          fraction, _mm256_castsi256_pd(_mm256_set1_epi64x(
                        static_cast<long long>(0x7fffffffffffffffull))));
      const auto adjust =
          _mm256_cmp_pd(magnitude, _mm256_set1_pd(0.5), _CMP_GE_OQ);
      const auto signed_one = _mm256_or_pd(
          _mm256_set1_pd(1.0),
          _mm256_and_pd(value, _mm256_set1_pd(-0.0)));
      return _mm256_blendv_pd(
          integral, _mm256_add_pd(integral, signed_one), adjust);
    }
#endif
#if VEC_WIDTH >= 512
    else {
      const auto fraction = _mm512_sub_pd(value, integral);
      const auto magnitude = _mm512_and_pd(
          fraction, _mm512_castsi512_pd(_mm512_set1_epi64(
                        static_cast<long long>(0x7fffffffffffffffull))));
      const auto adjust =
          _mm512_cmp_pd_mask(magnitude, _mm512_set1_pd(0.5), _CMP_GE_OQ);
      const auto signed_one = _mm512_or_pd(
          _mm512_set1_pd(1.0),
          _mm512_and_pd(value, _mm512_set1_pd(-0.0)));
      return _mm512_mask_add_pd(
          integral, adjust, integral, signed_one);
    }
#endif
  }
}

template <typename Op, Element T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_round_float(Raw value) {
  if constexpr (std::same_as<Op, RoundOp>)
    return x86_round_away_native<T>(value);
  else
    return x86_round_native_float<Op, T>(value);
}

template <typename Op, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_round_native_float16(Raw value) {
  constexpr int control = x86_rounding_control<Op>();
  if constexpr (sizeof(Raw) == 16)
    return _mm_castph_si128(
        _mm_roundscale_ph(_mm_castsi128_ph(value), control));
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_castph_si256(
        _mm256_roundscale_ph(_mm256_castsi256_ph(value), control));
#endif
#if VEC_WIDTH >= 512
  else
    return _mm512_castph_si512(
        _mm512_roundscale_ph(_mm512_castsi512_ph(value), control));
#endif
}

template <typename Op, Element T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_round_small_float(Raw value) {
  static_assert(
      std::same_as<T, float16_t> || std::same_as<T, bfloat16_t>);
  const auto expand = []<typename Float>(
                          Raw input, Float& low, Float& high) {
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(input, low, high);
    else
      x86_float16_to_float32_pair(input, low, high);
  };
  const auto narrow = []<typename Float>(Float low, Float high) -> Raw {
    if constexpr (std::same_as<T, bfloat16_t>) {
#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
      return x86_float32_pair_to_bfloat16(
          x86_canonicalize_float32_nan(low),
          x86_canonicalize_float32_nan(high));
#else
      return x86_float32_pair_to_bfloat16(low, high);
#endif
    } else {
      return x86_float32_pair_to_float16(low, high);
    }
  };
  const auto apply = [](auto part) {
    return x86_round_float<Op, float32_t>(part);
  };

#define VECOPS_VEC_X86_ROUND_SMALL(Float)                              \
  Float low, high;                                                     \
  expand(value, low, high);                                           \
  return narrow(apply(low), apply(high))
  if constexpr (sizeof(Raw) == 16) {
    VECOPS_VEC_X86_ROUND_SMALL(__m128);
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    VECOPS_VEC_X86_ROUND_SMALL(__m256);
  }
#endif
#if VEC_WIDTH >= 512
  else {
    VECOPS_VEC_X86_ROUND_SMALL(__m512);
  }
#endif
#undef VECOPS_VEC_X86_ROUND_SMALL
}

template <typename Op>
struct X86RoundingWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    using T = ElementOf<Tag>;
    using Raw = decltype(value.value);
    static_assert(Index >= 0 && Index < num_words(tag));
    if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
      if constexpr (!std::same_as<Op, RoundOp>)
        return NativeWordVec<Tag>{
            x86_round_native_float16<Op>(value.value)};
#endif
      return NativeWordVec<Tag>{
          x86_round_small_float<Op, T>(value.value)};
    } else if constexpr (std::same_as<T, bfloat16_t>) {
      return NativeWordVec<Tag>{
          x86_round_small_float<Op, T>(value.value)};
    } else {
      return NativeWordVec<Tag>{x86_round_float<Op, T>(value.value)};
    }
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return blend(tag, inactive, mask, call<Index>(op, tag, value));
  }
};

#define VECOPS_VEC_REGISTER_X86_ROUNDING(OpType)                       \
  template <>                                                          \
  struct NativeWordImpl<X86Backend, OpType>                            \
      : X86RoundingWordImpl<OpType> {}

VECOPS_VEC_REGISTER_X86_ROUNDING(FloorOp);
VECOPS_VEC_REGISTER_X86_ROUNDING(CeilOp);
VECOPS_VEC_REGISTER_X86_ROUNDING(TruncOp);
VECOPS_VEC_REGISTER_X86_ROUNDING(RoundOp);
VECOPS_VEC_REGISTER_X86_ROUNDING(RoundEvenOp);
VECOPS_VEC_REGISTER_X86_ROUNDING(NearbyIntOp);
VECOPS_VEC_REGISTER_X86_ROUNDING(RintOp);

#undef VECOPS_VEC_REGISTER_X86_ROUNDING

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_ROUNDING_H
