// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_X86_REDUCTION_H
#define VECOPS_VEC_DETAILS_X86_REDUCTION_H

/**
 * @file Reduction.h
 * @brief x86 backend implementations for reduction operations.
 */

#include <cstring>

#include "vecops/vec/details/x86/Arithmetic.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                       Reduction word implementations                       //
/* **************************************************************************** */

template <typename Op, VectorTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_reduction_combine(
    Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
  if constexpr (std::same_as<Op, ReduceAddOp>) return add(tag, a, b);
  else if constexpr (std::same_as<Op, ReduceMaxOp>) return max(tag, a, b);
  else if constexpr (std::same_as<Op, ReduceMinOp>) return min(tag, a, b);
  else static_assert(dispatch_dependent_false<Op>);
}

template <typename Op, Element T, typename Raw>
VECOPS_ALWAYS_INLINE T x86_horizontal_reduction(Raw value) {
  if constexpr (sizeof(Raw) > 16) {
    using HalfTag = FixedTag<
        T, static_cast<nint_t>(sizeof(Raw) / (2 * sizeof(T)))>;
    using HalfRaw =
        typename RepresentationTraits<X86Backend, HalfTag>::RawVec;
    HalfRaw lower;
    HalfRaw upper;
    constexpr auto domain = x86_reg_domain_of_v<T>;
    if constexpr (sizeof(Raw) == 32) {
      if constexpr (domain == x86_reg_domain::f32) {
        lower = _mm256_castps256_ps128(value);
        upper = _mm256_extractf128_ps(value, 1);
      } else if constexpr (domain == x86_reg_domain::f64) {
        lower = _mm256_castpd256_pd128(value);
        upper = _mm256_extractf128_pd(value, 1);
      } else {
        lower = _mm256_castsi256_si128(value);
        upper = _mm256_extracti128_si256(value, 1);
      }
    }
#if VEC_WIDTH >= 512
    else {
      if constexpr (domain == x86_reg_domain::f32) {
        lower = _mm512_castps512_ps256(value);
        upper = _mm512_extractf32x8_ps(value, 1);
      } else if constexpr (domain == x86_reg_domain::f64) {
        lower = _mm512_castpd512_pd256(value);
        upper = _mm512_extractf64x4_pd(value, 1);
      } else {
        lower = _mm512_castsi512_si256(value);
        upper = _mm512_extracti64x4_epi64(value, 1);
      }
    }
#endif
    const auto combined = x86_reduction_combine<Op>(
        HalfTag{}, NativeWordVec<HalfTag>{lower},
        NativeWordVec<HalfTag>{upper});
    return x86_horizontal_reduction<Op, T>(combined.value);
  } else {
    using WordTag = FixedTag<
        T, static_cast<nint_t>(sizeof(Raw) / sizeof(T))>;
    NativeWordVec<WordTag> reduced{value};
    const auto fold = [&]<int Bytes>() VECOPS_INLINE_LAMBDA {
      const auto bits = x86_arithmetic_to_integer_bits<T>(reduced.value);
      const auto shifted_bits = _mm_srli_si128(bits, Bytes);
      const auto shifted = x86_arithmetic_from_integer_bits<
          T, Raw>(shifted_bits);
      reduced = x86_reduction_combine<Op>(
          WordTag{}, reduced, NativeWordVec<WordTag>{shifted});
    };
    if constexpr (sizeof(T) <= 8) fold.template operator()<8>();
    if constexpr (sizeof(T) <= 4) fold.template operator()<4>();
    if constexpr (sizeof(T) <= 2) fold.template operator()<2>();
    if constexpr (sizeof(T) <= 1) fold.template operator()<1>();
    T result;
    std::memcpy(&result, &reduced.value, sizeof(T));
    return result;
  }
}

template <typename Op, nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE ElementOf<Tag> x86_simd_reduction(
    Tag tag, NativeWordVec<Tag> value, const NativeWordMask<Tag>* mask) {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  using T = ElementOf<Tag>;
  using MaskRaw = typename Traits::RawMask;
  constexpr nint_t valid = valid_word_lanes<Index, Tag>();
  if constexpr (valid < Traits::word_lanes) {
    const MaskRaw prefix = x86_mask_prefix<T, MaskRaw>(valid);
    const MaskRaw active = mask == nullptr
        ? prefix : x86_mask_and(mask->value, prefix);
    value = blend(
        tag, fill_word(tag, reduction_identity<Op, T>()),
        NativeWordMask<Tag>{active}, value);
  } else if (mask != nullptr) {
    value = blend(
        tag, fill_word(tag, reduction_identity<Op, T>()), *mask, value);
  }
  return x86_horizontal_reduction<Op, T>(value.value);
}

template <typename Op>
struct X86ReductionWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = typename Traits::RawVec;
    constexpr nint_t valid = valid_word_lanes<Index, Tag>();
    constexpr nint_t lanes = static_cast<nint_t>(sizeof(Raw) / sizeof(T));
#if defined(HAS_AVX512F)
    if constexpr (sizeof(Raw) == 64) {
      using MaskRaw = typename Traits::RawMask;
      const MaskRaw active = x86_mask_prefix<T, MaskRaw>(valid);
      if constexpr (std::same_as<Op, ReduceAddOp>) {
        if constexpr (std::same_as<T, float32_t>)
          return _mm512_mask_reduce_add_ps(active, value.value);
        else if constexpr (std::same_as<T, float64_t>)
          return _mm512_mask_reduce_add_pd(active, value.value);
        else if constexpr (std::same_as<T, int32_t> || std::same_as<T, uint32_t>)
          return static_cast<T>(_mm512_mask_reduce_add_epi32(active, value.value));
        else if constexpr (std::same_as<T, int64_t> || std::same_as<T, uint64_t>)
          return static_cast<T>(_mm512_mask_reduce_add_epi64(active, value.value));
      } else if constexpr (std::same_as<Op, ReduceMaxOp>) {
        if constexpr (std::same_as<T, float32_t>)
          return _mm512_mask_reduce_max_ps(active, value.value);
        else if constexpr (std::same_as<T, float64_t>)
          return _mm512_mask_reduce_max_pd(active, value.value);
        else if constexpr (std::same_as<T, int32_t>)
          return static_cast<T>(_mm512_mask_reduce_max_epi32(active, value.value));
        else if constexpr (std::same_as<T, uint32_t>)
          return static_cast<T>(_mm512_mask_reduce_max_epu32(active, value.value));
        else if constexpr (std::same_as<T, int64_t>)
          return static_cast<T>(_mm512_mask_reduce_max_epi64(active, value.value));
        else if constexpr (std::same_as<T, uint64_t>)
          return static_cast<T>(_mm512_mask_reduce_max_epu64(active, value.value));
      } else if constexpr (std::same_as<Op, ReduceMinOp>) {
        if constexpr (std::same_as<T, float32_t>)
          return _mm512_mask_reduce_min_ps(active, value.value);
        else if constexpr (std::same_as<T, float64_t>)
          return _mm512_mask_reduce_min_pd(active, value.value);
        else if constexpr (std::same_as<T, int32_t>)
          return static_cast<T>(_mm512_mask_reduce_min_epi32(active, value.value));
        else if constexpr (std::same_as<T, uint32_t>)
          return static_cast<T>(_mm512_mask_reduce_min_epu32(active, value.value));
        else if constexpr (std::same_as<T, int64_t>)
          return static_cast<T>(_mm512_mask_reduce_min_epi64(active, value.value));
        else if constexpr (std::same_as<T, uint64_t>)
          return static_cast<T>(_mm512_mask_reduce_min_epu64(active, value.value));
      }
    }
#endif
#if defined(HAS_AVX512_FP16)
    if constexpr (std::same_as<T, float16_t> && valid == lanes) {
      if constexpr (sizeof(Raw) == 16) {
        if constexpr (std::same_as<Op, ReduceAddOp>)
          return static_cast<T>(_mm_reduce_add_ph(_mm_castsi128_ph(value.value)));
        else if constexpr (std::same_as<Op, ReduceMaxOp>)
          return static_cast<T>(_mm_reduce_max_ph(_mm_castsi128_ph(value.value)));
        else
          return static_cast<T>(_mm_reduce_min_ph(_mm_castsi128_ph(value.value)));
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32) {
        if constexpr (std::same_as<Op, ReduceAddOp>)
          return static_cast<T>(_mm256_reduce_add_ph(_mm256_castsi256_ph(value.value)));
        else if constexpr (std::same_as<Op, ReduceMaxOp>)
          return static_cast<T>(_mm256_reduce_max_ph(_mm256_castsi256_ph(value.value)));
        else
          return static_cast<T>(_mm256_reduce_min_ph(_mm256_castsi256_ph(value.value)));
      }
#endif
#if VEC_WIDTH >= 512
      else if constexpr (sizeof(Raw) == 64) {
        if constexpr (std::same_as<Op, ReduceAddOp>)
          return static_cast<T>(_mm512_reduce_add_ph(_mm512_castsi512_ph(value.value)));
        else if constexpr (std::same_as<Op, ReduceMaxOp>)
          return static_cast<T>(_mm512_reduce_max_ph(_mm512_castsi512_ph(value.value)));
        else
          return static_cast<T>(_mm512_reduce_min_ph(_mm512_castsi512_ph(value.value)));
      }
#endif
    }
#endif
    return x86_simd_reduction<Op, Index>(tag, value, nullptr);
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = typename Traits::RawVec;
#if defined(HAS_AVX512F)
    if constexpr (sizeof(Raw) == 64) {
      using MaskRaw = typename Traits::RawMask;
      const MaskRaw valid = x86_mask_prefix<T, MaskRaw>(
          valid_word_lanes<Index, Tag>());
      const MaskRaw active = x86_mask_and(mask.value, valid);
      if constexpr (std::same_as<Op, ReduceAddOp>) {
        if constexpr (std::same_as<T, float32_t>)
          return _mm512_mask_reduce_add_ps(active, value.value);
        else if constexpr (std::same_as<T, float64_t>)
          return _mm512_mask_reduce_add_pd(active, value.value);
        else if constexpr (std::same_as<T, int32_t> || std::same_as<T, uint32_t>)
          return static_cast<T>(_mm512_mask_reduce_add_epi32(active, value.value));
        else if constexpr (std::same_as<T, int64_t> || std::same_as<T, uint64_t>)
          return static_cast<T>(_mm512_mask_reduce_add_epi64(active, value.value));
      } else if constexpr (std::same_as<Op, ReduceMaxOp>) {
        if constexpr (std::same_as<T, float32_t>)
          return _mm512_mask_reduce_max_ps(active, value.value);
        else if constexpr (std::same_as<T, float64_t>)
          return _mm512_mask_reduce_max_pd(active, value.value);
        else if constexpr (std::same_as<T, int32_t>)
          return static_cast<T>(_mm512_mask_reduce_max_epi32(active, value.value));
        else if constexpr (std::same_as<T, uint32_t>)
          return static_cast<T>(_mm512_mask_reduce_max_epu32(active, value.value));
        else if constexpr (std::same_as<T, int64_t>)
          return static_cast<T>(_mm512_mask_reduce_max_epi64(active, value.value));
        else if constexpr (std::same_as<T, uint64_t>)
          return static_cast<T>(_mm512_mask_reduce_max_epu64(active, value.value));
      } else if constexpr (std::same_as<Op, ReduceMinOp>) {
        if constexpr (std::same_as<T, float32_t>)
          return _mm512_mask_reduce_min_ps(active, value.value);
        else if constexpr (std::same_as<T, float64_t>)
          return _mm512_mask_reduce_min_pd(active, value.value);
        else if constexpr (std::same_as<T, int32_t>)
          return static_cast<T>(_mm512_mask_reduce_min_epi32(active, value.value));
        else if constexpr (std::same_as<T, uint32_t>)
          return static_cast<T>(_mm512_mask_reduce_min_epu32(active, value.value));
        else if constexpr (std::same_as<T, int64_t>)
          return static_cast<T>(_mm512_mask_reduce_min_epi64(active, value.value));
        else if constexpr (std::same_as<T, uint64_t>)
          return static_cast<T>(_mm512_mask_reduce_min_epu64(active, value.value));
      }
    }
#endif
    return x86_simd_reduction<Op, Index>(tag, value, &mask);
  }
};

template <>
struct NativeWordImpl<X86Backend, ReduceAddOp>
    : X86ReductionWordImpl<ReduceAddOp> {};
template <>
struct NativeWordImpl<X86Backend, ReduceMaxOp>
    : X86ReductionWordImpl<ReduceMaxOp> {};
template <>
struct NativeWordImpl<X86Backend, ReduceMinOp>
    : X86ReductionWordImpl<ReduceMinOp> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_REDUCTION_H
