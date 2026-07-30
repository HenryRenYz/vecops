#ifndef VECOPS_VEC_DETAILS_X86_REDUCTION_H
#define VECOPS_VEC_DETAILS_X86_REDUCTION_H

/**
 * @file Reduction.h
 * @brief x86 backend implementations for reduction operations.
 */

#include <cstddef>
#include <cstring>

#include "vecops/vec/details/x86/Arithmetic.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                       Reduction word implementations                       //
/* **************************************************************************** */

template <typename Op, nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE ElementOf<Tag> x86_scalar_reduction(
    Tag tag, NativeWordVec<Tag> value, const NativeWordMask<Tag>* mask) {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  using T = ElementOf<Tag>;
  constexpr std::size_t lanes = sizeof(typename Traits::RawVec) / sizeof(T);
  alignas(64) T values[lanes];
  std::memcpy(values, &value.value, sizeof(value.value));
  T result = reduction_identity<Op, T>();
  const nint_t valid = x86_valid_word_lanes<Index, Tag>();
  for (nint_t lane = 0; lane < valid; ++lane) {
    if (mask == nullptr || execute_word<Index, X86Backend>(
                               GetMaskLaneOp{}, tag, *mask, lane)) {
      const T lane_value = values[static_cast<std::size_t>(lane)];
      if constexpr (std::same_as<Op, ReduceAddOp>) {
        result = scalar_arithmetic_binary<T>(
            result, lane_value, [](auto a, auto b) { return a + b; });
      } else if constexpr (std::same_as<Op, ReduceMaxOp>) {
        if (lane_value > result) result = lane_value;
      } else if constexpr (std::same_as<Op, ReduceMinOp>) {
        if (lane_value < result) result = lane_value;
      }
    }
  }
  return result;
}

template <typename Op>
struct X86ReductionWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = typename Traits::RawVec;
    constexpr nint_t valid = x86_valid_word_lanes<Index, Tag>();
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
    return x86_scalar_reduction<Op, Index>(tag, value, nullptr);
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
          x86_valid_word_lanes<Index, Tag>());
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
    return x86_scalar_reduction<Op, Index>(tag, value, &mask);
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
