#ifndef VECOPS_VEC_DETAILS_X86_SEQUENCE_H
#define VECOPS_VEC_DETAILS_X86_SEQUENCE_H

/**
 * @file Sequence.h
 * @brief x86 lane sequences using a constant lane-index vector followed by
 * native vector multiply and add.
 */

#include <array>
#include <cstring>

#include "vecops/vec/details/x86/Arithmetic.h"

namespace vecops::vec::details {

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_iota_f32_madd(Raw a, Raw b, Raw c) {
#if defined(HAS_FMA)
  if constexpr (sizeof(Raw) == 16) return _mm_fmadd_ps(a, b, c);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) return _mm256_fmadd_ps(a, b, c);
#endif
#if VEC_WIDTH >= 512
  else return _mm512_fmadd_ps(a, b, c);
#endif
#else
  if constexpr (sizeof(Raw) == 16) return _mm_add_ps(_mm_mul_ps(a, b), c);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
#if VEC_WIDTH >= 512
  else return _mm512_add_ps(_mm512_mul_ps(a, b), c);
#endif
#endif
}

template <Element T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_iota_small_float_madd(
    Raw index, Raw step, Raw start) {
  using F32Raw = std::conditional_t<
      sizeof(Raw) == 16, __m128,
      std::conditional_t<sizeof(Raw) == 32, __m256, __m512>>;
  F32Raw index_low;
  F32Raw index_high;
  F32Raw step_low;
  F32Raw step_high;
  F32Raw start_low;
  F32Raw start_high;
  if constexpr (std::same_as<T, bfloat16_t>) {
    x86_bfloat16_to_float32_pair(index, index_low, index_high);
    x86_bfloat16_to_float32_pair(step, step_low, step_high);
    x86_bfloat16_to_float32_pair(start, start_low, start_high);
    return x86_float32_pair_to_bfloat16(
        x86_iota_f32_madd(index_low, step_low, start_low),
        x86_iota_f32_madd(index_high, step_high, start_high));
  } else {
    x86_float16_to_float32_pair(index, index_low, index_high);
    x86_float16_to_float32_pair(step, step_low, step_high);
    x86_float16_to_float32_pair(start, start_low, start_high);
    return x86_float32_pair_to_float16(
        x86_iota_f32_madd(index_low, step_low, start_low),
        x86_iota_f32_madd(index_high, step_high, start_high));
  }
}

template <Element T>
VECOPS_ALWAYS_INLINE constexpr T x86_iota_index_value(nint_t index) {
  if constexpr (std::same_as<T, float16_t>)
    return T(static_cast<_Float16>(index));
  else
    return static_cast<T>(index);
}

template <>
struct NativeWordImpl<X86Backend, IotaOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      IotaOp, Tag tag, ElementOf<Tag> start) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = typename Traits::RawVec;
    constexpr std::size_t lanes = sizeof(Raw) / sizeof(T);
    static constexpr auto indices = [] {
      std::array<T, lanes> values{};
      for (std::size_t lane = 0; lane < lanes; ++lane) {
        values[lane] = x86_iota_index_value<T>(
            Index * static_cast<nint_t>(lanes) +
            static_cast<nint_t>(lane));
      }
      return values;
    }();
    NativeWordVec<Tag> index_word{};
    std::memcpy(&index_word.value, indices.data(), sizeof(Raw));
    return NativeWordImpl<X86Backend, AddOp>::template call<Index>(
        AddOp{}, tag,
        NativeWordImpl<X86Backend, FillOp>::template call<Index>(
            FillOp{}, tag, start),
        index_word);
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      IotaOp, Tag tag, ElementOf<Tag> start, ElementOf<Tag> step) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = typename Traits::RawVec;
    constexpr std::size_t lanes = sizeof(Raw) / sizeof(T);
    static constexpr auto indices = [] {
      std::array<T, lanes> values{};
      for (std::size_t lane = 0; lane < lanes; ++lane) {
        values[lane] = x86_iota_index_value<T>(
            Index * static_cast<nint_t>(lanes) +
            static_cast<nint_t>(lane));
      }
      return values;
    }();
    NativeWordVec<Tag> index_word{};
    std::memcpy(&index_word.value, indices.data(), sizeof(Raw));
    if constexpr (
        std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>) {
      const auto step_word =
          NativeWordImpl<X86Backend, FillOp>::template call<Index>(
              FillOp{}, tag, step);
      const auto start_word =
          NativeWordImpl<X86Backend, FillOp>::template call<Index>(
              FillOp{}, tag, start);
      return NativeWordVec<Tag>{x86_iota_small_float_madd<T>(
          index_word.value, step_word.value, start_word.value)};
    }
    if constexpr (::vecops::is_float_v<T>) {
      return NativeWordImpl<X86Backend, FmaddOp>::template call<Index>(
          FmaddOp{}, tag, index_word,
          NativeWordImpl<X86Backend, FillOp>::template call<Index>(
              FillOp{}, tag, step),
          NativeWordImpl<X86Backend, FillOp>::template call<Index>(
              FillOp{}, tag, start));
    }
    const auto product = NativeWordImpl<X86Backend, MulOp>::template call<Index>(
        MulOp{}, tag, index_word,
        NativeWordImpl<X86Backend, FillOp>::template call<Index>(
            FillOp{}, tag, step));
    return NativeWordImpl<X86Backend, AddOp>::template call<Index>(
        AddOp{}, tag,
        NativeWordImpl<X86Backend, FillOp>::template call<Index>(
            FillOp{}, tag, start),
        product);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_SEQUENCE_H
