#ifndef VECOPS_NVEC_DETAILS_X86_ARITHMETIC_H
#define VECOPS_NVEC_DETAILS_X86_ARITHMETIC_H

#include <cstring>

#include "vecops/nvec/details/Dispatch.h"

namespace vecops::nvec::details {

template <Element T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_emulated_small_float_add(Raw a, Raw b) {
  static_assert(
      std::same_as<T, float16_t> || std::same_as<T, bfloat16_t>);
  constexpr std::size_t lanes = sizeof(Raw) / sizeof(T);
  alignas(64) T a_lanes[lanes];
  alignas(64) T b_lanes[lanes];
  alignas(64) T result_lanes[lanes];
  std::memcpy(a_lanes, &a, sizeof(Raw));
  std::memcpy(b_lanes, &b, sizeof(Raw));
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    result_lanes[lane] = a_lanes[lane] + b_lanes[lane];
  }
  Raw result;
  std::memcpy(&result, result_lanes, sizeof(Raw));
  return result;
}

template <>
struct NativeWordImpl<X86Backend, AddOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      AddOp, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = decltype(a.value);
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto result = [&]() {
      if constexpr (std::same_as<T, bfloat16_t>) {
        return x86_emulated_small_float_add<T>(a.value, b.value);
      } else if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
        if constexpr (sizeof(Raw) == 16) {
          return _mm_castph_si128(_mm_add_ph(
              _mm_castsi128_ph(a.value), _mm_castsi128_ph(b.value)));
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          return _mm256_castph_si256(_mm256_add_ph(
              _mm256_castsi256_ph(a.value),
              _mm256_castsi256_ph(b.value)));
        }
#endif
#if VEC_WIDTH >= 512
        else {
          return _mm512_castph_si512(_mm512_add_ph(
              _mm512_castsi512_ph(a.value),
              _mm512_castsi512_ph(b.value)));
        }
#endif
#else
        return x86_emulated_small_float_add<T>(a.value, b.value);
#endif
      } else if constexpr (std::same_as<T, float32_t>) {
        if constexpr (sizeof(Raw) == 16) return _mm_add_ps(a.value, b.value);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32)
          return _mm256_add_ps(a.value, b.value);
#endif
#if VEC_WIDTH >= 512
        else return _mm512_add_ps(a.value, b.value);
#endif
      } else if constexpr (std::same_as<T, float64_t>) {
        if constexpr (sizeof(Raw) == 16) return _mm_add_pd(a.value, b.value);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32)
          return _mm256_add_pd(a.value, b.value);
#endif
#if VEC_WIDTH >= 512
        else return _mm512_add_pd(a.value, b.value);
#endif
      } else if constexpr (
          std::same_as<T, int8_t> || std::same_as<T, uint8_t>) {
        if constexpr (sizeof(Raw) == 16)
          return _mm_add_epi8(a.value, b.value);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32)
          return _mm256_add_epi8(a.value, b.value);
#endif
#if VEC_WIDTH >= 512
        else return _mm512_add_epi8(a.value, b.value);
#endif
      } else if constexpr (
          std::same_as<T, int16_t> || std::same_as<T, uint16_t>) {
        if constexpr (sizeof(Raw) == 16)
          return _mm_add_epi16(a.value, b.value);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32)
          return _mm256_add_epi16(a.value, b.value);
#endif
#if VEC_WIDTH >= 512
        else return _mm512_add_epi16(a.value, b.value);
#endif
      } else if constexpr (
          std::same_as<T, int32_t> || std::same_as<T, uint32_t>) {
        if constexpr (sizeof(Raw) == 16)
          return _mm_add_epi32(a.value, b.value);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32)
          return _mm256_add_epi32(a.value, b.value);
#endif
#if VEC_WIDTH >= 512
        else return _mm512_add_epi32(a.value, b.value);
#endif
      } else if constexpr (
          std::same_as<T, int64_t> || std::same_as<T, uint64_t>) {
        if constexpr (sizeof(Raw) == 16)
          return _mm_add_epi64(a.value, b.value);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32)
          return _mm256_add_epi64(a.value, b.value);
#endif
#if VEC_WIDTH >= 512
        else return _mm512_add_epi64(a.value, b.value);
#endif
      } else {
        static_assert(
            dispatch_dependent_false<T>,
            "x86 add has no implementation for this element type");
      }
    }();
    return NativeWordVec<Tag>{result};
  }
};

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_X86_ARITHMETIC_H
