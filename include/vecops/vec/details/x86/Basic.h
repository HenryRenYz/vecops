#ifndef VECOPS_VEC_DETAILS_X86_BASIC_H
#define VECOPS_VEC_DETAILS_X86_BASIC_H

/**
 * @file Basic.h
 * @brief x86 backend implementations for fill, mask, blend, lane access,
 * interleave, and shuffle operations using SSE/AVX/AVX2/AVX-512 intrinsics.
 *
 * Mask operations on AVX-512 use __mmask* types directly; on pre-AVX-512
 * targets, masks are stored in regular vector or GPR registers. Lane
 * shuffle uses either vperm* (whole-register) or vpshufb (local 16-byte)
 * depending on the operation.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {

#if !defined(CPU_CAPABILITY_AVX512)
template <nint_t ElementBytes, typename Raw>

/* **************************************************************************** */
//                   Mask helpers and word implementations                    //
/* **************************************************************************** */

VECOPS_ALWAYS_INLINE Raw x86_vector_mask_prefix(nint_t count) {
  constexpr nint_t lanes = static_cast<nint_t>(sizeof(Raw)) / ElementBytes;
  count = std::clamp<nint_t>(count, 0, lanes);
  if constexpr (sizeof(Raw) == 16) {
    if constexpr (ElementBytes == 1) {
      const auto index = _mm_setr_epi8(
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
      return _mm_cmpgt_epi8(_mm_set1_epi8(static_cast<int8_t>(count)), index);
    } else if constexpr (ElementBytes == 2) {
      const auto index = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
      return _mm_cmpgt_epi16(_mm_set1_epi16(static_cast<int16_t>(count)), index);
    } else if constexpr (ElementBytes == 4) {
      const auto index = _mm_setr_epi32(0, 1, 2, 3);
      return _mm_cmpgt_epi32(_mm_set1_epi32(static_cast<int32_t>(count)), index);
    } else if constexpr (ElementBytes == 8) {
      // Duplicated 32-bit indices avoid requiring a 64-bit compare on SSE.
      const auto index = _mm_setr_epi32(0, 0, 1, 1);
      return _mm_cmpgt_epi32(_mm_set1_epi32(static_cast<int32_t>(count)), index);
    } else {
      static_assert(dispatch_dependent_false<Raw>, "unsupported mask granularity");
    }
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    if constexpr (ElementBytes == 1) {
      const auto index = _mm256_setr_epi8(
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
      return _mm256_cmpgt_epi8(_mm256_set1_epi8(static_cast<int8_t>(count)), index);
    } else if constexpr (ElementBytes == 2) {
      const auto index = _mm256_setr_epi16(
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
      return _mm256_cmpgt_epi16(_mm256_set1_epi16(static_cast<int16_t>(count)), index);
    } else if constexpr (ElementBytes == 4) {
      const auto index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int32_t>(count)), index);
    } else if constexpr (ElementBytes == 8) {
      const auto index = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
      return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int32_t>(count)), index);
    } else {
      static_assert(dispatch_dependent_false<Raw>, "unsupported mask granularity");
    }
  }
#endif
  else {
    static_assert(dispatch_dependent_false<Raw>, "unsupported x86 mask width");
  }
}
#endif

template <Element T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_mask_prefix(nint_t count) {
  constexpr nint_t element_bytes = static_cast<nint_t>(sizeof(T));
  constexpr nint_t lanes =
#if defined(CPU_CAPABILITY_AVX512)
      static_cast<nint_t>(sizeof(Raw) * 8);
#else
      static_cast<nint_t>(sizeof(Raw)) / element_bytes;
#endif
  count = std::clamp<nint_t>(count, 0, lanes);
#if defined(CPU_CAPABILITY_AVX512)
  if (count == 0) return Raw{0};
  if (count == lanes) return static_cast<Raw>(~Raw{0});
  return static_cast<Raw>((uint64_t{1} << count) - 1);
#else
  return x86_vector_mask_prefix<element_bytes, Raw>(count);
#endif
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_mask_and(Raw a, Raw b) {
#if defined(CPU_CAPABILITY_AVX512)
  return static_cast<Raw>(a & b);
#else
  if constexpr (sizeof(Raw) == 16) return _mm_and_si128(a, b);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) return _mm256_and_si256(a, b);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 mask width");
#endif
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_mask_or(Raw a, Raw b) {
#if defined(CPU_CAPABILITY_AVX512)
  return static_cast<Raw>(a | b);
#else
  if constexpr (sizeof(Raw) == 16) return _mm_or_si128(a, b);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) return _mm256_or_si256(a, b);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 mask width");
#endif
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_mask_xor(Raw a, Raw b) {
#if defined(CPU_CAPABILITY_AVX512)
  return static_cast<Raw>(a ^ b);
#else
  if constexpr (sizeof(Raw) == 16) return _mm_xor_si128(a, b);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) return _mm256_xor_si256(a, b);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 mask width");
#endif
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_mask_andnot(Raw a, Raw b) {
#if defined(CPU_CAPABILITY_AVX512)
  return static_cast<Raw>((~a) & b);
#else
  if constexpr (sizeof(Raw) == 16) return _mm_andnot_si128(a, b);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) return _mm256_andnot_si256(a, b);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 mask width");
#endif
}

template <nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t x86_valid_word_lanes() {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  constexpr nint_t start = Index * Traits::word_lanes;
  constexpr nint_t remaining = Traits::logical_lanes - start;
  if constexpr (remaining <= 0) return 0;
  else if constexpr (remaining < Traits::word_lanes) return remaining;
  else return Traits::word_lanes;
}

template <>
struct NativeWordImpl<X86Backend, FillOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      FillOp, Tag, ElementOf<Tag> value) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = typename Traits::RawVec;
    static_assert(Index >= 0 && Index < Traits::word_count);

    const auto fill_i8 = []<typename R = Raw>(int8_t x) -> R {
      if constexpr (sizeof(R) == 16) return _mm_set1_epi8(x);
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(R) == 32) return _mm256_set1_epi8(x);
#endif
#if VEC_WIDTH >= 512
      else if constexpr (sizeof(R) == 64) return _mm512_set1_epi8(x);
#endif
      else static_assert(dispatch_dependent_false<R>, "unsupported x86 word width");
    };
    const auto fill_i16 = []<typename R = Raw>(int16_t x) -> R {
      if constexpr (sizeof(R) == 16) return _mm_set1_epi16(x);
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(R) == 32) return _mm256_set1_epi16(x);
#endif
#if VEC_WIDTH >= 512
      else if constexpr (sizeof(R) == 64) return _mm512_set1_epi16(x);
#endif
      else static_assert(dispatch_dependent_false<R>, "unsupported x86 word width");
    };
    const auto fill_i32 = []<typename R = Raw>(int32_t x) -> R {
      if constexpr (sizeof(R) == 16) return _mm_set1_epi32(x);
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(R) == 32) return _mm256_set1_epi32(x);
#endif
#if VEC_WIDTH >= 512
      else if constexpr (sizeof(R) == 64) return _mm512_set1_epi32(x);
#endif
      else static_assert(dispatch_dependent_false<R>, "unsupported x86 word width");
    };
    const auto fill_i64 = []<typename R = Raw>(int64_t x) -> R {
      if constexpr (sizeof(R) == 16) return _mm_set1_epi64x(x);
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(R) == 32) return _mm256_set1_epi64x(x);
#endif
#if VEC_WIDTH >= 512
      else if constexpr (sizeof(R) == 64) return _mm512_set1_epi64(x);
#endif
      else static_assert(dispatch_dependent_false<R>, "unsupported x86 word width");
    };

    if constexpr (std::same_as<T, bfloat16_t>) {
      return NativeWordVec<Tag>{fill_i16(std::bit_cast<int16_t>(value.to_bits()))};
    } else if constexpr (std::same_as<T, float16_t>) {
      return NativeWordVec<Tag>{fill_i16(std::bit_cast<int16_t>(value.to_bits()))};
    } else if constexpr (std::same_as<T, float32_t>) {
      if constexpr (sizeof(Raw) == 16) return NativeWordVec<Tag>{_mm_set1_ps(value)};
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32) return NativeWordVec<Tag>{_mm256_set1_ps(value)};
#endif
#if VEC_WIDTH >= 512
      else if constexpr (sizeof(Raw) == 64) return NativeWordVec<Tag>{_mm512_set1_ps(value)};
#endif
      else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
    } else if constexpr (std::same_as<T, float64_t>) {
      if constexpr (sizeof(Raw) == 16) return NativeWordVec<Tag>{_mm_set1_pd(value)};
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32) return NativeWordVec<Tag>{_mm256_set1_pd(value)};
#endif
#if VEC_WIDTH >= 512
      else if constexpr (sizeof(Raw) == 64) return NativeWordVec<Tag>{_mm512_set1_pd(value)};
#endif
      else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
    } else if constexpr (std::same_as<T, int8_t>) {
      return NativeWordVec<Tag>{fill_i8(value)};
    } else if constexpr (std::same_as<T, uint8_t>) {
      return NativeWordVec<Tag>{fill_i8(std::bit_cast<int8_t>(value))};
    } else if constexpr (std::same_as<T, int16_t>) {
      return NativeWordVec<Tag>{fill_i16(value)};
    } else if constexpr (std::same_as<T, uint16_t>) {
      return NativeWordVec<Tag>{fill_i16(std::bit_cast<int16_t>(value))};
    } else if constexpr (std::same_as<T, int32_t>) {
      return NativeWordVec<Tag>{fill_i32(value)};
    } else if constexpr (std::same_as<T, uint32_t>) {
      return NativeWordVec<Tag>{fill_i32(std::bit_cast<int32_t>(value))};
    } else if constexpr (std::same_as<T, int64_t>) {
      return NativeWordVec<Tag>{fill_i64(value)};
    } else if constexpr (std::same_as<T, uint64_t>) {
      return NativeWordVec<Tag>{fill_i64(std::bit_cast<int64_t>(value))};
    } else {
      static_assert(
          dispatch_dependent_false<T>,
          "x86 fill has no implementation for this element type");
    }
  }
};

template <>
struct NativeWordImpl<X86Backend, MaskFillOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      MaskFillOp, Tag, bool value) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    using Raw = typename Traits::RawMask;
    const nint_t valid = x86_valid_word_lanes<Index, Tag>();
    return NativeWordMask<Tag>{
        value ? x86_mask_prefix<ElementOf<Tag>, Raw>(valid) : Raw{}};
  }
};

template <>
struct NativeWordImpl<X86Backend, MaskWhileLtOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      MaskWhileLtOp, Tag, nint_t a, nint_t b) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    using Raw = typename Traits::RawMask;
    const nint_t base = Index * Traits::word_lanes;
    const nint_t count = std::clamp<nint_t>(
        b - a - base, 0, x86_valid_word_lanes<Index, Tag>());
    return NativeWordMask<Tag>{x86_mask_prefix<ElementOf<Tag>, Raw>(count)};
  }
};

template <>
struct NativeWordImpl<X86Backend, MaskWhileGeOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      MaskWhileGeOp, Tag, nint_t a, nint_t b) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    using Raw = typename Traits::RawMask;
    const nint_t base = Index * Traits::word_lanes;
    const nint_t valid = x86_valid_word_lanes<Index, Tag>();
    const nint_t first = std::clamp<nint_t>(b - a - base, 0, valid);
    const Raw active = x86_mask_prefix<ElementOf<Tag>, Raw>(valid);
    const Raw before = x86_mask_prefix<ElementOf<Tag>, Raw>(first);
    return NativeWordMask<Tag>{x86_mask_andnot(before, active)};
  }
};

template <>
struct NativeWordImpl<X86Backend, BlendOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BlendOp, Tag, NativeWordVec<Tag> v0,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> v1) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = typename Traits::RawVec;
    static_assert(Index >= 0 && Index < Traits::word_count);

#if defined(CPU_CAPABILITY_AVX512)
#define VECOPS_VEC_X86_RETURN_BLEND(Suffix)                               \
    do {                                                                   \
      if constexpr (sizeof(Raw) == 16)                                    \
        return NativeWordVec<Tag>{_mm_mask_blend_##Suffix(mask.value, v0.value, v1.value)}; \
      else if constexpr (sizeof(Raw) == 32)                               \
        return NativeWordVec<Tag>{_mm256_mask_blend_##Suffix(mask.value, v0.value, v1.value)}; \
      else                                                                 \
        return NativeWordVec<Tag>{_mm512_mask_blend_##Suffix(mask.value, v0.value, v1.value)}; \
    } while (false)
    if constexpr (std::same_as<T, bfloat16_t>) VECOPS_VEC_X86_RETURN_BLEND(epi16);
    else if constexpr (std::same_as<T, float16_t>) VECOPS_VEC_X86_RETURN_BLEND(epi16);
    else if constexpr (std::same_as<T, float32_t>) VECOPS_VEC_X86_RETURN_BLEND(ps);
    else if constexpr (std::same_as<T, float64_t>) VECOPS_VEC_X86_RETURN_BLEND(pd);
    else if constexpr (std::same_as<T, int8_t>) VECOPS_VEC_X86_RETURN_BLEND(epi8);
    else if constexpr (std::same_as<T, uint8_t>) VECOPS_VEC_X86_RETURN_BLEND(epi8);
    else if constexpr (std::same_as<T, int16_t>) VECOPS_VEC_X86_RETURN_BLEND(epi16);
    else if constexpr (std::same_as<T, uint16_t>) VECOPS_VEC_X86_RETURN_BLEND(epi16);
    else if constexpr (std::same_as<T, int32_t>) VECOPS_VEC_X86_RETURN_BLEND(epi32);
    else if constexpr (std::same_as<T, uint32_t>) VECOPS_VEC_X86_RETURN_BLEND(epi32);
    else if constexpr (std::same_as<T, int64_t>) VECOPS_VEC_X86_RETURN_BLEND(epi64);
    else if constexpr (std::same_as<T, uint64_t>) VECOPS_VEC_X86_RETURN_BLEND(epi64);
    else static_assert(dispatch_dependent_false<T>, "unsupported x86 blend element type");
#undef VECOPS_VEC_X86_RETURN_BLEND
#else
    const auto blend_integer = []<typename R, typename M>(R a, M m, R b) -> R {
      if constexpr (sizeof(R) == 16) {
        return _mm_blendv_epi8(a, b, m);
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(R) == 32) {
        return _mm256_blendv_epi8(a, b, m);
      }
#endif
      else static_assert(dispatch_dependent_false<R>, "unsupported x86 word width");
    };
    if constexpr (std::same_as<T, bfloat16_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else if constexpr (std::same_as<T, float16_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else if constexpr (std::same_as<T, float32_t>) {
      if constexpr (sizeof(Raw) == 16)
        return NativeWordVec<Tag>{_mm_blendv_ps(v0.value, v1.value, _mm_castsi128_ps(mask.value))};
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32)
        return NativeWordVec<Tag>{_mm256_blendv_ps(v0.value, v1.value, _mm256_castsi256_ps(mask.value))};
#endif
      else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
    } else if constexpr (std::same_as<T, float64_t>) {
      if constexpr (sizeof(Raw) == 16)
        return NativeWordVec<Tag>{_mm_blendv_pd(v0.value, v1.value, _mm_castsi128_pd(mask.value))};
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32)
        return NativeWordVec<Tag>{_mm256_blendv_pd(v0.value, v1.value, _mm256_castsi256_pd(mask.value))};
#endif
      else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
    } else if constexpr (std::same_as<T, int8_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else if constexpr (std::same_as<T, uint8_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else if constexpr (std::same_as<T, int16_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else if constexpr (std::same_as<T, uint16_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else if constexpr (std::same_as<T, int32_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else if constexpr (std::same_as<T, uint32_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else if constexpr (std::same_as<T, int64_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else if constexpr (std::same_as<T, uint64_t>) {
      return NativeWordVec<Tag>{blend_integer(v0.value, mask.value, v1.value)};
    } else {
      static_assert(dispatch_dependent_false<T>, "unsupported x86 blend element type");
    }
#endif
  }
};

#define VECOPS_VEC_X86_MASK_BINARY_IMPL(Op, Helper)                       \
  template <>                                                              \
  struct NativeWordImpl<X86Backend, Op> {                                  \
    template <nint_t Index, VectorTag Tag>                                 \
    static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(                  \
        Op, Tag, NativeWordMask<Tag> a, NativeWordMask<Tag> b) {           \
      using Traits = RepresentationTraits<X86Backend, Tag>;                \
      static_assert(Index >= 0 && Index < Traits::word_count);             \
      return NativeWordMask<Tag>{Helper(a.value, b.value)};                \
    }                                                                      \
  }

VECOPS_VEC_X86_MASK_BINARY_IMPL(MaskAndOp, x86_mask_and);
VECOPS_VEC_X86_MASK_BINARY_IMPL(MaskOrOp, x86_mask_or);
VECOPS_VEC_X86_MASK_BINARY_IMPL(MaskXorOp, x86_mask_xor);
VECOPS_VEC_X86_MASK_BINARY_IMPL(MaskAndNotOp, x86_mask_andnot);

#undef VECOPS_VEC_X86_MASK_BINARY_IMPL

template <>
struct NativeWordImpl<X86Backend, MaskNotOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      MaskNotOp, Tag, NativeWordMask<Tag> value) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto active = x86_mask_prefix<ElementOf<Tag>, typename Traits::RawMask>(
        x86_valid_word_lanes<Index, Tag>());
#if defined(CPU_CAPABILITY_AVX512)
    const auto inverted = static_cast<typename Traits::RawMask>(~value.value);
#else
    const auto inverted = [&] {
      if constexpr (sizeof(typename Traits::RawMask) == 16)
        return _mm_xor_si128(value.value, _mm_set1_epi32(-1));
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(typename Traits::RawMask) == 32)
        return _mm256_xor_si256(value.value, _mm256_set1_epi32(-1));
#endif
      else static_assert(
          dispatch_dependent_false<Tag>, "unsupported x86 mask width");
    }();
#endif
    return NativeWordMask<Tag>{x86_mask_and(inverted, active)};
  }
};

template <>
struct NativeWordImpl<X86Backend, GetVecLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      GetVecLaneOp, Tag, NativeWordVec<Tag> value, nint_t lane) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    constexpr std::size_t lanes = sizeof(typename Traits::RawVec) / sizeof(T);
    alignas(64) T values[lanes];
    std::memcpy(values, &value.value, sizeof(value.value));
    return values[static_cast<std::size_t>(lane)];
  }
};

template <>
struct NativeWordImpl<X86Backend, SetVecLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      SetVecLaneOp, Tag, NativeWordVec<Tag> value,
      nint_t lane, ElementOf<Tag> replacement) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    constexpr std::size_t lanes = sizeof(typename Traits::RawVec) / sizeof(T);
    alignas(64) T values[lanes];
    std::memcpy(values, &value.value, sizeof(value.value));
    values[static_cast<std::size_t>(lane)] = replacement;
    std::memcpy(&value.value, values, sizeof(value.value));
    return value;
  }
};

template <>
struct NativeWordImpl<X86Backend, GetMaskLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE bool call(
      GetMaskLaneOp, Tag, NativeWordMask<Tag> value, nint_t lane) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
#if defined(CPU_CAPABILITY_AVX512)
    return ((static_cast<uint64_t>(value.value) >> lane) & 1U) != 0;
#else
    alignas(64) std::byte bytes[sizeof(typename Traits::RawMask)];
    std::memcpy(bytes, &value.value, sizeof(value.value));
    return bytes[static_cast<std::size_t>(lane * sizeof(ElementOf<Tag>))]
        != std::byte{};
#endif
  }
};

template <>
struct NativeWordImpl<X86Backend, SetMaskLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      SetMaskLaneOp, Tag, NativeWordMask<Tag> value,
      nint_t lane, bool replacement) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    using Raw = typename Traits::RawMask;
    const Raw below = x86_mask_prefix<ElementOf<Tag>, Raw>(lane);
    const Raw through = x86_mask_prefix<ElementOf<Tag>, Raw>(lane + 1);
    const Raw bit = x86_mask_xor(below, through);
    const Raw cleared = x86_mask_andnot(bit, value.value);
    return NativeWordMask<Tag>{replacement ? x86_mask_or(cleared, bit) : cleared};
  }
};

template <>
struct NativeWordImpl<X86Backend, BitCastOp> {
  template <nint_t Index, VectorTag ToTag,
            Element From, nint_t FromLanes, typename FromRaw>
  static VECOPS_ALWAYS_INLINE NativeWordVec<ToTag> call(
      BitCastOp, ToTag, X86Vector<From, FromLanes, FromRaw> value) {
    using ToTraits = RepresentationTraits<X86Backend, ToTag>;
    static_assert(Index >= 0 && Index < ToTraits::word_count);
    static_assert(
        sizeof(FromRaw) == sizeof(typename ToTraits::RawVec),
        "word bitcast requires equal physical word widths");
    const auto bits = [&] {
      if constexpr (std::same_as<From, bfloat16_t>) return value.value;
      else if constexpr (std::same_as<From, float16_t>) return value.value;
      else if constexpr (std::same_as<From, float32_t>) {
        if constexpr (sizeof(FromRaw) == 16) return _mm_castps_si128(value.value);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(FromRaw) == 32) return _mm256_castps_si256(value.value);
#endif
#if VEC_WIDTH >= 512
        else return _mm512_castps_si512(value.value);
#endif
      } else if constexpr (std::same_as<From, float64_t>) {
        if constexpr (sizeof(FromRaw) == 16) return _mm_castpd_si128(value.value);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(FromRaw) == 32) return _mm256_castpd_si256(value.value);
#endif
#if VEC_WIDTH >= 512
        else return _mm512_castpd_si512(value.value);
#endif
      } else if constexpr (std::same_as<From, int8_t>) return value.value;
      else if constexpr (std::same_as<From, uint8_t>) return value.value;
      else if constexpr (std::same_as<From, int16_t>) return value.value;
      else if constexpr (std::same_as<From, uint16_t>) return value.value;
      else if constexpr (std::same_as<From, int32_t>) return value.value;
      else if constexpr (std::same_as<From, uint32_t>) return value.value;
      else if constexpr (std::same_as<From, int64_t>) return value.value;
      else if constexpr (std::same_as<From, uint64_t>) return value.value;
      else static_assert(
          dispatch_dependent_false<From>,
          "x86 bitcast has no implementation for this source element type");
    }();

    using To = ElementOf<ToTag>;
    if constexpr (std::same_as<To, bfloat16_t>) {
      return NativeWordVec<ToTag>{bits};
    } else if constexpr (std::same_as<To, float16_t>) {
      return NativeWordVec<ToTag>{bits};
    } else if constexpr (std::same_as<To, float32_t>) {
      if constexpr (sizeof(bits) == 16)
        return NativeWordVec<ToTag>{_mm_castsi128_ps(bits)};
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(bits) == 32)
        return NativeWordVec<ToTag>{_mm256_castsi256_ps(bits)};
#endif
#if VEC_WIDTH >= 512
      else
        return NativeWordVec<ToTag>{_mm512_castsi512_ps(bits)};
#endif
    } else if constexpr (std::same_as<To, float64_t>) {
      if constexpr (sizeof(bits) == 16)
        return NativeWordVec<ToTag>{_mm_castsi128_pd(bits)};
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(bits) == 32)
        return NativeWordVec<ToTag>{_mm256_castsi256_pd(bits)};
#endif
#if VEC_WIDTH >= 512
      else
        return NativeWordVec<ToTag>{_mm512_castsi512_pd(bits)};
#endif
    } else if constexpr (std::same_as<To, int8_t>) {
      return NativeWordVec<ToTag>{bits};
    } else if constexpr (std::same_as<To, uint8_t>) {
      return NativeWordVec<ToTag>{bits};
    } else if constexpr (std::same_as<To, int16_t>) {
      return NativeWordVec<ToTag>{bits};
    } else if constexpr (std::same_as<To, uint16_t>) {
      return NativeWordVec<ToTag>{bits};
    } else if constexpr (std::same_as<To, int32_t>) {
      return NativeWordVec<ToTag>{bits};
    } else if constexpr (std::same_as<To, uint32_t>) {
      return NativeWordVec<ToTag>{bits};
    } else if constexpr (std::same_as<To, int64_t>) {
      return NativeWordVec<ToTag>{bits};
    } else if constexpr (std::same_as<To, uint64_t>) {
      return NativeWordVec<ToTag>{bits};
    } else {
      static_assert(
          dispatch_dependent_false<To>,
          "x86 bitcast has no implementation for this target element type");
    }
  }

  template <nint_t Index, VectorTag ToTag, VectorTag FromTag,
            Element From, nint_t FromLanes, typename FromRaw>
  static VECOPS_ALWAYS_INLINE NativeWordVec<ToTag> call(
      BitCastOp op, ToTag to, FromTag,
      X86Vector<From, FromLanes, FromRaw> value) {
    static_assert(std::same_as<From, ElementOf<FromTag>>);
    return call<Index>(op, to, value);
  }
};

/* **************************************************************************** */
//               Mask logical lane access and half operations                //
/* **************************************************************************** */

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE bool x86_get_mask_logical_lane(
    Tag tag, Mask<Tag> value, nint_t lane) {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  const nint_t word = lane / Traits::word_lanes;
  const nint_t offset = lane % Traits::word_lanes;
  if constexpr (Traits::word_count == 1) {
    return NativeWordImpl<X86Backend, GetMaskLaneOp>::template call<0>(
        GetMaskLaneOp{}, tag, value, offset);
  } else {
    static_assert(Traits::word_count == 2, "rearrange base case accepts at most two words");
    return word == 0
        ? NativeWordImpl<X86Backend, GetMaskLaneOp>::template call<0>(
              GetMaskLaneOp{}, tag, value.words[0], offset)
        : NativeWordImpl<X86Backend, GetMaskLaneOp>::template call<1>(
              GetMaskLaneOp{}, tag, value.words[1], offset);
  }
}

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Tag> x86_set_mask_logical_lane(
    Tag tag, Mask<Tag> value, nint_t lane, bool replacement) {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  const nint_t word = lane / Traits::word_lanes;
  const nint_t offset = lane % Traits::word_lanes;
  if constexpr (Traits::word_count == 1) {
    return NativeWordImpl<X86Backend, SetMaskLaneOp>::template call<0>(
        SetMaskLaneOp{}, tag, value, offset, replacement);
  } else {
    static_assert(Traits::word_count == 2, "rearrange base case accepts at most two words");
    if (word == 0) {
      value.words[0] = NativeWordImpl<X86Backend, SetMaskLaneOp>::template call<0>(
          SetMaskLaneOp{}, tag, value.words[0], offset, replacement);
    } else {
      value.words[1] = NativeWordImpl<X86Backend, SetMaskLaneOp>::template call<1>(
          SetMaskLaneOp{}, tag, value.words[1], offset, replacement);
    }
    return value;
  }
}

template <bool Upper, VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Half<Tag>> x86_extract_mask_half(
    Tag tag, Mask<Tag> value) {
  using OutTag = Half<Tag>;
  Mask<OutTag> result{};
  constexpr nint_t half = RepresentationTraits<X86Backend, OutTag>::logical_lanes;
  for (nint_t lane = 0; lane < half; ++lane) {
    result = x86_set_mask_logical_lane(
        OutTag{}, result, lane,
        x86_get_mask_logical_lane(tag, value, lane + (Upper ? half : 0)));
  }
  return result;
}

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Tag> x86_concat_mask_halves(
    Tag tag, Mask<Half<Tag>> lower, Mask<Half<Tag>> upper) {
  using HalfTag = Half<Tag>;
  Mask<Tag> result{};
  constexpr nint_t half = RepresentationTraits<X86Backend, HalfTag>::logical_lanes;
  for (nint_t lane = 0; lane < half; ++lane) {
    result = x86_set_mask_logical_lane(
        tag, result, lane,
        x86_get_mask_logical_lane(HalfTag{}, lower, lane));
    result = x86_set_mask_logical_lane(
        tag, result, half + lane,
        x86_get_mask_logical_lane(HalfTag{}, upper, lane));
  }
  return result;
}

#define VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH(Expression, Message)     \
  if constexpr (std::same_as<T, bfloat16_t>) return (Expression);         \
  else if constexpr (std::same_as<T, float16_t>) return (Expression);     \
  else if constexpr (std::same_as<T, float32_t>) return (Expression);     \
  else if constexpr (std::same_as<T, float64_t>) return (Expression);     \
  else if constexpr (std::same_as<T, int8_t>) return (Expression);        \
  else if constexpr (std::same_as<T, uint8_t>) return (Expression);       \
  else if constexpr (std::same_as<T, int16_t>) return (Expression);       \
  else if constexpr (std::same_as<T, uint16_t>) return (Expression);      \
  else if constexpr (std::same_as<T, int32_t>) return (Expression);       \
  else if constexpr (std::same_as<T, uint32_t>) return (Expression);      \
  else if constexpr (std::same_as<T, int64_t>) return (Expression);       \
  else if constexpr (std::same_as<T, uint64_t>) return (Expression);      \
  else static_assert(dispatch_dependent_false<T>, Message)

/* **************************************************************************** */
//                Lower, upper, and concatenation operations                 //
/* **************************************************************************** */

template <VectorTag Tag>
  requires (
      RepresentationTraits<X86Backend, Tag>::word_count == 1 ||
      RepresentationTraits<X86Backend, Half<Tag>>::word_count == 1)
struct NativeImpl<X86Backend, LowerOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(LowerOp, Tag, Vec<Tag> value) {
    using T = ElementOf<Tag>;
    using InRaw = typename RepresentationTraits<X86Backend, Tag>::RawVec;
    using OutRaw = typename RepresentationTraits<X86Backend, Half<Tag>>::RawVec;
    if constexpr (RepresentationTraits<X86Backend, Tag>::word_count == 2) {
      VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH(
          value.words[0], "unsupported x86 lower element type");
    } else if constexpr (sizeof(InRaw) == sizeof(OutRaw)) {
      VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH(
          Vec<Half<Tag>>{value.value}, "unsupported x86 lower element type");
    } else {
      if constexpr (std::same_as<T, bfloat16_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else if constexpr (std::same_as<T, float16_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else if constexpr (std::same_as<T, float32_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castps256_ps128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castps512_ps256(value.value)};
      } else if constexpr (std::same_as<T, float64_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castpd256_pd128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castpd512_pd256(value.value)};
      } else if constexpr (std::same_as<T, int8_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else if constexpr (std::same_as<T, uint8_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else if constexpr (std::same_as<T, int16_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else if constexpr (std::same_as<T, uint16_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else if constexpr (std::same_as<T, int32_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else if constexpr (std::same_as<T, uint32_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else if constexpr (std::same_as<T, int64_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else if constexpr (std::same_as<T, uint64_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_castsi256_si128(value.value)};
        else return Vec<Half<Tag>>{_mm512_castsi512_si256(value.value)};
      } else {
        static_assert(dispatch_dependent_false<T>, "unsupported x86 lower element type");
      }
    }
  }
  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(LowerOp, Tag tag, Mask<Tag> value) {
    if constexpr (RepresentationTraits<X86Backend, Tag>::word_count == 2) {
      return value.words[0];
    } else {
      return x86_extract_mask_half<false>(tag, value);
    }
  }
};

template <VectorTag Tag>
  requires (
      RepresentationTraits<X86Backend, Tag>::word_count == 1 ||
      RepresentationTraits<X86Backend, Half<Tag>>::word_count == 1)
struct NativeImpl<X86Backend, UpperOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(UpperOp, Tag, Vec<Tag> value) {
    using T = ElementOf<Tag>;
    using InTraits = RepresentationTraits<X86Backend, Tag>;
    using InRaw = typename InTraits::RawVec;
    using OutRaw = typename RepresentationTraits<X86Backend, Half<Tag>>::RawVec;
    if constexpr (RepresentationTraits<X86Backend, Tag>::word_count == 2) {
      VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH(
          value.words[1], "unsupported x86 upper element type");
    } else if constexpr (sizeof(InRaw) == sizeof(OutRaw)) {
      constexpr int logical_bytes = static_cast<int>(
          InTraits::logical_lanes * static_cast<nint_t>(sizeof(T)));
      constexpr int half_bytes = logical_bytes / 2;
      if constexpr (std::same_as<T, float32_t>) {
        if constexpr (logical_bytes == 8)
          return Vec<Half<Tag>>{_mm_shuffle_ps(
              value.value, value.value, _MM_SHUFFLE(0, 0, 0, 1))};
        else return Vec<Half<Tag>>{_mm_shuffle_ps(
            value.value, value.value, _MM_SHUFFLE(0, 0, 3, 2))};
      } else if constexpr (std::same_as<T, float64_t>) {
        return Vec<Half<Tag>>{_mm_shuffle_pd(
            value.value, value.value, _MM_SHUFFLE2(0, 1))};
      } else if constexpr (std::same_as<T, bfloat16_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else if constexpr (std::same_as<T, float16_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else if constexpr (std::same_as<T, int8_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else if constexpr (std::same_as<T, uint8_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else if constexpr (std::same_as<T, int16_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else if constexpr (std::same_as<T, uint16_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else if constexpr (std::same_as<T, int32_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else if constexpr (std::same_as<T, uint32_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else if constexpr (std::same_as<T, int64_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else if constexpr (std::same_as<T, uint64_t>) {
        if constexpr (logical_bytes == 16)
          return Vec<Half<Tag>>{_mm_shuffle_epi32(
              value.value, _MM_SHUFFLE(0, 0, 3, 2))};
        else return Vec<Half<Tag>>{_mm_srli_si128(value.value, half_bytes)};
      } else {
        static_assert(dispatch_dependent_false<T>, "unsupported x86 upper element type");
      }
    } else {
      if constexpr (std::same_as<T, bfloat16_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else if constexpr (std::same_as<T, float16_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else if constexpr (std::same_as<T, float32_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extractf128_ps(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extractf32x8_ps(value.value, 1)};
      } else if constexpr (std::same_as<T, float64_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extractf128_pd(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extractf64x4_pd(value.value, 1)};
      } else if constexpr (std::same_as<T, int8_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else if constexpr (std::same_as<T, uint8_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else if constexpr (std::same_as<T, int16_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else if constexpr (std::same_as<T, uint16_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else if constexpr (std::same_as<T, int32_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else if constexpr (std::same_as<T, uint32_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else if constexpr (std::same_as<T, int64_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else if constexpr (std::same_as<T, uint64_t>) {
        if constexpr (sizeof(InRaw) == 32)
          return Vec<Half<Tag>>{_mm256_extracti128_si256(value.value, 1)};
        else return Vec<Half<Tag>>{_mm512_extracti32x8_epi32(value.value, 1)};
      } else {
        static_assert(dispatch_dependent_false<T>, "unsupported x86 upper element type");
      }
    }
  }
  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(UpperOp, Tag tag, Mask<Tag> value) {
    if constexpr (RepresentationTraits<X86Backend, Tag>::word_count == 2) {
      return value.words[1];
    } else {
      return x86_extract_mask_half<true>(tag, value);
    }
  }
};

template <VectorTag Tag>
  requires (
      RepresentationTraits<X86Backend, Tag>::word_count == 1 ||
      RepresentationTraits<X86Backend, Half<Tag>>::word_count == 1)
struct NativeImpl<X86Backend, ConcatOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatOp, Tag tag, Vec<Half<Tag>> lower, Vec<Half<Tag>> upper) {
    using T = ElementOf<Tag>;
    using OutTraits = RepresentationTraits<X86Backend, Tag>;
    using OutRaw = typename OutTraits::RawVec;
    using InRaw = typename RepresentationTraits<X86Backend, Half<Tag>>::RawVec;
    if constexpr (RepresentationTraits<X86Backend, Tag>::word_count == 2) {
      VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH(
          ::vecops::vec::from_words(tag, lower, upper),
          "unsupported x86 concat element type");
    } else if constexpr (sizeof(OutRaw) == sizeof(InRaw)) {
      constexpr int logical_bytes = static_cast<int>(
          OutTraits::logical_lanes * static_cast<nint_t>(sizeof(T)));
      constexpr int half_bytes = logical_bytes / 2;
      const auto lower_bits = [&] {
        if constexpr (std::same_as<T, float32_t>)
          return _mm_castps_si128(lower.value);
        else if constexpr (std::same_as<T, float64_t>)
          return _mm_castpd_si128(lower.value);
        else return lower.value;
      }();
      const auto upper_bits = [&] {
        if constexpr (std::same_as<T, float32_t>)
          return _mm_castps_si128(upper.value);
        else if constexpr (std::same_as<T, float64_t>)
          return _mm_castpd_si128(upper.value);
        else return upper.value;
      }();
      const auto bits = [&] {
        if constexpr (half_bytes == 1)
          return _mm_unpacklo_epi8(lower_bits, upper_bits);
        else if constexpr (half_bytes == 2)
          return _mm_unpacklo_epi16(lower_bits, upper_bits);
        else if constexpr (half_bytes == 4)
          return _mm_unpacklo_epi32(lower_bits, upper_bits);
        else if constexpr (half_bytes == 8)
          return _mm_unpacklo_epi64(lower_bits, upper_bits);
        else {
          const auto kept_lower = _mm_srli_si128(
              _mm_slli_si128(lower_bits, 16 - half_bytes),
              16 - half_bytes);
          return _mm_or_si128(
              kept_lower, _mm_slli_si128(upper_bits, half_bytes));
        }
      }();
      if constexpr (std::same_as<T, bfloat16_t>) return Vec<Tag>{bits};
      else if constexpr (std::same_as<T, float16_t>) return Vec<Tag>{bits};
      else if constexpr (std::same_as<T, float32_t>)
        return Vec<Tag>{_mm_castsi128_ps(bits)};
      else if constexpr (std::same_as<T, float64_t>)
        return Vec<Tag>{_mm_castsi128_pd(bits)};
      else if constexpr (std::same_as<T, int8_t>) return Vec<Tag>{bits};
      else if constexpr (std::same_as<T, uint8_t>) return Vec<Tag>{bits};
      else if constexpr (std::same_as<T, int16_t>) return Vec<Tag>{bits};
      else if constexpr (std::same_as<T, uint16_t>) return Vec<Tag>{bits};
      else if constexpr (std::same_as<T, int32_t>) return Vec<Tag>{bits};
      else if constexpr (std::same_as<T, uint32_t>) return Vec<Tag>{bits};
      else if constexpr (std::same_as<T, int64_t>) return Vec<Tag>{bits};
      else if constexpr (std::same_as<T, uint64_t>) return Vec<Tag>{bits};
      else static_assert(dispatch_dependent_false<T>, "unsupported x86 concat element type");
    } else {
      const auto concat_integer = []<typename R>(R lower_raw, R upper_raw) {
        if constexpr (sizeof(OutRaw) == 32)
          return _mm256_inserti128_si256(
              _mm256_castsi128_si256(lower_raw), upper_raw, 1);
        else return _mm512_inserti64x4(
            _mm512_castsi256_si512(lower_raw), upper_raw, 1);
      };
      if constexpr (std::same_as<T, float32_t>) {
        if constexpr (sizeof(OutRaw) == 32)
          return Vec<Tag>{_mm256_insertf128_ps(
              _mm256_castps128_ps256(lower.value), upper.value, 1)};
        else return Vec<Tag>{_mm512_insertf32x8(
            _mm512_castps256_ps512(lower.value), upper.value, 1)};
      } else if constexpr (std::same_as<T, float64_t>) {
        if constexpr (sizeof(OutRaw) == 32)
          return Vec<Tag>{_mm256_insertf128_pd(
              _mm256_castpd128_pd256(lower.value), upper.value, 1)};
        else return Vec<Tag>{_mm512_insertf64x4(
            _mm512_castpd256_pd512(lower.value), upper.value, 1)};
      } else if constexpr (std::same_as<T, bfloat16_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else if constexpr (std::same_as<T, float16_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else if constexpr (std::same_as<T, int8_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else if constexpr (std::same_as<T, uint8_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else if constexpr (std::same_as<T, int16_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else if constexpr (std::same_as<T, uint16_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else if constexpr (std::same_as<T, int32_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else if constexpr (std::same_as<T, uint32_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else if constexpr (std::same_as<T, int64_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else if constexpr (std::same_as<T, uint64_t>) {
        return Vec<Tag>{concat_integer(lower.value, upper.value)};
      } else {
        static_assert(dispatch_dependent_false<T>, "unsupported x86 concat element type");
      }
    }
  }
  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      ConcatOp, Tag tag, Mask<Half<Tag>> lower, Mask<Half<Tag>> upper) {
    if constexpr (RepresentationTraits<X86Backend, Tag>::word_count == 2) {
      return ::vecops::vec::mask_from_words(tag, lower, upper);
    } else {
      return x86_concat_mask_halves(tag, lower, upper);
    }
  }
};

/* **************************************************************************** */
//                  Parity extraction and concatenation                      //
/* **************************************************************************** */

template <bool Odd, VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> x86_concat_parity(Vec<Tag> a, Vec<Tag> b) {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  using T = ElementOf<Tag>;
  using Raw = typename Traits::RawVec;
  constexpr int logical_bytes = static_cast<int>(
      Traits::logical_lanes * static_cast<nint_t>(sizeof(T)));
  constexpr int select32 = Odd ? _MM_SHUFFLE(3, 1, 3, 1)
                               : _MM_SHUFFLE(2, 0, 2, 0);
  constexpr int select16 = Odd ? _MM_SHUFFLE(3, 1, 3, 1)
                               : _MM_SHUFFLE(2, 0, 2, 0);

  const auto f32 = []<typename R>(R av, R bv) -> R {
    if constexpr (sizeof(R) == 16) {
      if constexpr (logical_bytes <= 8) {
        if constexpr (Odd) {
          return _mm_unpacklo_ps(
              _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(av), 4)),
              _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(bv), 4)));
        } else {
          return _mm_unpacklo_ps(av, bv);
        }
      } else {
        return _mm_shuffle_ps(av, bv, select32);
      }
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) {
      const auto mixed = _mm256_shuffle_ps(av, bv, select32);
      return _mm256_castpd_ps(_mm256_permute4x64_pd(
          _mm256_castps_pd(mixed), _MM_SHUFFLE(3, 1, 2, 0)));
    }
#endif
#if VEC_WIDTH >= 512
    else {
      const auto mixed = _mm512_shuffle_ps(av, bv, select32);
      const auto order = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
      return _mm512_castpd_ps(_mm512_permutexvar_pd(
          order, _mm512_castps_pd(mixed)));
    }
#endif
  };
  const auto f64 = []<typename R>(R av, R bv) -> R {
    if constexpr (sizeof(R) == 16) {
      return _mm_shuffle_pd(av, bv, Odd ? 3 : 0);
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) {
      const auto mixed = _mm256_shuffle_pd(av, bv, Odd ? 0xf : 0x0);
      return _mm256_permute4x64_pd(
          mixed, _MM_SHUFFLE(3, 1, 2, 0));
    }
#endif
#if VEC_WIDTH >= 512
    else {
      const auto mixed = _mm512_shuffle_pd(av, bv, Odd ? 0xff : 0x00);
      const auto order = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
      return _mm512_permutexvar_pd(order, mixed);
    }
#endif
  };
  const auto i16 = []<typename R>(R av, R bv) -> R {
    if constexpr (sizeof(R) == 16) {
      if constexpr (logical_bytes <= 4) {
        if constexpr (Odd)
          return _mm_unpacklo_epi16(
              _mm_srli_si128(av, 2), _mm_srli_si128(bv, 2));
        else return _mm_unpacklo_epi16(av, bv);
      } else {
        auto picked_a = _mm_shufflelo_epi16(av, select16);
        auto picked_b = _mm_shufflelo_epi16(bv, select16);
        if constexpr (logical_bytes <= 8) {
          return _mm_blend_epi16(picked_a, picked_b, 0xcc);
        } else {
          picked_a = _mm_shufflehi_epi16(picked_a, select16);
          picked_b = _mm_shufflehi_epi16(picked_b, select16);
          return _mm_castps_si128(_mm_shuffle_ps(
              _mm_castsi128_ps(picked_a), _mm_castsi128_ps(picked_b),
              select32));
        }
      }
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) {
      auto picked_a = _mm256_shufflelo_epi16(av, select16);
      auto picked_b = _mm256_shufflelo_epi16(bv, select16);
      picked_a = _mm256_shufflehi_epi16(picked_a, select16);
      picked_b = _mm256_shufflehi_epi16(picked_b, select16);
      const auto mixed = _mm256_castps_si256(_mm256_shuffle_ps(
          _mm256_castsi256_ps(picked_a), _mm256_castsi256_ps(picked_b),
          select32));
      return _mm256_permute4x64_epi64(
          mixed, _MM_SHUFFLE(3, 1, 2, 0));
    }
#endif
#if VEC_WIDTH >= 512
    else {
      auto picked_a = _mm512_shufflelo_epi16(av, select16);
      auto picked_b = _mm512_shufflelo_epi16(bv, select16);
      picked_a = _mm512_shufflehi_epi16(picked_a, select16);
      picked_b = _mm512_shufflehi_epi16(picked_b, select16);
      const auto mixed = _mm512_castps_si512(_mm512_shuffle_ps(
          _mm512_castsi512_ps(picked_a), _mm512_castsi512_ps(picked_b),
          select32));
      const auto order = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
      return _mm512_permutexvar_epi64(order, mixed);
    }
#endif
  };
  const auto i8 = []<typename R>(R av, R bv) -> R {
    if constexpr (sizeof(R) == 16) {
      if constexpr (logical_bytes <= 2) {
        if constexpr (Odd)
          return _mm_unpacklo_epi8(
              _mm_srli_si128(av, 1), _mm_srli_si128(bv, 1));
        else return _mm_unpacklo_epi8(av, bv);
      } else {
        const auto index = [&] {
          if constexpr (logical_bytes <= 4)
            return Odd
                ? _mm_set1_epi16(0x0301)
                : _mm_set1_epi16(0x0200);
          else if constexpr (logical_bytes <= 8)
            return Odd
                ? _mm_set1_epi32(0x07050301)
                : _mm_set1_epi32(0x06040200);
          else
            return Odd
                ? _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1,
                               15, 13, 11, 9, 7, 5, 3, 1)
                : _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0,
                               14, 12, 10, 8, 6, 4, 2, 0);
        }();
        const auto picked_a = _mm_shuffle_epi8(av, index);
        const auto picked_b = _mm_shuffle_epi8(bv, index);
        if constexpr (logical_bytes <= 4)
          return _mm_blend_epi16(picked_a, picked_b, 0xaa);
        else if constexpr (logical_bytes <= 8)
          return _mm_blend_epi16(picked_a, picked_b, 0xcc);
        else return _mm_blend_epi16(picked_a, picked_b, 0xf0);
      }
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) {
      const auto index = Odd
          ? _mm256_set_epi8(
                15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1,
                15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1)
          : _mm256_set_epi8(
                14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0,
                14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);
      const auto picked_a = _mm256_shuffle_epi8(av, index);
      const auto picked_b = _mm256_shuffle_epi8(bv, index);
      const auto mixed = _mm256_blend_epi32(picked_a, picked_b, 0xcc);
      return _mm256_permute4x64_epi64(
          mixed, _MM_SHUFFLE(3, 1, 2, 0));
    }
#endif
#if VEC_WIDTH >= 512
    else {
      const auto index = Odd
          ? _mm512_set_epi8(
                15,13,11,9,7,5,3,1,15,13,11,9,7,5,3,1,
                15,13,11,9,7,5,3,1,15,13,11,9,7,5,3,1,
                15,13,11,9,7,5,3,1,15,13,11,9,7,5,3,1,
                15,13,11,9,7,5,3,1,15,13,11,9,7,5,3,1)
          : _mm512_set_epi8(
                14,12,10,8,6,4,2,0,14,12,10,8,6,4,2,0,
                14,12,10,8,6,4,2,0,14,12,10,8,6,4,2,0,
                14,12,10,8,6,4,2,0,14,12,10,8,6,4,2,0,
                14,12,10,8,6,4,2,0,14,12,10,8,6,4,2,0);
      const auto picked_a = _mm512_shuffle_epi8(av, index);
      const auto picked_b = _mm512_shuffle_epi8(bv, index);
      const auto mixed = _mm512_mask_blend_epi32(
          __mmask16{0xcccc}, picked_a, picked_b);
      const auto order = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
      return _mm512_permutexvar_epi64(order, mixed);
    }
#endif
  };
  const auto i32 = [&]<typename R>(R av, R bv) -> R {
    if constexpr (sizeof(R) == 16)
      return _mm_castps_si128(f32(
          _mm_castsi128_ps(av), _mm_castsi128_ps(bv)));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32)
      return _mm256_castps_si256(f32(
          _mm256_castsi256_ps(av), _mm256_castsi256_ps(bv)));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_castps_si512(f32(
        _mm512_castsi512_ps(av), _mm512_castsi512_ps(bv)));
#endif
  };
  const auto i64 = [&]<typename R>(R av, R bv) -> R {
    if constexpr (sizeof(R) == 16)
      return _mm_castpd_si128(f64(
          _mm_castsi128_pd(av), _mm_castsi128_pd(bv)));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32)
      return _mm256_castpd_si256(f64(
          _mm256_castsi256_pd(av), _mm256_castsi256_pd(bv)));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_castpd_si512(f64(
        _mm512_castsi512_pd(av), _mm512_castsi512_pd(bv)));
#endif
  };

  if constexpr (std::same_as<T, bfloat16_t>) return Vec<Tag>{i16(a.value, b.value)};
  else if constexpr (std::same_as<T, float16_t>) return Vec<Tag>{i16(a.value, b.value)};
  else if constexpr (std::same_as<T, float32_t>) return Vec<Tag>{f32(a.value, b.value)};
  else if constexpr (std::same_as<T, float64_t>) return Vec<Tag>{f64(a.value, b.value)};
  else if constexpr (std::same_as<T, int8_t>) return Vec<Tag>{i8(a.value, b.value)};
  else if constexpr (std::same_as<T, uint8_t>) return Vec<Tag>{i8(a.value, b.value)};
  else if constexpr (std::same_as<T, int16_t>) return Vec<Tag>{i16(a.value, b.value)};
  else if constexpr (std::same_as<T, uint16_t>) return Vec<Tag>{i16(a.value, b.value)};
  else if constexpr (std::same_as<T, int32_t>) return Vec<Tag>{i32(a.value, b.value)};
  else if constexpr (std::same_as<T, uint32_t>) return Vec<Tag>{i32(a.value, b.value)};
  else if constexpr (std::same_as<T, int64_t>) return Vec<Tag>{i64(a.value, b.value)};
  else if constexpr (std::same_as<T, uint64_t>) return Vec<Tag>{i64(a.value, b.value)};
  else static_assert(dispatch_dependent_false<T>, "unsupported x86 parity element type");
}

template <bool Odd, VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Half<Tag>> x86_extract_parity(Vec<Tag> value) {
  using T = ElementOf<Tag>;
  using InTraits = RepresentationTraits<X86Backend, Tag>;
  using InRaw = typename InTraits::RawVec;
  using OutRaw = typename RepresentationTraits<X86Backend, Half<Tag>>::RawVec;
  if constexpr (InTraits::word_count == 2) {
    return x86_concat_parity<Odd, Half<Tag>>(
        value.words[0], value.words[1]);
  } else {
    const auto packed = x86_concat_parity<Odd, Tag>(value, value);
    if constexpr (sizeof(InRaw) == sizeof(OutRaw)) {
      if constexpr (std::same_as<T, bfloat16_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, float16_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, float32_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, float64_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, int8_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, uint8_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, int16_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, uint16_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, int32_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, uint32_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, int64_t>) return Vec<Half<Tag>>{packed.value};
      else if constexpr (std::same_as<T, uint64_t>) return Vec<Half<Tag>>{packed.value};
      else static_assert(dispatch_dependent_false<T>, "unsupported x86 parity element type");
    } else if constexpr (std::same_as<T, float32_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castps256_ps128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castps512_ps256(packed.value)};
    } else if constexpr (std::same_as<T, float64_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castpd256_pd128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castpd512_pd256(packed.value)};
    } else if constexpr (std::same_as<T, bfloat16_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else if constexpr (std::same_as<T, float16_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else if constexpr (std::same_as<T, int8_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else if constexpr (std::same_as<T, uint8_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else if constexpr (std::same_as<T, int16_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else if constexpr (std::same_as<T, uint16_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else if constexpr (std::same_as<T, int32_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else if constexpr (std::same_as<T, uint32_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else if constexpr (std::same_as<T, int64_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else if constexpr (std::same_as<T, uint64_t>) {
      if constexpr (sizeof(InRaw) == 32)
        return Vec<Half<Tag>>{_mm256_castsi256_si128(packed.value)};
      else return Vec<Half<Tag>>{_mm512_castsi512_si256(packed.value)};
    } else {
      static_assert(dispatch_dependent_false<T>, "unsupported x86 parity element type");
    }
  }
}

template <VectorTag Tag>
  requires (
      RepresentationTraits<X86Backend, Tag>::word_count == 1 ||
      RepresentationTraits<X86Backend, Half<Tag>>::word_count == 1)
struct NativeImpl<X86Backend, EvenOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(EvenOp, Tag, Vec<Tag> value) {
    using T = ElementOf<Tag>;
    VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH(
        (x86_extract_parity<false, Tag>(value)), "unsupported x86 even element type");
  }
};

template <VectorTag Tag>
  requires (
      RepresentationTraits<X86Backend, Tag>::word_count == 1 ||
      RepresentationTraits<X86Backend, Half<Tag>>::word_count == 1)
struct NativeImpl<X86Backend, OddOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(OddOp, Tag, Vec<Tag> value) {
    using T = ElementOf<Tag>;
    VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH(
        (x86_extract_parity<true, Tag>(value)), "unsupported x86 odd element type");
  }
};

template <VectorTag Tag>
  requires (RepresentationTraits<X86Backend, Tag>::word_count == 1)
struct NativeImpl<X86Backend, ConcatEvenOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatEvenOp, Tag, Vec<Tag> a, Vec<Tag> b) {
    using T = ElementOf<Tag>;
    VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH(
        (x86_concat_parity<false, Tag>(a, b)), "unsupported x86 concat_even element type");
  }
};

template <VectorTag Tag>
  requires (RepresentationTraits<X86Backend, Tag>::word_count == 1)
struct NativeImpl<X86Backend, ConcatOddOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatOddOp, Tag, Vec<Tag> a, Vec<Tag> b) {
    using T = ElementOf<Tag>;
    VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH(
        (x86_concat_parity<true, Tag>(a, b)), "unsupported x86 concat_odd element type");
  }
};

/* **************************************************************************** */
//                       Vector interleave operations                        //
/* **************************************************************************** */

template <VectorTag Tag>
  requires (RepresentationTraits<X86Backend, Tag>::word_count == 1)
struct NativeImpl<X86Backend, InterleaveOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      InterleaveOp, Tag, Vec<Half<Tag>> a, Vec<Half<Tag>> b) {
    using T = ElementOf<Tag>;
    using OutRaw = typename RepresentationTraits<X86Backend, Tag>::RawVec;
    const auto f32 = []<typename R>(R av, R bv) {
      if constexpr (sizeof(OutRaw) == 16) {
        return _mm_unpacklo_ps(av, bv);
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(OutRaw) == 32) {
        const auto expanded_a = _mm256_castsi256_ps(_mm256_permute4x64_epi64(
            _mm256_castsi128_si256(_mm_castps_si128(av)),
            _MM_SHUFFLE(0, 1, 1, 0)));
        const auto expanded_b = _mm256_castsi256_ps(_mm256_permute4x64_epi64(
            _mm256_castsi128_si256(_mm_castps_si128(bv)),
            _MM_SHUFFLE(0, 1, 1, 0)));
        return _mm256_unpacklo_ps(expanded_a, expanded_b);
      }
#endif
#if VEC_WIDTH >= 512
      else {
        const auto order = _mm512_set_epi64(2, 3, 3, 2, 0, 1, 1, 0);
        const auto expanded_a = _mm512_castsi512_ps(_mm512_permutexvar_epi64(
            order, _mm512_castsi256_si512(_mm256_castps_si256(av))));
        const auto expanded_b = _mm512_castsi512_ps(_mm512_permutexvar_epi64(
            order, _mm512_castsi256_si512(_mm256_castps_si256(bv))));
        return _mm512_unpacklo_ps(expanded_a, expanded_b);
      }
#endif
    };
    const auto f64 = []<typename R>(R av, R bv) {
      if constexpr (sizeof(OutRaw) == 16) {
        return _mm_unpacklo_pd(av, bv);
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(OutRaw) == 32) {
        const auto expanded_a = _mm256_castsi256_pd(_mm256_permute4x64_epi64(
            _mm256_castsi128_si256(_mm_castpd_si128(av)),
            _MM_SHUFFLE(0, 1, 1, 0)));
        const auto expanded_b = _mm256_castsi256_pd(_mm256_permute4x64_epi64(
            _mm256_castsi128_si256(_mm_castpd_si128(bv)),
            _MM_SHUFFLE(0, 1, 1, 0)));
        return _mm256_unpacklo_pd(expanded_a, expanded_b);
      }
#endif
#if VEC_WIDTH >= 512
      else {
        const auto order = _mm512_set_epi64(2, 3, 3, 2, 0, 1, 1, 0);
        const auto expanded_a = _mm512_castsi512_pd(_mm512_permutexvar_epi64(
            order, _mm512_castsi256_si512(_mm256_castpd_si256(av))));
        const auto expanded_b = _mm512_castsi512_pd(_mm512_permutexvar_epi64(
            order, _mm512_castsi256_si512(_mm256_castpd_si256(bv))));
        return _mm512_unpacklo_pd(expanded_a, expanded_b);
      }
#endif
    };
    const auto integer = []<int ElementBytes, typename R>(R av, R bv) {
      if constexpr (sizeof(OutRaw) == 16) {
        if constexpr (ElementBytes == 1) return _mm_unpacklo_epi8(av, bv);
        else if constexpr (ElementBytes == 2) return _mm_unpacklo_epi16(av, bv);
        else if constexpr (ElementBytes == 4) return _mm_unpacklo_epi32(av, bv);
        else return _mm_unpacklo_epi64(av, bv);
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(OutRaw) == 32) {
        const auto expanded_a = _mm256_permute4x64_epi64(
            _mm256_castsi128_si256(av), _MM_SHUFFLE(0, 1, 1, 0));
        const auto expanded_b = _mm256_permute4x64_epi64(
            _mm256_castsi128_si256(bv), _MM_SHUFFLE(0, 1, 1, 0));
        if constexpr (ElementBytes == 1)
          return _mm256_unpacklo_epi8(expanded_a, expanded_b);
        else if constexpr (ElementBytes == 2)
          return _mm256_unpacklo_epi16(expanded_a, expanded_b);
        else if constexpr (ElementBytes == 4)
          return _mm256_unpacklo_epi32(expanded_a, expanded_b);
        else return _mm256_unpacklo_epi64(expanded_a, expanded_b);
      }
#endif
#if VEC_WIDTH >= 512
      else {
        const auto order = _mm512_set_epi64(2, 3, 3, 2, 0, 1, 1, 0);
        const auto expanded_a = _mm512_permutexvar_epi64(
            order, _mm512_castsi256_si512(av));
        const auto expanded_b = _mm512_permutexvar_epi64(
            order, _mm512_castsi256_si512(bv));
        if constexpr (ElementBytes == 1)
          return _mm512_unpacklo_epi8(expanded_a, expanded_b);
        else if constexpr (ElementBytes == 2)
          return _mm512_unpacklo_epi16(expanded_a, expanded_b);
        else if constexpr (ElementBytes == 4)
          return _mm512_unpacklo_epi32(expanded_a, expanded_b);
        else return _mm512_unpacklo_epi64(expanded_a, expanded_b);
      }
#endif
    };

    if constexpr (std::same_as<T, bfloat16_t>)
      return Vec<Tag>{integer.template operator()<2>(a.value, b.value)};
    else if constexpr (std::same_as<T, float16_t>)
      return Vec<Tag>{integer.template operator()<2>(a.value, b.value)};
    else if constexpr (std::same_as<T, float32_t>) return Vec<Tag>{f32(a.value, b.value)};
    else if constexpr (std::same_as<T, float64_t>) return Vec<Tag>{f64(a.value, b.value)};
    else if constexpr (std::same_as<T, int8_t>)
      return Vec<Tag>{integer.template operator()<1>(a.value, b.value)};
    else if constexpr (std::same_as<T, uint8_t>)
      return Vec<Tag>{integer.template operator()<1>(a.value, b.value)};
    else if constexpr (std::same_as<T, int16_t>)
      return Vec<Tag>{integer.template operator()<2>(a.value, b.value)};
    else if constexpr (std::same_as<T, uint16_t>)
      return Vec<Tag>{integer.template operator()<2>(a.value, b.value)};
    else if constexpr (std::same_as<T, int32_t>)
      return Vec<Tag>{integer.template operator()<4>(a.value, b.value)};
    else if constexpr (std::same_as<T, uint32_t>)
      return Vec<Tag>{integer.template operator()<4>(a.value, b.value)};
    else if constexpr (std::same_as<T, int64_t>)
      return Vec<Tag>{integer.template operator()<8>(a.value, b.value)};
    else if constexpr (std::same_as<T, uint64_t>)
      return Vec<Tag>{integer.template operator()<8>(a.value, b.value)};
    else static_assert(
        dispatch_dependent_false<T>, "unsupported x86 interleave element type");
  }
};

#undef VECOPS_VEC_X86_EXACT_REARRANGE_DISPATCH

#define VECOPS_VEC_X86_RETURN_UNPACK(Direction, Suffix)                  \
  do {                                                                    \
    if constexpr (sizeof(Raw) == 16)                                     \
      return NativeWordVec<Tag>{_mm_unpack##Direction##_##Suffix(a.value, b.value)}; \
    else if constexpr (sizeof(Raw) == 32)                                \
      return NativeWordVec<Tag>{_mm256_unpack##Direction##_##Suffix(a.value, b.value)}; \
    else                                                                  \
      return NativeWordVec<Tag>{_mm512_unpack##Direction##_##Suffix(a.value, b.value)}; \
  } while (false)

#define VECOPS_VEC_X86_DEFINE_LOCAL_INTERLEAVE(OpType, Direction)        \
  template <>                                                             \
  struct NativeWordImpl<X86Backend, OpType> {                             \
    template <nint_t Index, VectorTag Tag>                                \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                  \
        OpType, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {        \
      using Traits = RepresentationTraits<X86Backend, Tag>;               \
      using T = ElementOf<Tag>;                                           \
      using Raw = typename Traits::RawVec;                                \
      static_assert(Index >= 0 && Index < Traits::word_count);            \
      if constexpr (std::same_as<T, bfloat16_t>)                          \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi16);                  \
      else if constexpr (std::same_as<T, float16_t>)                      \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi16);                  \
      else if constexpr (std::same_as<T, float32_t>)                      \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, ps);                     \
      else if constexpr (std::same_as<T, float64_t>)                      \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, pd);                     \
      else if constexpr (std::same_as<T, int8_t>)                         \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi8);                   \
      else if constexpr (std::same_as<T, uint8_t>)                        \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi8);                   \
      else if constexpr (std::same_as<T, int16_t>)                        \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi16);                  \
      else if constexpr (std::same_as<T, uint16_t>)                       \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi16);                  \
      else if constexpr (std::same_as<T, int32_t>)                        \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi32);                  \
      else if constexpr (std::same_as<T, uint32_t>)                       \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi32);                  \
      else if constexpr (std::same_as<T, int64_t>)                        \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi64);                  \
      else if constexpr (std::same_as<T, uint64_t>)                       \
        VECOPS_VEC_X86_RETURN_UNPACK(Direction, epi64);                  \
      else static_assert(                                                 \
          dispatch_dependent_false<T>, "unsupported x86 local interleave element type"); \
    }                                                                     \
  }

VECOPS_VEC_X86_DEFINE_LOCAL_INTERLEAVE(LocalInterleaveLowerOp, lo);
VECOPS_VEC_X86_DEFINE_LOCAL_INTERLEAVE(LocalInterleaveUpperOp, hi);

#undef VECOPS_VEC_X86_DEFINE_LOCAL_INTERLEAVE
#undef VECOPS_VEC_X86_RETURN_UNPACK

template <bool Odd, nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_interleave_parity_word(
    Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  using T = ElementOf<Tag>;
  using Raw = typename Traits::RawVec;
  static_assert(Index >= 0 && Index < Traits::word_count);

  const auto f32 = []<typename R>(R av, R bv) -> R {
    constexpr int select = Odd ? _MM_SHUFFLE(3, 1, 3, 1)
                               : _MM_SHUFFLE(2, 0, 2, 0);
    if constexpr (sizeof(R) == 16) {
      const auto mixed = _mm_shuffle_ps(av, bv, select);
      return _mm_shuffle_ps(mixed, mixed, _MM_SHUFFLE(3, 1, 2, 0));
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) {
      const auto mixed = _mm256_shuffle_ps(av, bv, select);
      return _mm256_shuffle_ps(mixed, mixed, _MM_SHUFFLE(3, 1, 2, 0));
    }
#endif
#if VEC_WIDTH >= 512
    else {
      const auto mixed = _mm512_shuffle_ps(av, bv, select);
      return _mm512_shuffle_ps(mixed, mixed, _MM_SHUFFLE(3, 1, 2, 0));
    }
#endif
  };
  const auto f64 = []<typename R>(R av, R bv) -> R {
    constexpr int select = Odd ? 0xff : 0x00;
    if constexpr (sizeof(R) == 16) return _mm_shuffle_pd(av, bv, select);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) return _mm256_shuffle_pd(av, bv, select);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_shuffle_pd(av, bv, select);
#endif
  };
  const auto i16 = []<typename R>(R av, R bv) -> R {
    constexpr int select = Odd ? _MM_SHUFFLE(3, 3, 1, 1)
                               : _MM_SHUFFLE(2, 2, 0, 0);
    if constexpr (sizeof(R) == 16) {
      auto picked = Odd ? _mm_shufflelo_epi16(av, select)
                        : _mm_shufflelo_epi16(bv, select);
      picked = _mm_shufflehi_epi16(picked, select);
      return Odd ? _mm_blend_epi16(picked, bv, 0xaa)
                 : _mm_blend_epi16(av, picked, 0xaa);
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) {
      auto picked = Odd ? _mm256_shufflelo_epi16(av, select)
                        : _mm256_shufflelo_epi16(bv, select);
      picked = _mm256_shufflehi_epi16(picked, select);
      return Odd ? _mm256_blend_epi16(picked, bv, 0xaa)
                 : _mm256_blend_epi16(av, picked, 0xaa);
    }
#endif
#if VEC_WIDTH >= 512
    else {
      auto picked = Odd ? _mm512_shufflelo_epi16(av, select)
                        : _mm512_shufflelo_epi16(bv, select);
      picked = _mm512_shufflehi_epi16(picked, select);
      return Odd
          ? _mm512_mask_blend_epi16(__mmask32{0xaaaaaaaaU}, picked, bv)
          : _mm512_mask_blend_epi16(__mmask32{0xaaaaaaaaU}, av, picked);
    }
#endif
  };
  const auto i8 = []<typename R>(R av, R bv) -> R {
    if constexpr (sizeof(R) == 16) {
      if constexpr (Odd) {
        return _mm_or_si128(
            _mm_srli_epi16(av, 8),
            _mm_slli_epi16(_mm_srli_epi16(bv, 8), 8));
      } else {
        return _mm_or_si128(
            _mm_srli_epi16(_mm_slli_epi16(av, 8), 8),
            _mm_slli_epi16(bv, 8));
      }
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) {
      if constexpr (Odd) {
        return _mm256_or_si256(
            _mm256_srli_epi16(av, 8),
            _mm256_slli_epi16(_mm256_srli_epi16(bv, 8), 8));
      } else {
        return _mm256_or_si256(
            _mm256_srli_epi16(_mm256_slli_epi16(av, 8), 8),
            _mm256_slli_epi16(bv, 8));
      }
    }
#endif
#if VEC_WIDTH >= 512
    else {
      if constexpr (Odd) {
        return _mm512_or_si512(
            _mm512_srli_epi16(av, 8),
            _mm512_slli_epi16(_mm512_srli_epi16(bv, 8), 8));
      } else {
        return _mm512_or_si512(
            _mm512_srli_epi16(_mm512_slli_epi16(av, 8), 8),
            _mm512_slli_epi16(bv, 8));
      }
    }
#endif
  };

  if constexpr (std::same_as<T, bfloat16_t>) {
    return NativeWordVec<Tag>{i16(a.value, b.value)};
  } else if constexpr (std::same_as<T, float16_t>) {
    return NativeWordVec<Tag>{i16(a.value, b.value)};
  } else if constexpr (std::same_as<T, float32_t>) {
    return NativeWordVec<Tag>{f32(a.value, b.value)};
  } else if constexpr (std::same_as<T, float64_t>) {
    return NativeWordVec<Tag>{f64(a.value, b.value)};
  } else if constexpr (std::same_as<T, int8_t>) {
    return NativeWordVec<Tag>{i8(a.value, b.value)};
  } else if constexpr (std::same_as<T, uint8_t>) {
    return NativeWordVec<Tag>{i8(a.value, b.value)};
  } else if constexpr (std::same_as<T, int16_t>) {
    return NativeWordVec<Tag>{i16(a.value, b.value)};
  } else if constexpr (std::same_as<T, uint16_t>) {
    return NativeWordVec<Tag>{i16(a.value, b.value)};
  } else if constexpr (std::same_as<T, int32_t>) {
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_castps_si128(f32(
          _mm_castsi128_ps(a.value), _mm_castsi128_ps(b.value)))};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<Tag>{_mm256_castps_si256(f32(
          _mm256_castsi256_ps(a.value), _mm256_castsi256_ps(b.value)))};
#endif
#if VEC_WIDTH >= 512
    else
      return NativeWordVec<Tag>{_mm512_castps_si512(f32(
          _mm512_castsi512_ps(a.value), _mm512_castsi512_ps(b.value)))};
#endif
  } else if constexpr (std::same_as<T, uint32_t>) {
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_castps_si128(f32(
          _mm_castsi128_ps(a.value), _mm_castsi128_ps(b.value)))};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<Tag>{_mm256_castps_si256(f32(
          _mm256_castsi256_ps(a.value), _mm256_castsi256_ps(b.value)))};
#endif
#if VEC_WIDTH >= 512
    else
      return NativeWordVec<Tag>{_mm512_castps_si512(f32(
          _mm512_castsi512_ps(a.value), _mm512_castsi512_ps(b.value)))};
#endif
  } else if constexpr (std::same_as<T, int64_t>) {
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_castpd_si128(f64(
          _mm_castsi128_pd(a.value), _mm_castsi128_pd(b.value)))};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<Tag>{_mm256_castpd_si256(f64(
          _mm256_castsi256_pd(a.value), _mm256_castsi256_pd(b.value)))};
#endif
#if VEC_WIDTH >= 512
    else
      return NativeWordVec<Tag>{_mm512_castpd_si512(f64(
          _mm512_castsi512_pd(a.value), _mm512_castsi512_pd(b.value)))};
#endif
  } else if constexpr (std::same_as<T, uint64_t>) {
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_castpd_si128(f64(
          _mm_castsi128_pd(a.value), _mm_castsi128_pd(b.value)))};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<Tag>{_mm256_castpd_si256(f64(
          _mm256_castsi256_pd(a.value), _mm256_castsi256_pd(b.value)))};
#endif
#if VEC_WIDTH >= 512
    else
      return NativeWordVec<Tag>{_mm512_castpd_si512(f64(
          _mm512_castsi512_pd(a.value), _mm512_castsi512_pd(b.value)))};
#endif
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "unsupported x86 interleave parity element type");
  }
}

template <>
struct NativeWordImpl<X86Backend, InterleaveEvenOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      InterleaveEvenOp, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    return x86_interleave_parity_word<false, Index>(tag, a, b);
  }
};

template <>
struct NativeWordImpl<X86Backend, InterleaveOddOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      InterleaveOddOp, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    return x86_interleave_parity_word<true, Index>(tag, a, b);
  }
};

/* **************************************************************************** */
//                           Shuffle operations                              //
/* **************************************************************************** */

template <nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_local_shuffle_word(
    Tag, NativeWordVec<Tag> value,
    NativeWordVec<IndexTag<Tag>> indices) {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  using T = ElementOf<Tag>;
  using Raw = typename Traits::RawVec;
  static_assert(Index >= 0 && Index < Traits::word_count);
  static_assert(
      sizeof(Raw) ==
      sizeof(typename RepresentationTraits<X86Backend, IndexTag<Tag>>::RawVec));

  const auto shuffle_i8 = []<typename R>(R v, R i) -> R {
    if constexpr (sizeof(R) == 16) return _mm_shuffle_epi8(v, i);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) return _mm256_shuffle_epi8(v, i);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_shuffle_epi8(v, i);
#endif
  };
  const auto shuffle_i16 = [&]<typename R>(R v, R i) -> R {
    if constexpr (sizeof(R) == 16) {
      const auto doubled = _mm_slli_epi32(i, 1);
      const auto high_bytes = _mm_slli_epi32(i, 9);
#ifdef HAS_AVX512F
      const auto byte_indices = _mm_ternarylogic_epi32(
          doubled, high_bytes, _mm_set1_epi16(0x0100),
          _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
#else
      const auto byte_indices = _mm_or_si128(
          _mm_or_si128(doubled, high_bytes), _mm_set1_epi16(0x0100));
#endif
      return shuffle_i8(v, byte_indices);
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) {
      const auto doubled = _mm256_slli_epi32(i, 1);
      const auto high_bytes = _mm256_slli_epi32(i, 9);
#ifdef HAS_AVX512F
      const auto byte_indices = _mm256_ternarylogic_epi32(
          doubled, high_bytes, _mm256_set1_epi16(0x0100),
          _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
#else
      const auto byte_indices = _mm256_or_si256(
          _mm256_or_si256(doubled, high_bytes),
          _mm256_set1_epi16(0x0100));
#endif
      return shuffle_i8(v, byte_indices);
    }
#endif
#if VEC_WIDTH >= 512
    else {
      const auto doubled = _mm512_slli_epi32(i, 1);
      const auto high_bytes = _mm512_slli_epi32(i, 9);
      const auto byte_indices = _mm512_ternarylogic_epi32(
          doubled, high_bytes, _mm512_set1_epi16(0x0100),
          _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
      return shuffle_i8(v, byte_indices);
    }
#endif
  };
  const auto shuffle_f32 = []<typename R, typename I>(R v, I i) -> R {
    if constexpr (sizeof(R) == 16) return _mm_permutevar_ps(v, i);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32) return _mm256_permutevar_ps(v, i);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_permutevar_ps(v, i);
#endif
  };
  const auto shuffle_f64 = []<typename R, typename I>(R v, I i) -> R {
    if constexpr (sizeof(R) == 16)
      return _mm_permutevar_pd(v, _mm_slli_epi64(i, 1));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32)
      return _mm256_permutevar_pd(v, _mm256_slli_epi64(i, 1));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_permutevar_pd(v, _mm512_slli_epi64(i, 1));
#endif
  };
  const auto shuffle_i32 = [&]<typename R>(R v, R i) -> R {
    if constexpr (sizeof(R) == 16)
      return _mm_castps_si128(shuffle_f32(_mm_castsi128_ps(v), i));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32)
      return _mm256_castps_si256(shuffle_f32(_mm256_castsi256_ps(v), i));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_castps_si512(
        shuffle_f32(_mm512_castsi512_ps(v), i));
#endif
  };
  const auto shuffle_i64 = [&]<typename R>(R v, R i) -> R {
    if constexpr (sizeof(R) == 16)
      return _mm_castpd_si128(shuffle_f64(_mm_castsi128_pd(v), i));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(R) == 32)
      return _mm256_castpd_si256(shuffle_f64(_mm256_castsi256_pd(v), i));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_castpd_si512(
        shuffle_f64(_mm512_castsi512_pd(v), i));
#endif
  };

  if constexpr (std::same_as<T, bfloat16_t>) {
    return NativeWordVec<Tag>{shuffle_i16(value.value, indices.value)};
  } else if constexpr (std::same_as<T, float16_t>) {
    return NativeWordVec<Tag>{shuffle_i16(value.value, indices.value)};
  } else if constexpr (std::same_as<T, float32_t>) {
    return NativeWordVec<Tag>{shuffle_f32(value.value, indices.value)};
  } else if constexpr (std::same_as<T, float64_t>) {
    return NativeWordVec<Tag>{shuffle_f64(value.value, indices.value)};
  } else if constexpr (std::same_as<T, int8_t>) {
    return NativeWordVec<Tag>{shuffle_i8(value.value, indices.value)};
  } else if constexpr (std::same_as<T, uint8_t>) {
    return NativeWordVec<Tag>{shuffle_i8(value.value, indices.value)};
  } else if constexpr (std::same_as<T, int16_t>) {
    return NativeWordVec<Tag>{shuffle_i16(value.value, indices.value)};
  } else if constexpr (std::same_as<T, uint16_t>) {
    return NativeWordVec<Tag>{shuffle_i16(value.value, indices.value)};
  } else if constexpr (std::same_as<T, int32_t>) {
    return NativeWordVec<Tag>{shuffle_i32(value.value, indices.value)};
  } else if constexpr (std::same_as<T, uint32_t>) {
    return NativeWordVec<Tag>{shuffle_i32(value.value, indices.value)};
  } else if constexpr (std::same_as<T, int64_t>) {
    return NativeWordVec<Tag>{shuffle_i64(value.value, indices.value)};
  } else if constexpr (std::same_as<T, uint64_t>) {
    return NativeWordVec<Tag>{shuffle_i64(value.value, indices.value)};
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "unsupported x86 local shuffle element type");
  }
}

template <>
struct NativeWordImpl<X86Backend, LocalShuffleOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LocalShuffleOp, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<IndexTag<Tag>> indices) {
    return x86_local_shuffle_word<Index>(tag, value, indices);
  }
};

template <>
struct NativeWordImpl<X86Backend, ShuffleOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ShuffleOp, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<IndexTag<Tag>> indices) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = typename Traits::RawVec;
    static_assert(Index >= 0 && Index < Traits::word_count);

    if constexpr (sizeof(Raw) == 16) {
      return x86_local_shuffle_word<Index>(tag, value, indices);
    } else {
      const auto shuffle_i8 = []<typename R>(R v, R i) -> R {
        if constexpr (sizeof(R) == 32) {
#ifdef HAS_AVX512VBMI
          return _mm256_permutexvar_epi8(i, v);
#else
          const auto lane0 = _mm256_permute2f128_si256(v, v, 0x00);
          const auto lane1 = _mm256_permute2f128_si256(v, v, 0x11);
          const auto lower = _mm256_shuffle_epi8(lane0, i);
#ifdef HAS_AVX512BW
          const auto lane_mask = _mm256_movepi8_mask(
              _mm256_slli_epi16(i, 3));
          return _mm256_mask_shuffle_epi8(lower, lane_mask, lane1, i);
#else
          const auto lane_mask = _mm256_slli_epi16(i, 3);
          const auto upper = _mm256_shuffle_epi8(lane1, i);
          return _mm256_blendv_epi8(lower, upper, lane_mask);
#endif
#endif
        }
#if VEC_WIDTH >= 512
        else {
#ifdef HAS_AVX512VBMI
          return _mm512_permutexvar_epi8(i, v);
#else
          const auto lane0 = _mm512_shuffle_i32x4(
              v, v, _MM_SHUFFLE(0, 0, 0, 0));
          const auto lane1 = _mm512_shuffle_i32x4(
              v, v, _MM_SHUFFLE(1, 1, 1, 1));
          const auto lane2 = _mm512_shuffle_i32x4(
              v, v, _MM_SHUFFLE(2, 2, 2, 2));
          const auto lane3 = _mm512_shuffle_i32x4(
              v, v, _MM_SHUFFLE(3, 3, 3, 3));
          const auto low_lane_bit = _mm512_movepi8_mask(
              _mm512_slli_epi16(i, 3));
          const auto high_lane_bit = _mm512_movepi8_mask(
              _mm512_slli_epi16(i, 2));
          const auto lower02 = _mm512_shuffle_epi8(lane0, i);
          const auto lower13 = _mm512_shuffle_epi8(lane2, i);
          const auto selected01 = _mm512_mask_shuffle_epi8(
              lower02, low_lane_bit, lane1, i);
          const auto selected23 = _mm512_mask_shuffle_epi8(
              lower13, low_lane_bit, lane3, i);
          return _mm512_mask_blend_epi8(
              high_lane_bit, selected01, selected23);
#endif
        }
#endif
      };
      const auto to_byte_indices = []<typename R>(R i) -> R {
        if constexpr (sizeof(R) == 32) {
          const auto doubled = _mm256_slli_epi32(i, 1);
          const auto high_bytes = _mm256_slli_epi32(i, 9);
#ifdef HAS_AVX512F
          return _mm256_ternarylogic_epi32(
              doubled, high_bytes, _mm256_set1_epi16(0x0100),
              _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
#else
          return _mm256_or_si256(
              _mm256_or_si256(doubled, high_bytes),
              _mm256_set1_epi16(0x0100));
#endif
        }
#if VEC_WIDTH >= 512
        else {
          const auto doubled = _mm512_slli_epi32(i, 1);
          const auto high_bytes = _mm512_slli_epi32(i, 9);
          return _mm512_ternarylogic_epi32(
              doubled, high_bytes, _mm512_set1_epi16(0x0100),
              _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
        }
#endif
      };
      const auto shuffle_i16 = [&]<typename R>(R v, R i) -> R {
        if constexpr (sizeof(R) == 32) {
#ifdef HAS_AVX512BW
          return _mm256_permutexvar_epi16(i, v);
#else
          return shuffle_i8(v, to_byte_indices(i));
#endif
        }
#if VEC_WIDTH >= 512
        else return _mm512_permutexvar_epi16(i, v);
#endif
      };
      const auto shuffle_f32 = []<typename R, typename I>(R v, I i) -> R {
        if constexpr (sizeof(R) == 32)
          return _mm256_permutevar8x32_ps(v, i);
#if VEC_WIDTH >= 512
        else return _mm512_permutexvar_ps(i, v);
#endif
      };
      const auto shuffle_i32 = []<typename R>(R v, R i) -> R {
        if constexpr (sizeof(R) == 32)
          return _mm256_permutevar8x32_epi32(v, i);
#if VEC_WIDTH >= 512
        else return _mm512_permutexvar_epi32(i, v);
#endif
      };
      const auto shuffle_f64 = []<typename R, typename I>(R v, I i) -> R {
        if constexpr (sizeof(R) == 32) {
#ifdef HAS_AVX512F
          return _mm256_permutexvar_pd(i, v);
#else
          const auto low_words = _mm256_slli_epi64(i, 1);
          const auto high_words = _mm256_slli_epi64(i, 33);
          const auto word_indices = _mm256_or_si256(
              _mm256_or_si256(low_words, high_words),
              _mm256_set1_epi64x(0x00000001'00000000));
          return _mm256_castps_pd(_mm256_permutevar8x32_ps(
              _mm256_castpd_ps(v), word_indices));
#endif
        }
#if VEC_WIDTH >= 512
        else return _mm512_permutexvar_pd(i, v);
#endif
      };
      const auto shuffle_i64 = [&]<typename R>(R v, R i) -> R {
        if constexpr (sizeof(R) == 32) {
#ifdef HAS_AVX512F
          return _mm256_permutexvar_epi64(i, v);
#else
          return _mm256_castpd_si256(
              shuffle_f64(_mm256_castsi256_pd(v), i));
#endif
        }
#if VEC_WIDTH >= 512
        else return _mm512_permutexvar_epi64(i, v);
#endif
      };

      if constexpr (std::same_as<T, bfloat16_t>) {
        return NativeWordVec<Tag>{shuffle_i16(value.value, indices.value)};
      } else if constexpr (std::same_as<T, float16_t>) {
        return NativeWordVec<Tag>{shuffle_i16(value.value, indices.value)};
      } else if constexpr (std::same_as<T, float32_t>) {
        return NativeWordVec<Tag>{shuffle_f32(value.value, indices.value)};
      } else if constexpr (std::same_as<T, float64_t>) {
        return NativeWordVec<Tag>{shuffle_f64(value.value, indices.value)};
      } else if constexpr (std::same_as<T, int8_t>) {
        return NativeWordVec<Tag>{shuffle_i8(value.value, indices.value)};
      } else if constexpr (std::same_as<T, uint8_t>) {
        return NativeWordVec<Tag>{shuffle_i8(value.value, indices.value)};
      } else if constexpr (std::same_as<T, int16_t>) {
        return NativeWordVec<Tag>{shuffle_i16(value.value, indices.value)};
      } else if constexpr (std::same_as<T, uint16_t>) {
        return NativeWordVec<Tag>{shuffle_i16(value.value, indices.value)};
      } else if constexpr (std::same_as<T, int32_t>) {
        return NativeWordVec<Tag>{shuffle_i32(value.value, indices.value)};
      } else if constexpr (std::same_as<T, uint32_t>) {
        return NativeWordVec<Tag>{shuffle_i32(value.value, indices.value)};
      } else if constexpr (std::same_as<T, int64_t>) {
        return NativeWordVec<Tag>{shuffle_i64(value.value, indices.value)};
      } else if constexpr (std::same_as<T, uint64_t>) {
        return NativeWordVec<Tag>{shuffle_i64(value.value, indices.value)};
      } else {
        static_assert(
            dispatch_dependent_false<T>,
            "unsupported x86 shuffle element type");
      }
    }
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_BASIC_H
