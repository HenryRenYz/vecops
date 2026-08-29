#ifndef VECOPS_VEC_DETAILS_X86_ARITHMETIC_H
#define VECOPS_VEC_DETAILS_X86_ARITHMETIC_H

/**
 * @file Arithmetic.h
 * @brief x86 backend arithmetic operations using SSE/AVX/AVX2/AVX-512
 * intrinsics.
 *
 * Small-float types (bfloat16, float16) are widened to float32 for
 * computation since x86 lacks native fp16/bf16 arithmetic. Integer
 * operations use ISA-specific intrinsics with unsigned bitcast for
 * modular (wrap-around) semantics.
 */

#include <cmath>
#include <cstring>
#include <limits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {

template <Element T, typename Raw, typename Operation>

/* **************************************************************************** */
//                 Small-float and integer arithmetic helpers                 //
/* **************************************************************************** */

VECOPS_ALWAYS_INLINE Raw x86_emulated_small_float_binary(
    Raw a, Raw b, Operation operation) {
  static_assert(
      std::same_as<T, float16_t> || std::same_as<T, bfloat16_t>);
  constexpr std::size_t lanes = sizeof(Raw) / sizeof(T);
  alignas(64) T a_lanes[lanes];
  alignas(64) T b_lanes[lanes];
  alignas(64) T result_lanes[lanes];
  std::memcpy(a_lanes, &a, sizeof(Raw));
  std::memcpy(b_lanes, &b, sizeof(Raw));
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    result_lanes[lane] = operation(a_lanes[lane], b_lanes[lane]);
  }
  Raw result;
  std::memcpy(&result, result_lanes, sizeof(Raw));
  return result;
}

template <Element T, typename Raw, typename Operation>
VECOPS_ALWAYS_INLINE Raw x86_emulated_small_float_unary(
    Raw value, Operation operation) {
  static_assert(
      std::same_as<T, float16_t> || std::same_as<T, bfloat16_t>);
  constexpr std::size_t lanes = sizeof(Raw) / sizeof(T);
  alignas(64) T input[lanes];
  alignas(64) T result_lanes[lanes];
  std::memcpy(input, &value, sizeof(Raw));
  for (std::size_t lane = 0; lane < lanes; ++lane)
    result_lanes[lane] = operation(input[lane]);
  Raw result;
  std::memcpy(&result, result_lanes, sizeof(Raw));
  return result;
}

#if defined(HAS_AVX512_BF16)
VECOPS_ALWAYS_INLINE __m128bh x86_cast_si128_to_bfloat16(__m128i value) {
  union {
    __m128i integer;
    __m128bh bfloat;
  } cast{.integer = value};
  return cast.bfloat;
}

VECOPS_ALWAYS_INLINE __m128i x86_cast_bfloat16_to_si128(__m128bh value) {
  union {
    __m128i integer;
    __m128bh bfloat;
  } cast{.bfloat = value};
  return cast.integer;
}

#if VEC_WIDTH >= 256
VECOPS_ALWAYS_INLINE __m256bh x86_cast_si256_to_bfloat16(__m256i value) {
  union {
    __m256i integer;
    __m256bh bfloat;
  } cast{.integer = value};
  return cast.bfloat;
}

VECOPS_ALWAYS_INLINE __m256i x86_cast_bfloat16_to_si256(__m256bh value) {
  union {
    __m256i integer;
    __m256bh bfloat;
  } cast{.bfloat = value};
  return cast.integer;
}
#endif

#if VEC_WIDTH >= 512
VECOPS_ALWAYS_INLINE __m512i x86_cast_bfloat16_to_si512(__m512bh value) {
  union {
    __m512i integer;
    __m512bh bfloat;
  } cast{.bfloat = value};
  return cast.integer;
}
#endif
#endif

VECOPS_ALWAYS_INLINE void x86_bfloat16_to_float32_pair(
    __m128i value, __m128& low, __m128& high) {
#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  const auto widened = _mm256_cvtpbh_ps(x86_cast_si128_to_bfloat16(value));
  low = _mm256_castps256_ps128(widened);
  high = _mm256_extractf128_ps(widened, 1);
#else
  low = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(value), 16));
  high = _mm_castsi128_ps(_mm_slli_epi32(
      _mm_cvtepu16_epi32(_mm_srli_si128(value, 8)), 16));
#endif
}

VECOPS_ALWAYS_INLINE __m128i x86_float32_pair_to_bfloat16(
    __m128 low, __m128 high) {
#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  return x86_cast_bfloat16_to_si128(_mm_cvtne2ps_pbh(high, low));
#else
  const auto convert = [](__m128 value) {
    const auto bits = _mm_castps_si128(value);
    const auto lsb = _mm_and_si128(
        _mm_srli_epi32(bits, 16), _mm_set1_epi32(1));
    auto rounded = _mm_srli_epi32(_mm_add_epi32(
        bits, _mm_add_epi32(lsb, _mm_set1_epi32(0x7fff))), 16);
    const auto ordered = _mm_castps_si128(_mm_cmpord_ps(value, value));
    return _mm_blendv_epi8(
        _mm_set1_epi32(0x7fc0), rounded, ordered);
  };
  return _mm_packus_epi32(convert(low), convert(high));
#endif
}

#if VEC_WIDTH >= 256
VECOPS_ALWAYS_INLINE void x86_bfloat16_to_float32_pair(
    __m256i value, __m256& low, __m256& high) {
#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  const auto widened = _mm512_cvtpbh_ps(x86_cast_si256_to_bfloat16(value));
  low = _mm512_castps512_ps256(widened);
  high = _mm512_extractf32x8_ps(widened, 1);
#else
  const auto low128 = _mm256_castsi256_si128(value);
  const auto high128 = _mm256_extractf128_si256(value, 1);
  low = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(low128), 16));
  high = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(high128), 16));
#endif
}

VECOPS_ALWAYS_INLINE __m256i x86_float32_pair_to_bfloat16(
    __m256 low, __m256 high) {
#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  return x86_cast_bfloat16_to_si256(_mm256_cvtne2ps_pbh(high, low));
#else
  const auto convert = [](__m256 value) {
    const auto bits = _mm256_castps_si256(value);
    const auto lsb = _mm256_and_si256(
        _mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    auto rounded = _mm256_srli_epi32(_mm256_add_epi32(
        bits, _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff))), 16);
    const auto ordered = _mm256_castps_si256(
        _mm256_cmp_ps(value, value, _CMP_ORD_Q));
    return _mm256_blendv_epi8(
        _mm256_set1_epi32(0x7fc0), rounded, ordered);
  };
  const auto packed = _mm256_packus_epi32(convert(low), convert(high));
  return _mm256_permute4x64_epi64(packed, 0xd8);
#endif
}
#endif

#if VEC_WIDTH >= 512
VECOPS_ALWAYS_INLINE void x86_bfloat16_to_float32_pair(
    __m512i value, __m512& low, __m512& high) {
  const auto low256 = _mm512_castsi512_si256(value);
  const auto high256 = _mm512_extracti64x4_epi64(value, 1);
#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  low = _mm512_cvtpbh_ps(x86_cast_si256_to_bfloat16(low256));
  high = _mm512_cvtpbh_ps(x86_cast_si256_to_bfloat16(high256));
#else
  low = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(low256), 16));
  high = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(high256), 16));
#endif
}

VECOPS_ALWAYS_INLINE __m512i x86_float32_pair_to_bfloat16(
    __m512 low, __m512 high) {
#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  return x86_cast_bfloat16_to_si512(_mm512_cvtne2ps_pbh(high, low));
#else
  const auto low_low = _mm512_castps512_ps256(low);
  const auto low_high = _mm512_extractf32x8_ps(low, 1);
  const auto high_low = _mm512_castps512_ps256(high);
  const auto high_high = _mm512_extractf32x8_ps(high, 1);
  const auto packed_low =
      x86_float32_pair_to_bfloat16(low_low, low_high);
  const auto packed_high =
      x86_float32_pair_to_bfloat16(high_low, high_high);
  return _mm512_inserti64x4(
      _mm512_castsi256_si512(packed_low), packed_high, 1);
#endif
}
#endif

#if defined(HAS_F16C)
VECOPS_ALWAYS_INLINE void x86_float16_to_float32_pair(
    __m128i value, __m128& low, __m128& high) {
  low = _mm_cvtph_ps(value);
  high = _mm_cvtph_ps(_mm_srli_si128(value, 8));
}

VECOPS_ALWAYS_INLINE __m128i x86_float32_pair_to_float16(
    __m128 low, __m128 high) {
  const auto low16 = _mm_cvtps_ph(
      low, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const auto high16 = _mm_cvtps_ph(
      high, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return _mm_or_si128(low16, _mm_slli_si128(high16, 8));
}

#if VEC_WIDTH >= 256
VECOPS_ALWAYS_INLINE void x86_float16_to_float32_pair(
    __m256i value, __m256& low, __m256& high) {
  low = _mm256_cvtph_ps(_mm256_castsi256_si128(value));
  high = _mm256_cvtph_ps(_mm256_extractf128_si256(value, 1));
}

VECOPS_ALWAYS_INLINE __m256i x86_float32_pair_to_float16(
    __m256 low, __m256 high) {
  const auto low16 = _mm256_cvtps_ph(
      low, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const auto high16 = _mm256_cvtps_ph(
      high, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return _mm256_insertf128_si256(
      _mm256_castsi128_si256(low16), high16, 1);
}
#endif

#if VEC_WIDTH >= 512
VECOPS_ALWAYS_INLINE void x86_float16_to_float32_pair(
    __m512i value, __m512& low, __m512& high) {
  const auto low256 = _mm512_castsi512_si256(value);
  const auto high256 = _mm512_extracti64x4_epi64(value, 1);
  __m256 part_low;
  __m256 part_high;
  x86_float16_to_float32_pair(low256, part_low, part_high);
  low = _mm512_insertf32x8(
      _mm512_castps256_ps512(part_low), part_high, 1);
  x86_float16_to_float32_pair(high256, part_low, part_high);
  high = _mm512_insertf32x8(
      _mm512_castps256_ps512(part_low), part_high, 1);
}

VECOPS_ALWAYS_INLINE __m512i x86_float32_pair_to_float16(
    __m512 low, __m512 high) {
  const auto low16 = _mm512_cvtps_ph(
      low, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const auto high16 = _mm512_cvtps_ph(
      high, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return _mm512_inserti64x4(
      _mm512_castsi256_si512(low16), high16, 1);
}
#endif
#endif

#if !defined(HAS_F16C)
VECOPS_ALWAYS_INLINE void x86_float16_to_float32_pair(
    __m128i value, __m128& low, __m128& high) {
  alignas(16) float16_t input[8];
  alignas(16) float low_values[4];
  alignas(16) float high_values[4];
  std::memcpy(input, &value, sizeof(value));
  for (int lane = 0; lane < 4; ++lane) {
    low_values[lane] = static_cast<float>(input[lane]);
    high_values[lane] = static_cast<float>(input[lane + 4]);
  }
  std::memcpy(&low, low_values, sizeof(low));
  std::memcpy(&high, high_values, sizeof(high));
}

VECOPS_ALWAYS_INLINE __m128i x86_float32_pair_to_float16(
    __m128 low, __m128 high) {
  alignas(16) float low_values[4];
  alignas(16) float high_values[4];
  alignas(16) float16_t output[8];
  std::memcpy(low_values, &low, sizeof(low));
  std::memcpy(high_values, &high, sizeof(high));
  for (int lane = 0; lane < 4; ++lane) {
    output[lane] = float16_t(low_values[lane]);
    output[lane + 4] = float16_t(high_values[lane]);
  }
  __m128i result;
  std::memcpy(&result, output, sizeof(result));
  return result;
}
#endif

template <typename Op, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_float32_binary(Raw a, Raw b) {
#define VECOPS_VEC_X86_FLOAT32_BINARY(Width)                            \
  if constexpr (std::same_as<Op, AddOp>)                               \
    return _mm##Width##_add_ps(a, b);                                  \
  else if constexpr (std::same_as<Op, SubOp>)                          \
    return _mm##Width##_sub_ps(a, b);                                  \
  else if constexpr (std::same_as<Op, MulOp>)                          \
    return _mm##Width##_mul_ps(a, b);                                  \
  else if constexpr (std::same_as<Op, DivOp>)                          \
    return _mm##Width##_div_ps(a, b);                                  \
  else if constexpr (std::same_as<Op, MinOp>)                          \
    return _mm##Width##_min_ps(a, b);                                  \
  else if constexpr (std::same_as<Op, MaxOp>)                          \
    return _mm##Width##_max_ps(a, b);                                  \
  else                                                                 \
    static_assert(                                                     \
        dispatch_dependent_false<Op>,                                  \
        "unsupported x86 float32 binary operation")
  if constexpr (sizeof(Raw) == 16) {
    VECOPS_VEC_X86_FLOAT32_BINARY();
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    VECOPS_VEC_X86_FLOAT32_BINARY(256);
  }
#endif
#if VEC_WIDTH >= 512
  else {
    VECOPS_VEC_X86_FLOAT32_BINARY(512);
  }
#endif
#undef VECOPS_VEC_X86_FLOAT32_BINARY
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_canonicalize_float32_nan(Raw value) {
  if constexpr (sizeof(Raw) == 16) {
    const auto nan = _mm_cmpunord_ps(value, value);
    const auto canonical =
        _mm_castsi128_ps(_mm_set1_epi32(0x7fc00000));
    return _mm_blendv_ps(value, canonical, nan);
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    const auto nan = _mm256_cmp_ps(value, value, _CMP_UNORD_Q);
    const auto canonical =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));
    return _mm256_blendv_ps(value, canonical, nan);
  }
#endif
#if VEC_WIDTH >= 512
  else {
    const auto nan = _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q);
    const auto canonical =
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000));
    return _mm512_mask_blend_ps(nan, value, canonical);
  }
#endif
}

template <Element T, typename Op, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_small_float_binary_simd(
    Raw a, Raw b) {
  static_assert(
      std::same_as<T, float16_t> || std::same_as<T, bfloat16_t>);
  const auto expand = []<typename Float>(
                          Raw value, Float& low, Float& high) {
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(value, low, high);
    else
      x86_float16_to_float32_pair(value, low, high);
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

  if constexpr (sizeof(Raw) == 16) {
    __m128 a_low, a_high, b_low, b_high;
    expand(a, a_low, a_high);
    expand(b, b_low, b_high);
    return narrow(
        x86_float32_binary<Op>(a_low, b_low),
        x86_float32_binary<Op>(a_high, b_high));
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    __m256 a_low, a_high, b_low, b_high;
    expand(a, a_low, a_high);
    expand(b, b_low, b_high);
    return narrow(
        x86_float32_binary<Op>(a_low, b_low),
        x86_float32_binary<Op>(a_high, b_high));
  }
#endif
#if VEC_WIDTH >= 512
  else {
    __m512 a_low, a_high, b_low, b_high;
    expand(a, a_low, a_high);
    expand(b, b_low, b_high);
    return narrow(
        x86_float32_binary<Op>(a_low, b_low),
        x86_float32_binary<Op>(a_high, b_high));
  }
#endif
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_mullo_epi8(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16) {
    const auto even = _mm_mullo_epi16(a, b);
    const auto odd = _mm_mullo_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8));
    return _mm_or_si128(
        _mm_slli_epi16(odd, 8), _mm_and_si128(even, _mm_set1_epi16(0xff)));
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    const auto even = _mm256_mullo_epi16(a, b);
    const auto odd = _mm256_mullo_epi16(
        _mm256_srli_epi16(a, 8), _mm256_srli_epi16(b, 8));
    return _mm256_or_si256(
        _mm256_slli_epi16(odd, 8),
        _mm256_and_si256(even, _mm256_set1_epi16(0xff)));
  }
#endif
#if VEC_WIDTH >= 512
  else {
    const auto even = _mm512_mullo_epi16(a, b);
    const auto odd = _mm512_mullo_epi16(
        _mm512_srli_epi16(a, 8), _mm512_srli_epi16(b, 8));
    return _mm512_or_si512(
        _mm512_slli_epi16(odd, 8),
        _mm512_and_si512(even, _mm512_set1_epi16(0xff)));
  }
#endif
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_mullo_epi64_fallback(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16) {
    const auto lo_lo = _mm_mul_epu32(a, b);
    const auto a_hi = _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 1));
    const auto b_hi = _mm_shuffle_epi32(b, _MM_SHUFFLE(3, 3, 1, 1));
    const auto cross = _mm_add_epi64(
        _mm_mul_epu32(a_hi, b), _mm_mul_epu32(a, b_hi));
    return _mm_add_epi64(lo_lo, _mm_slli_epi64(cross, 32));
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    const auto lo_lo = _mm256_mul_epu32(a, b);
    const auto a_hi = _mm256_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 1));
    const auto b_hi = _mm256_shuffle_epi32(b, _MM_SHUFFLE(3, 3, 1, 1));
    const auto cross = _mm256_add_epi64(
        _mm256_mul_epu32(a_hi, b), _mm256_mul_epu32(a, b_hi));
    return _mm256_add_epi64(lo_lo, _mm256_slli_epi64(cross, 32));
  }
#endif
#if VEC_WIDTH >= 512
  else {
    const auto lo_lo = _mm512_mul_epu32(a, b);
    const auto a_hi = _mm512_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 1));
    const auto b_hi = _mm512_shuffle_epi32(b, _MM_SHUFFLE(3, 3, 1, 1));
    const auto cross = _mm512_add_epi64(
        _mm512_mul_epu32(a_hi, b), _mm512_mul_epu32(a, b_hi));
    return _mm512_add_epi64(lo_lo, _mm512_slli_epi64(cross, 32));
  }
#endif
}

/* **************************************************************************** */
//                   Binary arithmetic word implementation                    //
/* **************************************************************************** */

template <typename Op>
struct X86ArithmeticWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = decltype(a.value);
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto result = [&]() {
      if constexpr (std::same_as<T, bfloat16_t>) {
        return x86_small_float_binary_simd<T, Op>(a.value, b.value);
      } else if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
        if constexpr (sizeof(Raw) == 16) {
          if constexpr (std::same_as<Op, AddOp>) return _mm_castph_si128(_mm_add_ph(_mm_castsi128_ph(a.value), _mm_castsi128_ph(b.value)));
          else if constexpr (std::same_as<Op, SubOp>) return _mm_castph_si128(_mm_sub_ph(_mm_castsi128_ph(a.value), _mm_castsi128_ph(b.value)));
          else if constexpr (std::same_as<Op, MulOp>) return _mm_castph_si128(_mm_mul_ph(_mm_castsi128_ph(a.value), _mm_castsi128_ph(b.value)));
          else if constexpr (std::same_as<Op, DivOp>) return _mm_castph_si128(_mm_div_ph(_mm_castsi128_ph(a.value), _mm_castsi128_ph(b.value)));
          else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation");
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          if constexpr (std::same_as<Op, AddOp>) return _mm256_castph_si256(_mm256_add_ph(_mm256_castsi256_ph(a.value), _mm256_castsi256_ph(b.value)));
          else if constexpr (std::same_as<Op, SubOp>) return _mm256_castph_si256(_mm256_sub_ph(_mm256_castsi256_ph(a.value), _mm256_castsi256_ph(b.value)));
          else if constexpr (std::same_as<Op, MulOp>) return _mm256_castph_si256(_mm256_mul_ph(_mm256_castsi256_ph(a.value), _mm256_castsi256_ph(b.value)));
          else if constexpr (std::same_as<Op, DivOp>) return _mm256_castph_si256(_mm256_div_ph(_mm256_castsi256_ph(a.value), _mm256_castsi256_ph(b.value)));
          else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation");
        }
#endif
#if VEC_WIDTH >= 512
        else {
          if constexpr (std::same_as<Op, AddOp>) return _mm512_castph_si512(_mm512_add_ph(_mm512_castsi512_ph(a.value), _mm512_castsi512_ph(b.value)));
          else if constexpr (std::same_as<Op, SubOp>) return _mm512_castph_si512(_mm512_sub_ph(_mm512_castsi512_ph(a.value), _mm512_castsi512_ph(b.value)));
          else if constexpr (std::same_as<Op, MulOp>) return _mm512_castph_si512(_mm512_mul_ph(_mm512_castsi512_ph(a.value), _mm512_castsi512_ph(b.value)));
          else if constexpr (std::same_as<Op, DivOp>) return _mm512_castph_si512(_mm512_div_ph(_mm512_castsi512_ph(a.value), _mm512_castsi512_ph(b.value)));
          else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation");
        }
#endif
#elif defined(HAS_F16C)
        return x86_small_float_binary_simd<T, Op>(a.value, b.value);
#else
        return x86_emulated_small_float_binary<T>(
            a.value, b.value, [](T x, T y) {
              if constexpr (std::same_as<Op, AddOp>) return x + y;
              else if constexpr (std::same_as<Op, SubOp>) return x - y;
              else if constexpr (std::same_as<Op, MulOp>) return x * y;
              else if constexpr (std::same_as<Op, DivOp>) return x / y;
              else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation");
            });
#endif
      } else if constexpr (std::same_as<T, float32_t>) {
#define VECOPS_VEC_X86_FLOAT_BINARY(Width, Suffix)                       \
        if constexpr (std::same_as<Op, AddOp>) return _mm##Width##_add_##Suffix(a.value, b.value); \
        else if constexpr (std::same_as<Op, SubOp>) return _mm##Width##_sub_##Suffix(a.value, b.value); \
        else if constexpr (std::same_as<Op, MulOp>) return _mm##Width##_mul_##Suffix(a.value, b.value); \
        else if constexpr (std::same_as<Op, DivOp>) return _mm##Width##_div_##Suffix(a.value, b.value); \
        else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation")
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_FLOAT_BINARY(, ps);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_FLOAT_BINARY(256, ps);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_FLOAT_BINARY(512, ps);
#endif
#undef VECOPS_VEC_X86_FLOAT_BINARY
      } else if constexpr (std::same_as<T, float64_t>) {
#define VECOPS_VEC_X86_FLOAT_BINARY(Width, Suffix)                       \
        if constexpr (std::same_as<Op, AddOp>) return _mm##Width##_add_##Suffix(a.value, b.value); \
        else if constexpr (std::same_as<Op, SubOp>) return _mm##Width##_sub_##Suffix(a.value, b.value); \
        else if constexpr (std::same_as<Op, MulOp>) return _mm##Width##_mul_##Suffix(a.value, b.value); \
        else if constexpr (std::same_as<Op, DivOp>) return _mm##Width##_div_##Suffix(a.value, b.value); \
        else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation")
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_FLOAT_BINARY(, pd);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_FLOAT_BINARY(256, pd);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_FLOAT_BINARY(512, pd);
#endif
#undef VECOPS_VEC_X86_FLOAT_BINARY
      } else if constexpr (std::same_as<T, int8_t> || std::same_as<T, uint8_t>) {
        if constexpr (std::same_as<Op, MulOp>) return x86_mullo_epi8(a.value, b.value);
#define VECOPS_VEC_X86_INT_BINARY(Width, Bits)                           \
        if constexpr (std::same_as<Op, AddOp>) return _mm##Width##_add_epi##Bits(a.value, b.value); \
        else if constexpr (std::same_as<Op, SubOp>) return _mm##Width##_sub_epi##Bits(a.value, b.value); \
        else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation")
        else if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_INT_BINARY(, 8);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_INT_BINARY(256, 8);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_INT_BINARY(512, 8);
#endif
#undef VECOPS_VEC_X86_INT_BINARY
      } else if constexpr (std::same_as<T, int16_t> || std::same_as<T, uint16_t>) {
#define VECOPS_VEC_X86_INT16_BINARY(Width)                               \
        if constexpr (std::same_as<Op, AddOp>) return _mm##Width##_add_epi16(a.value, b.value); \
        else if constexpr (std::same_as<Op, SubOp>) return _mm##Width##_sub_epi16(a.value, b.value); \
        else if constexpr (std::same_as<Op, MulOp>) return _mm##Width##_mullo_epi16(a.value, b.value); \
        else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation")
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_INT16_BINARY();
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_INT16_BINARY(256);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_INT16_BINARY(512);
#endif
#undef VECOPS_VEC_X86_INT16_BINARY
      } else if constexpr (std::same_as<T, int32_t> || std::same_as<T, uint32_t>) {
#define VECOPS_VEC_X86_INT32_BINARY(Width)                               \
        if constexpr (std::same_as<Op, AddOp>) return _mm##Width##_add_epi32(a.value, b.value); \
        else if constexpr (std::same_as<Op, SubOp>) return _mm##Width##_sub_epi32(a.value, b.value); \
        else if constexpr (std::same_as<Op, MulOp>) return _mm##Width##_mullo_epi32(a.value, b.value); \
        else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation")
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_INT32_BINARY();
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_INT32_BINARY(256);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_INT32_BINARY(512);
#endif
#undef VECOPS_VEC_X86_INT32_BINARY
      } else if constexpr (std::same_as<T, int64_t> || std::same_as<T, uint64_t>) {
        if constexpr (std::same_as<Op, MulOp>) {
#if defined(HAS_AVX512DQ)
          if constexpr (sizeof(Raw) == 16) return _mm_mullo_epi64(a.value, b.value);
#if VEC_WIDTH >= 256
          else if constexpr (sizeof(Raw) == 32) return _mm256_mullo_epi64(a.value, b.value);
#endif
#if VEC_WIDTH >= 512
          else return _mm512_mullo_epi64(a.value, b.value);
#endif
#else
          return x86_mullo_epi64_fallback(a.value, b.value);
#endif
        }
#define VECOPS_VEC_X86_INT64_BINARY(Width)                               \
        if constexpr (std::same_as<Op, AddOp>) return _mm##Width##_add_epi64(a.value, b.value); \
        else if constexpr (std::same_as<Op, SubOp>) return _mm##Width##_sub_epi64(a.value, b.value); \
        else static_assert(dispatch_dependent_false<Op>, "unsupported x86 arithmetic operation")
        else if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_INT64_BINARY();
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_INT64_BINARY(256);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_INT64_BINARY(512);
#endif
#undef VECOPS_VEC_X86_INT64_BINARY
      } else {
        static_assert(
            dispatch_dependent_false<T>,
            "unsupported x86 arithmetic element type");
      }
    }();
    return NativeWordVec<Tag>{result};
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    using T = ElementOf<Tag>;
    using Raw = decltype(a.value);
#if defined(CPU_CAPABILITY_AVX512)
    const auto masked = [&]() {
#define VECOPS_VEC_X86_MASK_BINARY(Width, Name, Suffix)                  \
      if constexpr (std::same_as<Op, AddOp>) {                            \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)      \
          return _mm##Width##_maskz_add_##Suffix(mask.value, a.value, b.value); \
        else return _mm##Width##_mask_add_##Suffix(inactive.value, mask.value, a.value, b.value); \
      } else if constexpr (std::same_as<Op, SubOp>) {                     \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)      \
          return _mm##Width##_maskz_sub_##Suffix(mask.value, a.value, b.value); \
        else return _mm##Width##_mask_sub_##Suffix(inactive.value, mask.value, a.value, b.value); \
      } else if constexpr (std::same_as<Op, MulOp>) {                     \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)      \
          return _mm##Width##_maskz_##Name##_##Suffix(mask.value, a.value, b.value); \
        else return _mm##Width##_mask_##Name##_##Suffix(inactive.value, mask.value, a.value, b.value); \
      } else if constexpr (std::same_as<Op, DivOp>) {                     \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)      \
          return _mm##Width##_maskz_div_##Suffix(mask.value, a.value, b.value); \
        else return _mm##Width##_mask_div_##Suffix(inactive.value, mask.value, a.value, b.value); \
      }                                                                  \
      else static_assert(dispatch_dependent_false<Op>, "unsupported x86 masked arithmetic operation")
      if constexpr (std::same_as<T, float32_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASK_BINARY(, mul, ps);
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASK_BINARY(256, mul, ps);
        else VECOPS_VEC_X86_MASK_BINARY(512, mul, ps);
      } else if constexpr (std::same_as<T, float64_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASK_BINARY(, mul, pd);
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASK_BINARY(256, mul, pd);
        else VECOPS_VEC_X86_MASK_BINARY(512, mul, pd);
      } else if constexpr (std::same_as<T, bfloat16_t>) {
        return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(
            BlendOp{}, tag, inactive, mask, call<Index>(op, tag, a, b)).value;
      } else if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
        if constexpr (sizeof(Raw) == 16) {
          const auto raw_a = _mm_castsi128_ph(a.value);
          const auto raw_b = _mm_castsi128_ph(b.value);
          if constexpr (std::same_as<Op, AddOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm_castph_si128(_mm_maskz_add_ph(mask.value, raw_a, raw_b));
            else return _mm_castph_si128(_mm_mask_add_ph(
                _mm_castsi128_ph(inactive.value), mask.value, raw_a, raw_b));
          } else if constexpr (std::same_as<Op, SubOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm_castph_si128(_mm_maskz_sub_ph(mask.value, raw_a, raw_b));
            else return _mm_castph_si128(_mm_mask_sub_ph(
                _mm_castsi128_ph(inactive.value), mask.value, raw_a, raw_b));
          } else if constexpr (std::same_as<Op, MulOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm_castph_si128(_mm_maskz_mul_ph(mask.value, raw_a, raw_b));
            else return _mm_castph_si128(_mm_mask_mul_ph(
                _mm_castsi128_ph(inactive.value), mask.value, raw_a, raw_b));
          } else if constexpr (std::same_as<Op, DivOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm_castph_si128(_mm_maskz_div_ph(mask.value, raw_a, raw_b));
            else return _mm_castph_si128(_mm_mask_div_ph(
                _mm_castsi128_ph(inactive.value), mask.value, raw_a, raw_b));
          } else {
            static_assert(
                dispatch_dependent_false<Op>,
                "unsupported x86 masked arithmetic operation");
          }
        } else if constexpr (sizeof(Raw) == 32) {
          const auto raw_a = _mm256_castsi256_ph(a.value);
          const auto raw_b = _mm256_castsi256_ph(b.value);
          if constexpr (std::same_as<Op, AddOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm256_castph_si256(_mm256_maskz_add_ph(mask.value, raw_a, raw_b));
            else return _mm256_castph_si256(_mm256_mask_add_ph(
                _mm256_castsi256_ph(inactive.value), mask.value, raw_a, raw_b));
          } else if constexpr (std::same_as<Op, SubOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm256_castph_si256(_mm256_maskz_sub_ph(mask.value, raw_a, raw_b));
            else return _mm256_castph_si256(_mm256_mask_sub_ph(
                _mm256_castsi256_ph(inactive.value), mask.value, raw_a, raw_b));
          } else if constexpr (std::same_as<Op, MulOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm256_castph_si256(_mm256_maskz_mul_ph(mask.value, raw_a, raw_b));
            else return _mm256_castph_si256(_mm256_mask_mul_ph(
                _mm256_castsi256_ph(inactive.value), mask.value, raw_a, raw_b));
          } else if constexpr (std::same_as<Op, DivOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm256_castph_si256(_mm256_maskz_div_ph(mask.value, raw_a, raw_b));
            else return _mm256_castph_si256(_mm256_mask_div_ph(
                _mm256_castsi256_ph(inactive.value), mask.value, raw_a, raw_b));
          } else {
            static_assert(
                dispatch_dependent_false<Op>,
                "unsupported x86 masked arithmetic operation");
          }
        } else {
          const auto raw_a = _mm512_castsi512_ph(a.value);
          const auto raw_b = _mm512_castsi512_ph(b.value);
          if constexpr (std::same_as<Op, AddOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm512_castph_si512(_mm512_maskz_add_ph(mask.value, raw_a, raw_b));
            else return _mm512_castph_si512(_mm512_mask_add_ph(
                _mm512_castsi512_ph(inactive.value), mask.value, raw_a, raw_b));
          } else if constexpr (std::same_as<Op, SubOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm512_castph_si512(_mm512_maskz_sub_ph(mask.value, raw_a, raw_b));
            else return _mm512_castph_si512(_mm512_mask_sub_ph(
                _mm512_castsi512_ph(inactive.value), mask.value, raw_a, raw_b));
          } else if constexpr (std::same_as<Op, MulOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm512_castph_si512(_mm512_maskz_mul_ph(mask.value, raw_a, raw_b));
            else return _mm512_castph_si512(_mm512_mask_mul_ph(
                _mm512_castsi512_ph(inactive.value), mask.value, raw_a, raw_b));
          } else if constexpr (std::same_as<Op, DivOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)
              return _mm512_castph_si512(_mm512_maskz_div_ph(mask.value, raw_a, raw_b));
            else return _mm512_castph_si512(_mm512_mask_div_ph(
                _mm512_castsi512_ph(inactive.value), mask.value, raw_a, raw_b));
          } else {
            static_assert(
                dispatch_dependent_false<Op>,
                "unsupported x86 masked arithmetic operation");
          }
        }
#else
        return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(
            BlendOp{}, tag, inactive, mask, call<Index>(op, tag, a, b)).value;
#endif
      } else if constexpr (std::same_as<T, int8_t> || std::same_as<T, uint8_t>) {
        if constexpr (std::same_as<Op, MulOp>) {
          return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(
              BlendOp{}, tag, inactive, mask, call<Index>(op, tag, a, b)).value;
        } else if constexpr (sizeof(Raw) == 16) {
          if constexpr (std::same_as<Op, AddOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return _mm_maskz_add_epi8(mask.value, a.value, b.value);
            else return _mm_mask_add_epi8(inactive.value, mask.value, a.value, b.value);
          } else if constexpr (std::same_as<Op, SubOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return _mm_maskz_sub_epi8(mask.value, a.value, b.value);
            else return _mm_mask_sub_epi8(inactive.value, mask.value, a.value, b.value);
          }
          else static_assert(dispatch_dependent_false<Op>, "unsupported x86 masked arithmetic operation");
        } else if constexpr (sizeof(Raw) == 32) {
          if constexpr (std::same_as<Op, AddOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return _mm256_maskz_add_epi8(mask.value, a.value, b.value);
            else return _mm256_mask_add_epi8(inactive.value, mask.value, a.value, b.value);
          } else if constexpr (std::same_as<Op, SubOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return _mm256_maskz_sub_epi8(mask.value, a.value, b.value);
            else return _mm256_mask_sub_epi8(inactive.value, mask.value, a.value, b.value);
          }
          else static_assert(dispatch_dependent_false<Op>, "unsupported x86 masked arithmetic operation");
        } else {
          if constexpr (std::same_as<Op, AddOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return _mm512_maskz_add_epi8(mask.value, a.value, b.value);
            else return _mm512_mask_add_epi8(inactive.value, mask.value, a.value, b.value);
          } else if constexpr (std::same_as<Op, SubOp>) {
            if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return _mm512_maskz_sub_epi8(mask.value, a.value, b.value);
            else return _mm512_mask_sub_epi8(inactive.value, mask.value, a.value, b.value);
          }
          else static_assert(dispatch_dependent_false<Op>, "unsupported x86 masked arithmetic operation");
        }
      } else if constexpr (std::same_as<T, int16_t> || std::same_as<T, uint16_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASK_BINARY(, mullo, epi16);
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASK_BINARY(256, mullo, epi16);
        else VECOPS_VEC_X86_MASK_BINARY(512, mullo, epi16);
      } else if constexpr (std::same_as<T, int32_t> || std::same_as<T, uint32_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASK_BINARY(, mullo, epi32);
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASK_BINARY(256, mullo, epi32);
        else VECOPS_VEC_X86_MASK_BINARY(512, mullo, epi32);
      } else if constexpr (std::same_as<T, int64_t> || std::same_as<T, uint64_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASK_BINARY(, mullo, epi64);
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASK_BINARY(256, mullo, epi64);
        else VECOPS_VEC_X86_MASK_BINARY(512, mullo, epi64);
      } else {
        static_assert(dispatch_dependent_false<T>, "unsupported x86 masked arithmetic element type");
      }
#undef VECOPS_VEC_X86_MASK_BINARY
    }();
    return NativeWordVec<Tag>{masked};
#else
    return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, call<Index>(op, tag, a, b));
#endif
  }
};

template <>
struct NativeWordImpl<X86Backend, AddOp> : X86ArithmeticWordImpl<AddOp> {};
template <>
struct NativeWordImpl<X86Backend, SubOp> : X86ArithmeticWordImpl<SubOp> {};
template <>
struct NativeWordImpl<X86Backend, MulOp> : X86ArithmeticWordImpl<MulOp> {};
template <>
struct NativeWordImpl<X86Backend, DivOp> : X86ArithmeticWordImpl<DivOp> {};

/* **************************************************************************** */
//                   Extrema helpers and word implementation                   //
/* **************************************************************************** */

template <Element T, typename Raw, typename Op>
VECOPS_ALWAYS_INLINE Raw x86_emulated_extrema(Raw a, Raw b, Op) {
  constexpr std::size_t lanes = sizeof(Raw) / sizeof(T);
  alignas(64) T a_lanes[lanes];
  alignas(64) T b_lanes[lanes];
  alignas(64) T result_lanes[lanes];
  std::memcpy(a_lanes, &a, sizeof(Raw));
  std::memcpy(b_lanes, &b, sizeof(Raw));
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    if constexpr (std::same_as<Op, MinOp>)
      result_lanes[lane] = a_lanes[lane] < b_lanes[lane]
          ? a_lanes[lane] : b_lanes[lane];
    else if constexpr (std::same_as<Op, MaxOp>)
      result_lanes[lane] = a_lanes[lane] > b_lanes[lane]
          ? a_lanes[lane] : b_lanes[lane];
    else
      static_assert(dispatch_dependent_false<Op>);
  }
  Raw result;
  std::memcpy(&result, result_lanes, sizeof(Raw));
  return result;
}

template <Element T, typename Raw, typename Op>
VECOPS_ALWAYS_INLINE Raw x86_extrema_epi64_fallback(Raw a, Raw b, Op) {
  static_assert(std::same_as<T, int64_t> || std::same_as<T, uint64_t>);
  if constexpr (sizeof(Raw) == 16) {
    auto compare_a = a;
    auto compare_b = b;
    if constexpr (std::same_as<T, uint64_t>) {
      const auto sign = _mm_set1_epi64x(std::numeric_limits<int64_t>::min());
      compare_a = _mm_xor_si128(compare_a, sign);
      compare_b = _mm_xor_si128(compare_b, sign);
    }
    const auto select_a = [&] {
      if constexpr (std::same_as<Op, MinOp>)
        return _mm_cmpgt_epi64(compare_b, compare_a);
      else if constexpr (std::same_as<Op, MaxOp>)
        return _mm_cmpgt_epi64(compare_a, compare_b);
      else
        static_assert(dispatch_dependent_false<Op>);
    }();
    return _mm_blendv_epi8(b, a, select_a);
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    auto compare_a = a;
    auto compare_b = b;
    if constexpr (std::same_as<T, uint64_t>) {
      const auto sign =
          _mm256_set1_epi64x(std::numeric_limits<int64_t>::min());
      compare_a = _mm256_xor_si256(compare_a, sign);
      compare_b = _mm256_xor_si256(compare_b, sign);
    }
    const auto select_a = [&] {
      if constexpr (std::same_as<Op, MinOp>)
        return _mm256_cmpgt_epi64(compare_b, compare_a);
      else if constexpr (std::same_as<Op, MaxOp>)
        return _mm256_cmpgt_epi64(compare_a, compare_b);
      else
        static_assert(dispatch_dependent_false<Op>);
    }();
    return _mm256_blendv_epi8(b, a, select_a);
  }
#endif
  else {
    static_assert(dispatch_dependent_false<Raw>);
  }
}

template <typename Op>
struct X86ExtremaWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using T = ElementOf<Tag>;
    using Raw = decltype(a.value);
    static_assert(Index >= 0 && Index < num_words(tag));
    const auto result = [&]() {
#define VECOPS_VEC_X86_EXTREMA(Width, Suffix)                           \
      if constexpr (std::same_as<Op, MinOp>)                            \
        return _mm##Width##_min_##Suffix(a.value, b.value);             \
      else if constexpr (std::same_as<Op, MaxOp>)                       \
        return _mm##Width##_max_##Suffix(a.value, b.value);             \
      else static_assert(dispatch_dependent_false<Op>)
      if constexpr (std::same_as<T, bfloat16_t>) {
        return x86_small_float_binary_simd<T, Op>(a.value, b.value);
      } else if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
        if constexpr (sizeof(Raw) == 16) {
          if constexpr (std::same_as<Op, MinOp>) return _mm_castph_si128(
              _mm_min_ph(_mm_castsi128_ph(a.value), _mm_castsi128_ph(b.value)));
          else return _mm_castph_si128(
              _mm_max_ph(_mm_castsi128_ph(a.value), _mm_castsi128_ph(b.value)));
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          if constexpr (std::same_as<Op, MinOp>) return _mm256_castph_si256(
              _mm256_min_ph(
                  _mm256_castsi256_ph(a.value), _mm256_castsi256_ph(b.value)));
          else return _mm256_castph_si256(_mm256_max_ph(
              _mm256_castsi256_ph(a.value), _mm256_castsi256_ph(b.value)));
        }
#endif
#if VEC_WIDTH >= 512
        else {
          if constexpr (std::same_as<Op, MinOp>) return _mm512_castph_si512(
              _mm512_min_ph(
                  _mm512_castsi512_ph(a.value), _mm512_castsi512_ph(b.value)));
          else return _mm512_castph_si512(_mm512_max_ph(
              _mm512_castsi512_ph(a.value), _mm512_castsi512_ph(b.value)));
        }
#endif
#elif defined(HAS_F16C)
        return x86_small_float_binary_simd<T, Op>(a.value, b.value);
#else
        return x86_emulated_extrema<T>(a.value, b.value, Op{});
#endif
      } else if constexpr (std::same_as<T, float32_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, ps);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, ps);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_EXTREMA(512, ps);
#endif
      } else if constexpr (std::same_as<T, float64_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, pd);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, pd);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_EXTREMA(512, pd);
#endif
      } else if constexpr (std::same_as<T, int8_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, epi8);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, epi8);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_EXTREMA(512, epi8);
#endif
      } else if constexpr (std::same_as<T, uint8_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, epu8);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, epu8);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_EXTREMA(512, epu8);
#endif
      } else if constexpr (std::same_as<T, int16_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, epi16);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, epi16);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_EXTREMA(512, epi16);
#endif
      } else if constexpr (std::same_as<T, uint16_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, epu16);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, epu16);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_EXTREMA(512, epu16);
#endif
      } else if constexpr (std::same_as<T, int32_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, epi32);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, epi32);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_EXTREMA(512, epi32);
#endif
      } else if constexpr (std::same_as<T, uint32_t>) {
        if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, epu32);
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, epu32);
#endif
#if VEC_WIDTH >= 512
        else VECOPS_VEC_X86_EXTREMA(512, epu32);
#endif
      } else if constexpr (
          std::same_as<T, int64_t> || std::same_as<T, uint64_t>) {
#if defined(HAS_AVX512DQ)
        if constexpr (std::same_as<T, int64_t>) {
          if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, epi64);
#if VEC_WIDTH >= 256
          else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, epi64);
#endif
#if VEC_WIDTH >= 512
          else VECOPS_VEC_X86_EXTREMA(512, epi64);
#endif
        } else {
          if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_EXTREMA(, epu64);
#if VEC_WIDTH >= 256
          else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_EXTREMA(256, epu64);
#endif
#if VEC_WIDTH >= 512
          else VECOPS_VEC_X86_EXTREMA(512, epu64);
#endif
        }
#else
        return x86_extrema_epi64_fallback<T>(a.value, b.value, Op{});
#endif
      } else {
        static_assert(dispatch_dependent_false<T>);
      }
#undef VECOPS_VEC_X86_EXTREMA
    }();
    return NativeWordVec<Tag>{result};
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
#if defined(CPU_CAPABILITY_AVX512)
    using T = ElementOf<Tag>;
    using Raw = decltype(a.value);
#define VECOPS_VEC_X86_MASKED_EXTREMA(Width, Suffix)                    \
    if constexpr (std::same_as<Op, MinOp>) {                            \
      if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)      \
        return NativeWordVec<Tag>{_mm##Width##_maskz_min_##Suffix(      \
            mask.value, a.value, b.value)};                            \
      else return NativeWordVec<Tag>{_mm##Width##_mask_min_##Suffix(   \
          inactive.value, mask.value, a.value, b.value)};              \
    } else if constexpr (std::same_as<Op, MaxOp>) {                     \
      if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)      \
        return NativeWordVec<Tag>{_mm##Width##_maskz_max_##Suffix(      \
            mask.value, a.value, b.value)};                            \
      else return NativeWordVec<Tag>{_mm##Width##_mask_max_##Suffix(   \
          inactive.value, mask.value, a.value, b.value)};              \
    } else static_assert(dispatch_dependent_false<Op>)
    if constexpr (std::same_as<T, float32_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, ps);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, ps);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, ps);
    } else if constexpr (std::same_as<T, float64_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, pd);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, pd);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, pd);
    } else if constexpr (std::same_as<T, int8_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, epi8);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, epi8);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, epi8);
    } else if constexpr (std::same_as<T, uint8_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, epu8);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, epu8);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, epu8);
    } else if constexpr (std::same_as<T, int16_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, epi16);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, epi16);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, epi16);
    } else if constexpr (std::same_as<T, uint16_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, epu16);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, epu16);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, epu16);
    } else if constexpr (std::same_as<T, int32_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, epi32);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, epi32);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, epi32);
    } else if constexpr (std::same_as<T, uint32_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, epu32);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, epu32);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, epu32);
    } else if constexpr (std::same_as<T, int64_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, epi64);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, epi64);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, epi64);
    } else if constexpr (std::same_as<T, uint64_t>) {
      if constexpr (sizeof(Raw) == 16) VECOPS_VEC_X86_MASKED_EXTREMA(, epu64);
      else if constexpr (sizeof(Raw) == 32) VECOPS_VEC_X86_MASKED_EXTREMA(256, epu64);
      else VECOPS_VEC_X86_MASKED_EXTREMA(512, epu64);
    }
#if defined(HAS_AVX512_FP16)
    else if constexpr (std::same_as<T, float16_t>) {
#define VECOPS_VEC_X86_MASKED_EXTREMA_PH(Width, CastTo, CastFrom)       \
      if constexpr (std::same_as<Op, MinOp>) {                          \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)    \
          return NativeWordVec<Tag>{CastFrom(_mm##Width##_maskz_min_ph(\
              mask.value, CastTo(a.value), CastTo(b.value)))};         \
        else return NativeWordVec<Tag>{CastFrom(_mm##Width##_mask_min_ph(\
            CastTo(inactive.value), mask.value,                        \
            CastTo(a.value), CastTo(b.value)))};                       \
      } else {                                                          \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)    \
          return NativeWordVec<Tag>{CastFrom(_mm##Width##_maskz_max_ph(\
              mask.value, CastTo(a.value), CastTo(b.value)))};         \
        else return NativeWordVec<Tag>{CastFrom(_mm##Width##_mask_max_ph(\
            CastTo(inactive.value), mask.value,                        \
            CastTo(a.value), CastTo(b.value)))};                       \
      }
      if constexpr (sizeof(Raw) == 16) {
        VECOPS_VEC_X86_MASKED_EXTREMA_PH(
            , _mm_castsi128_ph, _mm_castph_si128);
      } else if constexpr (sizeof(Raw) == 32) {
        VECOPS_VEC_X86_MASKED_EXTREMA_PH(
            256, _mm256_castsi256_ph, _mm256_castph_si256);
      } else {
        VECOPS_VEC_X86_MASKED_EXTREMA_PH(
            512, _mm512_castsi512_ph, _mm512_castph_si512);
      }
#undef VECOPS_VEC_X86_MASKED_EXTREMA_PH
    }
#endif
#undef VECOPS_VEC_X86_MASKED_EXTREMA
#endif
    const auto computed = call<Index>(op, tag, a, b);
    return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, computed);
  }
};

template <>
struct NativeWordImpl<X86Backend, MinOp> : X86ExtremaWordImpl<MinOp> {};
template <>
struct NativeWordImpl<X86Backend, MaxOp> : X86ExtremaWordImpl<MaxOp> {};

/* **************************************************************************** */
//                    Unary arithmetic word implementation                    //
/* **************************************************************************** */

template <typename Op>
struct X86UnaryArithmeticWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = decltype(value.value);
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto result = [&]() {
      if constexpr (
          std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>) {
        if constexpr (sizeof(Raw) == 16) {
          if constexpr (std::same_as<Op, NegOp>)
            return _mm_xor_si128(value.value, _mm_set1_epi16(
                static_cast<short>(0x8000u)));
          else
            return _mm_and_si128(value.value, _mm_set1_epi16(0x7fff));
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          if constexpr (std::same_as<Op, NegOp>)
            return _mm256_xor_si256(value.value, _mm256_set1_epi16(
                static_cast<short>(0x8000u)));
          else
            return _mm256_and_si256(value.value, _mm256_set1_epi16(0x7fff));
        }
#endif
#if VEC_WIDTH >= 512
        else {
          if constexpr (std::same_as<Op, NegOp>)
            return _mm512_xor_si512(value.value, _mm512_set1_epi16(
                static_cast<short>(0x8000u)));
          else
            return _mm512_and_si512(value.value, _mm512_set1_epi16(0x7fff));
        }
#endif
      } else if constexpr (std::same_as<T, float32_t>) {
        if constexpr (sizeof(Raw) == 16) {
          if constexpr (std::same_as<Op, NegOp>)
            return _mm_xor_ps(value.value, _mm_set1_ps(-0.0F));
          else
            return _mm_andnot_ps(_mm_set1_ps(-0.0F), value.value);
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          if constexpr (std::same_as<Op, NegOp>)
            return _mm256_xor_ps(value.value, _mm256_set1_ps(-0.0F));
          else
            return _mm256_andnot_ps(_mm256_set1_ps(-0.0F), value.value);
        }
#endif
#if VEC_WIDTH >= 512
        else {
          if constexpr (std::same_as<Op, NegOp>)
            return _mm512_xor_ps(value.value, _mm512_set1_ps(-0.0F));
          else
            return _mm512_andnot_ps(_mm512_set1_ps(-0.0F), value.value);
        }
#endif
      } else if constexpr (std::same_as<T, float64_t>) {
        if constexpr (sizeof(Raw) == 16) {
          if constexpr (std::same_as<Op, NegOp>)
            return _mm_xor_pd(value.value, _mm_set1_pd(-0.0));
          else
            return _mm_andnot_pd(_mm_set1_pd(-0.0), value.value);
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          if constexpr (std::same_as<Op, NegOp>)
            return _mm256_xor_pd(value.value, _mm256_set1_pd(-0.0));
          else
            return _mm256_andnot_pd(_mm256_set1_pd(-0.0), value.value);
        }
#endif
#if VEC_WIDTH >= 512
        else {
          if constexpr (std::same_as<Op, NegOp>)
            return _mm512_xor_pd(value.value, _mm512_set1_pd(-0.0));
          else
            return _mm512_andnot_pd(_mm512_set1_pd(-0.0), value.value);
        }
#endif
      } else if constexpr (std::is_unsigned_v<T> && std::same_as<Op, AbsOp>) {
        return value.value;
      } else if constexpr (sizeof(Raw) == 16) {
#define VECOPS_VEC_X86_UNARY_INT128(Bits)                              \
        if constexpr (std::same_as<Op, NegOp>)                         \
          return _mm_sub_epi##Bits(_mm_setzero_si128(), value.value);  \
        else if constexpr (Bits == 64) {                               \
          const auto sign = _mm_cmpgt_epi64(_mm_setzero_si128(), value.value); \
          return _mm_sub_epi64(_mm_xor_si128(value.value, sign), sign);\
        } else                                                          \
          return _mm_abs_epi##Bits(value.value)
        if constexpr (sizeof(T) == 1) VECOPS_VEC_X86_UNARY_INT128(8);
        else if constexpr (sizeof(T) == 2) VECOPS_VEC_X86_UNARY_INT128(16);
        else if constexpr (sizeof(T) == 4) VECOPS_VEC_X86_UNARY_INT128(32);
        else VECOPS_VEC_X86_UNARY_INT128(64);
#undef VECOPS_VEC_X86_UNARY_INT128
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32) {
#define VECOPS_VEC_X86_UNARY_INT256(Bits)                              \
        if constexpr (std::same_as<Op, NegOp>)                         \
          return _mm256_sub_epi##Bits(                                 \
              _mm256_setzero_si256(), value.value);                    \
        else if constexpr (Bits == 64) {                               \
          const auto sign = _mm256_cmpgt_epi64(                        \
              _mm256_setzero_si256(), value.value);                    \
          return _mm256_sub_epi64(                                     \
              _mm256_xor_si256(value.value, sign), sign);              \
        } else                                                          \
          return _mm256_abs_epi##Bits(value.value)
        if constexpr (sizeof(T) == 1) VECOPS_VEC_X86_UNARY_INT256(8);
        else if constexpr (sizeof(T) == 2) VECOPS_VEC_X86_UNARY_INT256(16);
        else if constexpr (sizeof(T) == 4) VECOPS_VEC_X86_UNARY_INT256(32);
        else VECOPS_VEC_X86_UNARY_INT256(64);
#undef VECOPS_VEC_X86_UNARY_INT256
      }
#endif
#if VEC_WIDTH >= 512
      else {
#define VECOPS_VEC_X86_UNARY_INT512(Bits)                              \
        if constexpr (std::same_as<Op, NegOp>)                         \
          return _mm512_sub_epi##Bits(                                 \
              _mm512_setzero_si512(), value.value);                    \
        else                                                            \
          return _mm512_abs_epi##Bits(value.value)
        if constexpr (sizeof(T) == 1) VECOPS_VEC_X86_UNARY_INT512(8);
        else if constexpr (sizeof(T) == 2) VECOPS_VEC_X86_UNARY_INT512(16);
        else if constexpr (sizeof(T) == 4) VECOPS_VEC_X86_UNARY_INT512(32);
        else VECOPS_VEC_X86_UNARY_INT512(64);
#undef VECOPS_VEC_X86_UNARY_INT512
      }
#endif
    }();
    return NativeWordVec<Tag>{result};
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, call<Index>(op, tag, value));
  }
};

template <>
struct NativeWordImpl<X86Backend, NegOp>
    : X86UnaryArithmeticWordImpl<NegOp> {};

template <>
struct NativeWordImpl<X86Backend, AbsOp>
    : X86UnaryArithmeticWordImpl<AbsOp> {};

/* **************************************************************************** */
//                  Floating-point unary word implementation                  //
/* **************************************************************************** */

template <typename Op>
struct X86FloatingUnaryWordImpl {
  template <typename RawFloat>
  static VECOPS_ALWAYS_INLINE RawFloat compute_float32(RawFloat value) {
    if constexpr (sizeof(RawFloat) == 16) {
      if constexpr (std::same_as<Op, SqrtOp>) return _mm_sqrt_ps(value);
      else if constexpr (std::same_as<Op, RcpOp>) {
#if defined(CPU_CAPABILITY_AVX512)
        return _mm_rcp14_ps(value);
#else
        return _mm_rcp_ps(value);
#endif
      } else if constexpr (std::same_as<Op, RsqrtOp>) {
#if defined(CPU_CAPABILITY_AVX512)
        return _mm_rsqrt14_ps(value);
#else
        return _mm_rsqrt_ps(value);
#endif
      } else static_assert(dispatch_dependent_false<Op>);
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(RawFloat) == 32) {
      if constexpr (std::same_as<Op, SqrtOp>) return _mm256_sqrt_ps(value);
      else if constexpr (std::same_as<Op, RcpOp>) {
#if defined(CPU_CAPABILITY_AVX512)
        return _mm256_rcp14_ps(value);
#else
        return _mm256_rcp_ps(value);
#endif
      } else if constexpr (std::same_as<Op, RsqrtOp>) {
#if defined(CPU_CAPABILITY_AVX512)
        return _mm256_rsqrt14_ps(value);
#else
        return _mm256_rsqrt_ps(value);
#endif
      } else static_assert(dispatch_dependent_false<Op>);
    }
#endif
#if VEC_WIDTH >= 512
    else {
      if constexpr (std::same_as<Op, SqrtOp>) return _mm512_sqrt_ps(value);
      else if constexpr (std::same_as<Op, RcpOp>) return _mm512_rcp14_ps(value);
      else if constexpr (std::same_as<Op, RsqrtOp>) return _mm512_rsqrt14_ps(value);
      else static_assert(dispatch_dependent_false<Op>);
    }
#endif
  }

  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    using T = ElementOf<Tag>;
    using Raw = decltype(value.value);
    static_assert(Index >= 0 && Index < num_words(tag));
    const auto result = [&]() {
      if constexpr (std::same_as<T, bfloat16_t>) {
        if constexpr (sizeof(Raw) == 16) {
          __m128 low;
          __m128 high;
          x86_bfloat16_to_float32_pair(value.value, low, high);
          return x86_float32_pair_to_bfloat16(
              compute_float32(low), compute_float32(high));
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          __m256 low;
          __m256 high;
          x86_bfloat16_to_float32_pair(value.value, low, high);
          return x86_float32_pair_to_bfloat16(
              compute_float32(low), compute_float32(high));
        }
#endif
#if VEC_WIDTH >= 512
        else {
          __m512 low;
          __m512 high;
          x86_bfloat16_to_float32_pair(value.value, low, high);
          return x86_float32_pair_to_bfloat16(
              compute_float32(low), compute_float32(high));
        }
#endif
      } else if constexpr (std::same_as<T, float16_t>) {
#if defined(HAS_AVX512_FP16)
        if constexpr (sizeof(Raw) == 16) {
          const auto raw = _mm_castsi128_ph(value.value);
          if constexpr (std::same_as<Op, SqrtOp>)
            return _mm_castph_si128(_mm_sqrt_ph(raw));
          else if constexpr (std::same_as<Op, RcpOp>)
            return _mm_castph_si128(_mm_rcp_ph(raw));
          else
            return _mm_castph_si128(_mm_rsqrt_ph(raw));
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          const auto raw = _mm256_castsi256_ph(value.value);
          if constexpr (std::same_as<Op, SqrtOp>)
            return _mm256_castph_si256(_mm256_sqrt_ph(raw));
          else if constexpr (std::same_as<Op, RcpOp>)
            return _mm256_castph_si256(_mm256_rcp_ph(raw));
          else
            return _mm256_castph_si256(_mm256_rsqrt_ph(raw));
        }
#endif
#if VEC_WIDTH >= 512
        else {
          const auto raw = _mm512_castsi512_ph(value.value);
          if constexpr (std::same_as<Op, SqrtOp>)
            return _mm512_castph_si512(_mm512_sqrt_ph(raw));
          else if constexpr (std::same_as<Op, RcpOp>)
            return _mm512_castph_si512(_mm512_rcp_ph(raw));
          else
            return _mm512_castph_si512(_mm512_rsqrt_ph(raw));
        }
#endif
#elif defined(HAS_F16C)
        if constexpr (sizeof(Raw) == 16) {
          __m128 low;
          __m128 high;
          x86_float16_to_float32_pair(value.value, low, high);
          return x86_float32_pair_to_float16(
              compute_float32(low), compute_float32(high));
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          __m256 low;
          __m256 high;
          x86_float16_to_float32_pair(value.value, low, high);
          return x86_float32_pair_to_float16(
              compute_float32(low), compute_float32(high));
        }
#endif
#if VEC_WIDTH >= 512
        else {
          __m512 low;
          __m512 high;
          x86_float16_to_float32_pair(value.value, low, high);
          return x86_float32_pair_to_float16(
              compute_float32(low), compute_float32(high));
        }
#endif
#else
        return x86_emulated_small_float_unary<T>(
            value.value, [](T input) {
              const auto widened = static_cast<float>(input);
              if constexpr (std::same_as<Op, SqrtOp>)
                return static_cast<T>(std::sqrt(widened));
              else if constexpr (std::same_as<Op, RcpOp>)
                return static_cast<T>(1.0F / widened);
              else
                return static_cast<T>(1.0F / std::sqrt(widened));
            });
#endif
      } else if constexpr (std::same_as<T, float32_t>) {
        return compute_float32(value.value);
      } else if constexpr (std::same_as<T, float64_t>) {
        if constexpr (sizeof(Raw) == 16) {
          if constexpr (std::same_as<Op, SqrtOp>) return _mm_sqrt_pd(value.value);
#if defined(CPU_CAPABILITY_AVX512)
          else if constexpr (std::same_as<Op, RcpOp>) return _mm_rcp14_pd(value.value);
          else return _mm_rsqrt14_pd(value.value);
#else
          else if constexpr (std::same_as<Op, RcpOp>)
            return _mm_div_pd(_mm_set1_pd(1.0), value.value);
          else return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(value.value));
#endif
        }
#if VEC_WIDTH >= 256
        else if constexpr (sizeof(Raw) == 32) {
          if constexpr (std::same_as<Op, SqrtOp>) return _mm256_sqrt_pd(value.value);
#if defined(CPU_CAPABILITY_AVX512)
          else if constexpr (std::same_as<Op, RcpOp>) return _mm256_rcp14_pd(value.value);
          else return _mm256_rsqrt14_pd(value.value);
#else
          else if constexpr (std::same_as<Op, RcpOp>)
            return _mm256_div_pd(_mm256_set1_pd(1.0), value.value);
          else return _mm256_div_pd(
              _mm256_set1_pd(1.0), _mm256_sqrt_pd(value.value));
#endif
        }
#endif
#if VEC_WIDTH >= 512
        else {
          if constexpr (std::same_as<Op, SqrtOp>) return _mm512_sqrt_pd(value.value);
          else if constexpr (std::same_as<Op, RcpOp>) return _mm512_rcp14_pd(value.value);
          else return _mm512_rsqrt14_pd(value.value);
        }
#endif
      }
    }();
    return NativeWordVec<Tag>{result};
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
#if defined(CPU_CAPABILITY_AVX512)
    using T = ElementOf<Tag>;
    using Raw = decltype(value.value);
#define VECOPS_VEC_X86_MASKED_FLOAT_UNARY(Width, Suffix)               \
    if constexpr (std::same_as<Op, SqrtOp>) {                          \
      if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
        return NativeWordVec<Tag>{_mm##Width##_maskz_sqrt_##Suffix(   \
            mask.value, value.value)};                                \
      else return NativeWordVec<Tag>{_mm##Width##_mask_sqrt_##Suffix( \
          inactive.value, mask.value, value.value)};                  \
    } else if constexpr (std::same_as<Op, RcpOp>) {                    \
      if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
        return NativeWordVec<Tag>{_mm##Width##_maskz_rcp14_##Suffix(  \
            mask.value, value.value)};                                \
      else return NativeWordVec<Tag>{_mm##Width##_mask_rcp14_##Suffix(\
          inactive.value, mask.value, value.value)};                  \
    } else {                                                           \
      if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
        return NativeWordVec<Tag>{_mm##Width##_maskz_rsqrt14_##Suffix(\
            mask.value, value.value)};                                \
      else return NativeWordVec<Tag>{_mm##Width##_mask_rsqrt14_##Suffix(\
          inactive.value, mask.value, value.value)};                  \
    }
    if constexpr (std::same_as<T, float32_t>) {
      if constexpr (sizeof(Raw) == 16) {
        VECOPS_VEC_X86_MASKED_FLOAT_UNARY(, ps);
      } else if constexpr (sizeof(Raw) == 32) {
        VECOPS_VEC_X86_MASKED_FLOAT_UNARY(256, ps);
      } else {
        VECOPS_VEC_X86_MASKED_FLOAT_UNARY(512, ps);
      }
    } else if constexpr (std::same_as<T, float64_t>) {
      if constexpr (sizeof(Raw) == 16) {
        VECOPS_VEC_X86_MASKED_FLOAT_UNARY(, pd);
      } else if constexpr (sizeof(Raw) == 32) {
        VECOPS_VEC_X86_MASKED_FLOAT_UNARY(256, pd);
      } else {
        VECOPS_VEC_X86_MASKED_FLOAT_UNARY(512, pd);
      }
    }
#undef VECOPS_VEC_X86_MASKED_FLOAT_UNARY
#endif
    return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, call<Index>(op, tag, value));
  }
};

template <>
struct NativeWordImpl<X86Backend, SqrtOp>
    : X86FloatingUnaryWordImpl<SqrtOp> {};
template <>
struct NativeWordImpl<X86Backend, RcpOp>
    : X86FloatingUnaryWordImpl<RcpOp> {};
template <>
struct NativeWordImpl<X86Backend, RsqrtOp>
    : X86FloatingUnaryWordImpl<RsqrtOp> {};

/* **************************************************************************** */
//                          FMA word implementation                           //
/* **************************************************************************** */

template <typename Op>
struct X86FmaWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordVec<Tag> c) {
    using T = ElementOf<Tag>;
    using Raw = decltype(a.value);
    static_assert(Index >= 0 && Index < num_words(tag));
#if defined(HAS_FMA)
#define VECOPS_VEC_X86_FMA(Width, Suffix)                               \
    if constexpr (std::same_as<Op, FmaddOp>)                            \
      return NativeWordVec<Tag>{_mm##Width##_fmadd_##Suffix(            \
          a.value, b.value, c.value)};                                  \
    else if constexpr (std::same_as<Op, FmsubOp>)                       \
      return NativeWordVec<Tag>{_mm##Width##_fmsub_##Suffix(            \
          a.value, b.value, c.value)};                                  \
    else if constexpr (std::same_as<Op, FnmaddOp>)                      \
      return NativeWordVec<Tag>{_mm##Width##_fnmadd_##Suffix(           \
          a.value, b.value, c.value)};                                  \
    else if constexpr (std::same_as<Op, FnmsubOp>)                      \
      return NativeWordVec<Tag>{_mm##Width##_fnmsub_##Suffix(           \
          a.value, b.value, c.value)}
    if constexpr (std::same_as<T, float32_t>) {
      if constexpr (sizeof(Raw) == 16) {
        VECOPS_VEC_X86_FMA(, ps);
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32) {
        VECOPS_VEC_X86_FMA(256, ps);
      }
#endif
#if VEC_WIDTH >= 512
      else {
        VECOPS_VEC_X86_FMA(512, ps);
      }
#endif
    } else if constexpr (std::same_as<T, float64_t>) {
      if constexpr (sizeof(Raw) == 16) {
        VECOPS_VEC_X86_FMA(, pd);
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32) {
        VECOPS_VEC_X86_FMA(256, pd);
      }
#endif
#if VEC_WIDTH >= 512
      else {
        VECOPS_VEC_X86_FMA(512, pd);
      }
#endif
    }
#undef VECOPS_VEC_X86_FMA
#endif
#if defined(HAS_AVX512_FP16)
#define VECOPS_VEC_X86_FMA_PH(Width, CastTo, CastFrom)                  \
    if constexpr (std::same_as<Op, FmaddOp>)                            \
      return NativeWordVec<Tag>{CastFrom(_mm##Width##_fmadd_ph(         \
          CastTo(a.value), CastTo(b.value), CastTo(c.value)))};         \
    else if constexpr (std::same_as<Op, FmsubOp>)                       \
      return NativeWordVec<Tag>{CastFrom(_mm##Width##_fmsub_ph(         \
          CastTo(a.value), CastTo(b.value), CastTo(c.value)))};         \
    else if constexpr (std::same_as<Op, FnmaddOp>)                      \
      return NativeWordVec<Tag>{CastFrom(_mm##Width##_fnmadd_ph(        \
          CastTo(a.value), CastTo(b.value), CastTo(c.value)))};         \
    else if constexpr (std::same_as<Op, FnmsubOp>)                      \
      return NativeWordVec<Tag>{CastFrom(_mm##Width##_fnmsub_ph(        \
          CastTo(a.value), CastTo(b.value), CastTo(c.value)))}
    if constexpr (std::same_as<T, float16_t>) {
      if constexpr (sizeof(Raw) == 16) {
        VECOPS_VEC_X86_FMA_PH(, _mm_castsi128_ph, _mm_castph_si128);
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32) {
        VECOPS_VEC_X86_FMA_PH(
            256, _mm256_castsi256_ph, _mm256_castph_si256);
      }
#endif
#if VEC_WIDTH >= 512
      else {
        VECOPS_VEC_X86_FMA_PH(
            512, _mm512_castsi512_ph, _mm512_castph_si512);
      }
#endif
    }
#undef VECOPS_VEC_X86_FMA_PH
#endif
    return synthesize_fma_word<X86Backend, Op, Index>(tag, a, b, c);
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordVec<Tag> c, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    using T = ElementOf<Tag>;
    using Raw = decltype(a.value);
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (
        std::same_as<Policy, PreserveArithmeticInactive> ||
        std::same_as<Policy, ZeroArithmeticInactive>) {
#define VECOPS_VEC_X86_MASKED_FMA(Width, Suffix)                        \
      if constexpr (std::same_as<Op, FmaddOp>) {                        \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
          return NativeWordVec<Tag>{_mm##Width##_maskz_fmadd_##Suffix(  \
              mask.value, a.value, b.value, c.value)};                  \
        else return NativeWordVec<Tag>{_mm##Width##_mask_fmadd_##Suffix(\
            a.value, mask.value, b.value, c.value)};                    \
      } else if constexpr (std::same_as<Op, FmsubOp>) {                 \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
          return NativeWordVec<Tag>{_mm##Width##_maskz_fmsub_##Suffix(  \
              mask.value, a.value, b.value, c.value)};                  \
        else return NativeWordVec<Tag>{_mm##Width##_mask_fmsub_##Suffix(\
            a.value, mask.value, b.value, c.value)};                    \
      } else if constexpr (std::same_as<Op, FnmaddOp>) {                \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
          return NativeWordVec<Tag>{_mm##Width##_maskz_fnmadd_##Suffix( \
              mask.value, a.value, b.value, c.value)};                  \
        else return NativeWordVec<Tag>{_mm##Width##_mask_fnmadd_##Suffix(\
            a.value, mask.value, b.value, c.value)};                    \
      } else if constexpr (std::same_as<Op, FnmsubOp>) {                \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
          return NativeWordVec<Tag>{_mm##Width##_maskz_fnmsub_##Suffix( \
              mask.value, a.value, b.value, c.value)};                  \
        else return NativeWordVec<Tag>{_mm##Width##_mask_fnmsub_##Suffix(\
            a.value, mask.value, b.value, c.value)};                    \
      }
      if constexpr (std::same_as<T, float32_t>) {
        if constexpr (sizeof(Raw) == 16) {
          VECOPS_VEC_X86_MASKED_FMA(, ps);
        } else if constexpr (sizeof(Raw) == 32) {
          VECOPS_VEC_X86_MASKED_FMA(256, ps);
        } else {
          VECOPS_VEC_X86_MASKED_FMA(512, ps);
        }
      } else if constexpr (std::same_as<T, float64_t>) {
        if constexpr (sizeof(Raw) == 16) {
          VECOPS_VEC_X86_MASKED_FMA(, pd);
        } else if constexpr (sizeof(Raw) == 32) {
          VECOPS_VEC_X86_MASKED_FMA(256, pd);
        } else {
          VECOPS_VEC_X86_MASKED_FMA(512, pd);
        }
      }
#undef VECOPS_VEC_X86_MASKED_FMA
#if defined(HAS_AVX512_FP16)
#define VECOPS_VEC_X86_MASKED_FMA_PH(Width, CastTo, CastFrom)           \
      if constexpr (std::same_as<Op, FmaddOp>) {                        \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
          return NativeWordVec<Tag>{CastFrom(                           \
              _mm##Width##_maskz_fmadd_ph(                              \
                  mask.value, CastTo(a.value), CastTo(b.value),         \
                  CastTo(c.value)))};                                   \
        else return NativeWordVec<Tag>{CastFrom(                        \
            _mm##Width##_mask_fmadd_ph(                                 \
                CastTo(a.value), mask.value, CastTo(b.value),           \
                CastTo(c.value)))};                                     \
      } else if constexpr (std::same_as<Op, FmsubOp>) {                 \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
          return NativeWordVec<Tag>{CastFrom(                           \
              _mm##Width##_maskz_fmsub_ph(                              \
                  mask.value, CastTo(a.value), CastTo(b.value),         \
                  CastTo(c.value)))};                                   \
        else return NativeWordVec<Tag>{CastFrom(                        \
            _mm##Width##_mask_fmsub_ph(                                 \
                CastTo(a.value), mask.value, CastTo(b.value),           \
                CastTo(c.value)))};                                     \
      } else if constexpr (std::same_as<Op, FnmaddOp>) {                \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
          return NativeWordVec<Tag>{CastFrom(                           \
              _mm##Width##_maskz_fnmadd_ph(                             \
                  mask.value, CastTo(a.value), CastTo(b.value),         \
                  CastTo(c.value)))};                                   \
        else return NativeWordVec<Tag>{CastFrom(                        \
            _mm##Width##_mask_fnmadd_ph(                                \
                CastTo(a.value), mask.value, CastTo(b.value),           \
                CastTo(c.value)))};                                     \
      } else if constexpr (std::same_as<Op, FnmsubOp>) {                \
        if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)     \
          return NativeWordVec<Tag>{CastFrom(                           \
              _mm##Width##_maskz_fnmsub_ph(                             \
                  mask.value, CastTo(a.value), CastTo(b.value),         \
                  CastTo(c.value)))};                                   \
        else return NativeWordVec<Tag>{CastFrom(                        \
            _mm##Width##_mask_fnmsub_ph(                                \
                CastTo(a.value), mask.value, CastTo(b.value),           \
                CastTo(c.value)))};                                     \
      }
      if constexpr (std::same_as<T, float16_t>) {
        if constexpr (sizeof(Raw) == 16) {
          VECOPS_VEC_X86_MASKED_FMA_PH(
              , _mm_castsi128_ph, _mm_castph_si128);
        } else if constexpr (sizeof(Raw) == 32) {
          VECOPS_VEC_X86_MASKED_FMA_PH(
              256, _mm256_castsi256_ph, _mm256_castph_si256);
        } else {
          VECOPS_VEC_X86_MASKED_FMA_PH(
              512, _mm512_castsi512_ph, _mm512_castph_si512);
        }
      }
#undef VECOPS_VEC_X86_MASKED_FMA_PH
#endif
    }
#endif
    const auto computed = call<Index>(op, tag, a, b, c);
    return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, computed);
  }
};

template <>
struct NativeWordImpl<X86Backend, FmaddOp> : X86FmaWordImpl<FmaddOp> {};
template <>
struct NativeWordImpl<X86Backend, FmsubOp> : X86FmaWordImpl<FmsubOp> {};
template <>
struct NativeWordImpl<X86Backend, FnmaddOp> : X86FmaWordImpl<FnmaddOp> {};
template <>
struct NativeWordImpl<X86Backend, FnmsubOp> : X86FmaWordImpl<FnmsubOp> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_ARITHMETIC_H
