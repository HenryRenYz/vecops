// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_X86_COMPARISON_H
#define VECOPS_VEC_DETAILS_X86_COMPARISON_H

/**
 * @file Comparison.h
 * @brief x86 backend implementations for comparison and classification operations.
 */

#include <limits>
#include <type_traits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/x86/Basic.h"

namespace vecops::vec::details {

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_vec_and(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16)
    return _mm_and_si128(a, b);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_castps_si256(_mm256_and_ps(
        _mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64)
    return _mm512_and_si512(a, b);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_vec_or(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16)
    return _mm_or_si128(a, b);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_castps_si256(_mm256_or_ps(
        _mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64)
    return _mm512_or_si512(a, b);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_vec_xor(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16)
    return _mm_xor_si128(a, b);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_castps_si256(_mm256_xor_ps(
        _mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64)
    return _mm512_xor_si512(a, b);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_vec_all_ones() {
  if constexpr (sizeof(Raw) == 16) return _mm_set1_epi32(-1);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ));
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64) return _mm512_set1_epi32(-1);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_combine_128(__m128i low, __m128i high) {
  if constexpr (sizeof(Raw) == 16) {
    (void)high;
    return low;
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
  }
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported split x86 word");
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_cmpeq_epi16_vector(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16) return _mm_cmpeq_epi16(a, b);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
#if defined(HAS_AVX2)
    return _mm256_cmpeq_epi16(a, b);
#else
    return x86_combine_128<Raw>(
        _mm_cmpeq_epi16(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b)),
        _mm_cmpeq_epi16(
            _mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)));
#endif
  }
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64)
    return _mm512_movm_epi16(_mm512_cmpeq_epi16_mask(a, b));
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_cmpgt_epi16_vector(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16) return _mm_cmpgt_epi16(a, b);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
#if defined(HAS_AVX2)
    return _mm256_cmpgt_epi16(a, b);
#else
    return x86_combine_128<Raw>(
        _mm_cmpgt_epi16(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b)),
        _mm_cmpgt_epi16(
            _mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)));
#endif
  }
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64)
    return _mm512_movm_epi16(_mm512_cmpgt_epi16_mask(a, b));
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_set1_epi16(uint16_t bits) {
  if constexpr (sizeof(Raw) == 16)
    return _mm_set1_epi16(::vecops::bitcast<int16_t>(bits));
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_set1_epi16(::vecops::bitcast<int16_t>(bits));
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64)
    return _mm512_set1_epi16(::vecops::bitcast<int16_t>(bits));
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported x86 word width");
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_cmpgt_epu16_vector(Raw a, Raw b) {
  const auto sign = x86_set1_epi16<Raw>(0x8000u);
  return x86_cmpgt_epi16_vector(x86_vec_xor(a, sign), x86_vec_xor(b, sign));
}

template <typename T, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_vector_mask_to_native(Raw value) {
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (sizeof(T) == 2) {
    if constexpr (sizeof(Raw) == 16) return _mm_movepi16_mask(value);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_movepi16_mask(value);
#endif
#if VEC_WIDTH >= 512
    else if constexpr (sizeof(Raw) == 64) return _mm512_movepi16_mask(value);
#endif
  }
  static_assert(sizeof(T) == 2, "vector-to-k conversion is only used for 16-bit floats");
#else
  return value;
#endif
}


/* **************************************************************************** */
//             Comparison and classification word implementations             //
/* **************************************************************************** */

template <typename Op>
consteval int x86_float_compare_predicate() {
  if constexpr (std::same_as<Op, CmpEqOp>) return _CMP_EQ_OQ;
  else if constexpr (std::same_as<Op, CmpNeOp>) return _CMP_NEQ_UQ;
  else if constexpr (std::same_as<Op, CmpLtOp>) return _CMP_LT_OQ;
  else if constexpr (std::same_as<Op, CmpGtOp>) return _CMP_GT_OQ;
  else if constexpr (std::same_as<Op, CmpLeOp>) return _CMP_LE_OQ;
  else if constexpr (std::same_as<Op, CmpGeOp>) return _CMP_GE_OQ;
  else static_assert(dispatch_dependent_false<Op>, "unsupported float comparison");
}

#if defined(HAS_AVX512_FP16)
template <typename Op, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_compare_float16_native(Raw a, Raw b) {
  constexpr int predicate = x86_float_compare_predicate<Op>();
  if constexpr (sizeof(Raw) == 16)
    return _mm_cmp_ph_mask(
        _mm_castsi128_ph(a), _mm_castsi128_ph(b), predicate);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_cmp_ph_mask(
        _mm256_castsi256_ph(a), _mm256_castsi256_ph(b), predicate);
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64)
    return _mm512_cmp_ph_mask(
        _mm512_castsi512_ph(a), _mm512_castsi512_ph(b), predicate);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported FP16 word width");
}

template <typename Op, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_compare_float16_native_masked(
    RawMask mask, Raw a, Raw b) {
  constexpr int predicate = x86_float_compare_predicate<Op>();
  if constexpr (sizeof(Raw) == 16)
    return _mm_mask_cmp_ph_mask(
        mask, _mm_castsi128_ph(a), _mm_castsi128_ph(b), predicate);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_mask_cmp_ph_mask(
        mask, _mm256_castsi256_ph(a), _mm256_castsi256_ph(b), predicate);
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64)
    return _mm512_mask_cmp_ph_mask(
        mask, _mm512_castsi512_ph(a), _mm512_castsi512_ph(b), predicate);
#endif
  else static_assert(dispatch_dependent_false<Raw>, "unsupported FP16 word width");
}
#endif

template <typename Op, typename T, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_compare_small_float(Op, Raw a, Raw b) {
  static_assert(
      std::same_as<T, float16_t> || std::same_as<T, bfloat16_t>,
      "small-float comparison requires a 16-bit floating type");
#if defined(HAS_AVX512_FP16)
  if constexpr (std::same_as<T, float16_t>)
    return x86_compare_float16_native<Op, RawMask>(a, b);
  else {
#endif
  constexpr uint16_t exponent = std::same_as<T, float16_t> ? 0x7c00u : 0x7f80u;
  constexpr uint16_t magnitude = 0x7fffu;
  const auto abs_mask = x86_set1_epi16<Raw>(magnitude);
  const auto exponent_value = x86_set1_epi16<Raw>(exponent);
  const auto zero = x86_set1_epi16<Raw>(0);
  const auto abs_a = x86_vec_and(a, abs_mask);
  const auto abs_b = x86_vec_and(b, abs_mask);
  const auto nan_a = x86_cmpgt_epu16_vector(abs_a, exponent_value);
  const auto nan_b = x86_cmpgt_epu16_vector(abs_b, exponent_value);
  const auto ordered = x86_vec_xor(x86_vec_or(nan_a, nan_b), x86_vec_all_ones<Raw>());
  const auto bit_equal = x86_cmpeq_epi16_vector(a, b);
  const auto both_zero = x86_vec_and(
      x86_cmpeq_epi16_vector(abs_a, zero),
      x86_cmpeq_epi16_vector(abs_b, zero));
  const auto equal = x86_vec_and(x86_vec_or(bit_equal, both_zero), ordered);

  Raw result;
  if constexpr (std::same_as<Op, CmpEqOp>) {
    result = equal;
  } else if constexpr (std::same_as<Op, CmpNeOp>) {
    // C++ != is true for unordered operands as well as ordered unequal ones.
    result = x86_vec_xor(equal, x86_vec_all_ones<Raw>());
  } else {
    const auto sign = x86_set1_epi16<Raw>(0x8000u);
    const auto key_a = x86_vec_xor(
        a, x86_vec_or(sign, x86_cmpgt_epi16_vector(zero, a)));
    const auto key_b = x86_vec_xor(
        b, x86_vec_or(sign, x86_cmpgt_epi16_vector(zero, b)));
    const auto unequal = x86_vec_xor(equal, x86_vec_all_ones<Raw>());
    const auto a_gt_b = x86_vec_and(
        x86_vec_and(x86_cmpgt_epu16_vector(key_a, key_b), ordered), unequal);
    const auto a_lt_b = x86_vec_and(
        x86_vec_and(x86_cmpgt_epu16_vector(key_b, key_a), ordered), unequal);
    if constexpr (std::same_as<Op, CmpLtOp>) result = a_lt_b;
    else if constexpr (std::same_as<Op, CmpGtOp>) result = a_gt_b;
    else if constexpr (std::same_as<Op, CmpLeOp>) result = x86_vec_or(a_lt_b, equal);
    else if constexpr (std::same_as<Op, CmpGeOp>) result = x86_vec_or(a_gt_b, equal);
    else static_assert(dispatch_dependent_false<Op>, "unsupported small-float comparison");
  }
  return x86_vector_mask_to_native<T, RawMask>(result);
#if defined(HAS_AVX512_FP16)
  }
#endif
}

template <typename Op, typename T, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_compare_standard_float(Op, Raw a, Raw b) {
  constexpr int predicate = x86_float_compare_predicate<Op>();
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16) return _mm_cmp_ps_mask(a, b, predicate);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_cmp_ps_mask(a, b, predicate);
#endif
#if VEC_WIDTH >= 512
    else if constexpr (sizeof(Raw) == 64) return _mm512_cmp_ps_mask(a, b, predicate);
#endif
  } else if constexpr (std::same_as<T, float64_t>) {
    if constexpr (sizeof(Raw) == 16) return _mm_cmp_pd_mask(a, b, predicate);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_cmp_pd_mask(a, b, predicate);
#endif
#if VEC_WIDTH >= 512
    else if constexpr (sizeof(Raw) == 64) return _mm512_cmp_pd_mask(a, b, predicate);
#endif
  } else static_assert(dispatch_dependent_false<T>, "unsupported x86 float type");
#else
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<Op, CmpEqOp>) return _mm_castps_si128(_mm_cmpeq_ps(a, b));
      else if constexpr (std::same_as<Op, CmpNeOp>) return _mm_castps_si128(_mm_cmpneq_ps(a, b));
      else if constexpr (std::same_as<Op, CmpLtOp>) return _mm_castps_si128(_mm_cmplt_ps(a, b));
      else if constexpr (std::same_as<Op, CmpGtOp>) return _mm_castps_si128(_mm_cmpgt_ps(a, b));
      else if constexpr (std::same_as<Op, CmpLeOp>) return _mm_castps_si128(_mm_cmple_ps(a, b));
      else return _mm_castps_si128(_mm_cmpge_ps(a, b));
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_castps_si256(_mm256_cmp_ps(a, b, predicate));
#endif
  } else if constexpr (std::same_as<T, float64_t>) {
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<Op, CmpEqOp>) return _mm_castpd_si128(_mm_cmpeq_pd(a, b));
      else if constexpr (std::same_as<Op, CmpNeOp>) return _mm_castpd_si128(_mm_cmpneq_pd(a, b));
      else if constexpr (std::same_as<Op, CmpLtOp>) return _mm_castpd_si128(_mm_cmplt_pd(a, b));
      else if constexpr (std::same_as<Op, CmpGtOp>) return _mm_castpd_si128(_mm_cmpgt_pd(a, b));
      else if constexpr (std::same_as<Op, CmpLeOp>) return _mm_castpd_si128(_mm_cmple_pd(a, b));
      else return _mm_castpd_si128(_mm_cmpge_pd(a, b));
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_castpd_si256(_mm256_cmp_pd(a, b, predicate));
#endif
  } else static_assert(dispatch_dependent_false<T>, "unsupported x86 float type");
#endif
  VECOPS_UNREACHABLE();
}

template <typename Op>
consteval int x86_integer_predicate() {
  if constexpr (std::same_as<Op, CmpEqOp>) return _MM_CMPINT_EQ;
  else if constexpr (std::same_as<Op, CmpNeOp>) return _MM_CMPINT_NE;
  else if constexpr (std::same_as<Op, CmpLtOp>) return _MM_CMPINT_LT;
  else if constexpr (std::same_as<Op, CmpGtOp>) return _MM_CMPINT_GT;
  else if constexpr (std::same_as<Op, CmpLeOp>) return _MM_CMPINT_LE;
  else if constexpr (std::same_as<Op, CmpGeOp>) return _MM_CMPINT_GE;
  else static_assert(dispatch_dependent_false<Op>, "unsupported integer comparison");
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_signed_gt_vector(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16) {
    if constexpr (sizeof(T) == 1) return _mm_cmpgt_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm_cmpgt_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm_cmpgt_epi32(a, b);
    else if constexpr (sizeof(T) == 8) return _mm_cmpgt_epi64(a, b);
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
#if defined(HAS_AVX2)
    if constexpr (sizeof(T) == 1) return _mm256_cmpgt_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm256_cmpgt_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm256_cmpgt_epi32(a, b);
    else if constexpr (sizeof(T) == 8) return _mm256_cmpgt_epi64(a, b);
#else
    return x86_combine_128<Raw>(
        x86_signed_gt_vector<T>(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b)),
        x86_signed_gt_vector<T>(
            _mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)));
#endif
  }
#endif
  VECOPS_UNREACHABLE();
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_integer_sign_vector() {
  if constexpr (sizeof(Raw) == 16) {
    if constexpr (sizeof(T) == 1) return _mm_set1_epi8(static_cast<char>(0x80));
    else if constexpr (sizeof(T) == 2) return _mm_set1_epi16(static_cast<int16_t>(0x8000u));
    else if constexpr (sizeof(T) == 4) return _mm_set1_epi32(static_cast<int32_t>(0x80000000u));
    else return _mm_set1_epi64x(static_cast<int64_t>(0x8000000000000000ull));
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    const auto half = x86_integer_sign_vector<T, __m128i>();
    return x86_combine_128<Raw>(half, half);
  }
#endif
  VECOPS_UNREACHABLE();
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_eq_vector(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16) {
    if constexpr (sizeof(T) == 1) return _mm_cmpeq_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm_cmpeq_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm_cmpeq_epi32(a, b);
    else return _mm_cmpeq_epi64(a, b);
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
#if defined(HAS_AVX2)
    if constexpr (sizeof(T) == 1) return _mm256_cmpeq_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm256_cmpeq_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm256_cmpeq_epi32(a, b);
    else return _mm256_cmpeq_epi64(a, b);
#else
    return x86_combine_128<Raw>(
        x86_eq_vector<T>(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b)),
        x86_eq_vector<T>(
            _mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)));
#endif
  }
#endif
  VECOPS_UNREACHABLE();
}

template <typename Op, typename T, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_compare_integer(Op, Raw a, Raw b) {
  static_assert(std::integral<T>, "integer comparison requires an integer dtype");
#if defined(CPU_CAPABILITY_AVX512)
#define VECOPS_VEC_X86_AVX512_INT_COMPARE(Type, SignedName, UnsignedName) \
  if constexpr (std::same_as<T, Type>) {                                  \
    if constexpr (sizeof(Raw) == 16) return _mm_cmp_##SignedName##_mask(a, b, x86_integer_predicate<Op>()); \
    else if constexpr (sizeof(Raw) == 32) return _mm256_cmp_##SignedName##_mask(a, b, x86_integer_predicate<Op>()); \
    else return _mm512_cmp_##SignedName##_mask(a, b, x86_integer_predicate<Op>()); \
  } else if constexpr (std::same_as<T, std::make_unsigned_t<Type>>) {     \
    if constexpr (sizeof(Raw) == 16) return _mm_cmp_##UnsignedName##_mask(a, b, x86_integer_predicate<Op>()); \
    else if constexpr (sizeof(Raw) == 32) return _mm256_cmp_##UnsignedName##_mask(a, b, x86_integer_predicate<Op>()); \
    else return _mm512_cmp_##UnsignedName##_mask(a, b, x86_integer_predicate<Op>()); \
  } else
  VECOPS_VEC_X86_AVX512_INT_COMPARE(int8_t, epi8, epu8)
  VECOPS_VEC_X86_AVX512_INT_COMPARE(int16_t, epi16, epu16)
  VECOPS_VEC_X86_AVX512_INT_COMPARE(int32_t, epi32, epu32)
  VECOPS_VEC_X86_AVX512_INT_COMPARE(int64_t, epi64, epu64) {
    static_assert(dispatch_dependent_false<T>, "unsupported AVX-512 integer dtype");
  }
#undef VECOPS_VEC_X86_AVX512_INT_COMPARE
#else
  Raw result;
  if constexpr (std::same_as<Op, CmpEqOp> || std::same_as<Op, CmpNeOp>) {
    result = x86_eq_vector<T>(a, b);
    if constexpr (std::same_as<Op, CmpNeOp>)
      result = x86_vec_xor(result, x86_vec_all_ones<Raw>());
  } else {
    if constexpr (std::is_unsigned_v<T>) {
      const auto sign = x86_integer_sign_vector<T, Raw>();
      a = x86_vec_xor(a, sign);
      b = x86_vec_xor(b, sign);
    }
    const auto gt = x86_signed_gt_vector<T>(a, b);
    const auto lt = x86_signed_gt_vector<T>(b, a);
    if constexpr (std::same_as<Op, CmpGtOp>) result = gt;
    else if constexpr (std::same_as<Op, CmpLtOp>) result = lt;
    else if constexpr (std::same_as<Op, CmpGeOp>)
      result = x86_vec_xor(lt, x86_vec_all_ones<Raw>());
    else if constexpr (std::same_as<Op, CmpLeOp>)
      result = x86_vec_xor(gt, x86_vec_all_ones<Raw>());
    else static_assert(dispatch_dependent_false<Op>, "unsupported integer comparison");
  }
  return result;
#endif
}

template <typename Op, typename T, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_compare_raw(Op op, Raw a, Raw b) {
  if constexpr (std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>)
    return x86_compare_small_float<Op, T, RawMask>(op, a, b);
  else if constexpr (std::same_as<T, float32_t> || std::same_as<T, float64_t>)
    return x86_compare_standard_float<Op, T, RawMask>(op, a, b);
  else if constexpr (
      std::same_as<T, int8_t> || std::same_as<T, uint8_t> ||
      std::same_as<T, int16_t> || std::same_as<T, uint16_t> ||
      std::same_as<T, int32_t> || std::same_as<T, uint32_t> ||
      std::same_as<T, int64_t> || std::same_as<T, uint64_t>)
    return x86_compare_integer<Op, T, RawMask>(op, a, b);
  else static_assert(
      dispatch_dependent_false<T>,
      "x86 comparison has no implementation for this element type");
}

template <typename Op, typename T, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_classify_small_float(Op, Raw value) {
#if defined(HAS_AVX512_FP16)
  if constexpr (std::same_as<T, float16_t>) {
    if constexpr (std::same_as<Op, IsNanOp>) {
      if constexpr (sizeof(Raw) == 16)
        return _mm_cmp_ph_mask(
            _mm_castsi128_ph(value), _mm_castsi128_ph(value), _CMP_UNORD_Q);
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32)
        return _mm256_cmp_ph_mask(
            _mm256_castsi256_ph(value), _mm256_castsi256_ph(value),
            _CMP_UNORD_Q);
#endif
#if VEC_WIDTH >= 512
      else
        return _mm512_cmp_ph_mask(
            _mm512_castsi512_ph(value), _mm512_castsi512_ph(value),
            _CMP_UNORD_Q);
#endif
    } else {
      const auto positive = x86_set1_epi16<Raw>(0x7c00u);
      if constexpr (std::same_as<Op, IsPosInfOp>)
        return x86_compare_float16_native<CmpEqOp, RawMask>(value, positive);
      else if constexpr (std::same_as<Op, IsNegInfOp>)
        return x86_compare_float16_native<CmpEqOp, RawMask>(
            value, x86_set1_epi16<Raw>(0xfc00u));
      else if constexpr (std::same_as<Op, IsInfOp>)
        return x86_compare_float16_native<CmpEqOp, RawMask>(
            x86_vec_and(value, x86_set1_epi16<Raw>(0x7fffu)), positive);
      else if constexpr (std::same_as<Op, IsFiniteOp>)
        return x86_vector_mask_to_native<T, RawMask>(
            x86_cmpgt_epu16_vector(
                positive,
                x86_vec_and(value, x86_set1_epi16<Raw>(0x7fffu))));
      else if constexpr (std::same_as<Op, IsNormalOp>) {
        const auto absolute =
            x86_vec_and(value, x86_set1_epi16<Raw>(0x7fffu));
        const auto below_inf = x86_cmpgt_epu16_vector(positive, absolute);
        const auto below_min = x86_cmpgt_epu16_vector(
            x86_set1_epi16<Raw>(0x0400u), absolute);
        return x86_vector_mask_to_native<T, RawMask>(x86_vec_and(
            below_inf,
            x86_vec_xor(below_min, x86_vec_all_ones<Raw>())));
      } else if constexpr (std::same_as<Op, SignBitOp>)
        return x86_vector_mask_to_native<T, RawMask>(x86_cmpeq_epi16_vector(
            x86_vec_and(value, x86_set1_epi16<Raw>(0x8000u)),
            x86_set1_epi16<Raw>(0x8000u)));
      else static_assert(dispatch_dependent_false<Op>, "unsupported classification");
    }
  } else {
#endif
  constexpr uint16_t exponent = std::same_as<T, float16_t> ? 0x7c00u : 0x7f80u;
  const auto absolute = x86_vec_and(value, x86_set1_epi16<Raw>(0x7fffu));
  Raw result;
  if constexpr (std::same_as<Op, IsNanOp>)
    result = x86_cmpgt_epu16_vector(absolute, x86_set1_epi16<Raw>(exponent));
  else if constexpr (std::same_as<Op, IsPosInfOp>)
    result = x86_cmpeq_epi16_vector(value, x86_set1_epi16<Raw>(exponent));
  else if constexpr (std::same_as<Op, IsNegInfOp>)
    result = x86_cmpeq_epi16_vector(
        value, x86_set1_epi16<Raw>(static_cast<uint16_t>(exponent | 0x8000u)));
  else if constexpr (std::same_as<Op, IsInfOp>)
    result = x86_cmpeq_epi16_vector(absolute, x86_set1_epi16<Raw>(exponent));
  else if constexpr (std::same_as<Op, IsFiniteOp>)
    result = x86_cmpgt_epu16_vector(
        x86_set1_epi16<Raw>(exponent), absolute);
  else if constexpr (std::same_as<Op, IsNormalOp>) {
    constexpr uint16_t minimum_normal =
        std::same_as<T, float16_t> ? 0x0400u : 0x0080u;
    const auto below_inf = x86_cmpgt_epu16_vector(
        x86_set1_epi16<Raw>(exponent), absolute);
    const auto below_min = x86_cmpgt_epu16_vector(
        x86_set1_epi16<Raw>(minimum_normal), absolute);
    result = x86_vec_and(
        below_inf, x86_vec_xor(below_min, x86_vec_all_ones<Raw>()));
  } else if constexpr (std::same_as<Op, SignBitOp>)
    result = x86_cmpeq_epi16_vector(
        x86_vec_and(value, x86_set1_epi16<Raw>(0x8000u)),
        x86_set1_epi16<Raw>(0x8000u));
  else static_assert(dispatch_dependent_false<Op>, "unsupported x86 classification");
  return x86_vector_mask_to_native<T, RawMask>(result);
#if defined(HAS_AVX512_FP16)
  }
#endif
}

#if defined(HAS_AVX512_FP16)
template <typename Op, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_classify_float16_native_masked(
    RawMask mask, Raw value) {
  if constexpr (std::same_as<Op, IsNanOp>) {
    if constexpr (sizeof(Raw) == 16)
      return _mm_mask_cmp_ph_mask(
          mask, _mm_castsi128_ph(value), _mm_castsi128_ph(value),
          _CMP_UNORD_Q);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_mask_cmp_ph_mask(
          mask, _mm256_castsi256_ph(value), _mm256_castsi256_ph(value),
          _CMP_UNORD_Q);
#endif
#if VEC_WIDTH >= 512
    else
      return _mm512_mask_cmp_ph_mask(
          mask, _mm512_castsi512_ph(value), _mm512_castsi512_ph(value),
          _CMP_UNORD_Q);
#endif
  } else {
    const auto positive = x86_set1_epi16<Raw>(0x7c00u);
    if constexpr (std::same_as<Op, IsPosInfOp>)
      return x86_compare_float16_native_masked<CmpEqOp>(
          mask, value, positive);
    else if constexpr (std::same_as<Op, IsNegInfOp>)
      return x86_compare_float16_native_masked<CmpEqOp>(
          mask, value, x86_set1_epi16<Raw>(0xfc00u));
    else if constexpr (std::same_as<Op, IsInfOp>)
      return x86_compare_float16_native_masked<CmpEqOp>(
          mask,
          x86_vec_and(value, x86_set1_epi16<Raw>(0x7fffu)),
          positive);
    else static_assert(dispatch_dependent_false<Op>, "unsupported classification");
  }
}
#endif

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_float_infinity(bool negative) {
  if constexpr (std::same_as<T, float32_t>) {
    const float value = negative
        ? -std::numeric_limits<float>::infinity()
        : std::numeric_limits<float>::infinity();
    if constexpr (sizeof(Raw) == 16) return _mm_set1_ps(value);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_set1_ps(value);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_set1_ps(value);
#endif
  } else if constexpr (std::same_as<T, float64_t>) {
    const double value = negative
        ? -std::numeric_limits<double>::infinity()
        : std::numeric_limits<double>::infinity();
    if constexpr (sizeof(Raw) == 16) return _mm_set1_pd(value);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_set1_pd(value);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_set1_pd(value);
#endif
  } else static_assert(dispatch_dependent_false<T>, "unsupported float dtype");
}

template <typename Op, typename T, typename RawMask, typename Raw>
VECOPS_ALWAYS_INLINE RawMask x86_classify_raw(Op op, Raw value) {
  if constexpr (std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>) {
    return x86_classify_small_float<Op, T, RawMask>(op, value);
  } else if constexpr (std::same_as<T, float32_t> || std::same_as<T, float64_t>) {
    if constexpr (std::same_as<Op, IsNanOp>) {
      return x86_compare_standard_float<CmpNeOp, T, RawMask>(CmpNeOp{}, value, value);
    } else if constexpr (std::same_as<Op, IsPosInfOp>) {
      return x86_compare_standard_float<CmpEqOp, T, RawMask>(
          CmpEqOp{}, value, x86_float_infinity<T, Raw>(false));
    } else if constexpr (std::same_as<Op, IsNegInfOp>) {
      return x86_compare_standard_float<CmpEqOp, T, RawMask>(
          CmpEqOp{}, value, x86_float_infinity<T, Raw>(true));
    } else if constexpr (std::same_as<Op, IsInfOp>) {
      const auto positive = x86_compare_standard_float<CmpEqOp, T, RawMask>(
          CmpEqOp{}, value, x86_float_infinity<T, Raw>(false));
      const auto negative = x86_compare_standard_float<CmpEqOp, T, RawMask>(
          CmpEqOp{}, value, x86_float_infinity<T, Raw>(true));
#if defined(CPU_CAPABILITY_AVX512)
      return static_cast<RawMask>(positive | negative);
#else
      return x86_vec_or(positive, negative);
#endif
    } else if constexpr (
        std::same_as<Op, IsFiniteOp> ||
        std::same_as<Op, IsNormalOp> ||
        std::same_as<Op, SignBitOp>) {
      const auto bits = [&] {
        if constexpr (std::same_as<T, float32_t>) {
          if constexpr (sizeof(Raw) == 16) return _mm_castps_si128(value);
#if VEC_WIDTH >= 256
          else if constexpr (sizeof(Raw) == 32)
            return _mm256_castps_si256(value);
#endif
#if VEC_WIDTH >= 512
          else return _mm512_castps_si512(value);
#endif
        } else {
          if constexpr (sizeof(Raw) == 16) return _mm_castpd_si128(value);
#if VEC_WIDTH >= 256
          else if constexpr (sizeof(Raw) == 32)
            return _mm256_castpd_si256(value);
#endif
#if VEC_WIDTH >= 512
          else return _mm512_castpd_si512(value);
#endif
        }
      }();
      if constexpr (std::same_as<Op, SignBitOp>) {
#if defined(CPU_CAPABILITY_AVX512)
        if constexpr (std::same_as<T, float32_t>) {
          if constexpr (sizeof(Raw) == 16) return _mm_movepi32_mask(bits);
          else if constexpr (sizeof(Raw) == 32)
            return _mm256_movepi32_mask(bits);
          else return _mm512_movepi32_mask(bits);
        } else {
          if constexpr (sizeof(Raw) == 16) return _mm_movepi64_mask(bits);
          else if constexpr (sizeof(Raw) == 32)
            return _mm256_movepi64_mask(bits);
          else return _mm512_movepi64_mask(bits);
        }
#else
        using S = std::conditional_t<
            std::same_as<T, float32_t>, int32_t, int64_t>;
        return x86_vector_mask_to_native<T, RawMask>(
            x86_signed_gt_vector<S>(
                x86_bit_zero_raw<decltype(bits)>(), bits));
#endif
      } else {
        const auto absolute = [&] {
          if constexpr (std::same_as<T, float32_t>) {
            if constexpr (sizeof(Raw) == 16)
              return _mm_andnot_ps(_mm_set1_ps(-0.0F), value);
#if VEC_WIDTH >= 256
            else if constexpr (sizeof(Raw) == 32)
              return _mm256_andnot_ps(_mm256_set1_ps(-0.0F), value);
#endif
#if VEC_WIDTH >= 512
            else return _mm512_castsi512_ps(_mm512_andnot_si512(
                _mm512_set1_epi32(static_cast<int32_t>(0x80000000u)),
                _mm512_castps_si512(value)));
#endif
          } else {
            if constexpr (sizeof(Raw) == 16)
              return _mm_andnot_pd(_mm_set1_pd(-0.0), value);
#if VEC_WIDTH >= 256
            else if constexpr (sizeof(Raw) == 32)
              return _mm256_andnot_pd(_mm256_set1_pd(-0.0), value);
#endif
#if VEC_WIDTH >= 512
            else return _mm512_castsi512_pd(_mm512_andnot_si512(
                _mm512_set1_epi64(
                    static_cast<int64_t>(0x8000000000000000ull)),
                _mm512_castpd_si512(value)));
#endif
          }
        }();
        const auto minimum_normal = [&] {
          if constexpr (std::same_as<T, float32_t>) {
            if constexpr (sizeof(Raw) == 16)
              return _mm_set1_ps(std::numeric_limits<float32_t>::min());
#if VEC_WIDTH >= 256
            else if constexpr (sizeof(Raw) == 32)
              return _mm256_set1_ps(std::numeric_limits<float32_t>::min());
#endif
#if VEC_WIDTH >= 512
            else return _mm512_set1_ps(
                std::numeric_limits<float32_t>::min());
#endif
          } else {
            if constexpr (sizeof(Raw) == 16)
              return _mm_set1_pd(std::numeric_limits<float64_t>::min());
#if VEC_WIDTH >= 256
            else if constexpr (sizeof(Raw) == 32)
              return _mm256_set1_pd(std::numeric_limits<float64_t>::min());
#endif
#if VEC_WIDTH >= 512
            else return _mm512_set1_pd(
                std::numeric_limits<float64_t>::min());
#endif
          }
        }();
        const auto below_inf = x86_compare_standard_float<
            CmpLtOp, T, RawMask>(
                CmpLtOp{}, absolute, x86_float_infinity<T, Raw>(false));
        if constexpr (std::same_as<Op, IsFiniteOp>) return below_inf;
        const auto normal = x86_compare_standard_float<
            CmpGeOp, T, RawMask>(
                CmpGeOp{}, absolute, minimum_normal);
#if defined(CPU_CAPABILITY_AVX512)
        return static_cast<RawMask>(below_inf & normal);
#else
        return x86_vec_and(below_inf, normal);
#endif
      }
    } else static_assert(dispatch_dependent_false<Op>, "unsupported x86 classification");
  } else static_assert(
      dispatch_dependent_false<T>,
      "x86 classification has no implementation for this element type");
}

template <typename Op>
struct X86BinaryComparisonImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    return NativeWordMask<Tag>{
        x86_compare_raw<Op, T, typename Traits::RawMask>(op, a.value, b.value)};
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask) {
    auto result = call<Index>(op, tag, a, b);
    result.value = x86_mask_and(result.value, mask.value);
    return result;
  }
};

template <typename Op>
struct X86ClassificationImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    return NativeWordMask<Tag>{
        x86_classify_raw<Op, T, typename Traits::RawMask>(op, value.value)};
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value, NativeWordMask<Tag> mask) {
#if defined(HAS_AVX512_FP16)
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    if constexpr (
        std::same_as<T, float16_t> &&
        (std::same_as<Op, IsNanOp> ||
         std::same_as<Op, IsPosInfOp> ||
         std::same_as<Op, IsNegInfOp> ||
         std::same_as<Op, IsInfOp>)) {
      static_assert(Index >= 0 && Index < Traits::word_count);
      return NativeWordMask<Tag>{
          x86_classify_float16_native_masked<Op>(mask.value, value.value)};
    } else {
#endif
    auto result = call<Index>(op, tag, value);
    result.value = x86_mask_and(result.value, mask.value);
    return result;
#if defined(HAS_AVX512_FP16)
    }
#endif
  }
};

#define VECOPS_VEC_X86_BINARY_COMPARISON(OpType)                         \
  template <> struct NativeWordImpl<X86Backend, OpType>                  \
      : X86BinaryComparisonImpl<OpType> {}

VECOPS_VEC_X86_BINARY_COMPARISON(CmpEqOp);
VECOPS_VEC_X86_BINARY_COMPARISON(CmpNeOp);
VECOPS_VEC_X86_BINARY_COMPARISON(CmpLtOp);
VECOPS_VEC_X86_BINARY_COMPARISON(CmpGtOp);
VECOPS_VEC_X86_BINARY_COMPARISON(CmpLeOp);
VECOPS_VEC_X86_BINARY_COMPARISON(CmpGeOp);
#undef VECOPS_VEC_X86_BINARY_COMPARISON

#define VECOPS_VEC_X86_CLASSIFICATION(OpType)                            \
  template <> struct NativeWordImpl<X86Backend, OpType>                  \
      : X86ClassificationImpl<OpType> {}

VECOPS_VEC_X86_CLASSIFICATION(IsNanOp);
VECOPS_VEC_X86_CLASSIFICATION(IsPosInfOp);
VECOPS_VEC_X86_CLASSIFICATION(IsNegInfOp);
VECOPS_VEC_X86_CLASSIFICATION(IsInfOp);
VECOPS_VEC_X86_CLASSIFICATION(IsFiniteOp);
VECOPS_VEC_X86_CLASSIFICATION(IsNormalOp);
VECOPS_VEC_X86_CLASSIFICATION(SignBitOp);
#undef VECOPS_VEC_X86_CLASSIFICATION

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_COMPARISON_H
