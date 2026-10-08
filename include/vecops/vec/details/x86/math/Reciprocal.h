// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_X86_MATH_RECIPROCAL_H
#define VECOPS_VEC_DETAILS_X86_MATH_RECIPROCAL_H

/**
 * @file Reciprocal.h
 * @brief x86 backend implementations for rcp and rsqrt (all accuracy tiers).
 *
 * Tiers follow the shared reciprocal contract:
 *
 * | Tier     | Chain                                          | Contract            |
 * |----------|------------------------------------------------|---------------------|
 * | Estimate | RCP14/RCPPS or RSQRT14/RSQRTPS only           | rel. error <= 2^-7  |
 * | Fast     | estimate + 1 Markstein refinement step         | rel. error <= 2^-15 |
 * | Strict   | estimate + 2 Markstein steps, or 1/x resp.     | ULP error <= 1      |
 * |          | 1/sqrt(x) on tiers lacking FMA or estimates   |                     |
 *
 * The Markstein step (r = 1 - x*y evaluated inside one FMA, then
 * y' = fma(r, y, y)) squares the previous error and adds a single rounding,
 * so each step trades one accuracy class for at most half an ULP. The
 * reciprocal form is an exact-residual refinement; the rsqrt form rounds
 * y*y first, and the full-domain accuracy probe verifies the composed chain
 * still respects the 1 ULP Strict bound.
 *
 * Configurations without usable estimate instructions (f64 before AVX512)
 * or without FMA (Strict on pre-FMA targets) evaluate the IEEE composition
 * 1/x resp. 1/sqrt(x), which every tier contract permits. Lanes outside the
 * plain estimate domain — subnormal inputs, reciprocals of huge magnitudes,
 * zero, infinity — are repaired on a cold branch that recomputes the
 * affected lanes in long double. Long double is the x87 80-bit format on
 * this project's Linux x86 targets, covering the full f32/f64 subnormal
 * range exactly, so special-input results are identical across tiers and
 * gradual subnormal outputs appear exactly when the tier requires them.
 * The inline path only sees normal inputs, so its outputs are always
 * normal and need no flush.
 */

#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/x86/Arithmetic.h"
#include "vecops/vec/details/x86/Basic.h"
#include "vecops/vec/details/x86/Types.h"

namespace vecops::vec::details {

template <Accuracy Tier>
constexpr bool x86_recip_gradual() {
#ifdef VECOPS_PRESERVE_SUBNORMALS
  return Tier == Accuracy::Strict;
#else
  return false;
#endif
}

#if defined(CPU_CAPABILITY_AVX512)
inline constexpr bool x86_recip_has_avx512 = true;
#else
inline constexpr bool x86_recip_has_avx512 = false;
#endif


/* **************************************************************************** */
//    estimate instructions                                                     //
/* **************************************************************************** */

template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_rcp_estimate(
    Tag, NativeWordVec<Tag> x) {
  using Raw = decltype(x.value);
  static_assert(
      std::same_as<ElementOf<Tag>, float32_t> ||
      (std::same_as<ElementOf<Tag>, float64_t> && x86_recip_has_avx512));
  if constexpr (std::same_as<ElementOf<Tag>, float32_t>) {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_rcp14_ps(x.value)};
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<Tag>{_mm256_rcp14_ps(x.value)};
    else
      return NativeWordVec<Tag>{_mm512_rcp14_ps(x.value)};
#else
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_rcp_ps(x.value)};
    else
      return NativeWordVec<Tag>{_mm256_rcp_ps(x.value)};
#endif
  } else {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_rcp14_pd(x.value)};
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<Tag>{_mm256_rcp14_pd(x.value)};
    else
      return NativeWordVec<Tag>{_mm512_rcp14_pd(x.value)};
#endif
  }
}

template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_rsqrt_estimate(
    Tag, NativeWordVec<Tag> x) {
  using Raw = decltype(x.value);
  if constexpr (std::same_as<ElementOf<Tag>, float32_t>) {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_rsqrt14_ps(x.value)};
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<Tag>{
          _mm256_rsqrt14_ps(x.value)};
    else
      return NativeWordVec<Tag>{
          _mm512_rsqrt14_ps(x.value)};
#else
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_rsqrt_ps(x.value)};
    else
      return NativeWordVec<Tag>{_mm256_rsqrt_ps(x.value)};
#endif
  } else {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(Raw) == 16)
      return NativeWordVec<Tag>{_mm_rsqrt14_pd(x.value)};
    else if constexpr (sizeof(Raw) == 32)
      return NativeWordVec<Tag>{
          _mm256_rsqrt14_pd(x.value)};
    else
      return NativeWordVec<Tag>{
          _mm512_rsqrt14_pd(x.value)};
#endif
  }
}

/* **************************************************************************** */
//    refinement chains                                                         //
/* **************************************************************************** */

template <nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_recip_markstein(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag> y) {
  const auto one = fill_word(
      tag, ElementOf<Tag>(1));
  const auto residual =
      fnmadd(tag, x, y, one);
  return fmadd(tag, residual, y, y);
}

template <nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_rsqrt_markstein(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag> y) {
  const auto one = fill_word(
      tag, ElementOf<Tag>(1));
  const auto half = fill_word(
      tag, ElementOf<Tag>(0.5));
  const auto half_y = mul(tag, y, half);
  const auto square = mul(tag, y, y);
  const auto residual =
      fnmadd(tag, x, square, one);
  return fmadd(tag, residual, half_y, y);
}

/** FMA-less Newton step, used only by the Fast tier on pre-FMA targets. */
template <nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_recip_plain(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag> y) {
  const auto one = fill_word(
      tag, ElementOf<Tag>(1));
  const auto residual = sub(
      tag, one, mul(tag, x, y));
  return add(
      tag, y, mul(tag, residual, y));
}

template <nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_rsqrt_plain(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag> y) {
  const auto one = fill_word(
      tag, ElementOf<Tag>(1));
  const auto half = fill_word(
      tag, ElementOf<Tag>(0.5));
  const auto half_y = mul(tag, y, half);
  const auto square = mul(tag, y, y);
  const auto residual = sub(
      tag, one, mul(tag, x, square));
  return add(
      tag, y, mul(tag, residual, half_y));
}

/* **************************************************************************** */
//    cold-path lane repair                                                     //
/* **************************************************************************** */

/**
 * Long-double recomputation for lanes outside the plain estimate domain.
 * Covers subnormal or tiny inputs, reciprocals of huge magnitudes (whose
 * results are subnormal), zero, and infinity. rsqrt(-0) reads +inf here,
 * matching the estimate-instruction convention instead of 1/sqrt(-0) = -inf.
 */
template <bool IsRsqrt, Element T>
T x86_recip_scalar_fixup(T value) {
  if constexpr (IsRsqrt) {
    if (value == T(0)) return std::numeric_limits<T>::infinity();
    const long double widened = static_cast<long double>(value);
    return static_cast<T>(1.0L / std::sqrt(widened));
  } else {
    const long double widened = static_cast<long double>(value);
    return static_cast<T>(1.0L / widened);
  }
}

template <Accuracy Tier, Element T>
T x86_recip_flush(T result) {
  if constexpr (x86_recip_gradual<Tier>()) {
    return result;
  } else {
    const double widened = static_cast<double>(result);
    if (widened != 0.0 && std::abs(widened) <
                          static_cast<double>(std::numeric_limits<T>::min()))
      return static_cast<T>(widened * 0.0);
    return result;
  }
}

/**
 * Per-lane repair entry. Runs only after a vector compare proved that some
 * lane is outside the estimate domain, so the store/reload round-trip never
 * touches the hot path. NaN lanes never match the predicates and keep the
 * NaN the vector chain already produced.
 */
template <Accuracy Tier, bool IsRsqrt, Element T, typename Raw>
Raw x86_recip_repair_word(Raw raw_x, Raw raw_y) {
  constexpr std::size_t lanes = sizeof(Raw) / sizeof(T);
  alignas(64) T inputs[lanes];
  alignas(64) T outputs[lanes];
  std::memcpy(inputs, &raw_x, sizeof(Raw));
  std::memcpy(outputs, &raw_y, sizeof(Raw));
  // One binade of slack above the smallest normal: the estimate seeds sit
  // right at their exponent-range edge there and the exhaustive sweep found
  // a single 2-ULP Strict outlier, so those lanes also take the long-double
  // repair path.
  const T smallest = std::same_as<T, float32_t> ? T(0x1p-125f) : T(0x1p-1021);
  const T huge = std::same_as<T, float32_t> ? T(0x1p125f) : T(0x1p1021);
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const T value = inputs[lane];
    const bool bad = std::abs(value) < smallest ||
        value > huge || value < -huge;
    if (bad) {
      outputs[lane] = x86_recip_flush<Tier, T>(
          x86_recip_scalar_fixup<IsRsqrt, T>(value));
    }
  }
  Raw raw;
  std::memcpy(&raw, outputs, sizeof(Raw));
  return raw;
}

/* **************************************************************************** */
//    word-level tier dispatch (f32 / f64)                                     //
/* **************************************************************************** */

/**
 * The smallest input whose reciprocal is still normal; beyond it the result
 * would be subnormal and the lane needs the repair path.
 */
template <Element T>
constexpr T x86_recip_huge_bound() {
  return std::same_as<T, float32_t> ? T(0x1p126f) : T(0x1p1022);
}

template <Accuracy Tier, bool IsRsqrt, nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_recip_word(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
#if defined(HAS_FMA)
  constexpr bool fused = true;
#else
  constexpr bool fused = false;
#endif
  constexpr bool has_estimate =
      std::same_as<T, float32_t> ||
      (std::same_as<T, float64_t> && x86_recip_has_avx512);
  constexpr bool estimate_based =
      has_estimate && (Tier != Accuracy::Strict || fused);

  NativeWordVec<Tag> y = [&] {
    if constexpr (!estimate_based) {
      // No estimate (f64 pre-AVX512) or no FMA for the Strict ladder: the
      // IEEE composition meets every tier bound.
      if constexpr (IsRsqrt) {
        const auto root = sqrt(tag, x);
        return div(tag, fill_word(tag, T(1)), root);
      } else {
        return div(tag, fill_word(tag, T(1)), x);
      }
    } else if constexpr (Tier == Accuracy::Estimate) {
      if constexpr (IsRsqrt) return x86_rsqrt_estimate<Tag>(tag, x);
      else return x86_rcp_estimate<Tag>(tag, x);
    } else if constexpr (Tier == Accuracy::Fast && fused) {
      const auto estimate = [&] {
        if constexpr (IsRsqrt) return x86_rsqrt_estimate<Tag>(tag, x);
        else return x86_rcp_estimate<Tag>(tag, x);
      }();
      if constexpr (IsRsqrt)
        return x86_rsqrt_markstein<Index>(tag, x, estimate);
      else return x86_recip_markstein<Index>(tag, x, estimate);
    } else if constexpr (Tier == Accuracy::Fast) {
      const auto estimate = [&] {
        if constexpr (IsRsqrt) return x86_rsqrt_estimate<Tag>(tag, x);
        else return x86_rcp_estimate<Tag>(tag, x);
      }();
      if constexpr (IsRsqrt) return x86_rsqrt_plain<Index>(tag, x, estimate);
      else return x86_recip_plain<Index>(tag, x, estimate);
    } else {
      const auto estimate = [&] {
        if constexpr (IsRsqrt) return x86_rsqrt_estimate<Tag>(tag, x);
        else return x86_rcp_estimate<Tag>(tag, x);
      }();
      if constexpr (IsRsqrt)
        return x86_rsqrt_markstein<Index>(
            tag, x, x86_rsqrt_markstein<Index>(tag, x, estimate));
      else
        return x86_recip_markstein<Index>(
            tag, x, x86_recip_markstein<Index>(tag, x, estimate));
    }
  }();

  const T small_bound = std::same_as<T, float32_t> ? T(0x1p-125f) : T(0x1p-1021);
  const T huge_bound = std::same_as<T, float32_t> ? T(0x1p125f) : T(0x1p1021);
  const auto small = cmplt(tag, x, fill_word(tag, small_bound));
  // Both ops repair infinities: the inline Markstein chains would form
  // inf*0 NaNs on those lanes.
  const auto bad = x86_mask_word_or<Tag>(
      small,
      x86_mask_word_or<Tag>(
          cmpgt(tag, x, fill_word(tag, huge_bound)), cmplt(tag, x, fill_word(tag, -huge_bound))));
  if (x86_mask_word_any<Tag>(bad)) {
    y = NativeWordVec<Tag>{x86_recip_repair_word<Tier, IsRsqrt, T>(
        x.value, y.value)};
  }
  return y;
}

/* **************************************************************************** */
//    f16 / bf16 widening route and registration                                //
/* **************************************************************************** */

/**
 * Flushes outputs that are subnormal in the f16 format but normal in f32
 * (f16's smallest normal is 2^-14). Only f16 halves use it; bf16 shares the
 * f32 exponent range, so the f32 pipeline's own bounds already implement
 * the bf16 subnormal contract.
 */
template <Accuracy Tier, nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_recip_flush_narrow(
    Tag tag, NativeWordVec<Tag> y) {
  if constexpr (x86_recip_gradual<Tier>()) {
    return y;
  } else {
    using T = ElementOf<Tag>;
      const T smallest = T(0x1p-14f);
    const auto magnitude = x86_mask_word_and<Tag>(
        cmplt(tag, y, fill_word(tag, smallest)), cmpgt(tag, y, fill_word(tag, -smallest)));
    const auto flush = x86_mask_word_and<Tag>(
        magnitude, cmpne(tag, y, fill_word(tag, T(0))));
    return blend(
        tag, y, flush, mul(tag, y, fill_word(tag, T(0))));
  }
}

/** Widens a narrow word to two f32 words, runs the f32 path, narrows back. */
template <Accuracy Tier, bool IsRsqrt, nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_recip_narrow_word(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  using Raw = decltype(x.value);
  using FloatTag = FixedTag<
      float32_t, static_cast<nint_t>(sizeof(Raw) / sizeof(float32_t))>;
  constexpr FloatTag float_tag{};
  // f16 needs the f16-bound flush before narrowing; bf16 inherits the f32
  // pipeline's own subnormal handling.
  constexpr bool narrow_flush = std::same_as<T, float16_t>;
  if constexpr (sizeof(Raw) == 16) {
    __m128 low;
    __m128 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto finish = [&](NativeWordVec<FloatTag> half) {
      if constexpr (narrow_flush)
        return x86_recip_flush_narrow<Tier, 0>(float_tag, half);
      else
        return half;
    };
    const auto result_low = finish(
        x86_recip_word<Tier, IsRsqrt, 0>(
            float_tag, NativeWordVec<FloatTag>{low}));
    const auto result_high = finish(
        x86_recip_word<Tier, IsRsqrt, 0>(
            float_tag, NativeWordVec<FloatTag>{high}));
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    __m256 low;
    __m256 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto finish = [&](NativeWordVec<FloatTag> half) {
      if constexpr (narrow_flush)
        return x86_recip_flush_narrow<Tier, 0>(float_tag, half);
      else
        return half;
    };
    const auto result_low = finish(
        x86_recip_word<Tier, IsRsqrt, 0>(
            float_tag, NativeWordVec<FloatTag>{low}));
    const auto result_high = finish(
        x86_recip_word<Tier, IsRsqrt, 0>(
            float_tag, NativeWordVec<FloatTag>{high}));
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#endif
#if VEC_WIDTH >= 512
  else {
    __m512 low;
    __m512 high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto finish = [&](NativeWordVec<FloatTag> half) {
      if constexpr (narrow_flush)
        return x86_recip_flush_narrow<Tier, 0>(float_tag, half);
      else
        return half;
    };
    const auto result_low = finish(
        x86_recip_word<Tier, IsRsqrt, 0>(
            float_tag, NativeWordVec<FloatTag>{low}));
    const auto result_high = finish(
        x86_recip_word<Tier, IsRsqrt, 0>(
            float_tag, NativeWordVec<FloatTag>{high}));
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#endif
}

template <Accuracy A, bool IsRsqrt>
struct X86RecipWordImpl {
  template <nint_t Index, FloatingTag Tag, typename Op>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    static_assert(
        std::same_as<Op, RsqrtOp<A>> || std::same_as<Op, RcpOp<A>>);
    if constexpr (
        std::same_as<ElementOf<Tag>, float32_t> ||
        std::same_as<ElementOf<Tag>, float64_t>)
      return x86_recip_word<A, IsRsqrt, Index>(tag, value);
    else
      return x86_recip_narrow_word<A, IsRsqrt, Index>(tag, value);
  }

  template <nint_t Index, FloatingTag Tag, typename Op, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    // Sanitize inactive lanes first: the narrowing conversions and the
    // repair-branch compares must not observe stale values.
    const auto zero = fill_word(
        tag, ElementOf<Tag>{});
    const auto safe = blend(
        tag, zero, mask, value);
    return blend(
        tag, inactive, mask, call<Index>(op, tag, safe));
  }
};

template <Accuracy A>
struct NativeWordImpl<X86Backend, RsqrtOp<A>> : X86RecipWordImpl<A, true> {};

template <Accuracy A>
struct NativeWordImpl<X86Backend, RcpOp<A>> : X86RecipWordImpl<A, false> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_MATH_RECIPROCAL_H
