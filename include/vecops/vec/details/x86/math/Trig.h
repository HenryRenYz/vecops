// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// The project copyright above applies to project-owned portions only.
// SPDX-FileCopyrightText: 2010-2025 Naoki Shibata and contributors
// SLEEF-derived sets retain BSL-1.0; other project-owned portions use MIT.
// Original source notice: Copyright Naoki Shibata and contributors 2010 - 2025.
// The compared older reference carries the corresponding 2010 - 2024 notice.
// Complete upstream terms: LICENSES/SLEEF-BSL-1.0.txt.
// The existing LLVM design reference is retained; no verbatim LLVM
// function was identified. See THIRD_PARTY_NOTICES.md and LICENSES/LLVM-libc.txt.
// SPDX-License-Identifier: MIT AND BSL-1.0
#ifndef VECOPS_VEC_DETAILS_X86_MATH_TRIG_H
#define VECOPS_VEC_DETAILS_X86_MATH_TRIG_H

/**
 * @file Trig.h
 * @brief x86 backend implementations for the trigonometric families.
 *
 * The fused and tangent hot kernels use quadrant reduction to
 * [-pi/4, pi/4]; unary sin/cos use a cheaper half-period reduction to
 * [-pi/2, pi/2] and one odd polynomial. Both use split pi constants and FMA
 * polynomial evaluation. Their design follows SLEEF's
 * SIMD sin/cos/tan kernels (src/libm/sleefsimd{sp,dp}.c, Boost Software
 * License 1.0) and LLVM libc's table-free small-range kernels
 * (libc/src/__support/math/{sin,cos,tan}{f,}.h, Apache-2.0 WITH
 * LLVM-exception). Coefficients below are the ordinary Taylor coefficients,
 * selected at compile time for the documented tier and reduction interval.
 * The Strict f32/f64 unary coefficients are the SLEEF u35 sets; the remaining
 * coefficients are ordinary Taylor coefficients.
 *
 * Radian inputs outside the inexpensive Cody-Waite interval take a cold
 * per-lane long-double libm path. This keeps the common path branch-free and
 * gives the complete finite domain a high-quality Payne-Hanek reduction from
 * the platform libm. Strict additionally repairs lanes close to roots or tan
 * poles, where a tiny range-reduction error is many ULPs of the result.
 *
 * Pi-scaled operations never form pi*x before reduction. The vector path
 * rounds 2*x and subtracts an exact half-integer; the cold path reduces by
 * the exactly representable periods 1 or 2 before multiplying the small
 * remainder by long-double pi. Integer and half-integer lanes are repaired
 * explicitly, including the C23/oneMKL signs of zero and infinity.
 *
 * bf16 and f16 Strict widen exactly to paired f32 words. Under AVX512-FP16,
 * every f16 Fast/Estimate operation instead uses native-half reduction/FMA
 * kernels. Large arguments and sensitive tangent poles still fall back to
 * paired f32 so the common path remains fast without weakening its contract.
 */

#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <type_traits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/x86/Arithmetic.h"
#include "vecops/vec/details/x86/Basic.h"
#include "vecops/vec/details/x86/Types.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    Reduction and polynomial kernels                                           //
/* **************************************************************************** */

template <typename T>
constexpr T x86_trig_radian_hot_limit() {
#if defined(HAS_FMA)
  if constexpr (std::same_as<T, float32_t>) return T(8192.0f);
  else return T(65536.0);
#else
  // SLEEF uses the same conservative first-stage bounds when a fused
  // subtraction is unavailable.
  if constexpr (std::same_as<T, float32_t>) return T(125.0f);
  else return T(15.0);
#endif
}

template <typename T>
constexpr T x86_trig_pi_hot_limit() {
  // 2*x and the half-integer subtraction are exact through these bounds.
  if constexpr (std::same_as<T, float32_t>) return T(0x1p21f);
  else return T(0x1p50);
}

template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_reduce_radians(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag> q) {
  using T = ElementOf<Tag>;
  if constexpr (std::same_as<T, float32_t>) {
    auto r = fnmadd(tag, q, fill_word(tag, T(1.5703125f)), x);
    r = fnmadd(tag, q, fill_word(tag, T(0.00048351287841796875f)), r);
    r = fnmadd(tag, q, fill_word(tag, T(3.1385570764541626e-7f)), r);
    return fnmadd(tag, q, fill_word(tag, T(6.0771006282767104e-11f)), r);
  } else {
    auto r = fnmadd(
        tag, q, fill_word(tag, T(0x1.921fb54442d18p+0)), x);
    return fnmadd(
        tag, q, fill_word(tag, T(0x1.1a62633145c07p-54)), r);
  }
}

template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_reduce_pi(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag> q) {
  using T = ElementOf<Tag>;
  if constexpr (std::same_as<T, float32_t>) {
    auto r = fnmadd(tag, q, fill_word(tag, T(3.140625f)), x);
    r = fnmadd(tag, q, fill_word(tag, T(0.0009670257568359375f)), r);
    r = fnmadd(tag, q, fill_word(tag, T(6.2771141529083252e-7f)), r);
    return fnmadd(tag, q, fill_word(tag, T(1.2154201256553421e-10f)), r);
  } else {
    auto r = fnmadd(
        tag, q, fill_word(tag, T(0x1.921fb54442d18p+1)), x);
    return fnmadd(
        tag, q, fill_word(tag, T(0x1.1a62633145c07p-53)), r);
  }
}

/** q modulo four, represented in the same floating format as q. */
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_quadrant(
    Tag tag, NativeWordVec<Tag> q) {
  using T = ElementOf<Tag>;
  const auto four = fill_word(tag, T(4));
  const auto blocks = floor(tag, mul(tag, q, fill_word(tag, T(0.25))));
  return fnmadd(tag, blocks, four, q);
}

template <Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_sin_poly(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  const auto z = mul(tag, x, x);
  NativeWordVec<Tag> p;
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (Tier == Accuracy::Strict) {
      p = fmadd(tag, z, fill_word(tag, T(1.0 / 362880.0)),
                fill_word(tag, T(-1.0 / 5040.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 120.0)));
    } else if constexpr (Tier == Accuracy::Fast) {
      p = fill_word(tag, T(1.0 / 120.0));
    } else {
      p = fill_word(tag, T(0));
    }
  } else {
    if constexpr (Tier == Accuracy::Strict) {
      p = fmadd(tag, z, fill_word(tag, T(1.0 / 355687428096000.0)),
                fill_word(tag, T(-1.0 / 1307674368000.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 6227020800.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 39916800.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 362880.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 5040.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 120.0)));
    } else if constexpr (Tier == Accuracy::Fast) {
      p = fmadd(tag, z, fill_word(tag, T(1.0 / 362880.0)),
                fill_word(tag, T(-1.0 / 5040.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 120.0)));
    } else {
      p = fill_word(tag, T(1.0 / 120.0));
    }
  }
  p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 6.0)));
  return fmadd(tag, mul(tag, z, x), p, x);
}

template <Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_cos_poly(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  const auto z = mul(tag, x, x);
  NativeWordVec<Tag> p;
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (Tier == Accuracy::Strict) {
      p = fmadd(tag, z, fill_word(tag, T(-1.0 / 3628800.0)),
                fill_word(tag, T(1.0 / 40320.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 720.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 24.0)));
    } else {
      p = fmadd(tag, z, fill_word(tag, T(-1.0 / 720.0)),
                fill_word(tag, T(1.0 / 24.0)));
    }
  } else {
    if constexpr (Tier == Accuracy::Strict) {
      p = fmadd(tag, z, fill_word(tag, T(1.0 / 20922789888000.0)),
                fill_word(tag, T(-1.0 / 87178291200.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 479001600.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 3628800.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 40320.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 720.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 24.0)));
    } else if constexpr (Tier == Accuracy::Fast) {
      p = fmadd(tag, z, fill_word(tag, T(-1.0 / 3628800.0)),
                fill_word(tag, T(1.0 / 40320.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 720.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 24.0)));
    } else {
      p = fmadd(tag, z, fill_word(tag, T(-1.0 / 720.0)),
                fill_word(tag, T(1.0 / 24.0)));
    }
  }
  p = fmadd(tag, z, p, fill_word(tag, T(-0.5)));
  return fmadd(tag, z, p, fill_word(tag, T(1)));
}

/** Odd sine polynomial on [-pi/2, pi/2], used by unary sin and cos. */
template <Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_wide_sin_poly(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  const auto z = mul(tag, x, x);
  NativeWordVec<Tag> p;
  if constexpr (std::same_as<T, float32_t>) {
    if constexpr (Tier == Accuracy::Strict) {
      p = fmadd(tag, z, fill_word(tag, T(2.6083159809786594e-6f)),
                fill_word(tag, T(-0.00019810690719168633f)));
      p = fmadd(tag, z, p, fill_word(tag, T(0.00833307858556509f)));
      p = fmadd(tag, z, p, fill_word(tag, T(-0.16666659712791443f)));
    } else if constexpr (Tier == Accuracy::Fast) {
      p = fmadd(tag, z, fill_word(tag, T(-1.0 / 5040.0)),
                fill_word(tag, T(1.0 / 120.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 6.0)));
    } else {
      p = fmadd(tag, z, fill_word(tag, T(1.0 / 120.0)),
                fill_word(tag, T(-1.0 / 6.0)));
    }
  } else {
    if constexpr (Tier == Accuracy::Strict) {
      p = fmadd(tag, z, fill_word(tag, T(-7.9725595500903787e-18)),
                fill_word(tag, T(2.810099727108632e-15)));
      p = fmadd(tag, z, p, fill_word(tag, T(-7.6471221911815883e-13)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.605904306056645e-10)));
      p = fmadd(tag, z, p, fill_word(tag, T(-2.5052108376350205e-8)));
      p = fmadd(tag, z, p, fill_word(tag, T(2.7557319223919875e-6)));
      p = fmadd(tag, z, p, fill_word(tag, T(-0.00019841269841269616)));
      p = fmadd(tag, z, p, fill_word(tag, T(0.0083333333333333297)));
      p = fmadd(tag, z, p, fill_word(tag, T(-0.16666666666666666)));
    } else if constexpr (Tier == Accuracy::Fast) {
      p = fmadd(tag, z, fill_word(tag, T(1.0 / 6227020800.0)),
                fill_word(tag, T(-1.0 / 39916800.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 362880.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 5040.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 120.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 6.0)));
    } else {
      p = fmadd(tag, z, fill_word(tag, T(1.0 / 362880.0)),
                fill_word(tag, T(-1.0 / 5040.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(1.0 / 120.0)));
      p = fmadd(tag, z, p, fill_word(tag, T(-1.0 / 6.0)));
    }
  }
  return fmadd(tag, mul(tag, z, x), p, x);
}

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier, FloatingTag Tag>
  requires (Kind != TrigKind::Tan)
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_hot_sin_or_cos(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag>& reduced) {
  using T = ElementOf<Tag>;
  NativeWordVec<Tag> q;
  NativeWordVec<Tag> quadrant_source;
  if constexpr (Kind == TrigKind::Sin) {
    if constexpr (Unit == TrigUnit::Radians) {
      q = round_even(tag, mul(
          tag, x, fill_word(tag, T(0.31830988618379067154))));
      reduced = x86_trig_reduce_pi(tag, x, q);
    } else {
      q = round_even(tag, x);
      reduced = mul(
          tag, sub(tag, x, q),
          fill_word(tag, T(3.14159265358979323846)));
    }
    quadrant_source = q;
  } else {
    if constexpr (Unit == TrigUnit::Radians) {
      q = round_even(tag, sub(
          tag, mul(tag, x, fill_word(tag, T(0.31830988618379067154))),
          fill_word(tag, T(0.5))));
      quadrant_source = fmadd(
          tag, q, fill_word(tag, T(2)), fill_word(tag, T(1)));
      reduced = x86_trig_reduce_radians(tag, x, quadrant_source);
    } else {
      q = round_even(tag, sub(tag, x, fill_word(tag, T(0.5))));
      quadrant_source = fmadd(
          tag, q, fill_word(tag, T(2)), fill_word(tag, T(1)));
      reduced = mul(
          tag,
          fnmadd(tag, quadrant_source, fill_word(tag, T(0.5)), x),
          fill_word(tag, T(3.14159265358979323846)));
    }
  }
  const auto poly = x86_trig_wide_sin_poly<Tier>(tag, reduced);
  const auto quadrant = x86_trig_quadrant(tag, quadrant_source);
  if constexpr (Kind == TrigKind::Sin) {
    const auto flip = x86_mask_word_or<Tag>(
        cmpeq(tag, quadrant, fill_word(tag, T(1))),
        cmpeq(tag, quadrant, fill_word(tag, T(3))));
    return blend(tag, poly, flip, neg(tag, poly));
  } else {
    const auto flip = cmpeq(tag, quadrant, fill_word(tag, T(1)));
    return blend(tag, poly, flip, neg(tag, poly));
  }
}

/** Applies q's quadrant to sin(r), cos(r). */
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE void x86_trig_reconstruct(
    Tag tag, NativeWordVec<Tag> q, NativeWordVec<Tag> sin_r,
    NativeWordVec<Tag> cos_r, NativeWordVec<Tag>& sin_x,
    NativeWordVec<Tag>& cos_x) {
  using T = ElementOf<Tag>;
  const auto one = cmpeq(tag, q, fill_word(tag, T(1)));
  const auto two = cmpeq(tag, q, fill_word(tag, T(2)));
  const auto three = cmpeq(tag, q, fill_word(tag, T(3)));
  const auto neg_sin = neg(tag, sin_r);
  const auto neg_cos = neg(tag, cos_r);
  sin_x = blend(tag, sin_r, one, cos_r);
  sin_x = blend(tag, sin_x, two, neg_sin);
  sin_x = blend(tag, sin_x, three, neg_cos);
  cos_x = blend(tag, cos_r, one, neg_sin);
  cos_x = blend(tag, cos_x, two, neg_cos);
  cos_x = blend(tag, cos_x, three, sin_r);
}

template <TrigUnit Unit, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE void x86_trig_hot_sincos(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag>& sin_x,
    NativeWordVec<Tag>& cos_x, NativeWordVec<Tag>& reduced) {
  using T = ElementOf<Tag>;
  NativeWordVec<Tag> q;
  if constexpr (Unit == TrigUnit::Radians) {
    q = round_even(tag, mul(
        tag, x, fill_word(tag, T(0.63661977236758134308))));
    reduced = x86_trig_reduce_radians(tag, x, q);
  } else {
    q = round_even(tag, mul(tag, x, fill_word(tag, T(2))));
    reduced = fnmadd(tag, q, fill_word(tag, T(0.5)), x);
    reduced = mul(
        tag, reduced,
        fill_word(tag, T(3.14159265358979323846264338327950288L)));
  }
  const auto quadrant = x86_trig_quadrant(tag, q);
  x86_trig_reconstruct(
      tag, quadrant, x86_trig_sin_poly<Tier>(tag, reduced),
      x86_trig_cos_poly<Tier>(tag, reduced), sin_x, cos_x);
}

/* **************************************************************************** */
//    Scalar cold path and lane repair                                           //
/* **************************************************************************** */

template <TrigKind Kind, TrigUnit Unit, typename T>
T x86_trig_scalar(T input) {
  const long double x = static_cast<long double>(input);
  if (!std::isfinite(x)) {
    if (std::isnan(x)) return static_cast<T>(x + x);
    return std::numeric_limits<T>::quiet_NaN();
  }
  if constexpr (Unit == TrigUnit::Radians) {
    if constexpr (Kind == TrigKind::Sin)
      return static_cast<T>(std::sin(x));
    else if constexpr (Kind == TrigKind::Cos)
      return static_cast<T>(std::cos(x));
    else
      return static_cast<T>(std::tan(x));
  } else {
    const long double frac = std::remainder(x, 1.0L);
    const bool integer = frac == 0.0L;
    const bool half_integer = std::abs(frac) == 0.5L;
    if constexpr (Kind == TrigKind::Sin) {
      if (integer) return std::copysign(T(0), input);
      const long double reduced = std::remainder(x, 2.0L);
      return static_cast<T>(std::sin(std::numbers::pi_v<long double> * reduced));
    } else if constexpr (Kind == TrigKind::Cos) {
      if (half_integer) return T(0);
      if (integer) {
        const bool odd = std::abs(std::remainder(x, 2.0L)) == 1.0L;
        return odd ? T(-1) : T(1);
      }
      const long double reduced = std::remainder(x, 2.0L);
      return static_cast<T>(std::cos(std::numbers::pi_v<long double> * reduced));
    } else {
      if (integer) {
        const bool odd = std::abs(std::remainder(x, 2.0L)) == 1.0L;
        return std::copysign(T(0), odd ? -input : input);
      }
      if (half_integer) {
        const long double n = std::floor(x);
        const bool odd = std::abs(std::remainder(n, 2.0L)) == 1.0L;
        return std::copysign(
            std::numeric_limits<T>::infinity(), odd ? T(-1) : T(1));
      }
      return static_cast<T>(
          std::tan(std::numbers::pi_v<long double> * frac));
    }
  }
}

template <Accuracy Tier, TrigKind Kind, TrigUnit Unit, typename T>
bool x86_trig_lane_needs_repair(T input, T output) {
  if (!std::isfinite(static_cast<long double>(input))) return true;
  const T limit = Unit == TrigUnit::Radians
      ? x86_trig_radian_hot_limit<T>() : x86_trig_pi_hot_limit<T>();
  if (std::abs(input) > limit) return true;
  if constexpr (Unit == TrigUnit::Pi) {
    const long double frac = std::remainder(
        static_cast<long double>(input), 1.0L);
    if (frac == 0.0L || std::abs(frac) == 0.5L) return true;
  }
  if constexpr (Tier == Accuracy::Strict) {
    const T root = T(0x1p-10);
    if (input != T(0) && std::abs(output) < root) return true;
    if constexpr (Kind == TrigKind::Tan)
      if (std::abs(output) > T(1) / root) return true;
  }
  return false;
}

template <Accuracy Tier, TrigKind Kind, TrigUnit Unit, typename T, typename Raw>
VECOPS_NOINLINE Raw x86_trig_repair_word(Raw raw_input, Raw raw_output) {
  constexpr std::size_t lanes = sizeof(Raw) / sizeof(T);
  alignas(64) T inputs[lanes];
  alignas(64) T outputs[lanes];
  std::memcpy(inputs, &raw_input, sizeof(Raw));
  std::memcpy(outputs, &raw_output, sizeof(Raw));
  for (std::size_t lane = 0; lane < lanes; ++lane)
    if (x86_trig_lane_needs_repair<Tier, Kind, Unit>(
            inputs[lane], outputs[lane]))
      outputs[lane] = x86_trig_scalar<Kind, Unit>(inputs[lane]);
  Raw result;
  std::memcpy(&result, outputs, sizeof(Raw));
  return result;
}

template <Accuracy Tier, TrigUnit Unit, typename T, typename Raw>
VECOPS_NOINLINE void x86_sincos_repair_word(
    Raw raw_input, Raw& raw_sin, Raw& raw_cos) {
  constexpr std::size_t lanes = sizeof(Raw) / sizeof(T);
  alignas(64) T inputs[lanes];
  alignas(64) T sin_values[lanes];
  alignas(64) T cos_values[lanes];
  std::memcpy(inputs, &raw_input, sizeof(Raw));
  std::memcpy(sin_values, &raw_sin, sizeof(Raw));
  std::memcpy(cos_values, &raw_cos, sizeof(Raw));
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const bool repair_sin = x86_trig_lane_needs_repair<
        Tier, TrigKind::Sin, Unit>(inputs[lane], sin_values[lane]);
    const bool repair_cos = x86_trig_lane_needs_repair<
        Tier, TrigKind::Cos, Unit>(inputs[lane], cos_values[lane]);
    if (repair_sin || repair_cos) {
      sin_values[lane] = x86_trig_scalar<TrigKind::Sin, Unit>(inputs[lane]);
      cos_values[lane] = x86_trig_scalar<TrigKind::Cos, Unit>(inputs[lane]);
    }
  }
  std::memcpy(&raw_sin, sin_values, sizeof(Raw));
  std::memcpy(&raw_cos, cos_values, sizeof(Raw));
}

/* **************************************************************************** */
//    Wide and narrow word kernels                                               //
/* **************************************************************************** */

template <TrigUnit Unit, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordMask<Tag> x86_trig_cold_mask(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag> reduced) {
  using T = ElementOf<Tag>;
  const T limit = Unit == TrigUnit::Radians
      ? x86_trig_radian_hot_limit<T>() : x86_trig_pi_hot_limit<T>();
  auto cold = cmpgt(tag, abs(tag, x), fill_word(tag, limit));
  cold = x86_mask_word_or<Tag>(cold, cmpne(tag, x, x));
  if constexpr (Unit == TrigUnit::Pi)
    cold = x86_mask_word_or<Tag>(
        cold, cmpeq(tag, reduced, fill_word(tag, T(0))));
  return cold;
}

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_word(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  const T limit = Unit == TrigUnit::Radians
      ? x86_trig_radian_hot_limit<T>() : x86_trig_pi_hot_limit<T>();
  const auto outside = cmpgt(tag, abs(tag, x), fill_word(tag, limit));
  const auto unordered = cmpne(tag, x, x);
  const auto sanitize = x86_mask_word_or<Tag>(outside, unordered);
  const auto safe = blend(tag, x, sanitize, fill_word(tag, T(0)));
  NativeWordVec<Tag> reduced;
  auto result = [&] {
    if constexpr (Kind == TrigKind::Tan) {
      NativeWordVec<Tag> sin_x;
      NativeWordVec<Tag> cos_x;
      x86_trig_hot_sincos<Unit, Tier>(
          tag, safe, sin_x, cos_x, reduced);
      return div(tag, sin_x, cos_x);
    } else {
      return x86_trig_hot_sin_or_cos<Kind, Unit, Tier>(
          tag, safe, reduced);
    }
  }();
  if constexpr (Kind != TrigKind::Cos)
    result = blend(
        tag, result, cmpeq(tag, x, fill_word(tag, T(0))), x);
  auto cold = x86_trig_cold_mask<Unit>(tag, x, reduced);
  if constexpr (Tier == Accuracy::Strict) {
    const T root = T(0x1p-10);
    const auto nonzero = cmpne(tag, x, fill_word(tag, T(0)));
    auto near = x86_mask_word_and<Tag>(
        cmplt(tag, abs(tag, result), fill_word(tag, root)), nonzero);
    if constexpr (Kind == TrigKind::Tan)
      near = x86_mask_word_or<Tag>(
          near, cmpgt(tag, abs(tag, result), fill_word(tag, T(1) / root)));
    cold = x86_mask_word_or<Tag>(cold, near);
  }
  if (x86_mask_word_any<Tag>(cold))
    result = NativeWordVec<Tag>{
        x86_trig_repair_word<Tier, Kind, Unit, T>(x.value, result.value)};
  return result;
}

template <TrigUnit Unit, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE void x86_sincos_word(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag>& sin_x,
    NativeWordVec<Tag>& cos_x) {
  using T = ElementOf<Tag>;
  const T limit = Unit == TrigUnit::Radians
      ? x86_trig_radian_hot_limit<T>() : x86_trig_pi_hot_limit<T>();
  auto sanitize = cmpgt(tag, abs(tag, x), fill_word(tag, limit));
  sanitize = x86_mask_word_or<Tag>(sanitize, cmpne(tag, x, x));
  const auto safe = blend(tag, x, sanitize, fill_word(tag, T(0)));
  NativeWordVec<Tag> reduced;
  x86_trig_hot_sincos<Unit, Tier>(tag, safe, sin_x, cos_x, reduced);
  sin_x = blend(
      tag, sin_x, cmpeq(tag, x, fill_word(tag, T(0))), x);
  auto cold = x86_trig_cold_mask<Unit>(tag, x, reduced);
  if constexpr (Tier == Accuracy::Strict) {
    const T root = T(0x1p-10);
    const auto nonzero = cmpne(tag, x, fill_word(tag, T(0)));
    const auto near_sin = x86_mask_word_and<Tag>(
        cmplt(tag, abs(tag, sin_x), fill_word(tag, root)), nonzero);
    const auto near_cos = cmplt(tag, abs(tag, cos_x), fill_word(tag, root));
    cold = x86_mask_word_or<Tag>(
        cold, x86_mask_word_or<Tag>(near_sin, near_cos));
  }
  if (x86_mask_word_any<Tag>(cold))
    x86_sincos_repair_word<Tier, Unit, T>(
        x.value, sin_x.value, cos_x.value);
}

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier,
          nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_narrow_word(
    Tag tag, NativeWordVec<Tag> x) {
  using T = ElementOf<Tag>;
  using Raw = decltype(x.value);
  using FloatTag = FixedTag<
      float32_t, static_cast<nint_t>(sizeof(Raw) / sizeof(float32_t))>;
  constexpr FloatTag float_tag{};
  if constexpr (sizeof(Raw) == 16) {
    __m128 low, high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_trig_word<Kind, Unit, Tier>(
        float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_trig_word<Kind, Unit, Tier>(
        float_tag, NativeWordVec<FloatTag>{high});
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    __m256 low, high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_trig_word<Kind, Unit, Tier>(
        float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_trig_word<Kind, Unit, Tier>(
        float_tag, NativeWordVec<FloatTag>{high});
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
    __m512 low, high;
    if constexpr (std::same_as<T, bfloat16_t>)
      x86_bfloat16_to_float32_pair(x.value, low, high);
    else
      x86_float16_to_float32_pair(x.value, low, high);
    const auto result_low = x86_trig_word<Kind, Unit, Tier>(
        float_tag, NativeWordVec<FloatTag>{low});
    const auto result_high = x86_trig_word<Kind, Unit, Tier>(
        float_tag, NativeWordVec<FloatTag>{high});
    if constexpr (std::same_as<T, bfloat16_t>)
      return NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(
          result_low.value, result_high.value)};
    else
      return NativeWordVec<Tag>{x86_float32_pair_to_float16(
          result_low.value, result_high.value)};
  }
#endif
}

template <TrigUnit Unit, Accuracy Tier, nint_t Index, FloatingTag Tag>
VECOPS_ALWAYS_INLINE void x86_sincos_narrow_word(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag>& sin_x,
    NativeWordVec<Tag>& cos_x) {
  using T = ElementOf<Tag>;
  using Raw = decltype(x.value);
  using FloatTag = FixedTag<
      float32_t, static_cast<nint_t>(sizeof(Raw) / sizeof(float32_t))>;
  constexpr FloatTag float_tag{};
#define VECOPS_X86_SINCOS_NARROW_BODY(RawFloat)                         \
    RawFloat low, high;                                                 \
    if constexpr (std::same_as<T, bfloat16_t>)                          \
      x86_bfloat16_to_float32_pair(x.value, low, high);                 \
    else                                                                \
      x86_float16_to_float32_pair(x.value, low, high);                  \
    NativeWordVec<FloatTag> sin_low, cos_low, sin_high, cos_high;       \
    x86_sincos_word<Unit, Tier>(                                       \
        float_tag, NativeWordVec<FloatTag>{low}, sin_low, cos_low);     \
    x86_sincos_word<Unit, Tier>(                                       \
        float_tag, NativeWordVec<FloatTag>{high}, sin_high, cos_high);  \
    if constexpr (std::same_as<T, bfloat16_t>) {                        \
      sin_x = NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(          \
          sin_low.value, sin_high.value)};                             \
      cos_x = NativeWordVec<Tag>{x86_float32_pair_to_bfloat16(          \
          cos_low.value, cos_high.value)};                             \
    } else {                                                            \
      sin_x = NativeWordVec<Tag>{x86_float32_pair_to_float16(           \
          sin_low.value, sin_high.value)};                             \
      cos_x = NativeWordVec<Tag>{x86_float32_pair_to_float16(           \
          cos_low.value, cos_high.value)};                             \
    }
  if constexpr (sizeof(Raw) == 16) {
    VECOPS_X86_SINCOS_NARROW_BODY(__m128)
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    VECOPS_X86_SINCOS_NARROW_BODY(__m256)
  }
#endif
#if VEC_WIDTH >= 512
  else {
    VECOPS_X86_SINCOS_NARROW_BODY(__m512)
  }
#endif
#undef VECOPS_X86_SINCOS_NARROW_BODY
}

#if defined(HAS_AVX512_FP16)

/* **************************************************************************** */
//    Native AVX512-FP16 Fast/Estimate kernels                                  //
/* **************************************************************************** */

/**
 * Native-half hot domain. The quadrant integer must remain exact in fp16;
 * larger lanes fall back to the existing paired-f32 path. The radian bound is
 * deliberately tighter because its split-pi reduction also accumulates in
 * fp16, whereas the pi-scaled subtraction is exact through 1024.
 */
template <TrigUnit Unit>
VECOPS_ALWAYS_INLINE float16_t x86_trig_f16_hot_limit() {
  if constexpr (Unit == TrigUnit::Radians) return float16_t(32.0f);
  else return float16_t(1024.0f);
}

template <Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_f16_sin_poly(
    Tag tag, NativeWordVec<Tag> x) {
  const auto z = mul(tag, x, x);
  auto p = fill_word(tag, float16_t(-1.0f / 6.0f));
  if constexpr (Tier == Accuracy::Fast)
    p = fmadd(tag, z, fill_word(tag, float16_t(1.0f / 120.0f)), p);
  return fmadd(tag, mul(tag, z, x), p, x);
}

template <Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_f16_cos_poly(
    Tag tag, NativeWordVec<Tag> x) {
  const auto z = mul(tag, x, x);
  auto p = fill_word(tag, float16_t(-0.5f));
  if constexpr (Tier == Accuracy::Fast)
    p = fmadd(tag, z, fill_word(tag, float16_t(1.0f / 24.0f)), p);
  return fmadd(tag, z, p, fill_word(tag, float16_t(1.0f)));
}

template <TrigUnit Unit, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE void x86_sincos_f16_native_hot(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag>& sin_x,
    NativeWordVec<Tag>& cos_x) {
  NativeWordVec<Tag> q;
  NativeWordVec<Tag> r;
  if constexpr (Unit == TrigUnit::Radians) {
    q = round_even(tag, mul(
        tag, x, fill_word(tag, float16_t(0.63661977236758134308f))));
    r = fnmadd(tag, q, fill_word(tag, float16_t(1.5703125f)), x);
    r = fnmadd(
        tag, q, fill_word(tag, float16_t(0.00048351287841796875f)), r);
  } else {
    q = round_even(tag, mul(tag, x, fill_word(tag, float16_t(2.0f))));
    r = fnmadd(tag, q, fill_word(tag, float16_t(0.5f)), x);
    r = mul(tag, r, fill_word(tag, float16_t(3.14159265358979323846f)));
  }
  x86_trig_reconstruct(
      tag, x86_trig_quadrant(tag, q),
      x86_trig_f16_sin_poly<Tier>(tag, r),
      x86_trig_f16_cos_poly<Tier>(tag, r), sin_x, cos_x);
  if constexpr (Unit == TrigUnit::Pi) {
    const auto fraction = abs(tag, sub(tag, x, round_even(tag, x)));
    const auto integer =
        cmpeq(tag, fraction, fill_word(tag, float16_t(0.0f)));
    const auto half =
        cmpeq(tag, fraction, fill_word(tag, float16_t(0.5f)));
    sin_x = blend(
        tag, sin_x, integer,
        copysign(tag, fill_word(tag, float16_t(0.0f)), x));
    cos_x = blend(tag, cos_x, half, fill_word(tag, float16_t(0.0f)));
  }
}

template <TrigUnit Unit, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordMask<Tag> x86_trig_f16_cold_mask(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag> cos_x) {
  auto cold = cmpgt(
      tag, abs(tag, x), fill_word(tag, x86_trig_f16_hot_limit<Unit>()));
  cold = x86_mask_word_or<Tag>(cold, cmpne(tag, x, x));
  // The half-precision quotient is least stable close to tangent poles.
  return x86_mask_word_or<Tag>(
      cold, cmplt(tag, abs(tag, cos_x),
                  fill_word(tag, float16_t(0.125f))));
}

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier,
          nint_t Index, FloatingTag Tag>
  requires (Tier != Accuracy::Strict &&
            std::same_as<ElementOf<Tag>, float16_t>)
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> x86_trig_f16_native_word(
    Tag tag, NativeWordVec<Tag> x) {
  NativeWordVec<Tag> sin_x;
  NativeWordVec<Tag> cos_x;
  x86_sincos_f16_native_hot<Unit, Tier>(tag, x, sin_x, cos_x);
  auto result = [&] {
    if constexpr (Kind == TrigKind::Sin) return sin_x;
    else if constexpr (Kind == TrigKind::Cos) return cos_x;
    else return div(tag, sin_x, cos_x);
  }();
  if constexpr (Kind != TrigKind::Cos)
    result = blend(
        tag, result, cmpeq(tag, x, fill_word(tag, float16_t(0.0f))), x);

  auto cold = x86_trig_f16_cold_mask<Unit>(tag, x, cos_x);
  if constexpr (Kind != TrigKind::Tan) {
    // A small cosine is harmless when only sin/cos is requested.
    cold = cmpgt(
        tag, abs(tag, x), fill_word(tag, x86_trig_f16_hot_limit<Unit>()));
    cold = x86_mask_word_or<Tag>(cold, cmpne(tag, x, x));
  }
  if (x86_mask_word_any<Tag>(cold)) {
    const auto widened =
        x86_trig_narrow_word<Kind, Unit, Tier, Index>(tag, x);
    result = blend(tag, result, cold, widened);
  }
  return result;
}

template <TrigUnit Unit, Accuracy Tier, nint_t Index, FloatingTag Tag>
  requires (Tier != Accuracy::Strict &&
            std::same_as<ElementOf<Tag>, float16_t>)
VECOPS_ALWAYS_INLINE void x86_sincos_f16_native_word(
    Tag tag, NativeWordVec<Tag> x, NativeWordVec<Tag>& sin_x,
    NativeWordVec<Tag>& cos_x) {
  x86_sincos_f16_native_hot<Unit, Tier>(tag, x, sin_x, cos_x);
  sin_x = blend(
      tag, sin_x, cmpeq(tag, x, fill_word(tag, float16_t(0.0f))), x);
  auto cold = cmpgt(
      tag, abs(tag, x), fill_word(tag, x86_trig_f16_hot_limit<Unit>()));
  cold = x86_mask_word_or<Tag>(cold, cmpne(tag, x, x));
  if (x86_mask_word_any<Tag>(cold)) {
    NativeWordVec<Tag> wide_sin;
    NativeWordVec<Tag> wide_cos;
    x86_sincos_narrow_word<Unit, Tier, Index>(
        tag, x, wide_sin, wide_cos);
    sin_x = blend(tag, sin_x, cold, wide_sin);
    cos_x = blend(tag, cos_x, cold, wide_cos);
  }
}

#endif // HAS_AVX512_FP16

/* **************************************************************************** */
//    NativeWordImpl registration                                                //
/* **************************************************************************** */

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier>
struct X86TrigWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      TrigOp<Kind, Unit, Tier>, Tag tag, NativeWordVec<Tag> value) {
    if constexpr (
        std::same_as<ElementOf<Tag>, float32_t> ||
        std::same_as<ElementOf<Tag>, float64_t>)
      return x86_trig_word<Kind, Unit, Tier>(tag, value);
#if defined(HAS_AVX512_FP16)
    else if constexpr (std::same_as<ElementOf<Tag>, float16_t> &&
                       Tier != Accuracy::Strict)
      return x86_trig_f16_native_word<Kind, Unit, Tier, Index>(tag, value);
#endif
    else
      return x86_trig_narrow_word<Kind, Unit, Tier, Index>(tag, value);
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      TrigOp<Kind, Unit, Tier> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    const auto safe = blend(
        tag, fill_word(tag, ElementOf<Tag>(0)), mask, value);
    return blend(tag, inactive, mask, call<Index>(op, tag, safe));
  }
};

template <TrigUnit Unit, Accuracy Tier>
struct X86SinCosWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE void call(
      SinCosOp<Unit, Tier>, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag>& sin_out, NativeWordVec<Tag>& cos_out) {
    if constexpr (
        std::same_as<ElementOf<Tag>, float32_t> ||
        std::same_as<ElementOf<Tag>, float64_t>)
      x86_sincos_word<Unit, Tier>(tag, value, sin_out, cos_out);
#if defined(HAS_AVX512_FP16)
    else if constexpr (std::same_as<ElementOf<Tag>, float16_t> &&
                       Tier != Accuracy::Strict)
      x86_sincos_f16_native_word<Unit, Tier, Index>(
          tag, value, sin_out, cos_out);
#endif
    else
      x86_sincos_narrow_word<Unit, Tier, Index>(
          tag, value, sin_out, cos_out);
  }
};

template <TrigKind Kind, TrigUnit Unit, Accuracy Tier>
struct NativeWordImpl<X86Backend, TrigOp<Kind, Unit, Tier>>
    : X86TrigWordImpl<Kind, Unit, Tier> {};

template <TrigUnit Unit, Accuracy Tier>
struct NativeWordImpl<X86Backend, SinCosOp<Unit, Tier>>
    : X86SinCosWordImpl<Unit, Tier> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_MATH_TRIG_H
