#ifndef VECOPS_VEC_DETAILS_SCALAR_MATH_TRIG_H
#define VECOPS_VEC_DETAILS_SCALAR_MATH_TRIG_H

/**
 * @file Trig.h
 * @brief Scalar backend implementations for the trigonometric families.
 *
 * Radian functions use the matching scalar libm overload. The pi-scaled
 * family first reduces the input in half-revolutions and evaluates only a
 * small long-double argument; it never forms `pi * x` before reduction.
 * Integer and half-integer inputs are handled separately so the C23 exact
 * values, signed zeros, and tangent poles do not depend on an approximation
 * of pi. All accuracy tiers share this libm-backed scalar implementation.
 */

#include <cmath>
#include <limits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Wordwise.h"
#include "vecops/vec/details/scalar/Basic.h"

namespace vecops::vec::details {

inline constexpr long double kScalarTrigPi =
    3.141592653589793238462643383279502884L;

template <typename T>
VECOPS_ALWAYS_INLINE T scalar_trig_narrow(long double value) {
  if constexpr (std::same_as<T, float64_t>)
    return static_cast<T>(static_cast<double>(value));
  else if constexpr (std::same_as<T, float32_t>)
    return static_cast<T>(static_cast<float>(value));
  else
    return T{static_cast<float>(value)};
}

template <typename T>
VECOPS_ALWAYS_INLINE long double scalar_trig_widen(T value) {
  if constexpr (
      std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>)
    return static_cast<long double>(static_cast<float>(value));
  else
    return static_cast<long double>(value);
}

/** Returns the parity of an exactly integral long-double value. */
VECOPS_ALWAYS_INLINE bool scalar_trig_integer_is_odd(long double value) {
  return std::fmod(std::fabs(value), 2.0L) == 1.0L;
}

template <typename T>
VECOPS_ALWAYS_INLINE void scalar_sincospi_value(
    T value, T& sin_result, T& cos_result) {
  const long double x = scalar_trig_widen(value);

  if (std::isnan(x)) {
    sin_result = cos_result = scalar_trig_narrow<T>(x);
    return;
  }
  if (std::isinf(x)) {
    const long double nan = std::numeric_limits<long double>::quiet_NaN();
    sin_result = cos_result = scalar_trig_narrow<T>(nan);
    return;
  }

  long double integral;
  if (std::modf(x, &integral) == 0.0L) {
    // C23: sinpi(+-n) has the sign of x, whereas cospi(n) is (-1)^n.
    sin_result = scalar_trig_narrow<T>(std::copysign(0.0L, x));
    cos_result = scalar_trig_narrow<T>(
        scalar_trig_integer_is_odd(integral) ? -1.0L : 1.0L);
    return;
  }

  long double twice_integral;
  if (std::modf(2.0L * x, &twice_integral) == 0.0L) {
    // The integer below the half-integer determines the quadrant.
    const long double lower = std::floor(x);
    sin_result = scalar_trig_narrow<T>(
        scalar_trig_integer_is_odd(lower) ? -1.0L : 1.0L);
    // C23 specifies +0 for cospi(n + 1/2), independent of n.
    cos_result = scalar_trig_narrow<T>(0.0L);
    return;
  }

  // Reduce x = n + r with r in (-1/2, 1/2). Since every non-integral
  // binary32/binary64 input has magnitude below the point where adding 1/2
  // loses information in long double, this subtraction is exact.
  const long double nearest = std::floor(x + 0.5L);
  const long double r = x - nearest;
  const bool negate = scalar_trig_integer_is_odd(nearest);

  long double s = std::sin(kScalarTrigPi * r);
  // Near a half-integer, evaluate cos(pi*r) as sin(pi*distance) so the
  // representable distance to the exact zero is not obscured by pi rounding.
  const long double ar = std::fabs(r);
  long double c = ar > 0.25L
      ? std::sin(kScalarTrigPi * (0.5L - ar))
      : std::cos(kScalarTrigPi * r);
  if (negate) {
    s = -s;
    c = -c;
  }
  sin_result = scalar_trig_narrow<T>(s);
  cos_result = scalar_trig_narrow<T>(c);
}

template <typename T>
VECOPS_ALWAYS_INLINE T scalar_tanpi_value(T value) {
  const long double x = scalar_trig_widen(value);

  if (std::isnan(x)) return scalar_trig_narrow<T>(x);
  if (std::isinf(x)) {
    return scalar_trig_narrow<T>(
        std::numeric_limits<long double>::quiet_NaN());
  }

  long double integral;
  if (std::modf(x, &integral) == 0.0L) {
    // C23 integer-zero sign: negative XOR odd integer.
    const bool negative = std::signbit(x) ^
        scalar_trig_integer_is_odd(integral);
    return scalar_trig_narrow<T>(
        std::copysign(0.0L, negative ? -1.0L : 1.0L));
  }

  long double twice_integral;
  if (std::modf(2.0L * x, &twice_integral) == 0.0L) {
    // tanpi(n + 1/2) is +inf for even n and -inf for odd n.
    const bool negative = scalar_trig_integer_is_odd(std::floor(x));
    const long double inf = std::numeric_limits<long double>::infinity();
    return scalar_trig_narrow<T>(negative ? -inf : inf);
  }

  const long double nearest = std::floor(x + 0.5L);
  const long double r = x - nearest;
  const long double ar = std::fabs(r);
  long double result;
  if (ar > 0.25L) {
    // tan(pi*r) = sign(r) / tan(pi*(1/2-|r|)); this keeps the small
    // distance to a pole intact instead of subtracting from an approximate
    // representation of pi/2.
    const long double distance = 0.5L - ar;
    result = std::copysign(
        1.0L / std::tan(kScalarTrigPi * distance), r);
  } else {
    result = std::tan(kScalarTrigPi * r);
  }
  return scalar_trig_narrow<T>(result);
}

template <TrigKind Kind, typename T>
VECOPS_ALWAYS_INLINE T scalar_radian_trig_value(T value) {
  if constexpr (std::same_as<T, float64_t>) {
    const double x = static_cast<double>(value);
    if constexpr (Kind == TrigKind::Sin) return static_cast<T>(std::sin(x));
    else if constexpr (Kind == TrigKind::Cos)
      return static_cast<T>(std::cos(x));
    else
      return static_cast<T>(std::tan(x));
  } else {
    const float x = static_cast<float>(value);
    const float result = [&] {
      if constexpr (Kind == TrigKind::Sin) return std::sin(x);
      else if constexpr (Kind == TrigKind::Cos) return std::cos(x);
      else return std::tan(x);
    }();
    return static_cast<T>(result);
  }
}

template <TrigKind Kind, TrigUnit Unit, typename T>
VECOPS_ALWAYS_INLINE T scalar_trig_value(T value) {
  if constexpr (Unit == TrigUnit::Radians) {
    return scalar_radian_trig_value<Kind>(value);
  } else if constexpr (Kind == TrigKind::Tan) {
    return scalar_tanpi_value(value);
  } else {
    T sin_result;
    T cos_result;
    scalar_sincospi_value(value, sin_result, cos_result);
    if constexpr (Kind == TrigKind::Sin) return sin_result;
    else return cos_result;
  }
}

template <TrigKind Kind, TrigUnit Unit, Accuracy A>
struct ScalarTrigWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      TrigOp<Kind, Unit, A>, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)
      value[lane] = scalar_trig_value<Kind, Unit, T>(value[lane]);
    return value;
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      TrigOp<Kind, Unit, A> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, value), mask, inactive);
  }
};

template <TrigKind Kind, TrigUnit Unit, Accuracy A>
struct NativeWordImpl<ScalarBackend, TrigOp<Kind, Unit, A>>
    : ScalarTrigWordImpl<Kind, Unit, A> {};

template <TrigUnit Unit, Accuracy A>
struct NativeWordImpl<ScalarBackend, SinCosOp<Unit, A>> {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE void call(
      SinCosOp<Unit, A>, Tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag>& sin_out, NativeWordVec<Tag>& cos_out) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      if constexpr (Unit == TrigUnit::Pi) {
        scalar_sincospi_value(value[lane], sin_out[lane], cos_out[lane]);
      } else {
        sin_out[lane] = scalar_radian_trig_value<TrigKind::Sin, T>(
            value[lane]);
        cos_out[lane] = scalar_radian_trig_value<TrigKind::Cos, T>(
            value[lane]);
      }
    }
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_MATH_TRIG_H
