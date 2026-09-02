#ifndef VECOPS_VEC_DETAILS_ROUNDING_H
#define VECOPS_VEC_DETAILS_ROUNDING_H

/**
 * @file Rounding.h
 * @brief Backend-independent rounding traits, scalar reference operations,
 * and multi-word dispatch.
 */

#include <cmath>
#include <limits>

#include "vecops/vec/details/Arithmetic.h"

namespace vecops::vec::details {

#define VECOPS_VEC_REGISTER_ROUNDING_OP(OpType)                         \
  template <>                                                          \
  struct EnableElementwiseWordBatching<OpType> : std::true_type {};    \
  template <typename Backend, FloatingTag Tag>                          \
  struct GenericImpl<Backend, OpType, Tag>                              \
      : UnaryArithmeticGenericImpl<Backend, OpType, Tag> {}

VECOPS_VEC_REGISTER_ROUNDING_OP(FloorOp);
VECOPS_VEC_REGISTER_ROUNDING_OP(CeilOp);
VECOPS_VEC_REGISTER_ROUNDING_OP(TruncOp);
VECOPS_VEC_REGISTER_ROUNDING_OP(RoundOp);
VECOPS_VEC_REGISTER_ROUNDING_OP(RoundEvenOp);
VECOPS_VEC_REGISTER_ROUNDING_OP(NearbyIntOp);
VECOPS_VEC_REGISTER_ROUNDING_OP(RintOp);

#undef VECOPS_VEC_REGISTER_ROUNDING_OP

template <typename Op, Element T>
VECOPS_ALWAYS_INLINE T scalar_rounding_value(T value) {
  static_assert(::vecops::is_float_v<T>);
  const double input = static_cast<double>(value);
  const double result = [&] {
    if constexpr (std::same_as<Op, FloorOp>) {
      return std::floor(input);
    } else if constexpr (std::same_as<Op, CeilOp>) {
      return std::ceil(input);
    } else if constexpr (std::same_as<Op, TruncOp>) {
      return std::trunc(input);
    } else if constexpr (std::same_as<Op, RoundOp>) {
      return std::round(input);
    } else if constexpr (std::same_as<Op, NearbyIntOp>) {
      return std::nearbyint(input);
    } else if constexpr (std::same_as<Op, RintOp>) {
      // Some optimized builds fold std::rint under an assumed default mode.
      // nearbyint has the same value semantics and reliably observes the
      // dynamic environment here; rint's inexact exception remains optional.
      return std::nearbyint(input);
    } else if constexpr (std::same_as<Op, RoundEvenOp>) {
      if (!std::isfinite(input) || input == 0.0) return input;
      const double magnitude = std::abs(input);
      constexpr int digits = std::numeric_limits<T>::digits;
      const double integral_threshold = std::ldexp(1.0, digits - 1);
      if (magnitude >= integral_threshold) return input;
      const double lower = std::floor(magnitude);
      const double fraction = magnitude - lower;
      const double rounded = fraction < 0.5
          ? lower
          : fraction > 0.5
              ? lower + 1.0
              : std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0;
      return std::copysign(rounded, input);
    } else {
      static_assert(dispatch_dependent_false<Op>);
    }
  }();
  return static_cast<T>(result);
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_ROUNDING_H
