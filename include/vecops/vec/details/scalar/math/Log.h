// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SCALAR_MATH_LOG_H
#define VECOPS_VEC_DETAILS_SCALAR_MATH_LOG_H

/**
 * @file Log.h
 * @brief Scalar backend implementations for the log family (all tiers).
 *
 * The scalar backend computes every tier from the IEEE libm logarithm of
 * the widest convenient format: f32 directly, f64 directly, and f16/bf16
 * through a float intermediate. The composed result satisfies every
 * accuracy tier (libm logarithms are within 1 ULP of the correctly rounded
 * value), so tiers differ only nominally here. Special inputs follow the
 * shared log contract: log(+-0) = -inf, log(+inf) = +inf, log(x < 0) = NaN,
 * NaN propagates, and log(1) = +0. Log outputs are never subnormal and
 * subnormal inputs are exact in the wider evaluation format, so no flush
 * or renormalization is needed.
 */

#include <cmath>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Basic.h"
#include "vecops/vec/details/Wordwise.h"

namespace vecops::vec::details {

template <LogBase Base, Element T>
VECOPS_ALWAYS_INLINE T scalar_log_value(T value) {
  const double widened = static_cast<double>(value);
  const double result = [&] {
    if constexpr (Base == LogBase::E) return std::log(widened);
    else if constexpr (Base == LogBase::Base2) return std::log2(widened);
    else return std::log10(widened);
  }();
  if constexpr (
      std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>)
    return static_cast<T>(static_cast<float>(result));
  else
    return static_cast<T>(result);
}

template <LogBase Base, Accuracy A>
struct ScalarLogWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A>, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)
      value[lane] = scalar_log_value<Base, T>(value[lane]);
    return value;
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, value), mask, inactive);
  }
};

template <LogBase Base, Accuracy A>
struct NativeWordImpl<ScalarBackend, LogOp<Base, A>>
    : ScalarLogWordImpl<Base, A> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_MATH_LOG_H
