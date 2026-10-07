// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SCALAR_MATH_RECIPROCAL_H
#define VECOPS_VEC_DETAILS_SCALAR_MATH_RECIPROCAL_H

/**
 * @file Reciprocal.h
 * @brief Scalar backend implementations for rcp and rsqrt (all tiers).
 *
 * The scalar backend computes both operations from IEEE division and square
 * root, which satisfies every accuracy tier (the composed result stays
 * within 1 ULP of the correctly-rounded value). Tiers therefore differ only
 * in their subnormal-output policy. Special inputs follow the shared
 * reciprocal contract: rcp(+-0) = +-inf, rcp(+-inf) = +-0, rsqrt(+-0) = +inf,
 * rsqrt(x < 0) = NaN, NaN propagates.
 */

#include <cmath>
#include <limits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Basic.h"
#include "vecops/vec/details/Wordwise.h"

namespace vecops::vec::details {

template <Accuracy Tier, Element T>
VECOPS_ALWAYS_INLINE T scalar_recip_flush(T result) {
#ifdef VECOPS_PRESERVE_SUBNORMALS
  if constexpr (Tier == Accuracy::Strict) return result;
#endif
  const double widened = static_cast<double>(result);
  if (widened != 0.0 &&
      std::abs(widened) < static_cast<double>(std::numeric_limits<T>::min()))
    return static_cast<T>(widened * 0.0);
  return result;
}

template <Accuracy Tier, Element T>
VECOPS_ALWAYS_INLINE T scalar_rcp_value(T value) {
  const T result = [&] {
    if constexpr (
        std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>)
      return static_cast<T>(1.0F / static_cast<float>(value));
    else
      return static_cast<T>(T(1) / value);
  }();
  return scalar_recip_flush<Tier>(result);
}

template <Accuracy Tier, Element T>
VECOPS_ALWAYS_INLINE T scalar_rsqrt_value(T value) {
  if (value == T(0)) return std::numeric_limits<T>::infinity();
  const T result = [&] {
    if constexpr (
        std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>)
      return static_cast<T>(1.0F / std::sqrt(static_cast<float>(value)));
    else
      return static_cast<T>(T(1) / std::sqrt(value));
  }();
  // rsqrt never produces subnormal outputs for normal inputs; the flush is
  // kept purely for uniformity with the shared contract.
  return scalar_recip_flush<Tier>(result);
}

template <Accuracy A, bool IsRsqrt>
struct ScalarRecipWordImpl {
  template <nint_t Index, FloatingTag Tag, typename Op>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(
        std::same_as<Op, RsqrtOp<A>> || std::same_as<Op, RcpOp<A>>);
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      if constexpr (IsRsqrt)
        value[lane] = scalar_rsqrt_value<A>(value[lane]);
      else
        value[lane] = scalar_rcp_value<A>(value[lane]);
    }
    return value;
  }

  template <nint_t Index, FloatingTag Tag, typename Op, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, value), mask, inactive);
  }
};

template <Accuracy A>
struct NativeWordImpl<ScalarBackend, RsqrtOp<A>>
    : ScalarRecipWordImpl<A, true> {};

template <Accuracy A>
struct NativeWordImpl<ScalarBackend, RcpOp<A>>
    : ScalarRecipWordImpl<A, false> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_MATH_RECIPROCAL_H
