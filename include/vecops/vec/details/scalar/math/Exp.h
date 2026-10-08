// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SCALAR_MATH_EXP_H
#define VECOPS_VEC_DETAILS_SCALAR_MATH_EXP_H

/**
 * @file Exp.h
 * @brief Scalar backend implementations for the exp family (exp, exp2, exp10).
 *
 * Accuracy comes from the platform libm: double/float exp and exp2 for the
 * matching widths, and a long-double (or double for narrow types) exp2 of
 * x*log2(10) for the base-10 family, which std does not provide. The
 * subnormal flush contract is applied by scalar_exp_flush, mirroring the
 * vector backends.
 */

#include <cmath>
#include <limits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Basic.h"
#include "vecops/vec/details/Wordwise.h"

namespace vecops::vec::details {

/** log2(10) with enough digits for every precision below long double. */
inline constexpr long double kScalarLog2Base10 = 3.321928094887362347870319429489L;

/**
 * Shared post-processing: subnormal results are flushed to zero unless the
 * tier must preserve them (mirrors the vector backends' contract).
 */
template <Accuracy A, Element T>
VECOPS_ALWAYS_INLINE T scalar_exp_flush(T result) {
#ifdef VECOPS_PRESERVE_SUBNORMALS
  if constexpr (A == Accuracy::Strict) return result;
#endif
  const double widened = static_cast<double>(result);
  if (widened > 0.0 &&
      widened < static_cast<double>(std::numeric_limits<T>::min()))
    return T(0.0F);
  return result;
}

template <Accuracy A, Element T>
VECOPS_ALWAYS_INLINE T scalar_exp_value(T value) {
  const T result = [&] {
    if constexpr (std::same_as<T, float64_t>)
      return static_cast<T>(std::exp(value));
    else
      return static_cast<T>(std::exp(static_cast<float>(value)));
  }();
  return scalar_exp_flush<A>(result);
}

template <Accuracy A, Element T>
VECOPS_ALWAYS_INLINE T scalar_exp2_value(T value) {
  const T result = [&] {
    if constexpr (std::same_as<T, float64_t>)
      return static_cast<T>(std::exp2(value));
    else
      return static_cast<T>(std::exp2(static_cast<float>(value)));
  }();
  return scalar_exp_flush<A>(result);
}

template <Accuracy A, Element T>
VECOPS_ALWAYS_INLINE T scalar_exp10_value(T value) {
  // 10^x = 2^(x * log2(10)); the scaled argument stays exact to well below
  // the target precision, and std has no exp10 to lean on.
  const T result = [&] {
    if constexpr (std::same_as<T, float64_t>) {
      return static_cast<T>(
          std::exp2(static_cast<long double>(value) * kScalarLog2Base10));
    } else {
      return static_cast<T>(std::exp2(
          static_cast<double>(static_cast<float>(value)) *
          static_cast<double>(kScalarLog2Base10)));
    }
  }();
  return scalar_exp_flush<A>(result);
}

template <ExpBase Base, Accuracy A, Element T>
VECOPS_ALWAYS_INLINE T scalar_exp_family_value(T value) {
  if constexpr (Base == ExpBase::E) return scalar_exp_value<A>(value);
  else if constexpr (Base == ExpBase::Base2)
    return scalar_exp2_value<A>(value);
  else return scalar_exp10_value<A>(value);
}

template <ExpBase Base, Accuracy A, bool NegativeOnly>
struct ScalarExpWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ExpOp<Base, A, NegativeOnly>, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)
      value[lane] = scalar_exp_family_value<Base, A>(value[lane]);
    return value;
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ExpOp<Base, A, NegativeOnly>, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag,
        call<Index>(ExpOp<Base, A, NegativeOnly>{}, tag, value), mask,
        inactive);
  }
};

template <ExpBase Base, Accuracy A, bool NegativeOnly>
struct NativeWordImpl<ScalarBackend, ExpOp<Base, A, NegativeOnly>>
    : ScalarExpWordImpl<Base, A, NegativeOnly> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_MATH_EXP_H
