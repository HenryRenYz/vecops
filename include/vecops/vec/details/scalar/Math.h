#ifndef VECOPS_VEC_DETAILS_SCALAR_MATH_H
#define VECOPS_VEC_DETAILS_SCALAR_MATH_H

#include <cmath>
#include <limits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                     Exponential scalar implementations                     //
/* **************************************************************************** */

template <Accuracy A, Element T>
VECOPS_ALWAYS_INLINE T scalar_exp_value(T value) {
  const T result = [&] {
    if constexpr (std::same_as<T, float64_t>)
      return static_cast<T>(std::exp(value));
    else
      return static_cast<T>(std::exp(static_cast<float>(value)));
  }();
#ifdef VECOPS_PRESERVE_SUBNORMALS
  if constexpr (A == Accuracy::Strict) return result;
#endif
  const double widened = static_cast<double>(result);
  if (widened > 0.0 &&
      widened < static_cast<double>(std::numeric_limits<T>::min()))
    return T(0.0F);
  return result;
}

template <Accuracy A, bool NegativeOnly>
struct ScalarExpWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ExpOp<A, NegativeOnly>, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)
      value[lane] = scalar_exp_value<A>(value[lane]);
    return value;
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ExpOp<A, NegativeOnly>, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    static_assert(Index >= 0 && Index < num_words(tag));
    for (nint_t lane = 0; lane < native_word_size(tag); ++lane) {
      if (mask.bits.test(static_cast<std::size_t>(lane)))
        inactive[lane] = scalar_exp_value<A>(value[lane]);
    }
    return inactive;
  }
};

template <Accuracy A, bool NegativeOnly>
struct NativeWordImpl<ScalarBackend, ExpOp<A, NegativeOnly>>
    : ScalarExpWordImpl<A, NegativeOnly> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_MATH_H
