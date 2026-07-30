#ifndef VECOPS_VEC_DETAILS_SCALAR_MATH_H
#define VECOPS_VEC_DETAILS_SCALAR_MATH_H

#include <cmath>
#include <limits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                     Exponential scalar implementations                     //
/* **************************************************************************** */

template <typename Op>
inline constexpr bool exp_op_is_strict =
    std::same_as<Op, ExpOp> || std::same_as<Op, ExpNegOp>;

template <Element T>
VECOPS_ALWAYS_INLINE T scalar_exp_value(T value, bool strict) {
  const T result = [&] {
    if constexpr (std::same_as<T, float64_t>)
      return static_cast<T>(std::exp(value));
    else
      return static_cast<T>(std::exp(static_cast<float>(value)));
  }();
#ifdef VECOPS_PRESERVE_SUBNORMALS
  if (strict) return result;
#else
  static_cast<void>(strict);
#endif
  const double widened = static_cast<double>(result);
  if (widened > 0.0 &&
      widened < static_cast<double>(std::numeric_limits<T>::min()))
    return T(0.0F);
  return result;
}

template <typename Op>
struct ScalarExpWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)
      value[lane] = scalar_exp_value(value[lane], exp_op_is_strict<Op>);
    return value;
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    static_assert(Index >= 0 && Index < num_words(tag));
    for (nint_t lane = 0; lane < native_word_size(tag); ++lane) {
      if (mask.bits.test(static_cast<std::size_t>(lane)))
        inactive[lane] = scalar_exp_value(
            value[lane], exp_op_is_strict<Op>);
    }
    return inactive;
  }
};

#define VECOPS_VEC_DEFINE_SCALAR_EXP(OpType)                           \
  template <>                                                          \
  struct NativeWordImpl<ScalarBackend, OpType>                         \
      : ScalarExpWordImpl<OpType> {}

VECOPS_VEC_DEFINE_SCALAR_EXP(ExpOp);
VECOPS_VEC_DEFINE_SCALAR_EXP(ExpFastOp);
VECOPS_VEC_DEFINE_SCALAR_EXP(ExpEstOp);
VECOPS_VEC_DEFINE_SCALAR_EXP(ExpNegOp);
VECOPS_VEC_DEFINE_SCALAR_EXP(ExpNegFastOp);
VECOPS_VEC_DEFINE_SCALAR_EXP(ExpNegEstOp);

#undef VECOPS_VEC_DEFINE_SCALAR_EXP

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_MATH_H
