#ifndef VECOPS_VEC_DETAILS_SCALAR_REDUCTION_H
#define VECOPS_VEC_DETAILS_SCALAR_REDUCTION_H

#include <algorithm>

#include "vecops/vec/details/scalar/Arithmetic.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                       Reduction word implementations                       //
/* **************************************************************************** */

template <nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t scalar_valid_reduce_lanes(Tag tag) {
  constexpr nint_t word_lanes =
      RepresentationTraits<ScalarBackend, Tag>::word_lanes;
  return std::clamp<nint_t>(
      size(tag) - Index * word_lanes, 0, word_lanes);
}

template <typename Op>
struct ScalarReductionWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    using T = ElementOf<Tag>;
    T result = reduction_identity<Op, T>();
    const nint_t valid = scalar_valid_reduce_lanes<Index>(tag);
    for (nint_t lane = 0; lane < valid; ++lane) {
      if constexpr (std::same_as<Op, ReduceAddOp>) {
        result = scalar_arithmetic_binary<T>(
            result, value[lane], [](auto a, auto b) { return a + b; });
      } else if constexpr (std::same_as<Op, ReduceMaxOp>) {
        if (value[lane] > result) result = value[lane];
      } else if constexpr (std::same_as<Op, ReduceMinOp>) {
        if (value[lane] < result) result = value[lane];
      }
    }
    return result;
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask) {
    using T = ElementOf<Tag>;
    T result = reduction_identity<Op, T>();
    const nint_t valid = scalar_valid_reduce_lanes<Index>(tag);
    for (nint_t lane = 0; lane < valid; ++lane) {
      if (mask.bits.test(static_cast<std::size_t>(lane))) {
        if constexpr (std::same_as<Op, ReduceAddOp>) {
          result = scalar_arithmetic_binary<T>(
              result, value[lane], [](auto a, auto b) { return a + b; });
        } else if constexpr (std::same_as<Op, ReduceMaxOp>) {
          if (value[lane] > result) result = value[lane];
        } else if constexpr (std::same_as<Op, ReduceMinOp>) {
          if (value[lane] < result) result = value[lane];
        }
      }
    }
    return result;
  }
};

template <>
struct NativeWordImpl<ScalarBackend, ReduceAddOp>
    : ScalarReductionWordImpl<ReduceAddOp> {};
template <>
struct NativeWordImpl<ScalarBackend, ReduceMaxOp>
    : ScalarReductionWordImpl<ReduceMaxOp> {};
template <>
struct NativeWordImpl<ScalarBackend, ReduceMinOp>
    : ScalarReductionWordImpl<ReduceMinOp> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_REDUCTION_H
