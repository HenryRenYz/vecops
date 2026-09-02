#ifndef VECOPS_VEC_DETAILS_SCALAR_SEQUENCE_H
#define VECOPS_VEC_DETAILS_SCALAR_SEQUENCE_H

/** @file Sequence.h @brief Scalar lane-sequence implementation. */

#include <type_traits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {

template <Element T>
VECOPS_ALWAYS_INLINE T scalar_iota_value(
    T start, T step, nint_t index) {
  if constexpr (std::integral<T>) {
    using U = std::make_unsigned_t<T>;
    const U start_bits = [&] {
      if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(start);
      else return start;
    }();
    const U step_bits = [&] {
      if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(step);
      else return step;
    }();
    const U result = static_cast<U>(
        start_bits + step_bits * static_cast<U>(index));
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(result);
    else return result;
  } else if constexpr (std::same_as<T, float64_t>) {
    return static_cast<T>(start + static_cast<T>(index) * step);
  } else {
    const T lane = static_cast<T>(index);
    const float32_t result = static_cast<float32_t>(start) +
        static_cast<float32_t>(lane) * static_cast<float32_t>(step);
    return static_cast<T>(result);
  }
}

template <>
struct NativeWordImpl<ScalarBackend, IotaOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      IotaOp op, Tag tag, ElementOf<Tag> start) {
    return call<Index>(
        op, tag, start, static_cast<ElementOf<Tag>>(1));
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      IotaOp, Tag, ElementOf<Tag> start, ElementOf<Tag> step) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    NativeWordVec<Tag> result{};
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      result[lane] = scalar_iota_value(
          start, step, Index * Traits::word_lanes + lane);
    }
    return result;
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_SEQUENCE_H
