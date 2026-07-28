#ifndef VECOPS_NVEC_DETAILS_SCALAR_ARITHMETIC_H
#define VECOPS_NVEC_DETAILS_SCALAR_ARITHMETIC_H

#include <bit>
#include <type_traits>

#include "vecops/nvec/details/Dispatch.h"

namespace vecops::nvec::details {

template <>
struct NativeWordImpl<ScalarBackend, AddOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      AddOp, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    NativeWordVec<Tag> result{};
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      if constexpr (
          std::same_as<T, bfloat16_t> || std::same_as<T, float16_t> ||
          std::same_as<T, float32_t> || std::same_as<T, float64_t>) {
        result[lane] = a[lane] + b[lane];
      } else if constexpr (std::same_as<T, int8_t>) {
        result[lane] = ::vecops::bitcast<T>(static_cast<uint8_t>(
            ::vecops::bitcast<uint8_t>(a[lane]) +
            ::vecops::bitcast<uint8_t>(b[lane])));
      } else if constexpr (std::same_as<T, uint8_t>) {
        result[lane] = static_cast<T>(a[lane] + b[lane]);
      } else if constexpr (std::same_as<T, int16_t>) {
        result[lane] = ::vecops::bitcast<T>(static_cast<uint16_t>(
            ::vecops::bitcast<uint16_t>(a[lane]) +
            ::vecops::bitcast<uint16_t>(b[lane])));
      } else if constexpr (std::same_as<T, uint16_t>) {
        result[lane] = static_cast<T>(a[lane] + b[lane]);
      } else if constexpr (std::same_as<T, int32_t>) {
        result[lane] = ::vecops::bitcast<T>(
            ::vecops::bitcast<uint32_t>(a[lane]) +
            ::vecops::bitcast<uint32_t>(b[lane]));
      } else if constexpr (std::same_as<T, uint32_t>) {
        result[lane] = a[lane] + b[lane];
      } else if constexpr (std::same_as<T, int64_t>) {
        result[lane] = ::vecops::bitcast<T>(
            ::vecops::bitcast<uint64_t>(a[lane]) +
            ::vecops::bitcast<uint64_t>(b[lane]));
      } else if constexpr (std::same_as<T, uint64_t>) {
        result[lane] = a[lane] + b[lane];
      } else {
        static_assert(
            dispatch_dependent_false<T>,
            "scalar add has no implementation for this element type");
      }
    }
    return result;
  }
};

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_SCALAR_ARITHMETIC_H
