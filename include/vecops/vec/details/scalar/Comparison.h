// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SCALAR_COMPARISON_H
#define VECOPS_VEC_DETAILS_SCALAR_COMPARISON_H

#include <bit>
#include <cmath>
#include <limits>
#include <type_traits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//             Comparison and classification word implementations             //
/* **************************************************************************** */

template <typename Op, typename T>
VECOPS_ALWAYS_INLINE bool scalar_compare_lane(Op, T a, T b) {
  static_assert(
      std::same_as<T, bfloat16_t> || std::same_as<T, float16_t> ||
      std::same_as<T, float32_t> || std::same_as<T, float64_t> ||
      std::same_as<T, int8_t> || std::same_as<T, uint8_t> ||
      std::same_as<T, int16_t> || std::same_as<T, uint16_t> ||
      std::same_as<T, int32_t> || std::same_as<T, uint32_t> ||
      std::same_as<T, int64_t> || std::same_as<T, uint64_t>,
      "scalar comparison has no implementation for this element type");
  if constexpr (std::same_as<Op, CmpEqOp>) return a == b;
  else if constexpr (std::same_as<Op, CmpNeOp>) return a != b;
  else if constexpr (std::same_as<Op, CmpLtOp>) return a < b;
  else if constexpr (std::same_as<Op, CmpGtOp>) return a > b;
  else if constexpr (std::same_as<Op, CmpLeOp>) return a <= b;
  else if constexpr (std::same_as<Op, CmpGeOp>) return a >= b;
  else {
    static_assert(dispatch_dependent_false<Op>, "unsupported scalar comparison");
  }
}

template <typename Op, typename T>
VECOPS_ALWAYS_INLINE bool scalar_classify_lane(Op, T value) {
  static_assert(
      std::same_as<T, bfloat16_t> || std::same_as<T, float16_t> ||
      std::same_as<T, float32_t> || std::same_as<T, float64_t>,
      "scalar classification has no implementation for this element type");
  // Preserve float64 exactly. The old implementation narrowed it to float32,
  // which incorrectly classified large finite doubles as infinity.
  const auto widened = [&] {
    if constexpr (std::same_as<T, float64_t>) return value;
    else return static_cast<float32_t>(value);
  }();
  if constexpr (std::same_as<Op, IsNanOp>) {
    return std::isnan(widened);
  } else if constexpr (std::same_as<Op, IsPosInfOp>) {
    return std::isinf(widened) && !std::signbit(widened);
  } else if constexpr (std::same_as<Op, IsNegInfOp>) {
    return std::isinf(widened) && std::signbit(widened);
  } else if constexpr (std::same_as<Op, IsInfOp>) {
    return std::isinf(widened);
  } else if constexpr (
      std::same_as<Op, IsFiniteOp> ||
      std::same_as<Op, IsNormalOp> ||
      std::same_as<Op, SignBitOp>) {
    const auto bits = [&] {
      if constexpr (std::same_as<T, bfloat16_t> ||
                    std::same_as<T, float16_t>)
        return static_cast<uint64_t>(value.to_bits());
      else if constexpr (std::same_as<T, float32_t>)
        return static_cast<uint64_t>(::vecops::bitcast<uint32_t>(value));
      else
        return ::vecops::bitcast<uint64_t>(value);
    }();
    constexpr uint64_t sign = sizeof(T) == 2 ? 0x8000ull
                                  : sizeof(T) == 4 ? 0x80000000ull
                                                   : 0x8000000000000000ull;
    constexpr uint64_t infinity =
        std::same_as<T, bfloat16_t> ? 0x7f80ull
        : std::same_as<T, float16_t> ? 0x7c00ull
        : std::same_as<T, float32_t> ? 0x7f800000ull
                                     : 0x7ff0000000000000ull;
    constexpr uint64_t minimum_normal =
        std::same_as<T, bfloat16_t> ? 0x0080ull
        : std::same_as<T, float16_t> ? 0x0400ull
        : std::same_as<T, float32_t> ? 0x00800000ull
                                     : 0x0010000000000000ull;
    const uint64_t absolute = bits & ~sign;
    if constexpr (std::same_as<Op, IsFiniteOp>)
      return absolute < infinity;
    else if constexpr (std::same_as<Op, IsNormalOp>)
      return absolute >= minimum_normal && absolute < infinity;
    else
      return (bits & sign) != 0;
  } else {
    static_assert(dispatch_dependent_false<Op>, "unsupported scalar classification");
  }
}

template <typename Op>
struct ScalarBinaryComparisonImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    NativeWordMask<Tag> result{};
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      result.bits.set(
          static_cast<std::size_t>(lane),
          scalar_compare_lane(op, a[lane], b[lane]));
    }
    return result;
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask) {
    auto result = call<Index>(op, tag, a, b);
    result.bits &= mask.bits;
    return result;
  }
};

template <typename Op>
struct ScalarClassificationImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    NativeWordMask<Tag> result{};
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      result.bits.set(
          static_cast<std::size_t>(lane),
          scalar_classify_lane(op, value[lane]));
    }
    return result;
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value, NativeWordMask<Tag> mask) {
    auto result = call<Index>(op, tag, value);
    result.bits &= mask.bits;
    return result;
  }
};

#define VECOPS_VEC_SCALAR_BINARY_COMPARISON(OpType)                      \
  template <>                                                             \
  struct NativeWordImpl<ScalarBackend, OpType>                            \
      : ScalarBinaryComparisonImpl<OpType> {}

VECOPS_VEC_SCALAR_BINARY_COMPARISON(CmpEqOp);
VECOPS_VEC_SCALAR_BINARY_COMPARISON(CmpNeOp);
VECOPS_VEC_SCALAR_BINARY_COMPARISON(CmpLtOp);
VECOPS_VEC_SCALAR_BINARY_COMPARISON(CmpGtOp);
VECOPS_VEC_SCALAR_BINARY_COMPARISON(CmpLeOp);
VECOPS_VEC_SCALAR_BINARY_COMPARISON(CmpGeOp);

#undef VECOPS_VEC_SCALAR_BINARY_COMPARISON

#define VECOPS_VEC_SCALAR_CLASSIFICATION(OpType)                         \
  template <>                                                             \
  struct NativeWordImpl<ScalarBackend, OpType>                            \
      : ScalarClassificationImpl<OpType> {}

VECOPS_VEC_SCALAR_CLASSIFICATION(IsNanOp);
VECOPS_VEC_SCALAR_CLASSIFICATION(IsPosInfOp);
VECOPS_VEC_SCALAR_CLASSIFICATION(IsNegInfOp);
VECOPS_VEC_SCALAR_CLASSIFICATION(IsInfOp);
VECOPS_VEC_SCALAR_CLASSIFICATION(IsFiniteOp);
VECOPS_VEC_SCALAR_CLASSIFICATION(IsNormalOp);
VECOPS_VEC_SCALAR_CLASSIFICATION(SignBitOp);

#undef VECOPS_VEC_SCALAR_CLASSIFICATION

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_COMPARISON_H
