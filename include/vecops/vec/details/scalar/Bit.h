// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SCALAR_BIT_H
#define VECOPS_VEC_DETAILS_SCALAR_BIT_H

#include <bit>
#include <limits>
#include <type_traits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Basic.h"
#include "vecops/vec/details/Wordwise.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                   Bitwise operation word implementations                   //
/* **************************************************************************** */

template <typename T>
VECOPS_ALWAYS_INLINE std::make_unsigned_t<T> scalar_bit_bits(T value) {
  using U = std::make_unsigned_t<T>;
  if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(value);
  else return value;
}

template <typename T>
VECOPS_ALWAYS_INLINE T scalar_bit_value(std::make_unsigned_t<T> value) {
  if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(value);
  else return value;
}

#define VECOPS_VEC_DEFINE_SCALAR_BIT_BINARY(OpType, Expression)          \
  template <>                                                            \
  struct NativeWordImpl<ScalarBackend, OpType> {                         \
    template <nint_t Index, IntegerTag Tag>                              \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {       \
      using Traits = RepresentationTraits<ScalarBackend, Tag>;          \
      using T = ElementOf<Tag>;                                          \
      using U = std::make_unsigned_t<T>;                                 \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      NativeWordVec<Tag> result{};                                       \
      for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {         \
        const U ua = scalar_bit_bits(a[lane]);                           \
        const U ub = scalar_bit_bits(b[lane]);                           \
        result[lane] = scalar_bit_value<T>(static_cast<U>(Expression));  \
      }                                                                  \
      return result;                                                      \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag, typename Policy>             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b, \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      return scalar_masked_merge<Index>(                                 \
          tag, call<Index>(op, tag, a, b), mask, inactive);              \
    }                                                                    \
  }

VECOPS_VEC_DEFINE_SCALAR_BIT_BINARY(BitAndOp, ua & ub);
VECOPS_VEC_DEFINE_SCALAR_BIT_BINARY(BitOrOp, ua | ub);
VECOPS_VEC_DEFINE_SCALAR_BIT_BINARY(BitXorOp, ua ^ ub);
VECOPS_VEC_DEFINE_SCALAR_BIT_BINARY(BitAndNotOp, (~ua) & ub);

#undef VECOPS_VEC_DEFINE_SCALAR_BIT_BINARY

template <>
struct NativeWordImpl<ScalarBackend, BitNotOp> {
  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitNotOp, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    using U = std::make_unsigned_t<T>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      value[lane] = scalar_bit_value<T>(
          static_cast<U>(~scalar_bit_bits(value[lane])));
    }
    return value;
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitNotOp op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    const auto complemented = call<Index>(op, tag, value);
    for (nint_t lane = 0; lane < native_word_size(tag); ++lane) {
      if (mask.bits.test(static_cast<std::size_t>(lane)))
        inactive[lane] = complemented[lane];
    }
    return inactive;
  }
};

template <typename Op, typename T>
VECOPS_ALWAYS_INLINE T scalar_bit_count(T value) {
  using U = std::make_unsigned_t<T>;
  const U bits = scalar_bit_bits(value);
  int count;
  if constexpr (std::same_as<Op, PopCountOp>)
    count = std::popcount(bits);
  else if constexpr (std::same_as<Op, CountLeadingZeroOp>)
    count = std::countl_zero(bits);
  else if constexpr (std::same_as<Op, CountLeadingOneOp>)
    count = std::countl_one(bits);
  else if constexpr (std::same_as<Op, CountTrailingZeroOp>)
    count = std::countr_zero(bits);
  else if constexpr (std::same_as<Op, CountTrailingOneOp>)
    count = std::countr_one(bits);
  else
    static_assert(dispatch_dependent_false<Op>, "unknown bit-count op");
  return scalar_bit_value<T>(static_cast<U>(count));
}

#define VECOPS_VEC_DEFINE_SCALAR_BIT_COUNT(OpType)                      \
  template <>                                                           \
  struct NativeWordImpl<ScalarBackend, OpType> {                        \
    template <nint_t Index, IntegerTag Tag>                             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value) {                        \
      using Traits = RepresentationTraits<ScalarBackend, Tag>;         \
      using T = ElementOf<Tag>;                                         \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)         \
        value[lane] = scalar_bit_count<OpType>(value[lane]);            \
      return value;                                                      \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, typename Policy>            \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      return scalar_masked_merge<Index>(                                \
          tag, call<Index>(op, tag, value), mask, inactive);            \
    }                                                                   \
  }

VECOPS_VEC_DEFINE_SCALAR_BIT_COUNT(PopCountOp);
VECOPS_VEC_DEFINE_SCALAR_BIT_COUNT(CountLeadingZeroOp);
VECOPS_VEC_DEFINE_SCALAR_BIT_COUNT(CountLeadingOneOp);
VECOPS_VEC_DEFINE_SCALAR_BIT_COUNT(CountTrailingZeroOp);
VECOPS_VEC_DEFINE_SCALAR_BIT_COUNT(CountTrailingOneOp);

#undef VECOPS_VEC_DEFINE_SCALAR_BIT_COUNT

template <typename T>
VECOPS_ALWAYS_INLINE T scalar_shift_left(T value, int count) {
  using U = std::make_unsigned_t<T>;
  constexpr int width = std::numeric_limits<U>::digits;
  if (count >= width) return T{};
  return scalar_bit_value<T>(static_cast<U>(scalar_bit_bits(value) << count));
}

template <typename T>
VECOPS_ALWAYS_INLINE T scalar_shift_right(T value, int count) {
  using U = std::make_unsigned_t<T>;
  constexpr int width = std::numeric_limits<U>::digits;
  const U bits = scalar_bit_bits(value);
  if constexpr (std::is_unsigned_v<T>) {
    return count >= width ? T{} : static_cast<T>(bits >> count);
  } else {
    const bool negative = (bits >> (width - 1)) != 0;
    if (count >= width)
      return scalar_bit_value<T>(negative ? std::numeric_limits<U>::max() : U{});
    if (count == 0) return value;
    U shifted = static_cast<U>(bits >> count);
    if (negative) shifted |= static_cast<U>(
        std::numeric_limits<U>::max() << (width - count));
    return scalar_bit_value<T>(shifted);
  }
}

#define VECOPS_VEC_DEFINE_SCALAR_SHIFT(OpType, Helper)                   \
  template <>                                                            \
  struct NativeWordImpl<ScalarBackend, OpType> {                         \
    template <nint_t Index, IntegerTag Tag, nint_t Count>                \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> value, meta::Const<Count>) {     \
      using Traits = RepresentationTraits<ScalarBackend, Tag>;          \
      using T = ElementOf<Tag>;                                          \
      using U = std::make_unsigned_t<T>;                                 \
      constexpr int width = std::numeric_limits<U>::digits;             \
      static_assert(Count >= 0);                                         \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      constexpr int effective_count = Count >= width                    \
          ? width                                                        \
          : static_cast<int>(Count);                                     \
      for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)          \
        value[lane] = Helper(value[lane], effective_count);              \
      return value;                                                       \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag>                              \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> value, int count) {              \
      using Traits = RepresentationTraits<ScalarBackend, Tag>;          \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)          \
        value[lane] = Helper(value[lane], count);                        \
      return value;                                                       \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag>                              \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> value,                           \
        NativeWordVec<Tag> counts) {                                    \
      using Traits = RepresentationTraits<ScalarBackend, Tag>;          \
      using T = ElementOf<Tag>;                                          \
      using U = std::make_unsigned_t<T>;                                 \
      constexpr U width = std::numeric_limits<U>::digits;               \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {        \
        const U count = static_cast<U>(counts[lane]);                    \
        value[lane] = Helper(                                            \
            value[lane], count >= width ? static_cast<int>(width)       \
                                        : static_cast<int>(count));      \
      }                                                                  \
      return value;                                                       \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag, nint_t Count, typename Policy>\
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        meta::Const<Count> count, NativeWordMask<Tag> mask,             \
        NativeWordVec<Tag> inactive, Policy) {                           \
      return scalar_masked_merge<Index>(                                 \
          tag, call<Index>(op, tag, value, count), mask, inactive);      \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag, typename Policy>             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType op, Tag tag, NativeWordVec<Tag> value, int count,        \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      return scalar_masked_merge<Index>(                                 \
          tag, call<Index>(op, tag, value, count), mask, inactive);      \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag, typename Policy>             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        NativeWordVec<Tag> counts, NativeWordMask<Tag> mask,            \
        NativeWordVec<Tag> inactive, Policy) {                           \
      return scalar_masked_merge<Index>(                                 \
          tag, call<Index>(op, tag, value, counts), mask, inactive);     \
    }                                                                    \
  }

VECOPS_VEC_DEFINE_SCALAR_SHIFT(BitShiftLeftOp, scalar_shift_left);
VECOPS_VEC_DEFINE_SCALAR_SHIFT(BitShiftRightOp, scalar_shift_right);

#undef VECOPS_VEC_DEFINE_SCALAR_SHIFT

template <typename Op, typename T>
VECOPS_ALWAYS_INLINE T scalar_rotate(T value, int count) {
  using U = std::make_unsigned_t<T>;
  const U bits = scalar_bit_bits(value);
  const U rotated = [&] {
    if constexpr (std::same_as<Op, RotateLeftOp>)
      return std::rotl(bits, count);
    else
      return std::rotr(bits, count);
  }();
  return scalar_bit_value<T>(rotated);
}

#define VECOPS_VEC_DEFINE_SCALAR_ROTATE(OpType)                         \
  template <>                                                           \
  struct NativeWordImpl<ScalarBackend, OpType> {                        \
    template <nint_t Index, IntegerTag Tag, nint_t Count>               \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value, meta::Const<Count>) {    \
      using Traits = RepresentationTraits<ScalarBackend, Tag>;         \
      using T = ElementOf<Tag>;                                         \
      using U = std::make_unsigned_t<T>;                                \
      constexpr nint_t width = std::numeric_limits<U>::digits;         \
      constexpr nint_t remainder = Count % width;                       \
      constexpr int normalized = static_cast<int>(                     \
          remainder < 0 ? remainder + width : remainder);              \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)         \
        value[lane] = scalar_rotate<OpType>(value[lane], normalized);   \
      return value;                                                      \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag>                             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value, int count) {             \
      using Traits = RepresentationTraits<ScalarBackend, Tag>;         \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)         \
        value[lane] = scalar_rotate<OpType>(value[lane], count);        \
      return value;                                                      \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag>                             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value,                          \
        NativeWordVec<Tag> counts) {                                   \
      using Traits = RepresentationTraits<ScalarBackend, Tag>;         \
      using T = ElementOf<Tag>;                                         \
      using U = std::make_unsigned_t<T>;                                \
      constexpr U mask = std::numeric_limits<U>::digits - 1;           \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {       \
        const int count = static_cast<int>(                             \
            scalar_bit_bits(counts[lane]) & mask);                     \
        value[lane] = scalar_rotate<OpType>(value[lane], count);        \
      }                                                                 \
      return value;                                                      \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, nint_t Count, typename Policy>\
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        meta::Const<Count> count, NativeWordMask<Tag> mask,             \
        NativeWordVec<Tag> inactive, Policy) {                          \
      return scalar_masked_merge<Index>(                                \
          tag, call<Index>(op, tag, value, count), mask, inactive);     \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, typename Policy>            \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value, int count,        \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      return scalar_masked_merge<Index>(                                \
          tag, call<Index>(op, tag, value, count), mask, inactive);     \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, typename Policy>            \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        NativeWordVec<Tag> counts, NativeWordMask<Tag> mask,            \
        NativeWordVec<Tag> inactive, Policy) {                          \
      return scalar_masked_merge<Index>(                                \
          tag, call<Index>(op, tag, value, counts), mask, inactive);    \
    }                                                                   \
  }

VECOPS_VEC_DEFINE_SCALAR_ROTATE(RotateLeftOp);
VECOPS_VEC_DEFINE_SCALAR_ROTATE(RotateRightOp);

#undef VECOPS_VEC_DEFINE_SCALAR_ROTATE

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_BIT_H
