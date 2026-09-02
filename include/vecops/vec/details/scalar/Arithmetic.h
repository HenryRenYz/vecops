#ifndef VECOPS_VEC_DETAILS_SCALAR_ARITHMETIC_H
#define VECOPS_VEC_DETAILS_SCALAR_ARITHMETIC_H

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <type_traits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Basic.h"
#include "vecops/vec/details/Wordwise.h"

/**
 * @file Arithmetic.h
 * @brief Scalar backend word-level implementations for arithmetic operations.
 *
 * Each operation is implemented lane-by-lane within a 16-byte ScalarVector
 * word. Signed integer types are bitcast to unsigned before arithmetic to
 * produce modular (wrap-around) semantics matching SIMD behavior. Masked
 * variants compose the computed result with an inactive-fill vector via a
 * policy tag (PreserveArithmeticInactive, ZeroArithmeticInactive, or
 * MergeArithmeticInactive).
 */

namespace vecops::vec::details {

/**
 * Applies a binary scalar operation lane-by-lane. Signed integer types are
 * bitcast to unsigned before the operation to produce modular (wrap-around)
 * semantics that match SIMD instruction behavior.
 */

/* **************************************************************************** */
//    scalar_arithmetic_binary helper                                          //
/* **************************************************************************** */

template <typename T, typename Operation>
VECOPS_ALWAYS_INLINE T scalar_arithmetic_binary(
    T a, T b, Operation operation) {
  if constexpr (
      std::same_as<T, bfloat16_t> || std::same_as<T, float16_t> ||
      std::same_as<T, float32_t> || std::same_as<T, float64_t>) {
    return operation(a, b);
  } else if constexpr (std::same_as<T, int8_t>) {
    return ::vecops::bitcast<T>(static_cast<uint8_t>(operation(
        ::vecops::bitcast<uint8_t>(a), ::vecops::bitcast<uint8_t>(b))));
  } else if constexpr (std::same_as<T, uint8_t>) {
    return static_cast<T>(operation(a, b));
  } else if constexpr (std::same_as<T, int16_t>) {
    return ::vecops::bitcast<T>(static_cast<uint16_t>(operation(
        ::vecops::bitcast<uint16_t>(a), ::vecops::bitcast<uint16_t>(b))));
  } else if constexpr (std::same_as<T, uint16_t>) {
    return static_cast<T>(operation(a, b));
  } else if constexpr (std::same_as<T, int32_t>) {
    return ::vecops::bitcast<T>(static_cast<uint32_t>(operation(
        ::vecops::bitcast<uint32_t>(a), ::vecops::bitcast<uint32_t>(b))));
  } else if constexpr (std::same_as<T, uint32_t>) {
    return static_cast<T>(operation(a, b));
  } else if constexpr (std::same_as<T, int64_t>) {
    return ::vecops::bitcast<T>(static_cast<uint64_t>(operation(
        ::vecops::bitcast<uint64_t>(a), ::vecops::bitcast<uint64_t>(b))));
  } else if constexpr (std::same_as<T, uint64_t>) {
    return static_cast<T>(operation(a, b));
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "unsupported scalar arithmetic element type");
  }
}

/* **************************************************************************** */
//    ScalarArithmeticWordImpl and registrations                               //
/* **************************************************************************** */

template <typename Op>
struct ScalarArithmeticWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      if constexpr (std::same_as<Op, AddOp>) {
        a[lane] = scalar_arithmetic_binary<T>(
            a[lane], b[lane], [](auto x, auto y) { return x + y; });
      } else if constexpr (std::same_as<Op, SubOp>) {
        a[lane] = scalar_arithmetic_binary<T>(
            a[lane], b[lane], [](auto x, auto y) { return x - y; });
      } else if constexpr (std::same_as<Op, MulOp>) {
        a[lane] = scalar_arithmetic_binary<T>(
            a[lane], b[lane], [](auto x, auto y) { return x * y; });
      } else if constexpr (std::same_as<Op, DivOp>) {
        static_assert(::vecops::is_float_v<T>);
        a[lane] = static_cast<T>(a[lane] / b[lane]);
      } else if constexpr (std::same_as<Op, MinOp>) {
        a[lane] = std::min(a[lane], b[lane]);
      } else if constexpr (std::same_as<Op, MaxOp>) {
        a[lane] = std::max(a[lane], b[lane]);
      } else {
        static_assert(
            dispatch_dependent_false<Op>,
            "unsupported scalar arithmetic operation");
      }
    }
    return a;
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, a, b), mask, inactive);
  }
};

template <>
struct NativeWordImpl<ScalarBackend, AddOp>
    : ScalarArithmeticWordImpl<AddOp> {};

template <>
struct NativeWordImpl<ScalarBackend, SubOp>
    : ScalarArithmeticWordImpl<SubOp> {};

template <typename Op, std::integral T>
VECOPS_ALWAYS_INLINE T scalar_saturating_add_sub(T a, T b) {
  using U = std::make_unsigned_t<T>;
  const U ua = [&] {
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(a);
    else return a;
  }();
  const U ub = [&] {
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(b);
    else return b;
  }();
  const U wrapped = [&] {
    if constexpr (std::same_as<Op, SaturatedAddOp>)
      return static_cast<U>(ua + ub);
    else
      return static_cast<U>(ua - ub);
  }();

  if constexpr (std::is_unsigned_v<T>) {
    if constexpr (std::same_as<Op, SaturatedAddOp>)
      return wrapped < ua ? std::numeric_limits<T>::max() : wrapped;
    else
      return ua < ub ? T{0} : wrapped;
  } else {
    constexpr U sign = U{1} << (sizeof(T) * 8 - 1);
    const bool overflow = [&] {
      if constexpr (std::same_as<Op, SaturatedAddOp>)
        return (((ua ^ wrapped) & (ub ^ wrapped)) & sign) != 0;
      else
        return (((ua ^ ub) & (ua ^ wrapped)) & sign) != 0;
    }();
    if (overflow)
      return (ua & sign) != 0
          ? std::numeric_limits<T>::min()
          : std::numeric_limits<T>::max();
    return ::vecops::bitcast<T>(wrapped);
  }
}

template <typename Op>
struct ScalarSaturatingArithmeticWordImpl {
  template <nint_t Index, VectorTag Tag>
    requires std::integral<ElementOf<Tag>>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)
      a[lane] = scalar_saturating_add_sub<Op>(a[lane], b[lane]);
    return a;
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
    requires std::integral<ElementOf<Tag>>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, a, b), mask, inactive);
  }
};

template <>
struct NativeWordImpl<ScalarBackend, SaturatedAddOp>
    : ScalarSaturatingArithmeticWordImpl<SaturatedAddOp> {};

template <>
struct NativeWordImpl<ScalarBackend, SaturatedSubOp>
    : ScalarSaturatingArithmeticWordImpl<SaturatedSubOp> {};

template <>
struct NativeWordImpl<ScalarBackend, MulOp>
    : ScalarArithmeticWordImpl<MulOp> {};

template <>
struct NativeWordImpl<ScalarBackend, DivOp>
    : ScalarArithmeticWordImpl<DivOp> {};

template <>
struct NativeWordImpl<ScalarBackend, MinOp>
    : ScalarArithmeticWordImpl<MinOp> {};

template <>
struct NativeWordImpl<ScalarBackend, MaxOp>
    : ScalarArithmeticWordImpl<MaxOp> {};

template <Element T>
  requires (::vecops::is_float_v<T>)
VECOPS_ALWAYS_INLINE T scalar_copy_sign(T magnitude, T sign) {
  if constexpr (std::same_as<T, bfloat16_t> ||
                std::same_as<T, float16_t>) {
    return T::from_bits(static_cast<uint16_t>(
        (magnitude.to_bits() & 0x7fffu) | (sign.to_bits() & 0x8000u)));
  } else if constexpr (std::same_as<T, float32_t>) {
    const auto magnitude_bits = ::vecops::bitcast<uint32_t>(magnitude);
    const auto sign_bits = ::vecops::bitcast<uint32_t>(sign);
    return ::vecops::bitcast<T>(
        (magnitude_bits & 0x7fffffffu) | (sign_bits & 0x80000000u));
  } else {
    const auto magnitude_bits = ::vecops::bitcast<uint64_t>(magnitude);
    const auto sign_bits = ::vecops::bitcast<uint64_t>(sign);
    return ::vecops::bitcast<T>(
        (magnitude_bits & 0x7fffffffffffffffull) |
        (sign_bits & 0x8000000000000000ull));
  }
}

template <>
struct NativeWordImpl<ScalarBackend, CopySignOp> {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      CopySignOp, Tag, NativeWordVec<Tag> magnitude,
      NativeWordVec<Tag> sign) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)
      magnitude[lane] = scalar_copy_sign<T>(magnitude[lane], sign[lane]);
    return magnitude;
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      CopySignOp op, Tag tag, NativeWordVec<Tag> magnitude,
      NativeWordVec<Tag> sign, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, magnitude, sign), mask, inactive);
  }
};

/* **************************************************************************** */
//    ScalarClampWordImpl                                                      //
/* **************************************************************************** */

struct ScalarClampWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ClampOp, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> lower, NativeWordVec<Tag> upper) {
    const auto bounded_low =
        ScalarArithmeticWordImpl<MaxOp>::template call<Index>(
            MaxOp{}, tag, value, lower);
    return ScalarArithmeticWordImpl<MinOp>::template call<Index>(
        MinOp{}, tag, bounded_low, upper);
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ClampOp, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> lower, NativeWordVec<Tag> upper,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive,
      Policy policy) {
    const auto bounded_low =
        ScalarArithmeticWordImpl<MaxOp>::template call<Index>(
            MaxOp{}, tag, value, lower);
    return ScalarArithmeticWordImpl<MinOp>::template call<Index>(
        MinOp{}, tag, bounded_low, upper, mask, inactive, policy);
  }
};

template <>
struct NativeWordImpl<ScalarBackend, ClampOp> : ScalarClampWordImpl {};

/* **************************************************************************** */
//    ScalarFmaWordImpl and registrations                                      //
/* **************************************************************************** */

template <typename Op>
struct ScalarFmaWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordVec<Tag> c) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      if constexpr (::vecops::is_float_v<T>) {
        if constexpr (std::same_as<Op, FmaddOp>)
          a[lane] = static_cast<T>(a[lane] * b[lane] + c[lane]);
        else if constexpr (std::same_as<Op, FmsubOp>)
          a[lane] = static_cast<T>(a[lane] * b[lane] - c[lane]);
        else if constexpr (std::same_as<Op, FnmaddOp>)
          a[lane] = static_cast<T>(-(a[lane] * b[lane]) + c[lane]);
        else if constexpr (std::same_as<Op, FnmsubOp>)
          a[lane] = static_cast<T>(-(a[lane] * b[lane]) - c[lane]);
        else
          static_assert(dispatch_dependent_false<Op>);
      } else {
        const T product = scalar_arithmetic_binary<T>(
            a[lane], b[lane], [](auto x, auto y) { return x * y; });
        if constexpr (std::same_as<Op, FmaddOp>)
          a[lane] = scalar_arithmetic_binary<T>(
              product, c[lane], [](auto x, auto y) { return x + y; });
        else if constexpr (std::same_as<Op, FmsubOp>)
          a[lane] = scalar_arithmetic_binary<T>(
              product, c[lane], [](auto x, auto y) { return x - y; });
        else if constexpr (std::same_as<Op, FnmaddOp>)
          a[lane] = scalar_arithmetic_binary<T>(
              c[lane], product, [](auto x, auto y) { return x - y; });
        else if constexpr (std::same_as<Op, FnmsubOp>) {
          const T sum = scalar_arithmetic_binary<T>(
              product, c[lane], [](auto x, auto y) { return x + y; });
          a[lane] = scalar_arithmetic_binary<T>(
              T{}, sum, [](auto x, auto y) { return x - y; });
        } else
          static_assert(dispatch_dependent_false<Op>);
      }
    }
    return a;
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordVec<Tag> c, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, a, b, c), mask, inactive);
  }
};

template <>
struct NativeWordImpl<ScalarBackend, FmaddOp>
    : ScalarFmaWordImpl<FmaddOp> {};
template <>
struct NativeWordImpl<ScalarBackend, FmsubOp>
    : ScalarFmaWordImpl<FmsubOp> {};
template <>
struct NativeWordImpl<ScalarBackend, FnmaddOp>
    : ScalarFmaWordImpl<FnmaddOp> {};
template <>
struct NativeWordImpl<ScalarBackend, FnmsubOp>
    : ScalarFmaWordImpl<FnmsubOp> {};

/* **************************************************************************** */
//    scalar_unary_arithmetic helper                                           //
/* **************************************************************************** */

template <Element T>
VECOPS_ALWAYS_INLINE T scalar_unary_arithmetic(T value, bool absolute) {
  if constexpr (std::same_as<T, bfloat16_t>) {
    const uint16_t bits = value.to_bits();
    return bfloat16_t::from_bits(
        absolute ? static_cast<uint16_t>(bits & 0x7fffu)
                 : static_cast<uint16_t>(bits ^ 0x8000u));
  } else if constexpr (std::same_as<T, float16_t>) {
    const uint16_t bits = value.to_bits();
    return float16_t::from_bits(
        absolute ? static_cast<uint16_t>(bits & 0x7fffu)
                 : static_cast<uint16_t>(bits ^ 0x8000u));
  } else if constexpr (std::same_as<T, float32_t>) {
    const uint32_t bits = ::vecops::bitcast<uint32_t>(value);
    return ::vecops::bitcast<T>(
        absolute ? bits & 0x7fffffffu : bits ^ 0x80000000u);
  } else if constexpr (std::same_as<T, float64_t>) {
    const uint64_t bits = ::vecops::bitcast<uint64_t>(value);
    return ::vecops::bitcast<T>(
        absolute ? bits & 0x7fffffffffffffffull
                 : bits ^ 0x8000000000000000ull);
  } else {
    using U = std::make_unsigned_t<T>;
    if constexpr (std::is_unsigned_v<T>) {
      return absolute ? value : static_cast<T>(U{} - value);
    } else {
      const U bits = ::vecops::bitcast<U>(value);
      const U result = absolute && (bits >> (sizeof(T) * 8 - 1)) == 0
          ? bits
          : static_cast<U>(U{} - bits);
      return ::vecops::bitcast<T>(result);
    }
  }
}

/* **************************************************************************** */
//    ScalarUnaryArithmeticWordImpl — NegOp, AbsOp and registrations          //
/* **************************************************************************** */

template <typename Op>
struct ScalarUnaryArithmeticWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      value[lane] = scalar_unary_arithmetic(
          value[lane], std::same_as<Op, AbsOp>);
    }
    return value;
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, value), mask, inactive);
  }
};

template <>
struct NativeWordImpl<ScalarBackend, NegOp>
    : ScalarUnaryArithmeticWordImpl<NegOp> {};

template <>
struct NativeWordImpl<ScalarBackend, AbsOp>
    : ScalarUnaryArithmeticWordImpl<AbsOp> {};

/* **************************************************************************** */
//    ScalarFloatingUnaryWordImpl — SqrtOp and registrations              //
/* **************************************************************************** */

template <typename Op>
struct ScalarFloatingUnaryWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
      if constexpr (
          std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>) {
        if constexpr (std::same_as<Op, SqrtOp>) {
          value[lane] = static_cast<T>(
              std::sqrt(static_cast<float>(value[lane])));
        } else {
          static_assert(dispatch_dependent_false<Op>);
        }
      } else {
        if constexpr (std::same_as<Op, SqrtOp>)
          value[lane] = static_cast<T>(std::sqrt(value[lane]));
        else
          static_assert(dispatch_dependent_false<Op>);
      }
    }
    return value;
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, value), mask, inactive);
  }
};

template <>
struct NativeWordImpl<ScalarBackend, SqrtOp>
    : ScalarFloatingUnaryWordImpl<SqrtOp> {};
} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_ARITHMETIC_H
