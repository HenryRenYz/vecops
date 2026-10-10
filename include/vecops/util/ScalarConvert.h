// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT

#ifndef VECOPS_SCALARCONVERT_H
#define VECOPS_SCALARCONVERT_H

#include <type_traits>
#include <numeric>
#include <cmath>
#include <bit>
#include <cstdint>
#include <limits>
#include "CoreTypes.h"
#include "TypeTraits.h"

namespace vecops {

namespace details {

template <typename T, typename = void/*SFINAE*/>
struct DTypePromote {
  using type = T;
};

template <typename T>
  requires (is_int_v<T> && is_none_v<T, uint32_t> && sizeof(T) <= sizeof(int32_t))
struct DTypePromote<T, void> {
  using type = int32_t;
};

template <typename T>
  requires (is_float_v<T> && sizeof(T) <= sizeof(float32_t))
struct DTypePromote<T, void> {
  using type = float32_t;
};

} // namespace details

/**
 * Scalar numeric conversion. Integer narrowing saturates; float-to-integer
 * conversion truncates toward zero, saturates at both endpoints and maps NaN
 * to zero, including builds using fast-math. Use wrap_convert for low bits.
 */
template <typename TOut, typename TIn>
VECOPS_INLINE constexpr TOut convert(TIn v) {
  using TPromoteIn = details::DTypePromote<TIn>::type;
  using TPromoteOut = details::DTypePromote<TOut>::type;
  if constexpr (is_int_v<TOut> && is_int_v<TIn> && sizeof(TOut) < sizeof(TIn)) {
    if constexpr (is_unsigned_int_v<TIn>) {
      const auto HI = TIn(std::numeric_limits<TOut>::max());
      return TOut(std::min(HI, v));
    } else {
      using Wide = std::conditional_t<(sizeof(TPromoteOut) < sizeof(TIn)), TIn, TPromoteOut>;
      const auto HI = Wide(std::numeric_limits<TOut>::max());
      const auto LO = Wide(std::numeric_limits<TOut>::min());
      return TOut(std::max(LO, std::min(HI, Wide(v))));
    }
  } else if constexpr (is_int_v<TOut> && is_int_v<TIn> && sizeof(TOut) == sizeof(TIn)) {
    if constexpr (is_unsigned_int_v<TOut>) {
      const auto LO = TIn(0);
      return TOut(std::max(v, LO));
    } else if constexpr (is_unsigned_int_v<TIn>) {
      const auto HI = TIn(std::numeric_limits<TOut>::max());
      return TOut(std::min(v, HI));
    } else {
      return TOut(v);
    }
  } else if constexpr (is_int_v<TOut> && is_float_v<TIn>) {
    // Clamp in the floating-point domain before converting: a plain
    // static_cast<uint>(negative float) is undefined behavior, and
    // compilers resolve that UB differently at different optimization
    // levels. Clamping the already-converted integer is too late — the
    // UB has already produced a wrapped value.
    const auto f = static_cast<TPromoteIn>(v);
    using Bits = std::conditional_t<sizeof(TPromoteIn) == 4, uint32_t, uint64_t>;
    constexpr Bits sign = Bits(1) << (sizeof(Bits) * 8 - 1);
    constexpr Bits infinity = sizeof(Bits) == 4 ? Bits(0x7f800000u) : Bits(0x7ff0000000000000ull);
    // Bit classification survives -ffast-math; isnan alone does not.
    if ((std::bit_cast<Bits>(f) & ~sign) > infinity) return TOut(0);
    constexpr auto hi = TPromoteIn(std::numeric_limits<TOut>::max());
    constexpr auto lo = TPromoteIn(std::numeric_limits<TOut>::min());
    // Return the integer endpoint directly. Its floating representation can
    // round outside the destination range (e.g. INT32_MAX becomes 2^31).
    if (f >= hi) return std::numeric_limits<TOut>::max();
    if (f <= lo) return std::numeric_limits<TOut>::min();
    return TOut(f);
  } else {
    return TOut(TPromoteOut(TPromoteIn(v)));
  }
}

/**
 * Integer narrowing with deterministic low-bit/modulo semantics.
 */
template <typename TOut, typename TIn>
  requires (
      std::is_integral_v<TOut> && std::is_integral_v<TIn> &&
      sizeof(TOut) < sizeof(TIn))
VECOPS_INLINE constexpr TOut wrap_convert(TIn v) {
  using UIn = std::make_unsigned_t<TIn>;
  using UOut = std::make_unsigned_t<TOut>;
  const UOut low = static_cast<UOut>(static_cast<UIn>(v));
  return static_cast<TOut>(low);
}

} // namespace vecops

#endif //VECOPS_SCALARCONVERT_H
