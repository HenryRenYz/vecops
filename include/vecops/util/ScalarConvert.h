//
// Created by renyz on 2026/3/21.
//

#ifndef VECOPS_SCALARCONVERT_H
#define VECOPS_SCALARCONVERT_H

#include <type_traits>
#include <numeric>
#include <cmath>
#include "CoreTypes.h"
#include "TypeTraits.h"

namespace vecops {

namespace details {

template <typename T, typename = void/*SFINAE*/>
struct DTypePromote {
  using Type = T;
};

template <typename T>
struct DTypePromote<T, std::enable_if_t<(is_int<T> && is_none<T, uint32_t> && sizeof(T) <= sizeof(int32_t))>> {
  using Type = int32_t;
};

template <typename T>
struct DTypePromote<T, std::enable_if_t<(is_float<T> && sizeof(T) <= sizeof(float32_t))>> {
  using Type = float32_t;
};

} // namespace details

template <typename TOut, typename TIn>
VECOPS_INLINE constexpr TOut convert(TIn v) {
  using TPromoteIn = details::DTypePromote<TIn>::Type;
  using TPromoteOut = details::DTypePromote<TOut>::Type;
  if constexpr (is_int<TOut> && sizeof(TOut) < sizeof(TIn)) {
    if constexpr (is_unsigned_int<TIn>) {
      const auto HI = TIn(std::numeric_limits<TOut>::max());
      return TOut(std::min(HI, v));
    } else {
      using Wide = std::conditional_t<(sizeof(TPromoteOut) < sizeof(TIn)), TIn, TPromoteOut>;
      const auto HI = Wide(std::numeric_limits<TOut>::max());
      const auto LO = Wide(std::numeric_limits<TOut>::min());
      return TOut(std::max(LO, std::min(HI, Wide(v))));
    }
  } else if constexpr (is_int<TOut> && is_int<TIn> && sizeof(TOut) == sizeof(TIn)) {
    if constexpr (is_unsigned_int<TOut>) {
      const auto LO = TIn(0);
      return TOut(std::max(v, LO));
    } else if constexpr (is_unsigned_int<TIn>) {
      const auto HI = TIn(std::numeric_limits<TOut>::max());
      return TOut(std::min(v, HI));
    } else {
      return TOut(v);
    }
  } else if constexpr (is_int<TOut> && is_float<TIn>) {
    // Clamp in the floating-point domain before converting: a plain
    // static_cast<uint>(negative float) is undefined behavior, and
    // compilers resolve that UB differently at different optimization
    // levels. Clamping the already-converted integer is too late — the
    // UB has already produced a wrapped value.
    const auto f = static_cast<TPromoteIn>(v);
    if (std::isnan(static_cast<double>(f))) return TOut(0);
    if constexpr (is_unsigned_int<TOut>) {
      constexpr auto hi = [] {
        using F = TPromoteIn;
        // Largest float exactly representable in TOut.
        F value = F(1);
        while (value * F(2) <= F(std::numeric_limits<TOut>::max())) {
          value *= F(2);
        }
        return value - F(1) == F(std::numeric_limits<TOut>::max())
            ? F(std::numeric_limits<TOut>::max())
            : value - F(1);
      }();
      return TOut(std::min(std::max(f, TPromoteIn(0)), hi));
    } else {
      using F = TPromoteIn;
      constexpr F hi = F(std::numeric_limits<TOut>::max());
      constexpr F lo = F(std::numeric_limits<TOut>::min());
      return TOut(std::min(std::max(f, lo), hi));
    }
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
