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
      const auto u_max = TIn(std::numeric_limits<TOut>::max());
      return TOut(std::min(u_max, v));
    } else {
      using Wide = std::conditional_t<(sizeof(TPromoteOut) < sizeof(TIn)), TIn, TPromoteOut>;
      const auto HI = Wide(std::numeric_limits<TOut>::max());
      const auto LO = Wide(std::numeric_limits<TOut>::min());
      return TOut(std::max(LO, std::min(HI, Wide(v))));
    }
  } else {
    return TOut(TPromoteOut(TPromoteIn(v)));
  }
}

} // namespace vecops

#endif //VECOPS_SCALARCONVERT_H
