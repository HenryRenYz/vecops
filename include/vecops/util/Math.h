//
// Created by renyz on 2026/3/17.
//

#ifndef VECOPS_MATH_H
#define VECOPS_MATH_H

#include <bit>
#include <concepts>
#include <type_traits>

#include "Assertion.h"
#include "CoreTypes.h"
#include "Features.h"

#ifdef ARCH_X86_FAMILY
  #include <immintrin.h>
#endif

namespace vecops {

template <typename T>
constexpr int log2_floor(T x) noexcept {
  if (x == 0) return -1;
  return std::bit_width(std::make_unsigned_t<T>(x)) - 1;
}

VECOPS_ALWAYS_INLINE constexpr bool is_aligned(int alignment, const void * p) {
  if (!std::is_constant_evaluated()) {
    VECOPS_ASSERT(
        alignment > 0 && (alignment & (alignment - 1)) == 0,
        "Alignment must be positive and power of 2: %d", alignment);
  }
  return (nuint_t(p) & (alignment - 1)) == 0;
}

VECOPS_ALWAYS_INLINE constexpr uint32_t tailing_mask(int32_t n) {
  if (std::is_constant_evaluated()) {
    return n >= 32 ? uint32_t(-1) : ((1u << n) - 1);
  }
  #ifdef HAS_BMI2
  return _bzhi_u32(uint32_t(-1), n);
  #else
  return n >= 32 ? uint32_t(-1) : ((1u << n) - 1);
  #endif
}

VECOPS_ALWAYS_INLINE constexpr uint64_t tailing_mask(int64_t n) {
  if (std::is_constant_evaluated()) {
    return n >= 64 ? uint64_t(-1) : ((1uLL << n) - 1);
  }
  #ifdef HAS_BMI2
  return _bzhi_u64(uint64_t(-1), n);
  #else
  return n >= 64 ? uint64_t(-1) : ((1uLL << n) - 1);
  #endif
}

template <std::integral T, typename X>
  requires std::convertible_to<X, T>
VECOPS_ALWAYS_INLINE constexpr T floor_div(T value, X divisor) {
  const T d = static_cast<T>(divisor);
  VECOPS_ASSERT(d > 0, "floor_div divisor must be positive");
  const T quotient = value / d;
  const T remainder = value % d;
  if constexpr (std::signed_integral<T>) {
    return quotient - static_cast<T>(remainder < 0);
  } else {
    return quotient;
  }
}

template <std::integral T, typename X>
  requires std::convertible_to<X, T>
VECOPS_ALWAYS_INLINE constexpr T ceil_div(T value, X divisor) {
  const T d = static_cast<T>(divisor);
  VECOPS_ASSERT(d > 0, "ceil_div divisor must be positive");
  const T quotient = value / d;
  const T remainder = value % d;
  return quotient + static_cast<T>(remainder > 0);
}

template <std::integral T, typename X>
  requires std::convertible_to<X, T>
VECOPS_ALWAYS_INLINE constexpr T align_down(T value, X alignment) {
  const T a = static_cast<T>(alignment);
  if constexpr (std::signed_integral<T>) {
    VECOPS_ASSERT(value >= 0, "align_down value must be non-negative");
  }
  return floor_div(value, a) * a;
}

template <std::integral T, typename X>
  requires std::convertible_to<X, T>
VECOPS_ALWAYS_INLINE constexpr T align_up(T value, X alignment) {
  const T a = static_cast<T>(alignment);
  if constexpr (std::signed_integral<T>) {
    VECOPS_ASSERT(value >= 0, "align_up value must be non-negative");
  }
  return ceil_div(value, a) * a;
}

template <std::integral T, typename X>
  requires std::convertible_to<X, T>
VECOPS_ALWAYS_INLINE constexpr T cdiv(T value, X divisor) {
  return ceil_div(value, divisor);
}

} // vecops

#endif //VECOPS_MATH_H
