//
// Created by renyz on 2026/3/13.
//

#ifndef VECOPS_CORETYPES_H
#define VECOPS_CORETYPES_H

#include <cstdint>  // for standard int defs

#include "vecops/Features.h"
#include "vecops/CoreDefs.h"
#include "vecops/util/BFloat16.h"
#include "vecops/util/Float16.h"

namespace vecops {

/**
 * Float types
 */
using bfloat16_t = BFloat16;
using float16_t = Float16;
using float32_t = float;
using float64_t = double;
/**
 * Signed native int, having the same width as machine word.
 */
using nint_t = ptrdiff_t;
/**
 * Unsigned native int, having the same width as machine word.
 */
using nuint_t = size_t;

/**
 * Integer types
 */
using std::int8_t;
using std::uint8_t;
using std::int16_t;
using std::uint16_t;
using std::int32_t;
using std::uint32_t;
using std::int64_t;
using std::uint64_t;

template <typename T>
struct TypeTraits {
  static constexpr bool is_integer = std::is_integral_v<T>;
  static constexpr bool is_signed = std::is_signed_v<T>;
  static constexpr bool is_float = std::is_floating_point_v<T>;
  static constexpr size_t bits = sizeof(T) * 8;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float16 = false;
};

template <>
struct TypeTraits<bfloat16_t> {
  static constexpr bool is_integer = false;
  static constexpr bool is_signed = true;
  static constexpr bool is_float = true;
  static constexpr size_t bits = 16;
  static constexpr bool is_bfloat16 = true;
  static constexpr bool is_float16 = false;
};

template <>
struct TypeTraits<float16_t> {
  static constexpr bool is_integer = false;
  static constexpr bool is_signed = true;
  static constexpr bool is_float = true;
  static constexpr size_t bits = 16;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float16 = true;
};

template <typename T> constexpr bool IsIntV = TypeTraits<T>::is_integer;
template <typename T> constexpr bool IsFloatV = TypeTraits<T>::is_float;
template <typename T> constexpr bool IsSignedV = TypeTraits<T>::is_signed;
template <typename T> constexpr bool IsBfloat16V = TypeTraits<T>::is_bfloat16;
template <typename T> constexpr bool IsFloat16V = TypeTraits<T>::is_float16;
template <typename T> constexpr bool IsStandardFloatV = IsFloatV<T> && !IsBfloat16V<T> && !IsFloat16V<T>;
template <typename T> constexpr size_t TypeBitsV = TypeTraits<T>::bits;


} // namespace vecops

#endif //VECOPS_CORETYPES_H
