// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_TESTS_VEC_TESTTYPES_H
#define VECOPS_TESTS_VEC_TESTTYPES_H

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <string>
#include <tuple>

#include "vecops/CoreTypes.h"

namespace vec_test {

template <typename... Ts>
struct TypeCatalog {
  using GTestTypes = ::testing::Types<Ts...>;
  using Tuple = std::tuple<Ts...>;

  static constexpr std::size_t size = sizeof...(Ts);

  template <std::size_t Index>
  using At = std::tuple_element_t<Index, Tuple>;
};

using AllElements = TypeCatalog<
    vecops::bfloat16_t,
    vecops::float16_t,
    vecops::float32_t,
    vecops::float64_t,
    vecops::int8_t,
    vecops::uint8_t,
    vecops::int16_t,
    vecops::uint16_t,
    vecops::int32_t,
    vecops::uint32_t,
    vecops::int64_t,
    vecops::uint64_t>;

using FloatingElements = TypeCatalog<
    vecops::bfloat16_t,
    vecops::float16_t,
    vecops::float32_t,
    vecops::float64_t>;

using IntegerElements = TypeCatalog<
    vecops::int8_t,
    vecops::uint8_t,
    vecops::int16_t,
    vecops::uint16_t,
    vecops::int32_t,
    vecops::uint32_t,
    vecops::int64_t,
    vecops::uint64_t>;

using AllElementTypes = AllElements::GTestTypes;
using FloatingElementTypes = FloatingElements::GTestTypes;
using IntegerElementTypes = IntegerElements::GTestTypes;

template <std::size_t Index>
using ElementAt = AllElements::At<Index>;

template <std::size_t Index>
using FloatingElementAt = FloatingElements::At<Index>;

template <std::size_t Index>
using IntegerElementAt = IntegerElements::At<Index>;

struct ElementTypeName {
  template <typename T>
  static std::string GetName(int) {
    if constexpr (std::same_as<T, vecops::bfloat16_t>) return "BFloat16";
    else if constexpr (std::same_as<T, vecops::float16_t>) return "Float16";
    else if constexpr (std::same_as<T, vecops::float32_t>) return "Float32";
    else if constexpr (std::same_as<T, vecops::float64_t>) return "Float64";
    else if constexpr (std::same_as<T, vecops::int8_t>) return "Int8";
    else if constexpr (std::same_as<T, vecops::uint8_t>) return "UInt8";
    else if constexpr (std::same_as<T, vecops::int16_t>) return "Int16";
    else if constexpr (std::same_as<T, vecops::uint16_t>) return "UInt16";
    else if constexpr (std::same_as<T, vecops::int32_t>) return "Int32";
    else if constexpr (std::same_as<T, vecops::uint32_t>) return "UInt32";
    else if constexpr (std::same_as<T, vecops::int64_t>) return "Int64";
    else if constexpr (std::same_as<T, vecops::uint64_t>) return "UInt64";
    else {
      static_assert(std::same_as<T, void>, "unregistered vec test type");
      return {};
    }
  }
};

} // namespace vec_test

#endif // VECOPS_TESTS_VEC_TESTTYPES_H
