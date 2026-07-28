#ifndef VECOPS_TESTS_NVEC_TESTHELPERS_H
#define VECOPS_TESTS_NVEC_TESTHELPERS_H

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>
#include <utility>

#include "vecops/nvec/VecBase.h"

namespace nvec_test {

using AllElementTypes = ::testing::Types<
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

template <typename T>
bool values_identical(T expected, T actual) {
  if constexpr (
      std::same_as<T, vecops::bfloat16_t> ||
      std::same_as<T, vecops::float16_t>) {
    return expected.to_bits() == actual.to_bits();
  } else if constexpr (std::same_as<T, vecops::float32_t>) {
    return ::vecops::bitcast<std::uint32_t>(expected) ==
           ::vecops::bitcast<std::uint32_t>(actual);
  } else if constexpr (std::same_as<T, vecops::float64_t>) {
    return ::vecops::bitcast<std::uint64_t>(expected) ==
           ::vecops::bitcast<std::uint64_t>(actual);
  } else {
    return expected == actual;
  }
}

/** Visits every scalable shape valid for T on the minimum 16-byte word. */
template <typename T, typename Visitor>
void for_each_scalable_shape(Visitor&& visitor) {
  visitor.template operator()<vecops::nvec::ScalableTag<T, -1>>();
  if constexpr (sizeof(T) <= 4) {
    visitor.template operator()<vecops::nvec::ScalableTag<T, -2>>();
  }
  if constexpr (sizeof(T) <= 2) {
    visitor.template operator()<vecops::nvec::ScalableTag<T, -3>>();
  }
  visitor.template operator()<vecops::nvec::ScalableTag<T, 0>>();
  visitor.template operator()<vecops::nvec::ScalableTag<T, 1>>();
  visitor.template operator()<vecops::nvec::ScalableTag<T, 2>>();
}

} // namespace nvec_test

#endif // VECOPS_TESTS_NVEC_TESTHELPERS_H
