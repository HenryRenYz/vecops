// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <cmath>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "vecops/vec/Vec.h"
#include "vecops/execution/FloatControl.h"
#include "TestHelpers.h"

namespace v = vecops::vec;

template <typename T>
class MathSubnormalTest : public ::testing::Test {};
TYPED_TEST_SUITE(MathSubnormalTest, vec_test::FloatingElementTypes,
                 vec_test::ElementTypeName);

template <typename T, v::ExpBase Base, v::Accuracy A, bool NegativeOnly>
void check_exp_storage_contract() {
  // Well inside each storage format's subnormal-output interval. All
  // inputs are finite normal numbers, independently of Inf/NaN behavior.
  constexpr bool half = std::same_as<T, vecops::float16_t>;
  constexpr bool wide = std::same_as<T, double>;
  const T input = Base == v::ExpBase::E ? T(half ? -12 : wide ? -710 : -90)
      : Base == v::ExpBase::Base2 ? T(half ? -20 : wide ? -1030 : -130)
      : T(half ? -6 : wide ? -310 : -40);
  const v::ScalableTag<T, 1> tag;
  const auto result = v::ExpCpo<Base, NegativeOnly>{}(
      tag, v::fill(tag, input), v::opt::math::accuracy<A>);
#ifdef VECOPS_PRESERVE_SUBNORMALS
  constexpr bool gradual = A == v::Accuracy::Strict;
#else
  constexpr bool gradual = false;
#endif
  for (vecops::nint_t lane = 0; lane < v::size(tag); ++lane) {
    const double value = static_cast<double>(v::get(tag, result, lane));
    if constexpr (gradual) {
      EXPECT_GT(value, 0);
      EXPECT_LT(value, static_cast<double>(std::numeric_limits<T>::min()));
    } else {
      EXPECT_EQ(value, 0) << "Fast/Estimate and default Strict must flush";
    }
  }
  // Inactive values are caller-owned and must survive even when they are
  // subnormal. The flush contract applies only to computed active lanes.
  const auto inactive = v::fill(tag, std::numeric_limits<T>::denorm_min());
  const auto mask = v::mwhilelt(tag, vecops::nint_t{0}, vecops::nint_t{1});
  const auto merged = v::ExpCpo<Base, NegativeOnly>{}(
      tag, v::fill(tag, input), v::opt::math::accuracy<A>,
      v::opt::masked(mask), v::opt::merge(inactive));
  for (vecops::nint_t lane = 1; lane < v::size(tag); ++lane)
    EXPECT_EQ(static_cast<double>(v::get(tag, merged, lane)),
              static_cast<double>(std::numeric_limits<T>::denorm_min()));
}

template <typename T, v::ExpBase Base, v::Accuracy A>
void check_domains() {
  check_exp_storage_contract<T, Base, A, false>();
  check_exp_storage_contract<T, Base, A, true>();
}
template <typename T, v::ExpBase Base>
void check_tiers() {
  check_domains<T, Base, v::Accuracy::Strict>();
  check_domains<T, Base, v::Accuracy::Fast>();
  check_domains<T, Base, v::Accuracy::Estimate>();
}

TYPED_TEST(MathSubnormalTest, EveryExponentialBaseTierAndDomain) {
  check_tiers<TypeParam, v::ExpBase::E>();
  check_tiers<TypeParam, v::ExpBase::Base2>();
  check_tiers<TypeParam, v::ExpBase::Base10>();
}

template <typename T, v::Accuracy A>
void check_reciprocal_storage_contract() {
  using Bits = std::conditional_t<sizeof(T) == 8, std::uint64_t,
      std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint16_t>>;
  const v::ScalableTag<T> tag;
  for (bool negative : {false, true}) {
    const T input = negative ? T(-std::numeric_limits<T>::max())
                             : std::numeric_limits<T>::max();
    const bool previous = vecops::execution::current_thread_flush_subnormals();
    for (bool flush : {false, true}) {
      vecops::execution::set_current_thread_flush_subnormals(flush);
      const auto result = v::rcp(tag, v::fill(tag, input), v::opt::math::accuracy<A>);
      const Bits bits = std::bit_cast<Bits>(v::get(tag, result, 0));
      // Inspect the actual stored bits after restoring the mode: DAZ makes
      // floating comparisons alone incapable of detecting this regression.
      vecops::execution::set_current_thread_flush_subnormals(previous);
#ifdef VECOPS_PRESERVE_SUBNORMALS
      if constexpr (A == v::Accuracy::Strict) {
        if (!flush) EXPECT_NE(bits, Bits(negative) << (sizeof(T) * 8 - 1));
      } else
#endif
      {
        EXPECT_EQ(bits, Bits(negative) << (sizeof(T) * 8 - 1));
      }
    }
  }
}

TYPED_TEST(MathSubnormalTest, ReciprocalFlushPreservesSignWithAndWithoutHardwareFTZ) {
  check_reciprocal_storage_contract<TypeParam, v::Accuracy::Strict>();
  check_reciprocal_storage_contract<TypeParam, v::Accuracy::Fast>();
  check_reciprocal_storage_contract<TypeParam, v::Accuracy::Estimate>();
}

#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
template <typename T, v::Accuracy A>
void check_negative_nan() {
  const v::ScalableTag<T, 1> tag;
  const auto result = v::exp_neg(
      tag, v::fill(tag, std::numeric_limits<T>::quiet_NaN()),
      v::opt::math::accuracy<A>, v::opt::first(1), v::opt::merge(T(3)));
  EXPECT_TRUE(std::isnan(static_cast<double>(v::get(tag, result, 0))));
  for (vecops::nint_t lane = 1; lane < v::size(tag); ++lane)
    EXPECT_EQ(static_cast<double>(v::get(tag, result, lane)), 3);
}

TYPED_TEST(MathSubnormalTest, NegativeExponentialNaNsAndInactiveLanes) {
  check_negative_nan<TypeParam, v::Accuracy::Strict>();
  check_negative_nan<TypeParam, v::Accuracy::Fast>();
  check_negative_nan<TypeParam, v::Accuracy::Estimate>();
}
#endif
