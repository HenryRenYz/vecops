// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// @vecops-test-shards: 4

#include <gtest/gtest.h>

#include <cfenv>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "vecops/vec/Memory.h"
#include "vecops/vec/Rounding.h"
#include "TestHelpers.h"
#include "TestShard.h"

#pragma STDC FENV_ACCESS ON

namespace vec = vecops::vec;

template <typename T>
void run_rounding_test();
template <typename T>
void run_rounding_special_values_test();
template <typename T>
void run_rounding_environment_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == vec_test::FloatingElements::size);

namespace {

enum class RoundingKind {
  Floor,
  Ceil,
  Trunc,
  Round,
  RoundEven,
  NearbyInt,
  Rint
};

double reference_round_even(double value) {
  if (!std::isfinite(value) || value == 0.0) return value;
  const double magnitude = std::abs(value);
  const double lower = std::floor(magnitude);
  const double fraction = magnitude - lower;
  const double rounded = fraction < 0.5
      ? lower
      : fraction > 0.5
          ? lower + 1.0
          : std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0;
  return std::copysign(rounded, value);
}

template <RoundingKind Kind, typename T>
T expected_rounding(T value) {
  const double input = static_cast<double>(value);
  if constexpr (Kind == RoundingKind::Floor)
    return static_cast<T>(std::floor(input));
  else if constexpr (Kind == RoundingKind::Ceil)
    return static_cast<T>(std::ceil(input));
  else if constexpr (Kind == RoundingKind::Trunc)
    return static_cast<T>(std::trunc(input));
  else if constexpr (Kind == RoundingKind::Round)
    return static_cast<T>(std::round(input));
  else if constexpr (Kind == RoundingKind::RoundEven)
    return static_cast<T>(reference_round_even(input));
  else if constexpr (Kind == RoundingKind::NearbyInt)
    return static_cast<T>(std::nearbyint(input));
  else
    return static_cast<T>(std::rint(input));
}

template <RoundingKind Kind, vec::FloatingTag Tag, typename... Options>
vec::Vec<Tag> invoke_rounding(
    Tag, vec::Vec<Tag> value, Options&&... options) {
  if constexpr (Kind == RoundingKind::Floor)
    return vec::floor(value, std::forward<Options>(options)...);
  else if constexpr (Kind == RoundingKind::Ceil)
    return vec::ceil(value, std::forward<Options>(options)...);
  else if constexpr (Kind == RoundingKind::Trunc)
    return vec::trunc(value, std::forward<Options>(options)...);
  else if constexpr (Kind == RoundingKind::Round)
    return vec::round(value, std::forward<Options>(options)...);
  else if constexpr (Kind == RoundingKind::RoundEven)
    return vec::round_even(value, std::forward<Options>(options)...);
  else if constexpr (Kind == RoundingKind::NearbyInt)
    return vec::nearbyint(value, std::forward<Options>(options)...);
  else
    return vec::rint(value, std::forward<Options>(options)...);
}

template <typename T>
T rounding_operand(vecops::nint_t lane) {
  constexpr double values[] = {
      -3.75, -2.5, -1.5, -0.5, -0.0, 0.0,
      0.25, 0.5, 1.5, 2.5, 3.75, 17.0};
  return static_cast<T>(values[lane % 12]);
}

template <RoundingKind Kind, bool FullOptions, vec::FloatingTag Tag>
void verify_rounding_shape(Tag tag) {
  using T = vec::ElementOf<Tag>;
  auto value = vec::zeros(tag);
  auto merge = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  const T scalar_merge = static_cast<T>(13.0);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    value = vec::set(tag, value, lane, rounding_operand<T>(lane));
    merge = vec::set(tag, merge, lane, static_cast<T>(lane + 19.0));
    mask = vec::set(tag, mask, lane, (lane % 3) != 1);
  }

  const auto result = invoke_rounding<Kind>(tag, value);
  const auto explicit_result = [&] {
    if constexpr (Kind == RoundingKind::Floor) return vec::floor(tag, value);
    else if constexpr (Kind == RoundingKind::Ceil) return vec::ceil(tag, value);
    else if constexpr (Kind == RoundingKind::Trunc) return vec::trunc(tag, value);
    else if constexpr (Kind == RoundingKind::Round) return vec::round(tag, value);
    else if constexpr (Kind == RoundingKind::RoundEven) return vec::round_even(tag, value);
    else if constexpr (Kind == RoundingKind::NearbyInt) return vec::nearbyint(tag, value);
    else return vec::rint(tag, value);
  }();
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T expected = expected_rounding<Kind>(rounding_operand<T>(lane));
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, result, lane)))
        << "kind=" << static_cast<int>(Kind) << ", lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, explicit_result, lane)))
        << "kind=" << static_cast<int>(Kind)
        << ", explicit lane=" << lane;
  }

  if constexpr (FullOptions) {
    const auto default_masked = invoke_rounding<Kind>(
        tag, value, vec::opt::masked(mask));
    const auto zero_masked = invoke_rounding<Kind>(
        tag, value, vec::opt::masked(mask), vec::opt::zero);
    const auto scalar_masked = invoke_rounding<Kind>(
        tag, value, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
    const auto vector_masked = invoke_rounding<Kind>(
        tag, value, vec::opt::masked(mask), vec::opt::merge(merge));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const bool active = (lane % 3) != 1;
      const T input = rounding_operand<T>(lane);
      const T expected = expected_rounding<Kind>(input);
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : input,
          vec::get(tag, default_masked, lane)));
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : T{}, vec::get(tag, zero_masked, lane)));
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : scalar_merge,
          vec::get(tag, scalar_masked, lane)));
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : static_cast<T>(lane + 19.0),
          vec::get(tag, vector_masked, lane)));
    }
  }
}

template <typename T>
void verify_half_boundary_neighbors() {
  if constexpr (std::same_as<T, float> || std::same_as<T, double>) {
    using Tag = vec::ScalableTag<T>;
    const T half = static_cast<T>(0.5);
    const T below = std::nextafter(half, static_cast<T>(0));
    const T above = std::nextafter(half, static_cast<T>(1));
    EXPECT_TRUE(vec_test::values_identical(
        static_cast<T>(0),
        vec::get(Tag{}, vec::round(vec::fill(Tag{}, below)), 0)));
    EXPECT_TRUE(vec_test::values_identical(
        static_cast<T>(1),
        vec::get(Tag{}, vec::round(vec::fill(Tag{}, above)), 0)));
  }
}

struct RestoreRoundingMode {
  int mode = std::fegetround();
  ~RestoreRoundingMode() { std::fesetround(mode); }
};

} // namespace

template <typename T>
void run_rounding_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::FloatingTag Tag>() {
    constexpr bool options = vec_test::exhaustive_options_shape<Tag>;
    verify_rounding_shape<RoundingKind::Floor, options>(Tag{});
    verify_rounding_shape<RoundingKind::Ceil, options>(Tag{});
    verify_rounding_shape<RoundingKind::Trunc, options>(Tag{});
    verify_rounding_shape<RoundingKind::Round, options>(Tag{});
    verify_rounding_shape<RoundingKind::RoundEven, options>(Tag{});
    verify_rounding_shape<RoundingKind::NearbyInt, options>(Tag{});
    verify_rounding_shape<RoundingKind::Rint, options>(Tag{});
  });
  verify_half_boundary_neighbors<T>();
}

template <typename T>
void run_rounding_special_values_test() {
  using Tag = vec::ScalableTag<T>;
  const Tag tag{};
  const T positive_zero = static_cast<T>(0.0);
  const T negative_zero = static_cast<T>(-0.0);
  const T infinity = static_cast<T>(
      std::numeric_limits<double>::infinity());
  const T nan = static_cast<T>(
      std::numeric_limits<double>::quiet_NaN());
  const auto verify = [&](auto op) {
    const T positive_result = vec::get(
        tag, op(vec::fill(tag, positive_zero)), 0);
    const T negative_result = vec::get(
        tag, op(vec::fill(tag, negative_zero)), 0);
    const T infinity_result = vec::get(
        tag, op(vec::fill(tag, infinity)), 0);
    const T nan_result = vec::get(tag, op(vec::fill(tag, nan)), 0);
    EXPECT_FALSE(std::signbit(static_cast<double>(positive_result)));
    EXPECT_TRUE(std::signbit(static_cast<double>(negative_result)));
    EXPECT_TRUE(std::isinf(static_cast<double>(infinity_result)));
    EXPECT_TRUE(std::isnan(static_cast<double>(nan_result)));
  };
  verify(vec::floor);
  verify(vec::ceil);
  verify(vec::trunc);
  verify(vec::round);
  verify(vec::round_even);
  verify(vec::nearbyint);
  verify(vec::rint);
}

template <typename T>
void run_rounding_environment_test() {
  using Tag = vec::ScalableTag<T>;
  const Tag tag{};
  RestoreRoundingMode restore;
  constexpr int modes[] = {
      FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  for (const int mode : modes) {
    ASSERT_EQ(std::fesetround(mode), 0);
    for (const double source : {-1.25, 1.25}) {
      const T input = static_cast<T>(source);
      const T expected = static_cast<T>(
          std::nearbyint(static_cast<double>(input)));
      const auto value = vec::fill(tag, input);
      std::feclearexcept(FE_ALL_EXCEPT);
      const auto nearby_result = vec::nearbyint(value);
      EXPECT_EQ(std::fetestexcept(FE_INEXACT), 0);
      EXPECT_TRUE(vec_test::values_identical(
          expected, vec::get(tag, nearby_result, 0)));
      EXPECT_TRUE(vec_test::values_identical(
          expected, vec::get(tag, vec::rint(value), 0)));
    }
  }
}

using ShardType = vec_test::FloatingElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_rounding_test<ShardType>();
template void run_rounding_special_values_test<ShardType>();
template void run_rounding_environment_test<ShardType>();

#else

template <typename T>
class VecRoundingTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecRoundingTest,
    vec_test::FloatingElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecRoundingTest, CoversEveryOperationLaneShapeAndPopulation) {
  run_rounding_test<TypeParam>();
}

TYPED_TEST(VecRoundingTest, PreservesSpecialValueClassesAndZeroSigns) {
  run_rounding_special_values_test<TypeParam>();
}

TYPED_TEST(VecRoundingTest, NearbyIntAndRintUseCurrentEnvironmentMode) {
  run_rounding_environment_test<TypeParam>();
}

#endif
