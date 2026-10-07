// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// @vecops-test-shards: 4

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "vecops/vec/Arithmetic.h"
#include "TestHelpers.h"
#include "TestShard.h"

namespace vec = vecops::vec;

template <typename T>
void run_floating_unary_test();
template <typename T>
void run_floating_unary_special_values_test();
template <typename T>
void run_fixed_floating_unary_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == vec_test::FloatingElements::size);

enum class FloatingUnaryKind { Sqrt };

template <typename T>
T floating_unary_operand(vecops::nint_t lane, bool active = true) {
  if (!active) return static_cast<T>(-static_cast<float>(lane + 1));
  constexpr float values[] = {
      0.25F, 0.5F, 1.0F, 2.0F, 3.0F, 4.0F, 9.0F, 16.0F};
  return static_cast<T>(values[lane % 8]);
}

template <FloatingUnaryKind Kind, typename T>
T expected_floating_unary(T value) {
  const double widened = static_cast<double>(value);
  return static_cast<T>(std::sqrt(widened));
}

template <typename T>
void expect_floating_unary_near(T expected, T actual, bool estimate) {
  const double expected_value = static_cast<double>(expected);
  const double actual_value = static_cast<double>(actual);
  if (!estimate) {
    const double tolerance = std::same_as<T, vecops::bfloat16_t>
        ? std::abs(expected_value) * 0.008
        : std::same_as<T, vecops::float16_t>
            ? std::abs(expected_value) * 0.001
            : std::abs(expected_value) * 1.0e-12 + 1.0e-12;
    EXPECT_NEAR(expected_value, actual_value, tolerance);
  } else {
    EXPECT_LE(
        std::abs(actual_value - expected_value),
        std::max(std::abs(expected_value), 1.0) * 0.01);
  }
}

template <FloatingUnaryKind Kind, vec::FloatingTag Tag, typename... Options>
vec::Vec<Tag> invoke_floating_unary(
    Tag tag, vec::Vec<Tag> value, Options&&... options) {
  return vec::sqrt(value, std::forward<Options>(options)...);
}

template <FloatingUnaryKind Kind, bool FullOptions = true, vec::FloatingTag Tag>
void verify_floating_unary(Tag tag) {
  using T = vec::ElementOf<Tag>;
  constexpr bool estimate = false;
  auto value = vec::zeros(tag);
  auto vector_merge = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  const T scalar_merge = static_cast<T>(13.0F);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const bool active = (lane % 3) != 0;
    value = vec::set(
        tag, value, lane, floating_unary_operand<T>(lane, active));
    vector_merge = vec::set(
        tag, vector_merge, lane, static_cast<T>(lane + 17.0F));
    mask = vec::set(tag, mask, lane, active);
  }

  auto positive = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    positive = vec::set(
        tag, positive, lane, floating_unary_operand<T>(lane));
  const auto result = invoke_floating_unary<Kind>(tag, positive);
  const auto unmasked_result =
      invoke_floating_unary<Kind>(tag, positive, vec::opt::unmasked);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T input = floating_unary_operand<T>(lane);
    expect_floating_unary_near(
        expected_floating_unary<Kind>(input),
        vec::get(tag, result, lane), estimate);
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, result, lane),
        vec::get(tag, unmasked_result, lane)))
        << "explicit unmasked lane=" << lane;
  }

  if constexpr (FullOptions) {
    const auto masked_default = invoke_floating_unary<Kind>(
        tag, value, vec::opt::masked(mask));
    const auto masked_zero = invoke_floating_unary<Kind>(
        tag, value, vec::opt::masked(mask), vec::opt::zero);
    const auto masked_scalar = invoke_floating_unary<Kind>(
        tag, value, vec::opt::merge(scalar_merge), vec::opt::masked(mask));
    const auto masked_vector = invoke_floating_unary<Kind>(
        tag, value, vec::opt::masked(mask), vec::opt::merge(vector_merge));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const bool active = (lane % 3) != 0;
      const T input = floating_unary_operand<T>(lane, active);
      if (active) {
        const T expected = expected_floating_unary<Kind>(input);
        expect_floating_unary_near(
            expected, vec::get(tag, masked_default, lane), estimate);
        expect_floating_unary_near(
            expected, vec::get(tag, masked_zero, lane), estimate);
        expect_floating_unary_near(
            expected, vec::get(tag, masked_scalar, lane), estimate);
        expect_floating_unary_near(
            expected, vec::get(tag, masked_vector, lane), estimate);
      } else {
        EXPECT_TRUE(vec_test::values_identical(
            input, vec::get(tag, masked_default, lane)));
        EXPECT_TRUE(vec_test::values_identical(
            T{}, vec::get(tag, masked_zero, lane)));
        EXPECT_TRUE(vec_test::values_identical(
            scalar_merge, vec::get(tag, masked_scalar, lane)));
        EXPECT_TRUE(vec_test::values_identical(
            static_cast<T>(lane + 17.0F),
            vec::get(tag, masked_vector, lane)));
      }
    }
  }
}

template <typename T>
void run_floating_unary_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::FloatingTag Tag>() {
    constexpr bool options = vec_test::exhaustive_options_shape<Tag>;
    verify_floating_unary<FloatingUnaryKind::Sqrt, options>(Tag{});
  });
}

template <typename T>
void run_floating_unary_special_values_test() {
  using Tag = vec::ScalableTag<T>;
  const auto nan = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
  const auto infinity = static_cast<T>(std::numeric_limits<double>::infinity());
  const auto sqrt_negative = vec::sqrt(vec::fill(Tag{}, static_cast<T>(-1)));
  const auto sqrt_nan = vec::sqrt(vec::fill(Tag{}, nan));
  const auto sqrt_infinity = vec::sqrt(vec::fill(Tag{}, infinity));
  EXPECT_TRUE(std::isnan(static_cast<double>(vec::get(Tag{}, sqrt_negative, 0))));
  EXPECT_TRUE(std::isnan(static_cast<double>(vec::get(Tag{}, sqrt_nan, 0))));
  EXPECT_TRUE(std::isinf(static_cast<double>(vec::get(Tag{}, sqrt_infinity, 0))));
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

template <typename T>
void run_fixed_floating_unary_test() {
  if constexpr (!std::same_as<T, vecops::float32_t>) {
    GTEST_SKIP() << "f32 is the representative >4-word floating type";
  } else {
    constexpr vecops::nint_t word_lanes =
        FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
    constexpr vecops::nint_t lanes = static_cast<vecops::nint_t>(
        std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
    using Tag = vec::FixedTag<T, lanes>;
    EXPECT_GT(vec::num_words(Tag{}), 4);
    verify_floating_unary<FloatingUnaryKind::Sqrt>(Tag{});
  }
}

#endif

using ShardType = vec_test::FloatingElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_floating_unary_test<ShardType>();
template void run_floating_unary_special_values_test<ShardType>();
#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
template void run_fixed_floating_unary_test<ShardType>();
#endif

#else

template <typename T>
class VecFloatingUnaryTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecFloatingUnaryTest,
    vec_test::FloatingElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecFloatingUnaryTest, CoversEveryLaneShapeAndPopulation) {
  run_floating_unary_test<TypeParam>();
}

TYPED_TEST(VecFloatingUnaryTest, PreservesIEEESpecialValueClasses) {
  run_floating_unary_special_values_test<TypeParam>();
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecFloatingUnaryTest, FixedSVEBatchesBeyondTupleLimit) {
  run_fixed_floating_unary_test<TypeParam>();
}
#endif

#endif
