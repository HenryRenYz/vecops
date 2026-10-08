// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// @vecops-test-shards: 12

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "vecops/vec/Arithmetic.h"
#include "vecops/vec/Memory.h"
#include "TestHelpers.h"
#include "TestShard.h"

namespace vec = vecops::vec;

template <int Index>
void run_clamp_case();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == 12);

namespace {

template <typename T>
T clamp_test_value(int value) {
  if constexpr (std::unsigned_integral<T>)
    return static_cast<T>(value + 32);
  else
    return static_cast<T>(value);
}

template <typename T>
T expected_clamp(T value, T lower, T upper) {
  const T bounded_low = value > lower ? value : lower;
  return bounded_low < upper ? bounded_low : upper;
}

template <vec::VectorTag Tag>
void verify_clamp_shape(Tag tag) {
  using T = vec::ElementOf<Tag>;
  const auto lanes = static_cast<std::size_t>(vec::size(tag));
  std::vector<T> values(lanes);
  std::vector<T> lowers(lanes);
  std::vector<T> uppers(lanes);
  std::vector<T> merges(lanes);
  auto mask = vec::mfalse(tag);

  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    constexpr int value_pattern[] = {-9, -4, 1, 6, 9, 2};
    const auto slot = static_cast<std::size_t>(lane % 6);
    values[static_cast<std::size_t>(lane)] =
        clamp_test_value<T>(value_pattern[slot]);
    if (slot == 5) {
      // Bounds are deliberately reversed. clamp is defined as the exact
      // max-then-min composition rather than having an ordered-bound precondition.
      lowers[static_cast<std::size_t>(lane)] = clamp_test_value<T>(5);
      uppers[static_cast<std::size_t>(lane)] = clamp_test_value<T>(-5);
    } else {
      lowers[static_cast<std::size_t>(lane)] = clamp_test_value<T>(-4);
      uppers[static_cast<std::size_t>(lane)] = clamp_test_value<T>(6);
    }
    merges[static_cast<std::size_t>(lane)] =
        clamp_test_value<T>(static_cast<int>(lane % 9) + 11);
    mask = vec::set(tag, mask, lane, (lane % 3) != 1);
  }

  const auto value = vec::load(tag, values.data());
  const auto lower = vec::load(tag, lowers.data());
  const auto upper = vec::load(tag, uppers.data());
  const auto merge = vec::load(tag, merges.data());
  const T scalar_merge = clamp_test_value<T>(13);

  const auto explicit_result = vec::clamp(tag, value, lower, upper);
  const auto inferred_result = vec::clamp(value, lower, upper);
  const auto explicit_unmasked =
      vec::clamp(tag, value, lower, upper, vec::opt::unmasked);

  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const auto index = static_cast<std::size_t>(lane);
    const T expected = expected_clamp(
        values[index], lowers[index], uppers[index]);
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, explicit_result, lane)))
        << "explicit lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, inferred_result, lane)))
        << "inferred lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, explicit_unmasked, lane)))
        << "explicit unmasked lane=" << lane;
  }

  if constexpr (vec_test::exhaustive_options_shape<Tag>) {
    const auto default_masked =
        vec::clamp(value, lower, upper, vec::opt::masked(mask));
    const auto zero_masked = vec::clamp(
        value, lower, upper, vec::opt::masked(mask), vec::opt::zero);
    const auto scalar_masked = vec::clamp(
        value, lower, upper, vec::opt::masked(mask),
        vec::opt::merge(scalar_merge));
    const auto vector_masked = vec::clamp(
        value, lower, upper, vec::opt::masked(mask),
        vec::opt::merge(merge));

    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const auto index = static_cast<std::size_t>(lane);
      const bool active = (lane % 3) != 1;
      const T expected = expected_clamp(
          values[index], lowers[index], uppers[index]);
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : values[index],
          vec::get(tag, default_masked, lane)))
          << "default masked lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : T{}, vec::get(tag, zero_masked, lane)))
          << "zero masked lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : scalar_merge,
          vec::get(tag, scalar_masked, lane)))
          << "scalar merge lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : merges[index],
          vec::get(tag, vector_masked, lane)))
          << "vector merge lane=" << lane;
    }
  }
}

template <typename T>
void verify_floating_special_values() {
  if constexpr (::vecops::is_float_v<T>) {
    using Tag = vec::ScalableTag<T, 0>;
    const Tag tag{};
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T one = static_cast<T>(1);
    const T negative_one = static_cast<T>(-1);
    const T positive_zero = static_cast<T>(0.0F);
    const T negative_zero = static_cast<T>(-0.0F);
    const T infinity = std::numeric_limits<T>::infinity();

    const auto verify = [&](T value, T lower, T upper) {
      const auto value_vec = vec::fill(tag, value);
      const auto lower_vec = vec::fill(tag, lower);
      const auto upper_vec = vec::fill(tag, upper);
      const auto expected = vec::min(
          vec::max(value_vec, lower_vec), upper_vec);
      const auto actual = vec::clamp(value_vec, lower_vec, upper_vec);
      EXPECT_TRUE(vec_test::values_identical(
          vec::get(tag, expected, 0), vec::get(tag, actual, 0)));
    };

    verify(nan, negative_one, one);
    verify(one, nan, infinity);
    verify(one, negative_one, nan);
    verify(positive_zero, negative_zero, one);
    verify(negative_zero, positive_zero, one);
    verify(infinity, negative_one, one);
    verify(-infinity, negative_one, one);
  }
}

template <typename T>
void verify_clamp_type() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_clamp_shape(Tag{});
  });
  verify_floating_special_values<T>();
}

} // namespace

template <int Index>
void run_clamp_case() {
  static_assert(Index >= 0 && Index < 12);
  using T = vec_test::ElementAt<Index>;
  verify_clamp_type<T>();
}

template void run_clamp_case<VECOPS_TEST_SHARD_INDEX>();

#else

TEST(ClampTest, EveryElementTypeAndScalableShape) {
  []<std::size_t... Index>(std::index_sequence<Index...>) {
    (run_clamp_case<static_cast<int>(Index)>(), ...);
  }(std::make_index_sequence<12>{});
}

#endif
