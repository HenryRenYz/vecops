// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// @vecops-test-shards: 8

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <type_traits>
#include <utility>

#include "TestHelpers.h"
#include "TestShard.h"
#include "vecops/vec/Arithmetic.h"

namespace vec = vecops::vec;

template <typename T>
void run_saturating_arithmetic_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == vec_test::IntegerElements::size);

template <typename T>
constexpr T reference_saturating_add(T a, T b) {
  if constexpr (std::is_signed_v<T>) {
    const __int128 result =
        static_cast<__int128>(a) + static_cast<__int128>(b);
    if (result > static_cast<__int128>(std::numeric_limits<T>::max()))
      return std::numeric_limits<T>::max();
    if (result < static_cast<__int128>(std::numeric_limits<T>::min()))
      return std::numeric_limits<T>::min();
    return static_cast<T>(result);
  } else {
    const unsigned __int128 result =
        static_cast<unsigned __int128>(a) +
        static_cast<unsigned __int128>(b);
    return result > static_cast<unsigned __int128>(
                        std::numeric_limits<T>::max())
        ? std::numeric_limits<T>::max()
        : static_cast<T>(result);
  }
}

template <typename T>
constexpr T reference_saturating_sub(T a, T b) {
  if constexpr (std::is_signed_v<T>) {
    const __int128 result =
        static_cast<__int128>(a) - static_cast<__int128>(b);
    if (result > static_cast<__int128>(std::numeric_limits<T>::max()))
      return std::numeric_limits<T>::max();
    if (result < static_cast<__int128>(std::numeric_limits<T>::min()))
      return std::numeric_limits<T>::min();
    return static_cast<T>(result);
  } else {
    return a < b ? T{0} : static_cast<T>(a - b);
  }
}

template <typename T>
constexpr T reference_wrapping_add(T a, T b) {
  using U = std::make_unsigned_t<T>;
  const U ua = [&] {
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(a);
    else return a;
  }();
  const U ub = [&] {
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(b);
    else return b;
  }();
  const U result = static_cast<U>(ua + ub);
  if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(result);
  else return result;
}

template <typename T>
constexpr auto saturating_cases() {
  constexpr T lo = std::numeric_limits<T>::min();
  constexpr T hi = std::numeric_limits<T>::max();
  if constexpr (std::is_signed_v<T>) {
    return std::array<std::pair<T, T>, 16>{
        std::pair{hi, T{1}}, std::pair{hi, hi},
        std::pair{lo, T{-1}}, std::pair{lo, lo},
        std::pair{hi, T{-1}}, std::pair{lo, T{1}},
        std::pair{T{0}, lo}, std::pair{T{0}, hi},
        std::pair{T{-1}, hi}, std::pair{T{1}, lo},
        std::pair{T{42}, T{19}}, std::pair{T{-42}, T{19}},
        std::pair{T{42}, T{-19}}, std::pair{T{-42}, T{-19}},
        std::pair{T{0}, T{0}}, std::pair{T{1}, T{-1}}};
  } else {
    return std::array<std::pair<T, T>, 16>{
        std::pair{hi, T{1}}, std::pair{hi, hi},
        std::pair{lo, T{1}}, std::pair{lo, hi},
        std::pair{hi, T{0}}, std::pair{T{1}, hi},
        std::pair{T{1}, T{1}}, std::pair{T{42}, T{19}},
        std::pair{T{19}, T{42}}, std::pair{T{0}, T{0}},
        std::pair{static_cast<T>(hi - 1), T{1}},
        std::pair{static_cast<T>(hi - 1), T{2}},
        std::pair{T{2}, T{1}}, std::pair{T{1}, T{2}},
        std::pair{static_cast<T>(hi / 2), static_cast<T>(hi / 2)},
        std::pair{static_cast<T>(hi / 2 + 1),
                  static_cast<T>(hi / 2 + 1)}};
  }
}

template <vec::VectorTag Tag>
void verify_saturating_arithmetic(Tag tag) {
  using T = vec::ElementOf<Tag>;
  constexpr auto cases = saturating_cases<T>();
  constexpr T scalar_merge = T{23};
  constexpr T vector_merge_value = T{17};
  const auto vector_merge = vec::fill(tag, vector_merge_value);

  for (std::size_t base = 0; base < cases.size();
       base += static_cast<std::size_t>(vec::size(tag))) {
    auto a = vec::zeros(tag);
    auto b = vec::zeros(tag);
    auto mask = vec::mfalse(tag);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const auto index =
          (base + static_cast<std::size_t>(lane)) % cases.size();
      a = vec::set(tag, a, lane, cases[index].first);
      b = vec::set(tag, b, lane, cases[index].second);
      mask = vec::set(tag, mask, lane, (index % 2) == 0);
    }

    const auto add = vec::add(tag, a, b, vec::opt::saturate);
    const auto inferred_add = vec::add(a, b, vec::opt::saturate);
    const auto sub = vec::sub(tag, a, b, vec::opt::saturate);
    const auto explicit_wrap = vec::add(tag, a, b, vec::opt::wrap);
    const auto implicit_wrap = vec::add(tag, a, b);
    const auto masked_add = vec::add(
        tag, a, b, vec::opt::masked(mask), vec::opt::saturate);
    const auto zeroed_sub = vec::sub(
        tag, a, b, vec::opt::saturate, vec::opt::masked(mask),
        vec::opt::zero);
    const auto scalar_merged_add = vec::add(
        tag, a, b, vec::opt::saturate, vec::opt::masked(mask),
        vec::opt::merge(scalar_merge));
    const auto vector_merged_sub = vec::sub(
        tag, a, b, vec::opt::masked(mask),
        vec::opt::merge(vector_merge), vec::opt::saturate);

    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const auto index =
          (base + static_cast<std::size_t>(lane)) % cases.size();
      const auto [lhs, rhs] = cases[index];
      const T expected_add = reference_saturating_add(lhs, rhs);
      const T expected_sub = reference_saturating_sub(lhs, rhs);
      const bool active = (index % 2) == 0;
      EXPECT_EQ(expected_add, vec::get(tag, add, lane));
      EXPECT_EQ(expected_add, vec::get(tag, inferred_add, lane));
      EXPECT_EQ(expected_sub, vec::get(tag, sub, lane));
      EXPECT_EQ(
          reference_wrapping_add(lhs, rhs),
          vec::get(tag, explicit_wrap, lane));
      EXPECT_EQ(
          vec::get(tag, implicit_wrap, lane),
          vec::get(tag, explicit_wrap, lane));
      EXPECT_EQ(
          active ? expected_add : lhs,
          vec::get(tag, masked_add, lane));
      EXPECT_EQ(
          active ? expected_sub : T{0},
          vec::get(tag, zeroed_sub, lane));
      EXPECT_EQ(
          active ? expected_add : scalar_merge,
          vec::get(tag, scalar_merged_add, lane));
      EXPECT_EQ(
          active ? expected_sub : vector_merge_value,
          vec::get(tag, vector_merged_sub, lane));
    }
  }
}

template <typename T>
void run_saturating_arithmetic_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_saturating_arithmetic(Tag{});
  });
}

using ShardType = vec_test::IntegerElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_saturating_arithmetic_test<ShardType>();

#else

template <typename T>
class VecSaturatingArithmeticTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecSaturatingArithmeticTest,
    vec_test::IntegerElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecSaturatingArithmeticTest, BoundariesMasksAndEveryShape) {
  run_saturating_arithmetic_test<TypeParam>();
}

#endif
