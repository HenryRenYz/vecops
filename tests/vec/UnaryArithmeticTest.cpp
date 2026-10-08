// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// @vecops-test-shards: 12

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "vecops/vec/Arithmetic.h"
#include "TestHelpers.h"
#include "TestShard.h"

namespace vec = vecops::vec;

template <typename T>
void run_unary_arithmetic_test();
template <typename T>
void run_fixed_unary_arithmetic_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == vec_test::AllElements::size);

enum class UnaryKind { Neg, Abs };

template <typename T>
T unary_operand(vecops::nint_t lane) {
  if constexpr (std::same_as<T, vecops::bfloat16_t>) {
    constexpr std::uint16_t bits[] = {0x8000, 0, 0xffc1, 0x3f80, 0xbf80};
    return T::from_bits(bits[lane % 5]);
  } else if constexpr (std::same_as<T, vecops::float16_t>) {
    constexpr std::uint16_t bits[] = {0x8000, 0, 0xfe55, 0x3c00, 0xbc00};
    return T::from_bits(bits[lane % 5]);
  } else if constexpr (std::same_as<T, vecops::float32_t>) {
    constexpr std::uint32_t bits[] = {
        0x80000000u, 0, 0xffc12345u, 0x3f800000u, 0xbf800000u};
    return ::vecops::bitcast<T>(bits[lane % 5]);
  } else if constexpr (std::same_as<T, vecops::float64_t>) {
    constexpr std::uint64_t bits[] = {
        0x8000000000000000ull, 0, 0xfff8123456789abcull,
        0x3ff0000000000000ull, 0xbff0000000000000ull};
    return ::vecops::bitcast<T>(bits[lane % 5]);
  } else if constexpr (std::is_signed_v<T>) {
    if (lane == 0) return std::numeric_limits<T>::lowest();
    return static_cast<T>((lane % 2) == 0 ? lane + 1 : -lane - 1);
  } else {
    return static_cast<T>(lane * 17 + 3);
  }
}

template <UnaryKind Kind, typename T>
T expected_unary(T value) {
  constexpr bool absolute = Kind == UnaryKind::Abs;
  if constexpr (
      std::same_as<T, vecops::bfloat16_t> ||
      std::same_as<T, vecops::float16_t>) {
    const auto bits = value.to_bits();
    return T::from_bits(static_cast<std::uint16_t>(
        absolute ? bits & 0x7fffu : bits ^ 0x8000u));
  } else if constexpr (std::same_as<T, vecops::float32_t>) {
    const auto bits = ::vecops::bitcast<std::uint32_t>(value);
    return ::vecops::bitcast<T>(
        absolute ? bits & 0x7fffffffu : bits ^ 0x80000000u);
  } else if constexpr (std::same_as<T, vecops::float64_t>) {
    const auto bits = ::vecops::bitcast<std::uint64_t>(value);
    return ::vecops::bitcast<T>(absolute
        ? bits & 0x7fffffffffffffffull
        : bits ^ 0x8000000000000000ull);
  } else {
    using U = std::make_unsigned_t<T>;
    if constexpr (std::is_unsigned_v<T>) {
      return absolute ? value : static_cast<T>(U{} - value);
    } else {
      const U bits = ::vecops::bitcast<U>(value);
      const bool negative = (bits >> (sizeof(T) * 8 - 1)) != 0;
      const U result = absolute && !negative
          ? bits
          : static_cast<U>(U{} - bits);
      return ::vecops::bitcast<T>(result);
    }
  }
}

template <UnaryKind Kind, vec::VectorTag Tag, typename... Options>
vec::Vec<Tag> invoke_unary(
    Tag tag, vec::Vec<Tag> value, Options&&... options) {
  if constexpr (Kind == UnaryKind::Neg)
    return vec::neg(value, std::forward<Options>(options)...);
  else
    return vec::abs(value, std::forward<Options>(options)...);
}

template <UnaryKind Kind, bool FullOptions = true, vec::VectorTag Tag>
void verify_unary(Tag tag) {
  using T = vec::ElementOf<Tag>;
  auto value = vec::zeros(tag);
  auto vector_merge = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  const T scalar_merge = unary_operand<T>(23);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    value = vec::set(tag, value, lane, unary_operand<T>(lane));
    vector_merge = vec::set(
        tag, vector_merge, lane, unary_operand<T>(lane + 11));
    mask = vec::set(tag, mask, lane, (lane % 3) != 0);
  }

  const auto result = invoke_unary<Kind>(tag, value);
  const auto unmasked_result =
      invoke_unary<Kind>(tag, value, vec::opt::unmasked);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    EXPECT_TRUE(vec_test::values_identical(
        expected_unary<Kind>(unary_operand<T>(lane)),
        vec::get(tag, result, lane)))
        << "lane=" << lane << ", words=" << vec::num_words(tag);
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, result, lane),
        vec::get(tag, unmasked_result, lane)))
        << "explicit unmasked lane=" << lane;
  }

  if constexpr (FullOptions) {
    const auto masked_default = invoke_unary<Kind>(
        tag, value, vec::opt::masked(mask));
    const auto masked_zero = invoke_unary<Kind>(
        tag, value, vec::opt::masked(mask), vec::opt::zero);
    const auto masked_scalar = invoke_unary<Kind>(
        tag, value, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
    const auto masked_vector = invoke_unary<Kind>(
        tag, value, vec::opt::masked(mask), vec::opt::merge(vector_merge));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T input = unary_operand<T>(lane);
      const T expected = expected_unary<Kind>(input);
      const bool active = (lane % 3) != 0;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : input,
          vec::get(tag, masked_default, lane))) << "default lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : T{},
          vec::get(tag, masked_zero, lane))) << "zero lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : scalar_merge,
          vec::get(tag, masked_scalar, lane))) << "scalar lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : unary_operand<T>(lane + 11),
          vec::get(tag, masked_vector, lane))) << "vector lane=" << lane;
    }
  }
}

template <typename T>
void run_unary_arithmetic_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    constexpr bool options = vec_test::exhaustive_options_shape<Tag>;
    verify_unary<UnaryKind::Neg, options>(Tag{});
    verify_unary<UnaryKind::Abs, options>(Tag{});
  });
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

template <typename T>
void run_fixed_unary_arithmetic_test() {
  if constexpr (
      !std::same_as<T, vecops::float32_t> &&
      !std::same_as<T, vecops::int32_t>) {
    GTEST_SKIP() << "f32 and i32 are the representative >4-word types";
  } else {
    constexpr vecops::nint_t word_lanes =
        FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
    constexpr vecops::nint_t lanes = static_cast<vecops::nint_t>(
        std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
    using Tag = vec::FixedTag<T, lanes>;
    EXPECT_GT(vec::num_words(Tag{}), 4);
    verify_unary<UnaryKind::Neg>(Tag{});
    verify_unary<UnaryKind::Abs, false>(Tag{});
  }
}

#endif

using ShardType = vec_test::ElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_unary_arithmetic_test<ShardType>();
#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
template void run_fixed_unary_arithmetic_test<ShardType>();
#endif

#else

template <typename T>
class VecUnaryArithmeticTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecUnaryArithmeticTest,
    vec_test::AllElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecUnaryArithmeticTest, CoversEveryLaneShapeAndPopulation) {
  run_unary_arithmetic_test<TypeParam>();
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecUnaryArithmeticTest, FixedSVEBatchesBeyondTupleLimit) {
  run_fixed_unary_arithmetic_test<TypeParam>();
}
#endif

#endif
