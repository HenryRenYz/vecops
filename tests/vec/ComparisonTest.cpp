// @vecops-test-shards: 12

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "vecops/vec/Comparison.h"
#include "TestHelpers.h"
#include "TestShard.h"

namespace vec = vecops::vec;

template <typename T>
void run_comparisons_test();
template <typename T>
void run_classification_test();
template <typename T>
void run_fixed_comparisons_test();
template <typename T>
void run_fixed_classification_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == vec_test::AllElements::size);

template <typename T>
T comparison_value(vecops::nint_t lane, bool rhs) {
  const int pattern = static_cast<int>(lane % 6);
  if constexpr (::vecops::is_float_v<T>) {
    if (pattern == 5) return std::numeric_limits<T>::quiet_NaN();
    if (pattern == 1) return static_cast<T>(rhs ? -0.0F : 0.0F);
    const float base = static_cast<float>((lane * 7) % 19) - 8.0F;
    if (pattern < 2) return static_cast<T>(base);
    if (pattern < 4) return static_cast<T>(base + (rhs ? 1.0F : 0.0F));
    return static_cast<T>(base + (rhs ? -1.0F : 0.0F));
  } else {
    using U = std::make_unsigned_t<T>;
    const U base = static_cast<U>(lane * 17 + 5);
    const U value = pattern < 2
        ? base
        : pattern < 4
            ? static_cast<U>(base + (rhs ? 1 : 0))
            : static_cast<U>(base - (rhs ? 1 : 0));
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(value);
    else return value;
  }
}

template <vec::VectorTag Tag>
void verify_comparisons(Tag tag) {
  using T = vec::ElementOf<Tag>;
  auto a = vec::zeros(tag);
  auto b = vec::zeros(tag);
  auto input_mask = vec::mfalse(tag);
  auto merge_mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    a = vec::set(tag, a, lane, comparison_value<T>(lane, false));
    b = vec::set(tag, b, lane, comparison_value<T>(lane, true));
    input_mask = vec::set(tag, input_mask, lane, (lane % 3) != 1);
    merge_mask = vec::set(tag, merge_mask, lane, (lane % 4) < 2);
  }

  const auto eq = vec::cmpeq(a, b);
  const auto ne = vec::cmpne(a, b);
  const auto lt = vec::cmplt(a, b);
  const auto gt = vec::cmpgt(a, b);
  const auto le = vec::cmple(a, b);
  const auto ge = vec::cmpge(a, b);
  const auto unmasked_eq = vec::cmpeq(a, b, vec::opt::unmasked);
  const auto unmasked_ne = vec::cmpne(a, b, vec::opt::unmasked);
  const auto unmasked_lt = vec::cmplt(a, b, vec::opt::unmasked);
  const auto unmasked_gt = vec::cmpgt(a, b, vec::opt::unmasked);
  const auto unmasked_le = vec::cmple(a, b, vec::opt::unmasked);
  const auto unmasked_ge = vec::cmpge(a, b, vec::opt::unmasked);
  const auto masked_eq = vec::cmpeq(a, b, vec::opt::masked(input_mask));
  const auto masked_ne = vec::cmpne(
      a, b, vec::opt::masked(input_mask), vec::opt::zero);
  const auto masked_lt = vec::cmplt(a, b, vec::opt::masked(input_mask));
  const auto masked_gt = vec::cmpgt(
      a, b, vec::opt::masked(input_mask), vec::opt::zero);
  const auto masked_le = vec::cmple(a, b, vec::opt::masked(input_mask));
  const auto masked_ge = vec::cmpge(
      a, b, vec::opt::masked(input_mask), vec::opt::zero);
  const auto merged_eq = vec::cmpeq(
      a,
      b,
      vec::opt::masked(input_mask),
      vec::opt::merge(merge_mask));

  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T x = comparison_value<T>(lane, false);
    const T y = comparison_value<T>(lane, true);
    const bool active = (lane % 3) != 1;
    EXPECT_EQ(x == y, vec::get(tag, eq, lane)) << "lane=" << lane;
    EXPECT_EQ(x != y, vec::get(tag, ne, lane)) << "lane=" << lane;
    EXPECT_EQ(x < y, vec::get(tag, lt, lane)) << "lane=" << lane;
    EXPECT_EQ(x > y, vec::get(tag, gt, lane)) << "lane=" << lane;
    EXPECT_EQ(x <= y, vec::get(tag, le, lane)) << "lane=" << lane;
    EXPECT_EQ(x >= y, vec::get(tag, ge, lane)) << "lane=" << lane;
    EXPECT_EQ(vec::get(tag, eq, lane), vec::get(tag, unmasked_eq, lane));
    EXPECT_EQ(vec::get(tag, ne, lane), vec::get(tag, unmasked_ne, lane));
    EXPECT_EQ(vec::get(tag, lt, lane), vec::get(tag, unmasked_lt, lane));
    EXPECT_EQ(vec::get(tag, gt, lane), vec::get(tag, unmasked_gt, lane));
    EXPECT_EQ(vec::get(tag, le, lane), vec::get(tag, unmasked_le, lane));
    EXPECT_EQ(vec::get(tag, ge, lane), vec::get(tag, unmasked_ge, lane));
    EXPECT_EQ(active && x == y, vec::get(tag, masked_eq, lane));
    EXPECT_EQ(active && x != y, vec::get(tag, masked_ne, lane));
    EXPECT_EQ(active && x < y, vec::get(tag, masked_lt, lane));
    EXPECT_EQ(active && x > y, vec::get(tag, masked_gt, lane));
    EXPECT_EQ(active && x <= y, vec::get(tag, masked_le, lane));
    EXPECT_EQ(active && x >= y, vec::get(tag, masked_ge, lane));
    EXPECT_EQ(
        active ? x == y : (lane % 4) < 2,
        vec::get(tag, merged_eq, lane));
  }
}

template <typename T>
void run_comparisons_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_comparisons(Tag{});
  });
}

template <typename T>
T classification_value(vecops::nint_t lane) {
  switch (lane % 7) {
    case 0: return std::numeric_limits<T>::quiet_NaN();
    case 1: return std::numeric_limits<T>::infinity();
    case 2: return -std::numeric_limits<T>::infinity();
    case 3:
      if constexpr (std::same_as<T, vecops::float64_t>)
        return std::numeric_limits<T>::max();
      else
        return std::numeric_limits<T>::max();
    case 4: return static_cast<T>(-17.25);
    case 5: return static_cast<T>(0.0);
    default: return static_cast<T>(42.5);
  }
}

template <vec::VectorTag Tag>
void verify_classification(Tag tag) {
  using T = vec::ElementOf<Tag>;
  auto value = vec::zeros(tag);
  auto input_mask = vec::mfalse(tag);
  auto merge_mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    value = vec::set(tag, value, lane, classification_value<T>(lane));
    input_mask = vec::set(tag, input_mask, lane, (lane % 2) == 0);
    merge_mask = vec::set(tag, merge_mask, lane, (lane % 3) == 0);
  }

  const auto nan = vec::isnan(value);
  const auto pos = vec::isposinf(value);
  const auto neg = vec::isneginf(value);
  const auto inf = vec::isinf(value);
  const auto unmasked_nan = vec::isnan(value, vec::opt::unmasked);
  const auto unmasked_pos = vec::isposinf(value, vec::opt::unmasked);
  const auto unmasked_neg = vec::isneginf(value, vec::opt::unmasked);
  const auto unmasked_inf = vec::isinf(value, vec::opt::unmasked);
  const auto masked_nan = vec::isnan(
      value, vec::opt::masked(input_mask));
  const auto masked_pos = vec::isposinf(
      value, vec::opt::masked(input_mask), vec::opt::zero);
  const auto masked_neg = vec::isneginf(
      value, vec::opt::masked(input_mask));
  const auto masked_inf = vec::isinf(
      value, vec::opt::masked(input_mask), vec::opt::zero);
  const auto merged_nan = vec::isnan(
      value,
      vec::opt::masked(input_mask),
      vec::opt::merge(merge_mask));

  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T x = classification_value<T>(lane);
    const double widened = static_cast<double>(x);
    const bool expected_nan = std::isnan(widened);
    const bool expected_pos = std::isinf(widened) && !std::signbit(widened);
    const bool expected_neg = std::isinf(widened) && std::signbit(widened);
    const bool expected_inf = std::isinf(widened);
    const bool active = (lane % 2) == 0;
    EXPECT_EQ(expected_nan, vec::get(tag, nan, lane)) << "lane=" << lane;
    EXPECT_EQ(expected_pos, vec::get(tag, pos, lane)) << "lane=" << lane;
    EXPECT_EQ(expected_neg, vec::get(tag, neg, lane)) << "lane=" << lane;
    EXPECT_EQ(expected_inf, vec::get(tag, inf, lane)) << "lane=" << lane;
    EXPECT_EQ(vec::get(tag, nan, lane), vec::get(tag, unmasked_nan, lane));
    EXPECT_EQ(vec::get(tag, pos, lane), vec::get(tag, unmasked_pos, lane));
    EXPECT_EQ(vec::get(tag, neg, lane), vec::get(tag, unmasked_neg, lane));
    EXPECT_EQ(vec::get(tag, inf, lane), vec::get(tag, unmasked_inf, lane));
    EXPECT_EQ(active && expected_nan, vec::get(tag, masked_nan, lane));
    EXPECT_EQ(active && expected_pos, vec::get(tag, masked_pos, lane));
    EXPECT_EQ(active && expected_neg, vec::get(tag, masked_neg, lane));
    EXPECT_EQ(active && expected_inf, vec::get(tag, masked_inf, lane));
    EXPECT_EQ(
        active ? expected_nan : (lane % 3) == 0,
        vec::get(tag, merged_nan, lane));
  }
}

template <typename T>
void run_classification_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_classification(Tag{});
  });
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

template <typename T>
void run_fixed_comparisons_test() {
  constexpr vecops::nint_t word_lanes =
      FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
  constexpr vecops::nint_t lanes = static_cast<vecops::nint_t>(
      std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
  using Tag = vec::FixedTag<T, lanes>;
  EXPECT_GT(vec::num_words(Tag{}), 4);
  verify_comparisons(Tag{});
}

template <typename T>
void run_fixed_classification_test() {
  constexpr vecops::nint_t word_lanes =
      FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
  constexpr vecops::nint_t lanes = static_cast<vecops::nint_t>(
      std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
  using Tag = vec::FixedTag<T, lanes>;
  EXPECT_GT(vec::num_words(Tag{}), 4);
  verify_classification(Tag{});
}

#endif

using ShardType = vec_test::ElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_comparisons_test<ShardType>();
#if VECOPS_TEST_SHARD_INDEX < 4
template void run_classification_test<ShardType>();
#endif
#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
template void run_fixed_comparisons_test<ShardType>();
#if VECOPS_TEST_SHARD_INDEX < 4
template void run_fixed_classification_test<ShardType>();
#endif
#endif

#else

template <typename T>
class VecComparisonElementTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecComparisonElementTest,
    vec_test::AllElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecComparisonElementTest, AllComparisonsCoverEveryLaneAndShape) {
  run_comparisons_test<TypeParam>();
}

template <typename T>
class VecClassificationTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecClassificationTest,
    vec_test::FloatingElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecClassificationTest, SpecialValuesCoverEveryLaneAndShape) {
  run_classification_test<TypeParam>();
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecComparisonElementTest, FixedSVEBatchesBeyondFourWords) {
  run_fixed_comparisons_test<TypeParam>();
}

TYPED_TEST(VecClassificationTest, FixedSVEBatchesBeyondFourWords) {
  run_fixed_classification_test<TypeParam>();
}
#endif

#endif
