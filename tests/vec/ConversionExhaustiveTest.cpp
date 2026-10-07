// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// @vecops-test-shards: 16

#include <gtest/gtest.h>

#include <limits>
#include <type_traits>
#include <typeinfo>

#include "vecops/vec/Conversion.h"
#include "vecops/util/ScalarConvert.h"
#include "TestHelpers.h"
#include "TestShard.h"

namespace vec = vecops::vec;

template <typename From>
void run_scalable_conversion_exhaustive_test();
template <int SourceBytes>
void run_fixed_conversion_exhaustive_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(
    VECOPS_TEST_SHARD_COUNT == vec_test::AllElements::size + 4);

namespace {

template <typename From, typename To>
From exhaustive_input(vecops::nint_t lane) {
  if constexpr (::vecops::is_float_v<From>) {
    const double magnitude = static_cast<double>(lane % 9) * 0.5;
    const double value = std::unsigned_integral<To>
        ? magnitude : ((lane % 2) == 0 ? magnitude : -magnitude);
    return static_cast<From>(value);
  } else if constexpr (std::signed_integral<From>) {
    if (lane == 0) return std::numeric_limits<From>::lowest();
    if (lane == 1) return std::numeric_limits<From>::max();
    return static_cast<From>((lane % 17) - 8);
  } else {
    if (lane == 0) return std::numeric_limits<From>::max();
    return static_cast<From>(lane * 37 + 3);
  }
}

template <vec::VectorTag FromTag, vec::VectorTag ToTag>
void verify_ordered_exhaustive_case() {
  using From = vec::ElementOf<FromTag>;
  using To = vec::ElementOf<ToTag>;
  ASSERT_EQ(vec::size(FromTag{}), vec::size(ToTag{}));
  SCOPED_TRACE(::testing::Message()
               << "From=" << typeid(From).name()
               << ", To=" << typeid(To).name()
               << ", lanes=" << vec::size(FromTag{})
               << ", from_words=" << vec::num_words(FromTag{})
               << ", to_words=" << vec::num_words(ToTag{}));

  auto source = vec::zeros(FromTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(FromTag{}); ++lane)
    source = vec::set(
        FromTag{}, source, lane, exhaustive_input<From, To>(lane));

  const auto implicit = vec::convert(ToTag{}, FromTag{}, source);
  const auto explicit_policy = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::ordered, vec::cvt::saturate);
  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane) {
    const To expected = ::vecops::convert<To>(
        exhaustive_input<From, To>(lane));
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(ToTag{}, implicit, lane))) << "lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(ToTag{}, explicit_policy, lane)))
        << "explicit lane=" << lane;
  }

  const auto from_word_lanes = vec::native_word_size(FromTag{});
  const auto to_word_lanes = vec::native_word_size(ToTag{});
  for (const auto boundary : {from_word_lanes, to_word_lanes}) {
    if (boundary > 0 && boundary < vec::size(ToTag{})) {
      for (const auto lane : {boundary - 1, boundary}) {
        const To expected = ::vecops::convert<To>(
            exhaustive_input<From, To>(lane));
        EXPECT_TRUE(vec_test::values_identical(
            expected, vec::get(ToTag{}, implicit, lane)))
            << "word-boundary lane=" << lane;
      }
    }
  }
}

template <typename From, typename To>
void verify_scalable_pair() {
  vec_test::for_each_scalable_conversion_shape<From, To>(
      []<vec::VectorTag FromTag, vec::VectorTag ToTag>() {
        verify_ordered_exhaustive_case<FromTag, ToTag>();
      });
}

} // namespace

template <typename From>
void run_scalable_conversion_exhaustive_test() {
  vec_test::for_each_element_type([&]<typename To>() {
    verify_scalable_pair<From, To>();
  });
}

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)

template <typename From, typename To>
void verify_fixed_pair() {
  vec_test::for_each_fixed_conversion_shape<From, To>(
      []<vec::VectorTag FromTag, vec::VectorTag ToTag>() {
        verify_ordered_exhaustive_case<FromTag, ToTag>();
      });
}

template <int SourceBytes>
void run_fixed_conversion_exhaustive_test() {
  if constexpr (SourceBytes == 1)
    verify_fixed_pair<vecops::int8_t, vecops::int64_t>();
  else if constexpr (SourceBytes == 2)
    verify_fixed_pair<vecops::int16_t, vecops::int8_t>();
  else if constexpr (SourceBytes == 4)
    verify_fixed_pair<vecops::int32_t, vecops::int8_t>();
  else if constexpr (SourceBytes == 8)
    verify_fixed_pair<vecops::int64_t, vecops::int8_t>();
}

#endif

#if VECOPS_TEST_SHARD_INDEX < 12
using ShardType = vec_test::ElementAt<VECOPS_TEST_SHARD_INDEX>;
template void run_scalable_conversion_exhaustive_test<ShardType>();
#elif !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
constexpr int shard_source_bytes = 1 << (VECOPS_TEST_SHARD_INDEX - 12);
template void run_fixed_conversion_exhaustive_test<shard_source_bytes>();
#endif

#else

template <typename T>
class VecConversionExhaustiveTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecConversionExhaustiveTest,
    vec_test::AllElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(
    VecConversionExhaustiveTest,
    EveryDestinationAndScalableWordPair) {
  run_scalable_conversion_exhaustive_test<TypeParam>();
}

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
TEST(VecConversionExhaustiveTest, EveryWordCountForWidthRatio) {
  run_fixed_conversion_exhaustive_test<1>();
  run_fixed_conversion_exhaustive_test<2>();
  run_fixed_conversion_exhaustive_test<4>();
  run_fixed_conversion_exhaustive_test<8>();
}
#endif

#endif
