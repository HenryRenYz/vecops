#include <gtest/gtest.h>

#include <limits>
#include <type_traits>
#include <typeinfo>

#include "vecops/vec/Conversion.h"
#include "vecops/util/ScalarConvert.h"
#include "TestHelpers.h"

#ifndef VECOPS_TEST_SOURCE_BYTES
#error "VECOPS_TEST_SOURCE_BYTES must select one conversion-matrix shard"
#endif

namespace vec = vecops::vec;

namespace {

#if VECOPS_TEST_SOURCE_BYTES == 1
using SourceTypes = ::testing::Types<vecops::int8_t, vecops::uint8_t>;
#elif VECOPS_TEST_SOURCE_BYTES == 2
using SourceTypes = ::testing::Types<
    vecops::bfloat16_t, vecops::float16_t,
    vecops::int16_t, vecops::uint16_t>;
#elif VECOPS_TEST_SOURCE_BYTES == 4
using SourceTypes = ::testing::Types<
    vecops::float32_t, vecops::int32_t, vecops::uint32_t>;
#elif VECOPS_TEST_SOURCE_BYTES == 8
using SourceTypes = ::testing::Types<
    vecops::float64_t, vecops::int64_t, vecops::uint64_t>;
#else
#error "unsupported conversion-matrix source width"
#endif

template <typename From, typename To>
From matrix_input(vecops::nint_t lane) {
  if constexpr (::vecops::IsFloatV<From>) {
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
void verify_ordered_matrix_case() {
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
        FromTag{}, source, lane, matrix_input<From, To>(lane));

  const auto implicit = vec::convert(ToTag{}, FromTag{}, source);
  const auto explicit_policy = vec::convert(
      ToTag{}, FromTag{}, source, vec::cvt::ordered, vec::cvt::saturate);
  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane) {
    const To expected = ::vecops::convert<To>(matrix_input<From, To>(lane));
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
            matrix_input<From, To>(lane));
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
        verify_ordered_matrix_case<FromTag, ToTag>();
      });
}

template <typename T>
class VecConversionMatrixTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecConversionMatrixTest, SourceTypes);

TYPED_TEST(VecConversionMatrixTest, EveryDestinationAndScalableWordPair) {
  using From = TypeParam;
  vec_test::for_each_element_type([&]<typename To>() {
    verify_scalable_pair<From, To>();
  });
}

#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)

template <typename From, typename To>
void verify_fixed_pair() {
  vec_test::for_each_fixed_conversion_shape<From, To>(
      []<vec::VectorTag FromTag, vec::VectorTag ToTag>() {
        verify_ordered_matrix_case<FromTag, ToTag>();
      });
}

TEST(VecConversionFixedMatrixTest, EveryWordCountForWidthRatio) {
#if VECOPS_TEST_SOURCE_BYTES == 1
  verify_fixed_pair<vecops::int8_t, vecops::int64_t>();
#elif VECOPS_TEST_SOURCE_BYTES == 2
  verify_fixed_pair<vecops::int16_t, vecops::int8_t>();
#elif VECOPS_TEST_SOURCE_BYTES == 4
  verify_fixed_pair<vecops::int32_t, vecops::int8_t>();
#elif VECOPS_TEST_SOURCE_BYTES == 8
  verify_fixed_pair<vecops::int64_t, vecops::int8_t>();
#endif
}

#endif

} // namespace
