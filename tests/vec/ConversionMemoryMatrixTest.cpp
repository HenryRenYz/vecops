#include <gtest/gtest.h>

#include <limits>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "vecops/vec/ConversionMemory.h"
#include "vecops/vec/Memory.h"
#include "vecops/util/ScalarConvert.h"
#include "TestHelpers.h"

#ifndef VECOPS_TEST_SOURCE_BYTES
#error "VECOPS_TEST_SOURCE_BYTES must select one conversion-memory shard"
#endif

namespace vec = vecops::vec;

namespace {

#if defined(VECOPS_TEST_SOURCE_ELEM)
using SourceTypes = ::testing::Types<VECOPS_TEST_SOURCE_ELEM>;
#elif VECOPS_TEST_SOURCE_BYTES == 1
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
#error "unsupported conversion-memory source width"
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
void verify_load_case() {
  using From = vec::ElementOf<FromTag>;
  using To = vec::ElementOf<ToTag>;
  const auto lanes = vec::size(FromTag{});
  ASSERT_EQ(lanes, vec::size(ToTag{}));
  SCOPED_TRACE(::testing::Message()
               << "load From=" << typeid(From).name()
               << ", To=" << typeid(To).name()
               << ", lanes=" << lanes
               << ", from_words=" << vec::num_words(FromTag{})
               << ", to_words=" << vec::num_words(ToTag{}));

  std::vector<From> input(static_cast<std::size_t>(lanes));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    input[static_cast<std::size_t>(lane)] = matrix_input<From, To>(lane);

  const auto implicit = vec::load_convert(ToTag{}, input.data());
  const auto explicit_policy = vec::load_convert(
      ToTag{}, input.data(), vec::cvt::ordered, vec::cvt::saturate);
  const auto wrapped = [&] {
    if constexpr (
        std::integral<From> && std::integral<To> &&
        sizeof(To) < sizeof(From)) {
      return vec::load_convert(
          ToTag{}, input.data(), vec::cvt::ordered, vec::cvt::wrap);
    } else {
      return implicit;
    }
  }();
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const To expected = ::vecops::convert<To>(
        input[static_cast<std::size_t>(lane)]);
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(ToTag{}, implicit, lane))) << "lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(ToTag{}, explicit_policy, lane)))
        << "explicit lane=" << lane;
    if constexpr (
        std::integral<From> && std::integral<To> &&
        sizeof(To) < sizeof(From)) {
      EXPECT_EQ(
          ::vecops::wrap_convert<To>(input[static_cast<std::size_t>(lane)]),
          vec::get(ToTag{}, wrapped, lane)) << "wrap lane=" << lane;
    }
  }
}

template <vec::VectorTag FromTag, vec::VectorTag ToTag>
void verify_store_case() {
  using From = vec::ElementOf<FromTag>;
  using To = vec::ElementOf<ToTag>;
  const auto lanes = vec::size(FromTag{});
  ASSERT_EQ(lanes, vec::size(ToTag{}));
  SCOPED_TRACE(::testing::Message()
               << "store From=" << typeid(From).name()
               << ", To=" << typeid(To).name()
               << ", lanes=" << lanes
               << ", from_words=" << vec::num_words(FromTag{})
               << ", to_words=" << vec::num_words(ToTag{}));

  std::vector<From> input(static_cast<std::size_t>(lanes));
  std::vector<To> implicit(static_cast<std::size_t>(lanes));
  std::vector<To> explicit_policy(static_cast<std::size_t>(lanes));
  std::vector<To> packed(static_cast<std::size_t>(lanes));
  std::vector<To> split(static_cast<std::size_t>(lanes));
  std::vector<To> wrapped(static_cast<std::size_t>(lanes));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    input[static_cast<std::size_t>(lane)] = matrix_input<From, To>(lane);

  const auto source = vec::load(FromTag{}, input.data());
  vec::store_convert(FromTag{}, implicit.data(), source);
  vec::store_convert(
      FromTag{}, explicit_policy.data(), source,
      vec::cvt::ordered, vec::cvt::saturate);
  vec::store_convert(
      FromTag{}, packed.data(), source,
      vec::cvt::ordered, vec::cvt::saturate, vec::mem::packed);
  vec::store_convert(
      FromTag{}, split.data(), source,
      vec::cvt::ordered, vec::cvt::saturate, vec::mem::split);
  if constexpr (
      std::integral<From> && std::integral<To> &&
      sizeof(To) < sizeof(From)) {
    vec::store_convert(
        FromTag{}, wrapped.data(), source,
        vec::cvt::ordered, vec::cvt::wrap, vec::mem::packed);
  }
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const To expected = ::vecops::convert<To>(
        input[static_cast<std::size_t>(lane)]);
    EXPECT_TRUE(vec_test::values_identical(
        expected, implicit[static_cast<std::size_t>(lane)]))
        << "lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected, explicit_policy[static_cast<std::size_t>(lane)]))
        << "explicit lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected, packed[static_cast<std::size_t>(lane)]))
        << "packed lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected, split[static_cast<std::size_t>(lane)]))
        << "split lane=" << lane;
    if constexpr (
        std::integral<From> && std::integral<To> &&
        sizeof(To) < sizeof(From)) {
      EXPECT_EQ(
          ::vecops::wrap_convert<To>(input[static_cast<std::size_t>(lane)]),
          wrapped[static_cast<std::size_t>(lane)]) << "wrap lane=" << lane;
    }
  }
}

template <vec::VectorTag FromTag, vec::VectorTag ToTag>
void verify_unordered_round_trip() {
  using From = vec::ElementOf<FromTag>;
  using To = vec::ElementOf<ToTag>;
  const auto lanes = vec::size(FromTag{});
  SCOPED_TRACE(::testing::Message()
               << "unordered From=" << typeid(From).name()
               << ", To=" << typeid(To).name()
               << ", lanes=" << lanes
               << ", from_words=" << vec::num_words(FromTag{})
               << ", to_words=" << vec::num_words(ToTag{}));
  std::vector<From> input(static_cast<std::size_t>(lanes));
  std::vector<From> output(static_cast<std::size_t>(lanes));
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const int small =
        (std::unsigned_integral<From> || std::unsigned_integral<To>)
        ? static_cast<int>(lane % 7) : static_cast<int>(lane % 7) - 3;
    input[static_cast<std::size_t>(lane)] = static_cast<From>(small);
  }
  const auto converted = vec::load_convert(
      ToTag{}, input.data(), vec::cvt::unordered);
  vec::store_convert(
      ToTag{}, output.data(), converted, vec::cvt::unordered);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const From expected = ::vecops::convert<From>(
        ::vecops::convert<To>(input[static_cast<std::size_t>(lane)]));
    EXPECT_TRUE(vec_test::values_identical(
        expected, output[static_cast<std::size_t>(lane)]))
        << "unordered lane=" << lane;
  }
}

template <typename From, typename To>
void verify_scalable_pair() {
  vec_test::for_each_scalable_conversion_shape<From, To>(
      []<vec::VectorTag FromTag, vec::VectorTag ToTag>() {
        verify_load_case<FromTag, ToTag>();
        verify_store_case<FromTag, ToTag>();
        verify_unordered_round_trip<FromTag, ToTag>();
      });
}

template <typename T>
class VecConversionMemoryMatrixTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecConversionMemoryMatrixTest, SourceTypes);

TYPED_TEST(
    VecConversionMemoryMatrixTest,
    IndependentLoadAndStoreEveryDestinationAndScalableWordPair) {
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
        verify_load_case<FromTag, ToTag>();
        verify_store_case<FromTag, ToTag>();
        verify_unordered_round_trip<FromTag, ToTag>();
      });
}

TEST(VecConversionMemoryFixedMatrixTest, EveryWordCountForWidthRatio) {
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
