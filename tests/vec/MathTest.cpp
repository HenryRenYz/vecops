#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "vecops/vec/Math.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

namespace {

enum class ExpTestTier { Strict, Fast, Estimate };

using FloatingTypes = ::testing::Types<
    vecops::bfloat16_t,
    vecops::float16_t,
    vecops::float32_t,
    vecops::float64_t>;

template <typename T>
double as_double(T value) {
  return static_cast<double>(value);
}

template <typename T>
std::uint64_t bits(T value) {
  if constexpr (std::same_as<T, float>) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  } else if constexpr (std::same_as<T, double>) {
    std::uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  } else {
    return value.to_bits();
  }
}

template <typename T>
T reference_exp(T value) {
  if constexpr (
      std::same_as<T, vecops::float16_t> ||
      std::same_as<T, vecops::bfloat16_t>) {
    return T(static_cast<float>(std::exp(
        static_cast<long double>(static_cast<float>(value)))));
  } else {
    return static_cast<T>(std::exp(static_cast<long double>(value)));
  }
}

template <ExpTestTier Tier, bool NegativeOnly, vec::FloatingTag Tag,
          typename... Options>
vec::Vec<Tag> invoke_exp(
    Tag tag, vec::Vec<Tag> value, Options&&... options) {
  if constexpr (NegativeOnly) {
    if constexpr (Tier == ExpTestTier::Strict)
      return vec::exp_neg(value, std::forward<Options>(options)...);
    else if constexpr (Tier == ExpTestTier::Fast)
      return vec::exp_neg_fast(value, std::forward<Options>(options)...);
    else
      return vec::exp_neg_est(value, std::forward<Options>(options)...);
  } else {
    if constexpr (Tier == ExpTestTier::Strict)
      return vec::exp(value, std::forward<Options>(options)...);
    else if constexpr (Tier == ExpTestTier::Fast)
      return vec::exp_fast(value, std::forward<Options>(options)...);
    else
      return vec::exp_est(value, std::forward<Options>(options)...);
  }
}

template <ExpTestTier Tier, typename T>
void expect_accurate(T input, T actual) {
  const T expected = reference_exp(input);
  ASSERT_EQ(std::isnan(as_double(expected)), std::isnan(as_double(actual)))
      << "x=" << as_double(input);
  if (std::isnan(as_double(expected))) return;
#ifndef VECOPS_PRESERVE_SUBNORMALS
  if (as_double(expected) > 0.0 &&
      as_double(expected) < as_double(std::numeric_limits<T>::min()) &&
      as_double(actual) == 0.0) {
    return;
  }
#endif
  const auto expected_bits = bits(expected);
  const auto actual_bits = bits(actual);
  const auto ulps = expected_bits > actual_bits
      ? expected_bits - actual_bits : actual_bits - expected_bits;
  if constexpr (Tier == ExpTestTier::Strict) {
    EXPECT_LE(ulps, 1u)
        << "x=" << as_double(input) << " expected=" << as_double(expected)
        << " actual=" << as_double(actual);
  } else if constexpr (Tier == ExpTestTier::Fast) {
    if (std::isfinite(as_double(expected)) &&
        as_double(expected) >= as_double(std::numeric_limits<T>::min()))
      EXPECT_LE(ulps, 4u) << "x=" << as_double(input);
  } else if (std::isfinite(as_double(expected)) &&
             as_double(expected) >= as_double(std::numeric_limits<T>::min())) {
    const long double exact = std::exp(
        static_cast<long double>(as_double(input)));
    const long double relative = std::abs(
        static_cast<long double>(as_double(actual)) / exact - 1.0L);
    EXPECT_TRUE(ulps <= 4u || relative <= 0.006L)
        << "x=" << as_double(input) << " ulp=" << ulps
        << " relative=" << static_cast<double>(relative);
  }
}

template <typename T>
constexpr double lower_bound() {
  if constexpr (std::same_as<T, double>) return -700.0;
  else if constexpr (std::same_as<T, float>) return -80.0;
  else return -9.0;
}

template <typename T>
constexpr double upper_bound() {
  if constexpr (std::same_as<T, double>) return 700.0;
  else if constexpr (std::same_as<T, float>) return 80.0;
  else return 9.0;
}

template <ExpTestTier Tier, bool NegativeOnly, bool FullOptions = true,
          vec::FloatingTag Tag>
void verify_lane_shape(Tag tag) {
  using T = vec::ElementOf<Tag>;
  auto input = vec::zeros(tag);
  auto vector_merge = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const bool active = lane % 3 != 0;
    const double valid = NegativeOnly
        ? -0.125 * static_cast<double>(lane % 9)
        : -1.5 + 0.25 * static_cast<double>(lane % 13);
    const T value = active
        ? T(valid)
        : (NegativeOnly ? T(4.0) : std::numeric_limits<T>::quiet_NaN());
    input = vec::set(tag, input, lane, value);
    vector_merge = vec::set(tag, vector_merge, lane, T(17.0 + lane));
    mask = vec::set(tag, mask, lane, active);
  }

  auto valid_input = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const double value = NegativeOnly
        ? -0.125 * static_cast<double>(lane % 9)
        : -1.5 + 0.25 * static_cast<double>(lane % 13);
    valid_input = vec::set(tag, valid_input, lane, T(value));
  }
  const auto result = invoke_exp<Tier, NegativeOnly>(tag, valid_input);
  const auto unmasked_result = invoke_exp<Tier, NegativeOnly>(
      tag, valid_input, vec::opt::unmasked);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    expect_accurate<Tier>(
        vec::get(tag, valid_input, lane), vec::get(tag, result, lane));
    EXPECT_EQ(
        bits(vec::get(tag, result, lane)),
        bits(vec::get(tag, unmasked_result, lane)))
        << "explicit unmasked lane=" << lane;
  }

  if constexpr (!FullOptions) return;
  const T scalar_merge = T(-7.0);
  const auto preserved = invoke_exp<Tier, NegativeOnly>(
      tag, input, vec::opt::masked(mask));
  const auto zeroed = invoke_exp<Tier, NegativeOnly>(
      tag, input, vec::opt::zero, vec::opt::masked(mask));
  const auto scalar = invoke_exp<Tier, NegativeOnly>(
      tag, input, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
  const auto vector = invoke_exp<Tier, NegativeOnly>(
      tag, input, vec::opt::merge(vector_merge), vec::opt::masked(mask));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const bool active = lane % 3 != 0;
    if (active) {
      const T expected_input = vec::get(tag, input, lane);
      expect_accurate<Tier>(expected_input, vec::get(tag, preserved, lane));
      expect_accurate<Tier>(expected_input, vec::get(tag, zeroed, lane));
      expect_accurate<Tier>(expected_input, vec::get(tag, scalar, lane));
      expect_accurate<Tier>(expected_input, vec::get(tag, vector, lane));
    } else {
      EXPECT_EQ(bits(vec::get(tag, input, lane)), bits(vec::get(tag, preserved, lane)));
      EXPECT_EQ(bits(T{}), bits(vec::get(tag, zeroed, lane)));
      EXPECT_EQ(bits(scalar_merge), bits(vec::get(tag, scalar, lane)));
      EXPECT_EQ(bits(vec::get(tag, vector_merge, lane)), bits(vec::get(tag, vector, lane)));
    }
  }
}

template <typename T, bool FullOptions = true, vec::FloatingTag Tag>
void verify_every_operation(Tag tag) {
  verify_lane_shape<ExpTestTier::Strict, false, FullOptions>(tag);
  verify_lane_shape<ExpTestTier::Fast, false, FullOptions>(tag);
  verify_lane_shape<ExpTestTier::Estimate, false, FullOptions>(tag);
  verify_lane_shape<ExpTestTier::Strict, true, FullOptions>(tag);
  verify_lane_shape<ExpTestTier::Fast, true, FullOptions>(tag);
  verify_lane_shape<ExpTestTier::Estimate, true, FullOptions>(tag);
}

template <ExpTestTier Tier, bool NegativeOnly, typename T>
void verify_dense_samples() {
  vec::ScalableTag<T, 0> tag;
  constexpr int sample_count = 4096;
  for (int base = 0; base <= sample_count; base += static_cast<int>(vec::size(tag))) {
    auto input = vec::zeros(tag);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const int index = std::min(base + static_cast<int>(lane), sample_count);
      const double lower = lower_bound<T>();
      const double upper = NegativeOnly ? 0.0 : upper_bound<T>();
      input = vec::set(tag, input, lane, T(
          lower + (upper - lower) * index / sample_count));
    }
    const auto result = invoke_exp<Tier, NegativeOnly>(tag, input);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
      expect_accurate<Tier>(
          vec::get(tag, input, lane), vec::get(tag, result, lane));
  }
}

template <typename T>
class VecMathTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecMathTest, FloatingTypes);

TYPED_TEST(VecMathTest, EveryOperationShapeAndOptions) {
  using T = TypeParam;
  vec_test::for_each_scalable_shape<T>([]<vec::FloatingTag Tag>() {
    constexpr bool options = vec_test::exhaustive_options_shape<Tag> &&
        (vec::scale_power<Tag> != vec_test::details::maximum_scalable_power ||
         std::same_as<T, vecops::float32_t>);
    verify_every_operation<T, options>(Tag{});
  });
}

TYPED_TEST(VecMathTest, DenseAccuracyAndNegativeDomain) {
  using T = TypeParam;
  verify_dense_samples<ExpTestTier::Strict, false, T>();
  verify_dense_samples<ExpTestTier::Fast, false, T>();
  verify_dense_samples<ExpTestTier::Estimate, false, T>();
  verify_dense_samples<ExpTestTier::Strict, true, T>();
  verify_dense_samples<ExpTestTier::Fast, true, T>();
  verify_dense_samples<ExpTestTier::Estimate, true, T>();
}

#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
TYPED_TEST(VecMathTest, IeeeEdges) {
  using T = TypeParam;
  vec::ScalableTag<T, 0> tag;
  auto input = vec::zeros(tag);
  constexpr double values[] = {0.0, -0.0, 1.0, -1.0};
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    T value = T(values[lane % 4]);
    if (lane == 2) value = std::numeric_limits<T>::infinity();
    if (lane == 3) value = -std::numeric_limits<T>::infinity();
    if (lane == 4) value = std::numeric_limits<T>::quiet_NaN();
    input = vec::set(tag, input, lane, value);
  }
  const auto result = vec::exp(input);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const double in = as_double(vec::get(tag, input, lane));
    const double out = as_double(vec::get(tag, result, lane));
    if (std::isnan(in)) EXPECT_TRUE(std::isnan(out));
    else if (in == std::numeric_limits<double>::infinity()) EXPECT_TRUE(std::isinf(out));
    else if (in == -std::numeric_limits<double>::infinity()) EXPECT_EQ(out, 0.0);
    else expect_accurate<ExpTestTier::Strict>(
        vec::get(tag, input, lane), vec::get(tag, result, lane));
  }
}
#endif

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecMathTest, FixedSVEBatchesBeyondTupleLimit) {
  using T = TypeParam;
  if constexpr (!std::same_as<T, vecops::float32_t>) {
    GTEST_SKIP() << "f32 is the representative >4-word floating type";
  } else {
    constexpr vecops::nint_t word_lanes =
        FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
    constexpr vecops::nint_t lanes = static_cast<vecops::nint_t>(
        std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
    using Tag = vec::FixedTag<T, lanes>;
    EXPECT_GT(vec::num_words(Tag{}), 4);
    verify_every_operation<T>(Tag{}, false);
  }
}
#endif

} // namespace
