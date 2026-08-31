// @vecops-test-shards: 72

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
#include "TestShard.h"

namespace vec = vecops::vec;

constexpr int kMathOperationCount = 18;
constexpr long double kLog2Base10Long = 3.321928094887362347870319429489L;

enum class MathOperation {
  ExpStrict,
  ExpFast,
  ExpEstimate,
  ExpNegStrict,
  ExpNegFast,
  ExpNegEstimate,
  Exp2Strict,
  Exp2Fast,
  Exp2Estimate,
  Exp2NegStrict,
  Exp2NegFast,
  Exp2NegEstimate,
  Exp10Strict,
  Exp10Fast,
  Exp10Estimate,
  Exp10NegStrict,
  Exp10NegFast,
  Exp10NegEstimate,
};

constexpr vec::ExpBase operation_base(MathOperation op) {
  return static_cast<vec::ExpBase>(static_cast<int>(op) / 6);
}

constexpr vec::Accuracy operation_tier(MathOperation op) {
  constexpr vec::Accuracy tiers[] = {
      vec::Accuracy::Strict, vec::Accuracy::Fast, vec::Accuracy::Estimate};
  return tiers[static_cast<int>(op) % 3];
}

constexpr bool operation_negative(MathOperation op) {
  return static_cast<int>(op) % 6 >= 3;
}

template <MathOperation Operation, typename T>
void run_math_shapes_test();
template <MathOperation Operation, typename T>
void run_math_dense_test();
template <typename T>
void run_math_ieee_edges_test();
template <MathOperation Operation, typename T>
void run_fixed_sve_math_test();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(
    VECOPS_TEST_SHARD_COUNT ==
    vec_test::FloatingElements::size * kMathOperationCount);

namespace {

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

template <vec::ExpBase Base, typename T>
T reference_exp(T value) {
  if constexpr (
      std::same_as<T, vecops::float16_t> ||
      std::same_as<T, vecops::bfloat16_t>) {
    const long double widened = static_cast<float>(value);
    if constexpr (Base == vec::ExpBase::E)
      return T(static_cast<float>(std::exp(widened)));
    else if constexpr (Base == vec::ExpBase::Base2)
      return T(static_cast<float>(std::exp2(widened)));
    else
      return T(static_cast<float>(
          std::exp2(widened * kLog2Base10Long)));
  } else {
    const long double widened = static_cast<long double>(value);
    if constexpr (Base == vec::ExpBase::E)
      return static_cast<T>(std::exp(widened));
    else if constexpr (Base == vec::ExpBase::Base2)
      return static_cast<T>(std::exp2(widened));
    else
      return static_cast<T>(std::exp2(widened * kLog2Base10Long));
  }
}

template <vec::Accuracy Tier, vec::ExpBase Base, bool NegativeOnly,
          vec::FloatingTag Tag, typename... Options>
vec::Vec<Tag> invoke_exp(
    Tag tag, vec::Vec<Tag> value, Options&&... options) {
  return vec::ExpCpo<Base, NegativeOnly>{}(
      tag,
      value,
      std::forward<Options>(options)...,
      vec::opt::math::accuracy<Tier>);
}

template <vec::Accuracy Tier, vec::ExpBase Base, bool NegativeOnly,
          vec::FloatingTag Tag>
vec::Vec<Tag> invoke_fixed_exp(Tag tag, vec::Vec<Tag> value) {
  return vec::FixedAccuracyExpCpo<Base, Tier, NegativeOnly>{}(tag, value);
}

/** Full-precision reference used for the Estimate relative-error budget. */
template <vec::ExpBase Base, typename T>
long double reference_exp_exact(T value) {
  const long double widened = [&] {
    if constexpr (
        std::same_as<T, vecops::float16_t> ||
        std::same_as<T, vecops::bfloat16_t>)
      return static_cast<long double>(static_cast<float>(value));
    else return static_cast<long double>(value);
  }();
  if constexpr (Base == vec::ExpBase::E) return std::exp(widened);
  else if constexpr (Base == vec::ExpBase::Base2) return std::exp2(widened);
  else return std::exp2(widened * kLog2Base10Long);
}

template <vec::Accuracy Tier, vec::ExpBase Base, typename T>
void expect_accurate(T input, T actual) {
  const T expected = reference_exp<Base>(input);
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
  if constexpr (Tier == vec::Accuracy::Strict) {
    EXPECT_LE(ulps, 1u)
        << "x=" << as_double(input) << " expected=" << as_double(expected)
        << " actual=" << as_double(actual);
  } else if constexpr (Tier == vec::Accuracy::Fast) {
    if (std::isfinite(as_double(expected)) &&
        as_double(expected) >= as_double(std::numeric_limits<T>::min()))
      EXPECT_LE(ulps, 4u) << "x=" << as_double(input);
  } else if (std::isfinite(as_double(expected)) &&
             as_double(expected) >= as_double(std::numeric_limits<T>::min())) {
    const long double exact = reference_exp_exact<Base>(input);
    const long double relative = std::abs(
        static_cast<long double>(as_double(actual)) / exact - 1.0L);
    EXPECT_TRUE(ulps <= 4u || relative <= 0.006L)
        << "x=" << as_double(input) << " ulp=" << ulps
        << " relative=" << static_cast<double>(relative);
  }
}

template <typename T, vec::ExpBase Base>
constexpr double lower_bound() {
  if constexpr (std::same_as<T, double>) {
    if constexpr (Base == vec::ExpBase::E) return -700.0;
    else if constexpr (Base == vec::ExpBase::Base2) return -1000.0;
    else return -300.0;
  } else if constexpr (std::same_as<T, float>) {
    if constexpr (Base == vec::ExpBase::E) return -80.0;
    else if constexpr (Base == vec::ExpBase::Base2) return -120.0;
    else return -37.0;
  } else {
    if constexpr (Base == vec::ExpBase::E) return -9.0;
    else if constexpr (Base == vec::ExpBase::Base2) return -9.0;
    else return -4.0;
  }
}

template <typename T, vec::ExpBase Base>
constexpr double upper_bound() {
  if constexpr (std::same_as<T, double>) {
    if constexpr (Base == vec::ExpBase::E) return 700.0;
    else if constexpr (Base == vec::ExpBase::Base2) return 1000.0;
    else return 300.0;
  } else if constexpr (std::same_as<T, float>) {
    if constexpr (Base == vec::ExpBase::E) return 80.0;
    else if constexpr (Base == vec::ExpBase::Base2) return 120.0;
    else return 37.0;
  } else {
    if constexpr (Base == vec::ExpBase::E) return 9.0;
    else if constexpr (Base == vec::ExpBase::Base2) return 9.0;
    else return 4.0;
  }
}

template <vec::Accuracy Tier, vec::ExpBase Base, bool NegativeOnly,
          bool FullOptions = true, vec::FloatingTag Tag>
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
  const auto result = invoke_exp<Tier, Base, NegativeOnly>(tag, valid_input);
  const auto fixed_result = invoke_fixed_exp<Tier, Base, NegativeOnly>(
      tag, valid_input);
  const auto unmasked_result = invoke_exp<Tier, Base, NegativeOnly>(
      tag, valid_input, vec::opt::unmasked);
  const auto default_result = [&] {
    if constexpr (Tier != vec::Accuracy::Strict) {
      return result;
    } else {
      return vec::ExpCpo<Base, NegativeOnly>{}(tag, valid_input);
    }
  }();
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    expect_accurate<Tier, Base>(
        vec::get(tag, valid_input, lane), vec::get(tag, result, lane));
    EXPECT_EQ(
        bits(vec::get(tag, result, lane)),
        bits(vec::get(tag, unmasked_result, lane)))
        << "explicit unmasked lane=" << lane;
    EXPECT_EQ(
        bits(vec::get(tag, result, lane)),
        bits(vec::get(tag, fixed_result, lane)))
        << "fixed-accuracy forwarding lane=" << lane;
    if constexpr (Tier == vec::Accuracy::Strict) {
      EXPECT_EQ(
          bits(vec::get(tag, result, lane)),
          bits(vec::get(tag, default_result, lane)))
          << "default strict lane=" << lane;
    }
  }

  if constexpr (!FullOptions) return;
  const T scalar_merge = T(-7.0);
  const auto preserved = invoke_exp<Tier, Base, NegativeOnly>(
      tag, input, vec::opt::masked(mask));
  const auto zeroed = invoke_exp<Tier, Base, NegativeOnly>(
      tag, input, vec::opt::zero, vec::opt::masked(mask));
  const auto scalar = invoke_exp<Tier, Base, NegativeOnly>(
      tag, input, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
  const auto vector = invoke_exp<Tier, Base, NegativeOnly>(
      tag, input, vec::opt::merge(vector_merge), vec::opt::masked(mask));
  const vecops::nint_t prefix_count = vec::size(tag) / 2;
  const auto prefix = invoke_exp<Tier, Base, NegativeOnly>(
      tag, input, vec::opt::first(prefix_count),
      vec::opt::merge(scalar_merge));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const bool active = lane % 3 != 0;
    if (active) {
      const T expected_input = vec::get(tag, input, lane);
      expect_accurate<Tier, Base>(expected_input, vec::get(tag, preserved, lane));
      expect_accurate<Tier, Base>(expected_input, vec::get(tag, zeroed, lane));
      expect_accurate<Tier, Base>(expected_input, vec::get(tag, scalar, lane));
      expect_accurate<Tier, Base>(expected_input, vec::get(tag, vector, lane));
    } else {
      EXPECT_EQ(bits(vec::get(tag, input, lane)), bits(vec::get(tag, preserved, lane)));
      EXPECT_EQ(bits(T{}), bits(vec::get(tag, zeroed, lane)));
      EXPECT_EQ(bits(scalar_merge), bits(vec::get(tag, scalar, lane)));
      EXPECT_EQ(bits(vec::get(tag, vector_merge, lane)), bits(vec::get(tag, vector, lane)));
    }
    if (lane < prefix_count) {
      expect_accurate<Tier, Base>(
          vec::get(tag, input, lane), vec::get(tag, prefix, lane));
    } else {
      EXPECT_EQ(
          bits(scalar_merge), bits(vec::get(tag, prefix, lane)));
    }
  }
}

template <MathOperation Operation, typename T, bool FullOptions = true,
          vec::FloatingTag Tag>
void verify_every_operation(Tag tag) {
  constexpr vec::ExpBase base = operation_base(Operation);
  constexpr vec::Accuracy tier = operation_tier(Operation);
  constexpr bool negative = operation_negative(Operation);
  if constexpr (tier == vec::Accuracy::Strict)
    verify_lane_shape<vec::Accuracy::Strict, base, negative, FullOptions>(tag);
  else if constexpr (tier == vec::Accuracy::Fast)
    verify_lane_shape<vec::Accuracy::Fast, base, negative, FullOptions>(tag);
  else
    verify_lane_shape<vec::Accuracy::Estimate, base, negative, FullOptions>(tag);
}

template <vec::Accuracy Tier, vec::ExpBase Base, bool NegativeOnly,
          typename T>
void verify_dense_samples() {
  vec::ScalableTag<T, 0> tag;
  constexpr int sample_count = 4096;
  for (int base = 0; base <= sample_count; base += static_cast<int>(vec::size(tag))) {
    auto input = vec::zeros(tag);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const int index = std::min(base + static_cast<int>(lane), sample_count);
      const double lower = lower_bound<T, Base>();
      const double upper = NegativeOnly ? 0.0 : upper_bound<T, Base>();
      input = vec::set(tag, input, lane, T(
          lower + (upper - lower) * index / sample_count));
    }
    const auto result = invoke_exp<Tier, Base, NegativeOnly>(tag, input);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
      expect_accurate<Tier, Base>(
          vec::get(tag, input, lane), vec::get(tag, result, lane));
  }
}

template <MathOperation Operation, typename T>
void verify_dense_for_operation() {
  constexpr vec::ExpBase b = operation_base(Operation);
  constexpr vec::Accuracy tier = operation_tier(Operation);
  constexpr bool negative = operation_negative(Operation);
  if constexpr (tier == vec::Accuracy::Strict)
    verify_dense_samples<vec::Accuracy::Strict, b, negative, T>();
  else if constexpr (tier == vec::Accuracy::Fast)
    verify_dense_samples<vec::Accuracy::Fast, b, negative, T>();
  else
    verify_dense_samples<vec::Accuracy::Estimate, b, negative, T>();
}

} // namespace

template <MathOperation Operation, typename T>
void run_math_shapes_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::FloatingTag Tag>() {
    constexpr bool options = vec_test::exhaustive_options_shape<Tag> &&
        (vec::scale_power_v<Tag> != vec_test::details::maximum_scalable_power ||
         std::same_as<T, vecops::float32_t>);
    verify_every_operation<Operation, T, options>(Tag{});
  });
}

template <MathOperation Operation, typename T>
void run_math_dense_test() {
  verify_dense_for_operation<Operation, T>();
}

#if !defined(VECOPS_MATH_ASSUME_VALID_INPUTS)
template <typename T>
void run_math_ieee_edges_test() {
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
    else expect_accurate<vec::Accuracy::Strict, vec::ExpBase::E>(
        vec::get(tag, input, lane), vec::get(tag, result, lane));
  }
}
#endif

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
template <MathOperation Operation, typename T>
void run_fixed_sve_math_test() {
  if constexpr (!std::same_as<T, vecops::float32_t>) {
    GTEST_SKIP() << "f32 is the representative >4-word floating type";
  } else {
    constexpr vecops::nint_t word_lanes =
        FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
    constexpr vecops::nint_t lanes = static_cast<vecops::nint_t>(
        std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
    using Tag = vec::FixedTag<T, lanes>;
    EXPECT_GT(vec::num_words(Tag{}), 4);
    verify_every_operation<Operation, T, false>(Tag{});
  }
}
#endif

constexpr auto shard_math_operation = static_cast<MathOperation>(
    VECOPS_TEST_SHARD_INDEX / vec_test::FloatingElements::size);
constexpr std::size_t shard_math_type_index =
    VECOPS_TEST_SHARD_INDEX % vec_test::FloatingElements::size;
using ShardType = vec_test::FloatingElementAt<shard_math_type_index>;
template void run_math_shapes_test<shard_math_operation, ShardType>();
template void run_math_dense_test<shard_math_operation, ShardType>();
#if !defined(VECOPS_MATH_ASSUME_VALID_INPUTS)
#if VECOPS_TEST_SHARD_INDEX < 4
template void run_math_ieee_edges_test<ShardType>();
#endif
#endif
#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
template void run_fixed_sve_math_test<shard_math_operation, ShardType>();
#endif

#else

template <typename T>
class VecMathTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecMathTest,
    vec_test::FloatingElementTypes,
    vec_test::ElementTypeName);

template <typename T>
void run_all_math_shapes() {
  run_math_shapes_test<MathOperation::ExpStrict, T>();
  run_math_shapes_test<MathOperation::ExpFast, T>();
  run_math_shapes_test<MathOperation::ExpEstimate, T>();
  run_math_shapes_test<MathOperation::ExpNegStrict, T>();
  run_math_shapes_test<MathOperation::ExpNegFast, T>();
  run_math_shapes_test<MathOperation::ExpNegEstimate, T>();
  run_math_shapes_test<MathOperation::Exp2Strict, T>();
  run_math_shapes_test<MathOperation::Exp2Fast, T>();
  run_math_shapes_test<MathOperation::Exp2Estimate, T>();
  run_math_shapes_test<MathOperation::Exp2NegStrict, T>();
  run_math_shapes_test<MathOperation::Exp2NegFast, T>();
  run_math_shapes_test<MathOperation::Exp2NegEstimate, T>();
  run_math_shapes_test<MathOperation::Exp10Strict, T>();
  run_math_shapes_test<MathOperation::Exp10Fast, T>();
  run_math_shapes_test<MathOperation::Exp10Estimate, T>();
  run_math_shapes_test<MathOperation::Exp10NegStrict, T>();
  run_math_shapes_test<MathOperation::Exp10NegFast, T>();
  run_math_shapes_test<MathOperation::Exp10NegEstimate, T>();
}

template <typename T>
void run_all_math_dense_cases() {
  run_math_dense_test<MathOperation::ExpStrict, T>();
  run_math_dense_test<MathOperation::ExpFast, T>();
  run_math_dense_test<MathOperation::ExpEstimate, T>();
  run_math_dense_test<MathOperation::ExpNegStrict, T>();
  run_math_dense_test<MathOperation::ExpNegFast, T>();
  run_math_dense_test<MathOperation::ExpNegEstimate, T>();
  run_math_dense_test<MathOperation::Exp2Strict, T>();
  run_math_dense_test<MathOperation::Exp2Fast, T>();
  run_math_dense_test<MathOperation::Exp2Estimate, T>();
  run_math_dense_test<MathOperation::Exp2NegStrict, T>();
  run_math_dense_test<MathOperation::Exp2NegFast, T>();
  run_math_dense_test<MathOperation::Exp2NegEstimate, T>();
  run_math_dense_test<MathOperation::Exp10Strict, T>();
  run_math_dense_test<MathOperation::Exp10Fast, T>();
  run_math_dense_test<MathOperation::Exp10Estimate, T>();
  run_math_dense_test<MathOperation::Exp10NegStrict, T>();
  run_math_dense_test<MathOperation::Exp10NegFast, T>();
  run_math_dense_test<MathOperation::Exp10NegEstimate, T>();
}

TYPED_TEST(VecMathTest, EveryOperationShapeAndOptions) {
  run_all_math_shapes<TypeParam>();
}

TYPED_TEST(VecMathTest, DenseAccuracyAndNegativeDomain) {
  run_all_math_dense_cases<TypeParam>();
}

#if !defined(VECOPS_MATH_ASSUME_VALID_INPUTS)
TYPED_TEST(VecMathTest, IeeeEdges) {
  run_math_ieee_edges_test<TypeParam>();
}
#endif

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecMathTest, FixedSVEBatchesBeyondTupleLimit) {
  if constexpr (!std::same_as<TypeParam, vecops::float32_t>) {
    GTEST_SKIP() << "f32 is the representative >4-word floating type";
  } else {
    run_fixed_sve_math_test<MathOperation::ExpStrict, TypeParam>();
    run_fixed_sve_math_test<MathOperation::ExpFast, TypeParam>();
    run_fixed_sve_math_test<MathOperation::ExpEstimate, TypeParam>();
    run_fixed_sve_math_test<MathOperation::ExpNegStrict, TypeParam>();
    run_fixed_sve_math_test<MathOperation::ExpNegFast, TypeParam>();
    run_fixed_sve_math_test<MathOperation::ExpNegEstimate, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp2Strict, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp2Fast, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp2Estimate, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp2NegStrict, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp2NegFast, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp2NegEstimate, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp10Strict, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp10Fast, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp10Estimate, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp10NegStrict, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp10NegFast, TypeParam>();
    run_fixed_sve_math_test<MathOperation::Exp10NegEstimate, TypeParam>();
  }
}
#endif

#endif
