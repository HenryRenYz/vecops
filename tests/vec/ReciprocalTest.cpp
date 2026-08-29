#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "vecops/vec/Vec.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

namespace {

enum class RecipKind { Rcp, Rsqrt };

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

template <bool IsRsqrt, typename T>
T reference(T value) {
  const long double widened = static_cast<double>(value);
  if constexpr (IsRsqrt) {
    if (value == T(0)) return std::numeric_limits<T>::infinity();
    return static_cast<T>(
        static_cast<double>(1.0L / std::sqrt(widened)));
  } else {
    return static_cast<T>(static_cast<double>(1.0L / widened));
  }
}

template <typename T>
std::uint64_t ulps_between(T expected, T actual) {
  const auto expected_bits = bits(expected);
  const auto actual_bits = bits(actual);
  return expected_bits > actual_bits ? expected_bits - actual_bits
                                     : actual_bits - expected_bits;
}

/**
 * Checks one tier result against the long-double reference under the
 * documented contracts:
 *   Strict   <= 1 ULP (element format), gradual/flushed subnormals per build
 *   Fast     <= 2^-15 relative (f32/f64), <= 2 ULP (f16/bf16)
 *   Estimate <= 2^-7 relative
 * Overflow boundaries relax the bit-exact checks: for the estimate tiers a
 * reciprocal that correctly rounds to +-inf may legitimately come out as a
 * finite value near the format maximum, and a result that lands exactly on
 * the smallest normal may flush in non-preserving builds when the f32 chain
 * sits one f32-ULP below the narrow-format boundary.
 */
template <vec::Accuracy Tier, bool IsRsqrt, typename T>
void expect_within_tier(T input, T actual) {
  const T expected = reference<IsRsqrt>(input);
  ASSERT_EQ(std::isnan(as_double(expected)), std::isnan(as_double(actual)))
      << "x=" << as_double(input);
  if (std::isnan(as_double(expected))) return;

  // Reciprocal overflow (rcp of tiny subnormals) is correctly +/-inf.
  if (std::isinf(as_double(expected))) {
    if constexpr (Tier == vec::Accuracy::Strict) {
      EXPECT_TRUE(vec_test::values_identical(expected, actual))
          << "x=" << as_double(input) << " actual=" << as_double(actual);
    } else {
      const bool near_overflow = std::isinf(as_double(actual)) ||
          std::abs(as_double(actual)) >=
              0.5 * static_cast<double>(std::numeric_limits<T>::max());
      EXPECT_TRUE(near_overflow)
          << "x=" << as_double(input) << " actual=" << as_double(actual);
    }
    return;
  }
  // rcp(+-inf) is exactly signed zero.
  if (expected == T(0)) {
    EXPECT_TRUE(vec_test::values_identical(expected, actual))
        << "x=" << as_double(input) << " actual=" << as_double(actual);
    return;
  }

  const double magnitude = std::abs(as_double(expected));
  const bool subnormal_output =
      magnitude > 0.0 &&
      magnitude < static_cast<double>(std::numeric_limits<T>::min());
  if (subnormal_output) {
    // Fast/Estimate (and non-preserving Strict) flush subnormal outputs.
    if constexpr (Tier == vec::Accuracy::Strict) {
#ifdef VECOPS_PRESERVE_SUBNORMALS
      EXPECT_LE(ulps_between(expected, actual), 1u)
          << "x=" << as_double(input) << " gradual";
#else
      EXPECT_TRUE(as_double(actual) == 0.0 || actual == expected)
          << "x=" << as_double(input) << " flushed";
#endif
    } else {
      EXPECT_TRUE(as_double(actual) == 0.0 || actual == expected)
          << "x=" << as_double(input) << " flushed";
    }
    return;
  }

  // A result one narrow-format ULP below the smallest normal flushes to zero
  // in non-preserving builds; accept that boundary behavior.
  if (std::abs(expected) == std::numeric_limits<T>::min() &&
      as_double(actual) == 0.0) {
#ifdef VECOPS_PRESERVE_SUBNORMALS
    if constexpr (Tier == vec::Accuracy::Strict) {
      ADD_FAILURE() << "x=" << as_double(input) << " boundary flush";
    }
#endif
    return;
  }

  if constexpr (Tier == vec::Accuracy::Strict) {
    EXPECT_LE(ulps_between(expected, actual), 1u)
        << "x=" << as_double(input) << " expected=" << as_double(expected)
        << " actual=" << as_double(actual);
  } else {
    const long double exact = IsRsqrt
        ? 1.0L / std::sqrt(static_cast<long double>(as_double(input)))
        : 1.0L / static_cast<long double>(as_double(input));
    const long double relative = std::abs(
        static_cast<long double>(as_double(actual)) / exact - 1.0L);
    if constexpr (Tier == vec::Accuracy::Fast) {
      // 2^-15 for the wide formats; 2 ULP of the narrow mantissa
      // (f16 keeps 11-bit precision, bf16 8-bit).
      const long double bound = [] {
        if constexpr (std::same_as<T, float> ||
                      std::same_as<T, double>)
          return 3.0517578125e-5L;
        else if constexpr (std::same_as<T, vecops::float16_t>)
          return 9.765625e-4L;
        else
          return 7.8125e-3L;
      }();
      EXPECT_LE(relative, bound)
          << "x=" << as_double(input) << " actual=" << as_double(actual);
    } else {
      EXPECT_LE(relative, 7.8125e-3L)
          << "x=" << as_double(input) << " actual=" << as_double(actual);
    }
  }
}

template <RecipKind Kind>
constexpr bool is_rsqrt = Kind == RecipKind::Rsqrt;

template <RecipKind Kind, vec::Accuracy Tier, vec::FloatingTag Tag,
          typename... Options>
vec::Vec<Tag> invoke_reciprocal(
    Tag tag, vec::Vec<Tag> value, Options&&... options) {
  if constexpr (Kind == RecipKind::Rsqrt)
    return vec::rsqrt(
        tag, value, std::forward<Options>(options)...,
        vec::opt::math::accuracy<Tier>);
  else
    return vec::rcp(
        tag, value, std::forward<Options>(options)...,
        vec::opt::math::accuracy<Tier>);
}

template <RecipKind Kind>
float reciprocal_operand(vecops::nint_t lane, bool active = true) {
  if (!active) return -static_cast<float>(lane + 1);
  // Powers of two and their odd neighbors exercise every branch of the
  // refinement chains.
  constexpr float values[] = {
      0.25F, 0.5F, 1.0F, 2.0F, 3.0F, 4.0F, 9.0F, 16.0F};
  return values[lane % 8];
}

template <
    RecipKind Kind, vec::Accuracy Tier, bool FullOptions,
    vec::FloatingTag Tag>
void exercise_reciprocal_tier(Tag tag) {
  using T = vec::ElementOf<Tag>;
  auto value = vec::zeros(tag);
  auto vector_merge = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  const T scalar_merge = static_cast<T>(13.0F);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const bool active = (lane % 3) != 0;
    value = vec::set(
        tag, value, lane,
        static_cast<T>(reciprocal_operand<Kind>(lane, active)));
    vector_merge = vec::set(
        tag, vector_merge, lane, static_cast<T>(lane + 17.0F));
    mask = vec::set(tag, mask, lane, active);
  }

  auto positive = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    positive = vec::set(
        tag, positive, lane, static_cast<T>(reciprocal_operand<Kind>(lane)));

  const auto result = invoke_reciprocal<Kind, Tier>(tag, positive);
  const auto unmasked =
      invoke_reciprocal<Kind, Tier>(tag, positive, vec::opt::unmasked);
  const auto first = invoke_reciprocal<Kind, Tier>(
      tag, positive, vec::opt::first(vec::size(tag)));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T input = static_cast<T>(reciprocal_operand<Kind>(lane));
    expect_within_tier<Tier, is_rsqrt<Kind>>(
        input, vec::get(tag, result, lane));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, result, lane), vec::get(tag, unmasked, lane)))
        << "unmasked lane=" << lane;
    // Estimate-seed instructions are not bit-deterministic across compiled
    // call shapes (masked vs unmasked forms); compare within the tier bound.
    expect_within_tier<Tier, is_rsqrt<Kind>>(
        input, vec::get(tag, first, lane));
  }

  if constexpr (FullOptions) {
    const auto masked_default =
        invoke_reciprocal<Kind, Tier>(tag, value, vec::opt::masked(mask));
    const auto masked_zero = invoke_reciprocal<Kind, Tier>(
        tag, value, vec::opt::masked(mask), vec::opt::zero);
    const auto masked_scalar = invoke_reciprocal<Kind, Tier>(
        tag, value, vec::opt::merge(scalar_merge),
        vec::opt::masked(mask));
    const auto masked_vector = invoke_reciprocal<Kind, Tier>(
        tag, value, vec::opt::masked(mask), vec::opt::merge(vector_merge));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const bool active = (lane % 3) != 0;
      const T input =
          static_cast<T>(reciprocal_operand<Kind>(lane, active));
      if (active) {
        expect_within_tier<Tier, is_rsqrt<Kind>>(
            input, vec::get(tag, masked_default, lane));
        expect_within_tier<Tier, is_rsqrt<Kind>>(
            input, vec::get(tag, masked_zero, lane));
        expect_within_tier<Tier, is_rsqrt<Kind>>(
            input, vec::get(tag, masked_scalar, lane));
        expect_within_tier<Tier, is_rsqrt<Kind>>(
            input, vec::get(tag, masked_vector, lane));
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

} // namespace

template <RecipKind Kind, bool FullOptions = true, vec::FloatingTag Tag>
void verify_reciprocal_shapes(Tag tag) {
  exercise_reciprocal_tier<Kind, vec::Accuracy::Strict, FullOptions>(tag);
  exercise_reciprocal_tier<Kind, vec::Accuracy::Fast, FullOptions>(tag);
  exercise_reciprocal_tier<Kind, vec::Accuracy::Estimate, FullOptions>(tag);
}

template <typename T>
void run_reciprocal_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::FloatingTag Tag>() {
    constexpr bool options = vec_test::exhaustive_options_shape<Tag>;
    verify_reciprocal_shapes<RecipKind::Rcp, options>(Tag{});
    verify_reciprocal_shapes<RecipKind::Rsqrt, options>(Tag{});
  });
}

template <vec::Accuracy Tier, typename T>
void run_reciprocal_accuracy_test() {
  using Tag = vec::ScalableTag<T>;
  const Tag tag{};
  const vecops::nint_t lanes = vec::size(tag);
  // Deterministic mixed magnitudes: random bit patterns across the domain.
  std::uint64_t state = 0x9e3779b97f4a7c15ull;
  const auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  std::vector<T> inputs(static_cast<std::size_t>(lanes));
  for (int block = 0; block < 4096; ++block) {
    for (vecops::nint_t i = 0; i < lanes; ++i) {
      const std::uint64_t bits = next();
      T value;
      if constexpr (sizeof(T) == 2) {
        value = T::from_bits(static_cast<std::uint16_t>(bits >> 48));
      } else if constexpr (sizeof(T) == 4) {
        std::memcpy(&value, &bits, 4);
      } else {
        std::memcpy(&value, &bits, 8);
      }
      inputs[static_cast<std::size_t>(i)] = value;
    }
    const auto v = vec::load(tag, inputs.data());
    const auto rcp_result =
        vec::rcp(tag, v, vec::opt::math::accuracy<Tier>);
    const auto rsqrt_result =
        vec::rsqrt(tag, v, vec::opt::math::accuracy<Tier>);
    for (vecops::nint_t i = 0; i < lanes; ++i) {
      const T input = inputs[static_cast<std::size_t>(i)];
      if (std::isnan(as_double(input))) continue;
      expect_within_tier<Tier, false>(input, vec::get(tag, rcp_result, i));
      expect_within_tier<Tier, true>(
          input, vec::get(tag, rsqrt_result, i));
    }
  }
}

namespace {

template <typename T, vec::Accuracy Tier>
void check_reciprocal_pair(
    T input, vec::Vec<vec::ScalableTag<T>>& rcp_result,
    vec::Vec<vec::ScalableTag<T>>& rsqrt_result) {
  const auto filled = vec::fill(vec::ScalableTag<T>{}, input);
  rcp_result = vec::rcp(
      vec::ScalableTag<T>{}, filled, vec::opt::math::accuracy<Tier>);
  rsqrt_result = vec::rsqrt(
      vec::ScalableTag<T>{}, filled, vec::opt::math::accuracy<Tier>);
}

} // namespace

template <typename T, vec::Accuracy Tier>
void run_reciprocal_special_values_for_tier() {
  using Tag = vec::ScalableTag<T>;
  const auto nan = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
  const auto infinity =
      static_cast<T>(std::numeric_limits<double>::infinity());
  const auto smallest = std::numeric_limits<T>::min();
  const auto largest = std::numeric_limits<T>::max();
  const auto smallest_subnormal = std::numeric_limits<T>::denorm_min();

  // rcp(+-0) = +-inf, rsqrt(+-0) = +inf
  {
    vec::Vec<Tag> rcp_result, rsqrt_result;
    check_reciprocal_pair<T, Tier>(T(0), rcp_result, rsqrt_result);
    const double got = as_double(vec::get(Tag{}, rcp_result, 0));
    EXPECT_TRUE(std::isinf(got));
    EXPECT_FALSE(std::signbit(got));
    EXPECT_TRUE(std::isinf(as_double(vec::get(Tag{}, rsqrt_result, 0))));
    EXPECT_FALSE(std::signbit(as_double(vec::get(Tag{}, rsqrt_result, 0))));
  }
  {
    vec::Vec<Tag> rcp_result, rsqrt_result;
    check_reciprocal_pair<T, Tier>(T(-0.0), rcp_result, rsqrt_result);
    const double got = as_double(vec::get(Tag{}, rcp_result, 0));
    EXPECT_TRUE(std::isinf(got));
    EXPECT_TRUE(std::signbit(got));
    EXPECT_TRUE(std::isinf(as_double(vec::get(Tag{}, rsqrt_result, 0))));
    EXPECT_FALSE(std::signbit(as_double(vec::get(Tag{}, rsqrt_result, 0))));
  }
  // rcp(+inf) = +0, rsqrt(+inf) = +0; rcp(-inf) = -0, rsqrt(-inf) = NaN.
  {
    vec::Vec<Tag> rcp_result, rsqrt_result;
    check_reciprocal_pair<T, Tier>(infinity, rcp_result, rsqrt_result);
    EXPECT_EQ(0.0, as_double(vec::get(Tag{}, rcp_result, 0)));
    EXPECT_EQ(0.0, as_double(vec::get(Tag{}, rsqrt_result, 0)));
  }
  {
    vec::Vec<Tag> rcp_result, rsqrt_result;
    check_reciprocal_pair<T, Tier>(-infinity, rcp_result, rsqrt_result);
    EXPECT_EQ(0.0, as_double(vec::get(Tag{}, rcp_result, 0)));
    EXPECT_TRUE(std::signbit(as_double(vec::get(Tag{}, rcp_result, 0))));
    EXPECT_TRUE(std::isnan(as_double(vec::get(Tag{}, rsqrt_result, 0))));
  }
  // NaN propagates through both ops.
  {
    vec::Vec<Tag> rcp_result, rsqrt_result;
    check_reciprocal_pair<T, Tier>(nan, rcp_result, rsqrt_result);
    EXPECT_TRUE(std::isnan(as_double(vec::get(Tag{}, rcp_result, 0))));
    EXPECT_TRUE(std::isnan(as_double(vec::get(Tag{}, rsqrt_result, 0))));
  }
  // Negative finite: rcp keeps the sign, rsqrt is NaN.
  {
    vec::Vec<Tag> rcp_result, rsqrt_result;
    check_reciprocal_pair<T, Tier>(T(-4), rcp_result, rsqrt_result);
    const double got = as_double(vec::get(Tag{}, rcp_result, 0));
    EXPECT_NEAR(got, -0.25, std::abs(got) * 0.01 + 1e-9);
    EXPECT_TRUE(std::isnan(as_double(vec::get(Tag{}, rsqrt_result, 0))));
  }
  // Subnormal inputs in every tier.
  {
    vec::Vec<Tag> rcp_result, rsqrt_result;
    check_reciprocal_pair<T, Tier>(
        smallest_subnormal, rcp_result, rsqrt_result);
    const T want = reference<false>(smallest_subnormal);
    if (std::isinf(as_double(want))) {
      EXPECT_TRUE(std::isinf(as_double(vec::get(Tag{}, rcp_result, 0))));
    } else {
      expect_within_tier<Tier, false>(
          smallest_subnormal, vec::get(Tag{}, rcp_result, 0));
    }
    expect_within_tier<Tier, true>(
        smallest_subnormal, vec::get(Tag{}, rsqrt_result, 0));
  }
  // Boundary normals in every tier.
  {
    vec::Vec<Tag> rcp_result, rsqrt_result;
    check_reciprocal_pair<T, Tier>(smallest, rcp_result, rsqrt_result);
    expect_within_tier<Tier, false>(
        smallest, vec::get(Tag{}, rcp_result, 0));
    expect_within_tier<Tier, true>(
        smallest, vec::get(Tag{}, rsqrt_result, 0));
  }
  {
    vec::Vec<Tag> rcp_result, rsqrt_result;
    check_reciprocal_pair<T, Tier>(largest, rcp_result, rsqrt_result);
    expect_within_tier<Tier, false>(
        largest, vec::get(Tag{}, rcp_result, 0));
    expect_within_tier<Tier, true>(
        largest, vec::get(Tag{}, rsqrt_result, 0));
  }
}

template <typename T>
void run_reciprocal_special_values_test() {
  run_reciprocal_special_values_for_tier<T, vec::Accuracy::Strict>();
  run_reciprocal_special_values_for_tier<T, vec::Accuracy::Fast>();
  run_reciprocal_special_values_for_tier<T, vec::Accuracy::Estimate>();
}

template <typename T>
void run_reciprocal_fixed_cpo_test() {
  using Tag = vec::ScalableTag<T>;
  const Tag tag{};
  const auto v = vec::fill(tag, T(7));
  const auto strict = vec::rcp(tag, v, vec::opt::math::strict);
  const auto fast = vec::rcp(tag, v, vec::opt::math::fast);
  const auto est = vec::rcp(tag, v, vec::opt::math::estimate);
  EXPECT_TRUE(vec_test::values_identical(
      vec::get(tag, strict, 0), vec::get(tag, vec::rcp_strict(tag, v), 0)));
  EXPECT_TRUE(vec_test::values_identical(
      vec::get(tag, fast, 0), vec::get(tag, vec::rcp_fast(tag, v), 0)));
  EXPECT_TRUE(vec_test::values_identical(
      vec::get(tag, est, 0), vec::get(tag, vec::rcp_est(tag, v), 0)));
  const auto strict_r = vec::rsqrt(tag, v, vec::opt::math::strict);
  const auto fast_r = vec::rsqrt(tag, v, vec::opt::math::fast);
  const auto est_r = vec::rsqrt(tag, v, vec::opt::math::estimate);
  EXPECT_TRUE(vec_test::values_identical(
      vec::get(tag, strict_r, 0),
      vec::get(tag, vec::rsqrt_strict(tag, v), 0)));
  EXPECT_TRUE(vec_test::values_identical(
      vec::get(tag, fast_r, 0), vec::get(tag, vec::rsqrt_fast(tag, v), 0)));
  EXPECT_TRUE(vec_test::values_identical(
      vec::get(tag, est_r, 0), vec::get(tag, vec::rsqrt_est(tag, v), 0)));
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

template <typename T>
void run_fixed_reciprocal_test() {
  if constexpr (!std::same_as<T, vecops::float32_t>) {
    GTEST_SKIP() << "f32 is the representative >4-word floating type";
  } else {
    constexpr vecops::nint_t word_lanes =
        FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
    constexpr vecops::nint_t lanes = static_cast<vecops::nint_t>(
        std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
    using Tag = vec::FixedTag<T, lanes>;
    EXPECT_GT(vec::num_words(Tag{}), 4);
    verify_reciprocal_shapes<RecipKind::Rcp, false>(Tag{});
    verify_reciprocal_shapes<RecipKind::Rsqrt, false>(Tag{});
  }
}

#endif

template <typename T>
class VecReciprocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecReciprocalTest,
    vec_test::FloatingElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecReciprocalTest, CoversEveryLaneShapeAndPopulation) {
  run_reciprocal_test<TypeParam>();
}

TYPED_TEST(VecReciprocalTest, StrictAccuracyHoldsOnMixedMagnitudes) {
  run_reciprocal_accuracy_test<vec::Accuracy::Strict, TypeParam>();
}

TYPED_TEST(VecReciprocalTest, FastAccuracyHoldsOnMixedMagnitudes) {
  run_reciprocal_accuracy_test<vec::Accuracy::Fast, TypeParam>();
}

TYPED_TEST(VecReciprocalTest, EstimateAccuracyHoldsOnMixedMagnitudes) {
  run_reciprocal_accuracy_test<vec::Accuracy::Estimate, TypeParam>();
}

TYPED_TEST(VecReciprocalTest, PreservesIEEESpecialValueClasses) {
  run_reciprocal_special_values_test<TypeParam>();
}

TYPED_TEST(VecReciprocalTest, FixedAccuracyCposMatchCanonicalForms) {
  run_reciprocal_fixed_cpo_test<TypeParam>();
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecReciprocalTest, FixedSVEBatchesBeyondTupleLimit) {
  run_fixed_reciprocal_test<TypeParam>();
}
#endif
