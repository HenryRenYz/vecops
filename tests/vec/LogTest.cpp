// @vecops-test-shards: 8
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

template <vec::LogBase Base, typename T>
long double reference_widened(T value) {
  const long double widened = static_cast<double>(value);
  if constexpr (Base == vec::LogBase::E) return std::log(widened);
  else if constexpr (Base == vec::LogBase::Base2) return std::log2(widened);
  else return std::log10(widened);
}

template <vec::LogBase Base, typename T>
T reference(T value) {
  return static_cast<T>(static_cast<double>(reference_widened<Base>(value)));
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
 * documented log-family contracts:
 *   Strict   <= 1 ULP (f32 native kernel on SVE; f16/bf16 via the f32
 *              pipeline), <= 4 ULP (f64, the optimized-routines libm class)
 *   Fast     <= 2^-13 relative (f32/f64), <= 2 ULP (f16/bf16)
 *   Estimate <= 2^-7 relative
 * Exact-class results are pinned bit-exactly in every tier: log(+-0) is
 * -inf, log(+inf) is +inf, and log(1) is +0. Logarithms cross zero at x=1,
 * so a sign disagreement is always a violation; the ULP distance is only
 * meaningful for same-sign values.
 */
template <vec::Accuracy Tier, vec::LogBase Base, typename T>
void expect_within_tier(T input, T actual) {
  const T expected = reference<Base>(input);
  ASSERT_EQ(std::isnan(as_double(expected)), std::isnan(as_double(actual)))
      << "x=" << as_double(input);
  if (std::isnan(as_double(expected))) return;

  // log(+-0) = -inf and log(+inf) = +inf are exact in every tier.
  if (std::isinf(as_double(expected))) {
    EXPECT_TRUE(vec_test::values_identical(expected, actual))
        << "x=" << as_double(input) << " actual=" << as_double(actual);
    return;
  }
  // log(1) = +0 exactly.
  if (expected == T(0)) {
    EXPECT_TRUE(vec_test::values_identical(T(0), actual))
        << "x=" << as_double(input) << " actual=" << as_double(actual);
    return;
  }
  // The zero crossing at x=1: a sign flip is outside every tier contract.
  ASSERT_EQ(std::signbit(as_double(expected)), std::signbit(as_double(actual)))
      << "x=" << as_double(input) << " expected=" << as_double(expected)
      << " actual=" << as_double(actual);

  if constexpr (Tier == vec::Accuracy::Strict) {
    const unsigned bound = std::same_as<T, double> ? 4u : 1u;
    EXPECT_LE(ulps_between(expected, actual), bound)
        << "x=" << as_double(input) << " expected=" << as_double(expected)
        << " actual=" << as_double(actual);
  } else {
    const long double exact = reference_widened<Base>(input);
    const long double relative = std::abs(
        static_cast<long double>(as_double(actual)) / exact - 1.0L);
    if constexpr (Tier == vec::Accuracy::Fast) {
      // 2^-13 for the wide formats; 2 ULP of the narrow mantissa.
      const long double bound = [] {
        if constexpr (std::same_as<T, float> ||
                      std::same_as<T, double>)
          return 1.220703125e-4L;
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

template <vec::LogBase Base, vec::Accuracy Tier, vec::FloatingTag Tag,
          typename... Options>
vec::Vec<Tag> invoke_log(Tag tag, vec::Vec<Tag> value, Options&&... options) {
  return vec::LogCpo<Base>{}(
      tag, value, std::forward<Options>(options)...,
      vec::opt::math::accuracy<Tier>);
}

// Powers of two, window edges (2/3, 4/3), and near-1 neighbors exercise
// every reduction branch of the log kernels.
template <vec::LogBase Base>
double log_operand(vecops::nint_t lane, bool active = true) {
  if (!active) return -static_cast<double>(lane + 1);
  constexpr double values[] = {
      0.25, 2.0 / 3.0, 1.0, 4.0 / 3.0, 2.0, 3.0, 8.0, 1024.0};
  return values[lane % 8];
}

template <
    vec::LogBase Base, vec::Accuracy Tier, bool FullOptions,
    vec::FloatingTag Tag>
void exercise_log_tier(Tag tag) {
  using T = vec::ElementOf<Tag>;
  auto value = vec::zeros(tag);
  auto vector_merge = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  const T scalar_merge = static_cast<T>(13.0F);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const bool active = (lane % 3) != 0;
    value = vec::set(
        tag, value, lane, static_cast<T>(log_operand<Base>(lane, active)));
    vector_merge = vec::set(
        tag, vector_merge, lane, static_cast<T>(lane + 17.0F));
    mask = vec::set(tag, mask, lane, active);
  }

  auto positive = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    positive = vec::set(
        tag, positive, lane, static_cast<T>(log_operand<Base>(lane)));

  const auto result = invoke_log<Base, Tier>(tag, positive);
  const auto unmasked =
      invoke_log<Base, Tier>(tag, positive, vec::opt::unmasked);
  const auto first = invoke_log<Base, Tier>(
      tag, positive, vec::opt::first(vec::size(tag)));
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T input = static_cast<T>(log_operand<Base>(lane));
    expect_within_tier<Tier, Base>(input, vec::get(tag, result, lane));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, result, lane), vec::get(tag, unmasked, lane)))
        << "unmasked lane=" << lane;
    expect_within_tier<Tier, Base>(input, vec::get(tag, first, lane));
  }

  if constexpr (FullOptions) {
    const auto masked_default =
        invoke_log<Base, Tier>(tag, value, vec::opt::masked(mask));
    const auto masked_zero = invoke_log<Base, Tier>(
        tag, value, vec::opt::masked(mask), vec::opt::zero);
    const auto masked_scalar = invoke_log<Base, Tier>(
        tag, value, vec::opt::merge(scalar_merge),
        vec::opt::masked(mask));
    const auto masked_vector = invoke_log<Base, Tier>(
        tag, value, vec::opt::masked(mask), vec::opt::merge(vector_merge));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const bool active = (lane % 3) != 0;
      const T input = static_cast<T>(log_operand<Base>(lane, active));
      if (active) {
        expect_within_tier<Tier, Base>(
            input, vec::get(tag, masked_default, lane));
        expect_within_tier<Tier, Base>(
            input, vec::get(tag, masked_zero, lane));
        expect_within_tier<Tier, Base>(
            input, vec::get(tag, masked_scalar, lane));
        expect_within_tier<Tier, Base>(
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

template <vec::LogBase Base, bool FullOptions = true, vec::FloatingTag Tag>
void verify_log_shapes(Tag tag) {
  exercise_log_tier<Base, vec::Accuracy::Strict, FullOptions>(tag);
  exercise_log_tier<Base, vec::Accuracy::Fast, FullOptions>(tag);
  exercise_log_tier<Base, vec::Accuracy::Estimate, FullOptions>(tag);
}

template <typename T>
void run_log_test() {
  vec_test::for_each_scalable_shape<T>([]<vec::FloatingTag Tag>() {
    constexpr bool options = vec_test::exhaustive_options_shape<Tag>;
    verify_log_shapes<vec::LogBase::E, options>(Tag{});
    verify_log_shapes<vec::LogBase::Base2, options>(Tag{});
    verify_log_shapes<vec::LogBase::Base10, options>(Tag{});
  });
}

template <vec::Accuracy Tier, typename T>
void run_log_accuracy_test() {
  using Tag = vec::ScalableTag<T>;
  const Tag tag{};
  const vecops::nint_t lanes = vec::size(tag);
  // Deterministic mixed magnitudes: random positive bit patterns plus a
  // dense cluster around the x=1 zero crossing.
  std::uint64_t state = 0x9e3779b97f4a7c15ull;
  const auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  auto input = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const std::uint64_t raw = next();
    double value;
    if (lane % 4 == 0) {
      value = 1.0 + (static_cast<double>(raw >> 11) / (1ull << 53) - 0.5) *
                        2.0e-3;
    } else {
      // Positive f32/f64 bit patterns with wide exponent spread.
      const std::uint64_t exponent = (raw >> 55) & 0xff;
      const double scale = std::ldexp(
          1.0, static_cast<int>(exponent) - 100);
      value = scale * (1.0 + static_cast<double>(raw >> 40) / (1ull << 24));
    }
    input = vec::set(tag, input, lane, static_cast<T>(value));
  }
  const auto out_e = invoke_log<vec::LogBase::E, Tier>(tag, input);
  const auto out_2 = invoke_log<vec::LogBase::Base2, Tier>(tag, input);
  const auto out_10 = invoke_log<vec::LogBase::Base10, Tier>(tag, input);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    const T x = vec::get(tag, input, lane);
    expect_within_tier<Tier, vec::LogBase::E>(
        x, vec::get(tag, out_e, lane));
    expect_within_tier<Tier, vec::LogBase::Base2>(
        x, vec::get(tag, out_2, lane));
    expect_within_tier<Tier, vec::LogBase::Base10>(
        x, vec::get(tag, out_10, lane));
  }
}

template <vec::LogBase Base, vec::Accuracy Tier, typename T>
void run_log_special_values_for_tier() {
  using Tag = vec::ScalableTag<T>;
  const auto nan = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
  const auto infinity =
      static_cast<T>(std::numeric_limits<double>::infinity());
  const auto smallest = std::numeric_limits<T>::min();
  const auto largest = std::numeric_limits<T>::max();
  const auto smallest_subnormal =
      std::numeric_limits<T>::denorm_min();
  const auto invoke_check = []<typename Check>(T input, Check check) {
    const auto filled = vec::fill(Tag{}, input);
    const auto result = vec::LogCpo<Base>{}(
        Tag{}, filled, vec::opt::math::accuracy<Tier>);
    check(result);
  };
  // log(+-0) = -inf for every base.
  invoke_check(T(0), [&](auto result) {
    EXPECT_EQ(
        -std::numeric_limits<double>::infinity(),
        as_double(vec::get(Tag{}, result, 0)));
  });
  invoke_check(T(-0.0), [&](auto result) {
    EXPECT_EQ(
        -std::numeric_limits<double>::infinity(),
        as_double(vec::get(Tag{}, result, 0)));
  });
  // log(+inf) = +inf.
  invoke_check(infinity, [&](auto result) {
    EXPECT_EQ(
        std::numeric_limits<double>::infinity(),
        as_double(vec::get(Tag{}, result, 0)));
  });
  // NaN propagates.
  invoke_check(nan, [&](auto result) {
    EXPECT_TRUE(std::isnan(as_double(vec::get(Tag{}, result, 0))));
  });
  // Negative finite and -inf are NaN for every base.
  invoke_check(T(-4), [&](auto result) {
    EXPECT_TRUE(std::isnan(as_double(vec::get(Tag{}, result, 0))));
  });
  invoke_check(-infinity, [&](auto result) {
    EXPECT_TRUE(std::isnan(as_double(vec::get(Tag{}, result, 0))));
  });
  // log(1) = +0 exactly.
  invoke_check(T(1), [&](auto result) {
    EXPECT_TRUE(vec_test::values_identical(
        T(0), vec::get(Tag{}, result, 0)));
  });
  // Powers of the base are exact integers (where representable).
  invoke_check(T(4), [&](auto result) {
    if constexpr (Base == vec::LogBase::Base2) {
      EXPECT_TRUE(vec_test::values_identical(
          T(2), vec::get(Tag{}, result, 0)));
    } else {
      expect_within_tier<Tier, Base>(
          T(4), vec::get(Tag{}, result, 0));
    }
  });
  // Subnormal inputs and boundary normals follow the tier contract.
  invoke_check(smallest_subnormal, [&](auto result) {
    expect_within_tier<Tier, Base>(
        smallest_subnormal, vec::get(Tag{}, result, 0));
  });
  invoke_check(smallest, [&](auto result) {
    expect_within_tier<Tier, Base>(
        smallest, vec::get(Tag{}, result, 0));
  });
  invoke_check(largest, [&](auto result) {
    expect_within_tier<Tier, Base>(
        largest, vec::get(Tag{}, result, 0));
  });
}

template <typename T>
void run_log_special_values_test() {
  for (const auto tier : {0, 1, 2}) {
    if (tier == 0) {
      run_log_special_values_for_tier<vec::LogBase::E, vec::Accuracy::Strict,
                                      T>();
      run_log_special_values_for_tier<vec::LogBase::Base2,
                                      vec::Accuracy::Strict, T>();
      run_log_special_values_for_tier<vec::LogBase::Base10,
                                      vec::Accuracy::Strict, T>();
    } else if (tier == 1) {
      run_log_special_values_for_tier<vec::LogBase::E, vec::Accuracy::Fast,
                                      T>();
      run_log_special_values_for_tier<vec::LogBase::Base2,
                                      vec::Accuracy::Fast, T>();
      run_log_special_values_for_tier<vec::LogBase::Base10,
                                      vec::Accuracy::Fast, T>();
    } else {
      run_log_special_values_for_tier<vec::LogBase::E,
                                      vec::Accuracy::Estimate, T>();
      run_log_special_values_for_tier<vec::LogBase::Base2,
                                      vec::Accuracy::Estimate, T>();
      run_log_special_values_for_tier<vec::LogBase::Base10,
                                      vec::Accuracy::Estimate, T>();
    }
  }
}

template <typename T>
void run_log_fixed_cpo_test() {
  using Tag = vec::ScalableTag<T>;
  const Tag tag{};
  const auto v = vec::fill(tag, T(7));
  {
    const auto strict = vec::log(tag, v, vec::opt::math::strict);
    const auto fast = vec::log(tag, v, vec::opt::math::fast);
    const auto est = vec::log(tag, v, vec::opt::math::estimate);
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, strict, 0), vec::get(tag, vec::log_strict(tag, v), 0)));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, fast, 0), vec::get(tag, vec::log_fast(tag, v), 0)));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, est, 0), vec::get(tag, vec::log_est(tag, v), 0)));
  }
  {
    const auto strict = vec::log2(tag, v, vec::opt::math::strict);
    const auto fast = vec::log2(tag, v, vec::opt::math::fast);
    const auto est = vec::log2(tag, v, vec::opt::math::estimate);
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, strict, 0),
        vec::get(tag, vec::log2_strict(tag, v), 0)));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, fast, 0), vec::get(tag, vec::log2_fast(tag, v), 0)));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, est, 0), vec::get(tag, vec::log2_est(tag, v), 0)));
  }
  {
    const auto strict = vec::log10(tag, v, vec::opt::math::strict);
    const auto fast = vec::log10(tag, v, vec::opt::math::fast);
    const auto est = vec::log10(tag, v, vec::opt::math::estimate);
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, strict, 0),
        vec::get(tag, vec::log10_strict(tag, v), 0)));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, fast, 0), vec::get(tag, vec::log10_fast(tag, v), 0)));
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, est, 0), vec::get(tag, vec::log10_est(tag, v), 0)));
  }
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

template <typename T>
void run_fixed_log_test() {
  if constexpr (!std::same_as<T, vecops::float32_t>) {
    GTEST_SKIP() << "f32 is the representative >4-word floating type";
  } else {
    constexpr vecops::nint_t word_lanes =
        FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
    constexpr vecops::nint_t lanes = static_cast<vecops::nint_t>(
        std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
    using Tag = vec::FixedTag<T, lanes>;
    EXPECT_GT(vec::num_words(Tag{}), 4);
    verify_log_shapes<vec::LogBase::E, false>(Tag{});
    verify_log_shapes<vec::LogBase::Base2, false>(Tag{});
    verify_log_shapes<vec::LogBase::Base10, false>(Tag{});
  }
}

#endif

template <typename T>
class VecLogTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecLogTest,
    vec_test::FloatingElementTypes,
    vec_test::ElementTypeName);

TYPED_TEST(VecLogTest, CoversEveryLaneShapeAndPopulation) {
  run_log_test<TypeParam>();
}

TYPED_TEST(VecLogTest, StrictAccuracyHoldsOnMixedMagnitudes) {
  run_log_accuracy_test<vec::Accuracy::Strict, TypeParam>();
}

TYPED_TEST(VecLogTest, FastAccuracyHoldsOnMixedMagnitudes) {
  run_log_accuracy_test<vec::Accuracy::Fast, TypeParam>();
}

TYPED_TEST(VecLogTest, EstimateAccuracyHoldsOnMixedMagnitudes) {
  run_log_accuracy_test<vec::Accuracy::Estimate, TypeParam>();
}

TYPED_TEST(VecLogTest, PreservesIEEESpecialValueClasses) {
  run_log_special_values_test<TypeParam>();
}

TYPED_TEST(VecLogTest, FixedAccuracyCposMatchCanonicalForms) {
  run_log_fixed_cpo_test<TypeParam>();
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
TYPED_TEST(VecLogTest, FixedSVEBatchesBeyondTupleLimit) {
  run_fixed_log_test<TypeParam>();
}
#endif
