// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "vecops/vec/Vec.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

namespace {

template <typename T>
double as_double(T value) {
  return static_cast<double>(value);
}

template <typename T>
using UIntOf = std::conditional_t<
    sizeof(T) == 8, std::uint64_t,
    std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint16_t>>;

template <typename T>
UIntOf<T> raw_bits(T value) {
  if constexpr (sizeof(T) <= 2) return value.to_bits();
  else {
    UIntOf<T> result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }
}

template <typename T>
std::uint64_t ordered_bits(T value) {
  using U = UIntOf<T>;
  constexpr U sign = U{1} << (sizeof(T) * 8 - 1);
  const U bits = raw_bits(value);
  return (bits & sign) ? static_cast<std::uint64_t>(static_cast<U>(~bits))
                       : static_cast<std::uint64_t>(bits | sign);
}

template <typename T>
std::uint64_t ulps_between(T expected, T actual) {
  const auto a = ordered_bits(expected);
  const auto b = ordered_bits(actual);
  return a > b ? a - b : b - a;
}

template <vec::TrigKind Kind, vec::TrigUnit Unit, typename T>
long double reference_wide(T input) {
  long double x = static_cast<long double>(as_double(input));
  if constexpr (Unit == vec::TrigUnit::Pi)
    x *= 3.141592653589793238462643383279502884L;
  if constexpr (Kind == vec::TrigKind::Sin) return std::sin(x);
  else if constexpr (Kind == vec::TrigKind::Cos) return std::cos(x);
  else return std::tan(x);
}

template <vec::TrigKind Kind, vec::TrigUnit Unit, typename T>
T reference(T input) {
  if constexpr (std::same_as<T, vecops::float64_t>)
    return static_cast<T>(reference_wide<Kind, Unit>(input));
  else
    return static_cast<T>(static_cast<float>(
        reference_wide<Kind, Unit>(input)));
}

template <vec::Accuracy Tier, vec::TrigKind Kind,
          vec::TrigUnit Unit, typename T>
void expect_within_contract(T input, T actual) {
  const T expected = reference<Kind, Unit>(input);
  ASSERT_EQ(std::isnan(as_double(expected)), std::isnan(as_double(actual)))
      << "x=" << as_double(input);
  if (std::isnan(as_double(expected))) return;
  ASSERT_EQ(std::isinf(as_double(expected)), std::isinf(as_double(actual)))
      << "x=" << as_double(input) << " actual=" << as_double(actual);
  if (std::isinf(as_double(expected))) {
    EXPECT_EQ(std::signbit(as_double(expected)),
              std::signbit(as_double(actual)));
    return;
  }
  if constexpr (Tier == vec::Accuracy::Strict) {
    const std::uint64_t bound = sizeof(T) <= 2 ? 1 : 4;
    EXPECT_LE(ulps_between(expected, actual), bound)
        << "x=" << as_double(input) << " expected=" << as_double(expected)
        << " actual=" << as_double(actual);
  } else {
    const long double epsilon = [] {
      if constexpr (Tier == vec::Accuracy::Fast) {
        if constexpr (std::same_as<T, vecops::bfloat16_t>) return 0x1p-4L;
        else if constexpr (std::same_as<T, vecops::float16_t>) return 0x1p-5L;
        else if constexpr (std::same_as<T, vecops::float32_t>) return 0x1p-12L;
        else return 0x1p-26L;
      } else {
        if constexpr (std::same_as<T, vecops::bfloat16_t>) return 0x1p-3L;
        else if constexpr (std::same_as<T, vecops::float16_t>) return 0x1p-4L;
        else if constexpr (std::same_as<T, vecops::float32_t>) return 0x1p-7L;
        else return 0x1p-13L;
      }
    }();
    const long double error = std::abs(
        static_cast<long double>(as_double(actual)) -
        reference_wide<Kind, Unit>(input));
    const long double scale = std::max(
        1.0L, std::abs(reference_wide<Kind, Unit>(input)));
    EXPECT_LE(error, epsilon * scale)
        << "x=" << as_double(input) << " actual=" << as_double(actual);
  }
}

template <vec::TrigKind Kind, vec::TrigUnit Unit,
          vec::Accuracy Tier, vec::FloatingTag Tag>
void exercise_unary(Tag tag, bool full_options) {
  using T = vec::ElementOf<Tag>;
  constexpr double radians[] = {
      -3.0, -2.25, -1.5, -0.75, -0.125, 0.0, 0.375, 1.0,
      1.625, 2.5, 3.0};
  constexpr double pi_units[] = {
      -1.375, -1.125, -0.875, -0.625, -0.375, -0.125,
      0.125, 0.375, 0.625, 0.875, 1.125, 1.375};
  auto input = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const double x = Unit == vec::TrigUnit::Pi
        ? pi_units[lane % std::size(pi_units)]
        : radians[lane % std::size(radians)];
    input = vec::set(tag, input, lane, static_cast<T>(x));
    mask = vec::set(tag, mask, lane, lane % 3 != 0);
  }

  const auto result = vec::TrigCpo<Kind, Unit>{}(
      tag, input, vec::opt::math::accuracy<Tier>);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    expect_within_contract<Tier, Kind, Unit>(
        vec::get(tag, input, lane), vec::get(tag, result, lane));

  if (full_options) {
    const auto masked = vec::TrigCpo<Kind, Unit>{}(
        tag, input, vec::opt::masked(mask), vec::opt::math::accuracy<Tier>);
    const auto zeroed = vec::TrigCpo<Kind, Unit>{}(
        tag, input, vec::opt::math::accuracy<Tier>,
        vec::opt::masked(mask), vec::opt::zero);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T x = vec::get(tag, input, lane);
      if (lane % 3 != 0) {
        expect_within_contract<Tier, Kind, Unit>(
            x, vec::get(tag, masked, lane));
        expect_within_contract<Tier, Kind, Unit>(
            x, vec::get(tag, zeroed, lane));
      } else {
        EXPECT_TRUE(vec_test::values_identical(
            x, vec::get(tag, masked, lane)));
        EXPECT_TRUE(vec_test::values_identical(
            T{}, vec::get(tag, zeroed, lane)));
      }
    }
  }
}

template <vec::TrigUnit Unit, vec::Accuracy Tier, vec::FloatingTag Tag>
void exercise_sincos(Tag tag, bool full_options) {
  using T = vec::ElementOf<Tag>;
  auto input = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const double x = Unit == vec::TrigUnit::Pi
        ? (static_cast<double>(lane % 11) - 5.0) * 0.1875
        : (static_cast<double>(lane % 13) - 6.0) * 0.375;
    input = vec::set(tag, input, lane, static_cast<T>(x));
    mask = vec::set(tag, mask, lane, lane % 3 != 0);
  }
  auto sin_out = vec::zeros(tag);
  auto cos_out = vec::zeros(tag);
  vec::SinCosCpo<Unit>{}(
      tag, input, sin_out, cos_out, vec::opt::math::accuracy<Tier>);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T x = vec::get(tag, input, lane);
    expect_within_contract<Tier, vec::TrigKind::Sin, Unit>(
        x, vec::get(tag, sin_out, lane));
    expect_within_contract<Tier, vec::TrigKind::Cos, Unit>(
        x, vec::get(tag, cos_out, lane));
  }
  if (full_options) {
    vec::SinCosCpo<Unit>{}(
        tag, input, sin_out, cos_out, vec::opt::masked(mask),
        vec::opt::math::accuracy<Tier>);
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T x = vec::get(tag, input, lane);
      if (lane % 3 != 0) {
        expect_within_contract<Tier, vec::TrigKind::Sin, Unit>(
            x, vec::get(tag, sin_out, lane));
        expect_within_contract<Tier, vec::TrigKind::Cos, Unit>(
            x, vec::get(tag, cos_out, lane));
      } else {
        EXPECT_TRUE(vec_test::values_identical(
            x, vec::get(tag, sin_out, lane)));
        EXPECT_TRUE(vec_test::values_identical(
            x, vec::get(tag, cos_out, lane)));
      }
    }
  }
}

template <vec::TrigKind Kind, vec::TrigUnit Unit, vec::FloatingTag Tag>
void exercise_unary_operation(Tag tag) {
  constexpr bool full = vec_test::exhaustive_options_shape<Tag>;
  const auto tiers = [&]<vec::Accuracy Tier>() {
    exercise_unary<Kind, Unit, Tier>(tag, full);
  };
  tiers.template operator()<vec::Accuracy::Strict>();
  tiers.template operator()<vec::Accuracy::Fast>();
  tiers.template operator()<vec::Accuracy::Estimate>();
}

template <vec::TrigUnit Unit, vec::FloatingTag Tag>
void exercise_fused_operation(Tag tag) {
  constexpr bool full = vec_test::exhaustive_options_shape<Tag>;
  exercise_sincos<Unit, vec::Accuracy::Strict>(tag, full);
  exercise_sincos<Unit, vec::Accuracy::Fast>(tag, full);
  exercise_sincos<Unit, vec::Accuracy::Estimate>(tag, full);
}

template <typename T>
void run_special_values() {
  using Tag = vec::ScalableTag<T>;
  const Tag tag{};
  const auto check = [&](auto op, T input) {
    return vec::get(tag, op(tag, vec::fill(tag, input)), 0);
  };
  const T pos_zero = T(0.0);
  const T neg_zero = T(-0.0);
  const T inf = static_cast<T>(std::numeric_limits<double>::infinity());
  const T nan = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(vec_test::values_identical(pos_zero, check(vec::sin, pos_zero)));
  EXPECT_TRUE(vec_test::values_identical(neg_zero, check(vec::sin, neg_zero)));
  EXPECT_TRUE(vec_test::values_identical(neg_zero, check(vec::tan, neg_zero)));
  EXPECT_TRUE(vec_test::values_identical(T(1), check(vec::cos, neg_zero)));
  EXPECT_TRUE(std::isnan(as_double(check(vec::sin, inf))));
  EXPECT_TRUE(std::isnan(as_double(check(vec::cos, inf))));
  EXPECT_TRUE(std::isnan(as_double(check(vec::tan, inf))));
  EXPECT_TRUE(std::isnan(as_double(check(vec::sinpi, nan))));

  EXPECT_TRUE(vec_test::values_identical(pos_zero, check(vec::sinpi, T(2))));
  EXPECT_TRUE(vec_test::values_identical(neg_zero, check(vec::sinpi, T(-2))));
  EXPECT_TRUE(vec_test::values_identical(T(-1), check(vec::cospi, T(1))));
  EXPECT_TRUE(vec_test::values_identical(T(1), check(vec::cospi, T(2))));
  EXPECT_TRUE(vec_test::values_identical(pos_zero, check(vec::cospi, T(0.5))));
  EXPECT_TRUE(vec_test::values_identical(pos_zero, check(vec::tanpi, T(2))));
  EXPECT_TRUE(vec_test::values_identical(neg_zero, check(vec::tanpi, T(1))));
  EXPECT_EQ(std::numeric_limits<double>::infinity(),
            as_double(check(vec::tanpi, T(0.5))));
  EXPECT_EQ(-std::numeric_limits<double>::infinity(),
            as_double(check(vec::tanpi, T(1.5))));
  EXPECT_TRUE(std::isnan(as_double(check(vec::sinpi, inf))));
  EXPECT_TRUE(std::isnan(as_double(check(vec::cospi, inf))));
  EXPECT_TRUE(std::isnan(as_double(check(vec::tanpi, inf))));
}

template <vec::Accuracy Tier>
void run_float16_exhaustive() {
  using T = vecops::float16_t;
  using Tag = vec::ScalableTag<T>;
  const Tag tag{};
  const auto lanes = vec::size(tag);
  for (std::uint32_t base = 0; base < 65536;
       base += static_cast<std::uint32_t>(lanes)) {
    auto input = vec::zeros(tag);
    for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
      const auto bits = static_cast<std::uint16_t>(
          std::min<std::uint32_t>(base + lane, 65535));
      input = vec::set(tag, input, lane, T::from_bits(bits));
    }
    const auto sin_out = vec::sin(
        tag, input, vec::opt::math::accuracy<Tier>);
    const auto cos_out = vec::cos(
        tag, input, vec::opt::math::accuracy<Tier>);
    const auto tan_out = vec::tan(
        tag, input, vec::opt::math::accuracy<Tier>);
    const auto sinpi_out = vec::sinpi(
        tag, input, vec::opt::math::accuracy<Tier>);
    const auto cospi_out = vec::cospi(
        tag, input, vec::opt::math::accuracy<Tier>);
    const auto tanpi_out = vec::tanpi(
        tag, input, vec::opt::math::accuracy<Tier>);
    auto fused_sin = vec::zeros(tag);
    auto fused_cos = vec::zeros(tag);
    vec::sincos(
        tag, input, fused_sin, fused_cos,
        vec::opt::math::accuracy<Tier>);
    auto fused_sinpi = vec::zeros(tag);
    auto fused_cospi = vec::zeros(tag);
    vec::sincospi(
        tag, input, fused_sinpi, fused_cospi,
        vec::opt::math::accuracy<Tier>);

    for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
      const T x = vec::get(tag, input, lane);
      expect_within_contract<
          Tier, vec::TrigKind::Sin, vec::TrigUnit::Radians>(
          x, vec::get(tag, sin_out, lane));
      expect_within_contract<
          Tier, vec::TrigKind::Cos, vec::TrigUnit::Radians>(
          x, vec::get(tag, cos_out, lane));
      expect_within_contract<
          Tier, vec::TrigKind::Tan, vec::TrigUnit::Radians>(
          x, vec::get(tag, tan_out, lane));
      expect_within_contract<
          Tier, vec::TrigKind::Sin, vec::TrigUnit::Radians>(
          x, vec::get(tag, fused_sin, lane));
      expect_within_contract<
          Tier, vec::TrigKind::Cos, vec::TrigUnit::Radians>(
          x, vec::get(tag, fused_cos, lane));

      const double xd = as_double(x);
      const double nearest = std::nearbyint(xd);
      const double fraction = std::abs(xd - nearest);
      // Integer and half-integer C23 results are checked bit-exactly by the
      // special-value suite. Skip them here because sin(pi*x) is not a valid
      // numerical reference for a mathematically exact zero or pole.
      if (std::isfinite(xd) && fraction != 0.0 && fraction != 0.5) {
        expect_within_contract<
            Tier, vec::TrigKind::Sin, vec::TrigUnit::Pi>(
            x, vec::get(tag, sinpi_out, lane));
        expect_within_contract<
            Tier, vec::TrigKind::Cos, vec::TrigUnit::Pi>(
            x, vec::get(tag, cospi_out, lane));
        expect_within_contract<
            Tier, vec::TrigKind::Tan, vec::TrigUnit::Pi>(
            x, vec::get(tag, tanpi_out, lane));
        expect_within_contract<
            Tier, vec::TrigKind::Sin, vec::TrigUnit::Pi>(
            x, vec::get(tag, fused_sinpi, lane));
        expect_within_contract<
            Tier, vec::TrigKind::Cos, vec::TrigUnit::Pi>(
            x, vec::get(tag, fused_cospi, lane));
      }
    }
  }
}

} // namespace

template <typename T>
class VecTrigTest : public ::testing::Test {};

TYPED_TEST_SUITE(
    VecTrigTest, vec_test::FloatingElementTypes,
    vec_test::ElementTypeName);

#if defined(VECOPS_TRIG_TEST_SIN)
TYPED_TEST(VecTrigTest, SinCoversEveryTierShapeAndPopulation) {
  exercise_unary_operation<
      vec::TrigKind::Sin, vec::TrigUnit::Radians>(
      vec::ScalableTag<TypeParam>{});
  vec_test::for_each_scalable_shape<TypeParam>([]<vec::FloatingTag Tag>() {
    // Accuracy tier selection is shape-independent. Keep the full tier and
    // population matrix on the native scalable shape above, and use Strict
    // as the word-batching/partial-word representative for every shape.
    exercise_unary<
        vec::TrigKind::Sin, vec::TrigUnit::Radians,
        vec::Accuracy::Strict>(Tag{}, false);
  });
}
#elif defined(VECOPS_TRIG_TEST_COS)
TYPED_TEST(VecTrigTest, CosCoversEveryTierAndPopulation) {
  exercise_unary_operation<
      vec::TrigKind::Cos, vec::TrigUnit::Radians>(
      vec::ScalableTag<TypeParam>{});
}
#elif defined(VECOPS_TRIG_TEST_TAN)
TYPED_TEST(VecTrigTest, TanCoversEveryTierAndPopulation) {
  exercise_unary_operation<
      vec::TrigKind::Tan, vec::TrigUnit::Radians>(
      vec::ScalableTag<TypeParam>{});
}
#elif defined(VECOPS_TRIG_TEST_SINPI)
TYPED_TEST(VecTrigTest, SinPiCoversEveryTierAndPopulation) {
  exercise_unary_operation<vec::TrigKind::Sin, vec::TrigUnit::Pi>(
      vec::ScalableTag<TypeParam>{});
}
#elif defined(VECOPS_TRIG_TEST_COSPI)
TYPED_TEST(VecTrigTest, CosPiCoversEveryTierAndPopulation) {
  exercise_unary_operation<vec::TrigKind::Cos, vec::TrigUnit::Pi>(
      vec::ScalableTag<TypeParam>{});
}
#elif defined(VECOPS_TRIG_TEST_TANPI)
TYPED_TEST(VecTrigTest, TanPiCoversEveryTierAndPopulation) {
  exercise_unary_operation<vec::TrigKind::Tan, vec::TrigUnit::Pi>(
      vec::ScalableTag<TypeParam>{});
}
#elif defined(VECOPS_TRIG_TEST_SINCOS)
TYPED_TEST(VecTrigTest, SinCosCoversEveryTierAndPopulation) {
  exercise_fused_operation<vec::TrigUnit::Radians>(
      vec::ScalableTag<TypeParam>{});
}
#elif defined(VECOPS_TRIG_TEST_SINCOSPI)
TYPED_TEST(VecTrigTest, SinCosPiCoversEveryTierAndMultiwordBatching) {
  exercise_fused_operation<vec::TrigUnit::Pi>(
      vec::ScalableTag<TypeParam>{});
  // The fused operation has a dedicated two-output batching path. One
  // two-word representative is sufficient because unary sin above already
  // covers every legal lane shape and partial-word predicate.
  exercise_sincos<vec::TrigUnit::Pi, vec::Accuracy::Strict>(
      vec::ScalableTag<TypeParam, 1>{}, false);
}
#elif defined(VECOPS_TRIG_TEST_SPECIAL)
TYPED_TEST(VecTrigTest, PreservesSpecialValuesAndPiLattice) {
  run_special_values<TypeParam>();
}
#elif defined(VECOPS_TRIG_TEST_F16_EXHAUSTIVE)
TEST(VecTrigFloat16ExhaustiveTest, FastAndEstimateMeetContracts) {
  run_float16_exhaustive<vec::Accuracy::Fast>();
  run_float16_exhaustive<vec::Accuracy::Estimate>();
}
#else
#error "one VECOPS_TRIG_TEST_* selector must be defined"
#endif
