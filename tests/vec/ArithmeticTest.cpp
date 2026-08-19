#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "vecops/vec/Arithmetic.h"
#include "vecops/vec/Memory.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

template <typename T>
class VecArithmeticElementTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecArithmeticElementTest, vec_test::AllElementTypes);

template <typename T>
T arithmetic_operand(vecops::nint_t lane, bool rhs) {
  if constexpr (
      std::same_as<T, vecops::bfloat16_t> ||
      std::same_as<T, vecops::float16_t> ||
      std::is_floating_point_v<T>) {
    const float value = rhs
        ? static_cast<float>((lane * 3 + 5) % 13) * 0.25F
        : static_cast<float>((lane * 5 + 2) % 17) - 8.0F;
    return static_cast<T>(value);
  } else {
    using U = std::make_unsigned_t<T>;
    const U bits = rhs
        ? static_cast<U>(lane * 29 + 7)
        : static_cast<U>(std::numeric_limits<U>::max() -
                         static_cast<U>(lane * 13));
    if constexpr (std::is_signed_v<T>) {
      return ::vecops::bitcast<T>(bits);
    } else {
      return bits;
    }
  }
}

enum class ArithmeticKind { Add, Sub, Mul, Div };
enum class ExtremaKind { Min, Max };
enum class FmaKind { Fmadd, Fmsub, Fnmadd, Fnmsub };

template <ArithmeticKind Kind, typename T>
T expected_arithmetic(T a, T b) {
  if constexpr (
      std::same_as<T, vecops::bfloat16_t> ||
      std::same_as<T, vecops::float16_t> ||
      std::is_floating_point_v<T>) {
    if constexpr (Kind == ArithmeticKind::Add) return static_cast<T>(a + b);
    else if constexpr (Kind == ArithmeticKind::Sub) return static_cast<T>(a - b);
    else if constexpr (Kind == ArithmeticKind::Mul) return static_cast<T>(a * b);
    else return static_cast<T>(a / b);
  } else {
    using U = std::make_unsigned_t<T>;
    const U unsigned_a = [&] {
      if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(a);
      else return static_cast<U>(a);
    }();
    const U unsigned_b = [&] {
      if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(b);
      else return static_cast<U>(b);
    }();
    const U result = [&] {
      if constexpr (Kind == ArithmeticKind::Add) {
        return static_cast<U>(unsigned_a + unsigned_b);
      } else if constexpr (Kind == ArithmeticKind::Sub) {
        return static_cast<U>(unsigned_a - unsigned_b);
      } else {
        return static_cast<U>(unsigned_a * unsigned_b);
      }
    }();
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(result);
    else return result;
  }
}

template <ArithmeticKind Kind, bool FullOptions = true, vec::VectorTag Tag>
void verify_arithmetic(Tag tag) {
  using T = vec::ElementOf<Tag>;
  std::vector<T> a_lanes(static_cast<std::size_t>(vec::size(tag)));
  std::vector<T> b_lanes(static_cast<std::size_t>(vec::size(tag)));
  std::vector<T> merge_lanes(static_cast<std::size_t>(vec::size(tag)));
  auto mask = vec::mfill(tag, false);
  const T scalar_merge = [] {
    if constexpr (
        std::same_as<T, vecops::bfloat16_t> ||
        std::same_as<T, vecops::float16_t> ||
        std::is_floating_point_v<T>) {
      return static_cast<T>(-2.5F);
    } else {
      return static_cast<T>(37);
    }
  }();
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    a_lanes[static_cast<std::size_t>(lane)] =
        arithmetic_operand<T>(lane, false);
    b_lanes[static_cast<std::size_t>(lane)] =
        arithmetic_operand<T>(lane, true);
    merge_lanes[static_cast<std::size_t>(lane)] =
        arithmetic_operand<T>(lane + 11, true);
    mask = vec::set(tag, mask, lane, (lane % 3) != 1);
  }
  const auto a = vec::load(tag, a_lanes.data());
  const auto b = vec::load(tag, b_lanes.data());
  const auto vector_merge = vec::load(tag, merge_lanes.data());

  const auto result = [&] {
    if constexpr (Kind == ArithmeticKind::Add) return vec::add(a, b);
    else if constexpr (Kind == ArithmeticKind::Sub) return vec::sub(a, b);
    else if constexpr (Kind == ArithmeticKind::Mul) return vec::mul(a, b);
    else return vec::div(a, b);
  }();
  const auto unmasked_result = [&] {
    if constexpr (Kind == ArithmeticKind::Add)
      return vec::add(a, b, vec::opt::unmasked);
    else if constexpr (Kind == ArithmeticKind::Sub)
      return vec::sub(a, b, vec::opt::unmasked);
    else if constexpr (Kind == ArithmeticKind::Mul)
      return vec::mul(a, b, vec::opt::unmasked);
    else
      return vec::div(a, b, vec::opt::unmasked);
  }();
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T expected = expected_arithmetic<Kind>(
        arithmetic_operand<T>(lane, false),
        arithmetic_operand<T>(lane, true));
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, result, lane)))
        << "lane=" << lane << ", words=" << vec::num_words(tag);
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, result, lane),
        vec::get(tag, unmasked_result, lane)))
        << "explicit unmasked lane=" << lane;
  }

  if constexpr (FullOptions) {
    const auto masked_default = [&] {
      if constexpr (Kind == ArithmeticKind::Add) return vec::add(a, b, vec::opt::masked(mask));
      else if constexpr (Kind == ArithmeticKind::Sub) return vec::sub(a, b, vec::opt::masked(mask));
      else if constexpr (Kind == ArithmeticKind::Mul) return vec::mul(a, b, vec::opt::masked(mask));
      else return vec::div(a, b, vec::opt::masked(mask));
    }();
    const auto masked_zero = [&] {
      if constexpr (Kind == ArithmeticKind::Add) return vec::add(a, b, vec::opt::masked(mask), vec::opt::zero);
      else if constexpr (Kind == ArithmeticKind::Sub) return vec::sub(a, b, vec::opt::masked(mask), vec::opt::zero);
      else if constexpr (Kind == ArithmeticKind::Mul) return vec::mul(a, b, vec::opt::masked(mask), vec::opt::zero);
      else return vec::div(a, b, vec::opt::masked(mask), vec::opt::zero);
    }();
    const auto masked_scalar_merge = [&] {
      if constexpr (Kind == ArithmeticKind::Add) return vec::add(a, b, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
      else if constexpr (Kind == ArithmeticKind::Sub) return vec::sub(a, b, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
      else if constexpr (Kind == ArithmeticKind::Mul) return vec::mul(a, b, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
      else return vec::div(a, b, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
    }();
    const auto masked_vector_merge = [&] {
      if constexpr (Kind == ArithmeticKind::Add) return vec::add(a, b, vec::opt::masked(mask), vec::opt::merge(vector_merge));
      else if constexpr (Kind == ArithmeticKind::Sub) return vec::sub(a, b, vec::opt::masked(mask), vec::opt::merge(vector_merge));
      else if constexpr (Kind == ArithmeticKind::Mul) return vec::mul(a, b, vec::opt::masked(mask), vec::opt::merge(vector_merge));
      else return vec::div(a, b, vec::opt::masked(mask), vec::opt::merge(vector_merge));
    }();
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T expected = expected_arithmetic<Kind>(
          arithmetic_operand<T>(lane, false),
          arithmetic_operand<T>(lane, true));
      const bool active = (lane % 3) != 1;
      const T default_expected = active
          ? expected
          : arithmetic_operand<T>(lane, false);
      EXPECT_TRUE(vec_test::values_identical(
          default_expected, vec::get(tag, masked_default, lane)))
          << "default masked lane=" << lane << ", words=" << vec::num_words(tag);
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : T{}, vec::get(tag, masked_zero, lane)))
          << "zero masked lane=" << lane << ", words=" << vec::num_words(tag);
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : scalar_merge,
          vec::get(tag, masked_scalar_merge, lane)))
          << "scalar merge lane=" << lane << ", words=" << vec::num_words(tag);
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : arithmetic_operand<T>(lane + 11, true),
          vec::get(tag, masked_vector_merge, lane)))
          << "vector merge lane=" << lane << ", words=" << vec::num_words(tag);
    }
  }
}

template <typename T>
T extrema_operand(vecops::nint_t lane, bool rhs) {
  if constexpr (::vecops::IsFloatV<T> || std::is_signed_v<T>) {
    const int base = static_cast<int>(lane % 11) - 5;
    return static_cast<T>(rhs
        ? base + ((lane % 2) == 0 ? 7 : -7)
        : base);
  } else {
    const unsigned base = 20U + static_cast<unsigned>(lane % 11);
    if (!rhs) return static_cast<T>(base);
    return static_cast<T>((lane % 2) == 0 ? base + 7U : base - 7U);
  }
}

template <ExtremaKind Kind, typename T>
T expected_extrema(T a, T b) {
  if constexpr (Kind == ExtremaKind::Min) return a < b ? a : b;
  else return a > b ? a : b;
}

template <ExtremaKind Kind, vec::VectorTag Tag, typename... Options>
vec::Vec<Tag> invoke_extrema(
    Tag tag, vec::Vec<Tag> a, vec::Vec<Tag> b, Options&&... options) {
  if constexpr (Kind == ExtremaKind::Min)
    return vec::min(a, b, std::forward<Options>(options)...);
  else
    return vec::max(a, b, std::forward<Options>(options)...);
}

template <ExtremaKind Kind, bool FullOptions = true, vec::VectorTag Tag>
void verify_extrema(Tag tag) {
  using T = vec::ElementOf<Tag>;
  std::vector<T> a_lanes(static_cast<std::size_t>(vec::size(tag)));
  std::vector<T> b_lanes(static_cast<std::size_t>(vec::size(tag)));
  std::vector<T> merge_lanes(static_cast<std::size_t>(vec::size(tag)));
  auto mask = vec::mfalse(tag);
  const T scalar_merge = static_cast<T>(9);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    a_lanes[static_cast<std::size_t>(lane)] =
        extrema_operand<T>(lane, false);
    b_lanes[static_cast<std::size_t>(lane)] =
        extrema_operand<T>(lane, true);
    merge_lanes[static_cast<std::size_t>(lane)] =
        extrema_operand<T>(lane + 17, true);
    mask = vec::set(tag, mask, lane, (lane % 3) != 0);
  }
  const auto a = vec::load(tag, a_lanes.data());
  const auto b = vec::load(tag, b_lanes.data());
  const auto vector_merge = vec::load(tag, merge_lanes.data());

  const auto result = invoke_extrema<Kind>(tag, a, b);
  const auto unmasked_result =
      invoke_extrema<Kind>(tag, a, b, vec::opt::unmasked);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T expected = expected_extrema<Kind>(
        extrema_operand<T>(lane, false), extrema_operand<T>(lane, true));
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, result, lane))) << "lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, result, lane),
        vec::get(tag, unmasked_result, lane)))
        << "explicit unmasked lane=" << lane;
  }

  if constexpr (FullOptions) {
    const auto masked_default = invoke_extrema<Kind>(
        tag, a, b, vec::opt::masked(mask));
    const auto masked_zero = invoke_extrema<Kind>(
        tag, a, b, vec::opt::masked(mask), vec::opt::zero);
    const auto masked_scalar = invoke_extrema<Kind>(
        tag, a, b, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
    const auto masked_vector = invoke_extrema<Kind>(
        tag, a, b, vec::opt::masked(mask), vec::opt::merge(vector_merge));

    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T a_lane = extrema_operand<T>(lane, false);
      const T expected = expected_extrema<Kind>(
          a_lane, extrema_operand<T>(lane, true));
      const bool active = (lane % 3) != 0;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : a_lane,
          vec::get(tag, masked_default, lane))) << "default lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : T{},
          vec::get(tag, masked_zero, lane))) << "zero lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : scalar_merge,
          vec::get(tag, masked_scalar, lane))) << "scalar lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : extrema_operand<T>(lane + 17, true),
          vec::get(tag, masked_vector, lane))) << "vector lane=" << lane;
    }
  }
}

template <FmaKind Kind, typename T>
T expected_fma(T a, T b, T c) {
  if constexpr (::vecops::IsFloatV<T>) {
    if constexpr (Kind == FmaKind::Fmadd)
      return static_cast<T>(a * b + c);
    else if constexpr (Kind == FmaKind::Fmsub)
      return static_cast<T>(a * b - c);
    else if constexpr (Kind == FmaKind::Fnmadd)
      return static_cast<T>(-(a * b) + c);
    else
      return static_cast<T>(-(a * b) - c);
  } else {
    using U = std::make_unsigned_t<T>;
    const auto bits = [](T value) {
      if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(value);
      else return static_cast<U>(value);
    };
    const U product = static_cast<U>(bits(a) * bits(b));
    const U result = [&] {
      if constexpr (Kind == FmaKind::Fmadd)
        return static_cast<U>(product + bits(c));
      else if constexpr (Kind == FmaKind::Fmsub)
        return static_cast<U>(product - bits(c));
      else if constexpr (Kind == FmaKind::Fnmadd)
        return static_cast<U>(bits(c) - product);
      else
        return static_cast<U>(U{} - product - bits(c));
    }();
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(result);
    else return result;
  }
}

template <FmaKind Kind, vec::VectorTag Tag, typename... Options>
vec::Vec<Tag> invoke_fma(
    Tag tag, vec::Vec<Tag> a, vec::Vec<Tag> b, vec::Vec<Tag> c,
    Options&&... options) {
  if constexpr (Kind == FmaKind::Fmadd)
    return vec::fmadd(a, b, c, std::forward<Options>(options)...);
  else if constexpr (Kind == FmaKind::Fmsub)
    return vec::fmsub(a, b, c, std::forward<Options>(options)...);
  else if constexpr (Kind == FmaKind::Fnmadd)
    return vec::fnmadd(a, b, c, std::forward<Options>(options)...);
  else
    return vec::fnmsub(a, b, c, std::forward<Options>(options)...);
}

template <FmaKind Kind, bool FullOptions = true, vec::VectorTag Tag>
void verify_fma(Tag tag) {
  using T = vec::ElementOf<Tag>;
  std::vector<T> a_lanes(static_cast<std::size_t>(vec::size(tag)));
  std::vector<T> b_lanes(static_cast<std::size_t>(vec::size(tag)));
  std::vector<T> c_lanes(static_cast<std::size_t>(vec::size(tag)));
  std::vector<T> merge_lanes(static_cast<std::size_t>(vec::size(tag)));
  auto mask = vec::mfalse(tag);
  const T scalar_merge = static_cast<T>(3);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    a_lanes[static_cast<std::size_t>(lane)] =
        arithmetic_operand<T>(lane, false);
    b_lanes[static_cast<std::size_t>(lane)] =
        arithmetic_operand<T>(lane, true);
    c_lanes[static_cast<std::size_t>(lane)] =
        arithmetic_operand<T>(lane + 7, true);
    merge_lanes[static_cast<std::size_t>(lane)] =
        arithmetic_operand<T>(lane + 13, false);
    mask = vec::set(tag, mask, lane, (lane % 3) != 0);
  }
  const auto a = vec::load(tag, a_lanes.data());
  const auto b = vec::load(tag, b_lanes.data());
  const auto c = vec::load(tag, c_lanes.data());
  const auto vector_merge = vec::load(tag, merge_lanes.data());

  const auto result = invoke_fma<Kind>(tag, a, b, c);
  const auto unmasked_result =
      invoke_fma<Kind>(tag, a, b, c, vec::opt::unmasked);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T expected = expected_fma<Kind>(
        arithmetic_operand<T>(lane, false), arithmetic_operand<T>(lane, true),
        arithmetic_operand<T>(lane + 7, true));
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(tag, result, lane))) << "lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        vec::get(tag, result, lane),
        vec::get(tag, unmasked_result, lane)))
        << "explicit unmasked lane=" << lane;
  }

  if constexpr (FullOptions) {
    const auto masked_default = invoke_fma<Kind>(
        tag, a, b, c, vec::opt::masked(mask));
    const auto masked_zero = invoke_fma<Kind>(
        tag, a, b, c, vec::opt::masked(mask), vec::opt::zero);
    const auto masked_scalar = invoke_fma<Kind>(
        tag, a, b, c, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
    const auto masked_vector = invoke_fma<Kind>(
        tag, a, b, c, vec::opt::masked(mask), vec::opt::merge(vector_merge));

    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T a_lane = arithmetic_operand<T>(lane, false);
      const T expected = expected_fma<Kind>(
          a_lane, arithmetic_operand<T>(lane, true),
          arithmetic_operand<T>(lane + 7, true));
      const bool active = (lane % 3) != 0;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : a_lane,
          vec::get(tag, masked_default, lane))) << "default lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : T{},
          vec::get(tag, masked_zero, lane))) << "zero lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : scalar_merge,
          vec::get(tag, masked_scalar, lane))) << "scalar lane=" << lane;
      EXPECT_TRUE(vec_test::values_identical(
          active ? expected : arithmetic_operand<T>(lane + 13, false),
          vec::get(tag, masked_vector, lane))) << "vector lane=" << lane;
    }
  }
}

TYPED_TEST(VecArithmeticElementTest, OperationsCoverEveryLaneAndScalableShape) {
  using T = TypeParam;
  vec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    constexpr bool options = vec_test::exhaustive_options_shape<Tag>;
    verify_arithmetic<ArithmeticKind::Add, options>(Tag{});
    verify_arithmetic<ArithmeticKind::Sub, options>(Tag{});
    verify_arithmetic<ArithmeticKind::Mul, options>(Tag{});
    if constexpr (::vecops::IsFloatV<T>)
      verify_arithmetic<ArithmeticKind::Div, options>(Tag{});
    verify_extrema<ExtremaKind::Min, options>(Tag{});
    verify_extrema<ExtremaKind::Max, options>(Tag{});
    verify_fma<FmaKind::Fmadd, options>(Tag{});
    verify_fma<FmaKind::Fmsub, options>(Tag{});
    verify_fma<FmaKind::Fnmadd, options>(Tag{});
    verify_fma<FmaKind::Fnmsub, options>(Tag{});
  });
}

TYPED_TEST(VecArithmeticElementTest, FloatingDivisionByZeroAndNaN) {
  using T = TypeParam;
  if constexpr (!::vecops::IsFloatV<T>) {
    GTEST_SKIP() << "floating-point edge case";
  } else {
    using Tag = vec::ScalableTag<T>;
    const auto one = vec::fill(Tag{}, static_cast<T>(1));
    const auto negative_one = vec::fill(Tag{}, static_cast<T>(-1));
    const auto zero = vec::fill(Tag{}, static_cast<T>(0.0));
    const auto nan = vec::fill(
        Tag{}, static_cast<T>(std::numeric_limits<double>::quiet_NaN()));
    const double positive = static_cast<double>(
        vec::get(Tag{}, vec::div(one, zero), 0));
    const double negative = static_cast<double>(
        vec::get(Tag{}, vec::div(negative_one, zero), 0));
    EXPECT_TRUE(std::isinf(positive));
    EXPECT_FALSE(std::signbit(positive));
    EXPECT_TRUE(std::isinf(negative));
    EXPECT_TRUE(std::signbit(negative));
    EXPECT_TRUE(std::isnan(static_cast<double>(
        vec::get(Tag{}, vec::div(zero, zero), 0))));
    EXPECT_TRUE(std::isnan(static_cast<double>(
        vec::get(Tag{}, vec::div(nan, one), 0))));
  }
}

TYPED_TEST(VecArithmeticElementTest, ExtremaRetainBackendFloatingSpecialValues) {
  using T = TypeParam;
  if constexpr (::vecops::IsFloatV<T>) {
    using Tag = vec::ScalableTag<T, 0>;
    const Tag tag{};
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T one = static_cast<T>(1);
    const T positive_zero = static_cast<T>(0.0F);
    const T negative_zero = static_cast<T>(-0.0F);
    const auto nan_vector = vec::fill(tag, nan);
    const auto one_vector = vec::fill(tag, one);
    const auto positive_zero_vector = vec::fill(tag, positive_zero);
    const auto negative_zero_vector = vec::fill(tag, negative_zero);

    const auto min_nan_first = vec::get(
        tag, vec::min(nan_vector, one_vector), 0);
    const auto min_nan_second = vec::get(
        tag, vec::min(one_vector, nan_vector), 0);
    const auto max_nan_first = vec::get(
        tag, vec::max(nan_vector, one_vector), 0);
    const auto max_nan_second = vec::get(
        tag, vec::max(one_vector, nan_vector), 0);

#if defined(CPU_CAPABILITY_GENERIC)
    EXPECT_TRUE(std::isnan(static_cast<double>(min_nan_first)));
    EXPECT_TRUE(vec_test::values_identical(one, min_nan_second));
    EXPECT_TRUE(std::isnan(static_cast<double>(max_nan_first)));
    EXPECT_TRUE(vec_test::values_identical(one, max_nan_second));
#elif defined(ARCH_X86_FAMILY)
    EXPECT_TRUE(vec_test::values_identical(one, min_nan_first));
    EXPECT_TRUE(std::isnan(static_cast<double>(min_nan_second)));
    EXPECT_TRUE(vec_test::values_identical(one, max_nan_first));
    EXPECT_TRUE(std::isnan(static_cast<double>(max_nan_second)));
#elif defined(CPU_CAPABILITY_SVE)
    EXPECT_TRUE(std::isnan(static_cast<double>(min_nan_first)));
    EXPECT_TRUE(std::isnan(static_cast<double>(min_nan_second)));
    EXPECT_TRUE(std::isnan(static_cast<double>(max_nan_first)));
    EXPECT_TRUE(std::isnan(static_cast<double>(max_nan_second)));
#endif

    const auto min_positive_negative = vec::get(
        tag, vec::min(positive_zero_vector, negative_zero_vector), 0);
    const auto min_negative_positive = vec::get(
        tag, vec::min(negative_zero_vector, positive_zero_vector), 0);
    const auto max_positive_negative = vec::get(
        tag, vec::max(positive_zero_vector, negative_zero_vector), 0);
    const auto max_negative_positive = vec::get(
        tag, vec::max(negative_zero_vector, positive_zero_vector), 0);

#if defined(CPU_CAPABILITY_GENERIC)
    EXPECT_FALSE(std::signbit(static_cast<double>(min_positive_negative)));
    EXPECT_TRUE(std::signbit(static_cast<double>(min_negative_positive)));
    EXPECT_FALSE(std::signbit(static_cast<double>(max_positive_negative)));
    EXPECT_TRUE(std::signbit(static_cast<double>(max_negative_positive)));
#elif defined(ARCH_X86_FAMILY)
    EXPECT_TRUE(std::signbit(static_cast<double>(min_positive_negative)));
    EXPECT_FALSE(std::signbit(static_cast<double>(min_negative_positive)));
    EXPECT_TRUE(std::signbit(static_cast<double>(max_positive_negative)));
    EXPECT_FALSE(std::signbit(static_cast<double>(max_negative_positive)));
#elif defined(CPU_CAPABILITY_SVE)
    EXPECT_TRUE(std::signbit(static_cast<double>(min_positive_negative)));
    EXPECT_TRUE(std::signbit(static_cast<double>(min_negative_positive)));
    EXPECT_FALSE(std::signbit(static_cast<double>(max_positive_negative)));
    EXPECT_FALSE(std::signbit(static_cast<double>(max_negative_positive)));
#endif
  }
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

TYPED_TEST(VecArithmeticElementTest, FixedSVEBatchesBeyondTupleLimit) {
  using T = TypeParam;
  if constexpr (
      !std::same_as<T, vecops::float32_t> &&
      !std::same_as<T, vecops::int32_t>) {
    GTEST_SKIP() << "f32 and i32 are the representative >4-word types";
  } else {
    constexpr vecops::nint_t word_lanes =
        FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
    constexpr vecops::nint_t array_lanes = static_cast<vecops::nint_t>(
        std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
    using Tag = vec::FixedTag<T, array_lanes>;

    EXPECT_GT(vec::num_words(Tag{}), 4);
    verify_arithmetic<ArithmeticKind::Add>(Tag{});
    verify_arithmetic<ArithmeticKind::Sub, false>(Tag{});
    verify_arithmetic<ArithmeticKind::Mul, false>(Tag{});
    if constexpr (::vecops::IsFloatV<T>) {
      verify_arithmetic<ArithmeticKind::Div, false>(Tag{});
    }
    verify_extrema<ExtremaKind::Min>(Tag{});
    verify_extrema<ExtremaKind::Max, false>(Tag{});
    verify_fma<FmaKind::Fmadd>(Tag{});
    verify_fma<FmaKind::Fmsub, false>(Tag{});
    verify_fma<FmaKind::Fnmadd, false>(Tag{});
    verify_fma<FmaKind::Fnmsub, false>(Tag{});
  }
}

#endif
