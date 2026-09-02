// @vecops-test-shards: 69

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <typeinfo>

#include "vecops/vec/Memory.h"
#include "vecops/vec/WideningDot.h"
#include "TestHelpers.h"
#include "TestShard.h"

namespace vec = vecops::vec;

template <int Index>
void run_widening_dot_case();

#if defined(VECOPS_TEST_SHARD_ACTIVE)

static_assert(VECOPS_TEST_SHARD_COUNT == 69);

namespace {

template <std::size_t Bytes, bool Signed>
using IntegerOf = std::conditional_t<
    Bytes == 1,
    std::conditional_t<Signed, vecops::int8_t, vecops::uint8_t>,
    std::conditional_t<
        Bytes == 2,
        std::conditional_t<Signed, vecops::int16_t, vecops::uint16_t>,
        std::conditional_t<
            Bytes == 4,
            std::conditional_t<Signed, vecops::int32_t, vecops::uint32_t>,
            std::conditional_t<Signed, vecops::int64_t, vecops::uint64_t>>>>;

template <typename T>
T source_value(vecops::nint_t lane, bool rhs) {
  if constexpr (::vecops::is_float_v<T>) {
    const int magnitude = static_cast<int>((lane * (rhs ? 3 : 5) + 1) % 7);
    return static_cast<T>((lane + (rhs ? 1 : 0)) % 3 == 0
                              ? -magnitude
                              : magnitude);
  } else if constexpr (std::signed_integral<T>) {
    if (lane == 0) return std::numeric_limits<T>::lowest();
    if (lane == 1) return std::numeric_limits<T>::max();
    const int value = static_cast<int>((lane * (rhs ? 13 : 9) + 5) % 31) - 15;
    return static_cast<T>(value);
  } else {
    if (lane == 0) return std::numeric_limits<T>::max();
    const auto value = static_cast<std::uint64_t>(
        lane * (rhs ? 37 : 29) + 11);
    return static_cast<T>(value);
  }
}

template <typename T>
T accumulator_value(vecops::nint_t lane) {
  if constexpr (::vecops::is_float_v<T>) {
    return static_cast<T>(static_cast<int>(lane % 5) - 2);
  } else if constexpr (std::signed_integral<T>) {
    if (lane == 0) return std::numeric_limits<T>::max();
    return static_cast<T>(static_cast<int>(lane * 7) - 19);
  } else {
    if (lane == 0) return std::numeric_limits<T>::max();
    return static_cast<T>(lane * 17 + 3);
  }
}

template <typename To, typename From>
std::make_unsigned_t<To> widened_integer_bits(From value) {
  using UTo = std::make_unsigned_t<To>;
  const To widened = static_cast<To>(value);
  if constexpr (std::signed_integral<To>) {
    return ::vecops::bitcast<UTo>(widened);
  } else {
    return widened;
  }
}

template <typename To, typename From1, typename From2>
To reference_dot_lane(vecops::nint_t output_lane, bool accumulating) {
  constexpr vecops::nint_t group = sizeof(To) == sizeof(From1)
      ? 1
      : static_cast<vecops::nint_t>(sizeof(To) / sizeof(From1));
  if constexpr (
      std::same_as<To, From1> && std::same_as<To, From2> &&
      ::vecops::is_float_v<To>) {
    const To a = source_value<From1>(output_lane, false);
    const To b = source_value<From2>(output_lane, true);
    if (!accumulating) return static_cast<To>(a * b);
  }
  if constexpr (std::integral<To>) {
    using UTo = std::make_unsigned_t<To>;
    UTo result = UTo{};
    if (accumulating) {
      const To initial = accumulator_value<To>(output_lane);
      if constexpr (std::signed_integral<To>)
        result = ::vecops::bitcast<UTo>(initial);
      else
        result = initial;
    }
    for (vecops::nint_t phase = 0; phase < group; ++phase) {
      const auto input_lane = output_lane * group + phase;
      const UTo a = widened_integer_bits<To>(
          source_value<From1>(input_lane, false));
      const UTo b = widened_integer_bits<To>(
          source_value<From2>(input_lane, true));
      result = static_cast<UTo>(result + static_cast<UTo>(a * b));
    }
    if constexpr (std::signed_integral<To>)
      return ::vecops::bitcast<To>(result);
    else
      return result;
  } else {
    To result = accumulating ? accumulator_value<To>(output_lane) : To{};
    for (vecops::nint_t phase = 0; phase < group; ++phase) {
      const auto input_lane = output_lane * group + phase;
      const To a = static_cast<To>(
          source_value<From1>(input_lane, false));
      const To b = static_cast<To>(
          source_value<From2>(input_lane, true));
      result = static_cast<To>(a * b + result);
    }
    return result;
  }
}

template <int Power, typename To, typename From1, typename From2>
void verify_widening_dot_shape_case() {
  using ToTag = vec::ScalableTag<To, Power>;
  using FromTag1 = std::conditional_t<
      std::same_as<To, From1>, ToTag, vec::ViewAs<From1, ToTag>>;
  using FromTag2 = std::conditional_t<
      std::same_as<To, From2>, ToTag, vec::ViewAs<From2, ToTag>>;
  constexpr bool exact =
      std::same_as<To, From1> && std::same_as<To, From2>;
  static_assert(exact || sizeof(From1) == sizeof(From2));
  static_assert(exact || sizeof(To) > sizeof(From1));

  SCOPED_TRACE(::testing::Message()
               << "To=" << typeid(To).name()
               << ", From1=" << typeid(From1).name()
               << ", From2=" << typeid(From2).name()
               << ", power=" << Power);

  auto a = vec::zeros(FromTag1{});
  auto b = vec::zeros(FromTag2{});
  auto c = vec::zeros(ToTag{});
  for (vecops::nint_t lane = 0; lane < vec::size(FromTag1{}); ++lane) {
    a = vec::set(
        FromTag1{}, a, lane, source_value<From1>(lane, false));
    b = vec::set(
        FromTag2{}, b, lane, source_value<From2>(lane, true));
  }
  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane)
    c = vec::set(ToTag{}, c, lane, accumulator_value<To>(lane));

  const auto explicit_result = vec::widening_dot(
      ToTag{}, FromTag1{}, FromTag2{}, a, b);
  const auto inferred_result = vec::widening_dot(ToTag{}, a, b);
  const auto explicit_accumulated = vec::widening_dot(
      ToTag{}, FromTag1{}, FromTag2{}, a, b, c);
  const auto inferred_accumulated = vec::widening_dot(ToTag{}, a, b, c);

  for (vecops::nint_t lane = 0; lane < vec::size(ToTag{}); ++lane) {
    const To expected = reference_dot_lane<To, From1, From2>(lane, false);
    const To expected_acc =
        reference_dot_lane<To, From1, From2>(lane, true);
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(ToTag{}, explicit_result, lane)))
        << "explicit lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected, vec::get(ToTag{}, inferred_result, lane)))
        << "inferred lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected_acc, vec::get(ToTag{}, explicit_accumulated, lane)))
        << "explicit accumulated lane=" << lane;
    EXPECT_TRUE(vec_test::values_identical(
        expected_acc, vec::get(ToTag{}, inferred_accumulated, lane)))
        << "inferred accumulated lane=" << lane;
  }
}

template <typename To, typename From1, typename From2>
void verify_widening_dot_type_case() {
  verify_widening_dot_shape_case<0, To, From1, From2>();
  verify_widening_dot_shape_case<1, To, From1, From2>();
}

template <int Index>
void run_floating_widening_case() {
  constexpr int choice = Index - 12;
  if constexpr (choice == 0)
    verify_widening_dot_type_case<
        vecops::float32_t, vecops::bfloat16_t, vecops::bfloat16_t>();
  else if constexpr (choice == 1)
    verify_widening_dot_type_case<
        vecops::float32_t, vecops::bfloat16_t, vecops::float16_t>();
  else if constexpr (choice == 2)
    verify_widening_dot_type_case<
        vecops::float32_t, vecops::float16_t, vecops::bfloat16_t>();
  else if constexpr (choice == 3)
    verify_widening_dot_type_case<
        vecops::float32_t, vecops::float16_t, vecops::float16_t>();
  else if constexpr (choice == 4)
    verify_widening_dot_type_case<
        vecops::float64_t, vecops::bfloat16_t, vecops::bfloat16_t>();
  else if constexpr (choice == 5)
    verify_widening_dot_type_case<
        vecops::float64_t, vecops::bfloat16_t, vecops::float16_t>();
  else if constexpr (choice == 6)
    verify_widening_dot_type_case<
        vecops::float64_t, vecops::float16_t, vecops::bfloat16_t>();
  else if constexpr (choice == 7)
    verify_widening_dot_type_case<
        vecops::float64_t, vecops::float16_t, vecops::float16_t>();
  else
    verify_widening_dot_type_case<
        vecops::float64_t, vecops::float32_t, vecops::float32_t>();
}

template <std::size_t FromBytes, int Outputs, int Choice>
void run_integer_width_case() {
  constexpr int pair = Choice / Outputs;
  constexpr int output = Choice % Outputs;
  using From1 = IntegerOf<FromBytes, (pair / 2) == 0>;
  using From2 = IntegerOf<FromBytes, (pair % 2) == 0>;
  constexpr std::size_t to_bytes = FromBytes << (output / 2 + 1);
  using To = IntegerOf<to_bytes, (output % 2) == 0>;
  verify_widening_dot_type_case<To, From1, From2>();
}

template <int Index>
void run_integer_widening_case() {
  constexpr int choice = Index - 21;
  if constexpr (choice < 24)
    run_integer_width_case<1, 6, choice>();
  else if constexpr (choice < 40)
    run_integer_width_case<2, 4, choice - 24>();
  else
    run_integer_width_case<4, 2, choice - 40>();
}

} // namespace

template <int Index>
void run_widening_dot_case() {
  static_assert(Index >= 0 && Index < 69);
  if constexpr (Index < 12) {
    using T = vec_test::ElementAt<Index>;
    verify_widening_dot_type_case<T, T, T>();
  } else if constexpr (Index < 21) {
    run_floating_widening_case<Index>();
  } else {
    run_integer_widening_case<Index>();
  }
}

template void run_widening_dot_case<VECOPS_TEST_SHARD_INDEX>();

#else

TEST(WideningDotTest, EveryLegalElementCombination) {
  []<std::size_t... Index>(std::index_sequence<Index...>) {
    (run_widening_dot_case<static_cast<int>(Index)>(), ...);
  }(std::make_index_sequence<69>{});
}

#endif
