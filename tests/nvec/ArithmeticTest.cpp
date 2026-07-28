#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "vecops/nvec/Arithmetic.h"
#include "TestHelpers.h"

namespace vec = vecops::nvec;

template <typename T>
class NVecArithmeticElementTest : public ::testing::Test {};

TYPED_TEST_SUITE(NVecArithmeticElementTest, nvec_test::AllElementTypes);

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

template <typename T>
T expected_add(T a, T b) {
  if constexpr (
      std::same_as<T, vecops::bfloat16_t> ||
      std::same_as<T, vecops::float16_t> ||
      std::is_floating_point_v<T>) {
    return static_cast<T>(a + b);
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
    const U sum = static_cast<U>(unsigned_a + unsigned_b);
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(sum);
    else return sum;
  }
}

template <vec::VectorTag Tag>
void verify_add(Tag tag) {
  using T = vec::ElementOf<Tag>;
  auto a = vec::zeros(tag);
  auto b = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    a = vec::set(tag, a, lane, arithmetic_operand<T>(lane, false));
    b = vec::set(tag, b, lane, arithmetic_operand<T>(lane, true));
  }

  const auto result = vec::add(tag, a, b);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T expected = expected_add(
        arithmetic_operand<T>(lane, false),
        arithmetic_operand<T>(lane, true));
    EXPECT_TRUE(nvec_test::values_identical(
        expected, vec::get(tag, result, lane)))
        << "lane=" << lane << ", words=" << vec::num_words(tag);
  }
}

TYPED_TEST(NVecArithmeticElementTest, AddCoversEveryLaneAndScalableShape) {
  using T = TypeParam;
  nvec_test::for_each_scalable_shape<T>([]<vec::VectorTag Tag>() {
    verify_add(Tag{});
  });
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

TYPED_TEST(NVecArithmeticElementTest, FixedSVEBatchesBeyondTupleLimit) {
  using T = TypeParam;
  constexpr vecops::nint_t word_lanes =
      FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
  constexpr vecops::nint_t array_lanes = static_cast<vecops::nint_t>(
      std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
  using Tag = vec::FixedTag<T, array_lanes>;

  EXPECT_GT(vec::num_words(Tag{}), 4);
  verify_add(Tag{});
}

#endif
