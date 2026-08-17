#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "vecops/vec/Bit.h"
#include "TestHelpers.h"

namespace vec = vecops::vec;

using IntegerElementTypes = ::testing::Types<
    vecops::int8_t, vecops::uint8_t,
    vecops::int16_t, vecops::uint16_t,
    vecops::int32_t, vecops::uint32_t,
    vecops::int64_t, vecops::uint64_t>;

template <typename T>
class VecBitElementTest : public ::testing::Test {};

TYPED_TEST_SUITE(VecBitElementTest, IntegerElementTypes);

template <typename T>
using UnsignedBits = std::make_unsigned_t<T>;

template <typename T>
UnsignedBits<T> bits_of(T value) {
  if constexpr (std::is_signed_v<T>)
    return ::vecops::bitcast<UnsignedBits<T>>(value);
  else
    return value;
}

template <typename T>
T from_bits(UnsignedBits<T> value) {
  if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(value);
  else return value;
}

template <typename T>
T bit_operand(vecops::nint_t lane, bool rhs) {
  using U = UnsignedBits<T>;
  constexpr int width = std::numeric_limits<U>::digits;
  U value = static_cast<U>(
      rhs ? (U{0x5b} + static_cast<U>(lane * 37))
          : (U{0xa6} ^ static_cast<U>(lane * 53)));
  value ^= static_cast<U>(U{1} << (lane % width));
  if (!rhs && lane % 3 == 0)
    value |= static_cast<U>(U{1} << (width - 1));
  return from_bits<T>(value);
}

template <typename T>
T expected_shift_left(T value, int count) {
  using U = UnsignedBits<T>;
  constexpr int width = std::numeric_limits<U>::digits;
  if (count < 0) return value;
  if (count >= width) return T{};
  return from_bits<T>(static_cast<U>(bits_of(value) << count));
}

template <typename T>
T expected_shift_right(T value, int count) {
  using U = UnsignedBits<T>;
  constexpr int width = std::numeric_limits<U>::digits;
  if (count < 0) return value;
  const U bits = bits_of(value);
  if constexpr (std::is_unsigned_v<T>) {
    return count >= width ? T{} : static_cast<T>(bits >> count);
  } else {
    const bool negative = (bits >> (width - 1)) != 0;
    if (count >= width)
      return from_bits<T>(negative ? std::numeric_limits<U>::max() : U{});
    if (count == 0) return value;
    U result = static_cast<U>(bits >> count);
    if (negative)
      result |= static_cast<U>(
          std::numeric_limits<U>::max() << (width - count));
    return from_bits<T>(result);
  }
}

template <vec::IntegerTag Tag>
void verify_bit_operations(Tag tag) {
  using T = vec::ElementOf<Tag>;
  using U = UnsignedBits<T>;
  auto a = vec::zeros(tag);
  auto b = vec::zeros(tag);
  auto defaults = vec::zeros(tag);
  auto mask = vec::mfalse(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    a = vec::set(tag, a, lane, bit_operand<T>(lane, false));
    b = vec::set(tag, b, lane, bit_operand<T>(lane, true));
    defaults = vec::set(tag, defaults, lane, bit_operand<T>(lane + 11, true));
    mask = vec::set(tag, mask, lane, lane % 3 != 1);
  }

  const auto and_value = vec::bit_and(a, b);
  const auto or_value = vec::bit_or(a, b);
  const auto xor_value = vec::bit_xor(a, b);
  const auto andnot_value = vec::bit_andnot(a, b);
  const auto not_value = vec::bit_not(a);
  const auto unmasked_and =
      vec::bit_and(a, b, vec::opt::unmasked);
  const auto unmasked_or = vec::bit_or(a, b, vec::opt::unmasked);
  const auto unmasked_xor = vec::bit_xor(a, b, vec::opt::unmasked);
  const auto unmasked_andnot =
      vec::bit_andnot(a, b, vec::opt::unmasked);
  const auto unmasked_not = vec::bit_not(a, vec::opt::unmasked);
  const T scalar_merge = bit_operand<T>(29, true);
  const auto masked_and = vec::bit_and(a, b, vec::opt::masked(mask));
  const auto masked_or = vec::bit_or(
      a, b, vec::opt::masked(mask), vec::opt::zero);
  const auto masked_xor = vec::bit_xor(
      a, b, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
  const auto masked_andnot = vec::bit_andnot(
      a, b, vec::opt::masked(mask), vec::opt::merge(defaults));
  const auto masked_not = vec::bit_not(a, vec::opt::masked(mask));
  const auto zero_not = vec::bit_not(
      a, vec::opt::masked(mask), vec::opt::zero);
  const auto scalar_not = vec::bit_not(
      a, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
  const auto vector_not = vec::bit_not(
      a, vec::opt::masked(mask), vec::opt::merge(defaults));

  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T av = bit_operand<T>(lane, false);
    const T bv = bit_operand<T>(lane, true);
    const T dv = bit_operand<T>(lane + 11, true);
    const U ua = bits_of(av);
    const U ub = bits_of(bv);
    const bool active = lane % 3 != 1;
    const T expected_and = from_bits<T>(static_cast<U>(ua & ub));
    const T expected_or = from_bits<T>(static_cast<U>(ua | ub));
    const T expected_xor = from_bits<T>(static_cast<U>(ua ^ ub));
    const T expected_andnot = from_bits<T>(static_cast<U>((~ua) & ub));
    const T expected_not = from_bits<T>(static_cast<U>(~ua));

    EXPECT_EQ(expected_and, vec::get(tag, and_value, lane)) << "lane=" << lane;
    EXPECT_EQ(expected_or, vec::get(tag, or_value, lane)) << "lane=" << lane;
    EXPECT_EQ(expected_xor, vec::get(tag, xor_value, lane)) << "lane=" << lane;
    EXPECT_EQ(expected_andnot, vec::get(tag, andnot_value, lane)) << "lane=" << lane;
    EXPECT_EQ(expected_not, vec::get(tag, not_value, lane)) << "lane=" << lane;
    EXPECT_EQ(vec::get(tag, and_value, lane),
              vec::get(tag, unmasked_and, lane));
    EXPECT_EQ(vec::get(tag, or_value, lane),
              vec::get(tag, unmasked_or, lane));
    EXPECT_EQ(vec::get(tag, xor_value, lane),
              vec::get(tag, unmasked_xor, lane));
    EXPECT_EQ(vec::get(tag, andnot_value, lane),
              vec::get(tag, unmasked_andnot, lane));
    EXPECT_EQ(vec::get(tag, not_value, lane),
              vec::get(tag, unmasked_not, lane));
    EXPECT_EQ(active ? expected_and : av, vec::get(tag, masked_and, lane));
    EXPECT_EQ(active ? expected_or : T{}, vec::get(tag, masked_or, lane));
    EXPECT_EQ(active ? expected_xor : scalar_merge,
              vec::get(tag, masked_xor, lane));
    EXPECT_EQ(active ? expected_andnot : dv,
              vec::get(tag, masked_andnot, lane));
    EXPECT_EQ(active ? expected_not : av, vec::get(tag, masked_not, lane));
    EXPECT_EQ(active ? expected_not : T{}, vec::get(tag, zero_not, lane));
    EXPECT_EQ(active ? expected_not : scalar_merge,
              vec::get(tag, scalar_not, lane));
    EXPECT_EQ(active ? expected_not : dv, vec::get(tag, vector_not, lane));
  }

  constexpr int width = std::numeric_limits<U>::digits;
  for (const int count : {-3, 0, 1, width - 1, width, width + 5}) {
    const auto left = vec::bit_shl(a, count);
    const auto right = vec::bit_shr(a, count);
    const auto unmasked_left =
        vec::bit_shl(a, count, vec::opt::unmasked);
    const auto unmasked_right =
        vec::bit_shr(a, count, vec::opt::unmasked);
    const auto masked_left = vec::bit_shl(
        a, count, vec::opt::masked(mask));
    const auto zero_right = vec::bit_shr(
        a, count, vec::opt::masked(mask), vec::opt::zero);
    const auto scalar_left = vec::bit_shl(
        a, count, vec::opt::masked(mask), vec::opt::merge(scalar_merge));
    const auto vector_right = vec::bit_shr(
        a, count, vec::opt::masked(mask), vec::opt::merge(defaults));
    for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
      const T av = bit_operand<T>(lane, false);
      const bool active = lane % 3 != 1;
      const T expected_left = expected_shift_left(av, count);
      const T expected_right = expected_shift_right(av, count);
      EXPECT_EQ(expected_left, vec::get(tag, left, lane))
          << "left lane=" << lane << ", count=" << count;
      EXPECT_EQ(expected_right, vec::get(tag, right, lane))
          << "right lane=" << lane << ", count=" << count;
      EXPECT_EQ(vec::get(tag, left, lane),
                vec::get(tag, unmasked_left, lane));
      EXPECT_EQ(vec::get(tag, right, lane),
                vec::get(tag, unmasked_right, lane));
      EXPECT_EQ(active ? expected_left : av,
                vec::get(tag, masked_left, lane));
      EXPECT_EQ(active ? expected_right : (count < 0 ? av : T{}),
                vec::get(tag, zero_right, lane));
      EXPECT_EQ(active ? expected_left : (count < 0 ? av : scalar_merge),
                vec::get(tag, scalar_left, lane));
      EXPECT_EQ(active ? expected_right : (count < 0 ? av : bit_operand<T>(lane + 11, true)),
                vec::get(tag, vector_right, lane));
    }
  }

  auto counts = vec::zeros(tag);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane)
    counts = vec::set(
        tag, counts, lane, static_cast<T>(lane % (width + 2)));
  const auto const_left = vec::bit_shl(a, vecops::meta::cint<3>);
  const auto const_right = vec::bit_shr(
      a, vecops::meta::cint<3>,
      vec::opt::masked(mask), vec::opt::zero);
  const auto dynamic_left = vec::bit_shl(
      a, vecops::meta::dyn<1>(2));
  const auto dynamic_right = vec::bit_shr(
      a, vecops::meta::dyn<1>(2), vec::opt::unmasked);
  const auto lane_left = vec::bit_shl(a, counts);
  const auto lane_right = vec::bit_shr(a, counts);
  const auto masked_lane_left = vec::bit_shl(
      a, counts, vec::opt::masked(mask), vec::opt::merge(defaults));
  const auto masked_lane_right = vec::bit_shr(
      a, counts, vec::opt::masked(mask), vec::opt::zero);
  for (vecops::nint_t lane = 0; lane < vec::size(tag); ++lane) {
    const T av = bit_operand<T>(lane, false);
    const T dv = bit_operand<T>(lane + 11, true);
    const bool active = lane % 3 != 1;
    const int count = static_cast<int>(lane % (width + 2));
    EXPECT_EQ(expected_shift_left(av, 3), vec::get(tag, const_left, lane));
    EXPECT_EQ(active ? expected_shift_right(av, 3) : T{},
              vec::get(tag, const_right, lane));
    EXPECT_EQ(expected_shift_left(av, 2), vec::get(tag, dynamic_left, lane));
    EXPECT_EQ(expected_shift_right(av, 2),
              vec::get(tag, dynamic_right, lane));
    EXPECT_EQ(expected_shift_left(av, count),
              vec::get(tag, lane_left, lane));
    EXPECT_EQ(expected_shift_right(av, count),
              vec::get(tag, lane_right, lane));
    EXPECT_EQ(active ? expected_shift_left(av, count) : dv,
              vec::get(tag, masked_lane_left, lane));
    EXPECT_EQ(active ? expected_shift_right(av, count) : T{},
              vec::get(tag, masked_lane_right, lane));
  }
}

TYPED_TEST(VecBitElementTest, CoversEveryLaneSubwordAndMultiwordShape) {
  using T = TypeParam;
  vec_test::for_each_scalable_shape<T>([]<vec::IntegerTag Tag>() {
    verify_bit_operations(Tag{});
  });
}

#if defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)

TYPED_TEST(VecBitElementTest, FixedSVEBatchesBeyondFourWords) {
  using T = TypeParam;
  constexpr vecops::nint_t word_lanes =
      FIXED_SVE_BITS / 8 / static_cast<vecops::nint_t>(sizeof(T));
  constexpr vecops::nint_t array_lanes = static_cast<vecops::nint_t>(
      std::bit_ceil(static_cast<std::uint64_t>(word_lanes * 4 + 1)));
  using Tag = vec::FixedTag<T, array_lanes>;

  EXPECT_GT(vec::num_words(Tag{}), 4);
  verify_bit_operations(Tag{});
}

#endif
